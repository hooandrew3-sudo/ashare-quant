"""特征矩阵、模型工厂与 Walk-Forward 训练协议。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.config import Config, ModelConfig
from quant.utils import setup_logging


def prepare_xy(
    factor_long: pd.DataFrame,
    label_long: pd.DataFrame,
    cfg: Config,
    factor_names: list[str] | None = None,
    feats_wide: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """因子长表 → 特征宽表（date, symbol, feat_*, label, excess）。

    feats_wide 可传入预构建的特征宽表（多周期/多种子训练复用，避免重复 pivot
    百万行因子表；列名须已带 f_ 前缀）。
    """
    names = factor_names or sorted(factor_long["factor"].unique().tolist())
    if feats_wide is None:
        filtered = factor_long[factor_long["factor"].isin(names)]
        feats = filtered.pivot_table(index=["date", "symbol"], columns="factor", values="value")
        feats = feats[[c for c in names if c in feats.columns]]
        feats = feats.rename(columns={c: f"f_{c}" for c in names if c in feats.columns})
    else:
        feats = feats_wide
    labels = label_long.set_index(["date", "symbol"])[["excess", "value"]]
    df = feats.join(labels, how="inner").reset_index()
    return df


def _make_model(cfg: Config, seed: int | None = None):
    """按配置构造模型：lightgbm 优先，gbt 兜底；regression 标签用回归器。"""
    seed = seed if seed is not None else cfg.run.seed
    name = cfg.model.name
    is_reg = cfg.model.label_mode == "regression"
    if name in ("lightgbm", "auto"):
        try:
            if is_reg:
                from lightgbm import LGBMRegressor

                params = dict(cfg.model.params)
                params.setdefault("random_state", seed)
                params.setdefault("verbosity", -1)
                return LGBMRegressor(**params)
            from lightgbm import LGBMClassifier

            params = dict(cfg.model.params)
            params.setdefault("random_state", seed)
            params.setdefault("verbosity", -1)
            return LGBMClassifier(**params)
        except ImportError:
            if name == "lightgbm":
                raise RuntimeError("配置指定 lightgbm 但未安装：pip install lightgbm")
    if is_reg:
        from sklearn.ensemble import GradientBoostingRegressor

        return GradientBoostingRegressor(random_state=seed, **cfg.model.gbt_params)
    from sklearn.ensemble import GradientBoostingClassifier

    return GradientBoostingClassifier(random_state=seed, **cfg.model.gbt_params)


def _metric(features: np.ndarray, y: np.ndarray) -> dict:
    """无需额外依赖的 AUC 近似（Mann-Whitney U）。"""
    pos = features[y == 1]
    neg = features[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return {"auc": None, "n_pos": int(len(pos)), "n_neg": int(len(neg))}
    rank = pd.Series(np.r_[pos, neg]).rank().to_numpy()
    u = rank[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2
    auc = u / (len(pos) * len(neg))
    return {"auc": float(auc), "n_pos": int(len(pos)), "n_neg": int(len(neg))}


def _max_horizon(cfg: ModelConfig) -> int:
    """标签视野上限：purge/embargo 长度取多周期配置的最大 horizon。"""
    horizons = list(getattr(cfg, "horizons", None) or [getattr(cfg, "horizon", 20)])
    return max(int(h) for h in horizons if h and int(h) > 0)


def _make_windows(
    dates: pd.DatetimeIndex, cfg: ModelConfig
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp, int]]:
    """滚动窗口：返回 [(train_start, train_end, test_start, test_end, fold)]。

    约束：train_end < test_start；窗口从数据末端向前滚动。
    purge：训练窗末端回退 max_horizon 个交易日 —— horizon=N 的重叠标签
    会使训练集尾部样本的收益窗口延伸进测试期，构成边界泄漏（embargo）。
    """
    test_days = cfg.test_months * 21
    train_days = cfg.train_years * 252
    embargo = _max_horizon(cfg)
    windows = []
    test_end = dates[-1]
    for fold in range(cfg.n_splits):
        test_start_idx = max(0, dates.searchsorted(test_end) - test_days)
        train_end_idx = test_start_idx - 1 - embargo
        train_start_idx = max(0, train_end_idx - train_days + 1)
        if test_start_idx <= 0 or train_end_idx <= 0 or test_start_idx >= len(dates) - 1:
            break
        windows.append(
            (
                dates[train_start_idx],
                dates[train_end_idx],
                dates[test_start_idx],
                test_end,
                fold + 1,
            )
        )
        test_end = dates[test_start_idx - 1] if test_start_idx > 0 else dates[0]
    return windows


def oos_region_start(dates: pd.DatetimeIndex, cfg: ModelConfig) -> pd.Timestamp | None:
    """全部 Walk-Forward 折覆盖的 OOS 区间起点（最早一折 test_start 回退 horizon）。

    因子准入 / 复合因子权重 / 门禁 t 值的数据窗口必须 ≤ 该日期，否则
    特征构造用到了 OOS 测试段自身的未来收益（结构性前视）。回退一个
    horizon 是因为 date≤D 的标签收益延伸至 D+horizon，须保证不越过
    最早的测试日。
    """
    windows = _make_windows(pd.DatetimeIndex(sorted(dates)), cfg)
    if not windows:
        return None
    earliest_test_start = min(pd.Timestamp(w[2]) for w in windows)
    idx = pd.DatetimeIndex(sorted(dates)).searchsorted(earliest_test_start)
    clean_idx = max(0, idx - _max_horizon(cfg))
    return pd.DatetimeIndex(sorted(dates))[clean_idx]


def walk_forward(
    xy: pd.DataFrame,
    cfg: Config,
    feature_cols: list[str],
    log=None,
    seed: int | None = None,
) -> dict:
    """执行 Walk-Forward，返回 OOS 预测、逐折指标、最终模型。"""
    log = log or setup_logging(cfg.run.verbose)
    dates = pd.DatetimeIndex(sorted(xy["date"].unique()))
    windows = _make_windows(dates, cfg.model)
    if not windows:
        raise RuntimeError("数据时间跨度不足以构成 Walk-Forward 窗口，请扩大数据范围")

    X_all = xy[feature_cols].to_numpy(dtype=float)
    y_all = xy["value"].to_numpy()
    excess_all = xy["excess"].to_numpy()
    date_arr = pd.to_datetime(xy["date"]).to_numpy()

    oos: list[pd.DataFrame] = []
    fold_metrics: list[dict] = []
    final_model = None
    is_reg = cfg.model.label_mode == "regression"

    for train_start, train_end, test_start, test_end, fold in windows:
        train_mask = (date_arr >= np.datetime64(train_start)) & (
            date_arr <= np.datetime64(train_end)
        )
        test_mask = (date_arr > np.datetime64(train_end)) & (date_arr <= np.datetime64(test_end))
        if train_mask.sum() < 100 or test_mask.sum() < 30:
            log.warning("fold %d 样本不足，跳过", fold)
            continue
        model = _make_model(cfg, seed=seed)
        model.fit(X_all[train_mask], y_all[train_mask])
        if is_reg:
            proba = model.predict(X_all[test_mask])
            m = {"auc": None, "n_pos": 0, "n_neg": 0}
        else:
            proba = model.predict_proba(X_all[test_mask])[:, 1]
            m = _metric(proba, y_all[test_mask])
        ic = _daily_rank_ic(date_arr[test_mask], proba, excess_all[test_mask])
        fold_metrics.append(
            {
                "fold": fold,
                "train_start": str(train_start.date()),
                "train_end": str(train_end.date()),
                "test_start": str(test_start.date()),
                "test_end": str(test_end.date()),
                "n_train": int(train_mask.sum()),
                "n_test": int(test_mask.sum()),
                **m,
                "rank_ic_excess": round(ic, 5) if ic == ic else None,
            }
        )
        log.info(
            "fold %d: train=%d test=%d auc=%s ic=%s",
            fold, int(train_mask.sum()), int(test_mask.sum()),
            fold_metrics[-1]["auc"], fold_metrics[-1]["rank_ic_excess"],
        )
        oos.append(
            pd.DataFrame(
                {
                    "date": xy.loc[test_mask, "date"].to_numpy(),
                    "symbol": xy.loc[test_mask, "symbol"].to_numpy(),
                    "score": proba,
                    "y_true": y_all[test_mask],
                    "excess": excess_all[test_mask],
                    "fold": fold,
                }
            )
        )
        final_model = model

    oos_df = pd.concat(oos, ignore_index=True) if oos else pd.DataFrame()
    metrics_df = pd.DataFrame(fold_metrics)
    return {
        "oos": oos_df,
        "fold_metrics": metrics_df,
        "final_model": final_model,
        "feature_cols": feature_cols,
    }


def _rank_ic(score: np.ndarray, excess: np.ndarray) -> float:
    mask = ~np.isnan(excess)
    if mask.sum() < 10:
        return float("nan")
    s = pd.Series(score[mask]).rank()
    e = pd.Series(excess[mask]).rank()
    return float(s.corr(e))


def _daily_rank_ic(dates: np.ndarray, score: np.ndarray, excess: np.ndarray) -> float:
    """逐日截面 Rank IC 的均值（业界标准口径）。

    此前实现把整折所有 (date, symbol) 混入单一 Spearman，混合了截面与时序
    变异，与其他模块的“逐日 IC 均值”不可比，且可能掩盖符号翻转。
    """
    df = pd.DataFrame({"date": dates, "score": score, "excess": excess})
    df = df.dropna(subset=["excess"])
    if df.empty:
        return float("nan")
    ics = []
    for _, g in df.groupby("date"):
        if len(g) < 10:
            continue
        ic = g["score"].rank().corr(g["excess"].rank())
        if ic == ic:
            ics.append(ic)
    if not ics:
        return float("nan")
    return float(np.mean(ics))


def ensemble_scores(oos_by_horizon: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """多周期集成：各周期 OOS 概率先做截面分位，再等权平均；保留 y_true/excess。"""
    if not oos_by_horizon:
        raise ValueError("无各周期 OOS 预测")
    frames: list[pd.DataFrame] = []
    labels_frames: list[pd.DataFrame] = []
    for horizon, df in sorted(oos_by_horizon.items()):
        d = df.copy()
        d[f"h{horizon}"] = d.groupby("date")["score"].rank(pct=True)
        frames.append(d[["date", "symbol", "fold", f"h{horizon}"]])
        if {"y_true", "excess"}.issubset(d.columns):
            labels_frames.append(d[["date", "symbol", "fold", "y_true", "excess"]])
    merged = frames[0]
    for f in frames[1:]:
        merged = merged.merge(f, on=["date", "symbol", "fold"], how="outer")
    cols = [c for c in merged.columns if c.startswith("h")]
    merged["score"] = merged[cols].mean(axis=1)
    out = merged[["date", "symbol", "score", "fold"]].dropna(subset=["score"])
    # 回灌标签（取最后一期标签列避免 merge 冲突）
    if labels_frames:
        labels = labels_frames[-1].drop_duplicates(subset=["date", "symbol", "fold"])
        out = out.merge(labels, on=["date", "symbol", "fold"], how="left")
    return out.sort_values(["date", "symbol"]).reset_index(drop=True)
