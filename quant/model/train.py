"""特征矩阵、模型工厂与 Walk-Forward 训练协议。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.config import Config
from quant.utils import setup_logging


def prepare_xy(
    factor_long: pd.DataFrame,
    label_long: pd.DataFrame,
    cfg: Config,
    factor_names: list[str] | None = None,
) -> pd.DataFrame:
    """因子长表 → 特征宽表（date, symbol, feat_*, label, excess）。"""
    names = factor_names or sorted(factor_long["factor"].unique().tolist())
    filtered = factor_long[factor_long["factor"].isin(names)]
    feats = filtered.pivot_table(index=["date", "symbol"], columns="factor", values="value")
    feats = feats[[c for c in names if c in feats.columns]]
    labels = label_long.set_index(["date", "symbol"])[["excess", "value"]]
    df = feats.join(labels, how="inner").reset_index()
    return df.rename(columns={c: f"f_{c}" for c in names if c in df.columns})


def _make_model(cfg: Config, seed: int | None = None):
    """按配置构造模型：lightgbm 优先，gbt 兜底；regression 标签用回归器。"""
    seed = seed if seed is not None else cfg.run.seed
    name = cfg.model.name
    is_reg = cfg.model.label_mode == "regression"
    if name in ("lightgbm", "auto"):
        try:
            if is_reg:
                from lightgbm import LGBMRegressor

                return LGBMRegressor(
                    random_state=seed, verbosity=-1, **cfg.model.params
                )
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


def _make_windows(
    dates: pd.DatetimeIndex, cfg: Config
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp, int]]:
    """滚动窗口：返回 [(train_start, train_end, test_start, test_end, fold)]。

    约束：train_end < test_start；窗口从数据末端向前滚动。
    """
    test_days = cfg.model.test_months * 21
    train_days = cfg.model.train_years * 252
    windows = []
    test_end = dates[-1]
    for fold in range(cfg.model.n_splits):
        test_start_idx = max(0, dates.searchsorted(test_end) - test_days)
        train_end_idx = test_start_idx - 1
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
    windows = _make_windows(dates, cfg)
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
        ic = _rank_ic(proba, excess_all[test_mask])
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
