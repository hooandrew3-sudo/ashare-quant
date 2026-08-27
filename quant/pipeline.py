"""研究流水线编排：数据 → 因子 → 模型 → 组合 → 回测 → 报告。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from quant.config import Config, load_config, to_dict
from quant.data.quality import check_benchmark, check_prices
from quant.data.storage import DataBundle, Storage
from quant.data.synthetic import generate_synthetic
from quant.factors.analysis import factor_ic_report, report_to_frame, save_ic_report
from quant.factors.compute import compute_all_factors
from quant.model.label import build_label
from quant.model.registry import ModelRegistry
from quant.model.train import prepare_xy, walk_forward
from quant.metrics.performance import compute_metrics
from quant.backtest.engine import BacktestEngine
from quant.backtest.stress import run_stress
from quant.portfolio.regime import compute_regime
from quant.portfolio.selection import build_target_weights
from quant.portfolio.risk_budget import build_target_weights_with_risk_budget
from quant.report.report import build_html_report
from quant.utils import ensure_dir, setup_logging


def prepare_data(cfg: Config, symbols: list[str] | None = None) -> DataBundle:
    """按配置准备数据快照并写入存储。"""
    cfg.validate()  # 工业级防线：synthetic 数据严禁写入真实数据目录
    log = setup_logging(cfg.run.verbose)
    storage = Storage(cfg.data.root)
    src = cfg.data.source
    start = end = None
    if src == "synthetic":
        bundle = generate_synthetic(
            n_stocks=cfg.data.demo.n_stocks,
            years=cfg.data.demo.years,
            start=cfg.data.demo.start,
            seed=cfg.run.seed,
        )
    elif src == "baostock":
        from quant.data.real import load_from_baostock
        from quant.data.baostock_sync import BaoStockSync

        end = cfg.data.sync.end or datetime.now().strftime("%Y-%m-%d")
        start = cfg.data.sync.start
        universe = cfg.data.sync.universe
        if universe == "csi800":
            with BaoStockSync(cache_dir=storage.root / "cache") as sync:
                symbols = sync.csi800_symbols()
            log.info("股票池: 中证800（%d 只）", len(symbols))
        elif universe == "all":
            with BaoStockSync(cache_dir=storage.root / "cache") as sync:
                basic = sync.stock_basic()
            symbols = basic.loc[basic["type"].astype(str) == "1", "symbol"].tolist()
            log.info("股票池: 全市场（%d 只）", len(symbols))
        elif not symbols:
            raise ValueError(
                "baostock 数据源需要 --symbols（逗号分隔）或 --universe csi800/all"
            )
        bundle = load_from_baostock(
            symbols,
            start,
            end,
            benchmark=cfg.data.benchmark,
            incremental=cfg.data.sync.incremental,
            manifest=storage._manifest,
            storage=storage,
            universe=universe,
        )
    elif src == "parquet":
        bundle = storage.load_bundle()
        bundle.validate()
        return bundle
    else:
        raise ValueError(f"未知数据源: {src}")

    storage._manifest["meta"] = {
        "universe": cfg.data.sync.universe if src == "baostock" else src,
        "source": src,
        "start": start,
        "end": end,
    }
    storage.save_manifest()
    bundle.validate()
    storage.save("prices", bundle.prices)
    storage.save("benchmark", bundle.benchmark, partition_by_symbol=False)
    # 可选数据集：为空时显式放行并告警（此前静默早退，磁盘残留过期数据无人知晓）
    for name, part_df in (
        ("meta", bundle.meta),
        ("industry", bundle.industry),
        ("fundamentals", bundle.fundamentals),
        ("constituents", getattr(bundle, "constituents", pd.DataFrame())),
    ):
        if part_df.empty:
            log.warning("数据集 %s 为空（上游未提供或拉取失败），保留存储现状", name)
        storage.save(name, part_df, partition_by_symbol=False, allow_empty=True)
    log.info("数据已写入 %s（%d 行 prices）", storage.processed, len(bundle.prices))
    return bundle


def prepare_fundamentals(cfg: Config) -> pd.DataFrame:
    """拉取全市场季度 ROE/股息率并写入 fundamentals.parquet。"""
    log = setup_logging(cfg.run.verbose)
    from quant.data.fundamentals_ak import fetch_fundamentals

    fund = fetch_fundamentals(verbose=cfg.run.verbose)
    if fund.empty:
        raise RuntimeError("财务数据拉取为空")
    storage = Storage(cfg.data.root)
    storage.save("fundamentals", fund, partition_by_symbol=False)
    log.info("财务数据已写入: %d 条 (symbol=%d)", len(fund), fund["symbol"].nunique())
    return fund


def prepare_cashflow(cfg: Config) -> pd.DataFrame:
    """拉取全市场季度现金流（OCF/每股收益/每股经营现金流）并写入 cashflow.parquet。"""
    log = setup_logging(cfg.run.verbose)
    from quant.data.fundamentals_ak import fetch_cashflow

    cf = fetch_cashflow(verbose=cfg.run.verbose)
    if cf.empty:
        raise RuntimeError("现金流数据拉取为空")
    storage = Storage(cfg.data.root)
    storage.save("cashflow", cf, partition_by_symbol=False)
    log.info(
        "现金流数据已写入: %d 条 (symbol=%d, 期数=%d)",
        len(cf), cf["symbol"].nunique(), cf["report_period"].nunique(),
    )
    return cf


def prepare_industry(cfg: Config) -> pd.DataFrame:
    """拉取全市场行业分类并写入 industry.parquet（激活行业中性化）。"""
    log = setup_logging(cfg.run.verbose)
    from quant.data.baostock_sync import BaoStockSync

    storage = Storage(cfg.data.root)
    with BaoStockSync(cache_dir=storage.root / "cache") as sync:
        ind = sync.industry()
    ind["as_of_date"] = pd.to_datetime(ind["as_of_date"], errors="coerce")
    ind = ind.dropna(subset=["symbol", "industry"])
    storage.save("industry", ind, partition_by_symbol=False)
    log.info("行业数据已写入: %d 条 (symbol=%d)", len(ind), ind["symbol"].nunique())
    return ind


def prepare_sentiment(cfg: Config) -> pd.DataFrame:
    """拉取月末调仓日公告情绪并写入 sentiment.parquet。"""
    log = setup_logging(cfg.run.verbose)
    from quant.data.sentiment_cninfo import fetch_sentiment_dates

    storage = Storage(cfg.data.root)
    prices = storage.load("prices")
    dates = pd.DatetimeIndex(sorted(pd.unique(prices["date"])))
    month_end = dates.to_series().groupby(dates.to_period("M")).max().tolist()
    day_strs = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in month_end]
    senti = fetch_sentiment_dates(day_strs, verbose=cfg.run.verbose)
    if senti.empty:
        raise RuntimeError("情绪数据拉取为空")
    storage.save("sentiment", senti, partition_by_symbol=False)
    log.info("情绪数据已写入: %d 条 (symbol=%d)", len(senti), senti["symbol"].nunique())
    return senti


def compute_history_members(cfg: Config) -> list[str]:
    """计算历史大市值成员（季度市值前 900 并集），输出 delta 清单。"""
    log = setup_logging(cfg.run.verbose)
    from quant.data.universe_history import historical_largecap_members

    storage = Storage(cfg.data.root)
    bundle = storage.load_bundle()
    members = historical_largecap_members(bundle.prices, bundle.fundamentals, top=900)
    current = set(bundle.prices["symbol"].unique())
    delta = sorted(set(members) - current)
    import json

    ensure_dir(cfg.data.root / "cache")
    (cfg.data.root / "cache" / "history_members_delta.json").write_text(
        json.dumps(delta), encoding="utf-8"
    )
    log.info("历史大市值成员: %d 个（新增 %d 个）", len(members), len(delta))
    return delta


def run_research(
    cfg: Config,
    bundle: DataBundle,
    output_root: str | Path = "artifacts",
    force: bool = False,
) -> dict[str, Any]:
    """执行因子→模型→组合→回测→报告全流程。"""
    cfg.validate()
    log = setup_logging(cfg.run.verbose)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + cfg.fingerprint()
    art = ensure_dir(Path(output_root) / run_id)
    storage = Storage(cfg.data.root)

    # 一致性防线：行业表不得包含行情外的股票（防残留/错配数据污染因子横截面）
    if not bundle.industry.empty:
        valid_syms = set(bundle.prices["symbol"].unique())
        bad = set(bundle.industry["symbol"].unique()) - valid_syms
        if bad:
            log.warning(
                "industry 含 %d 只不在行情中的股票（疑似残留数据），已禁用行业中性化",
                len(bad),
            )
            bundle.industry = pd.DataFrame()

    # 1. 质量校验
    rep_p = check_prices(bundle.prices)
    rep_b = check_benchmark(bundle.benchmark)
    if not rep_p.ok or not rep_b.ok:
        raise RuntimeError(f"数据质量不合格: {rep_p.summary()}; {rep_b.summary()}")
    log.info(rep_p.summary())

    # 2. 因子
    factor_long, cov = compute_all_factors(bundle, cfg, report_coverage=True)
    cov.to_parquet(art / "factor_coverage.parquet", index=False)

    # 3. 标签 + IC 报告（完整报告用于研究归档）
    label_long = build_label(bundle.prices, bundle.benchmark, cfg)
    # P0 防泄露：in-sample cutoff 必须对齐 Walk-Forward OOS 区间起点。
    # 此前按"全样本前 60%"取 cutoff，当样本 < ~6.3 年时 cutoff 会落在
    # 首折测试段之内 —— 复合因子权重/因子准入用到了测试段自身的未来收益，
    # 属结构性前视。现改为直接复用 _make_windows 的折边界（再回退一个 horizon）。
    from quant.model.train import oos_region_start

    all_dates = pd.DatetimeIndex(sorted(pd.unique(pd.to_datetime(factor_long["date"]))))
    train_cutoff = oos_region_start(all_dates, cfg.model)
    if train_cutoff is not None:
        log.info(
            "因子准入/复合因子窗口 cutoff: %s（= 最早一折 test_start 回退 horizon，杜绝 OOS 泄漏）",
            pd.Timestamp(train_cutoff).date(),
        )
    else:
        # 无有效 Walk-Forward 窗口时退回 60% 规则并显式告警
        if len(all_dates) > 252:
            train_cutoff = pd.Timestamp(all_dates[int(len(all_dates) * 0.6)])
            log.warning("无法推导 OOS 起点，cutoff 退回 60%% 规则: %s", train_cutoff.date())
    ic_report = factor_ic_report(factor_long, label_long, cfg)
    # 3.1 复合因子：方向显著因子等权合成，再纳入 IC 报告与特征
    if cfg.factors.composite:
        from quant.factors.composite import build_composite_factor

        ic_report_for_composite = ic_report
        rotate = bool(getattr(cfg.factors, "composite_regime_rotate", False))
        regime_ctx = None
        if train_cutoff is not None:
            f_in = factor_long[pd.to_datetime(factor_long["date"]) <= train_cutoff]
            l_in = label_long[pd.to_datetime(label_long["date"]) <= train_cutoff]
            ic_report_for_composite = factor_ic_report(f_in, l_in, cfg)
        if rotate and not bundle.benchmark.empty:
            # 牛熊分域：基准收盘 vs MA(composite_regime_ma)，仅用 ≤cutoff 样本估计
            bench_close = bundle.benchmark.set_index("date")["close"].sort_index()
            ma = bench_close.rolling(
                int(getattr(cfg.factors, "composite_regime_ma", 120)), min_periods=60
            ).mean()
            bull_all = (bench_close > ma).dropna()
            bull_dates = {pd.Timestamp(d) for d in bull_all[bull_all].index}
            if train_cutoff is not None:
                bull_dates = {
                    d for d in bull_dates if d <= pd.Timestamp(train_cutoff)
                }
            d_in = f_in if train_cutoff is not None else factor_long
            dl_in = l_in if train_cutoff is not None else label_long
            din_date = pd.to_datetime(d_in["date"])
            f_bull = d_in[din_date.isin(pd.to_datetime(sorted(bull_dates)))]
            l_bull = dl_in[pd.to_datetime(dl_in["date"]).isin(pd.to_datetime(sorted(bull_dates)))]
            f_bear = d_in[~din_date.isin(pd.to_datetime(sorted(bull_dates)))]
            l_bear = dl_in[~pd.to_datetime(dl_in["date"]).isin(pd.to_datetime(sorted(bull_dates)))]
            ic_bull = (
                factor_ic_report(f_bull, l_bull, cfg)
                if len(f_bull) and len(l_bull)
                else None
            )
            ic_bear = (
                factor_ic_report(f_bear, l_bear, cfg)
                if len(f_bear) and len(l_bear)
                else None
            )
            regime_ctx = (bull_dates, ic_bull, ic_bear)
            n_in_dates = int(pd.to_datetime(d_in["date"]).nunique())
            log.info(
                "复合因子 regime 轮动启用：牛市日 %d / 熊市日 %d（估计窗口 ≤%s）",
                len(bull_dates),
                n_in_dates - len([d for d in pd.to_datetime(d_in['date'].unique()) if pd.Timestamp(d) in bull_dates]),
                pd.Timestamp(train_cutoff).date() if train_cutoff is not None else "全样本",
            )
        composite = build_composite_factor(
            factor_long,
            ic_report_for_composite,
            n=cfg.factors.composite_n,
            min_t=cfg.factors.min_t_stat,
            corr_max=cfg.factors.composite_corr_max,
            require_stable_decay=cfg.factors.composite_require_decay,
            weight_by=cfg.factors.composite_weight,
            regime_bull_dates=regime_ctx[0] if regime_ctx else None,
            ic_report_bull=regime_ctx[1] if regime_ctx else None,
            ic_report_bear=regime_ctx[2] if regime_ctx else None,
        )
        if not composite.empty:
            factor_long = pd.concat([factor_long, composite], ignore_index=True)
            ic_report = factor_ic_report(factor_long, label_long, cfg)
            log.info("复合因子已生成（成分 %d 个）", cfg.factors.composite_n)
            from quant.factors.composite import factor_correlations

            ic_report["factor_corr"] = factor_correlations(factor_long).round(3).to_dict()
    factor_long.to_parquet(art / "factors.parquet", index=False)
    save_ic_report(ic_report, art / "ic_report.json")
    ic_frame = report_to_frame(ic_report)
    ic_frame.to_csv(art / "ic_report.csv", index=False)

    # 4. 因子准入（P0 防泄露：使用样本内前期数据做因子筛选，禁用全量 IC 窥探测试集）
    # train_cutoff 已在步骤 3 按 OOS 区间起点推导，此处直接复用（勿重复计算导致口径漂移）
    if train_cutoff is not None:
        f_in = factor_long[pd.to_datetime(factor_long["date"]) <= train_cutoff]
        l_in = label_long[pd.to_datetime(label_long["date"]) <= train_cutoff]
        ic_report_in = factor_ic_report(f_in, l_in, cfg)
        ranked_in = sorted(
            ic_report_in["factors"].items(),
            key=lambda kv: abs(kv[1]["rank_ic_mean"] or 0), reverse=True,
        )
        passed = ic_report_in["passed"]
        if not passed:
            log.warning("样本内窗口无因子通过准入（NW 校正后 t 值更严格），将按 |IC| 排序补齐特征")
    else:
        ranked_in = sorted(
            ic_report["factors"].items(),
            key=lambda kv: abs(kv[1]["rank_ic_mean"] or 0), reverse=True,
        )
        passed = ic_report["passed"]
    # 准入因子 + 复合因子（如已生成）强制入特征，再按 |IC| 补齐 ≥3 个。
    # 实验（v9 §10）：补满 top_n 会稀释模型（8.9% vs 12.4%/13.6%）——
    # 弱特征（如 earn_mom 负 IC）进入反而拖累 LightGBM；
    # 复合因子成员资格由准入严格把关，特征槽位维持最小补齐。
    selected = list(passed)
    if cfg.factors.composite and "composite" in set(factor_long["factor"].unique()):
        if "composite" not in selected:
            selected.append("composite")
    min_features = max(3, len(selected))
    ranked_source = ranked_in
    selected = selected + [
        n for n, _ in ranked_source if n not in selected
    ][: max(0, min_features - len(selected))]
    selected = selected[: cfg.factors.top_n]
    log.info("入选因子: %s", selected)
    # 特征宽表只构建一次（多周期 × 多种子复用，避免重复 pivot 百万行因子表）
    feats_wide = factor_long[factor_long["factor"].isin(selected)].pivot_table(
        index=["date", "symbol"], columns="factor", values="value"
    )
    feats_wide = feats_wide[[c for c in selected if c in feats_wide.columns]]
    feats_wide = feats_wide.rename(
        columns={c: f"f_{c}" for c in selected if c in feats_wide.columns}
    )
    xy = prepare_xy(
        factor_long, label_long, cfg, factor_names=selected, feats_wide=feats_wide
    )
    feature_cols = [c for c in xy.columns if c.startswith("f_")]
    if len(feature_cols) == 0:
        raise RuntimeError("无可用于建模的因子特征")

    # 5. Walk-Forward 多周期 × 多种子模型（降低单次模型噪声）
    horizons = cfg.model.horizons or [cfg.model.horizon]
    oos_by_horizon: dict[str, Any] = {}
    models: dict[str, Any] = {}
    fold_frames = []
    seeds = cfg.model.seeds[: cfg.model.n_seeds] if cfg.model.n_seeds > 1 else [cfg.run.seed]
    for h in horizons:
        label_h = build_label(bundle.prices, bundle.benchmark, cfg, horizon=h)
        xy_h = prepare_xy(
            factor_long, label_h, cfg, factor_names=selected, feats_wide=feats_wide
        )
        for seed in seeds:
            res_h = walk_forward(xy_h, cfg, feature_cols, log=log, seed=seed)
            key = f"{h}_{seed}"
            oos_by_horizon[key] = res_h["oos"]
            models[key] = res_h["final_model"]
            fm = res_h["fold_metrics"].copy()
            fm["horizon"] = h
            fm["seed"] = seed
            fold_frames.append(fm)
        log.info("周期 %d 日模型完成（%d 种子）", h, len(seeds))
    from quant.model.train import ensemble_scores

    oos = ensemble_scores(oos_by_horizon)
    # P1 概率校准（OOF）：按 fold 交叉拟合，避免用同一份 OOS 拟合+变换的乐观偏差
    _calibrator = None
    _final_cal = None
    try:
        from quant.model.calibration import ProbabilityCalibrator

        oos_flat = oos.dropna(subset=["score", "y_true"]).copy()
        fold_ids = sorted(oos_flat["fold"].unique()) if "fold" in oos_flat.columns else []
        n_folds = len(fold_ids)
        if n_folds >= 2 and len(oos_flat) >= 200 and oos_flat["y_true"].nunique() >= 2:
            calibrated = pd.Series(index=oos_flat.index, dtype=float)
            for f in fold_ids:
                # expanding 校准：折 k 仅用时间上更早的折 <k 拟合，
                # 禁止"未来折"信息回流（此前 fold != f 会用到更晚期间的结果）
                train_idx = oos_flat["fold"] < f
                val_idx = oos_flat["fold"] == f
                if train_idx.sum() < 100 or val_idx.sum() < 30:
                    calibrated.loc[val_idx] = oos_flat.loc[val_idx, "score"]
                    continue
                cal = ProbabilityCalibrator(method="isotonic")
                cal.fit(
                    oos_flat.loc[train_idx, "score"].to_numpy(),
                    oos_flat.loc[train_idx, "y_true"].to_numpy(),
                )
                calibrated.loc[val_idx] = cal.transform(oos_flat.loc[val_idx, "score"].to_numpy())
            oos = oos.copy()
            oos.loc[oos_flat.index, "score"] = calibrated
            _calibrator = "oof_isotonic"
            log.info("OOF 概率校准已应用（n=%d, folds=%d）", len(oos_flat), n_folds)
            # 部署校准器：在全量 OOS 上拟合最终单调校准曲线（仅用于实盘推理排序，
            # 单调变换对选股排序无方向性影响）
            _final_cal = ProbabilityCalibrator(method="isotonic")
            _final_cal.fit(
                oos_flat["score"].to_numpy(), oos_flat["y_true"].to_numpy()
            )
        else:
            log.info("概率校准跳过：样本不足或 fold 数不足（n=%d, folds=%d）", len(oos_flat), n_folds)
    except Exception as exc:  # noqa: BLE001
        log.warning("概率校准失败: %s", exc)
        _calibrator = None
    # 选股模式：composite 直接用复合因子排序，hybrid 融合模型概率（仅 model 模式走 ML）
    if cfg.model.selection_mode in ("composite", "hybrid"):
        comp = factor_long[factor_long["factor"] == "composite"][["date", "symbol", "value"]]
        comp["date"] = pd.to_datetime(comp["date"])
        comp = comp.rename(columns={"value": "comp_score"})
        oos = oos.merge(comp, on=["date", "symbol"], how="left")
        if cfg.model.selection_mode == "composite":
            oos["score"] = oos["comp_score"].fillna(oos["score"])
            log.info("选股模式: composite 直接排序（绕过 ML）")
        else:
            oos["score"] = (
                oos.groupby("date")["score"].rank(pct=True) * 0.5
                + oos.groupby("date")["comp_score"].rank(pct=True) * 0.5
            )
            log.info("选股模式: hybrid（ML 概率 + composite 融合）")
    fold_metrics = pd.concat(fold_frames, ignore_index=True)
    result = {
        "oos": oos,
        "fold_metrics": fold_metrics,
        "models": models,
        "final_model": models.get(f"{horizons[-1]}_{seeds[-1]}"),
        "feature_cols": feature_cols,
        "calibrator": _final_cal,
    }
    _comp_meta = (ic_report.get("factors") or {}).get("composite", {}) if isinstance(ic_report, dict) else {}
    # P0 门禁口径修正：composite_t 此前取自全样本 IC 报告（含全部 WF 测试段，
    # "用考试答案决定能否参加考试"）。现改用纯 OOS 区间（≥ 最早一折 test_start）
    # 重算复合因子 IC，t 值经 Newey-West 校正，作为上线门禁的真实判据。
    oos_comp_meta: dict = {}
    try:
        first_test_start = None
        fm_all = fold_metrics
        if not fm_all.empty and "test_start" in fm_all.columns:
            ts = pd.to_datetime(fm_all["test_start"]).min()
            first_test_start = pd.Timestamp(ts)
        if first_test_start is not None and "composite" in set(factor_long["factor"].unique()):
            comp_long = factor_long[
                (factor_long["factor"] == "composite")
                & (pd.to_datetime(factor_long["date"]) >= first_test_start)
            ]
            label_oos = label_long[pd.to_datetime(label_long["date"]) >= first_test_start]
            ic_oos = factor_ic_report(comp_long, label_oos, cfg)
            oos_comp_meta = (ic_oos.get("factors") or {}).get("composite", {}) or {}
            log.info(
                "复合因子 OOS t=%.2f（全样本 t=%s，门禁采用 OOS 口径）",
                oos_comp_meta.get("t_stat") if oos_comp_meta.get("t_stat") is not None else float("nan"),
                _comp_meta.get("t_stat"),
            )
    except Exception as exc:  # noqa: BLE001
        log.warning("OOS composite_t 计算失败，回退全样本口径: %s", exc)
    result["composite_t"] = oos_comp_meta.get("t_stat", _comp_meta.get("t_stat"))
    result["composite_rank_ic"] = oos_comp_meta.get(
        "rank_ic_mean", _comp_meta.get("rank_ic_mean")
    )
    result["composite_t_full_sample"] = _comp_meta.get("t_stat")
    registry = ModelRegistry(Path(output_root))
    data_fingerprint = pd.util.hash_pandas_object(bundle.prices).sum()
    run_id_model = registry.save_run(cfg, str(data_fingerprint), result, run_id=run_id)
    log.info("模型已注册: %s", run_id_model)

    # 6. 组合：月末调仓 + 状态机
    if oos.empty:
        raise RuntimeError("多周期集成未产生样本外预测")
    trading_dates = pd.DatetimeIndex(sorted(pd.unique(bundle.prices["date"])))
    last_trading_of_month = (
        trading_dates.to_series().groupby(trading_dates.to_period("M")).max()
    )
    score_dates = set(pd.to_datetime(oos["date"]).dt.normalize())
    month_ends = [
        pd.Timestamp(d) for d in last_trading_of_month if pd.Timestamp(d) in score_dates
    ]
    freq = cfg.portfolio.rebalance_freq
    step = {"M": 1, "2M": 2, "Q": 3}.get(freq, 1)
    rebalance_dates = month_ends[::-1][::step][::-1] if step > 1 else month_ends
    log.info("调仓频率=%s，调仓日 %d 个", freq, len(rebalance_dates))
    bench_close = bundle.benchmark.set_index("date")["close"]
    regime = compute_regime(bench_close, cfg.portfolio)
    target_weights = (
        build_target_weights_with_risk_budget(
            scores=oos,
            prices=bundle.prices,
            industry=bundle.industry,
            cfg=cfg.portfolio,
            rebalance_dates=rebalance_dates,
            constituents=getattr(bundle, "constituents", None),
        )
        if cfg.portfolio.weight_method == "risk_budget"
        else build_target_weights(
            scores=oos,
            prices=bundle.prices,
            industry=bundle.industry,
            cfg=cfg.portfolio,
            rebalance_dates=rebalance_dates,
            constituents=getattr(bundle, "constituents", None),
        )
    )
    log.info("组合方法: %s", cfg.portfolio.weight_method)
    target_weights.to_parquet(art / "target_weights.parquet", index=False)
    signal_health = None
    if cfg.portfolio.signal_health_enabled:
        from quant.portfolio.signal_health import compute_signal_health

        signal_health = compute_signal_health(oos, bundle.prices, bundle.benchmark, cfg.portfolio)
        signal_health.to_parquet(art / "signal_health.parquet", index=False)
        log.info("信号健康度已生成（%d 个交易日）", len(signal_health))
    smallcap_regime = None
    if cfg.portfolio.smallcap_regime_enabled:
        if hasattr(bundle, "smallcap") and not getattr(bundle, "smallcap").empty and "date" in getattr(bundle, "smallcap").columns:
            from quant.portfolio.regime import compute_smallcap_regime
            sc_close = bundle.smallcap.set_index("date")["close"].sort_index()
            smallcap_regime = compute_smallcap_regime(sc_close, cfg.portfolio)
            smallcap_regime.to_parquet(art / "smallcap_regime.parquet", index=False)
            log.info("小盘辅助择时已生成（%d 个交易日）", len(smallcap_regime))
        else:
            log.warning("smallcap_regime_enabled=True 但 bundle 无 smallcap 数据，已跳过")

    # 7. 回测（窗口 = 配置 ∩ 样本外区间）
    score_min, score_max = pd.Timestamp(oos["date"].min()), pd.Timestamp(oos["date"].max())
    bt_start = max(pd.Timestamp(cfg.backtest.start), score_min)
    bt_end = min(pd.Timestamp(cfg.backtest.end), score_max)
    if bt_start >= bt_end:
        bt_start, bt_end = score_min, score_max
    engine = BacktestEngine(bundle.prices, bundle.benchmark, cfg.backtest, cfg.portfolio)
    bt = engine.run(
        target_weights,
        regime,
        signal_health=signal_health,
        smallcap_regime=smallcap_regime,
        start=bt_start,
        end=bt_end,
    )
    bt.equity.to_parquet(art / "equity.parquet", index=False)
    bt.trades.to_parquet(art / "trades.parquet", index=False)
    bt.monthly.to_parquet(art / "monthly.parquet", index=False)

    # 8. 指标 + 压力测试
    metrics = compute_metrics(bt.equity, bt.monthly, bt.trades)
    # P0: OOS 模型性能基线（仅 Walk-Forward 样本外折，避免用 IS 指标麻痹判断）
    oos_model = {"oos_folds": int(fold_metrics["fold"].nunique()) if "fold" in fold_metrics.columns else 0}
    if "auc" in fold_metrics.columns and fold_metrics["auc"].notna().any():
        oos_model["oos_auc_mean"] = round(float(fold_metrics["auc"].mean()), 4)
    if "rank_ic_excess" in fold_metrics.columns:
        oos_ric = fold_metrics["rank_ic_excess"].dropna()
        if len(oos_ric):
            oos_model["oos_rank_ic_mean"] = round(float(oos_ric.mean()), 5)
            oos_model["oos_rank_ic_std"] = round(float(oos_ric.std()), 5)
            if oos_ric.std() > 0:
                oos_model["oos_rank_ic_ir"] = round(float(oos_ric.mean() / oos_ric.std()), 3)
            else:
                oos_model["oos_rank_ic_ir"] = None
    metrics["oos_model"] = oos_model
    stress = run_stress(bt.equity, cfg.stress.scenarios)
    bt.metrics, bt.stress = metrics, stress
    with open(art / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "stress": stress}, f, ensure_ascii=False, indent=2, default=str)

    # 9. 报告
    report_path = build_html_report(
        title=cfg.report.title,
        output_path=Path(cfg.report.output_dir) / f"{run_id}.html",
        metrics=metrics,
        equity=bt.equity,
        monthly=bt.monthly,
        stress=stress,
        ic_frame=ic_frame,
        trades=bt.trades,
        config_snapshot=to_dict(cfg),
    )
    log.info("报告: %s", report_path)
    # P2 模型漂移监控：基于 OOS 日度 IC 序列
    drift = None
    try:
        from quant.monitor.drift import ModelDriftMonitor

        drift_mon = ModelDriftMonitor(window=min(63, max(10, len(oos) // 10 or 1)))
        daily_ic = oos.copy()
        daily_ic["date"] = pd.to_datetime(daily_ic["date"])
        daily_ic = daily_ic.sort_values("date")

        def _daily_rank_ic(g: pd.DataFrame) -> float:
            mask = g["excess"].notna() & g["score"].notna()
            if mask.sum() < 10:
                return float("nan")
            return float(pd.Series(g.loc[mask, "score"]).rank().corr(g.loc[mask, "excess"].rank()))

        ic_by_date = daily_ic.groupby("date", group_keys=False).apply(_daily_rank_ic).dropna()
        for ic_val in ic_by_date:
            drift_mon.update(ic=ic_val, auc=None)
        drift = drift_mon.check()
        if drift:
            with open(art / "drift.json", "w", encoding="utf-8") as f:
                json.dump(drift, f, ensure_ascii=False, indent=2, default=str)
        log.info("模型漂移状态: %s", drift.get("status") if drift else "n/a")
    except Exception as exc:  # noqa: BLE001
        log.warning("漂移监控失败: %s", exc)

    return {
        "run_id": run_id,
        "artifacts": str(art),
        "report": str(report_path),
        "metrics": metrics,
        "stress": stress,
        "ic_report": ic_report,
        "target_weights": target_weights,
        "equity": bt.equity,
        "trades": bt.trades,
        "drift": drift,
    }


def live_signals(cfg: Config, output_root: str | Path = "artifacts") -> pd.DataFrame:
    """最新数据 → 因子 → 最新模型推理 → 今日调仓信号。"""
    cfg.validate()
    log = setup_logging(cfg.run.verbose)
    storage = Storage(cfg.data.root)
    bundle = storage.load_bundle()
    bundle.validate()
    registry = ModelRegistry(output_root)
    # 只认已写完 meta.json 的完整模型产物（训练中断的半成品不得参与推理）
    runs = sorted(
        p.name
        for p in registry.root.iterdir()
        if p.is_dir() and (p / "meta.json").exists()
    )
    if not runs:
        raise RuntimeError("无已训练模型，请先运行 --step model")
    sfp = cfg.strategy_fingerprint()
    latest = None
    mismatched: list[str] = []
    # 新 → 旧扫描：取第一个策略指纹与当前配置一致的完整模型。
    # 此前固定取 sorted[-1] 再硬失败——重训产物若写到其他 output_root，
    # 每日任务会拿到陈旧模型并整体失败（2026-08-26 事故根因）。
    for run_name in reversed(runs):
        meta_cand = registry.load_run(run_name).get("meta") or {}
        cand_sfp = meta_cand.get("strategy_fingerprint")
        if not cand_sfp:
            log.warning("模型 %s 缺少 strategy_fingerprint（旧产物），跳过", run_name)
            mismatched.append(run_name)
            continue
        if cand_sfp == sfp:
            latest = registry.load_run(run_name)
            meta = meta_cand
            if run_name != runs[-1]:
                log.warning(
                    "最新模型 %s 指纹不匹配已跳过，回退使用 %s（建议尽快用当前配置重训到同一 output_root）",
                    runs[-1], run_name,
                )
            break
        mismatched.append(run_name)
    if latest is None:
        raise RuntimeError(
            f"无策略指纹匹配的可用模型：{len(mismatched)} 个候选全部不一致"
            f"（期望 {sfp}）。请用当前配置执行 --step model 重训后再出信号。"
        )
    # 模型上线门禁：OOS 质量阈值（默认复合因子 t≥5），不达标按配置告警或硬失败
    from quant.model.gate import evaluate_model_gate

    gate = evaluate_model_gate(cfg, meta)
    model_run_id = meta.get("run_id", "?")
    gate["model"] = model_run_id
    latest_date_gate = pd.Timestamp(bundle.prices["date"].max())
    sig_dir = ensure_dir(Path(output_root) / "signals")
    (sig_dir / f"gate_{latest_date_gate.date()}.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    if cfg.model.gate_enabled and not gate["passed"]:
        detail = ", ".join(gate["failed_checks"])
        if cfg.model.gate_block_on_fail:
            raise RuntimeError(
                f"模型上线门禁未通过（{detail}）：模型 {model_run_id} 不满足当前配置阈值，"
                "禁止出信号。请按当前配置重训，或调整 model.gate_* 阈值。"
            )
        log.warning("模型上线门禁未通过（%s），当前为告警模式，继续出信号", detail)
    models = latest.get("models") or {}
    if not models and latest.get("model") is None:
        raise RuntimeError(f"模型 {model_run_id} 缺少 model.joblib")
    feature_cols = meta.get("feature_cols", [])

    factor_long, cov = compute_all_factors(bundle, cfg, report_coverage=True)
    # 特征包含 composite 时需同步构建复合因子（与 run_research 准入逻辑一致）
    if any(c.replace("f_", "") == "composite" for c in feature_cols):
        from quant.factors.analysis import factor_ic_report
        from quant.factors.composite import build_composite_factor
        from quant.model.label import build_label

        label_long = build_label(bundle.prices, bundle.benchmark, cfg)
        ic_report = factor_ic_report(factor_long, label_long, cfg)
        regime_ctx = None
        if bool(getattr(cfg.factors, "composite_regime_rotate", False)) and not bundle.benchmark.empty:
            # 与研究路径同构：全历史点内分域（推理日已知当日收盘前状态用昨日值，
            # 这里用截至最新数据的状态划分，无未来函数）
            bench_close = bundle.benchmark.set_index("date")["close"].sort_index()
            ma = bench_close.rolling(
                int(getattr(cfg.factors, "composite_regime_ma", 120)), min_periods=60
            ).mean()
            bull_all = (bench_close > ma).dropna()
            bull_dates = {pd.Timestamp(d) for d in bull_all[bull_all].index}
            din = pd.to_datetime(factor_long["date"])
            f_bull = factor_long[din.isin(pd.to_datetime(sorted(bull_dates)))]
            l_bull = label_long[pd.to_datetime(label_long["date"]).isin(pd.to_datetime(sorted(bull_dates)))]
            f_bear = factor_long[~din.isin(pd.to_datetime(sorted(bull_dates)))]
            l_bear = label_long[~pd.to_datetime(label_long["date"]).isin(pd.to_datetime(sorted(bull_dates)))]
            ic_bull = factor_ic_report(f_bull, l_bull, cfg) if len(f_bull) and len(l_bull) else None
            ic_bear = factor_ic_report(f_bear, l_bear, cfg) if len(f_bear) and len(l_bear) else None
            regime_ctx = (bull_dates, ic_bull, ic_bear)
        composite = build_composite_factor(
            factor_long,
            ic_report,
            n=cfg.factors.composite_n,
            min_t=cfg.factors.min_t_stat,
            corr_max=cfg.factors.composite_corr_max,
            require_stable_decay=cfg.factors.composite_require_decay,
            weight_by=cfg.factors.composite_weight,
            regime_bull_dates=regime_ctx[0] if regime_ctx else None,
            ic_report_bull=regime_ctx[1] if regime_ctx else None,
            ic_report_bear=regime_ctx[2] if regime_ctx else None,
        )
        if not composite.empty:
            log.info("复合因子已构建（成分 %d 个）", composite["factor"].nunique())
            factor_long = pd.concat([factor_long, composite], ignore_index=True)
    latest_date = factor_long["date"].max()
    feats = factor_long[factor_long["date"] == latest_date].pivot(
        index="symbol", columns="factor", values="value"
    )
    feats = feats[[c.replace("f_", "") for c in feature_cols]]
    feats.columns = feature_cols
    feats = feats.reindex(columns=feature_cols)
    feats = feats.fillna(0.0)
    X = feats.to_numpy(dtype=float)
    proba_cols = {}
    if models:
        for h, model in models.items():
            if model is not None:
                proba_cols[f"h{h}"] = model.predict_proba(X)[:, 1]
    else:
        proba_cols["h1"] = latest["model"].predict_proba(X)[:, 1]
    # 部署校准：与研究流程的 OOF 校准保持一致（校准后再做截面分位）
    raw_proba_cols = dict(proba_cols)
    calibrator = latest.get("calibrator")
    if calibrator is not None:
        for col in list(proba_cols):
            proba_cols[col] = calibrator.transform(proba_cols[col])
    proba_df = pd.DataFrame(proba_cols, index=feats.index)
    # 多周期集成：截面分位等权平均
    ensemble = proba_df.rank(pct=True).mean(axis=1)
    # 次级排序键：校准前原始集成分。isotonic 阶梯会把大量股票映射到同一
    # 分数档（1052 只仅 139 档），top-N 截断在并列组内退化为任意选择；
    # 原始概率是连续值，可确定性地恢复组内顺序，不改变组间校准序。
    raw_ensemble = pd.DataFrame(raw_proba_cols, index=feats.index).rank(pct=True).mean(axis=1)
    signals = pd.DataFrame(
        {"symbol": feats.index, "date": latest_date, "score": ensemble, "raw_score": raw_ensemble}
    )
    signals = signals.sort_values(
        ["score", "raw_score"], ascending=False
    ).reset_index(drop=True)
    out_dir = ensure_dir(Path(output_root) / "signals")
    signals.to_csv(out_dir / f"signals_{latest_date.date()}.csv", index=False)
    signals.to_json(out_dir / f"signals_{latest_date.date()}.json", orient="records", indent=2)
    # 因子覆盖度快照（原始值口径），供每日哨兵检测数据源失效
    cov_recent = cov[pd.to_datetime(cov["date"]) >= pd.to_datetime(cov["date"]).max() - pd.Timedelta(days=35)]
    cov_summary = {}
    for f, g in cov_recent.groupby("factor"):
        latest_row = g.loc[g["date"].idxmax()]
        cov_summary[f] = {
            "coverage_ratio_latest": round(float(latest_row["coverage_ratio"]), 4),
            "coverage_ratio_mean_20d": round(float(g["coverage_ratio"].tail(20).mean()), 4),
            "n_symbols": int(latest_row["n_symbols"]),
        }
    (out_dir / f"factor_coverage_{latest_date.date()}.json").write_text(
        json.dumps(
            {"date": str(latest_date.date()), "factors": cov_summary},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    log.info("今日信号已生成: %d 只, 日期 %s", len(signals), latest_date.date())
    return signals
