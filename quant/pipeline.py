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
    storage.save("meta", bundle.meta, partition_by_symbol=False)
    storage.save("industry", bundle.industry, partition_by_symbol=False)
    storage.save("fundamentals", bundle.fundamentals, partition_by_symbol=False)
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
    factor_long = compute_all_factors(bundle, cfg)

    # 3. 标签 + IC 报告（完整报告用于研究归档）
    label_long = build_label(bundle.prices, bundle.benchmark, cfg)
    # P0 防泄露：提前计算 in-sample cutoff（供 composite 和因子筛选共用）
    all_dates = pd.to_datetime(factor_long["date"]).sort_values().unique()
    train_cutoff = None
    if len(all_dates) > 252:
        cutoff_idx = int(len(all_dates) * 0.6)
        train_cutoff = pd.Timestamp(all_dates[cutoff_idx])
    ic_report = factor_ic_report(factor_long, label_long, cfg)
    # 3.1 复合因子：方向显著因子等权合成，再纳入 IC 报告与特征
    if cfg.factors.composite:
        from quant.factors.composite import build_composite_factor

        ic_report_for_composite = ic_report
        if train_cutoff is not None:
            f_in = factor_long[pd.to_datetime(factor_long["date"]) <= train_cutoff]
            l_in = label_long[pd.to_datetime(label_long["date"]) <= train_cutoff]
            ic_report_for_composite = factor_ic_report(f_in, l_in, cfg)
        composite = build_composite_factor(
            factor_long,
            ic_report_for_composite,
            n=cfg.factors.composite_n,
            min_t=cfg.factors.min_t_stat,
            corr_max=cfg.factors.composite_corr_max,
            require_stable_decay=cfg.factors.composite_require_decay,
            weight_by=cfg.factors.composite_weight,
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
    all_dates = pd.to_datetime(factor_long["date"]).sort_values().unique()
    train_cutoff = None
    if len(all_dates) > 252:
        cutoff_idx = int(len(all_dates) * 0.6)
        train_cutoff = pd.Timestamp(all_dates[cutoff_idx])
        log.info("因子准入筛选 cutoff: %s（保留前 %.0f%% 时间）", train_cutoff.date(), 60)
        f_in = factor_long[pd.to_datetime(factor_long["date"]) <= train_cutoff]
        l_in = label_long[pd.to_datetime(label_long["date"]) <= train_cutoff]
        ic_report_in = factor_ic_report(f_in, l_in, cfg)
        ranked_in = sorted(
            ic_report_in["factors"].items(),
            key=lambda kv: abs(kv[1]["rank_ic_mean"] or 0), reverse=True,
        )
        passed = ic_report_in["passed"]
    else:
        ranked_in = sorted(
            ic_report["factors"].items(),
            key=lambda kv: abs(kv[1]["rank_ic_mean"] or 0), reverse=True,
        )
        passed = ic_report["passed"]
    # 准入因子 + 复合因子（如已生成）强制入特征，再按 |IC| 补齐 ≥3 个
    selected = list(passed)
    if cfg.factors.composite and "composite" in set(factor_long["factor"].unique()):
        if "composite" not in selected:
            selected.append("composite")
    min_features = max(3, len(selected))
    ranked_source = ranked_in if train_cutoff is not None else ranked_in
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
                train_idx = oos_flat["fold"] != f
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
        )
        if cfg.portfolio.weight_method == "risk_budget"
        else build_target_weights(
            scores=oos,
            prices=bundle.prices,
            industry=bundle.industry,
            cfg=cfg.portfolio,
            rebalance_dates=rebalance_dates,
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
    latest = registry.load_run(runs[-1])
    meta = latest.get("meta") or {}
    sfp = meta.get("strategy_fingerprint")
    if sfp:
        if sfp != cfg.strategy_fingerprint():
            raise RuntimeError(
                f"模型 {runs[-1]} 的策略指纹与当前配置不一致（{sfp} vs "
                f"{cfg.strategy_fingerprint()}）。因子/模型/组合参数已变更，"
                "请按当前配置重新训练后再出信号。"
            )
    else:
        # 旧版本产物无 strategy_fingerprint：降级为告警而非阻断，
        # 避免每日任务因历史模型归档缺失而中断；强烈建议重训以启用严格门禁。
        log.warning(
            "模型 %s 缺少 strategy_fingerprint（旧版本产物），跳过严格策略指纹校验；"
            "建议用当前配置重训一次以启用门禁",
            runs[-1],
        )
    models = latest.get("models") or {}
    if not models and latest.get("model") is None:
        raise RuntimeError(f"模型 {runs[-1]} 缺少 model.joblib")
    feature_cols = meta.get("feature_cols", [])

    factor_long = compute_all_factors(bundle, cfg)
    # 特征包含 composite 时需同步构建复合因子（与 run_research 准入逻辑一致）
    if any(c.replace("f_", "") == "composite" for c in feature_cols):
        from quant.factors.analysis import factor_ic_report
        from quant.factors.composite import build_composite_factor
        from quant.model.label import build_label

        label_long = build_label(bundle.prices, bundle.benchmark, cfg)
        ic_report = factor_ic_report(factor_long, label_long, cfg)
        composite = build_composite_factor(
            factor_long,
            ic_report,
            n=cfg.factors.composite_n,
            min_t=cfg.factors.min_t_stat,
            corr_max=cfg.factors.composite_corr_max,
            require_stable_decay=cfg.factors.composite_require_decay,
            weight_by=cfg.factors.composite_weight,
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
    calibrator = latest.get("calibrator")
    if calibrator is not None:
        for col in list(proba_cols):
            proba_cols[col] = calibrator.transform(proba_cols[col])
    proba_df = pd.DataFrame(proba_cols, index=feats.index)
    # 多周期集成：截面分位等权平均
    ensemble = proba_df.rank(pct=True).mean(axis=1)
    signals = pd.DataFrame({"symbol": feats.index, "date": latest_date, "score": ensemble})
    signals = signals.sort_values("score", ascending=False).reset_index(drop=True)
    out_dir = ensure_dir(Path(output_root) / "signals")
    signals.to_csv(out_dir / f"signals_{latest_date.date()}.csv", index=False)
    signals.to_json(out_dir / f"signals_{latest_date.date()}.json", orient="records", indent=2)
    log.info("今日信号已生成: %d 只, 日期 %s", len(signals), latest_date.date())
    return signals
