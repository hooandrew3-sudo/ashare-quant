"""因子层测试：计算形状、预处理、IC 报告。"""

from __future__ import annotations

import pandas as pd

from quant.config import Config
from quant.data.synthetic import generate_synthetic
from quant.factors.analysis import factor_ic_report, report_to_frame
from quant.factors.compute import compute_all_factors
from quant.model.label import build_label


def _cfg() -> Config:
    cfg = Config()
    cfg.data.demo.n_stocks = 40
    cfg.data.demo.years = 4
    cfg.run.seed = 11
    return cfg


def test_factor_computation_shapes():
    cfg = _cfg()
    bundle = generate_synthetic(
        n_stocks=cfg.data.demo.n_stocks,
        years=cfg.data.demo.years,
        seed=cfg.run.seed,
    )
    factor_long = compute_all_factors(bundle, cfg)
    assert set(["date", "symbol", "factor", "value"]).issubset(factor_long.columns)
    n_factors = factor_long["factor"].nunique()
    # 13 个基础因子 + consensus_revision（分析师一致预期修正，数据不足时恒 0）
    assert n_factors == 14
    assert factor_long["value"].notna().mean() > 0.5


def test_ic_report_structure():
    cfg = _cfg()
    bundle = generate_synthetic(
        n_stocks=cfg.data.demo.n_stocks,
        years=cfg.data.demo.years,
        seed=cfg.run.seed,
    )
    factor_long = compute_all_factors(bundle, cfg)
    label_long = build_label(bundle.prices, bundle.benchmark, cfg)
    report = factor_ic_report(factor_long, label_long, cfg)
    assert "factors" in report
    assert len(report["factors"]) == 14  # 13 基础 + consensus_revision
    frame = report_to_frame(report)
    assert set(["factor", "ic_mean", "icir", "passed"]).issubset(frame.columns)


def test_factor_direction_negated_factors_positive():
    """已取负的因子在 compute 里已做「高值=更优」翻转，direction 应为 +1，避免双重取负。"""
    from quant.factors.definitions import factor_direction

    for name in ("low_vol", "crowding", "illiquidity", "max_ret", "size_proxy", "rev_5"):
        assert factor_direction(name) == 1, name
