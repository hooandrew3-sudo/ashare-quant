"""因子层测试：计算形状、预处理、IC 报告。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.config import Config
from quant.data.synthetic import generate_synthetic
from quant.factors.analysis import factor_ic_report, report_to_frame
from quant.factors.compute import compute_all_factors
from quant.model.label import build_label
from quant.data.storage import DataBundle


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
    assert n_factors == 19  # 17 + earn_quality/ocf_growth（现金流族）
    assert factor_long["value"].notna().mean() > 0.5


def test_neutralize_batched_equals_old():
    """批量中性化与旧实现（逐因子逐日 lstsq）在稠密输入下数值等价。"""
    from quant.factors.compute import _build_X_by_date, _neutralize_batched

    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2023-01-02", periods=30)
    syms = [f"S{i:02d}" for i in range(50)]
    df1 = pd.DataFrame(rng.normal(size=(30, 50)), index=dates, columns=syms)
    df2 = pd.DataFrame(rng.normal(size=(30, 50)), index=dates, columns=syms)
    industry = pd.Series({s: f"IND{i % 10}" for i, s in enumerate(syms)})
    dummies = pd.get_dummies(industry, prefix="ind").astype(float)
    size_panel = pd.DataFrame(rng.normal(size=(30, 50)), index=dates, columns=syms)
    X_by_date = _build_X_by_date(dates, syms, dummies, size_panel)
    new = _neutralize_batched([df1, df2], X_by_date)

    def old(df):
        out = df.copy()
        for dt, row in out.iterrows():
            X = pd.concat(
                [dummies, size_panel.loc[dt].to_frame("size")], axis=1
            ).fillna(0.0).to_numpy()
            y = row.values.astype(float)
            mask = ~np.isnan(y)
            beta, *_ = np.linalg.lstsq(X[mask], y[mask], rcond=None)
            out.loc[dt] = y - X @ beta
        return out

    np.testing.assert_allclose(new[0].values, old(df1).values, atol=1e-10)
    np.testing.assert_allclose(new[1].values, old(df2).values, atol=1e-10)


def test_factor_coverage_report():
    cfg = _cfg()
    bundle = generate_synthetic(
        n_stocks=cfg.data.demo.n_stocks,
        years=cfg.data.demo.years,
        seed=cfg.run.seed,
    )
    _long, cov = compute_all_factors(bundle, cfg, report_coverage=True)
    assert {"date", "factor", "coverage_ratio", "non_null", "n_symbols"}.issubset(cov.columns)
    assert (cov["coverage_ratio"] >= 0.0).all() and (cov["coverage_ratio"] <= 1.0).all()
    assert cov["factor"].nunique() >= 10


def test_consensus_snapshot_does_not_crash():
    """consensus 快照表（as_of_date 列）不得让因子管线崩溃（回归：2026-08-18）。"""
    cfg = _cfg()
    bundle = generate_synthetic(
        n_stocks=cfg.data.demo.n_stocks,
        years=cfg.data.demo.years,
        seed=cfg.run.seed,
    )
    consensus = pd.DataFrame(
        {
            "symbol": bundle.prices["symbol"].unique()[:10].tolist(),
            "as_of_date": pd.Timestamp("2026-08-10"),
            "year": [2026] * 10,
            "n_institutions": [3] * 10,
            "eps_min": [1.0] * 10,
            "eps_mean": [1.2] * 10,
            "eps_max": [1.5] * 10,
        }
    )
    bundle.consensus = consensus
    factor_long = compute_all_factors(bundle, cfg)
    rev = factor_long[factor_long["factor"] == "consensus_revision"]
    assert not rev.empty
    # 单日快照不足 30 个交易日位移 → 修正值应为 0（中性占位）
    assert (rev["value"] == 0.0).all()


def test_consensus_pip_shift():
    """一致预期快照（盘后采集）最早下一交易日可用：shift(1) 语义。"""
    from quant.factors.definitions import Panels, _f_consensus_revision

    idx = pd.bdate_range("2026-01-02", periods=60)
    close = pd.DataFrame(10.0, index=idx, columns=["A.SH", "B.SH"])
    cons = pd.DataFrame(np.nan, index=idx, columns=["A.SH"])
    cons.loc[idx[0], "A.SH"] = 1.0
    cons.loc[idx[30], "A.SH"] = 1.2
    p = Panels(
        close=close, volume=close, amount=close, turnover=close,
        pe=close, pb=close, roe=close, gross_margin=close,
        div_yield=close, consensus=cons,
    )
    out = _f_consensus_revision(p)
    # 采集日（idx[30]）当天：新快照不可用，且 30 日前快照尚未生效 → NaN
    assert pd.isna(out.loc[idx[30], "A.SH"])
    # 下一交易日：新快照生效 → +20%
    assert abs(out.loc[idx[31], "A.SH"] - 0.2) < 1e-9
    # 历史不足 30 个交易日：保持 NaN（供覆盖度哨兵观测）
    assert pd.isna(out.loc[idx[10], "A.SH"])


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
    assert len(report["factors"]) == 19  # 含现金流质量族
    frame = report_to_frame(report)
    assert set(["factor", "ic_mean", "icir", "passed"]).issubset(frame.columns)


def test_factor_direction_negated_factors_positive():
    """已取负的因子在 compute 里已做「高值=更优」翻转，direction 应为 +1，避免双重取负。"""
    from quant.factors.definitions import factor_direction

    for name in ("low_vol", "crowding", "illiquidity", "max_ret", "size_proxy", "rev_5"):
        assert factor_direction(name) == 1, name
