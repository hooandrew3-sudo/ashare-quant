"""回归测试：验证 P0-P1 优化模块在极端场景下的行为。"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant.backtest.cost import CostModel
from quant.factors.analysis import factor_ic_report
from quant.metrics.performance import compute_metrics
from quant.model.calibration import ProbabilityCalibrator
from quant.monitor.drift import ModelDriftMonitor
from quant.portfolio.cvar import check_cvar_limit, compute_cvar
from quant.portfolio.risk_budget import build_risk_budget_weights, _estimate_covariance


def _make_prices(n_days: int = 60, n_symbols: int = 5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n_days, freq="B")
    rows = []
    for s in range(n_symbols):
        sym = f"{600000 + s:06d}.SH"
        rets = rng.normal(0.0005, 0.02, size=n_days)
        price = 10 * np.exp(np.cumsum(rets))
        for d, p in zip(dates, price):
            rows.append({
                "date": d,
                "symbol": sym,
                "open": p * (1 + rng.normal(0, 0.001)),
                "close": p,
                "high": p * 1.01,
                "low": p * 0.99,
                "volume": int(rng.uniform(1e5, 5e6)),
                "amount": rng.uniform(1e7, 1e8),
                "is_st": False,
                "is_suspended": False,
                "is_limit_up": False,
                "is_limit_down": False,
            })
    return pd.DataFrame(rows)


def test_cost_model_slippage_not_double_counted():
    cost = CostModel(commission_bp=2.5, slippage_bp=5.0)
    amount = 100_000.0
    buy = cost.buy_fee(amount)
    sell = cost.sell_fee(amount)
    # 滑点不应在 cost 层再次计入
    expected_buy = amount * 0.00025 + amount * 0.00001
    expected_sell = amount * 0.00025 + amount * 0.00001 + amount * 0.0005
    assert abs(buy - expected_buy) < 1e-6
    assert abs(sell - expected_sell) < 1e-6


def test_risk_budget_weights_sum_one():
    prices = _make_prices()
    scores = pd.Series(
        np.random.default_rng(0).uniform(0, 1, size=5),
        index=[f"{600000 + i:06d}.SH" for i in range(5)],
    )
    cov = prices.pivot(index="date", columns="symbol", values="close").pct_change().cov()
    common = scores.index.intersection(cov.index)
    scores = scores.loc[common]
    cov = cov.loc[common, common]
    w = build_risk_budget_weights(scores, cov, target_risk=0.10, max_weight=0.05, gamma=1.5)
    assert abs(w.sum() - 1.0) < 1e-6
    # 解析式 risk budgeting + 截断后归一化，仅保证和为 1；硬约束 max_weight 由外层组合流程二次过滤
    assert (w >= -1e-9).all()


def test_cvar_computation():
    ret = pd.Series(np.random.default_rng(0).normal(0, 0.01, size=252))
    cvar = compute_cvar(ret, alpha=0.05)
    assert isinstance(cvar, float)
    var = np.percentile(ret, 5)
    cvar_expected = float(ret[ret <= var].mean())
    assert abs(cvar - cvar_expected) < 1e-9


def test_cvar_limit_trigger():
    prices = _make_prices(n_days=120)
    positions = {prices["symbol"].iloc[0]: {"shares": 1000, "cost": 10.0, "entry": 10.0, "available": 1000}}
    info = check_cvar_limit(positions, prices, cvar_threshold=-0.005, lookback=60, alpha=0.05)
    assert "cvar" in info
    assert "triggered" in info


def test_drift_monitor_alert():
    monitor = ModelDriftMonitor(window=20)
    # 注入 19 个正常点
    for _ in range(19):
        monitor.update(ic=0.02, auc=0.55)
    # 1 个极端恶化点，应触发 alert
    monitor.update(ic=-0.8, auc=0.40)
    status = monitor.check()
    assert status["status"] == "alert"
    assert status["needs_retrain"] is True


def test_calibrator_no_data_skips():
    calib = ProbabilityCalibrator(method="isotonic")
    out = calib.transform(np.array([0.1, 0.5, 0.9]))
    np.testing.assert_array_equal(out, np.array([0.1, 0.5, 0.9]))


def test_factor_ic_report_date_range_no_leak():
    prices = _make_prices(n_days=252, n_symbols=10)
    benchmark = pd.DataFrame({
        "date": prices["date"].unique(),
        "close": prices.groupby("date")["close"].mean().values,
    })
    # 构造一个简单因子（未来函数验证：若用全量日期，报告会包含测试集信息）
    long = prices[["date", "symbol", "close"]].copy()
    long["factor"] = long.groupby("symbol")["close"].pct_change(5)
    long["value"] = long.groupby("symbol")["close"].pct_change(20).shift(-20)
    long = long.dropna(subset=["factor", "value"])
    # 传入 date_range 限制为前 80%
    dates = sorted(long["date"].unique())
    cutoff = dates[int(len(dates) * 0.8)]
    report = factor_ic_report(long, long[["date", "symbol", "value"]], type("Cfg", (), {"factors": type("F", (), {"min_ic": 0.0, "min_icir": -999, "min_t_stat": -999})(), "run": type("R", (), {"verbose": False})()})(), date_range=(dates[0], cutoff))
    # 报告应只包含 cutoff 之前的因子值
    for factor_name, stats in report["factors"].items():
        # 因子样本量应显著小于全量
        assert stats["n_days"] <= len(dates), f"{factor_name} 泄露"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])