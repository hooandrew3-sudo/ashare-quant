"""验证 _turnover_breached 逻辑。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
from quant.config import PortfolioConfig
from quant.backtest.engine import BacktestEngine


def make_engine():
    eng = object.__new__(BacktestEngine)
    eng.pf = PortfolioConfig()
    eng.pf.max_turnover_annual = 4.0
    return eng


def test_breached():
    # 200 笔 × 50000 元成交，净值 100 万 → 换手 10x（双边 200*50000/1e6 = 10x）
    trade_rows = [
        {"date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i * 2),
         "status": "filled", "price": 50.0, "shares": 1000}
        for i in range(200)
    ]
    equity_rows = [
        {"date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i * 2),
         "portfolio_value": 1_000_000.0}
        for i in range(200)
    ]
    eng = make_engine()
    breached = BacktestEngine._turnover_breached(eng, trade_rows, equity_rows)
    print(f"breached (expect True): {breached}")
    assert breached is True
    return True


def test_not_breached():
    # 少量成交 → 换手低
    trade_rows = [
        {"date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i * 30),
         "status": "filled", "price": 50.0, "shares": 100}
        for i in range(12)
    ]
    equity_rows = [
        {"date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i * 30),
         "portfolio_value": 1_000_000.0}
        for i in range(12)
    ]
    eng = make_engine()
    breached = BacktestEngine._turnover_breached(eng, trade_rows, equity_rows)
    print(f"breached (expect False): {breached}")
    assert breached is False
    return True


def test_insufficient_data():
    eng = make_engine()
    breached = BacktestEngine._turnover_breached(eng, [], [])
    print(f"breached with empty (expect False): {breached}")
    assert breached is False
    return True


if __name__ == "__main__":
    test_breached()
    test_not_breached()
    test_insufficient_data()
    print("\nAll turnover breaker tests passed.")
