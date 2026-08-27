"""回测引擎测试：成本、撮合、T+1、约束、指标。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.backtest.cost import CostModel
from quant.backtest.engine import BacktestEngine
from quant.backtest.fills import FillSimulator
from quant.config import BacktestConfig, Config, PortfolioConfig
from quant.metrics.performance import compute_metrics, max_drawdown


def _mini_prices() -> pd.DataFrame:
    dates = pd.bdate_range("2023-01-02", periods=60)
    rows = []
    for i, sym in enumerate(["A.SH", "B.SH", "C.SH", "D.SH"]):
        close = 10.0 * (1 + np.linspace(0, 0.3, len(dates)) * (i % 2 + 1) / 2)
        open_p = close * 0.999
        rows.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "symbol": sym,
                    "open": open_p,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "volume": 1e6,
                    "amount": 1e6 * close * 100,
                    "turnover": 1.0,
                    "is_limit_up": False,
                    "is_limit_down": False,
                    "is_suspended": False,
                    "is_st": False,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def test_cost_model():
    c = CostModel()
    buy_fee = c.buy_fee(100_000)
    sell_fee = c.sell_fee(100_000)
    assert sell_fee > buy_fee  # 印花税
    assert c.round_lot(1234) == 1200
    assert c.buy_fee(1000) >= c.min_commission


def test_fill_limit_rules():
    prices = _mini_prices()
    # 制造 A.SH 在某日涨停、B.SH 跌停
    idx = prices.index[(prices["date"] == prices["date"].unique()[5]) & (prices["symbol"] == "A.SH")]
    prices.loc[idx, "is_limit_up"] = True
    idx2 = prices.index[(prices["date"] == prices["date"].unique()[6]) & (prices["symbol"] == "B.SH")]
    prices.loc[idx2, "is_limit_down"] = True
    sim = FillSimulator(prices, CostModel())
    d5 = prices["date"].unique()[5]
    r = sim.try_fill("A.SH", d5, "buy", 1000, 5)
    assert r.status == "skipped"
    d6 = prices["date"].unique()[6]
    r2 = sim.try_fill("B.SH", d6, "sell", 1000, 5)
    assert r2.status == "postponed"
    r3 = sim.try_fill("B.SH", d6, "sell", 1000, 0)
    assert r3.status == "dropped"


def test_fill_open_limit_rules():
    """开盘一字板（open vs preclose）应阻止买入/卖出，且 NaN 标记不误判。"""
    prices = _mini_prices()
    d = prices["date"].unique()[5]
    mask = (prices["date"] == d) & (prices["symbol"] == "A.SH")
    prices.loc[mask, "is_limit_up_open"] = True
    prices.loc[mask, "is_limit_down_open"] = False
    prices.loc[mask, "is_limit_up"] = False
    mask2 = (prices["date"] == d) & (prices["symbol"] == "B.SH")
    prices.loc[mask2, "is_limit_down_open"] = True
    prices.loc[mask2, "is_limit_up_open"] = False
    # 老数据缺列：不设列，验证 fallback 到收盘标记
    sim = FillSimulator(prices, CostModel())
    assert sim.try_fill("A.SH", d, "buy", 1000, 5).status == "skipped"
    assert sim.try_fill("B.SH", d, "sell", 1000, 5).status == "postponed"
    # NaN 标记不触发误判（bool(np.nan) 为 True 的经典坑）
    prices2 = _mini_prices()
    mask3 = (prices2["date"] == d) & (prices2["symbol"] == "C.SH")
    prices2.loc[mask3, "is_limit_up_open"] = np.nan
    prices2.loc[mask3, "is_limit_down_open"] = np.nan
    sim2 = FillSimulator(prices2, CostModel())
    assert sim2.try_fill("C.SH", d, "buy", 1000, 5).status == "filled"


def _engine_with_prices(prices):
    bench = pd.DataFrame(
        {"date": pd.unique(prices["date"]), "close": np.linspace(1000, 1300, len(pd.unique(prices["date"])))}
    )
    cfg = Config()
    bt_cfg, pf_cfg = cfg.backtest, cfg.portfolio
    bt_cfg.start, bt_cfg.end = str(prices["date"].min().date()), str(prices["date"].max().date())
    return BacktestEngine(prices, bench, bt_cfg, pf_cfg)


def test_duplicate_sell_no_phantom_cash():
    """同日多笔卖出（止损 + 调仓并发）不得重复入账（幽灵现金）。"""
    prices = _mini_prices()
    engine = _engine_with_prices(prices)
    d = prices["date"].unique()[10]
    positions = {"A.SH": {"shares": 100, "cost": 1000.0, "entry": 10.0, "available": 100}}
    cash, queue, trades, no_buy = 0.0, [], [], set()
    o1 = {"symbol": "A.SH", "side": "sell", "shares": 100, "exec_date": d, "days_left": 5, "reason": "stop_loss"}
    o2 = {"symbol": "A.SH", "side": "sell", "shares": 100, "exec_date": d, "days_left": 5, "reason": "rebalance_out"}
    cash = engine._execute(o1, d, positions, cash, queue, trades, no_buy)
    assert positions == {}  # 第一笔卖出清仓
    cash_after = engine._execute(o2, d, positions, cash, queue, trades, no_buy)
    assert cash_after == cash  # 第二笔不产生任何现金
    assert trades[-1]["status"] == "dropped"
    assert trades[-1]["reason"] == "no_position_or_available"


def test_buy_postponed_retries():
    """停牌买入应顺延重试，而非静默丢弃。"""
    prices = _mini_prices()
    d0, d1 = prices["date"].unique()[5], prices["date"].unique()[6]
    mask = (prices["date"] == d0) & (prices["symbol"] == "A.SH")
    prices.loc[mask, "is_suspended"] = True
    prices.loc[mask, ["open", "high", "low", "close"]] = np.nan
    engine = _engine_with_prices(prices)
    positions, cash, queue, trades, no_buy = {}, 10_000.0, [], [], set()
    order = {"symbol": "A.SH", "side": "buy", "shares": 100, "exec_date": d0, "days_left": 3, "reason": "rebalance"}
    cash = engine._execute(order, d0, positions, cash, queue, trades, no_buy)
    assert len(queue) == 1 and queue[0]["days_left"] == 2 and queue[0]["exec_date"] == d1
    cash = engine._execute(queue.pop(0), d1, positions, cash, queue, trades, no_buy)
    assert positions["A.SH"]["shares"] == 100
    assert cash < 10_000.0


def test_no_buy_after_stop_same_day():
    """止损卖出当日禁止回补同一标的（防同日卖出后再买入）。"""
    prices = _mini_prices()
    engine = _engine_with_prices(prices)
    d = prices["date"].unique()[10]
    positions = {"A.SH": {"shares": 100, "cost": 1000.0, "entry": 10.0, "available": 100}}
    cash, queue, trades, no_buy = 5_000.0, [], [], set()
    sell = {"symbol": "A.SH", "side": "sell", "shares": 100, "exec_date": d, "days_left": 5, "reason": "stop_loss"}
    cash = engine._execute(sell, d, positions, cash, queue, trades, no_buy)
    assert "A.SH" in no_buy
    buy = {"symbol": "A.SH", "side": "buy", "shares": 100, "exec_date": d, "days_left": 5, "reason": "rebalance"}
    cash2 = engine._execute(buy, d, positions, cash, queue, trades, no_buy)
    assert cash2 == cash
    assert trades[-1]["reason"] == "no_buy_after_stop"


def test_engine_end_to_end():
    prices = _mini_prices()
    bench = pd.DataFrame({"date": pd.unique(prices["date"]), "close": np.linspace(1000, 1300, 60)})
    cfg = Config()
    bt_cfg = cfg.backtest
    pf_cfg = cfg.portfolio
    bt_cfg.start, bt_cfg.end = str(prices["date"].min().date()), str(prices["date"].max().date())
    pf_cfg.top_n = 3
    # 月末调仓：持有 A/B/C
    dates = pd.DatetimeIndex(pd.unique(prices["date"]))
    month_ends = dates.to_series().groupby(dates.to_period("M")).max()
    tw = pd.DataFrame(
        {
            "date": [d for d in month_ends.tolist() for _ in range(3)],
            "symbol": ["A.SH", "B.SH", "C.SH"] * len(month_ends),
            "weight": [1 / 3] * (3 * len(month_ends)),
        }
    )
    engine = BacktestEngine(prices, bench, bt_cfg, pf_cfg)
    res = engine.run(tw, start=bt_cfg.start, end=bt_cfg.end)
    assert not res.equity.empty
    assert (res.equity["portfolio_value"] > 0).all()
    assert (res.equity["cash"] >= 0).all()
    if not res.trades.empty:
        assert res.trades["status"].isin(["filled", "skipped", "postponed", "dropped"]).all()
        filled = res.trades[res.trades["status"] == "filled"]
        assert (filled["shares"] % bt_cfg.lot_size == 0).all()


def test_metrics_correctness():
    equity = pd.DataFrame(
        {
            "date": pd.bdate_range("2023-01-02", periods=60),
            "portfolio_value": np.linspace(100, 120, 60),
            "benchmark_value": np.linspace(100, 110, 60),
            "cash": 0.0,
            "position_count": 3,
        }
    )
    monthly = pd.DataFrame(
        {"date": pd.date_range("2023-01-31", periods=2, freq="ME"),
         "return": [0.03, 0.04], "benchmark_return": [0.01, 0.01]}
    )
    trades = pd.DataFrame(
        {"side": ["sell", "sell"], "status": ["filled", "filled"],
         "price": [11.0, 12.0], "shares": [100, 100],
         "entry_price": [10.0, 13.0], "fee": [1.0, 1.0]}
    )
    m = compute_metrics(equity, monthly, trades)
    assert m["annualized_return"] > 0
    assert m["monthly_win_rate"] == 1.0
    assert m["sell_hit_rate"] == 0.5
    assert max_drawdown(equity["portfolio_value"]) >= -1e-9


def test_regime_levels():
    from quant.portfolio.regime import compute_regime

    cfg = Config()
    idx = pd.bdate_range("2022-01-03", periods=500)
    close = pd.Series(np.linspace(100, 200, len(idx)), index=idx)
    reg = compute_regime(close, cfg.portfolio)
    assert "target_exposure" in reg.columns
    assert reg["target_exposure"].dropna().between(0, 1).all()
