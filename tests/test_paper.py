"""纸面交易测试：成本/T+1 冻结、同日幂等、每日历史与订单流水。"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from quant.backtest.cost import CostModel
from quant.config import Config
from quant.data.storage import Storage
from quant.execution.broker import Order, OrderStatus
from quant.execution.paper import PaperBroker
from quant.execution.paper_runner import run_paper


def _one_day_prices(date: str = "2026-02-05", close: float = 10.0) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": pd.Timestamp(date),
                "symbol": "A.SH",
                "open": close,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": 1e6,
                "amount": 1e9,
                "turnover": 1.0,
                "is_limit_up": False,
                "is_limit_down": False,
                "is_suspended": False,
                "is_st": False,
            }
        ]
    )


def test_buy_applies_cost_and_slippage() -> None:
    prices = _one_day_prices()
    broker = PaperBroker(prices, initial_cash=100_000.0, cost=CostModel())
    broker.trade_date = "2026-02-05"
    order = Order(id="t_buy_A", symbol="A.SH", side="buy", shares=1000, price=10.0)
    broker.place_order(order)

    assert order.status == OrderStatus.FILLED
    assert order.fee > 0
    assert order.price == 10.0 * (1 + 5 / 10_000)  # 含滑点
    assert broker.get_cash() < 100_000.0 - 10_000.0  # 本金 + 费用
    assert broker.total_fees() > 0
    pos = broker.get_positions()["A.SH"]
    assert pos.shares == 1000
    assert pos.available == 0  # T+1 当日不可卖


def test_t1_settle_unlocks_and_sell_fills() -> None:
    prices = _one_day_prices()
    broker = PaperBroker(prices, initial_cash=100_000.0, cost=CostModel())
    broker.trade_date = "2026-02-05"
    broker.place_order(Order(id="t_buy_A", symbol="A.SH", side="buy", shares=1000, price=10.0))

    early = Order(id="t_sell_early", symbol="A.SH", side="sell", shares=1000, price=10.0)
    broker.place_order(early)
    assert early.status == OrderStatus.REJECTED  # 当日不可卖

    broker.settle("2026-02-06")
    assert broker.get_positions()["A.SH"].available == 1000
    cash_before = broker.get_cash()
    sell = Order(id="t_sell_ok", symbol="A.SH", side="sell", shares=1000, price=10.0)
    broker.place_order(sell)
    assert sell.status == OrderStatus.FILLED
    assert broker.get_cash() > cash_before
    assert "A.SH" not in broker.get_positions()


def _mini_storage(root, n_days: int, closes: dict[str, float]) -> Storage:
    dates = pd.bdate_range("2026-01-05", periods=n_days)
    rows = []
    for sym, close in closes.items():
        rows.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "symbol": sym,
                    "open": close,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "volume": 1e6,
                    "amount": 1e9,
                    "turnover": 1.0,
                    "is_limit_up": False,
                    "is_limit_down": False,
                    "is_suspended": False,
                    "is_st": False,
                }
            )
        )
    prices = pd.concat(rows, ignore_index=True)
    bench = pd.DataFrame(
        {"date": dates, "close": np.linspace(4000.0, 4100.0, len(dates))}
    )
    storage = Storage(root)
    storage.save("prices", prices)
    storage.save("benchmark", bench, partition_by_symbol=False)
    return storage


def _signals(date: str, scores: dict[str, float]) -> pd.DataFrame:
    df = pd.DataFrame(
        [{"symbol": s, "score": v, "date": pd.Timestamp(date)} for s, v in scores.items()]
    )
    return df.sort_values("score", ascending=False).reset_index(drop=True)


def test_runner_idempotent_same_day(tmp_path) -> None:
    root = tmp_path / "data"
    _mini_storage(root, n_days=25, closes={"A.SH": 10.0, "B.SH": 10.0, "C.SH": 10.0, "D.SH": 10.0})
    cfg = Config()
    cfg.data.root = root
    cfg.backtest.initial_cash = 2_100_000.0
    cfg.portfolio.top_n = 2
    cfg.portfolio.turnover_cap = 1.0

    out = tmp_path / "out"
    day1 = _signals("2026-02-06", {"A.SH": 0.9, "B.SH": 0.8, "C.SH": 0.7, "D.SH": 0.6})
    state = run_paper(cfg, out, signals=day1)
    assert state["date"] == "2026-02-06"
    assert state["filled"] == 2
    assert all(p["available"] == 0 for p in state["positions"].values())

    state2 = run_paper(cfg, out, signals=day1)
    assert state2["date"] == state["date"]
    hist = pd.read_csv(out / "paper" / "history.csv")
    orders = pd.read_csv(out / "paper" / "orders.csv")
    assert len(hist) == 1  # 同日重复运行不追加
    assert len(orders) == 2


def test_runner_daily_rebalance_with_history(tmp_path) -> None:
    root = tmp_path / "data"
    storage = _mini_storage(
        root, n_days=25, closes={"A.SH": 10.0, "B.SH": 10.0, "C.SH": 10.0, "D.SH": 10.0}
    )
    cfg = Config()
    cfg.data.root = root
    cfg.backtest.initial_cash = 2_100_000.0
    cfg.portfolio.top_n = 2
    cfg.portfolio.turnover_cap = 1.0

    out = tmp_path / "out"
    day1 = _signals("2026-02-06", {"A.SH": 0.9, "B.SH": 0.8, "C.SH": 0.7, "D.SH": 0.6})
    state1 = run_paper(cfg, out, signals=day1)
    assert set(state1["positions"]) == {"A.SH", "B.SH"}

    # 次日：A 跌出目标（应卖出），D 进入目标；T+1 已解锁
    dates = pd.bdate_range("2026-01-05", periods=26)
    rows = []
    for sym, close in {"A.SH": 9.0, "B.SH": 9.98, "C.SH": 9.5, "D.SH": 9.0}.items():
        rows.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "symbol": sym,
                    "open": close,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "volume": 1e6,
                    "amount": 1e9,
                    "turnover": 1.0,
                    "is_limit_up": False,
                    "is_limit_down": False,
                    "is_suspended": False,
                    "is_st": False,
                }
            )
        )
    storage.save("prices", pd.concat(rows, ignore_index=True))
    bench = pd.DataFrame(
        {"date": dates, "close": np.linspace(4000.0, 4120.0, len(dates))}
    )
    storage.save("benchmark", bench, partition_by_symbol=False)

    day2 = _signals("2026-02-09", {"D.SH": 0.95, "B.SH": 0.81, "A.SH": 0.1, "C.SH": 0.05})
    state2 = run_paper(cfg, out, signals=day2)
    assert state2["date"] == "2026-02-09"

    orders2 = pd.read_csv(out / "paper" / "orders.csv")
    sell_filled = orders2[
        (orders2["symbol"] == "A.SH") & (orders2["side"] == "sell")
        & (orders2["status"] == "filled")
    ]
    assert len(sell_filled) >= 1  # 关键回归：T+1 解锁后次日可卖出
    assert "A.SH" not in state2["positions"]

    hist = pd.read_csv(out / "paper" / "history.csv", parse_dates=["date"])
    assert len(hist) == 2
    assert list(hist["date"]) == sorted(hist["date"])
    assert (hist["total_fees"] > 0).all()
    assert (hist["benchmark_close"] > 0).all()
    state_json = json.loads((out / "paper" / "state.json").read_text(encoding="utf-8"))
    assert state_json["date"] == "2026-02-09"
