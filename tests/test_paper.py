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


def test_paper_last_price_date_bound() -> None:
    """_last_price 必须受 trade_date 约束，数据含未来行情时不得前视。"""
    prices = pd.concat(
        [_one_day_prices("2026-02-05", close=10.0), _one_day_prices("2026-02-06", close=20.0)],
        ignore_index=True,
    )
    broker = PaperBroker(prices)
    broker.trade_date = "2026-02-05"
    assert broker.last_price("A.SH") == 10.0
    broker.trade_date = "2026-02-06"
    assert broker.last_price("A.SH") == 20.0


def test_buy_shrinks_on_insufficient_cash() -> None:
    """现金不足时按可负担数量缩量成交，而非整单拒绝（与回测引擎一致）。"""
    prices = _one_day_prices()
    broker = PaperBroker(prices, initial_cash=30_000.0, cost=CostModel())
    broker.trade_date = "2026-02-05"
    order = Order(id="t_buy_big", symbol="A.SH", side="buy", shares=5000, price=10.0)
    broker.place_order(order)
    assert order.status == OrderStatus.FILLED
    pos = broker.get_positions()["A.SH"]
    assert 0 < pos.shares < 5000  # 部分成交
    assert broker.get_cash() >= 0.0


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


def test_paper_backtest_reconciliation() -> None:
    """同一目标组合下 PaperBroker 与 BacktestEngine 最终净值应接近（<2%）。"""
    from quant.backtest.cost import CostModel
    from quant.backtest.engine import BacktestEngine

    dates = pd.bdate_range("2023-01-02", periods=60)
    rows = []
    for sym, drift in [("A.SH", 0.10), ("B.SH", 0.20), ("C.SH", -0.05)]:
        close = 10.0 * (1 + drift * np.linspace(0, 1, len(dates)))
        rows.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "symbol": sym,
                    "open": close * 0.999,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "volume": 1e6,
                    "amount": 1e6 * close * 100,
                    "turnover": 1.0,
                    "is_limit_up": False,
                    "is_limit_down": False,
                    "is_limit_up_open": False,
                    "is_limit_down_open": False,
                    "is_suspended": False,
                    "is_st": False,
                }
            )
        )
    prices = pd.concat(rows, ignore_index=True)
    bench = pd.DataFrame(
        {"date": dates, "close": np.linspace(1000, 1100, len(dates))}
    )

    cfg = Config()
    bt_cfg, pf_cfg = cfg.backtest, cfg.portfolio
    bt_cfg.start, bt_cfg.end = str(dates[0].date()), str(dates[-1].date())
    pf_cfg.top_n = 3
    rebal = [dates[19], dates[39]]
    tw = pd.concat(
        [
            pd.DataFrame(
                {"date": rebal[0], "symbol": ["A.SH", "B.SH"], "weight": [0.5, 0.5]}
            ),
            pd.DataFrame(
                {
                    "date": rebal[1],
                    "symbol": ["A.SH", "B.SH", "C.SH"],
                    "weight": [1 / 3, 1 / 3, 1 / 3],
                }
            ),
        ],
        ignore_index=True,
    )

    engine = BacktestEngine(prices, bench, bt_cfg, pf_cfg)
    res = engine.run(tw, start=bt_cfg.start, end=bt_cfg.end)
    bt_nav = float(res.equity["portfolio_value"].iloc[-1])

    cost = CostModel(
        commission_bp=bt_cfg.commission_bp,
        stamp_bp=bt_cfg.stamp_bp,
        transfer_bp=bt_cfg.transfer_bp,
        slippage_bp=bt_cfg.slippage_bp,
        min_commission=bt_cfg.min_commission,
        lot_size=bt_cfg.lot_size,
    )
    broker = PaperBroker(prices, initial_cash=bt_cfg.initial_cash, cost=cost)

    def _rebalance(date, weights):
        broker.trade_date = str(date.date())
        broker.settle(broker.trade_date)
        pv = broker.portfolio_value()
        held = {s: p for s, p in broker.get_positions().items() if p.shares > 0}
        for sym, pos in held.items():
            if sym not in weights and pos.available > 0:
                broker.place_order(
                    Order(
                        id=f"{date.date()}_{sym}_out", symbol=sym, side="sell",
                        shares=pos.available, price=broker.last_price(sym),
                    )
                )
        for sym, w in weights.items():
            px = broker.last_price(sym)
            if px != px or px <= 0:
                continue
            cur_val = held[sym].shares * px if sym in held else 0.0
            target_val = pv * w
            diff = target_val - cur_val
            if diff > px * cost.lot_size:
                shares = int(
                    (diff - cost.buy_fee(diff)) // px // cost.lot_size * cost.lot_size
                )
                if shares > 0:
                    broker.place_order(
                        Order(
                            id=f"{date.date()}_{sym}_buy", symbol=sym, side="buy",
                            shares=shares, price=px,
                        )
                    )
            elif diff < -px * cost.lot_size:
                avail = held[sym].available if sym in held else 0
                shares = min(int(-diff // px // cost.lot_size * cost.lot_size), avail)
                if shares > 0:
                    broker.place_order(
                        Order(
                            id=f"{date.date()}_{sym}_sell", symbol=sym, side="sell",
                            shares=shares, price=px,
                        )
                    )

    _rebalance(rebal[0], {"A.SH": 0.5, "B.SH": 0.5})
    _rebalance(rebal[1], {"A.SH": 1 / 3, "B.SH": 1 / 3, "C.SH": 1 / 3})
    # 期末估值：把 paper 交易日推进到回测末日（真实每日任务会逐日推进）
    broker.trade_date = str(dates[-1].date())
    broker.settle(broker.trade_date)
    paper_nav = broker.portfolio_value()
    diff_pct = abs(paper_nav - bt_nav) / bt_cfg.initial_cash
    assert diff_pct < 0.02, f"paper/backtest NAV diff {diff_pct:.2%} >= 2%"


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
