"""纸面交易运行器：每日信号 → 目标组合 → PaperBroker 执行 → 状态存档。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quant.backtest.cost import CostModel
from quant.config import Config
from quant.data.storage import Storage
from quant.execution.broker import Order, OrderStatus, Position
from quant.execution.oms import OrderManager
from quant.execution.paper import PaperBroker
from quant.pipeline import live_signals
from quant.portfolio.selection import build_target_weights
from quant.utils import ensure_dir, setup_logging


def _benchmark_close(bundle, latest: pd.Timestamp) -> float | None:
    """取 latest 当日（或此前最近）的基准收盘价，用于与纸面组合对比。"""
    if bundle.benchmark.empty:
        return None
    bench = bundle.benchmark.copy()
    bench["date"] = pd.to_datetime(bench["date"])
    hist = bench[bench["date"] <= latest].sort_values("date")
    if hist.empty:
        return None
    return float(hist["close"].iloc[-1])


def _append_history(state_dir: Path, state: dict, bench_close: float | None) -> None:
    """每日一行净值历史（按 date 幂等追加），供收益/回撤/基准对比。"""
    init = float(state.get("initial_cash", 1_000_000.0))
    pnl = float(state["portfolio_value"]) - init
    row = {
        "date": state["date"],
        "portfolio_value": float(state["portfolio_value"]),
        "cash": float(state["cash"]),
        "n_positions": int(len(state["positions"])),
        "n_orders": int(state.get("orders", 0)),
        "total_fees": float(state.get("total_fees", 0.0)),
        "benchmark_close": bench_close,
        "pnl": pnl,
        "return_pct": pnl / init if init else 0.0,
    }
    hist_path = state_dir / "history.csv"
    df = pd.DataFrame([row])
    if hist_path.exists():
        old = pd.read_csv(hist_path)
        old = old[old["date"] != row["date"]]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(hist_path, index=False)


def _append_orders(state_dir: Path, orders: list[Order], date: pd.Timestamp) -> None:
    """按 date 幂等追加订单流水（含成交价与费用），避免覆盖历史。"""
    rows = [
        {
            "date": date,
            "symbol": o.symbol,
            "side": o.side,
            "shares": o.shares,
            "price": o.price,
            "fee": o.fee,
            "status": o.status.value,
            "reason": o.reason,
        }
        for o in orders
    ]
    orders_path = state_dir / "orders.csv"
    df = pd.DataFrame(rows)
    if orders_path.exists():
        old = pd.read_csv(orders_path)
        old = old[old["date"] != str(date)]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(orders_path, index=False)


def run_paper(
    cfg: Config,
    output_root: str | Path = "artifacts",
    signals: pd.DataFrame | None = None,
) -> dict:
    """按最新信号执行纸面调仓并持久化账户状态（幂等，可每日重复）。"""
    log = setup_logging(cfg.run.verbose)
    storage = Storage(cfg.data.root)
    bundle = storage.load_bundle()
    signals = signals if signals is not None else live_signals(cfg, output_root)
    if signals.empty:
        raise RuntimeError("无可用信号")

    latest = pd.Timestamp(bundle.prices["date"].max())
    latest_str = str(latest.date())
    state_dir = ensure_dir(Path(output_root) / "paper")
    state_file = state_dir / "state.json"
    cost = CostModel(
        commission_bp=cfg.backtest.commission_bp,
        stamp_bp=cfg.backtest.stamp_bp,
        transfer_bp=cfg.backtest.transfer_bp,
        slippage_bp=cfg.backtest.slippage_bp,
        min_commission=cfg.backtest.min_commission,
        lot_size=cfg.backtest.lot_size,
    )
    broker = PaperBroker(bundle.prices, initial_cash=cfg.backtest.initial_cash, cost=cost)

    if state_file.exists():
        state = json.loads(state_file.read_text(encoding="utf-8"))
        if state.get("date") == latest_str:
            log.info("纸面账户已按 %s 执行，跳过同日重复运行", latest.date())
            return state
        broker._cash = float(state.get("cash", cfg.backtest.initial_cash))
        broker._buy_dates = {str(k): str(v) for k, v in state.get("buy_dates", {}).items()}
        broker._total_fees = float(state.get("total_fees", 0.0))
        broker._positions = {}
        for sym, p in state.get("positions", {}).items():
            broker._positions[sym] = Position(
                symbol=sym,
                shares=int(p["shares"]),
                available=int(p.get("available", p["shares"])),
                cost=float(p.get("cost", 0.0)),
            )

    broker.trade_date = latest_str
    broker.settle(latest_str)  # T+1：解锁此前买入的持仓
    pv = broker.portfolio_value()
    held = {s: p for s, p in broker.get_positions().items() if p.shares > 0}

    # 当前持仓按市值折算为 prev_weights，喂给与回测一致的组合构建逻辑
    prev_rows = []
    for sym, pos in held.items():
        px = broker.last_price(sym)
        if px == px and px > 0:
            prev_rows.append({"date": latest, "symbol": sym, "weight": pos.shares * px / pv})
    prev_weights = pd.DataFrame(prev_rows)

    # 统一组合构建：复用回测的 build_target_weights（stickiness/turnover/行业/流动性约束）
    scores = signals.copy()
    if "date" not in scores.columns:
        scores["date"] = latest
    target = build_target_weights(
        scores=scores,
        prices=bundle.prices,
        industry=bundle.industry,
        cfg=cfg.portfolio,
        rebalance_dates=[latest],
        prev_weights=prev_weights if not prev_weights.empty else None,
    )
    if target.empty:
        raise RuntimeError("组合构建无目标持仓")
    target_map = dict(zip(target["symbol"], target["weight"]))
    target_syms = set(target_map)
    orders: list[Order] = []
    stamp = latest.strftime("%Y%m%d")

    for sym, pos in held.items():
        if sym not in target_syms and pos.available > 0:
            orders.append(
                Order(
                    id=f"{stamp}_sell_{sym}",
                    symbol=sym,
                    side="sell",
                    shares=pos.available,
                    price=broker.last_price(sym),
                )
            )
    for sym, w in target_map.items():
        px = broker.last_price(sym)
        if px != px or px <= 0:
            continue
        cur_val = held[sym].shares * px if sym in held else 0.0
        target_val = pv * float(w)
        diff = target_val - cur_val
        lot = cfg.backtest.lot_size
        if diff > px * lot:
            slip_bp = cost.effective_slippage_bp(diff, 0.0)
            fill_px = px * (1 + slip_bp / 10_000)
            fee_est = cost.buy_fee(diff)
            shares = int((diff - fee_est) // fill_px // lot * lot)  # 按含滑点成交价预留费用
            if shares > 0:
                orders.append(Order(id=f"{stamp}_buy_{sym}", symbol=sym, side="buy", shares=shares, price=px))
        elif diff < -px * lot:
            shares = int(-diff // px // lot * lot)
            shares = min(shares, held[sym].available) if sym in held else 0
            if shares > 0:
                orders.append(Order(id=f"{stamp}_sell_{sym}", symbol=sym, side="sell", shares=shares, price=px))

    oms = OrderManager(broker)
    for o in orders:
        oms.submit(o)
    filled = sum(1 for o in broker.get_orders() if o.status == OrderStatus.FILLED)

    state = {
        "date": latest_str,
        "initial_cash": cfg.backtest.initial_cash,
        "cash": broker.get_cash(),
        "portfolio_value": broker.portfolio_value(),
        "total_fees": broker.total_fees(),
        "positions": {
            s: {"shares": p.shares, "available": p.available, "cost": p.cost}
            for s, p in broker.get_positions().items()
            if p.shares > 0
        },
        "buy_dates": {
            s: bd for s, bd in broker._buy_dates.items() if s in broker._positions
        },
        "orders": len(orders),
        "filled": filled,
        "target_symbols": sorted(target_syms),
    }
    state_file.write_text(
        json.dumps(state, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    _append_history(state_dir, state, _benchmark_close(bundle, latest))
    _append_orders(state_dir, broker.get_orders(), latest)
    log.info(
        "纸面调仓完成: %d 笔订单(成交 %d), 组合市值 %.2f, 持仓 %d 只, 累计费用 %.2f",
        len(orders), filled, state["portfolio_value"], len(state["positions"]), state["total_fees"],
    )
    return state
