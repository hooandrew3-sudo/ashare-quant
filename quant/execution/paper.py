"""纸面交易通道：本地仿真成交，用于 dry-run 与每日模拟验证。

- 基于最近收盘价撮合，含佣金/印花税/过户费/滑点（与回测 CostModel 口径一致）；
- T+1 冻结：当日买入不可卖，settle(as_of) 按交易日解锁；
- 记录成交价与费用，便于对账与成本归因。
"""

from __future__ import annotations

import pandas as pd

from quant.backtest.cost import CostModel
from quant.execution.broker import Order, OrderStatus, Position


class PaperBroker:
    """基于最近收盘价的仿真撮合，含交易成本与 T+1 冻结。"""

    def __init__(
        self,
        prices: pd.DataFrame,
        initial_cash: float = 1_000_000.0,
        cost: CostModel | None = None,
    ):
        self._close = prices.pivot(index="date", columns="symbol", values="close")
        self._cash = initial_cash
        self._positions: dict[str, Position] = {}
        self._orders: list[Order] = []
        self._buy_dates: dict[str, str] = {}
        self._cost = cost or CostModel()
        self._total_fees = 0.0
        self.trade_date: str | None = None

    def connect(self) -> None:
        pass

    def get_cash(self) -> float:
        return self._cash

    def get_positions(self) -> dict[str, Position]:
        return dict(self._positions)

    def total_fees(self) -> float:
        return self._total_fees

    def _last_price(self, symbol: str) -> float:
        if symbol not in self._close.columns:
            return float("nan")
        series = self._close[symbol].dropna()
        return float(series.iloc[-1]) if len(series) else float("nan")

    def last_price(self, symbol: str) -> float:
        return self._last_price(symbol)

    def portfolio_value(self) -> float:
        return self._cash + sum(
            p.shares * self._last_price(s) for s, p in self._positions.items() if p.shares > 0
        )

    def settle(self, as_of: str) -> None:
        """T+1 解锁：as_of 晚于买入日的持仓全部变为可卖。"""
        asof = pd.Timestamp(as_of)
        for sym, pos in self._positions.items():
            buy_date = self._buy_dates.get(sym)
            if buy_date is None or asof > pd.Timestamp(buy_date):
                pos.available = pos.shares

    def place_order(self, order: Order) -> str:
        px = self._last_price(order.symbol)
        if px != px:
            order.status = OrderStatus.REJECTED
            order.reason = "no_price"
            self._orders.append(order)
            return order.id
        slip_bp = self._cost.effective_slippage_bp(px * order.shares, 0.0)
        if order.side == "buy":
            fill_px = px * (1 + slip_bp / 10_000)
            amount = fill_px * order.shares
            fee = self._cost.buy_fee(amount)
            if amount + fee > self._cash:
                order.status = OrderStatus.REJECTED
                order.reason = "insufficient_cash"
                self._orders.append(order)
                return order.id
            self._cash -= amount + fee
            pos = self._positions.get(order.symbol)
            if pos is None:
                self._positions[order.symbol] = Position(
                    symbol=order.symbol, shares=0, available=0, cost=0.0
                )
                pos = self._positions[order.symbol]
            pos.shares += order.shares
            pos.available = 0  # T+1：当日买入不可卖
            pos.cost = (pos.cost * (pos.shares - order.shares) + amount) / max(pos.shares, 1)
            if self.trade_date:
                self._buy_dates[order.symbol] = self.trade_date
            self._total_fees += fee
            order.price = fill_px
            order.fee = fee
        else:
            pos = self._positions.get(order.symbol)
            if pos is None or pos.available < order.shares:
                order.status = OrderStatus.REJECTED
                order.reason = "insufficient_available"
                self._orders.append(order)
                return order.id
            fill_px = px * (1 - slip_bp / 10_000)
            amount = fill_px * order.shares
            fee = self._cost.sell_fee(amount)
            self._cash += amount - fee
            pos.shares -= order.shares
            pos.available -= order.shares
            if pos.shares <= 0:
                self._positions.pop(order.symbol, None)
                self._buy_dates.pop(order.symbol, None)
            self._total_fees += fee
            order.price = fill_px
            order.fee = fee
        order.status = OrderStatus.FILLED
        self._orders.append(order)
        return order.id

    def cancel_order(self, order_id: str) -> None:
        for o in self._orders:
            if o.id == order_id and o.status == OrderStatus.PENDING:
                o.status = OrderStatus.CANCELLED

    def get_orders(self) -> list[Order]:
        return list(self._orders)
