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
        avg_amount: "pd.Series | None" = None,
    ):
        self._close = prices.pivot(index="date", columns="symbol", values="close")
        self._cash = initial_cash
        self._positions: dict[str, Position] = {}
        self._orders: list[Order] = []
        self._buy_dates: dict[str, str] = {}
        self._cost = cost or CostModel()
        # 近 20 日均成交额（symbol 索引）：启用 adaptive 滑点时与回测同口径。
        # 此前恒传 0，adaptive 配置被静默降级为固定 bp，纸面与回测口径分裂。
        self._avg_amount = avg_amount if avg_amount is not None else pd.Series(dtype=float)
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
        if self.trade_date is not None:
            # 日期上界：数据碰脏（含未来行情）时不得前视
            series = series[series.index <= pd.Timestamp(self.trade_date)]
        return float(series.iloc[-1]) if len(series) else float("nan")

    def last_price(self, symbol: str) -> float:
        return self._last_price(symbol)

    def portfolio_value(self) -> float:
        total = self._cash
        for s, p in self._positions.items():
            if p.shares <= 0:
                continue
            px = self._last_price(s)
            if px != px or px <= 0:
                # 无价持仓不得毒化整体市值（NaN 会连锁污染权重与调仓金额）
                continue
            total += p.shares * px
        return total

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
        slip_bp = self._cost.effective_slippage_bp(
            px * order.shares, float(self._avg_amount.get(order.symbol, 0.0) or 0.0)
        )
        if order.side == "buy":
            fill_px = px * (1 + slip_bp / 10_000)
            amount = fill_px * order.shares
            fee = self._cost.buy_fee(amount)
            if amount + fee > self._cash:
                # 现金不足：按可负担数量缩量（与回测引擎 _execute 语义一致），
                # 避免"整单拒绝"导致调仓目标落空、组合偏离
                shrink = order.shares
                while shrink > 0:
                    f = self._cost.buy_fee(fill_px * shrink)
                    if fill_px * shrink + f <= self._cash + 1e-6:
                        break
                    shrink -= self._cost.lot_size
                if shrink <= 0:
                    order.status = OrderStatus.REJECTED
                    order.reason = "insufficient_cash"
                    self._orders.append(order)
                    return order.id
                order.shares = shrink
                amount = fill_px * shrink
                fee = self._cost.buy_fee(amount)
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
