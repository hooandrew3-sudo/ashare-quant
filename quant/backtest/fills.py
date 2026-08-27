"""撮合模拟：开盘成交、涨跌停、停牌、顺延。"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quant.backtest.cost import CostModel


@dataclass
class FillResult:
    status: str  # filled | skipped | postponed | dropped
    price: float = 0.0
    shares: int = 0
    fee: float = 0.0
    reason: str = ""


class FillSimulator:
    """基于日线开盘价的撮合器。"""

    def __init__(self, prices: pd.DataFrame, cost: CostModel):
        self.cost = cost
        self._bars = prices.set_index(["date", "symbol"])
        close = prices.pivot(index="date", columns="symbol", values="close").ffill().sort_index()
        self._close = close
        # 预计算 numpy 矩阵 + 列索引，last_close 由 O(N) 布尔扫描降为 O(log D)
        self._close_index = close.index
        self._close_values = close.to_numpy(dtype=float)
        self._col_of: dict[str, int] = {s: i for i, s in enumerate(close.columns)}

    def try_fill(
        self,
        symbol: str,
        date: pd.Timestamp,
        side: str,
        shares: int,
        postpone_days_left: int,
    ) -> FillResult:
        if shares <= 0:
            return FillResult(status="dropped", reason="shares<=0")
        try:
            bar = self._bars.loc[(date, symbol)]
        except KeyError:
            return FillResult(status="postponed", reason="no_bar")
        if isinstance(bar, pd.DataFrame):
            bar = bar.iloc[-1]
        if bool(bar.get("is_suspended", False)) or pd.isna(bar.get("open")):
            return FillResult(status="postponed", reason="suspended")
        open_p = float(bar["open"])
        order_amount = open_p * shares
        daily_amount = float(bar.get("amount", 0.0) or 0.0)
        slip_bp = self.cost.effective_slippage_bp(order_amount, daily_amount)
        # 开盘一字板判定优先用 open-limit 标记（按 open vs preclose 计算）；
        # 老数据无该列时回退到收盘封板标记（保守近似）。
        limit_up_open = _flag(bar, "is_limit_up_open")
        limit_down_open = _flag(bar, "is_limit_down_open")
        if limit_up_open is None:
            limit_up_open = _flag(bar, "is_limit_up") or False
        if limit_down_open is None:
            limit_down_open = _flag(bar, "is_limit_down") or False
        if side == "buy":
            if limit_up_open:
                return FillResult(status="skipped", reason="limit_up_open")
            px = open_p * (1 + slip_bp / 10_000)
            shares = min(shares, self.cost.round_lot(shares))
            if shares <= 0:
                return FillResult(status="dropped", reason="lot")
            fee = self.cost.buy_fee(px * shares)
            return FillResult(status="filled", price=px, shares=shares, fee=fee)
        else:  # sell
            if limit_down_open:
                if postpone_days_left <= 0:
                    return FillResult(status="dropped", reason="limit_down_dropped")
                return FillResult(status="postponed", reason="limit_down_open")
            px = open_p * (1 - slip_bp / 10_000)
            fee = self.cost.sell_fee(px * shares)
            return FillResult(status="filled", price=px, shares=shares, fee=fee)

    def last_close(self, symbol: str, date: pd.Timestamp) -> float:
        """date 当日（或此前最近）的复权收盘价；预计算矩阵上二分查找。"""
        col = self._col_of.get(symbol)
        if col is None:
            return float("nan")
        pos = self._close_index.searchsorted(date, side="right")
        if pos <= 0:
            return float("nan")
        v = self._close_values[pos - 1, col]
        return float(v) if v == v else float("nan")


def _flag(bar, key: str) -> bool | None:
    """读取布尔标记；缺失或 NaN 返回 None（表示不可用）。"""
    if key not in bar or bar[key] is None:
        return None
    v = bar[key]
    if v is pd.NA:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return bool(v)
