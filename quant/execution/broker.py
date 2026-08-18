"""交易通道抽象：统一下单/查询接口。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable


class OrderStatus(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    id: str
    symbol: str
    side: str  # buy | sell
    shares: int
    price: float
    status: OrderStatus = OrderStatus.PENDING
    reason: str = ""
    fee: float = 0.0


@dataclass
class Position:
    symbol: str
    shares: int
    available: int
    cost: float


@runtime_checkable
class Broker(Protocol):
    def connect(self) -> None: ...

    def get_cash(self) -> float: ...

    def get_positions(self) -> dict[str, Position]: ...

    def place_order(self, order: Order) -> str: ...

    def cancel_order(self, order_id: str) -> None: ...

    def get_orders(self) -> list[Order]: ...
