"""订单管理系统：幂等、重试上限、全局熔断。"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field

from quant.execution.broker import Broker, Order


@dataclass
class OrderManager:
    broker: Broker
    max_retries: int = 3
    backoff_base: float = 1.0
    kill_threshold: int = 3
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("ashare.oms"))
    _submitted: set[str] = field(default_factory=set)
    _failures: int = 0
    _killed: bool = False

    def intent_id(self, symbol: str, side: str, shares: int, date: str) -> str:
        raw = f"{date}|{symbol}|{side}|{shares}".encode()
        return hashlib.sha1(raw).hexdigest()[:16]

    def submit(self, order: Order) -> str:
        if self._killed:
            self.logger.error("OMS 已熔断，拒绝下单 %s", order.symbol)
            return ""
        if order.id in self._submitted:
            self.logger.info("幂等跳过重复订单 %s", order.id)
            return order.id
        for attempt in range(1, self.max_retries + 1):
            try:
                order_id = self.broker.place_order(order)
                self._submitted.add(order.id)
                self._failures = 0
                return order_id
            except Exception as exc:  # noqa: BLE001
                self._failures += 1
                self.logger.warning("下单失败 %s (第 %d/%d 次): %s", order.symbol, attempt, self.max_retries, exc)
                if self._failures >= self.kill_threshold:
                    self._killed = True
                    self.logger.critical("连续失败 %d 次，OMS 熔断", self._failures)
                    break
                time.sleep(self.backoff_base * attempt)
        return ""

    def reset_kill(self) -> None:
        self._killed = False
        self._failures = 0
