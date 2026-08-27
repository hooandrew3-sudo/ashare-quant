"""订单管理系统：幂等、重试上限、全局熔断。

生产语义（区别于玩具实现）：
- 幂等键 = date|symbol|side（不含数量），并持久化到账本文件，
  进程重启后依然去重；此前 intent_id 掺入 shares 且仅存内存，
  "幂等"只在单进程内且数量不变时成立；
- 下单异常一律视为"结果未知"：网络超时时订单可能已到达券商，
  盲目重发 = 双倍仓位。默认不自动重试，需人工/对账确认后再补单
  （retry_on_error=True 仅适用于确认幂等的纸面通道）;
- 熔断按时间窗口计数：传输类连续失败 ≥ kill_threshold 次才熔断；
  券商明确拒单（place_order 正常返回失败码）不进入熔断计数。
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from quant.execution.broker import Broker, Order


class OrderUnknownError(RuntimeError):
    """下单结果未知（超时/断线）：禁止盲目重试，必须先查询券商侧状态。"""


@dataclass
class OrderManager:
    broker: Broker
    max_retries: int = 1
    backoff_base: float = 1.0
    kill_threshold: int = 5
    breaker_window_sec: float = 600.0
    ledger_path: Path | None = None  # 幂等账本（JSON），重启后仍可去重
    retry_on_error: bool = False     # 仅纸面等幂等通道可开启自动重试
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("ashare.oms"))
    _submitted: set[str] = field(default_factory=set)
    _failures: list[float] = field(default_factory=list)
    _killed: bool = False

    def __post_init__(self) -> None:
        self._load_ledger()

    def intent_id(self, symbol: str, side: str, shares: int | None = None, date: str = "") -> str:
        """幂等键：date|symbol|side。shares 不参与哈希——数量变化不代表新意图。"""
        raw = f"{date}|{symbol}|{side}".encode()
        return hashlib.sha1(raw).hexdigest()[:16]

    # ---- 幂等账本 ----
    def _load_ledger(self) -> None:
        if self.ledger_path is None:
            return
        try:
            if Path(self.ledger_path).exists():
                data = json.loads(Path(self.ledger_path).read_text(encoding="utf-8"))
                self._submitted.update(data.get("orders", []))
        except (json.JSONDecodeError, OSError) as exc:
            self.logger.warning("幂等账本加载失败（忽略）: %s", exc)

    def _record_ledger(self, order_id: str, broker_order_id: str) -> None:
        self._submitted.add(order_id)
        if self.ledger_path is None:
            return
        try:
            path = Path(self.ledger_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            data = {"orders": sorted(self._submitted)}
            if path.exists():
                try:
                    old = json.loads(path.read_text(encoding="utf-8"))
                    merged = set(old.get("orders", [])) | {order_id}
                    data["orders"] = sorted(merged)
                except json.JSONDecodeError:
                    pass
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(data), encoding="utf-8")
            tmp.replace(path)
        except OSError as exc:
            self.logger.warning("幂等账本写入失败: %s", exc)

    # ---- 提交 ----
    def submit(self, order: Order) -> str:
        if self._killed:
            self.logger.error("OMS 已熔断，拒绝下单 %s", order.symbol)
            return ""
        if order.id in self._submitted:
            self.logger.info("幂等跳过重复订单 %s", order.id)
            return order.id
        attempt = 0
        while attempt < max(1, self.max_retries):
            attempt += 1
            try:
                broker_order_id = self.broker.place_order(order)
                self._failures.clear()
                self._record_ledger(order.id, broker_order_id)
                return broker_order_id
            except Exception as exc:  # noqa: BLE001
                now = time.time()
                self._failures.append(now)
                self._failures = [t for t in self._failures if now - t <= self.breaker_window_sec]
                self.logger.error(
                    "下单失败 %s (第 %d/%d 次): %s",
                    order.symbol, attempt, self.max_retries, exc,
                )
                if len(self._failures) >= self.kill_threshold:
                    self._killed = True
                    self.logger.critical(
                        "窗口 %.0fs 内连续失败 %d 次，OMS 熔断（人工排查后调用 reset_kill 恢复）",
                        self.breaker_window_sec, len(self._failures),
                    )
                    break
                # 结果未知 ≠ 可重试：默认不重发，避免双倍仓位
                if not self.retry_on_error:
                    break
                time.sleep(self.backoff_base * attempt)
        return ""

    @property
    def killed(self) -> bool:
        return self._killed

    @property
    def failures_in_window(self) -> int:
        return len(self._failures)

    def reset_kill(self) -> None:
        self._killed = False
        self._failures.clear()
