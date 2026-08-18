"""影子交易（Shadow Trading）：记录模拟信号，晚间对账实盘成交差异。"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

LOGGER = logging.getLogger("ashare.shadow")


@dataclass
class ShadowBroker:
    """不发送真实订单，仅记录 shadow 执行结果。"""

    cost_bp: float = 7.5
    lot_size: int = 100
    log_path: str | None = None
    _records: list[dict] = field(default_factory=list)

    def simulate(self, order: dict[str, Any], date: str | None = None) -> dict:
        date = date or datetime.now().strftime("%Y-%m-%d")
        px = float(order.get("price", 0.0) or 0.0)
        shares = int(order.get("shares", 0) or 0)
        amount = abs(px * shares)
        fee = amount * self.cost_bp / 10_000
        rec = {
            "date": date,
            "symbol": order.get("symbol"),
            "side": order.get("side"),
            "shares": shares,
            "price": round(px, 4),
            "amount": round(amount, 2),
            "fee": round(fee, 2),
            "status": "shadow_filled",
        }
        self._records.append(rec)
        return rec

    def flush(self) -> pd.DataFrame:
        if not self._records:
            return pd.DataFrame(columns=["date", "symbol", "side", "shares", "price", "amount", "fee", "status"])
        df = pd.DataFrame(self._records)
        self._records.clear()
        if self.log_path:
            try:
                df.to_csv(self.log_path, mode="a", header=False, index=False, encoding="utf-8")
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("shadow 日志写入失败: %s", exc)
        return df

    def summary(self) -> dict:
        if not self._records:
            return {"n": 0}
        df = pd.DataFrame(self._records)
        return {
            "n": int(len(df)),
            "total_amount": float(df["amount"].sum()),
            "total_fee": float(df["fee"].sum()),
            "buy_pct": float((df["side"] == "buy").mean()) if "side" in df.columns else None,
        }