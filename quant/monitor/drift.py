"""模型漂移监控：IC / AUC 滚动统计 + 重训练触发器。"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

LOGGER = logging.getLogger("ashare.drift")


@dataclass
class ModelDriftMonitor:
    """滚动跟踪模型 OOS 表现，识别显著衰减。"""

    window: int = 63
    ic_history: list[float] = field(default_factory=list)
    auc_history: list[float] = field(default_factory=list)

    def update(self, ic: float | None, auc: float | None) -> None:
        if ic is not None and ic == ic:  # not NaN
            self.ic_history.append(float(ic))
            if len(self.ic_history) > self.window:
                self.ic_history.pop(0)
        if auc is not None and auc == auc:
            self.auc_history.append(float(auc))
            if len(self.auc_history) > self.window:
                self.auc_history.pop(0)

    def check(self) -> dict:
        if len(self.ic_history) < 10:
            return {"status": "collecting", "needs_retrain": False, "ic_decay": None}
        ic_arr = np.array(self.ic_history)
        ic_mean = ic_arr.mean()
        ic_std = ic_arr.std()
        latest = ic_arr[-1]
        decay = (latest - ic_mean) / ic_std if ic_std > 1e-12 else 0.0

        status = "ok"
        needs_retrain = False
        if latest < ic_mean - 2.0 * ic_std:
            status = "alert"
        if latest < ic_mean - 1.5 * ic_std:
            needs_retrain = True

        # 额外检测：近期 20 日累计 IC 为负且波动放大
        recent = ic_arr[-20:]
        recent_std = recent.std()
        if len(recent) >= 10 and recent.mean() < 0 and recent_std > ic_std * 1.5:
            status = "alert" if status == "ok" else status
            needs_retrain = True

        return {
            "status": status,
            "needs_retrain": needs_retrain,
            "ic_decay": round(float(decay), 3),
            "ic_mean": round(float(ic_mean), 5),
            "ic_std": round(float(ic_std), 5),
            "latest_ic": round(float(latest), 5),
            "samples": len(ic_arr),
        }