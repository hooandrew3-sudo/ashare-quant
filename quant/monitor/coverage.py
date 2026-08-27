"""因子覆盖度哨兵：检测数据源失效导致的因子静默退化。"""

from __future__ import annotations

import json
from pathlib import Path


def check_factor_coverage(signals_dir: str | Path, cfg) -> list[str]:
    """扫描 signals 目录下的 factor_coverage_*.json 历史，返回告警列表。

    对 coverage_watch 中的每个因子，从最新往回数连续覆盖度 < min_ratio 的
    天数；达到 coverage_min_days 时告警（因子可能因数据源失效而恒为 0）。
    """
    signals_dir = Path(signals_dir)
    files = sorted(signals_dir.glob("factor_coverage_*.json"))
    if len(files) < cfg.factors.coverage_min_days:
        return []
    history: list[dict[str, float]] = []
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        factors = data.get("factors") or {}
        history.append(
            {
                name: float(info.get("coverage_ratio_mean_20d") or 0.0)
                for name, info in factors.items()
            }
        )
    alerts: list[str] = []
    for factor in cfg.factors.coverage_watch:
        consecutive = 0
        for day in reversed(history):
            value = day.get(factor)
            if value is not None and value < cfg.factors.coverage_min_ratio:
                consecutive += 1
            else:
                break
        if consecutive >= cfg.factors.coverage_min_days:
            alerts.append(
                f"因子 {factor} 连续 {consecutive} 日覆盖度 < "
                f"{cfg.factors.coverage_min_ratio:.0%}，疑似数据源失效，"
                "该因子可能在静默退化为 0"
            )
    return alerts
