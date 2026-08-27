"""风控辅助：止损判定、组合偏离检查。"""

from __future__ import annotations

import pandas as pd


def stop_hit(entry_price: float, current_close: float, threshold: float) -> bool:
    """当前亏损超过阈值（-threshold）触发止损。"""
    if not entry_price or entry_price <= 0 or current_close != current_close:
        return False
    return (current_close / entry_price - 1.0) <= -abs(threshold)


def check_limits(
    positions: dict[str, float],
    industry_of: dict[str, str],
    max_weight: float,
    max_industry: float,
) -> list[str]:
    """检查组合约束，返回违规项描述。"""
    issues: list[str] = []
    total = sum(positions.values())
    if total <= 0:
        return issues
    for sym, w in positions.items():
        if w / total > max_weight + 1e-9:
            issues.append(f"{sym} 权重 {w/total:.2%} 超限")
    ind_w: dict[str, float] = {}
    for sym, w in positions.items():
        ind = industry_of.get(sym, "UNKNOWN")
        ind_w[ind] = ind_w.get(ind, 0) + w
    for ind, w in ind_w.items():
        if w / total > max_industry + 1e-9:
            issues.append(f"行业 {ind} 权重 {w/total:.2%} 超限")
    return issues
