"""组合尾部风险：CVaR / Expected Shortfall 计算与约束检查。"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("ashare.cvar")


def compute_portfolio_returns(
    positions: dict[str, dict],
    price_history: pd.DataFrame,
    lookback: int = 252,
) -> pd.Series:
    """根据当前持仓和历史价格，计算组合（市值）加权日收益序列（收益率，同 threshold 单位）。"""
    if not positions:
        return pd.Series(dtype=float)
    close_wide = price_history.pivot(index="date", columns="symbol", values="close").sort_index()
    # 多取一天：pct_change 会减少一个样本
    recent = close_wide.tail(lookback + 1)
    if recent.empty or len(recent) < 2:
        return pd.Series(dtype=float)
    ret = recent.pct_change().dropna(how="all")
    if ret.empty:
        return pd.Series(dtype=float)

    latest_prices = recent.iloc[-1]
    values: dict[str, float] = {}
    total_value = 0.0
    for sym, pos in positions.items():
        shares = pos.get("shares", 0)
        if shares <= 0 or sym not in ret.columns:
            continue
        px = float(latest_prices.get(sym, 0) or 0.0)
        value = shares * px
        if value > 0:
            values[sym] = value
            total_value += value

    if total_value <= 0:
        return pd.Series(0.0, index=ret.index, dtype=float)

    port_ret = pd.Series(0.0, index=ret.index, dtype=float)
    for sym, value in values.items():
        port_ret = port_ret.add(ret[sym].fillna(0.0) * value, fill_value=0.0)
    port_ret = port_ret / total_value
    return port_ret.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def compute_cvar(
    port_ret: pd.Series,
    alpha: float = 0.05,
) -> float:
    """历史模拟法 CVaR（Expected Shortfall）。"""
    if port_ret.empty or len(port_ret) < 10:
        return 0.0
    var = np.percentile(port_ret, alpha * 100)
    cvar = float(port_ret[port_ret <= var].mean())
    return float(cvar)


def check_cvar_limit(
    positions: dict[str, dict],
    price_history: pd.DataFrame,
    cvar_threshold: float = -0.005,
    lookback: int = 252,
    alpha: float = 0.05,
) -> dict:
    """CVaR 风控告警。"""
    port_ret = compute_portfolio_returns(positions, price_history, lookback)
    cvar = compute_cvar(port_ret, alpha)
    triggered = cvar < cvar_threshold
    result = {
        "cvar": round(cvar, 6),
        "threshold": cvar_threshold,
        "triggered": bool(triggered),
        "lookback": lookback,
        "alpha": alpha,
    }
    if triggered:
        LOGGER.warning(
            "CVaR 告警: %.4f < %.4f（尾部风险超限）", cvar, cvar_threshold
        )
    return result