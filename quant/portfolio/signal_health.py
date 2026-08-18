"""信号健康度：用「近期 top-N 组合相对基准的已实现超额」判断信号是否失效。

点内时间（无前视）：在任意时点 T，只用 T-window 的分数与 [T-window, T] 的已实现收益。
信号失效（近期超额为负）时下调目标仓位，与市场状态机（择时）解耦——这是针对
「市场涨但选股信号失效」这种场景的开关。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.config import PortfolioConfig


def compute_signal_health(
    scores: pd.DataFrame,
    prices: pd.DataFrame,
    benchmark: pd.DataFrame,
    cfg: PortfolioConfig,
) -> pd.DataFrame:
    """返回 date, signal_health（∈ [floor, 1.0]）。

    scores: long(date, symbol, score)；prices: long(date, symbol, close)；
    benchmark: (date, close)。window/floor/scale 取自 cfg。
    """
    top_n = cfg.top_n
    window = cfg.signal_health_window
    floor = cfg.signal_health_floor
    scale = cfg.signal_health_scale

    score_wide = scores.pivot_table(index="date", columns="symbol", values="score").sort_index()
    close_wide = (
        prices.pivot_table(index="date", columns="symbol", values="close")
        .ffill()
        .sort_index()
    )
    bench = benchmark.set_index("date")["close"].sort_index()

    score_dates = score_wide.index
    rows: list[dict] = []
    for i, T in enumerate(score_dates):
        if i < window:
            rows.append({"date": T, "signal_health": 1.0})
            continue
        T0 = score_dates[i - window]
        top = score_wide.loc[T0].sort_values(ascending=False).head(top_n).index
        s0 = close_wide.loc[T0, top]
        s1 = close_wide.loc[T, top]
        valid = s0.notna() & s1.notna() & (s0 > 0) & (s1 > 0)
        if int(valid.sum()) < 5:
            rows.append({"date": T, "signal_health": 1.0})
            continue
        basket_ret = float((s1[valid] / s0[valid] - 1.0).mean())
        b0 = float(bench.loc[:T0].iloc[-1])
        b1 = float(bench.loc[:T].iloc[-1])
        excess = basket_ret - (b1 / b0 - 1.0)
        health = float(np.clip(1.0 + excess / scale, floor, 1.0))
        rows.append({"date": T, "signal_health": health})
    return pd.DataFrame(rows)
