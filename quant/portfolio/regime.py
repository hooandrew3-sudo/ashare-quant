"""市场状态机：MA 系统 + 波动率目标 → 总仓位系数。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.config import PortfolioConfig


def compute_regime(benchmark_close: pd.Series, cfg: PortfolioConfig) -> pd.DataFrame:
    """返回 date, fast, slow, long, base_level, vol_scale, target_exposure。

    base_level：MA20 vs MA120 与 收盘 vs MA250 决定 1.0/0.5/0.3；
    vol_scale：组合目标波动 / 基准已实现波动（下限 0.2，上限 1.0）。
    """
    r = cfg.regime
    df = pd.DataFrame({"close": benchmark_close})
    if "date" not in df.columns:
        df = df.reset_index()
        if "date" not in df.columns:
            df = df.rename(columns={df.columns[0]: "date"})
    df["fast"] = df["close"].rolling(r.fast, min_periods=r.fast // 2).mean()
    df["slow"] = df["close"].rolling(r.slow, min_periods=r.slow // 2).mean()
    df["long"] = df["close"].rolling(r.long, min_periods=r.long // 2).mean()

    levels = r.levels
    df["base_level"] = np.where(
        df["close"] > df["long"],
        np.where(df["fast"] > df["slow"], levels[0], levels[1]),
        levels[2],
    )
    ret = df["close"].pct_change()
    realized = ret.rolling(cfg.vol_window, min_periods=10).std() * np.sqrt(252)
    df["vol_scale"] = (cfg.target_vol / realized).clip(0.2, 1.0)
    df["target_exposure"] = (df["base_level"] * df["vol_scale"]).clip(0.0, 1.0)
    return df[["date", "close", "fast", "slow", "long", "base_level", "vol_scale", "target_exposure"]]


def compute_smallcap_regime(smallcap_close: pd.Series, cfg: PortfolioConfig) -> pd.DataFrame:
    """小盘辅助择时：中证1000 收盘 < MA250 时降仓至 floor，否则 1.0。"""
    df = pd.DataFrame({"close": smallcap_close}).reset_index()
    df = df.rename(columns={df.columns[0]: "date"})
    long = df["close"].rolling(cfg.smallcap_long, min_periods=cfg.smallcap_long // 2).mean()
    df["smallcap_level"] = np.where(df["close"] > long, 1.0, cfg.smallcap_floor)
    return df[["date", "smallcap_level"]]
