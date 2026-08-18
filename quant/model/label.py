"""标签构造：未来 20 日相对基准超额收益 → 二分类/分位标签。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.config import Config


def build_label(
    prices: pd.DataFrame,
    benchmark: pd.DataFrame,
    cfg: Config,
    horizon: int | None = None,
) -> pd.DataFrame:
    """返回长表：date, symbol, excess(未来超额), value(标签)。

    label_mode:
      binary           — 0/1：是否跑赢基准
      quantile_contrast — 0/1：top 30% 为 1、bottom 30% 为 0，中间剔除
      regression       — 连续值：未来超额收益
    """
    horizon = horizon or cfg.model.horizon
    p = prices.sort_values(["symbol", "date"]).copy()
    p["fwd_ret"] = p.groupby("symbol")["close"].transform(lambda s: s.shift(-horizon) / s - 1.0)

    b = benchmark.sort_values("date").copy()
    b["fwd_ret"] = b["close"].shift(-horizon) / b["close"] - 1.0

    out = p[["date", "symbol", "fwd_ret"]].merge(
        b[["date", "fwd_ret"]], on="date", suffixes=("", "_bench")
    )
    out["excess"] = out["fwd_ret"] - out["fwd_ret_bench"]
    out = out.dropna(subset=["excess"])

    if cfg.model.label_mode == "top_quantile":
        thr = out.groupby("date")["excess"].transform(
            lambda x: x.quantile(1.0 - cfg.model.top_quantile)
        )
        out["value"] = (out["excess"] >= thr).astype(int)
    elif cfg.model.label_mode == "quantile_contrast":
        hi = out.groupby("date")["excess"].transform(lambda x: x.quantile(0.7))
        lo = out.groupby("date")["excess"].transform(lambda x: x.quantile(0.3))
        out["value"] = np.where(out["excess"] >= hi, 1, np.where(out["excess"] <= lo, 0, np.nan))
    elif cfg.model.label_mode == "regression":
        out["value"] = out["excess"]
    else:  # binary
        out["value"] = (out["excess"] > 0).astype(int)
    return out[["date", "symbol", "excess", "value"]].dropna(subset=["value"]).reset_index(drop=True)
