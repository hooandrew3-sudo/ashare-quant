"""Brinson 归因：配置效应 + 个股选择效应 + 交互效应。"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("ashare.brinson")


def brinson_attribution(
    portfolio_weights: pd.DataFrame,
    benchmark_weights: pd.DataFrame,
    returns: pd.DataFrame,
    date: str,
    by_industry: bool = False,
) -> pd.DataFrame:
    """按日期计算 Brinson-Fachler 归因（简化版）。

    Parameters
    ----------
    portfolio_weights: DataFrame with columns ['symbol', 'weight'] for target date.
    benchmark_weights: DataFrame with columns ['symbol', 'weight'] for target date.
    returns: DataFrame with columns ['symbol', 'return'] for target date.
    date: 日期字符串。
    by_industry: 若为 True 且存在行业信息，则按行业分组输出。

    Returns
    -------
    DataFrame with columns ['date', 'allocation', 'selection', 'interaction'].
    """
    pw = portfolio_weights.copy()
    bw = benchmark_weights.copy()
    rt = returns.copy()
    pw["date"] = pd.to_datetime(pw["date"])
    bw["date"] = pd.to_datetime(bw["date"])
    rt["date"] = pd.to_datetime(rt["date"])
    d = pd.Timestamp(date)

    pw = pw[pw["date"] == d].set_index("symbol")["weight"]
    bw = bw[bw["date"] == d].set_index("symbol")["weight"]
    rt = rt[rt["date"] == d].set_index("symbol")["return"]

    common = pw.index.union(bw.index).intersection(rt.index)
    pw = pw.reindex(common).fillna(0.0)
    bw = bw.reindex(common).fillna(0.0)
    rt = rt.reindex(common).fillna(0.0)

    total_pw = pw.sum()
    total_bw = bw.sum()
    if total_bw <= 0 or total_pw <= 0:
        return pd.DataFrame(
            [[d, 0.0, 0.0, 0.0]],
            columns=["date", "allocation", "selection", "interaction"],
        )

    # 总收益
    rp = (pw / total_pw * rt).sum()
    rb = (bw / total_bw * rt).sum()
    excess = rp - rb

    # 简化 Brinson（未含行业交叉项）
    allocation = ((pw - bw) / total_pw * rb).sum()
    selection = (bw / total_bw * (rt - rb)).sum()
    interaction = ((pw - bw) / total_pw * (rt - rb)).sum()
    # 数值微调求和为 excess
    diff = excess - (allocation + selection + interaction)
    interaction = interaction + diff

    row = {
        "date": d,
        "allocation": float(allocation),
        "selection": float(selection),
        "interaction": float(interaction),
        "excess": float(excess),
    }
    out = pd.DataFrame([row])

    if by_industry:
        try:
            ind_map = _industry_map()
            out = pd.concat([out, _brinson_by_industry(pw, bw, rt, d, ind_map)], ignore_index=True)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("行业归因失败: %s", exc)
    return out


def _industry_map() -> Optional[pd.Series]:
    """尝试从本地存储读取行业映射（若有）。"""
    try:
        from quant.data.storage import Storage
        from quant.config import Config
        storage = Storage(Config().data.root)
        ind = storage.load("industry")
        if "symbol" in ind.columns and "industry" in ind.columns:
            return ind.sort_values("as_of_date").drop_duplicates("symbol", keep="last").set_index("symbol")["industry"]
    except Exception:  # noqa: BLE001
        pass
    return None


def _brinson_by_industry(
    pw: pd.Series, bw: pd.Series, rt: pd.Series, d: pd.Timestamp, ind_map: pd.Series | None,
) -> pd.DataFrame:
    if ind_map is None or ind_map.empty:
        return pd.DataFrame()
    inds = ind_map.reindex(pw.index.union(bw.index).union(rt.index)).fillna("UNKNOWN")
    rows = []
    for ind, members in inds.groupby(inds):
        sub_pw = pw.reindex(members.index).fillna(0.0)
        sub_bw = bw.reindex(members.index).fillna(0.0)
        sub_rt = rt.reindex(members.index).fillna(0.0)
        total_pw = sub_pw.sum()
        total_bw = sub_bw.sum()
        if total_bw <= 0 or total_pw <= 0:
            continue
        rp = (sub_pw / total_pw * sub_rt).sum()
        rb = (sub_bw / total_bw * sub_rt).sum()
        allocation = ((sub_pw - sub_bw) / total_pw * rb).sum()
        selection = (sub_bw / total_bw * (sub_rt - rb)).sum()
        interaction = ((sub_pw - sub_bw) / total_pw * (sub_rt - rb)).sum()
        rows.append({
            "date": d,
            "industry": ind,
            "allocation": float(allocation),
            "selection": float(selection),
            "interaction": float(interaction),
            "excess": float(rp - rb),
        })
    return pd.DataFrame(rows)