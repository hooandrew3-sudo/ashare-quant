"""复合因子：方向显著 + 低相关成分的 ICIR/t 加权合成，提升 ICIR。"""

from __future__ import annotations

import pandas as pd


def factor_correlations(factor_long: pd.DataFrame) -> pd.DataFrame:
    """因子间截面秩相关矩阵（逐日排名后合并计算，用于去冗余）。"""
    wide = factor_long.pivot_table(index=["date", "symbol"], columns="factor", values="value")
    ranked = wide.groupby(level="date").rank()
    return ranked.corr(method="spearman")


def _select_components(
    ic_report: dict,
    corr: pd.DataFrame,
    n: int,
    min_t: float,
    corr_max: float,
    require_stable_decay: bool = True,
) -> list[str]:
    """贪心选择：|IC| 排序 + 方向显著 + 相关性 ≤ corr_max + 衰减稳定。"""
    ranked = sorted(
        ic_report["factors"].items(),
        key=lambda kv: abs(kv[1]["rank_ic_mean"] or 0),
        reverse=True,
    )
    candidates = [
        name
        for name, r in ranked
        if r.get("t_stat") is not None
        and abs(r["t_stat"]) >= min_t
        and r.get("rank_ic_mean") is not None
        and _decay_ok(r, require_stable_decay)
    ]
    if not candidates:
        return []
    selected = [candidates[0]]
    for name in candidates[1:]:
        if len(selected) >= n:
            break
        if all(abs(float(corr.loc[name, s])) <= corr_max for s in selected):
            selected.append(name)
    return selected


def _decay_ok(r: dict, require: bool) -> bool:
    """衰减稳定性：5 日与 20 日 IC 同号，且 20 日 |IC| ≥ 0.5×5 日 |IC|。"""
    if not require:
        return True
    d5 = r.get("decay", {}).get("5")
    d20 = r.get("decay", {}).get("20")
    if d5 is None or d20 is None:
        return False
    if d5 * d20 < 0:
        return False
    if abs(d5) > 1e-9 and abs(d20) < 0.5 * abs(d5):
        return False
    return True


def build_composite_factor(
    factor_long: pd.DataFrame,
    ic_report: dict,
    n: int = 5,
    min_t: float = 2.0,
    corr_max: float = 0.6,
    weight_by: str = "icir",  # icir | t | equal
    require_stable_decay: bool = True,
) -> pd.DataFrame:
    """返回 composite 因子的 long 表；不足 2 个可用因子时返回空。"""
    corr = factor_correlations(factor_long)
    picks = _select_components(
        ic_report, corr, n, min_t, corr_max, require_stable_decay=require_stable_decay
    )
    if len(picks) < 2:
        return pd.DataFrame(columns=["date", "symbol", "factor", "value"])
    sub = factor_long[factor_long["factor"].isin(picks)]
    wide = sub.pivot_table(index=["date", "symbol"], columns="factor", values="value")
    if weight_by == "icir":
        weights = {
            name: max(abs(ic_report["factors"][name]["icir"] or 0.05), 0.05)
            for name in picks
        }
    elif weight_by == "t":
        weights = {name: max(abs(ic_report["factors"][name]["t_stat"] or 1.0), 0.1) for name in picks}
    else:
        weights = {name: 1.0 for name in picks}
    total = sum(weights.values())
    weighted = sum(wide[name] * (w / total) for name, w in weights.items())
    comp = weighted.rename("value").reset_index()
    comp["factor"] = "composite"
    return comp[["date", "symbol", "factor", "value"]]
