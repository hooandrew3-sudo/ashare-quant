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


def _weights_of(report: dict, names: list[str], weight_by: str) -> dict[str, float]:
    if weight_by == "icir":
        return {
            name: max(abs(report["factors"][name]["icir"] or 0.05), 0.05)
            for name in names
        }
    if weight_by == "t":
        return {
            name: max(abs(report["factors"][name]["t_stat"] or 1.0), 0.1)
            for name in names
        }
    return {name: 1.0 for name in names}


def build_composite_factor(
    factor_long: pd.DataFrame,
    ic_report: dict,
    n: int = 5,
    min_t: float = 2.0,
    corr_max: float = 0.6,
    weight_by: str = "icir",  # icir | t | equal
    require_stable_decay: bool = True,
    regime_bull_dates: "pd.DatetimeIndex | set | None" = None,
    ic_report_bull: "dict | None" = None,
    ic_report_bear: "dict | None" = None,
) -> pd.DataFrame:
    """返回 composite 因子的 long 表；不足 2 个可用因子时返回空。

    regime 轮动（可选）：同时传入 regime_bull_dates（牛市日期集合）与
    牛/熊两个分域 IC 报告时，成分与权重分别按分域估计，逐日按当日所处
    状态选用——牛市偏进攻因子（动量/盈利改善），熊市偏防御因子
    （价值/低波）。分域样本不足或选不出 ≥2 个成分时，该日回退全窗口
    权重（保守降级，不产生空值）。
    """
    corr = factor_correlations(factor_long)
    picks = _select_components(
        ic_report, corr, n, min_t, corr_max, require_stable_decay=require_stable_decay
    )
    if len(picks) < 2:
        return pd.DataFrame(columns=["date", "symbol", "factor", "value"])
    sub = factor_long[factor_long["factor"].isin(picks)]
    wide = sub.pivot_table(index=["date", "symbol"], columns="factor", values="value")

    def _weighted(report: dict, names: list[str]) -> pd.Series:
        w = _weights_of(report, names, weight_by)
        total = sum(w.values())
        out = None
        for name, wv in w.items():
            term = wide[name] * (wv / total)
            out = term if out is None else out + term
        return out

    comp = _weighted(ic_report, picks)

    # ---- regime 条件化 ----
    if (
        regime_bull_dates is not None
        and ic_report_bull is not None
        and ic_report_bear is not None
    ):
        import numpy as np

        bull_picks = _select_components(
            ic_report_bull, corr, n, min_t, corr_max, require_stable_decay=require_stable_decay
        )
        bear_picks = _select_components(
            ic_report_bear, corr, n, min_t, corr_max, require_stable_decay=require_stable_decay
        )
        bull_set = {
            pd.Timestamp(d) for d in (
                list(regime_bull_dates)
                if not isinstance(regime_bull_dates, (set, frozenset))
                else regime_bull_dates
            )
        }
        date_index = wide.index.get_level_values("date")
        is_bull = np.array([d in bull_set for d in date_index], dtype=bool)
        bull_ok = len(bull_picks) >= 2
        bear_ok = len(bear_picks) >= 2
        if bull_ok or bear_ok:
            # 分域成分可能与全窗口 picks 不同：按并集重建宽表，保证 _weighted 可索引
            needed = set(picks) | set(bull_picks) | set(bear_picks)
            if not needed <= set(wide.columns):
                sub_all = factor_long[factor_long["factor"].isin(needed)]
                wide = sub_all.pivot_table(
                    index=["date", "symbol"], columns="factor", values="value"
                )
            bull_vals = _weighted(ic_report_bull, bull_picks).to_numpy(dtype=float) if bull_ok else comp.to_numpy(dtype=float)
            bear_vals = _weighted(ic_report_bear, bear_picks).to_numpy(dtype=float) if bear_ok else comp.to_numpy(dtype=float)
            merged = np.where(is_bull, bull_vals, bear_vals)
            comp = pd.Series(merged, index=wide.index)

    out = comp.rename("value").reset_index()
    out["factor"] = "composite"
    return out[["date", "symbol", "factor", "value"]]
