"""压力测试：指定历史场景内的回撤与收益。"""

from __future__ import annotations

import pandas as pd


def run_stress(
    equity: pd.DataFrame,
    scenarios: dict[str, list[str]],
) -> dict[str, dict]:
    """在指定区间内计算最大回撤与区间收益。"""
    out = {}
    if equity.empty:
        return out
    e = equity.set_index("date")["portfolio_value"]
    for name, (start, end) in scenarios.items():
        window = e.loc[(e.index >= pd.Timestamp(start)) & (e.index <= pd.Timestamp(end))]
        if len(window) < 2:
            out[name] = {
                "coverage_days": int(len(window)),
                "max_drawdown": None,
                "return": None,
                "ok": False,
                "reason": "insufficient_coverage",
            }
            continue
        if len(window) < 30:
            # 覆盖不足 30 个交易日的结果不具统计意义，明确标 FAIL 而非“通过”
            out[name] = {
                "coverage_days": int(len(window)),
                "max_drawdown": None,
                "return": None,
                "ok": False,
                "reason": "coverage_lt_30d",
            }
            continue
        peak = window.cummax()
        dd = float((window / peak - 1.0).min())
        ret = float(window.iloc[-1] / window.iloc[0] - 1.0)
        out[name] = {
            "coverage_days": int(len(window)),
            "max_drawdown": round(dd, 4),
            "return": round(ret, 4),
            "ok": True,
        }
    return out
