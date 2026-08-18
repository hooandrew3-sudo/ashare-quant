"""因子 IC 分析：Rank IC / ICIR / 分层收益 / 衰减 / 准入判定。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from quant.factors.definitions import FACTOR_SPECS, factor_direction
from quant.utils import setup_logging


def _rank_ic_series(factor_wide: pd.DataFrame, label_wide: pd.DataFrame) -> pd.Series:
    """逐日 Spearman Rank IC（用横截面秩的 Pearson 等价计算）。"""
    f_rank = factor_wide.rank(axis=1)
    l_rank = label_wide.reindex(index=factor_wide.index, columns=factor_wide.columns).rank(axis=1)
    f_rank = f_rank.sub(f_rank.mean(axis=1), axis=0)
    l_rank = l_rank.sub(l_rank.mean(axis=1), axis=0)
    f_std = f_rank.std(axis=1).replace(0, np.nan)
    l_std = l_rank.std(axis=1).replace(0, np.nan)
    ic = (f_rank * l_rank).mean(axis=1) / (f_std * l_std)
    return ic.dropna()


def _quantile_returns(factor_wide: pd.DataFrame, label_wide: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """分位数平均未来超额收益（日期 × 分位）。"""
    label = label_wide.reindex(index=factor_wide.index, columns=factor_wide.columns)
    # 按横截面排名分桶，避免大量并列值导致 qcut 边界重复
    bucket = (factor_wide.rank(axis=1, pct=True) * n).fillna(0.0).astype(int).clip(0, n - 1)
    out = {}
    for b in range(n):
        out[b] = label.where(bucket == b).mean(axis=1)
    return pd.DataFrame(out)


def factor_ic_report(
    factor_long: pd.DataFrame,
    label_long: pd.DataFrame,
    cfg,
    decay_days: tuple[int, ...] = (5, 10, 20, 40),
    date_range: tuple[pd.Timestamp, pd.Timestamp] | None = None,
) -> dict:
    """对每个因子输出 IC 统计、分位收益、衰减与准入结论。

    date_range: 可选 (start, end)，仅使用该日期窗口计算 IC（用于样本内准入，防止未来信息泄露）。
    """
    log = setup_logging(cfg.run.verbose)
    labels = label_long.set_index(["date", "symbol"])["value"]
    label_wide = labels.unstack()
    if date_range is not None:
        start, end = date_range
        label_wide = label_wide.loc[start:end]
    results = {}
    for name, group in factor_long.groupby("factor", sort=False):
        f_wide = group.pivot(index="date", columns="symbol", values="value")
        if date_range is not None:
            start, end = date_range
            f_wide = f_wide.loc[start:end]
        if f_wide.empty or label_wide.empty:
            continue
        spec = FACTOR_SPECS.get(name)
        direction = factor_direction(name) if spec else 1
        f_wide = group.pivot(index="date", columns="symbol", values="value")
        ic = _rank_ic_series(f_wide, label_wide)
        # 方向统一：direction<0 的因子，IC 取反后再看显著性
        ic_adj = ic * direction
        ic_mean = float(ic_adj.mean()) if len(ic_adj) else np.nan
        ic_std = float(ic_adj.std()) if len(ic_adj) else np.nan
        icir = ic_mean / ic_std if ic_std and ic_std > 0 else np.nan
        t_stat = ic_mean / (ic_std / np.sqrt(len(ic_adj))) if ic_std and len(ic_adj) > 1 else np.nan
        qret = _quantile_returns(f_wide, label_wide)
        monotonic = float(qret.iloc[:, -1].mean() - qret.iloc[:, 0].mean()) if len(qret) else np.nan
        decay = {}
        for h in decay_days:
            shifted = labels.groupby(level="symbol").shift(-h)
            sw = shifted.unstack().reindex(index=f_wide.index, columns=f_wide.columns)
            # 与 rank_ic_mean 保持一致：衰减 IC 也做方向调整，避免 direction<0 因子符号口径不一致。
            decay[h] = float((_rank_ic_series(f_wide, sw) * direction).mean()) if len(ic_adj) else np.nan

        passed = bool(
            abs(ic_mean) >= cfg.factors.min_ic
            and abs(icir) >= cfg.factors.min_icir
            and (t_stat is not None and abs(t_stat) >= cfg.factors.min_t_stat)
            and np.sign(ic_mean) == np.sign(direction)
        )
        results[name] = {
            "category": spec["category"] if spec else "",
            "direction": direction,
            "rank_ic_mean": round(ic_mean, 5) if ic_mean == ic_mean else None,
            "rank_ic_std": round(ic_std, 5) if ic_std == ic_std else None,
            "icir": round(icir, 4) if icir == icir else None,
            "t_stat": round(t_stat, 3) if t_stat == t_stat else None,
            "n_days": int(len(ic_adj)),
            "q1_mean": float(qret.iloc[:, 0].mean()) if len(qret) else np.nan,
            "q5_mean": float(qret.iloc[:, -1].mean()) if len(qret) else np.nan,
            "spread_q5_q1": round(monotonic, 6),
            "decay": {str(k): round(v, 5) if v == v else None for k, v in decay.items()},
            "passed": passed,
        }
        log.info(
            "IC[%s] mean=%.4f icir=%.3f t=%.2f spread=%.5f passed=%s",
            name, ic_mean if ic_mean == ic_mean else 0,
            icir if icir == icir else 0,
            t_stat if t_stat == t_stat else 0,
            monotonic if monotonic == monotonic else 0,
            passed,
        )
    report = {
        "min_ic": cfg.factors.min_ic,
        "min_icir": cfg.factors.min_icir,
        "factors": results,
        "passed": [k for k, v in results.items() if v["passed"]],
    }
    return report


def save_ic_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)


def report_to_frame(report: dict) -> pd.DataFrame:
    rows = []
    for name, r in report["factors"].items():
        rows.append(
            {
                "factor": name,
                "category": r["category"],
                "ic_mean": r["rank_ic_mean"],
                "icir": r["icir"],
                "t_stat": r["t_stat"],
                "spread_q5_q1": r["spread_q5_q1"],
                "passed": r["passed"],
            }
        )
    return pd.DataFrame(rows)
