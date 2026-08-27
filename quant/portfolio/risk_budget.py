"""风险预算（Risk Budgeting）组合：替代等权 Top-N 的收益增强模块。"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from quant.config import PortfolioConfig

LOGGER = logging.getLogger("ashare.risk_budget")


def _estimate_covariance(
    prices: pd.DataFrame,
    date: pd.Timestamp,
    window: int = 60,
    min_periods: int = 20,
) -> Optional[pd.DataFrame]:
    """估计截至 date 的滚动日收益协方差矩阵。"""
    hist = prices[prices["date"] <= date].sort_values("date")
    if hist.empty:
        return None
    close_wide = hist.pivot(index="date", columns="symbol", values="close")
    if close_wide.shape[0] < 2:
        return None
    ret = close_wide.pct_change().dropna(how="all")
    ret = ret.tail(window)
    if ret.shape[0] < min_periods:
        return None
    # drop 全为 NaN 的股票
    ret = ret.dropna(axis=1, how="all")
    cov = ret.cov(min_periods=min_periods)
    # 剔除非正定或奇异矩阵
    cov = cov.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if cov.shape[0] == 0:
        return None
    # 确保对称 + 对角 jitter，避免奇异矩阵（解析解仅用对角线 vols，对 off-diagonal 不敏感）
    cov = (cov + cov.T) / 2.0
    cov = cov + np.eye(cov.shape[0]) * 1e-3
    cov = pd.DataFrame(
        np.nan_to_num(cov.values, nan=0.0, posinf=0.0, neginf=0.0),
        index=cov.index,
        columns=cov.columns,
    )
    cov = (cov + cov.T) / 2.0
    return cov


def build_risk_budget_weights(
    scores: pd.Series,
    cov: pd.DataFrame,
    target_risk: float = 0.10,
    max_weight: float = 0.05,
    gamma: float = 1.5,
) -> pd.Series:
    """基于信号强度的风险预算求解最优权重。

    Parameters
    ----------
    scores : 信号强度（截面分位或概率），index=symbol
    cov : 协方差矩阵（symbol × symbol）
    target_risk : 组合目标年化波动率
    max_weight : 单票权重上限
    gamma : 信号强度幂指数（>1 放大头部权重，<1 更分散）

    Returns
    -------
    pd.Series : weight（symbol → weight）
    """
    common = scores.index.intersection(cov.index)
    if len(common) < 2:
        return pd.Series(dtype=float)
    scores = scores.loc[common]
    cov = cov.loc[common, common]

    # 风险预算 ∝ |score|^gamma
    raw_budget = np.abs(scores.values) ** gamma
    total = raw_budget.sum()
    if total <= 0:
        budgets = np.ones(len(scores)) / len(scores)
    else:
        budgets = raw_budget / total

    # 解析解：风险平价近似 w ∝ budgets / σ_i（当相关性均匀时）
    # 先用等权 risk parity 初始化
    diag = np.diag(cov.values)
    vols = np.sqrt(np.maximum(diag, 1e-12))
    # 工业稳健下限：防止极小波动率（如低流动/仙股）导致权重爆炸
    min_vol = max(float(np.percentile(vols, 10)), 1e-4)
    vols = np.clip(vols, min_vol, None)
    w = budgets / vols
    w = w / w.sum()

    # 截断 + 再归一化
    w = np.clip(w, 0.0, max_weight)
    total = w.sum()
    if total > 0:
        w = w / total
    else:
        w = np.ones(len(scores)) / len(scores)

    return pd.Series(w, index=common)


def build_target_weights_with_risk_budget(
    scores: pd.DataFrame,
    prices: pd.DataFrame,
    industry: pd.DataFrame,
    cfg: PortfolioConfig,
    rebalance_dates: list[pd.Timestamp],
    prev_weights: pd.DataFrame | None = None,
    constituents: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """风险预算版目标权重生成。

    与 selection.build_target_weights 接口一致，供 pipeline 统一调度。
    """
    out: list[pd.DataFrame] = []
    prev: dict[str, float] = {}
    if prev_weights is not None and not prev_weights.empty:
        last_date = prev_weights["date"].iloc[-1]
        prev = dict(
            zip(
                prev_weights.loc[prev_weights["date"] == last_date, "symbol"],
                prev_weights.loc[prev_weights["date"] == last_date, "weight"],
            )
        )

    scores_by_date = {pd.Timestamp(d): g for d, g in scores.groupby("date")}

    for date in rebalance_dates:
        if date not in scores_by_date:
            continue
        cov = _estimate_covariance(prices, date, window=cfg.vol_window, min_periods=cfg.vol_window // 2)
        if cov is None or cov.shape[0] < 5:
            LOGGER.warning(" %s 协方差估计不足（stock=%d），跳过风险预算，回退等权", date.date(), 0 if cov is None else cov.shape[0])
            from quant.portfolio.selection import build_target_weights

            tw = build_target_weights(scores, prices, industry, cfg, [date], prev_weights)
            if not tw.empty:
                out.append(tw)
                prev = dict(zip(tw["symbol"], tw["weight"]))
            continue

        sc = scores_by_date[date].copy()
        # 点内成分过滤（与 selection 同语义，消除幸存者偏差）
        if constituents is not None and not constituents.empty:
            from quant.portfolio.selection import members_asof

            mem = members_asof(constituents, date)
            if mem is not None:
                sc = sc[sc.index.isin(mem)]
        sc = sc.set_index("symbol")["score"]

        # 可交易过滤（先按 sc.index 对齐，防止 scores 含当日无行情的 symbol 时
        # 布尔索引抛 Unalignable 异常导致整个回测中断）
        day = prices[prices["date"] == date].set_index("symbol")
        if day.empty:
            continue
        day = day.reindex(sc.index)
        if "is_st" in day.columns:
            sc = sc[~day["is_st"].fillna(False).astype(bool)]
        if "is_suspended" in day.columns:
            sc = sc[~day["is_suspended"].fillna(False).astype(bool)]
        if "is_limit_up" in day.columns:
            sc = sc[~day["is_limit_up"].fillna(False).astype(bool)]
        if "close" in day.columns:
            sc = sc[(day["close"] > 0).fillna(False)]
        # 流动性过滤
        if "amount" in day.columns:
            recent = (
                prices[prices["date"] <= date]
                .sort_values("date")[["symbol", "amount"]]
                .groupby("symbol", group_keys=False)
                .tail(cfg.vol_window)
            )
            avg_amount = recent.groupby("symbol")["amount"].mean()
            sc = sc[avg_amount.reindex(sc.index).fillna(0) >= cfg.min_avg_amount]
        if len(sc) == 0:
            continue

        cov = cov.loc[cov.index.intersection(sc.index), cov.index.intersection(sc.index)]
        sc = sc.loc[cov.index]

        w = build_risk_budget_weights(
            sc, cov, target_risk=cfg.target_vol, max_weight=cfg.max_weight, gamma=1.5,
        )

        # 投粘性：旧持仓权重打 0.8 折
        if prev and cfg.stickiness < 1.0:
            w = w.copy()
            for sym in w.index:
                if sym in prev and prev[sym] > 0:
                    w[sym] = w[sym] * cfg.stickiness
            w = w / w.sum() if w.sum() > 0 else w

        # 换手控制：仅允许部分新名字
        max_new = max(1, int(round(cfg.top_n * cfg.turnover_cap)))
        new_count = 0
        chosen = []
        # 按信号降序排列，保留可交易且有权重的
        ordered = w.sort_values(ascending=False)
        for sym, weight in ordered.items():
            if len(chosen) >= cfg.top_n:
                break
            if sym in prev and prev.get(sym, 0) > 0:
                chosen.append(sym)
            elif new_count < max_new:
                chosen.append(sym)
                new_count += 1
        if not chosen:
            continue

        chosen_df = pd.DataFrame({"symbol": chosen})
        # 保留风险预算权重（此前在行业约束分支被等权覆盖，risk_budget 退化为等权）
        chosen_df["weight"] = chosen_df["symbol"].map(w).fillna(1.0 / len(chosen_df)).to_numpy()
        chosen_df["industry"] = chosen_df["symbol"].map(
            industry.sort_values("as_of_date").drop_duplicates("symbol", keep="last").set_index("symbol")["industry"]
            if "symbol" in industry.columns and "industry" in industry.columns and not industry.empty
            else pd.Series(dtype=object)
        )
        # 行业约束：直接在真实权重上做投影，超限行业剔除权重最小成员后归一化
        if "industry" in chosen_df.columns and not chosen_df["industry"].isna().all():
            min_count = max(10, cfg.top_n // 2)
            while True:
                total = chosen_df["weight"].sum()
                if total > 0:
                    ind_w = chosen_df.groupby("industry")["weight"].sum() / total
                else:
                    ind_w = pd.Series(dtype=float)
                bad = ind_w[ind_w > cfg.max_industry]
                if bad.empty:
                    break
                worst = bad.idxmax()
                cand = chosen_df[chosen_df["industry"] == worst]
                drop = cand.loc[cand["weight"].idxmin(), "symbol"]
                chosen_df = chosen_df[chosen_df["symbol"] != drop]
                if len(chosen_df) < min_count:
                    break
            chosen_df["weight"] = chosen_df["weight"] / chosen_df["weight"].sum()
        # 单票上限：water-filling 投影（clip 后归一化会把权重重新抬过上限，
        # 属约束失效；残额不分配 = 留现金，保守方向）
        from quant.portfolio.selection import cap_weights_waterfill

        capped = cap_weights_waterfill(chosen_df.set_index("symbol")["weight"], cfg.max_weight)
        chosen_df["weight"] = chosen_df["symbol"].map(capped).to_numpy()
        assert (chosen_df["weight"] <= cfg.max_weight + 1e-6).all(), (
            f"risk_budget 单票权重上限被击穿: max={chosen_df['weight'].max():.4f}"
        )

        chosen_df["date"] = date
        out.append(chosen_df[["date", "symbol", "weight"]])
        prev = dict(zip(chosen_df["symbol"], chosen_df["weight"]))

    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["date", "symbol", "weight"])
