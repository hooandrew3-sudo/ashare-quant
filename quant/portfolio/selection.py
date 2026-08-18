"""组合构建：top-N 等权 + 换手控制 + 行业/个股权重约束。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.config import PortfolioConfig


def _universe_ok(
    date: pd.Timestamp,
    prices: pd.DataFrame,
    industry: pd.DataFrame,
    cfg: PortfolioConfig,
    min_avg_amount: float = 50_000_000,
    amount_row: pd.Series | None = None,
) -> pd.DataFrame:
    """返回该调仓日可交易股票及行业映射。"""
    day = prices[prices["date"] == date]
    if day.empty:
        return pd.DataFrame(columns=["symbol", "industry", "ok"])
    day = day.set_index("symbol")
    if amount_row is not None:
        day["avg_amount"] = amount_row.reindex(day.index).fillna(0.0)
    elif "amount" in day.columns:
        recent = (
            prices[prices["date"] <= date]
            .sort_values("date")[["symbol", "amount"]]
            .groupby("symbol", group_keys=False)
            .tail(60)
        )
        day["avg_amount"] = recent.groupby("symbol")["amount"].mean().reindex(day.index).fillna(0.0)
    else:
        day["avg_amount"] = 0.0
    day["ok"] = True
    if "is_st" in day.columns:
        day.loc[day["is_st"].astype(bool), "ok"] = False
    if "is_suspended" in day.columns:
        day.loc[day["is_suspended"].astype(bool), "ok"] = False
    if "is_limit_up" in day.columns:
        day.loc[day["is_limit_up"].astype(bool), "ok"] = False
    day.loc[day["close"].isna(), "ok"] = False
    day.loc[day["avg_amount"] < min_avg_amount, "ok"] = False
    ind_map: pd.Series = pd.Series(dtype=object)
    if not industry.empty:
        ind_map = industry.sort_values("as_of_date").drop_duplicates("symbol", keep="last")
        ind_map = ind_map.set_index("symbol")["industry"]
    day["industry"] = ind_map.reindex(day.index)
    return day.reset_index()[["symbol", "industry", "ok", "avg_amount"]]


def build_target_weights(
    scores: pd.DataFrame,
    prices: pd.DataFrame,
    industry: pd.DataFrame,
    cfg: PortfolioConfig,
    rebalance_dates: list[pd.Timestamp],
    prev_weights: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """对每个调仓日生成目标权重（date, symbol, weight）。"""
    out: list[pd.DataFrame] = []
    prev: dict[str, float] = {}
    if prev_weights is not None and not prev_weights.empty:
        prev_grp = prev_weights.sort_values("date")
        last_date = prev_grp["date"].iloc[-1]
        prev = dict(
            zip(
                prev_grp.loc[prev_grp["date"] == last_date, "symbol"],
                prev_grp.loc[prev_grp["date"] == last_date, "weight"],
            )
        )

    scores_by_date = {pd.Timestamp(d): g for d, g in scores.groupby("date")}
    amount_wide = (
        prices.pivot(index="date", columns="symbol", values="amount")
        .rolling(60, min_periods=20)
        .mean()
    )

    for date in rebalance_dates:
        if date not in scores_by_date:
            continue
        amount_row = amount_wide.loc[date] if date in amount_wide.index else None
        uni = _universe_ok(date, prices, industry, cfg, min_avg_amount=cfg.min_avg_amount, amount_row=amount_row)
        if uni.empty:
            continue
        ok_syms = set(uni.loc[uni["ok"], "symbol"])
        sc = scores_by_date[date].copy()
        sc = sc[sc["symbol"].isin(ok_syms)]
        if sc.empty:
            continue
        # 粘性：旧持仓分数打 0.8 折
        sc["adj"] = sc["score"] * np.where(
            sc["symbol"].isin(prev.keys()), cfg.stickiness, 1.0
        )
        sc = sc.sort_values("adj", ascending=False)

        # 换手控制：最多替换 turnover_cap × N 个新名字
        max_new = max(1, int(round(cfg.top_n * cfg.turnover_cap)))
        chosen: list[str] = []
        new_count = 0
        for s in sc["symbol"]:
            if len(chosen) >= cfg.top_n:
                break
            if s in prev and prev.get(s, 0) > 0:
                chosen.append(s)
            elif new_count < max_new:
                chosen.append(s)
                new_count += 1
        if not chosen:
            continue

        # 行业约束：单行业权重 ≤ max_industry
        ind_of = uni.set_index("symbol")["industry"].to_dict()
        chosen_df = pd.DataFrame({"symbol": chosen})
        chosen_df["industry"] = chosen_df["symbol"].map(ind_of)
        while True:
            w = 1.0 / len(chosen_df)
            counts = chosen_df["industry"].value_counts()
            bad = counts[counts * w > cfg.max_industry]
            if bad.empty:
                break
            worst_industry = bad.index[0]
            drop_sym = chosen_df[chosen_df["industry"] == worst_industry].iloc[-1]["symbol"]
            chosen_df = chosen_df[chosen_df["symbol"] != drop_sym]
            if len(chosen_df) < max(10, cfg.top_n // 2):
                break

        # 市值分层约束：小市值股票（当日成交额 bottom 30%）数量 ≤ max_smallcap_ratio × N
        if getattr(cfg, "max_smallcap_ratio", 1.0) < 1.0:
            size_row = uni.set_index("symbol")["avg_amount"].rank(pct=True)
            smallcap_syms = set(size_row[size_row <= 0.3].index)
            max_small = max(1, int(round(len(chosen_df) * cfg.max_smallcap_ratio)))
            small_in = [s for s in chosen_df["symbol"] if s in smallcap_syms]
            if len(small_in) > max_small:
                chosen_set = set(chosen_df["symbol"])
                candidates = [
                    s for s in sc["symbol"]
                    if s not in chosen_set and s not in smallcap_syms
                ]
                for s in small_in[max_small:]:
                    if not candidates:
                        break
                    repl = candidates.pop(0)
                    chosen_df["symbol"] = chosen_df["symbol"].replace(s, repl)

        w = 1.0 / len(chosen_df)
        chosen_df["weight"] = w
        chosen_df["date"] = date
        out.append(chosen_df[["date", "symbol", "weight"]])
        prev = dict(zip(chosen_df["symbol"], chosen_df["weight"]))

    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["date", "symbol", "weight"]
    )
