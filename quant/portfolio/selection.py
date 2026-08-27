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


def cap_weights_waterfill(w: pd.Series | "pd.DataFrame", cap: float, weight_col: str = "weight") -> pd.Series:
    """权重封顶投影（water-filling）：超限部分按比例摊给未超限者。

    与"clip 后再归一化"的本质区别：归一化会把已截断的权重重新抬过上限。
    若全部触顶仍有剩余，残额不分配（等价于留现金，保守方向），sum ≤ 1。
    """
    if isinstance(w, pd.DataFrame):
        s = w[weight_col].astype(float)
    else:
        s = w.astype(float)
    if s.empty or cap >= 1.0:
        return s
    total = float(s.sum())
    if total <= 0:
        return s
    for _ in range(64):
        over = s > cap + 1e-12
        if not over.any():
            break
        excess = float((s[over] - cap).sum())
        s[over] = cap
        under = ~over
        room = s[under]
        denom = float(room.sum())
        if denom <= 1e-15:
            break
        add = excess * (room / denom)
        # 防止摊派后再次越限：只补到剩余额度
        headroom = np.maximum(cap - room, 0.0)
        take = np.minimum(add, headroom)
        s[under] = room + take
        excess -= float(take.sum())
        if excess <= 1e-15:
            break
    return s


def members_asof(constituents: pd.DataFrame, date: pd.Timestamp) -> set[str] | None:
    """返回 date 时点的指数成员（最近一期快照 ≤ date）。无更早快照时返回 None。"""
    import numpy as np

    if constituents is None or constituents.empty or "snapshot_date" not in constituents.columns:
        return None
    snaps = np.sort(constituents["snapshot_date"].unique())
    pos = snaps.searchsorted(np.datetime64(pd.Timestamp(date)), side="right") - 1
    if pos < 0:
        return None
    sd = snaps[pos]
    return set(
        constituents.loc[constituents["snapshot_date"] == sd, "symbol"].astype(str)
    )


def build_target_weights(
    scores: pd.DataFrame,
    prices: pd.DataFrame,
    industry: pd.DataFrame,
    cfg: PortfolioConfig,
    rebalance_dates: list[pd.Timestamp],
    prev_weights: pd.DataFrame | None = None,
    constituents: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """对每个调仓日生成目标权重（date, symbol, weight）。

    constituents 为点内成分快照（snapshot_date, symbol）时，按调仓日当时
    的真实指数成员过滤横截面——消除"用今日成分回测历史"的幸存者偏差。
    """
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
    pit_active = constituents is not None and not constituents.empty
    if pit_active:
        import logging

        logging.getLogger("ashare.portfolio").info(
            "幸存者偏差防线启用：按点内指数成分过滤（%d 期快照）",
            constituents["snapshot_date"].nunique(),
        )

    for date in rebalance_dates:
        if date not in scores_by_date:
            continue
        amount_row = amount_wide.loc[date] if date in amount_wide.index else None
        uni = _universe_ok(date, prices, industry, cfg, min_avg_amount=cfg.min_avg_amount, amount_row=amount_row)
        if uni.empty:
            continue
        ok_syms = set(uni.loc[uni["ok"], "symbol"])
        if pit_active:
            mem = members_asof(constituents, date)
            if mem is not None:
                ok_syms &= mem
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
        # 单票上限的硬下限：等权 1/N ≤ max_weight 要求 N ≥ ceil(1/max_weight)，
        # 否则行业剔除循环会把 N 削到 15 只 → 等权 6.7%，直接击穿 5% 合同上限
        min_names_cap = int(np.ceil(1.0 / max(cfg.max_weight, 1e-9)))
        hard_floor = max(10, cfg.top_n // 2, min_names_cap)
        while True:
            w = 1.0 / len(chosen_df)
            counts = chosen_df["industry"].value_counts()
            bad = counts[counts * w > cfg.max_industry]
            if bad.empty:
                break
            worst_industry = bad.index[0]
            drop_sym = chosen_df[chosen_df["industry"] == worst_industry].iloc[-1]["symbol"]
            if len(chosen_df) - 1 < hard_floor:
                break
            chosen_df = chosen_df[chosen_df["symbol"] != drop_sym]

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

        # 最终单票上限强制执行：等权超限（候选不足）时按 water-filling 封顶，
        # 残额留现金；任何路径下都不得输出超过 max_weight 的目标权重
        w_eq = 1.0 / len(chosen_df)
        if w_eq > cfg.max_weight + 1e-9:
            capped = cap_weights_waterfill(
                pd.Series(np.full(len(chosen_df), w_eq), index=chosen_df["symbol"]),
                cfg.max_weight,
            )
            chosen_df["weight"] = chosen_df["symbol"].map(capped).to_numpy()
        else:
            chosen_df["weight"] = w_eq
        assert (chosen_df["weight"] <= cfg.max_weight + 1e-6).all(), (
            f"单票权重上限被击穿: max={chosen_df['weight'].max():.4f} > {cfg.max_weight}"
        )
        chosen_df["date"] = date
        out.append(chosen_df[["date", "symbol", "weight"]])
        prev = dict(zip(chosen_df["symbol"], chosen_df["weight"]))

    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["date", "symbol", "weight"]
    )
