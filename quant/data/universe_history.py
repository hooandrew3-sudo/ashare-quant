"""历史大市值成员代理：用「季度总股本 × 公告日收盘价」估计历史市值，
取每季度前 top 名的并集，近似"曾入选大盘指数但已调出"的股票。"""

from __future__ import annotations

import pandas as pd


def quarterly_market_cap(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
) -> pd.DataFrame:
    """返回 long：symbol, as_of_date, market_cap。"""
    f = fundamentals.dropna(subset=["total_share"]).copy()
    f = f[f["total_share"] > 0]
    if f.empty:
        return pd.DataFrame(columns=["symbol", "as_of_date", "market_cap"])
    f["as_of_date"] = pd.to_datetime(f["as_of_date"]).astype("datetime64[ns]")
    p = prices.copy()
    p["date"] = pd.to_datetime(p["date"]).astype("datetime64[ns]")
    p = p.sort_values(["symbol", "date"])

    rows: list[dict] = []
    for sym, g in f.groupby("symbol", sort=False):
        pg = p[p["symbol"] == sym].sort_values("date")
        if pg.empty:
            continue
        # 公告日（或最近前一日）收盘价
        m = pd.merge_asof(
            g.sort_values("as_of_date"),
            pg[["date", "close"]],
            left_on="as_of_date",
            right_on="date",
            direction="backward",
        )
        cap = m["close"] * m["total_share"]
        rows.extend(
            {
                "symbol": sym,
                "as_of_date": ad,
                "market_cap": round(float(c), 0),
            }
            for ad, c in zip(m["as_of_date"], cap)
            if c == c and c > 0
        )
    return pd.DataFrame(rows)


def historical_largecap_members(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
    top: int = 900,
) -> list[str]:
    """各季度市值前 top 名的并集（按 as_of_date 截面排名）。"""
    cap = quarterly_market_cap(prices, fundamentals)
    if cap.empty:
        return []
    members: set[str] = set()
    for _, group in cap.groupby("as_of_date"):
        top_syms = group.nlargest(top, "market_cap")["symbol"].tolist()
        members.update(top_syms)
    return sorted(members)
