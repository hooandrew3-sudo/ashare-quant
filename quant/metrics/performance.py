"""绩效指标：收益/风险/回撤/胜率/归因。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def max_drawdown(series: pd.Series) -> float:
    if series.empty or series.iloc[0] <= 0:
        return 0.0
    peak = series.cummax()
    dd = (series / peak - 1.0).min()
    return float(dd)


def compute_metrics(
    equity: pd.DataFrame,
    monthly: pd.DataFrame,
    trades: pd.DataFrame,
    risk_free: float = 0.02,
) -> dict:
    if equity.empty or len(equity) < 2:
        return {}
    e = equity.set_index("date")
    pv = e["portfolio_value"]
    bv = e["benchmark_value"]
    years = max((e.index[-1] - e.index[0]).days / 365.25, 1e-9)

    total_ret = pv.iloc[-1] / pv.iloc[0] - 1.0
    bench_ret = bv.iloc[-1] / bv.iloc[0] - 1.0
    cagr = (pv.iloc[-1] / pv.iloc[0]) ** (1 / years) - 1.0
    daily = pv.pct_change().dropna()
    vol = float(daily.std() * np.sqrt(252)) if len(daily) else 0.0
    sharpe = (cagr - risk_free) / vol if vol > 0 else 0.0
    mdd = max_drawdown(pv)
    calmar = cagr / abs(mdd) if mdd < 0 else float("inf")

    excess_daily = daily - bv.pct_change().dropna().reindex(daily.index).fillna(0.0)
    ir = float(excess_daily.mean() / excess_daily.std() * np.sqrt(252)) if excess_daily.std() > 0 else 0.0

    # alpha/beta（OLS）
    y = daily.to_numpy()
    x = bv.pct_change().dropna().reindex(daily.index).fillna(0.0).to_numpy()
    A = np.vstack([x, np.ones(len(x))]).T
    beta, alpha = np.linalg.lstsq(A, y, rcond=None)[0]

    monthly_win_rate = float((monthly["return"] > 0).mean()) if len(monthly) else 0.0
    turnover = _turnover(trades, pv, years)
    cost_drag = float(trades["fee"].sum()) if not trades.empty else 0.0
    filled_sells = trades[(trades["side"] == "sell") & (trades["status"] == "filled")]
    hit_rate = float(
        (filled_sells["price"] > filled_sells["entry_price"]).mean()
    ) if len(filled_sells) else None

    return {
        "total_return": round(float(total_ret), 4),
        "benchmark_return": round(float(bench_ret), 4),
        "excess_return": round(float(total_ret - bench_ret), 4),
        "annualized_return": round(float(cagr), 4),
        "annualized_vol": round(vol, 4),
        "sharpe": round(float(sharpe), 3),
        "max_drawdown": round(mdd, 4),
        "calmar": round(float(calmar), 3) if calmar != float("inf") else None,
        "information_ratio": round(float(ir), 3),
        "alpha": round(float(alpha * 252), 5),
        "beta": round(float(beta), 3),
        "monthly_win_rate": round(monthly_win_rate, 4),
        "turnover_annual": round(float(turnover), 4),
        "cost_drag_total": round(cost_drag, 2),
        "sell_hit_rate": round(float(hit_rate), 4) if hit_rate is not None else None,
        "days": int(len(daily)),
    }


def _turnover(trades: pd.DataFrame, pv: pd.Series, years: float) -> float:
    """单边年化换手：min(买额, 卖额) ≈ (买额+卖额)/2 / 平均净值 / 年数。

    此前实现为“全程双边累计 / 平均净值”，既非单边也非年化，且与引擎
    max_turnover_annual 熔断口径不一致（阈值实际放大了 2 倍）。
    """
    if trades.empty:
        return 0.0
    t = trades[(trades["status"] == "filled")].copy()
    t["amount"] = t["price"] * t["shares"]
    buy_amt = float(t.loc[t["side"] == "buy", "amount"].sum())
    sell_amt = float(t.loc[t["side"] == "sell", "amount"].sum())
    one_side = (buy_amt + sell_amt) / 2.0
    avg_equity = float(pv.mean())
    annual = one_side / avg_equity / years if avg_equity > 0 and years > 0 else 0.0
    return annual
