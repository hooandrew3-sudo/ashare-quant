"""合成数据生成器：用于无外部依赖跑通全链路（demo/CI/单测）。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.data.storage import DataBundle


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def generate_synthetic(
    n_stocks: int = 200,
    years: int = 4,
    start: str = "2021-01-01",
    seed: int = 42,
) -> DataBundle:
    """生成带弱信号、涨跌停、停牌、ST、退市现象的合成 A 股数据。"""
    rng = _rng(seed)
    dates = pd.bdate_range(start=start, periods=years * 252)
    n_days = len(dates)

    # ---- 市场因子：牛/熊/急跌三 regime ----
    regime = rng.choice(["bull", "bear", "crash"], size=n_days, p=[0.6, 0.3, 0.1])
    mu_map = {"bull": 0.0006, "bear": -0.0012, "crash": -0.004}
    vol_map = {"bull": 0.008, "bear": 0.014, "crash": 0.022}
    mkt_mu = np.array([mu_map[r] for r in regime])
    mkt_vol = np.array([vol_map[r] for r in regime])
    mkt_ret = rng.normal(mkt_mu, mkt_vol)
    mkt_ret = np.clip(mkt_ret, -0.055, 0.055)
    bench_close = 1000.0 * np.cumprod(1 + mkt_ret)

    # ---- 股票池 ----
    symbols = [f"{600000 + i:06d}.SH" for i in range(n_stocks // 2)] + [
        f"{1 + i:06d}.SZ" for i in range(n_stocks - n_stocks // 2)
    ]
    industries = [f"IND_{i:02d}" for i in range(20)]
    skill = rng.normal(0, 1, n_stocks)  # 潜在"质地"（驱动弱动量/低波信号）
    beta = rng.uniform(0.7, 1.3, n_stocks)
    idio_vol = rng.uniform(0.008, 0.03, n_stocks)
    base_price = rng.uniform(8, 80, n_stocks)
    float_shares = rng.uniform(0.5e8, 2e9, n_stocks)

    # ST 与退市标记
    st_flags = rng.random(n_stocks) < 0.05
    delisted = rng.random(n_stocks) < 0.06
    delist_dates = {
        s: dates[int(rng.integers(int(n_days * 0.5), n_days))] for s, d in zip(symbols, delisted) if d
    }
    list_dates = {
        s: dates[0] - pd.Timedelta(days=int(rng.integers(90, 365 * 3))) for s in symbols
    }

    rows: list[dict] = []
    close_prev = np.array(base_price)
    idio_prev = np.zeros(n_stocks)
    suspended_until = np.zeros(n_stocks, dtype=int)

    for d in range(n_days):
        dt = dates[d]
        day_ret = beta * mkt_ret[d] + 0.0005 * skill + 0.12 * idio_prev
        noise = rng.normal(0, idio_vol)
        day_ret = day_ret + noise
        # 涨跌停封顶（非 ST ±9.8%，ST ±5%）
        limit_cap = np.where(st_flags, 0.05, 0.098)
        day_ret = np.clip(day_ret, -limit_cap, limit_cap)
        raw_close = close_prev * (1 + day_ret)

        # 停牌：约 2% 概率停 1-5 天
        suspend_now = (suspended_until <= d) & (rng.random(n_stocks) < 0.02)
        suspended_until = np.where(suspend_now, d + rng.integers(1, 6, n_stocks), suspended_until)
        is_suspended = suspended_until > d

        prev_c = np.where(d == 0, base_price, close_prev)
        open_p = prev_c * (1 + rng.normal(0, 0.004, n_stocks))
        high_p = np.maximum(open_p, raw_close) * (1 + rng.uniform(0, 0.008, n_stocks))
        low_p = np.minimum(open_p, raw_close) * (1 - rng.uniform(0, 0.008, n_stocks))
        volume = (
            float_shares * 0.005 * (1 + 8 * np.abs(day_ret)) * rng.lognormal(0, 0.4, n_stocks)
        ) / 100.0  # 手
        amount = volume * 100 * raw_close
        turnover = (volume * 100 / float_shares) * 100.0

        # 动量/量能信号：高 skill 股票近期放量（让 vol_ratio 有弱信号）
        volume = volume * (1 + 0.3 * np.clip(skill, 0, 1))

        ret_capped = np.where(d == 0, 0.0, raw_close / np.where(d == 0, base_price, close_prev) - 1)
        is_lu = ~is_suspended & (ret_capped >= limit_cap - 1e-9) & (d > 0)
        is_ld = ~is_suspended & (ret_capped <= -limit_cap + 1e-9) & (d > 0)

        pe = np.clip(15 + 20 * rng.normal(0, 0.4, n_stocks) - 5 * skill, 3, 200)
        pb = np.clip(1.5 + 2.5 * rng.normal(0, 0.4, n_stocks) + 0.5 * skill, 0.3, 30)
        roe = np.clip(0.08 + 0.03 * skill + rng.normal(0, 0.02, n_stocks), 0.0, 0.35)
        div_yield = np.clip(0.01 + 0.05 * np.clip(roe, 0, 0.25), 0.0, 0.08)

        for i, s in enumerate(symbols):
            if s in delist_dates and dt > delist_dates[s]:
                continue  # 已退市，不再产生数据
            if is_suspended[i]:
                rows.append(
                    {
                        "date": dt,
                        "symbol": s,
                        "open": np.nan, "high": np.nan, "low": np.nan, "close": np.nan,
                        "volume": 0.0, "amount": 0.0, "turnover": 0.0,
                        "pe": pe[i], "pb": pb[i],
                        "is_limit_up": False, "is_limit_down": False,
                        "is_suspended": True, "is_st": bool(st_flags[i]),
                        "adj_factor": 1.0,
                    }
                )
            else:
                rows.append(
                    {
                        "date": dt,
                        "symbol": s,
                        "open": open_p[i], "high": high_p[i], "low": low_p[i], "close": raw_close[i],
                        "volume": volume[i], "amount": amount[i], "turnover": turnover[i],
                        "pe": pe[i], "pb": pb[i],
                        "is_limit_up": bool(is_lu[i]), "is_limit_down": bool(is_ld[i]),
                        "is_suspended": False, "is_st": bool(st_flags[i]),
                        "adj_factor": 1.0,
                    }
                )

        close_prev = np.where(is_suspended, prev_c, raw_close)
        idio_prev = np.where(is_suspended, idio_prev, day_ret - beta * mkt_ret[d])

    prices = pd.DataFrame(rows)
    prices = prices.sort_values(["date", "symbol"]).reset_index(drop=True)

    benchmark = pd.DataFrame({"date": dates, "close": bench_close})
    meta = pd.DataFrame(
        {
            "symbol": symbols,
            "name": [f"SYN_{i}" for i in range(n_stocks)],
            "list_date": [list_dates[s] for s in symbols],
            "delist_date": [delist_dates.get(s, pd.NaT) for s in symbols],
            "industry": rng.choice(industries, size=n_stocks),
        }
    )
    industry = pd.DataFrame(
        {
            "symbol": symbols,
            "industry": meta["industry"].values,
            "as_of_date": dates[0],
        }
    )
    fundamentals = prices[["date", "symbol", "pe", "pb"]].copy()
    fundamentals["roe"] = np.nan
    fundamentals["div_yield"] = np.nan
    fundamentals["as_of_date"] = fundamentals["date"]
    # 用季度公告近似 ROE/股息率（仅演示）
    quarter_starts = pd.date_range(start=dates[0], periods=years * 4 + 1, freq="QS")
    for qs in quarter_starts:
        mask = fundamentals["date"] >= qs
        if not mask.any():
            continue
        idx = fundamentals.index[mask].to_numpy()
        symbols_arr = fundamentals.loc[idx, "symbol"].to_numpy()
        lookup = {s: i for i, s in enumerate(symbols)}
        roe_arr = np.array([0.08 + 0.03 * skill[lookup[s]] + rng.normal(0, 0.02) for s in symbols_arr])
        div_arr = np.array([0.01 + 0.05 * np.clip(0.08 + 0.03 * skill[lookup[s]], 0, 0.25) for s in symbols_arr])
        fundamentals.loc[idx, "roe"] = roe_arr
        fundamentals.loc[idx, "div_yield"] = div_arr

    return DataBundle(
        prices=prices,
        benchmark=benchmark,
        meta=meta,
        industry=industry,
        fundamentals=fundamentals,
    )
