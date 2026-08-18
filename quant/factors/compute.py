"""因子计算与预处理：长表输出，逐日横截面去极值/标准化/中性化。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.config import Config
from quant.data.storage import DataBundle
from quant.factors.definitions import FACTOR_SPECS, Panels
from quant.utils import setup_logging


def _wide(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """长表 → 宽表（date × symbol）。"""
    return df.pivot(index="date", columns="symbol", values=col)


def build_panels(bundle: DataBundle, cfg: Config) -> Panels:
    """从 DataBundle 构造宽表面板。"""
    prices = _attach_fundamentals(bundle.prices, bundle.fundamentals)
    prices = _attach_sentiment(prices, bundle.sentiment if hasattr(bundle, "sentiment") else None)
    prices = prices.sort_values(["date", "symbol"])
    # 排除仙股/极端离群值，避免污染逐日横截面的 winsorize/zscore/中性化统计量
    if cfg.data.universe.min_price > 0:
        median_close = prices.groupby("symbol")["close"].median()
        keep = set(median_close[median_close >= cfg.data.universe.min_price].index)
        prices = prices[prices["symbol"].isin(keep)]
    industry_map: pd.Series | None = None
    if not bundle.industry.empty:
        ind = bundle.industry.sort_values("as_of_date").drop_duplicates("symbol", keep="last")
        industry_map = ind.set_index("symbol")["industry"]
    elif not bundle.meta.empty and "industry" in bundle.meta.columns:
        industry_map = bundle.meta.set_index("symbol")["industry"]

    sentiment = None
    if "sentiment" in prices.columns:
        sentiment = _wide(prices, "sentiment")

    # 分析师一致预期：快照 long 表 → date × symbol（最近年度 EPS 均值）
    consensus = None
    if hasattr(bundle, "consensus") and not bundle.consensus.empty:
        c = bundle.consensus.copy()
        c["as_of_date"] = pd.to_datetime(c["as_of_date"], errors="coerce")
        c = c.dropna(subset=["as_of_date"])
        if c.empty:
            consensus = None
        else:
            # 仅保留最近年度、最近快照的一致预期（快照表列为 as_of_date，透视前统一为 date）
            c = (
                c.sort_values(["symbol", "as_of_date", "year"])
                .drop_duplicates("symbol", keep="last")
                .rename(columns={"as_of_date": "date"})
            )
            consensus = _wide(c, "eps_mean") if "eps_mean" in c.columns else None

    return Panels(
        close=_wide(prices, "close"),
        volume=_wide(prices, "volume"),
        amount=_wide(prices, "amount") if "amount" in prices.columns else pd.DataFrame(),
        turnover=_wide(prices, "turnover") if "turnover" in prices.columns else pd.DataFrame(),
        pe=_wide(prices, "pe") if "pe" in prices.columns else pd.DataFrame(),
        pb=_wide(prices, "pb") if "pb" in prices.columns else pd.DataFrame(),
        roe=_wide(prices, "roe") if "roe" in prices.columns else pd.DataFrame(),
        gross_margin=_wide(prices, "gross_margin") if "gross_margin" in prices.columns else pd.DataFrame(),
        div_yield=_wide(prices, "div_yield") if "div_yield" in prices.columns else pd.DataFrame(),
        sentiment=sentiment,
        forecast=_wide(prices, "forecast_growth") if "forecast_growth" in prices.columns else pd.DataFrame(),
        industry=industry_map,
        consensus=consensus,
    )


def _attach_sentiment(prices: pd.DataFrame, sentiment: pd.DataFrame | None) -> pd.DataFrame:
    """按 (date, symbol) 精确接入公告情绪；无公告日取 0（中性）。"""
    if sentiment is None or sentiment.empty:
        prices = prices.copy()
        prices["sentiment"] = 0.0
        return prices
    s = sentiment.copy()
    s["date"] = pd.to_datetime(s["date"]).astype("datetime64[ns]")
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"]).astype("datetime64[ns]")
    merged = prices.merge(s, on=["date", "symbol"], how="left")
    merged["sentiment"] = merged["sentiment"].fillna(0.0)
    return merged


def _attach_forecast(prices: pd.DataFrame, forecast: pd.DataFrame | None) -> pd.DataFrame:
    """业绩预告净利增速按公告日 merge_asof 接入日线（杜绝前视）。无预告取 0（中性）。"""
    if forecast is None or forecast.empty:
        prices = prices.copy()
        prices["forecast_growth"] = 0.0
        return prices
    f = forecast.copy()
    f["as_of_date"] = pd.to_datetime(f["as_of_date"]).astype("datetime64[ns]")
    prices = prices.sort_values(["symbol", "date"]).copy()
    prices["date"] = pd.to_datetime(prices["date"]).astype("datetime64[ns]")
    parts: list[pd.DataFrame] = []
    for sym, g in prices.groupby("symbol", sort=False):
        fg = f[f["symbol"] == sym].sort_values("as_of_date")
        if fg.empty:
            g = g.copy()
            g["forecast_growth"] = 0.0
            parts.append(g)
            continue
        m = pd.merge_asof(
            g,
            fg[["as_of_date", "forecast_growth"]],
            left_on="date",
            right_on="as_of_date",
            direction="backward",
        )
        m = m.drop(columns=["as_of_date"])
        m["forecast_growth"] = m["forecast_growth"].fillna(0.0)
        parts.append(m)
    return pd.concat(parts, ignore_index=True)


def _attach_fundamentals(prices: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    """将财务因子（ROE/股息率）按公告日 merge_asof 接入日线（杜绝前视）。

    无数据的股票取 0（中性占位）。
    """
    if fundamentals is None or fundamentals.empty:
        prices = prices.copy()
        prices["roe"] = 0.0
        prices["gross_margin"] = 0.0
        prices["div_yield"] = 0.0
        return prices
    f = fundamentals.copy()
    if "as_of_date" not in f.columns:
        if "date" in f.columns:
            f["as_of_date"] = pd.to_datetime(f["date"])
        else:
            prices = prices.copy()
            prices["roe"] = 0.0
            prices["gross_margin"] = 0.0
            prices["div_yield"] = 0.0
            return prices
    f["as_of_date"] = pd.to_datetime(f["as_of_date"]).astype("datetime64[ns]")
    prices = prices.sort_values(["symbol", "date"]).copy()
    prices["date"] = pd.to_datetime(prices["date"]).astype("datetime64[ns]")
    parts: list[pd.DataFrame] = []
    for sym, g in prices.groupby("symbol", sort=False):
        fg = f[f["symbol"] == sym].sort_values("as_of_date")
        if fg.empty:
            g = g.copy()
            g["roe"] = 0.0
            g["div_yield"] = 0.0
            parts.append(g)
            continue
        fg_cols = [c for c in ["as_of_date", "roe", "gross_margin", "div_yield"] if c in fg.columns]
        m = pd.merge_asof(
            g,
            fg[fg_cols],
            left_on="date",
            right_on="as_of_date",
            direction="backward",
        )
        m = m.drop(columns=["as_of_date"])
        for col in ("roe", "gross_margin", "div_yield"):
            if col not in m.columns:
                m[col] = 0.0
            else:
                m[col] = m[col].fillna(0.0)
        parts.append(m)
    return pd.concat(parts, ignore_index=True)


def _winsorize_zscore(df: pd.DataFrame, winsor: float) -> pd.DataFrame:
    """逐日横截面去极值 + z-score。"""
    lo, hi = df.quantile(winsor, axis=1), df.quantile(1 - winsor, axis=1)
    out = df.clip(lower=lo, upper=hi, axis=0)
    mean = out.mean(axis=1)
    std = out.std(axis=1).replace(0, np.nan)
    out = out.sub(mean, axis=0).div(std, axis=0)
    # 缺失值以横截面中位数填充（0 为 z-score 后的中性值）
    return out.fillna(0.0)


def _neutralize(df: pd.DataFrame, industry_map: pd.Series, size_panel: pd.DataFrame | None) -> pd.DataFrame:
    """行业 + 规模中性化：对 [行业哑变量, log(成交额)] 回归取残差。"""
    symbols = df.columns
    dummies = None
    if industry_map is not None and not industry_map.empty:
        inds = industry_map.reindex(symbols).dropna()
        if len(inds) >= 5 and inds.nunique() >= 2:
            dummies = pd.get_dummies(inds, prefix="ind").astype(float)
    if dummies is None and (size_panel is None or size_panel.empty):
        return df
    out = df.copy()
    for dt, row in out.iterrows():
        y = row.values.astype(float)
        parts: list[pd.DataFrame] = []
        if dummies is not None:
            parts.append(dummies)
        if size_panel is not None and dt in size_panel.index:
            size_row = size_panel.loc[dt].reindex(symbols).to_frame("size")
            parts.append(size_row)
        if not parts:
            continue
        X = pd.concat(parts, axis=1).fillna(0.0).to_numpy()
        mask = ~np.isnan(y)
        if mask.sum() < 10:
            continue
        beta, *_ = np.linalg.lstsq(X[mask], y[mask], rcond=None)
        out.loc[dt] = y - X @ beta
    return out


def compute_all_factors(
    bundle: DataBundle,
    cfg: Config,
    factors: list[str] | None = None,
) -> pd.DataFrame:
    """计算全部因子并做预处理，返回长表：date, symbol, factor, value。"""
    log = setup_logging(cfg.run.verbose)
    panels = build_panels(bundle, cfg)
    names = factors or list(FACTOR_SPECS.keys())
    frames: list[pd.DataFrame] = []
    for name in names:
        spec = FACTOR_SPECS[name]
        raw = spec["compute"](panels)
        clean = _winsorize_zscore(raw, cfg.factors.winsor)
        if cfg.factors.neutralize_industry or cfg.factors.neutralize_size:
            size_panel = None
            if cfg.factors.neutralize_size and not panels.amount.empty:
                size_panel = np.log(panels.amount.rolling(60, min_periods=20).mean() + 1.0)
            clean = _neutralize(clean, panels.industry, size_panel)
        long = clean.stack().rename("value").reset_index()
        long["factor"] = name
        frames.append(long[["date", "symbol", "factor", "value"]])
        log.info("factor %s computed: %d non-null", name, int(long["value"].notna().sum()))
    return pd.concat(frames, ignore_index=True)
