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
    prices = _attach_cashflow(prices, getattr(bundle, "cashflow", None))
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
            # 仅保留最近年度的一致预期。去重键必须是 (symbol, as_of_date)：
            # 此前按 symbol 全历史保留最后一条快照，面板在其最终快照日前
            # 全 NaN、之后恒定 → revision 因子静默退化为 0/NaN（死因子）
            c = (
                c.sort_values(["symbol", "as_of_date", "year"])
                .drop_duplicates(["symbol", "as_of_date"], keep="last")
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
        eps=_wide(prices, "eps") if "eps" in prices.columns else pd.DataFrame(),
        ocfps=_wide(prices, "ocfps") if "ocfps" in prices.columns else pd.DataFrame(),
        ocf_ytd=_wide(prices, "ocf_ytd") if "ocf_ytd" in prices.columns else pd.DataFrame(),
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
                # 先按时间前向填充再补 0：财务表各列披露节奏不同（如股息率
                # 只在年报行），若直接 fillna(0)，最新一行缺失该列会把上一
                # 季报的有效值打穿成 0，严重压低质量/红利类因子覆盖度
                m[col] = m[col].ffill().fillna(0.0)
        parts.append(m)
    return pd.concat(parts, ignore_index=True)


def _attach_cashflow(prices: pd.DataFrame, cashflow: pd.DataFrame | None) -> pd.DataFrame:
    """现金流数据按公告日 merge_asof 接入日线（点内时间，杜绝前视）。

    无数据的股票取 0（中性占位）；同 _attach_fundamentals 的 ffill 语义。
    """
    cols = ("eps", "ocfps", "ocf_ytd")
    if cashflow is None or cashflow.empty or "ocf_ytd" not in cashflow.columns:
        prices = prices.copy()
        for c in cols:
            prices[c] = 0.0
        return prices
    f = cashflow.copy()
    f["as_of_date"] = pd.to_datetime(f["as_of_date"]).astype("datetime64[ns]")
    prices = prices.sort_values(["symbol", "date"]).copy()
    prices["date"] = pd.to_datetime(prices["date"]).astype("datetime64[ns]")
    parts: list[pd.DataFrame] = []
    # 注意：右表不能带 symbol 列——与左表同名会在 merge_asof 中触发
    # _x/_y 后缀，导致左表 symbol 被顶替成 NaN、面板坍缩
    fg_cols = ["as_of_date"] + [c for c in cols if c in f.columns]
    for sym, g in prices.groupby("symbol", sort=False):
        fg = f[f["symbol"] == sym].sort_values("as_of_date")
        if fg.empty:
            g = g.copy()
            for c in cols:
                g[c] = 0.0
            parts.append(g)
            continue
        m = pd.merge_asof(
            g,
            fg[fg_cols],
            left_on="date",
            right_on="as_of_date",
            direction="backward",
        )
        m = m.drop(columns=["as_of_date"])
        for c in cols:
            if c not in m.columns:
                m[c] = 0.0
            else:
                m[c] = m[c].ffill().fillna(0.0)
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
    """行业 + 规模中性化（单因子兼容入口，语义与批量路径一致）。"""
    out = df.copy()
    if industry_map is not None and not industry_map.empty:
        inds = industry_map.reindex(df.columns).dropna()
        if len(inds) >= 5 and inds.nunique() >= 2:
            dummies = pd.get_dummies(inds, prefix="ind").astype(float)
            # 哑变量对齐到完整股票池：无行业映射的股票为全 0 行，
            # 保证 X 行数与横截面一致（旧实现依赖 size 面板隐式补齐）
            dummies = dummies.reindex(index=df.columns).fillna(0.0)
        else:
            dummies = None
    else:
        dummies = None
    if dummies is None and (size_panel is None or size_panel.empty):
        return out
    X_by_date = _build_X_by_date(out.index, out.columns, dummies, size_panel)
    return _neutralize_batched([out], X_by_date)[0]


def _build_X_by_date(
    dates, symbols, dummies: pd.DataFrame | None, size_panel: pd.DataFrame | None
) -> dict[pd.Timestamp, np.ndarray]:
    """预计算每日回归设计矩阵 [行业哑变量, log(成交额)]，跨因子复用。"""
    out: dict[pd.Timestamp, np.ndarray] = {}
    for dt in dates:
        parts: list[pd.DataFrame] = []
        if dummies is not None:
            parts.append(dummies)
        if size_panel is not None and dt in size_panel.index:
            parts.append(size_panel.loc[dt].reindex(symbols).to_frame("size"))
        if not parts:
            continue
        out[dt] = pd.concat(parts, axis=1).fillna(0.0).to_numpy()
    return out


def _neutralize_batched(
    frames: list[pd.DataFrame], X_by_date: dict[pd.Timestamp, np.ndarray]
) -> list[pd.DataFrame]:
    """批量中性化：单次 lstsq 同时求解全部因子的残差（多目标 RHS）。

    输入须为去极值/z-score 后的稠密面板（winsorize 已 fillna(0)），
    与旧实现（逐因子逐日单目标 lstsq）在稠密输入下数值等价，
    但 lstsq 调用次数从 n_factors × n_dates 降为 n_dates。
    """
    ref = frames[0]
    idx, cols = ref.index, ref.columns
    for f in frames[1:]:
        if not (f.index.equals(idx) and f.columns.equals(cols)):
            raise ValueError("批量中性化要求各因子面板索引/列完全一致")
    Y = np.stack([f.values for f in frames], axis=-1)  # dates × symbols × n_factors
    out = np.zeros_like(Y, dtype=float)
    for i, dt in enumerate(idx):
        X = X_by_date.get(dt)
        y = Y[i]
        if X is None:
            out[i] = y
            continue
        finite = np.isfinite(y).all(axis=1)
        if finite.sum() < 10:
            out[i] = y
            continue
        beta, *_ = np.linalg.lstsq(X[finite], y[finite], rcond=None)
        resid = y - X @ beta
        resid[~finite] = np.nan  # 防御：原 NaN 保持 NaN
        out[i] = resid
    return [pd.DataFrame(out[:, :, j], index=idx, columns=cols) for j in range(len(frames))]


def compute_all_factors(
    bundle: DataBundle,
    cfg: Config,
    factors: list[str] | None = None,
    report_coverage: bool = False,
) -> pd.DataFrame:
    """计算全部因子并做预处理，返回长表：date, symbol, factor, value。

    report_coverage=True 时返回 (factor_long, coverage_long)：
    coverage_long = date, factor, coverage_ratio, non_null, n_symbols（基于原始
    因子值，供因子健康度哨兵观测数据质量）。
    """
    log = setup_logging(cfg.run.verbose)
    panels = build_panels(bundle, cfg)
    names = factors or list(FACTOR_SPECS.keys())
    raw_by_name = {
        name: FACTOR_SPECS[name]["compute"](panels) for name in names
    }
    clean_by_name = {
        name: _winsorize_zscore(raw, cfg.factors.winsor)
        for name, raw in raw_by_name.items()
    }
    if cfg.factors.neutralize_industry or cfg.factors.neutralize_size:
        X_by_date = _build_factor_regressors(panels, cfg)
        if X_by_date:
            cleaned = _neutralize_batched(
                [clean_by_name[n] for n in names], X_by_date
            )
            clean_by_name = dict(zip(names, cleaned))
    frames: list[pd.DataFrame] = []
    for name in names:
        long = clean_by_name[name].stack().rename("value").reset_index()
        long["factor"] = name
        frames.append(long[["date", "symbol", "factor", "value"]])
        log.info("factor %s computed: %d non-null", name, int(long["value"].notna().sum()))
    out = pd.concat(frames, ignore_index=True)
    if report_coverage:
        cov_rows: list[dict] = []
        for name in names:
            raw = raw_by_name[name]
            for dt, row in raw.iterrows():
                cov_rows.append(
                    {
                        "date": dt,
                        "factor": name,
                        "non_null": int(row.notna().sum()),
                        "n_symbols": int(len(row)),
                    }
                )
        cov = pd.DataFrame(cov_rows)
        cov["coverage_ratio"] = cov["non_null"] / cov["n_symbols"].clip(lower=1)
        return out, cov[["date", "factor", "coverage_ratio", "non_null", "n_symbols"]]
    return out


def _build_factor_regressors(
    panels: Panels, cfg: Config
) -> dict[pd.Timestamp, np.ndarray]:
    """预计算每日中性化回归设计矩阵（行业哑变量 + log(成交额)），跨因子复用。"""
    symbols = panels.close.columns
    dummies = None
    if panels.industry is not None and not panels.industry.empty:
        inds = panels.industry.reindex(symbols).dropna()
        if len(inds) >= 5 and inds.nunique() >= 2:
            dummies = pd.get_dummies(inds, prefix="ind").astype(float)
            dummies = dummies.reindex(index=symbols).fillna(0.0)
    size_panel = None
    if cfg.factors.neutralize_size and not panels.amount.empty:
        size_panel = np.log(panels.amount.rolling(60, min_periods=20).mean() + 1.0)
    if dummies is None and (size_panel is None or size_panel.empty):
        return {}
    return _build_X_by_date(panels.close.index, symbols, dummies, size_panel)
