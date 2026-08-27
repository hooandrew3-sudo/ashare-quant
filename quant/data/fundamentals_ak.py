"""财务因子数据接入（akshare 批量季度接口）。

- ROE：stock_yjbb_em（业绩报表，按报告期拉全市场，含公告日期）
- 股息率：stock_fhps_em（分红送配，按报告期拉全市场，含公告日期）
- 现金流：stock_xjll_em（现金流量表，按报告期拉全市场）+ yjbb 的
  每股收益/每股经营现金流量（构建盈利含金量因子）
- 事件时点：以「最新公告日期」为可用时点，杜绝前视
- 输出：long (symbol, as_of_date, roe, div_yield) / (symbol, as_of_date, ocf_ytd, ...)
"""

from __future__ import annotations

import time
from datetime import date

import pandas as pd

from quant.utils import setup_logging


def normalize_code(code) -> str | None:
    """6 位代码 → '600519.SH'；仅保留沪深 A 股。"""
    s = str(code).strip()
    if len(s) != 6 or not s.isdigit():
        return None
    if s[0] in ("6", "9"):
        return f"{s}.SH"
    if s[0] in ("0", "3"):
        return f"{s}.SZ"
    return None  # 北交所等暂不支持


def report_quarters(start_year: int = 2020, end_year: int = 2026) -> list[str]:
    """报告期列表：YYYY-12-31, YYYY-03-31, ... 至 end_year 一季报。"""
    out: list[str] = []
    for y in range(start_year, end_year + 1):
        for md in ("0331", "0630", "0930", "1231"):
            q = f"{y}{md}"
            if q > f"{end_year}0331":
                break
            out.append(q)
    return out


def _pick_col(df: pd.DataFrame, *substrings: str) -> str | None:
    """按子串模糊匹配列名（akshare 列名可能随版本漂移）。"""
    for col in df.columns:
        s = str(col)
        if any(sub in s for sub in substrings):
            return col
    return None


def _fetch_yjbb(report_date: str, log) -> pd.DataFrame:
    import akshare as ak

    df = ak.stock_yjbb_em(date=report_date)
    if df is None or df.empty:
        return pd.DataFrame(columns=["symbol", "as_of_date", "roe"])
    code_col = _pick_col(df, "代码")
    roe_col = _pick_col(df, "净资产收益率")
    gm_col = _pick_col(df, "销售毛利率")
    ann_col = _pick_col(df, "公告日期")
    if not (code_col and roe_col and ann_col):
        log.warning("季度 %s yjbb 列名无法识别: %s", report_date, df.columns.tolist()[:6])
        return pd.DataFrame(columns=["symbol", "as_of_date", "roe", "gross_margin"])
    df = df.rename(
        columns={
            code_col: "code",
            roe_col: "roe_pct",
            ann_col: "ann_date",
            **({gm_col: "gm_pct"} if gm_col else {}),
        }
    )
    df["symbol"] = df["code"].map(normalize_code)
    df = df.dropna(subset=["symbol"])
    out = pd.DataFrame(
        {
            "symbol": df["symbol"],
            "as_of_date": pd.to_datetime(df["ann_date"], errors="coerce"),
            "roe": pd.to_numeric(df["roe_pct"], errors="coerce") / 100.0,
            "gross_margin": (
                pd.to_numeric(df["gm_pct"], errors="coerce") / 100.0
                if "gm_pct" in df.columns
                else pd.NA
            ),
        }
    )
    return out.dropna(subset=["as_of_date"])


def _fetch_fhps(report_date: str, log) -> pd.DataFrame:
    import akshare as ak

    df = ak.stock_fhps_em(date=report_date)
    if df is None or df.empty:
        return pd.DataFrame(columns=["symbol", "as_of_date", "div_yield"])
    code_col = _pick_col(df, "代码")
    div_col = _pick_col(df, "股息率")
    ann_col = _pick_col(df, "公告日期")
    share_col = _pick_col(df, "总股本")
    if not (code_col and div_col and ann_col):
        log.warning("季度 %s fhps 列名无法识别: %s", report_date, df.columns.tolist()[:6])
        return pd.DataFrame(columns=["symbol", "as_of_date", "div_yield"])
    df = df.rename(
        columns={
            code_col: "code",
            div_col: "div_pct",
            ann_col: "ann_date",
            **({share_col: "total_share"} if share_col else {}),
        }
    )
    df["symbol"] = df["code"].map(normalize_code)
    df = df.dropna(subset=["symbol"])
    out = pd.DataFrame(
        {
            "symbol": df["symbol"],
            "as_of_date": pd.to_datetime(df["ann_date"], errors="coerce"),
            # 东方财富「股息率」字段已是小数形式（0.0158 = 1.58%），无需除以 100
            "div_yield": pd.to_numeric(df["div_pct"], errors="coerce"),
            "total_share": (
                pd.to_numeric(df["total_share"], errors="coerce")
                if "total_share" in df.columns
                else pd.NA
            ),
        }
    )
    return out.dropna(subset=["as_of_date"])


def _fetch_yjbb_full(report_date: str, log) -> pd.DataFrame:
    """业绩报表全字段：ROE/毛利率/每股收益/每股经营现金流/净利润。"""
    import akshare as ak

    df = ak.stock_yjbb_em(date=report_date)
    if df is None or df.empty:
        return pd.DataFrame()
    code_col = _pick_col(df, "代码")
    roe_col = _pick_col(df, "净资产收益率")
    gm_col = _pick_col(df, "销售毛利率")
    ann_col = _pick_col(df, "公告日期")
    eps_col = _pick_col(df, "每股收益")
    ocfps_col = _pick_col(df, "每股经营现金流量")
    np_col = _pick_col(df, "净利润-净利润") or _pick_col(df, "净利润")
    if not (code_col and roe_col and ann_col):
        log.warning("季度 %s yjbb 列名无法识别: %s", report_date, df.columns.tolist()[:6])
        return pd.DataFrame()
    out = pd.DataFrame(
        {
            "symbol": df[code_col].map(normalize_code),
            "as_of_date": pd.to_datetime(df[ann_col], errors="coerce"),
            "roe": pd.to_numeric(df[roe_col], errors="coerce") / 100.0,
            "gross_margin": (
                pd.to_numeric(df[gm_col], errors="coerce") / 100.0 if gm_col else pd.NA
            ),
            "eps": pd.to_numeric(df[eps_col], errors="coerce") if eps_col else pd.NA,
            "ocfps": (
                pd.to_numeric(df[ocfps_col], errors="coerce") if ocfps_col else pd.NA
            ),
            "net_profit": (
                pd.to_numeric(df[np_col], errors="coerce") if np_col else pd.NA
            ),
        }
    )
    return out.dropna(subset=["symbol", "as_of_date"])


def _fetch_xjll(report_date: str, log) -> pd.DataFrame:
    """现金流量表（东财按报告期汇总）：经营活动现金流净额（YTD 累计，元）。"""
    import akshare as ak

    df = ak.stock_xjll_em(date=report_date)
    if df is None or df.empty:
        return pd.DataFrame(columns=["symbol", "as_of_date", "ocf_ytd"])
    code_col = _pick_col(df, "代码")
    ann_col = _pick_col(df, "公告日期")
    ocf_col = _pick_col(df, "经营性现金流-现金流量净额", "经营现金流-现金流量净额")
    if not (code_col and ann_col and ocf_col):
        log.warning("季度 %s xjll 列名无法识别: %s", report_date, df.columns.tolist()[:6])
        return pd.DataFrame(columns=["symbol", "as_of_date", "ocf_ytd"])
    out = pd.DataFrame(
        {
            "symbol": df[code_col].map(normalize_code),
            "as_of_date": pd.to_datetime(df[ann_col], errors="coerce"),
            "ocf_ytd": pd.to_numeric(df[ocf_col], errors="coerce"),
        }
    )
    return out.dropna(subset=["symbol", "as_of_date"])


def fetch_cashflow(
    quarters: list[str] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """拉取全部季度现金流数据，返回 long 表。

    输出列：symbol, as_of_date(公告日), report_period(报告期), ocf_ytd(元),
    eps(每股收益), ocfps(每股经营现金流量), net_profit(元)。
    用于盈利含金量（现金流/盈利覆盖）与 OCF 同比增长因子。
    """
    log = setup_logging(verbose)
    quarters = quarters or report_quarters(start_year=2021)
    parts: list[pd.DataFrame] = []
    for q in quarters:
        t0 = time.time()
        period = f"{q[:4]}-{q[4:6]}-{q[6:]}"
        try:
            x = _fetch_xjll(q, log)
            y = _fetch_yjbb_full(q, log)
            if x.empty and y.empty:
                continue
            m = pd.merge(
                x, y, on=["symbol", "as_of_date"], how="outer", suffixes=("", "_y")
            )
            # 两表公告日期可能差 1~2 天：以 symbol+period 对齐后取各自最新公告
            m["report_period"] = period
            parts.append(m)
            log.info(
                "季度 %s 完成: ocf=%d eps=%d (%.1fs)",
                q, len(x), len(y), time.time() - t0,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("季度 %s 拉取失败: %s", q, exc)
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["symbol", "as_of_date", "report_period", "ocf_ytd", "eps", "ocfps", "net_profit"]
    )
    out = out.dropna(subset=["symbol", "as_of_date"])
    out = out.sort_values(["symbol", "report_period", "as_of_date"]).drop_duplicates(
        subset=["symbol", "report_period"], keep="last"
    )
    return out.reset_index(drop=True)


def fetch_fundamentals(
    quarters: list[str] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """拉取全部季度 ROE 与股息率，返回 long 表。"""
    log = setup_logging(verbose)
    quarters = quarters or report_quarters(start_year=2021)
    parts: list[pd.DataFrame] = []
    for q in quarters:
        t0 = time.time()
        try:
            r = _fetch_yjbb(q, log)
            f = _fetch_fhps(q, log)
            if not r.empty or not f.empty:
                merged = pd.concat(
                    [
                        r[["symbol", "as_of_date", "roe", "gross_margin"]],
                        f[["symbol", "as_of_date", "div_yield", "total_share"]],
                    ],
                    ignore_index=True,
                )
                parts.append(merged)
            log.info("季度 %s 完成: roe=%d div=%d (%.1fs)", q, len(r), len(f), time.time() - t0)
        except Exception as exc:  # noqa: BLE001
            log.warning("季度 %s 拉取失败: %s", q, exc)
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["symbol", "as_of_date", "roe", "gross_margin", "div_yield", "total_share"]
    )
    out = out.dropna(subset=["symbol", "as_of_date"])
    # 同 (symbol, as_of_date) 合并
    out = out.groupby(["symbol", "as_of_date"], as_index=False).agg(
        {
            "roe": "first",
            "gross_margin": "first",
            "div_yield": "first",
            "total_share": "first",
        }
    )
    return out.sort_values(["symbol", "as_of_date"]).reset_index(drop=True)
