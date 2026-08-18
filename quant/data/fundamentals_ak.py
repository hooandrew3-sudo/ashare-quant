"""财务因子数据接入（akshare 批量季度接口）。

- ROE：stock_yjbb_em（业绩报表，按报告期拉全市场，含公告日期）
- 股息率：stock_fhps_em（分红送配，按报告期拉全市场，含公告日期）
- 事件时点：以「最新公告日期」为可用时点，杜绝前视
- 输出：long (symbol, as_of_date, roe, div_yield)
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
