"""真实数据接入：baostock（日线/复权，走生产级同步器）与 CSV 兜底。"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from quant.data.storage import DataBundle

log = logging.getLogger("ashare.sync")


def _fmt_date(d) -> str:
    return pd.Timestamp(d).strftime("%Y-%m-%d")


def load_from_baostock(
    symbols: list[str],
    start: str,
    end: str,
    benchmark: str = "sh.000906",
    incremental: bool = False,
    manifest: dict | None = None,
    storage: "Storage | None" = None,
    universe: str = "manual",
) -> DataBundle:
    """从 baostock 下载日线（前复权），委托生产级同步器。

    symbols 支持 'sh.600519' 或 '600519.SH' 两种格式；
    incremental=True 时基于 manifest 只拉取增量区间并与已有数据合并。
    """
    from quant.data.baostock_sync import BaoStockSync, bs_to_symbol, merge_incremental, symbol_to_bs

    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    norm_symbols = [
        s if "." in s and len(s.split(".")[0]) == 6 else bs_to_symbol(s) for s in symbols
    ]
    cache_dir = storage.root / "cache" if storage is not None else None
    with BaoStockSync(cache_dir=cache_dir) as sync:
        basic = sync.stock_basic()
        industry = sync.industry()
        sync_start = _fmt_date(start)
        inc = incremental
        if inc and storage is not None and manifest is not None:
            from quant.data.baostock_sync import incremental_window

            meta = manifest.get("meta", {})
            if meta.get("universe") != universe or meta.get("source") != "baostock":
                log.warning("manifest 股票池与本次不一致（%s vs %s），改为全量同步",
                            meta.get("universe"), universe)
                inc = False
            else:
                sync_start = incremental_window(manifest, "prices", end)
        if inc:
            log.info("增量同步窗口: %s → %s", sync_start, _fmt_date(end))
        else:
            log.info("全量同步窗口: %s → %s", sync_start, _fmt_date(end))
        calendar = sync.trading_calendar(sync_start, _fmt_date(end))
        raw, _ = sync.fetch_daily(norm_symbols, sync_start, _fmt_date(end), calendar=calendar, basic=basic)
        bench_start = (
            incremental_window(manifest, "benchmark", end)
            if inc and manifest is not None
            else _fmt_date(start)
        )
        bench_raw = sync._query(
            sync._bs.query_history_k_data_plus,
            symbol_to_bs(benchmark),
            "date,close",
            start_date=bench_start,
            end_date=_fmt_date(end),
            frequency="d",
            adjustflag="3",
        )

    if raw.empty or bench_raw is None or bench_raw.empty:
        raise RuntimeError("baostock 未返回任何数据，请检查日期范围与代码格式")
    if inc and storage is not None and storage.has("prices"):
        existing = storage.load("prices")
        raw = merge_incremental(existing, raw)

    benchmark_df = pd.DataFrame(
        {
            "date": pd.to_datetime(bench_raw["date"]),
            "close": pd.to_numeric(bench_raw["close"], errors="coerce"),
        }
    ).dropna()
    if inc and storage is not None and storage.has("benchmark"):
        existing_b = storage.load("benchmark")
        existing_b["date"] = pd.to_datetime(existing_b["date"])
        benchmark_df["date"] = pd.to_datetime(benchmark_df["date"])
        benchmark_df = (
            pd.concat([existing_b, benchmark_df], ignore_index=True)
            .drop_duplicates(subset=["date"], keep="last")
            .sort_values("date")
            .reset_index(drop=True)
        )
    meta = basic[basic["symbol"].isin(norm_symbols)][
        ["symbol", "name", "list_date", "delist_date"]
    ].copy()
    industry = industry[industry["symbol"].isin(norm_symbols)].copy()
    industry["as_of_date"] = pd.to_datetime(industry["as_of_date"], errors="coerce")
    return DataBundle(
        prices=raw,
        benchmark=benchmark_df,
        meta=meta,
        industry=industry,
        fundamentals=pd.DataFrame(),
    )


def load_csv_bundle(root: str | Path, benchmark_file: str = "benchmark.csv") -> DataBundle:
    """从目录读取每只股票一个 CSV（date,open,high,low,close,volume,amount,turnover）。"""
    root = Path(root)
    frames = []
    for f in sorted(root.glob("*.csv")):
        if f.name == benchmark_file:
            continue
        df = pd.read_csv(f, parse_dates=["date"])
        df["symbol"] = f.stem
        frames.append(df)
    prices = pd.concat(frames, ignore_index=True)
    bench = pd.read_csv(root / benchmark_file, parse_dates=["date"])
    return DataBundle(prices=prices, benchmark=bench)
