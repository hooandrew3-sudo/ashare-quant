"""一次性回填 2018-2020 日线（复用现有 1007 只股票），并同步基准指数 2018-2026。

后台运行，实时写进度到 stdout（调用方重定向到日志文件）。
"""

from __future__ import annotations

import socket
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant.data.baostock_sync import BaoStockSync, symbol_to_bs
from quant.data.storage import Storage


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> None:
    # 缩短底层 socket 超时，避免个别股票挂起拖慢整批；失败快速重试。
    socket.setdefaulttimeout(12)
    storage = Storage("data")
    existing = storage.load("prices")
    symbols = sorted(existing["symbol"].unique())
    log(f"START symbols={len(symbols)} range={existing['date'].min().date()}..{existing['date'].max().date()}")

    with BaoStockSync(cache_dir=Path("data/cache"), retries=2, pause_sec=0.0) as sync:
        calendar = sync.trading_calendar("2018-01-01", "2020-12-31")
        basic = sync.stock_basic()
        log(f"calendar={calendar.min().date()}..{calendar.max().date()} ({len(calendar)} days)")
        raw, stats = sync.fetch_daily(
            symbols, "2018-01-01", "2020-12-31", calendar=calendar, basic=basic
        )
        log(f"FETCHED rows={len(raw)} errors={len(stats.errors)}")
        if stats.errors:
            log(f"error sample={stats.errors[:10]}")

        # 基准指数 2018-2026（不复权）
        bench_raw = sync._query(
            sync._bs.query_history_k_data_plus,
            symbol_to_bs("000906.SH"),
            "date,close",
            start_date="2018-01-01",
            end_date=pd.Timestamp.today().strftime("%Y-%m-%d"),
            frequency="d",
            adjustflag="3",
        )
        benchmark = pd.DataFrame(
            {
                "date": pd.to_datetime(bench_raw["date"]),
                "close": pd.to_numeric(bench_raw["close"], errors="coerce"),
            }
        ).dropna()
        log(f"benchmark rows={len(benchmark)}")

    if raw.empty:
        log("ABORT: no rows fetched")
        return

    merged = pd.concat([raw, existing], ignore_index=True)
    merged = merged.drop_duplicates(subset=["date", "symbol"], keep="last")
    merged = merged.sort_values(["symbol", "date"]).reset_index(drop=True)
    log(f"MERGED rows={len(merged)} range={merged['date'].min().date()}..{merged['date'].max().date()}")

    storage.save("prices", merged)
    storage.save("benchmark", benchmark, partition_by_symbol=False)
    storage._manifest["meta"]["start"] = "2018-01-01"
    storage.save_manifest()
    log("DONE")


if __name__ == "__main__":
    main()
