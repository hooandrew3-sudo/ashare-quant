"""回填 2018-2021 退市股价格（消除退市股幸存者偏差），合并进 prices。

依赖：data/cache/delisted_missing.csv（由 baostock stock_basic 缓存推导的退市股缺口清单）。
"""

from __future__ import annotations

import socket
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant.data.baostock_sync import BaoStockSync
from quant.data.storage import Storage


def main() -> None:
    socket.setdefaulttimeout(12)
    missing = pd.read_csv("data/cache/delisted_missing.csv")
    missing["delist_date"] = pd.to_datetime(missing["delist_date"])
    # 只回填 2018 起退市的（2018 前退市的对 2018+ 回测无影响）
    target = missing[missing["delist_date"] >= "2018-01-01"].copy()
    symbols = sorted(target["symbol"].unique())
    print(f"target delisted symbols={len(symbols)}: {symbols}", flush=True)

    storage = Storage("data")
    existing = storage.load("prices")
    with BaoStockSync(cache_dir=Path("data/cache"), retries=2, pause_sec=0.0) as sync:
        calendar = sync.trading_calendar("2018-01-01", "2026-08-12")
        basic = sync.stock_basic()
        raw, stats = sync.fetch_daily(
            symbols, "2018-01-01", "2021-12-31", calendar=calendar, basic=basic
        )
    print(f"fetched rows={len(raw)} errors={len(stats.errors)}", flush=True)
    if stats.errors:
        print(f"errors sample={stats.errors[:10]}", flush=True)

    if raw.empty:
        print("ABORT: no rows fetched", flush=True)
        return
    merged = pd.concat([raw, existing], ignore_index=True)
    merged = merged.drop_duplicates(subset=["date", "symbol"], keep="last")
    merged = merged.sort_values(["symbol", "date"]).reset_index(drop=True)
    print(
        f"merged rows={len(merged)} range={merged['date'].min().date()}..{merged['date'].max().date()}",
        flush=True,
    )
    storage.save("prices", merged)
    storage._manifest["meta"]["delisted_backfilled"] = len(symbols)
    storage.save_manifest()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
