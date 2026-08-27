"""数据恢复：修复 synthetic 覆盖导致的真实数据污染。

恢复对象：
1. benchmark（000906.SH，2018-2026，不复权）
2. 20 只被 synthetic 覆盖的股票日线（600000-600009.SH + 000001-000010.SZ，前复权）
3. industry（从 cache/industry.parquet 重建）
4. meta（从 cache/stock_basic.parquet 重建）
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from quant.data.baostock_sync import BaoStockSync, symbol_to_bs
from quant.data.storage import Storage
from quant.utils import ensure_dir


POLLUTED_SYMS = [f"{600000 + i:06d}.SH" for i in range(10)] + [
    f"{1 + i:06d}.SZ" for i in range(10)
]


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> None:
    storage = Storage("data")
    start, end = "2018-01-01", "2026-08-17"

    with BaoStockSync(cache_dir=Path("data/cache"), retries=2, pause_sec=0.0) as sync:
        calendar = sync.trading_calendar(start, end)
        basic = sync.stock_basic()
        log(f"calendar: {calendar.min().date()}..{calendar.max().date()} ({len(calendar)} days)")

        # 1. 重新同步 20 只被污染股票
        raw, stats = sync.fetch_daily(POLLUTED_SYMS, start, end, calendar=calendar, basic=basic)
        log(f"polluted symbols refetched: rows={len(raw)} errors={len(stats.errors)}")
        if stats.errors:
            log(f"errors: {stats.errors[:5]}")

        # 2. 重新同步 benchmark（指数不复权）
        bench_raw = sync._query(
            sync._bs.query_history_k_data_plus,
            symbol_to_bs("000906.SH"),
            "date,close",
            start_date=start,
            end_date=end,
            frequency="d",
            adjustflag="3",
        )
        benchmark = pd.DataFrame(
            {
                "date": pd.to_datetime(bench_raw["date"]),
                "close": pd.to_numeric(bench_raw["close"], errors="coerce"),
            }
        ).dropna()
        log(f"benchmark refetched: rows={len(benchmark)}, {benchmark['date'].min().date()}..{benchmark['date'].max().date()}")

    # 3. 合并到现有 prices（覆盖被污染的 20 只）
    if not raw.empty:
        existing = storage.load("prices")
        existing = existing[~existing["symbol"].isin(POLLUTED_SYMS)]
        merged = pd.concat([raw, existing], ignore_index=True)
        merged = merged.drop_duplicates(subset=["date", "symbol"], keep="last")
        merged = merged.sort_values(["symbol", "date"]).reset_index(drop=True)
        storage.save("prices", merged)
        log(f"prices merged: rows={len(merged)}, symbols={merged['symbol'].nunique()}")

    # 4. benchmark 写入
    storage.save("benchmark", benchmark, partition_by_symbol=False)
    log("benchmark saved")

    # 5. 重建 industry（从 cache）
    from quant.data.baostock_sync import bs_to_symbol

    ind_cache = pd.read_parquet("data/cache/industry.parquet")
    ind = ind_cache[["code", "industry"]].copy()
    ind.columns = ["symbol", "industry"]
    # code 格式 'sh.600000' → '600000.SH'
    ind["symbol"] = ind["symbol"].map(bs_to_symbol)
    ind["as_of_date"] = pd.to_datetime("2026-08-10")
    ind = ind.dropna(subset=["industry"])
    # 只保留行情内股票
    valid_syms = set(merged["symbol"].unique())
    ind = ind[ind["symbol"].isin(valid_syms)]
    storage.save("industry", ind, partition_by_symbol=False)
    log(f"industry rebuilt: rows={len(ind)}, sectors={ind['industry'].nunique()}")

    # 6. 重建 meta
    if not basic.empty:
        meta = basic[basic["symbol"].isin(valid_syms)][
            ["symbol", "name", "list_date", "delist_date"]
        ].copy()
        storage.save("meta", meta, partition_by_symbol=False)
        log(f"meta rebuilt: rows={len(meta)}")

    # 7. 修正 manifest meta
    storage._manifest["meta"] = {
        "universe": "csi800",
        "source": "baostock",
        "start": start,
        "end": end,
    }
    storage.save_manifest()
    log("DONE")


if __name__ == "__main__":
    main()