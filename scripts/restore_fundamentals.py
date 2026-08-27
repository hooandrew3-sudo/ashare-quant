"""恢复 fundamentals：akshare 拉取 2021-2026 季度 ROE/股息率/毛利率。"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from quant.data.fundamentals_ak import fetch_fundamentals, report_quarters
from quant.data.storage import Storage


def main() -> None:
    print("=== fundamentals restore start ===", flush=True)
    storage = Storage("data")
    quarters = report_quarters(start_year=2021, end_year=2026)
    df = fetch_fundamentals(quarters=quarters, verbose=True)
    print(f"fetched: {len(df)} rows", flush=True)
    if df.empty:
        print("ABORT: no data fetched", flush=True)
        return
    # 过滤到行情内股票
    valid = set(storage.load("prices")["symbol"].unique())
    df = df[df["symbol"].isin(valid)]
    print(f"filtered to market: {len(df)} rows, {df['symbol'].nunique()} symbols", flush=True)
    storage.save("fundamentals", df, partition_by_symbol=False)
    print("fundamentals saved", flush=True)


if __name__ == "__main__":
    main()