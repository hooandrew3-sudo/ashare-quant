"""仅重建 industry 数据（从 cache/industry.parquet）。"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from quant.data.baostock_sync import bs_to_symbol
from quant.data.storage import Storage

storage = Storage("data")
valid_syms = set(storage.load("prices")["symbol"].unique())
print(f"valid symbols: {len(valid_syms)}")

ind_cache = pd.read_parquet("data/cache/industry.parquet")
ind = ind_cache[["code", "industry"]].copy()
ind.columns = ["symbol", "industry"]
ind["symbol"] = ind["symbol"].map(bs_to_symbol)
ind["as_of_date"] = pd.to_datetime("2026-08-10")
ind = ind.dropna(subset=["industry"])
ind = ind[ind["symbol"].isin(valid_syms)]
print(f"industry rows: {len(ind)}, sectors: {ind['industry'].nunique()}")
print(ind.head(5).to_string())

storage.save("industry", ind, partition_by_symbol=False)
print("industry saved")
