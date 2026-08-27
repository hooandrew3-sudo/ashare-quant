"""验证 consensus 存储和因子安全性。"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"C:\Users\hys18\Documents\ashare")

import pandas as pd
import numpy as np
from quant.data.storage import Storage
from quant.factors.definitions import Panels, _f_consensus_revision

storage = Storage("data")
cons = storage.load("consensus")
print(f"consensus stored: {len(cons)} rows, {cons['symbol'].nunique()} symbols")
print(cons.head(3).to_string())

# 验证因子管线在数据不足（仅1天快照）时返回0不报错
dates = pd.date_range("2026-01-01", periods=60, freq="B")
close = pd.DataFrame(10.0, index=dates, columns=["600519.SH", "000001.SZ"])
c_wide = pd.DataFrame(index=dates, columns=["600519.SH", "000001.SZ"], dtype=float)
c_wide.loc[dates[-1]] = [100.0, 20.0]
panels = Panels(
    close=close, volume=close * 0, amount=close * 1e8, turnover=close * 0,
    pe=close * 10, pb=close, roe=None, gross_margin=None, div_yield=None,
    consensus=c_wide,
)
rev = _f_consensus_revision(panels)
print(f"consensus_revision shape: {rev.shape}, all zero: {(rev == 0).all().all()}")
assert (rev == 0).all().all(), "should be 0 when insufficient data"
print("OK: consensus_revision safe with insufficient data")
