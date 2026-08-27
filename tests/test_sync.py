"""真实数据同步器与告警模块测试（纯函数部分，不依赖网络）。"""

from __future__ import annotations

import pandas as pd

from quant.data.baostock_sync import (
    bs_to_symbol,
    incremental_window,
    merge_incremental,
    symbol_to_bs,
)


def test_symbol_conversion_roundtrip():
    assert bs_to_symbol("sh.600519") == "600519.SH"
    assert bs_to_symbol("sz.000001") == "000001.SZ"
    assert symbol_to_bs("600519.SH") == "sh.600519"
    assert symbol_to_bs("000001.SZ") == "sz.000001"


def test_merge_incremental_dedup():
    existing = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "symbol": ["600519.SH", "600519.SH"],
            "close": [100.0, 101.0],
        }
    )
    new = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-03", "2024-01-04"]),
            "symbol": ["600519.SH", "600519.SH"],
            "close": [101.5, 102.0],
        }
    )
    merged = merge_incremental(existing, new)
    assert len(merged) == 3
    # 重叠日期以新数据为准
    overlap = merged[merged["date"] == pd.Timestamp("2024-01-03")]
    assert overlap.iloc[0]["close"] == 101.5


def test_merge_normalizes_flag_columns():
    """旧数据缺新列（is_limit_up_open）时，合并后必须规范化为 bool 而非 float/NaN。"""
    existing = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"]),
            "symbol": ["600519.SH"],
            "close": [100.0],
            "is_limit_up": [False],
        }
    )
    new = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "symbol": ["600519.SH", "600519.SH"],
            "close": [100.5, 101.0],
            "is_limit_up": [True, False],
            "is_limit_up_open": [True, False],
            "is_limit_down_open": [False, False],
        }
    )
    merged = merge_incremental(existing, new)
    assert merged["is_limit_up_open"].dtype == bool
    assert merged["is_limit_up_open"].isna().sum() == 0
    assert merged["is_limit_up"].dtype == bool
    assert bool(merged.loc[merged["date"] == pd.Timestamp("2024-01-02"), "is_limit_up_open"].iloc[0])


def test_incremental_window():
    manifest = {
        "datasets": {
            "prices": {
                "600519.SH": {"end": "2024-01-10"},
                "000001.SZ": {"end": "2024-01-09"},
            }
        }
    }
    start = incremental_window(manifest, "prices", "2024-01-31")
    assert start == "2024-01-05"  # max end 2024-01-10 - 5 天
    assert incremental_window(None, "prices", "2024-01-31") == "1990-12-19"


def test_notifiers():
    from quant.monitor.alerts import LogNotifier, MultiNotifier

    class Stub:
        def __init__(self):
            self.calls = []

        def send(self, title, body):
            self.calls.append((title, body))

    a, b = Stub(), Stub()
    multi = MultiNotifier([a, b])
    multi.send("t", "b")
    assert len(a.calls) == 1 and len(b.calls) == 1
    LogNotifier().send("t", "b")
