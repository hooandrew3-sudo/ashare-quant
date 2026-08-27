"""数据层测试：合成数据、质量校验、存储往返。"""

from __future__ import annotations

import pandas as pd

from quant.data.baostock_sync import BaoStockSync
from quant.data.quality import DataQualityError, check_benchmark, check_prices
from quant.data.storage import Storage
from quant.data.synthetic import generate_synthetic


def test_synthetic_schema():
    bundle = generate_synthetic(n_stocks=20, years=1, start="2023-01-01", seed=7)
    bundle.validate()
    assert set(["date", "symbol", "open", "high", "low", "close", "volume"]).issubset(
        bundle.prices.columns
    )
    assert bundle.prices["date"].is_monotonic_increasing


def test_quality_passes_on_synthetic():
    bundle = generate_synthetic(n_stocks=30, years=1, start="2023-01-01", seed=7)
    assert check_prices(bundle.prices).ok
    assert check_benchmark(bundle.benchmark).ok


def test_quality_detects_corruption():
    bundle = generate_synthetic(n_stocks=10, years=1, start="2023-01-01", seed=7)
    df = bundle.prices.copy()
    df.loc[df.index[0], "high"] = df.loc[df.index[0], "low"] - 1.0
    rep = check_prices(df)
    assert not rep.ok
    assert any("high" in e for e in rep.errors)


def test_storage_roundtrip(tmp_path):
    bundle = generate_synthetic(n_stocks=5, years=1, start="2023-01-01", seed=7)
    st = Storage(tmp_path / "data")
    st.save("prices", bundle.prices)
    st.save("benchmark", bundle.benchmark, partition_by_symbol=False)
    loaded = st.load("prices")
    assert len(loaded) == len(bundle.prices)
    assert st.has("prices")
    assert st.manifest_path.exists()


def test_fill_suspensions_clips_to_fetch_range():
    """回归：增量拉取时补行不得超出本次拉取范围，防止覆盖历史真实数据。"""
    sync = BaoStockSync(verbose=False)
    full_calendar = pd.bdate_range("2021-01-04", "2026-08-14")
    df = pd.DataFrame(
        [
            {"date": pd.Timestamp("2026-08-10"), "close": 10.0, "volume": 100},
            {"date": pd.Timestamp("2026-08-14"), "close": 9.5, "volume": 120},
        ]
    )
    out = sync._fill_suspensions(df.copy(), "600000.SH", full_calendar, basic=None)
    # 只补 8/10 ~ 8/14 区间内的缺失交易日，不得包含 2021~2026-08-07
    assert out["date"].min() == pd.Timestamp("2026-08-10")
    assert out["date"].max() == pd.Timestamp("2026-08-14")
    assert len(out) == 5  # 8/10-8/14 共 5 个交易日，中间 3 天补为停牌
    assert out.loc[out["close"].isna(), "is_suspended"].astype(bool).all()
