"""验证恢复后的数据完整性。"""
import pandas as pd
from pathlib import Path

p = Path("data/processed/prices")

for sym in ["000001.SZ", "600000.SH", "600519.SH", "600036.SH"]:
    f = p / f"{sym}.parquet"
    if f.exists():
        df = pd.read_parquet(f)
        print(
            f"{sym}: rows={len(df)}, {df['date'].min().date()}->{df['date'].max().date()}, "
            f"close_mean={df['close'].mean():.2f}"
        )

# benchmark
b = pd.read_parquet("data/processed/benchmark")
print(f"\nbenchmark: rows={len(b)}, {b['date'].min().date()}->{b['date'].max().date()}")

# industry
i = pd.read_parquet("data/processed/industry")
print(f"industry: rows={len(i)}, sectors={i['industry'].nunique()}")

# meta
m = pd.read_parquet("data/processed/meta")
print(f"meta: rows={len(m)}")

# 确认不再有 synthetic 污染特征（停牌率异常高）
for sym in ["000001.SZ", "600000.SH"]:
    df = pd.read_parquet(p / f"{sym}.parquet")
    susp = df["is_suspended"].mean()
    print(f"{sym} suspension rate: {susp:.2%}")
