"""审计数据探查脚本。"""
import pandas as pd
from pathlib import Path

p = Path("data/processed/prices")
files = sorted(p.glob("*.parquet"))
print(f"Stock count: {len(files)}")

df = pd.read_parquet(files[0])
print(f"Columns: {df.columns.tolist()}")
print(f"Date range: {df['date'].min()} -> {df['date'].max()}")
print(f"Rows: {len(df)}")
print("\nHead:")
print(df.head(2).to_string())
print("\nTail:")
print(df.tail(2).to_string())

# 全市场统计
all_dates = set()
for f in files[:50]:  # 抽样50只
    d = pd.read_parquet(f)
    all_dates.update(d["date"].tolist())
dates = sorted(all_dates)
print(f"\nSample 50 stocks date range: {dates[0]} -> {dates[-1]}")
print(f"Total trading days: {len(dates)}")

# benchmark
bench = pd.read_parquet("data/processed/benchmark")
print(f"\nBenchmark columns: {bench.columns.tolist()}")
print(f"Benchmark rows: {len(bench)}")
print(f"Benchmark date range: {bench['date'].min()} -> {bench['date'].max()}")

# industry
ind = pd.read_parquet("data/processed/industry")
print(f"\nIndustry columns: {ind.columns.tolist()}")
print(f"Industry rows: {len(ind)}")
print(ind.head(3).to_string())

# smallcap
sc = pd.read_parquet("data/processed/smallcap")
print(f"\nSmallcap columns: {sc.columns.tolist()}")
print(f"Smallcap rows: {len(sc)}")
if len(sc) > 0:
    print(f"Smallcap date range: {sc['date'].min()} -> {sc['date'].max()}")

# fundamentals
fund = pd.read_parquet("data/processed/fundamentals")
print(f"\nFundamentals columns: {fund.columns.tolist()}")
print(f"Fundamentals rows: {len(fund)}")
