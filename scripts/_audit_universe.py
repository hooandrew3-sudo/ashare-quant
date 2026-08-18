"""验证 point-in-time 选股池修正效果。"""
import pandas as pd
from pathlib import Path

p = Path("data/processed/prices")
prices = pd.concat([pd.read_parquet(f) for f in sorted(p.glob("*.parquet"))], ignore_index=True)
prices["date"] = pd.to_datetime(prices["date"])

# 旧方法：2021-2022 流动性 top300
old_amt = (
    prices[prices["date"].between("2021-01-01", "2022-12-31")]
    .groupby("symbol")["amount"].mean().sort_values(ascending=False)
)
old_keep = set(old_amt.head(300).index)

# 新方法：2019 年底前流动性 top300（point-in-time）
hist = prices[prices["date"] <= "2019-12-31"]
if hist.empty:
    hist = prices
new_amt = hist.groupby("symbol")["amount"].mean().sort_values(ascending=False)
new_keep = set(new_amt.head(300).index)

print(f"旧选股池(2021-22流动性): {len(old_keep)} 只")
print(f"新选股池(2019流动性): {len(new_keep)} 只")
print(f"重叠: {len(old_keep & new_keep)} 只")
print(f"新增(旧池没有): {len(new_keep - old_keep)} 只")
print(f"移除(旧池有、新池没有): {len(old_keep - new_keep)} 只")

# 检查新池中是否包含旧池里 2021-2022 才上市的新股
meta = pd.read_parquet("data/processed/meta")
if not meta.empty and "list_date" in meta.columns:
    md = meta.copy()
    md["list_date"] = pd.to_datetime(md["list_date"], errors="coerce")
    old_late = md[(md["symbol"].isin(old_keep)) & (md["list_date"] > "2019-12-31")]
    print(f"\n旧池中 2020 年后才上市的股票（后视偏差来源）: {len(old_late)} 只")
    print(old_late[["symbol", "name"]].head(5).to_string())
