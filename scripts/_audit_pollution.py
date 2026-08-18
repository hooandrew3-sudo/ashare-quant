"""验证 synthetic 数据污染范围。"""
import pandas as pd
from pathlib import Path

p = Path("data/processed/prices")

# synthetic 生成的股票代码范围（n_stocks=20 时）
syn_syms = [f"{600000 + i:06d}.SH" for i in range(10)] + [f"{1 + i:06d}.SZ" for i in range(10)]
print("Synthetic symbols:", syn_syms)

# 检查被污染股票 vs 真实股票
for sym in ["000001.SZ", "600000.SH", "600519.SH", "600036.SH", "000858.SZ"]:
    f = p / f"{sym}.parquet"
    if f.exists():
        df = pd.read_parquet(f)
        n = len(df)
        dmin = df["date"].min().date()
        dmax = df["date"].max().date()
        close_mean = df["close"].mean()
        # 检查是否停牌标记占比异常（synthetic 有 ~2% 停牌）
        susp = df["is_suspended"].mean() if "is_suspended" in df.columns else 0
        st = df["is_st"].mean() if "is_st" in df.columns else 0
        pe = df["pe"].mean() if "pe" in df.columns else 0
        # synthetic 的 pe 在 3~200 之间且均值 ~20，真实 A 股 pe 均值 ~30-60
        print(f"{sym}: rows={n}, {dmin}->{dmax}, close_mean={close_mean:.2f}, susp={susp:.2%}, st={st:.2%}, pe_mean={pe:.1f}")
    else:
        print(f"{sym}: NOT FOUND")
