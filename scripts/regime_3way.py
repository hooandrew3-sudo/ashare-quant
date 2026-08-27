"""Quarterly regime breakdown across all three runs."""
import pandas as pd
import numpy as np

runs = {
    "DEFAULT": "artifacts/20260821_175428_083ed44c76d3/equity.parquet",
    "REAL":    "artifacts/20260821_175428_6467e03ad13b/equity.parquet",
    "HYBRID":  "artifacts/20260821_184318_480975b879ed/equity.parquet",
}

data = {}
for n, p in runs.items():
    df = pd.read_parquet(p)
    df['date'] = pd.to_datetime(df['date'])
    df['exposure'] = (df['portfolio_value'] - df['cash']) / df['portfolio_value']
    df['quarter'] = df['date'].dt.to_period('Q')
    data[n] = df

print("=" * 110)
print(f"{'Quarter':<10} {'Metric':<14} {'DEFAULT':>14} {'REAL':>14} {'HYBRID':>14}  Winner")
print("=" * 110)

# Collect all quarters
all_q = sorted(set().union(*[set(d['quarter'].unique()) for d in data.values()]))

for q in all_q:
    rows = []
    for n in ["DEFAULT", "REAL", "HYBRID"]:
        df = data[n]
        sub = df[df['quarter'] == q]
        if len(sub) < 2:
            rows.append((None, None, None))
            continue
        pr = sub['portfolio_value'].iloc[-1] / sub['portfolio_value'].iloc[0] - 1
        br = sub['benchmark_value'].iloc[-1] / sub['benchmark_value'].iloc[0] - 1
        dd = (sub['portfolio_value']/sub['portfolio_value'].cummax() - 1).min()
        avg_exp = sub['exposure'].mean()
        rows.append((pr, br, dd, avg_exp))
    prs = [r[0] for r in rows]
    brs = [r[1] for r in rows]
    dds = [r[2] for r in rows]
    exps = [r[3] for r in rows]
    def f(v): return f"{v*100:>+.2f}%" if isinstance(v, float) else "    --"
    win_idx = prs.index(max(p for p in prs if p is not None)) if any(p is not None for p in prs) else 0
    winner = ["DEFAULT","REAL","HYBRID"][win_idx]
    for metric_name, vals in [("port_ret", prs), ("bench_ret", brs), ("excess", [p-b if (p is not None and b is not None) else None for p,b in zip(prs, brs)]), ("max_dd", dds), ("avg_exp", exps)]:
        print(f"{str(q):<10} {metric_name:<14} {f(vals[0]):>14} {f(vals[1]):>14} {f(vals[2]):>14}")
    print(f"  Winner: {winner}")
    print("-" * 110)
