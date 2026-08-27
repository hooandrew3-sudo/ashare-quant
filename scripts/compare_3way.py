"""Three-way comparison: default vs real vs hybrid on full-window."""
import json
from pathlib import Path

runs = {
    "DEFAULT": Path("artifacts/20260821_175428_083ed44c76d3/metrics.json"),
    "REAL":    Path("artifacts/20260821_175428_6467e03ad13b/metrics.json"),
    "HYBRID":  Path("artifacts/20260821_184318_480975b879ed/metrics.json"),
}

data = {k: json.load(open(v)) for k, v in runs.items()}

print("=" * 100)
print(f"{'METRIC':<30} {'DEFAULT':>16} {'REAL':>16} {'HYBRID':>16} {'D-R':>10} {'D-H':>10} {'R-H':>10}")
print("=" * 100)
for k, label in [
    ("total_return",        "Total Return"),
    ("annualized_return",   "Annualized Return"),
    ("annualized_vol",      "Annualized Vol"),
    ("sharpe",              "Sharpe"),
    ("max_drawdown",        "Max Drawdown"),
    ("calmar",              "Calmar"),
    ("information_ratio",   "Information Ratio"),
    ("alpha",               "Alpha"),
    ("beta",                "Beta"),
    ("monthly_win_rate",    "Monthly Win Rate"),
    ("turnover_annual",     "Annual Turnover"),
    ("cost_drag_total",     "Cost Drag"),
    ("sell_hit_rate",       "Sell Hit Rate"),
    ("days",                "Backtest Days"),
]:
    vals = {n: data[n]["metrics"].get(k) for n in runs}
    def f(v): return f"{v:.4f}" if isinstance(v, float) else str(v)
    dr = vals["DEFAULT"] - vals["REAL"] if all(isinstance(v, (int, float)) for v in [vals["DEFAULT"], vals["REAL"]]) else None
    dh = vals["DEFAULT"] - vals["HYBRID"] if all(isinstance(v, (int, float)) for v in [vals["DEFAULT"], vals["HYBRID"]]) else None
    rh = vals["REAL"] - vals["HYBRID"] if all(isinstance(v, (int, float)) for v in [vals["REAL"], vals["HYBRID"]]) else None
    def g(x): return f"{x:.4f}" if isinstance(x, float) else "-"
    print(f"{label:<30} {f(vals['DEFAULT']):>16} {f(vals['REAL']):>16} {f(vals['HYBRID']):>16} {g(dr):>10} {g(dh):>10} {g(rh):>10}")

print("-" * 100)
print("OOS MODEL QUALITY")
print("-" * 100)
for k, label in [
    ("oos_auc_mean",      "AUC mean"),
    ("oos_rank_ic_mean",  "Rank IC mean"),
    ("oos_rank_ic_ir",    "Rank IC IR"),
]:
    vals = {n: data[n]["metrics"]["oos_model"].get(k) for n in runs}
    def f(v): return f"{v:.4f}" if isinstance(v, float) else str(v)
    dr = vals["DEFAULT"] - vals["REAL"]
    dh = vals["DEFAULT"] - vals["HYBRID"]
    rh = vals["REAL"] - vals["HYBRID"]
    def g(x): return f"{x:.4f}" if isinstance(x, float) else "-"
    print(f"{label:<30} {f(vals['DEFAULT']):>16} {f(vals['REAL']):>16} {f(vals['HYBRID']):>16} {g(dr):>10} {g(dh):>10} {g(rh):>10}")

print("=" * 100)
print("STRESS SUB-PERIODS")
print("=" * 100)
print(f"{'scenario':<14} {'metric':<14} {'DEFAULT':>14} {'REAL':>14} {'HYBRID':>14} {'D-H':>14}")
print("-" * 100)
for scen in ["crash_2015", "bear_2018", "smallcap_2024", "rally_2024"]:
    for k in ["return", "max_drawdown"]:
        vals = {n: data[n]["stress"].get(scen, {}).get(k) for n in runs}
        def f(v): return f"{v:.4f}" if isinstance(v, float) else str(v)
        dh = (vals["DEFAULT"] - vals["HYBRID"]) if all(isinstance(v, (int, float)) for v in [vals["DEFAULT"], vals["HYBRID"]]) else None
        def g(x): return f"{x:.4f}" if isinstance(x, float) else "-"
        print(f"{scen:<14} {k:<14} {f(vals['DEFAULT']):>14} {f(vals['REAL']):>14} {f(vals['HYBRID']):>14} {g(dh):>14}")
    print("-" * 100)
