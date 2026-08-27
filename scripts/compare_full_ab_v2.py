"""ASCII-only cross-regime A/B comparison."""
import json, sys
from pathlib import Path

a = json.load(open(Path(sys.argv[1]) / "metrics.json"))
b = json.load(open(Path(sys.argv[2]) / "metrics.json"))

am, bm = a["metrics"], b["metrics"]
print("=" * 92)
print(f"{'METRIC':<30} {'DEFAULT':>18} {'REAL':>18} {'DIFF(D-R)':>14}")
print("=" * 92)
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
    va, vb = am.get(k), bm.get(k)
    diff = (va - vb) if (va is not None and vb is not None) else None
    def f(v): return f"{v:.4f}" if isinstance(v, float) else str(v)
    print(f"{label:<30} {f(va):>18} {f(vb):>18} {f(diff) if diff is not None else '-':>14}")

print("-" * 92)
print("OOS MODEL QUALITY (avg across folds / seeds / horizons)")
print("-" * 92)
for k, label in [
    ("oos_auc_mean",      "AUC mean"),
    ("oos_rank_ic_mean",  "Rank IC mean"),
    ("oos_rank_ic_ir",    "Rank IC IR"),
]:
    va = am["oos_model"].get(k); vb = bm["oos_model"].get(k)
    diff = va - vb
    def f(v): return f"{v:.4f}" if isinstance(v, float) else str(v)
    print(f"{label:<30} {f(va):>18} {f(vb):>18} {f(diff):>14}")

print("=" * 92)
print("STRESS SUB-PERIODS (cross-regime robustness)")
print("=" * 92)
print(f"{'scenario':<16} {'metric':<16} {'DEFAULT':>14} {'REAL':>14} {'DIFF':>14}  notes")
print("-" * 92)
for scen in ["crash_2015", "bear_2018", "smallcap_2024", "rally_2024"]:
    sa, sb = a["stress"].get(scen, {}), b["stress"].get(scen, {})
    for k in ["return", "max_drawdown"]:
        va, vb = sa.get(k), sb.get(k)
        diff = (va - vb) if (va is not None and vb is not None) else None
        def f(v): return f"{v:.4f}" if isinstance(v, float) else str(v)
        print(f"{scen:<16} {k:<16} {f(va):>14} {f(vb):>14} {f(diff) if diff is not None else '-':>14}")
    print(f"  ok={sa.get('ok')}/ok={sb.get('ok')}  "
          f"coverage_days={sa.get('coverage_days')}/{sb.get('coverage_days')}")
    print("-" * 92)
