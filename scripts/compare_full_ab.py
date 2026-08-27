"""Compare two metrics.json side-by-side for cross-regime robustness."""
import json, sys
from pathlib import Path

if len(sys.argv) != 3:
    print("Usage: python compare_full_ab.py <run_dir_a> <run_dir_b>")
    sys.exit(1)

a = json.load(open(Path(sys.argv[1]) / "metrics.json"))
b = json.load(open(Path(sys.argv[2]) / "metrics.json"))

print("=" * 88)
print(f"{'METRIC':<28} {'DEFAULT (full)':>20} {'REAL (full)':>20} {'DIFF (D-R)':>15}")
print("=" * 88)
keys = [
    ("total_return",        "总收益"),
    ("annualized_return",   "年化收益"),
    ("annualized_vol",      "年化波动"),
    ("sharpe",              "Sharpe"),
    ("max_drawdown",        "最大回撤"),
    ("calmar",              "Calmar"),
    ("information_ratio",   "信息比率"),
    ("alpha",               "Alpha"),
    ("beta",                "Beta"),
    ("monthly_win_rate",    "月胜率"),
    ("turnover_annual",     "年化换手"),
    ("cost_drag_total",     "成本拖累"),
    ("sell_hit_rate",       "卖出命中率"),
    ("days",                "回测天数"),
]
am, bm = a["metrics"], b["metrics"]
for k, label in keys:
    va = am.get(k); vb = bm.get(k)
    diff = (va - vb) if (va is not None and vb is not None) else None
    print(f"{label:<28} {va!r:>20} {vb!r:>20} {diff!r:>15}")

print("-" * 88)
print("OOS 模型质量（跨 fold 平均）")
print("-" * 88)
for k, label in [
    ("oos_auc_mean",      "AUC"),
    ("oos_rank_ic_mean",  "Rank IC"),
    ("oos_rank_ic_ir",    "Rank IC IR"),
    ("oos_folds",         "有效 fold 数"),
]:
    va = am["oos_model"].get(k); vb = bm["oos_model"].get(k)
    print(f"{label:<28} {va!r:>20} {vb!r:>20}")

print("=" * 88)
print("STRESS 子窗口（cross-regime robustness test）")
print("=" * 88)
print(f"{'scenario':<14} {'metric':<14} {'DEFAULT':>16} {'REAL':>16} {'DIFF':>16}")
print("-" * 88)
for scen in ["crash_2015", "bear_2018", "smallcap_2024", "rally_2024"]:
    sa = a["stress"].get(scen, {}); sb = b["stress"].get(scen, {})
    for k in ["return", "max_drawdown"]:
        va = sa.get(k); vb = sb.get(k)
        diff = (va - vb) if (va is not None and vb is not None) else None
        print(f"{scen:<14} {k:<14} {va!r:>16} {vb!r:>16} {diff!r:>16}")
    print(f"  ↳ ok={sa.get('ok')} / ok={sb.get('ok')} | "
          f"coverage_days {sa.get('coverage_days')}/{sb.get('coverage_days')}")
    print("-" * 88)
