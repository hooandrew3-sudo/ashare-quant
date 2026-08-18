"""真实 A 股数据回测对比：equal vs risk_budget，含/不含 CVaR。"""

from __future__ import annotations

import sys
import json
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from quant.config import Config
from quant.data.storage import Storage
from quant.pipeline import run_research


def load_real_bundle(cfg: Config) -> pd.DataFrame:
    """从已有 parquet 加载真实数据。"""
    storage = Storage(cfg.data.root)
    bundle = storage.load_bundle()
    bundle.validate()
    return bundle


def run_real_backtest(label: str, portfolio_overrides: dict) -> dict | None:
    cfg = Config()
    cfg.data.source = "parquet"
    cfg.data.root = Path("data")
    cfg.data.benchmark = "000906.SH"
    cfg.run.seed = 42
    cfg.run.verbose = False

    # 模型配置：用较少 fold 适配真实数据时间跨度
    cfg.model.horizons = [20]
    cfg.model.n_seeds = 1
    cfg.model.seeds = [42]
    cfg.model.n_splits = 3
    cfg.model.train_years = 2

    # 组合配置
    cfg.portfolio.top_n = 30
    cfg.portfolio.max_weight = 0.05
    cfg.portfolio.rebalance_freq = "M"
    cfg.portfolio.turnover_cap = 0.40
    cfg.portfolio.stickiness = 0.80

    # 回测配置
    cfg.backtest.start = "2021-06-01"
    cfg.backtest.end = "2022-12-07"
    cfg.backtest.initial_cash = 1_000_000

    # 因子配置
    cfg.factors.composite = True
    cfg.factors.composite_require_decay = True

    for k, v in portfolio_overrides.items():
        setattr(cfg.portfolio, k, v)

    try:
        print(f"\n{'='*70}")
        print(f"  REAL DATA BACKTEST: {label}")
        print(f"  Params: {portfolio_overrides}")
        print(f"{'='*70}")

        bundle = load_real_bundle(cfg)
        print(f"  Prices: {len(bundle.prices)} rows, {bundle.prices['symbol'].nunique()} symbols")
        print(f"  Date range: {bundle.prices['date'].min()} -> {bundle.prices['date'].max()}")
        print(f"  Benchmark: {len(bundle.benchmark)} rows")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out_root = Path("artifacts") / f"real_ab_{label}_{ts}"
        result = run_research(cfg, bundle, output_root=out_root)
        m = result["metrics"]
        oos = m.get("oos_model", {})
        drift = result.get("drift", {})

        row = {
            "label": label,
            "total_return": m["total_return"],
            "annualized_return": m["annualized_return"],
            "sharpe": m["sharpe"],
            "max_drawdown": m["max_drawdown"],
            "calmar": m["calmar"],
            "information_ratio": m["information_ratio"],
            "alpha": m["alpha"],
            "beta": m["beta"],
            "monthly_win_rate": m["monthly_win_rate"],
            "turnover_annual": m["turnover_annual"],
            "cost_drag_total": m["cost_drag_total"],
            "sell_hit_rate": m["sell_hit_rate"],
            "oos_auc": oos.get("oos_auc_mean"),
            "oos_rank_ic": oos.get("oos_rank_ic_mean"),
            "oos_rank_ic_ir": oos.get("oos_rank_ic_ir"),
            "drift_status": drift.get("status"),
            "drift_needs_retrain": drift.get("needs_retrain"),
        }
        print(f"\n  [OK] Sharpe={m['sharpe']:.3f}  Calmar={m['calmar']:.1f}  MDD={m['max_drawdown']:.2%}")
        print(f"       AnnRet={m['annualized_return']:.2%}  IR={m['information_ratio']:.3f}  Alpha={m['alpha']:.4f}")
        print(f"       OOS AUC={oos.get('oos_auc_mean')}  OOS IC={oos.get('oos_rank_ic_mean')}")
        print(f"       Drift={drift.get('status')}  Retrain={drift.get('needs_retrain')}")
        return row
    except Exception as exc:
        import traceback
        print(f"\n  [FAIL] {label}: {exc}")
        traceback.print_exc()
        return None


def main():
    rows = []

    configs = [
        ("baseline_equal", {"weight_method": "equal", "cvar_enabled": False}),
        ("risk_budget", {"weight_method": "risk_budget", "cvar_enabled": False}),
        ("equal_cvar", {"weight_method": "equal", "cvar_enabled": True, "cvar_threshold": -0.02}),
        ("risk_budget_cvar", {"weight_method": "risk_budget", "cvar_enabled": True, "cvar_threshold": -0.02}),
    ]

    for label, pf_overrides in configs:
        row = run_real_backtest(label, pf_overrides)
        if row is not None:
            rows.append(row)

    if not rows:
        print("\nAll configs failed. No result to compare.")
        return

    print(f"\n{'='*110}")
    print("  REAL A-SHARE DATA A/B COMPARISON SUMMARY")
    print(f"{'='*110}")

    df = pd.DataFrame(rows)
    for col in ["total_return", "annualized_return", "max_drawdown", "monthly_win_rate", "sell_hit_rate"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda v: f"{v:.2%}" if v is not None else "N/A")
    for col in ["sharpe", "calmar", "information_ratio", "beta", "turnover_annual"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda v: f"{v:.3f}" if v is not None else "N/A")
    for col in ["alpha"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda v: f"{v:.5f}" if v is not None else "N/A")
    for col in ["cost_drag_total"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda v: f"{v:,.2f}" if v is not None else "N/A")
    for col in ["oos_auc", "oos_rank_ic", "oos_rank_ic_ir"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda v: f"{v:.4f}" if v is not None else "N/A")

    print(df.to_string(index=False))
    print(f"{'='*110}")

    out_csv = Path("artifacts") / "real_ab_compare.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"\n  Saved: {out_csv}")


if __name__ == "__main__":
    main()