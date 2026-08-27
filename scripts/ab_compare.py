"""A/B 配置对比：在相同 synthetic 数据上对比不同组合/风控配置。"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from quant.config import Config
from quant.pipeline import prepare_data, run_research

_cached_bundle = None


def run_ab(
    label: str,
    portfolio_overrides: dict,
    backtest_overrides: dict | None = None,
) -> dict | None:
    global _cached_bundle
    cfg = Config()
    cfg.data.source = "synthetic"
    # 隔离：synthetic 演示数据写入独立目录，严禁覆盖 data/ 真实数据
    cfg.data.root = Path("data_demo")
    cfg.data.demo.n_stocks = 20
    cfg.data.demo.years = 2
    cfg.model.n_splits = 3
    cfg.model.horizons = [20]
    cfg.model.n_seeds = 1
    cfg.model.seeds = [42]
    cfg.run.seed = 42
    cfg.run.verbose = False

    for k, v in portfolio_overrides.items():
        setattr(cfg.portfolio, k, v)
    if backtest_overrides:
        for k, v in backtest_overrides.items():
            setattr(cfg.backtest, k, v)

    try:
        print(f"\n{'='*60}")
        print(f"  Run: {label}")
        print(f"  Params: {portfolio_overrides}")
        print(f"{'='*60}")

        if _cached_bundle is None:
            _cached_bundle = prepare_data(cfg)
        bundle = _cached_bundle

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out_root = Path("artifacts") / f"ab_{label}_{ts}"
        result = run_research(cfg, bundle, output_root=out_root)
        m = result["metrics"]
        row = {
            "label": label,
            "total_return": m["total_return"],
            "annualized_return": m["annualized_return"],
            "sharpe": m["sharpe"],
            "max_drawdown": m["max_drawdown"],
            "calmar": m["calmar"],
            "information_ratio": m["information_ratio"],
            "beta": m["beta"],
            "monthly_win_rate": m["monthly_win_rate"],
            "turnover_annual": m["turnover_annual"],
            "cost_drag_total": m["cost_drag_total"],
            "sell_hit_rate": m["sell_hit_rate"],
        }
        print(f"  [OK] Sharpe={m['sharpe']:.3f}  Calmar={m['calmar']:.1f}  MDD={m['max_drawdown']:.2%}")
        return row
    except Exception as exc:  # noqa: BLE001
        import traceback
        print(f"  [FAIL] config failed: {exc}")
        traceback.print_exc()
        return None


def main() -> None:
    rows: list[dict] = []

    configs = [
        (
            "baseline_equal",
            {"weight_method": "equal", "cvar_enabled": False},
            None,
        ),
        (
            "risk_budget_no_cvar",
            {"weight_method": "risk_budget", "cvar_enabled": False},
            None,
        ),
        (
            "risk_budget_cvar",
            {"weight_method": "risk_budget", "cvar_enabled": True},
            None,
        ),
        (
            "full_risk_mgmt",
            {
                "weight_method": "risk_budget",
                "cvar_enabled": True,
                "signal_health_enabled": True,
                "smallcap_regime_enabled": True,
            },
            None,
        ),
    ]

    for label, pf_overrides, bt_overrides in configs:
        row = run_ab(label, pf_overrides, bt_overrides)
        if row is not None:
            rows.append(row)

    if not rows:
        print("\nAll configs failed. No result to compare.")
        return

    print(f"\n{'='*100}")
    print("  A/B Comparison Summary")
    print(f"{'='*100}")
    df = pd.DataFrame(rows)
    fmt = {
        "total_return": "{:.2%}".format,
        "annualized_return": "{:.2%}".format,
        "sharpe": "{:.3f}".format,
        "max_drawdown": "{:.2%}".format,
        "calmar": "{:.1f}".format,
        "information_ratio": "{:.3f}".format,
        "beta": "{:.3f}".format,
        "monthly_win_rate": "{:.2%}".format,
        "turnover_annual": "{:.2f}".format,
        "cost_drag_total": "{:,.2f}".format,
        "sell_hit_rate": "{:.2%}".format,
    }
    for col, fmt_fn in fmt.items():
        if col in df.columns:
            df[col + "_fmt"] = df[col].apply(fmt_fn)

    print(df.to_string(
        index=False,
        columns=["label", "annualized_return", "sharpe", "max_drawdown", "calmar",
                 "information_ratio", "turnover_annual", "sell_hit_rate", "cost_drag_total"],
        header=["config", "ann_return", "sharpe", "mdd", "calmar",
                "info_ratio", "ann_turnover", "sell_hit", "cost_drag"],
    ))
    print(f"{'='*100}")

    out_csv = Path("artifacts") / "ab_compare.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"\n  Saved: {out_csv}")


if __name__ == "__main__":
    main()