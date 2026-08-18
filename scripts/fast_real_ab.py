"""真实 A 股数据快速对比：抽样 300 只股票，equal vs risk_budget。"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from quant.config import Config
from quant.data.storage import Storage
from quant.pipeline import run_research


def load_sampled_bundle(cfg: Config, n_symbols: int = 300, seed: int = 42) -> tuple:
    """加载全量 parquet，抽样 n_symbols 只（按日均成交额选流动性最好的）。"""
    storage = Storage(cfg.data.root)
    bundle = storage.load_bundle()

    prices = bundle.prices
    # 按 2021-2022 日均成交额排序（与回测区间一致），取前 n_symbols
    amt = (
        prices[prices["date"].between("2021-01-01", "2022-12-31")]
        .groupby("symbol")["amount"]
        .mean()
        .sort_values(ascending=False)
    )
    keep = set(amt.head(n_symbols).index)
    prices = prices[prices["symbol"].isin(keep)].copy()
    prices["date"] = pd.to_datetime(prices["date"])

    industry = bundle.industry
    if not industry.empty:
        industry = industry[industry["symbol"].isin(keep)].copy()

    fundamentals = bundle.fundamentals
    if not fundamentals.empty:
        # P0 数据污染修复中：当前 fundamentals 为 synthetic 残留，置空以使用 prices 自带 pe/pb
        fundamentals = pd.DataFrame(columns=["symbol", "as_of_date", "roe", "gross_margin", "div_yield"])

    sentiment = bundle.sentiment
    if not sentiment.empty:
        sentiment = sentiment[sentiment["symbol"].isin(keep)].copy()

    forecast = bundle.forecast
    if not forecast.empty:
        forecast = forecast[forecast["symbol"].isin(keep)].copy()

    meta = bundle.meta
    if not meta.empty:
        meta = meta[meta["symbol"].isin(keep)].copy()

    print(f"  Sampled {len(keep)} symbols")
    print(f"  Prices: {len(prices)} rows, date {prices['date'].min().date()} -> {prices['date'].max().date()}")
    print(f"  Benchmark: {len(bundle.benchmark)} rows")
    print(f"  Industry: {len(industry)} rows, {industry['industry'].nunique() if not industry.empty else 0} sectors")
    print(f"  Fundamentals: {len(fundamentals)} rows")

    return (
        pd.DataFrame(),
        {
            "prices": prices,
            "benchmark": bundle.benchmark,
            "meta": meta,
            "industry": industry,
            "fundamentals": fundamentals,
            "sentiment": sentiment,
            "forecast": forecast,
            "smallcap": bundle.smallcap,
        },
    )


def run_fast_ab(label: str, portfolio_overrides: dict) -> dict | None:
    cfg = Config()
    cfg.data.source = "parquet"
    cfg.data.root = Path("data")
    cfg.run.seed = 42
    cfg.run.verbose = False

    cfg.model.horizons = [20]
    cfg.model.n_seeds = 1
    cfg.model.seeds = [42]
    cfg.model.n_splits = 2
    cfg.model.train_years = 2
    cfg.model.test_months = 4

    cfg.portfolio.top_n = 20
    cfg.portfolio.max_weight = 0.08
    cfg.portfolio.rebalance_freq = "M"
    cfg.portfolio.turnover_cap = 0.40
    cfg.portfolio.stickiness = 0.80
    cfg.portfolio.min_avg_amount = 30_000_000

    cfg.backtest.start = "2021-06-01"
    cfg.backtest.end = "2022-12-07"
    cfg.backtest.initial_cash = 1_000_000

    cfg.factors.composite = True
    cfg.factors.composite_require_decay = True

    for k, v in portfolio_overrides.items():
        setattr(cfg.portfolio, k, v)

    try:
        print(f"\n{'='*70}")
        print(f"  FAST REAL BACKTEST: {label}")
        print(f"  Params: {portfolio_overrides}")
        print(f"{'='*70}")

        _, data = load_sampled_bundle(cfg, n_symbols=300)
        from quant.data.storage import DataBundle

        bundle = DataBundle(**data)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out_root = Path("artifacts") / f"fast_real_{label}_{ts}"
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
        }
        print(f"\n  [OK] Sharpe={m['sharpe']:.3f}  Calmar={m['calmar']:.1f}  MDD={m['max_drawdown']:.2%}")
        print(f"       AnnRet={m['annualized_return']:.2%}  IR={m['information_ratio']:.3f}")
        print(f"       OOS AUC={oos.get('oos_auc_mean')}  OOS IC={oos.get('oos_rank_ic_mean')}  ICIR={oos.get('oos_rank_ic_ir')}")
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
    ]
    for label, pf_overrides in configs:
        row = run_fast_ab(label, pf_overrides)
        if row is not None:
            rows.append(row)

    if not rows:
        print("\nAll configs failed.")
        return

    print(f"\n{'='*100}")
    print("  REAL A-SHARE FAST A/B COMPARISON")
    print(f"{'='*100}")
    df = pd.DataFrame(rows)
    out_csv = Path("artifacts") / "fast_real_ab_compare.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(df.to_string(index=False))
    print(f"  Saved: {out_csv}")


if __name__ == "__main__":
    main()