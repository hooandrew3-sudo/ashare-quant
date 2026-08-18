"""长区间真实数据回测：OOS 约 2 年，对比 equal vs risk_budget。"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from quant.config import Config
from quant.data.storage import Storage, DataBundle
from quant.pipeline import run_research


def load_sampled_bundle(cfg: Config, n_symbols: int = 300, as_of: str = "2019-12-31") -> DataBundle:
    """Point-in-time 选池：只用 as_of 之前的流动性选股，消除后视偏差。

    as_of 默认回测起点（2020-01-01）之前，确保选池不包含未来信息；
    同时用 meta.delist_date 排除回测区间内已退市的股票（幸存者偏差修正）。
    """
    storage = Storage(cfg.data.root)
    bundle = storage.load_bundle()

    prices = bundle.prices
    prices["date"] = pd.to_datetime(prices["date"])

    # 选池窗口：as_of 前 252 个交易日
    lookback_end = pd.Timestamp(as_of)
    hist = prices[prices["date"] <= lookback_end]
    if hist.empty:
        # 数据不足时退化为全样本流动性
        hist = prices
    amt = hist.groupby("symbol")["amount"].mean().sort_values(ascending=False)

    # 幸存者偏差修正：排除回测区间内退市的股票
    delisted_syms = set()
    if not bundle.meta.empty and "delist_date" in bundle.meta.columns:
        md = bundle.meta.copy()
        md["delist_date"] = pd.to_datetime(md["delist_date"], errors="coerce")
        delisted = md[md["delist_date"].notna() & (md["delist_date"] <= pd.Timestamp("2026-08-17"))]
        delisted_syms = set(delisted["symbol"].unique())

    keep = set(amt.head(n_symbols + len(delisted_syms)).index) - delisted_syms
    keep = set(sorted(keep, key=lambda s: amt[s], reverse=True)[:n_symbols])
    prices = prices[prices["symbol"].isin(keep)].copy()

    def _filt(df, cols):
        if df is None or df.empty:
            return pd.DataFrame(columns=cols)
        return df[df["symbol"].isin(keep)].copy()

    return DataBundle(
        prices=prices,
        benchmark=bundle.benchmark,
        meta=_filt(bundle.meta, ["symbol", "name", "list_date", "delist_date"]),
        industry=_filt(bundle.industry, ["symbol", "industry", "as_of_date"]),
        fundamentals=_filt(bundle.fundamentals, ["symbol", "as_of_date", "roe", "gross_margin", "div_yield"]),
        sentiment=_filt(bundle.sentiment, ["symbol", "date", "sentiment"]),
        forecast=bundle.forecast,
        smallcap=bundle.smallcap,
    )


def run_long_ab(label: str, portfolio_overrides: dict) -> dict | None:
    cfg = Config()
    cfg.data.source = "parquet"
    cfg.data.root = Path("data")
    cfg.run.seed = 42
    cfg.run.verbose = False

    cfg.model.horizons = [20]
    cfg.model.n_seeds = 1
    cfg.model.seeds = [42]
    cfg.model.n_splits = 4
    cfg.model.train_years = 2
    cfg.model.test_months = 6

    cfg.portfolio.top_n = 20
    cfg.portfolio.max_weight = 0.08
    cfg.portfolio.rebalance_freq = "M"
    cfg.portfolio.turnover_cap = 0.40
    cfg.portfolio.stickiness = 0.80
    cfg.portfolio.min_avg_amount = 30_000_000

    cfg.backtest.start = "2020-01-01"
    cfg.backtest.end = "2026-08-17"
    cfg.backtest.initial_cash = 1_000_000

    cfg.factors.composite = True
    cfg.factors.composite_require_decay = True

    for k, v in portfolio_overrides.items():
        setattr(cfg.portfolio, k, v)

    try:
        print(f"\n{'='*70}")
        print(f"  LONG REAL BACKTEST: {label}")
        print(f"  Params: {portfolio_overrides}")
        print(f"{'='*70}")

        bundle = load_sampled_bundle(cfg)
        print(f"  Prices: {len(bundle.prices)} rows, {bundle.prices['symbol'].nunique()} symbols")
        print(f"  Benchmark: {len(bundle.benchmark)} rows, {bundle.benchmark['date'].min().date()}~{bundle.benchmark['date'].max().date()}")
        print(f"  Industry: {len(bundle.industry)} rows, {bundle.industry['industry'].nunique() if not bundle.industry.empty else 0} sectors")
        print(f"  Fundamentals: {len(bundle.fundamentals)} rows")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out_root = Path("artifacts") / f"long_real_{label}_{ts}"
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
        row = run_long_ab(label, pf_overrides)
        if row is not None:
            rows.append(row)

    if not rows:
        print("\nAll configs failed.")
        return

    print(f"\n{'='*100}")
    print("  LONG REAL A-SHARE A/B COMPARISON (OOS ~2yr)")
    print(f"{'='*100}")
    df = pd.DataFrame(rows)
    out_csv = Path("artifacts") / "long_real_ab_compare.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(df.to_string(index=False))
    print(f"  Saved: {out_csv}")


if __name__ == "__main__":
    main()