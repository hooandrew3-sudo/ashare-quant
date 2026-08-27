"""验证 composite 直接选股模式（selection_mode=composite）正常。"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from quant.config import Config
from quant.data.storage import Storage, DataBundle
from quant.pipeline import run_research


def load_small_bundle(n_symbols: int = 100) -> DataBundle:
    storage = Storage("data")
    bundle = storage.load_bundle()
    prices = bundle.prices
    prices["date"] = pd.to_datetime(prices["date"])
    amt = prices[prices["date"] <= "2019-12-31"].groupby("symbol")["amount"].mean().sort_values(ascending=False)
    keep = set(amt.head(n_symbols).index)

    def _filt(df, cols):
        if df is None or df.empty:
            return pd.DataFrame(columns=cols)
        return df[df["symbol"].isin(keep)].copy()

    return DataBundle(
        prices=prices[prices["symbol"].isin(keep)].copy(),
        benchmark=bundle.benchmark,
        meta=_filt(bundle.meta, ["symbol", "name", "list_date", "delist_date"]),
        industry=_filt(bundle.industry, ["symbol", "industry", "as_of_date"]),
        fundamentals=_filt(bundle.fundamentals, ["symbol", "as_of_date", "roe", "gross_margin", "div_yield"]),
        sentiment=_filt(bundle.sentiment, ["symbol", "date", "sentiment"]),
        forecast=bundle.forecast,
        smallcap=bundle.smallcap,
    )


def main():
    for mode in ["model", "composite", "hybrid"]:
        cfg = Config()
        cfg.data.source = "parquet"
        cfg.data.root = Path("data")
        cfg.run.seed = 42
        cfg.run.verbose = False
        cfg.model.selection_mode = mode
        cfg.model.n_splits = 2
        cfg.model.test_months = 3
        cfg.portfolio.top_n = 15
        cfg.portfolio.max_weight = 0.10
        cfg.backtest.start = "2021-01-01"
        cfg.backtest.end = "2026-08-17"
        try:
            bundle = load_small_bundle(100)
            result = run_research(cfg, bundle, output_root=Path("artifacts") / f"mode_test_{mode}")
            m = result["metrics"]
            oos = m.get("oos_model", {})
            print(f"[{mode:>9}] Sharpe={m['sharpe']:.2f}  MDD={m['max_drawdown']:.2%}  "
                  f"OOS IC={oos.get('oos_rank_ic_mean')}  ICIR={oos.get('oos_rank_ic_ir')}  "
                  f"turnover={m['turnover_annual']:.1f}x")
        except Exception as exc:
            import traceback
            print(f"[{mode:>9}] FAILED: {exc}")
            traceback.print_exc()


if __name__ == "__main__":
    main()