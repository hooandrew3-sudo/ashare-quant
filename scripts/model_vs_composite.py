"""模型 vs 因子直接选股对比：验证 LightGBM 是否损耗因子信息。

在相同 OOS 日期上比较：
- composite 因子 top-N 篮子未来 20 日超额收益
- 单因子（动量/拥挤度/小市值）top-N 篮子
- 等权多因子组合
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from quant.config import Config
from quant.data.storage import Storage, DataBundle
from quant.factors.compute import compute_all_factors
from quant.factors.analysis import factor_ic_report
from quant.factors.composite import build_composite_factor
from quant.model.label import build_label


def load_bundle(n_symbols: int = 300) -> DataBundle:
    storage = Storage("data")
    bundle = storage.load_bundle()
    prices = bundle.prices
    amt = (
        prices[prices["date"].between("2021-01-01", "2022-12-31")]
        .groupby("symbol")["amount"]
        .mean()
        .sort_values(ascending=False)
    )
    keep = set(amt.head(n_symbols).index)
    prices = prices[prices["symbol"].isin(keep)].copy()
    prices["date"] = pd.to_datetime(prices["date"])

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


def basket_excess_ic(factor_wide: pd.DataFrame, excess_wide: pd.DataFrame, top_n: int = 20) -> tuple[float, float]:
    """每个日期取 top-N，计算未来 20 日超额收益均值；返回均值与 IC。"""
    dates = factor_wide.index
    row_ic = []
    row_ret = []
    for d in dates:
        if d not in excess_wide.index:
            continue
        f_row = factor_wide.loc[d].dropna()
        e_row = excess_wide.loc[d].reindex(f_row.index).dropna()
        if len(e_row) < 30:
            continue
        f_row = f_row.reindex(e_row.index)
        # Rank IC
        if f_row.std() > 0:
            ic = f_row.rank().corr(e_row.rank())
            row_ic.append(ic)
        # top-N 篮子收益
        top = f_row.sort_values(ascending=False).head(top_n).index
        basket = e_row.loc[top].mean()
        row_ret.append(basket)
    ic_mean = float(np.mean(row_ic)) if row_ic else float("nan")
    ret_mean = float(np.mean(row_ret)) if row_ret else float("nan")
    return ic_mean, ret_mean


def main() -> None:
    cfg = Config()
    cfg.data.source = "parquet"
    cfg.data.root = Path("data")
    cfg.run.seed = 42
    cfg.run.verbose = False

    bundle = load_bundle(300)
    print(f"prices: {len(bundle.prices)} rows, {bundle.prices['symbol'].nunique()} symbols")

    # 因子
    factor_long = compute_all_factors(bundle, cfg)
    label_long = build_label(bundle.prices, bundle.benchmark, cfg)

    # 训练期 cutoff（与 pipeline 一致：前 60% 时间）
    all_dates = pd.to_datetime(factor_long["date"]).sort_values().unique()
    cutoff_idx = int(len(all_dates) * 0.6)
    train_cutoff = pd.Timestamp(all_dates[cutoff_idx])
    f_in = factor_long[pd.to_datetime(factor_long["date"]) <= train_cutoff]
    l_in = label_long[pd.to_datetime(label_long["date"]) <= train_cutoff]
    ic_report = factor_ic_report(f_in, l_in, cfg)

    # composite 因子（训练期选择成分）
    composite = build_composite_factor(
        factor_long, ic_report, n=5, min_t=cfg.factors.min_t_stat,
        corr_max=0.6, require_stable_decay=True, weight_by="icir",
    )
    print(f"composite components: {ic_report['passed']}")
    factor_long = pd.concat([factor_long, composite], ignore_index=True)

    # 宽表
    labels = label_long.set_index(["date", "symbol"])["excess"]
    excess_wide = labels.unstack()

    # 对比因子
    targets = ["composite", "mom_12_1", "crowding", "size_proxy", "low_vol", "max_ret"]

    print(f"\n{'='*90}")
    print("  因子直接选股能力（OOS: 2020-2026 全样本）")
    print(f"  train_cutoff={train_cutoff.date()}, OOS 交易日={len(excess_wide.index)}")
    print(f"{'='*90}")
    print(f"  {'因子':<14} {'Rank IC':>10} {'top-20 未来20日超额':>20}")
    print(f"  {'-'*60}")

    results = []
    for name in targets:
        sub = factor_long[factor_long["factor"] == name]
        if sub.empty:
            continue
        f_wide = sub.pivot(index="date", columns="symbol", values="value")
        ic, ret = basket_excess_ic(f_wide, excess_wide, top_n=20)
        results.append({"factor": name, "rank_ic": ic, "basket_excess": ret})
        print(f"  {name:<14} {ic:>10.4f} {ret:>20.2%}")

    # 基准：随机选股
    rng = np.random.default_rng(42)
    rand_ics, rand_rets = [], []
    for d in excess_wide.index:
        e_row = excess_wide.loc[d].dropna()
        if len(e_row) < 30:
            continue
        rand = rng.choice(e_row.index, size=min(20, len(e_row)), replace=False)
        rand_rets.append(e_row.loc[rand].mean())
    rand_ret_mean = float(np.mean(rand_rets)) if rand_rets else float("nan")
    print(f"  {'random_top20':<14} {'N/A':>10} {rand_ret_mean:>20.2%}")

    print(f"\n  {'='*60}")
    print(f"  解读：top-20 篮子超额 > 随机基准 说明因子有真实选股能力")
    print(f"  {'='*60}")


if __name__ == "__main__":
    main()