"""验证业绩预告数据的 alpha：预告净利增速因子的 IC 与分层收益。"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from quant.config import Config
from quant.data.storage import Storage
from quant.model.label import build_label


def main():
    cfg = Config()
    storage = Storage("data")
    bundle = storage.load_bundle()

    forecast = bundle.forecast
    print(f"forecast: {len(forecast)} rows, {forecast['symbol'].nunique()} symbols")
    print(f"  cols: {forecast.columns.tolist()}")
    print(f"  as_of_date range: {forecast['as_of_date'].min()} ~ {forecast['as_of_date'].max()}")

    # 构造因子：预告净利增速（%），只保留有效值
    f = forecast.copy()
    f["as_of_date"] = pd.to_datetime(f["as_of_date"])
    f["value"] = pd.to_numeric(f["forecast_growth"], errors="coerce")
    f = f.dropna(subset=["value"])

    # 因子长表（date=symbol=as_of_date）
    factor_long = f[["as_of_date", "symbol", "value"]].rename(columns={"as_of_date": "date"})
    print(f"factor_long: {len(factor_long)} rows")

    # 标签：未来 20 日超额
    label_long = build_label(bundle.prices, bundle.benchmark, cfg)
    print(f"label_long: {len(label_long)} rows")

    # 合并：因子日需要与标签日对齐（因子在 as_of_date 当天生效）
    merged = factor_long.merge(
        label_long[["date", "symbol", "excess"]], on=["date", "symbol"], how="inner"
    )
    print(f"merged (因子与标签同日): {len(merged)} rows")

    if merged.empty:
        print("\n因子日期与标签日期无交集（as_of_date 非交易日），改用最近交易日对齐")
        # 用 ffill 对齐到交易日
        prices_dates = sorted(pd.to_datetime(bundle.prices["date"].unique()))
        price_idx = pd.DatetimeIndex(prices_dates)

        def _align(d):
            pos = price_idx.searchsorted(d)
            if pos >= len(price_idx):
                return price_idx[-1]
            return price_idx[pos]

        f["date"] = f["as_of_date"].map(_align)
        factor_long = f[["date", "symbol", "value"]]
        merged = factor_long.merge(
            label_long[["date", "symbol", "excess"]], on=["date", "symbol"], how="inner"
        )
        print(f"merged after alignment: {len(merged)} rows")

    if merged.empty:
        print("仍无交集，退出")
        return

    # 截面 Rank IC
    merged["date"] = pd.to_datetime(merged["date"])
    ics = []
    basket_rets = []
    for d, g in merged.groupby("date"):
        if len(g) < 20:
            continue
        ic = g["value"].rank().corr(g["excess"].rank())
        if ic == ic:
            ics.append(ic)
        # top 20% 篮子
        top = g.sort_values("value", ascending=False).head(max(1, len(g) // 5))
        basket_rets.append(top["excess"].mean())

    ic_mean = float(np.mean(ics)) if ics else float("nan")
    icir = float(np.mean(ics) / np.std(ics)) if ics and np.std(ics) > 0 else float("nan")
    t_stat = float(np.mean(ics) / (np.std(ics) / np.sqrt(len(ics)))) if ics and np.std(ics) > 0 else float("nan")
    basket_mean = float(np.mean(basket_rets)) if basket_rets else float("nan")

    print(f"\n=== 业绩预告增速因子（{len(ics)} 个截面日）===")
    print(f"  Rank IC 均值:   {ic_mean:.4f}")
    print(f"  ICIR:           {icir:.3f}")
    print(f"  t 值:           {t_stat:.2f}")
    print(f"  top20% 篮子未来20日超额: {basket_mean:.2%}")


if __name__ == "__main__":
    main()
