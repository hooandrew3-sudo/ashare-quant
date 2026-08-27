"""拉取全市场业绩预告（盈利预增/预减），写入 data/processed/forecast/all.parquet。

数据源：akshare stock_yjyg_em（按报告期批量，公告日点内时间）。
"""

from __future__ import annotations

import glob
import time
from pathlib import Path

import akshare as ak
import pandas as pd


def norm(code) -> str | None:
    s = str(code).strip().zfill(6)
    if len(s) != 6 or not s.isdigit():
        return None
    if s[0] in ("6", "9"):
        return f"{s}.SH"
    if s[0] in ("0", "3"):
        return f"{s}.SZ"
    return None


def main() -> None:
    existing_symbols = {
        f.split("\\")[-1].replace(".parquet", "")
        for f in glob.glob("data/processed/prices/*.parquet")
    }
    print(f"universe symbols={len(existing_symbols)}", flush=True)

    quarters = []
    for y in range(2018, 2027):
        for md in ("0331", "0630", "0930", "1231"):
            q = f"{y}{md}"
            if q > "20260630":
                break
            quarters.append(q)
    print(f"quarters={len(quarters)} {quarters[0]}..{quarters[-1]}", flush=True)

    rows = []
    for q in quarters:
        df = None
        for attempt in range(3):
            try:
                df = ak.stock_yjyg_em(date=q)
                break
            except Exception as exc:  # noqa: BLE001
                if attempt == 2:
                    print(f"FAIL {q}: {exc}", flush=True)
                time.sleep(1.5)
        if df is None or df.empty:
            continue
        if "预测指标" not in df.columns:
            continue
        sub = df[df["预测指标"] == "归属于上市公司股东的净利润"].copy()
        sub["symbol"] = sub["股票代码"].map(norm)
        sub = sub.dropna(subset=["symbol"])
        sub["as_of_date"] = pd.to_datetime(sub["公告日期"], errors="coerce")
        sub["forecast_growth"] = pd.to_numeric(sub["业绩变动幅度"], errors="coerce")
        sub["forecast_type"] = sub["预告类型"].astype(str)
        rows.append(sub[["symbol", "as_of_date", "forecast_growth", "forecast_type"]])
        print(f"  {q}: +{len(sub)} rows", flush=True)
        time.sleep(0.15)

    out = pd.concat(rows, ignore_index=True).dropna(subset=["as_of_date", "forecast_growth"])
    out = out.drop_duplicates(subset=["symbol", "as_of_date"], keep="last")
    out["as_of_date"] = pd.to_datetime(out["as_of_date"]).astype("datetime64[ns]")
    Path("data/processed/forecast").mkdir(parents=True, exist_ok=True)
    out.to_parquet("data/processed/forecast/all.parquet", index=False)
    print(f"forecast rows={len(out)} symbols={out['symbol'].nunique()}", flush=True)

    in_univ = out[out["symbol"].isin(existing_symbols)]
    print(f"in-universe rows={len(in_univ)} symbols={in_univ['symbol'].nunique()}", flush=True)
    print(
        f"date range={in_univ['as_of_date'].min().date()}..{in_univ['as_of_date'].max().date()}",
        flush=True,
    )
    print("growth desc:", flush=True)
    print(in_univ["forecast_growth"].describe().round(2).to_string(), flush=True)
    print("type counts:", flush=True)
    print(in_univ["forecast_type"].value_counts().to_dict(), flush=True)


if __name__ == "__main__":
    main()
