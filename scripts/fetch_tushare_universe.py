"""Tushare 历史指数成分 + 全量股票池 + 申万行业回填（消除幸存者偏差与行业前视）。

前置：
  1. pip install tushare
  2. 在 .env 配置 TUSHARE_TOKEN（已在 .gitignore，勿硬编码）
  3. 确认账户积分覆盖接口：stock_basic（基础）、index_weight（月频成分，需较高积分）、
     index_classify / index_member（申万行业）

用法：
  python scripts/fetch_tushare_universe.py --start 2018-01-01 --end 2026-08-12

注意：Tushare 接口字段/积分门槛会随版本调整，本脚本为骨架，字段名需以实际返回为准校验。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


def get_pro():
    import tushare as ts

    token = os.getenv("TUSHARE_TOKEN", "")
    if not token:
        raise RuntimeError("缺少 TUSHARE_TOKEN（.env 或环境变量）")
    return ts.pro_api(token)


def fetch_stock_basic(pro) -> pd.DataFrame:
    """全量股票（含退市 L / 暂停 P / 退市 D）→ symbol, name, list_date, delist_date。"""
    parts = []
    for status in ("L", "D", "P"):
        df = pro.stock_basic(
            exchange="",
            list_status=status,
            fields="ts_code,symbol,name,list_date,delist_date,list_status",
        )
        if df is not None and not df.empty:
            parts.append(df)
    basic = pd.concat(parts, ignore_index=True).drop_duplicates("ts_code")
    basic = basic.rename(columns={"ts_code": "symbol"})
    for col in ("list_date", "delist_date"):
        basic[col] = pd.to_datetime(basic[col], errors="coerce")
    return basic[["symbol", "name", "list_date", "delist_date"]]


def fetch_csi800_members(pro, start: str, end: str) -> pd.DataFrame:
    """中证800 月度成分（point-in-time）。

    中证800 = 沪深300(000300.SH) + 中证500(000905.SH)；若账户支持 000906.SH，
    可直接用中证800 指数代码一次拉取。这里用 300+500 并集兜底。
    """
    out = []
    for code in ("000300.SH", "000905.SH"):
        df = pro.index_weight(index_code=code, start_date=start, end_date=end)
        if df is not None and not df.empty:
            out.append(df[["con_code", "trade_date"]])
    members = pd.concat(out, ignore_index=True).drop_duplicates()
    members = members.rename(columns={"con_code": "symbol", "trade_date": "date"})
    members["date"] = pd.to_datetime(members["date"])
    return members


def fetch_sw_industry(pro) -> pd.DataFrame:
    """申万行业分类。

    当前快照：index_classify(level="L1", src="SW2021")。
    point-in-time：若积分支持，遍历申万一级行业指数调用 index_member(index_code=...)，
    得到每只股票的 con_code / in_date / out_date，据此构造行业归属的生效区间。
    """
    df = pro.index_classify(level="L1", src="SW2021")
    # TODO(行业 point-in-time)：把 index_member 的 in_date/out_date 映射成
    #   (symbol, industry, in_date, out_date)，替换静态快照。
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default="2026-08-12")
    args = ap.parse_args()

    pro = get_pro()
    basic = fetch_stock_basic(pro)
    members = fetch_csi800_members(pro, args.start, args.end)
    industry = fetch_sw_industry(pro)

    out_dir = Path("data/raw_tushare")
    out_dir.mkdir(parents=True, exist_ok=True)
    basic.to_parquet(out_dir / "stock_basic.parquet", index=False)
    members.to_parquet(out_dir / "csi800_members.parquet", index=False)
    industry.to_parquet(out_dir / "sw_industry.parquet", index=False)
    print(f"stock_basic={len(basic)} csi800_members={len(members)} sw_industry={len(industry)}")


if __name__ == "__main__":
    main()
