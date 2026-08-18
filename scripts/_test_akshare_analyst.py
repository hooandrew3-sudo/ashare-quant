"""测试 akshare 分析师一致预期接口。"""
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import akshare as ak
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)


def test_detail():
    print("=== stock_analyst_detail_em ===")
    try:
        df = ak.stock_analyst_detail_em(symbol="600519")  # 茅台
        print(f"rows={len(df)}, cols={df.columns.tolist()}")
        print(df.head(5).to_string())
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {str(e)[:200]}")


def test_profit_forecast():
    print("\n=== stock_profit_forecast_em ===")
    try:
        df = ak.stock_profit_forecast_em(symbol="600519")
        print(f"rows={len(df)}, cols={df.columns.tolist()}")
        print(df.head(5).to_string())
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {str(e)[:200]}")


if __name__ == "__main__":
    test_detail()
    test_profit_forecast()
