"""测试同花顺分析师一致预期 EPS 接口。"""
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import akshare as ak
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)


def test_ths(symbol="600519"):
    print(f"=== stock_profit_forecast_ths({symbol}, 预测年报每股收益) ===")
    try:
        df = ak.stock_profit_forecast_ths(symbol=symbol, indicator="预测年报每股收益")
        print(f"rows={len(df)}, cols={df.columns.tolist()}")
        print(df.to_string())
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {str(e)[:300]}")


def test_ths_indicators():
    print("\n=== 尝试其他 indicator ===")
    for ind in ["预测年报净利润", "预测年报营业收入", "预测年报每股净资产"]:
        try:
            df = ak.stock_profit_forecast_ths(symbol="600519", indicator=ind)
            print(f"indicator={ind}: rows={len(df)}, cols={df.columns.tolist()}")
            if len(df):
                print(df.head(2).to_string())
        except Exception as e:
            print(f"indicator={ind}: ERROR {str(e)[:100]}")


if __name__ == "__main__":
    test_ths()
    test_ths_indicators()
