"""检查 akshare 接口签名。"""
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import akshare as ak
import inspect

for name in ["stock_analyst_detail_em", "stock_analyst_rank_em", "stock_profit_forecast_em", "stock_profit_forecast_ths"]:
    fn = getattr(ak, name)
    try:
        sig = inspect.signature(fn)
        print(f"{name}{sig}")
    except Exception as e:
        print(f"{name}: {e}")
