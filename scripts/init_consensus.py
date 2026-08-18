"""初始化分析师一致预期快照：拉取股票池当前一致预期 EPS 并保存。

用法：python scripts/init_consensus.py [n_symbols]
- n_symbols 默认 300（核心流动性池）；采集约需 5-10 分钟
- 之后每日调用 quant/data/consensus_ths.py 的 save_snapshot 累积
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from quant.data.consensus_ths import fetch_consensus_snapshot, save_snapshot
from quant.data.storage import Storage


def main(n_symbols: int = 300) -> None:
    storage = Storage("data")
    prices = storage.load("prices")
    prices["date"] = pd.to_datetime(prices["date"])

    # 用 2025-2026 流动性选池（当前快照，采集起点）
    amt = prices[prices["date"] >= "2025-01-01"].groupby("symbol")["amount"].mean().sort_values(ascending=False)
    keep = list(amt.head(n_symbols).index)
    print(f"采集股票池: {len(keep)} 只（2025 年日均额 top{n_symbols}）", flush=True)

    snapshot = fetch_consensus_snapshot(keep, verbose=True, pause_sec=0.15)
    print(f"快照完成: {len(snapshot)} 行, {snapshot['symbol'].nunique() if len(snapshot) else 0} 只有机构覆盖", flush=True)

    if snapshot.empty:
        print("无数据，退出", flush=True)
        return

    save_snapshot(snapshot, "data")

    # 摘要
    latest = snapshot[snapshot["year"] == snapshot["year"].max()]
    print(f"\n一致预期覆盖摘要（最近年度 {latest['year'].iloc[0]}）:", flush=True)
    print(f"  有机构覆盖: {latest['symbol'].nunique()} 只", flush=True)
    print(f"  平均机构数: {latest['n_institutions'].mean():.0f}", flush=True)
    print(f"  一致预期EPS均值: {latest['eps_mean'].mean():.2f}", flush=True)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    main(n)
