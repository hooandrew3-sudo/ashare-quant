"""Streamlit 看板：数据状态 / 因子 IC / 最近回测 / 今日信号 / 纸面账户。

启动：streamlit run quant/monitor/dashboard.py -- --artifacts artifacts
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

try:
    import streamlit as st
except ImportError:  # pragma: no cover
    raise SystemExit("需要安装 streamlit：pip install streamlit")


def _latest_dir(root: Path) -> Path | None:
    runs = [p for p in root.iterdir() if p.is_dir() and (p / "metrics.json").exists()]
    return max(runs, key=lambda p: p.stat().st_mtime) if runs else None


def main(artifacts: str = "artifacts", data_root: str = "data") -> None:
    st.set_page_config(page_title="A股量化看板", layout="wide")
    st.title("A股多因子概率决策系统 · 运营看板")
    root = Path(artifacts)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["数据状态", "因子 IC", "回测结果", "今日信号", "纸面账户"]
    )

    with tab1:
        st.subheader("数据 manifest")
        manifest = Path(data_root) / "manifest.json"
        if manifest.exists():
            st.json(json.loads(manifest.read_text(encoding="utf-8")))
        else:
            st.info("暂无数据 manifest（data/ 目录尚未初始化）")

    with tab2:
        st.subheader("因子准入报告")
        ic_files = sorted(root.glob("*/ic_report.csv"))
        if not ic_files:
            st.info("暂无因子 IC 报告，请先运行流水线")
        else:
            df = pd.read_csv(ic_files[-1])
            st.dataframe(df, use_container_width=True)

    with tab3:
        run = _latest_dir(root)
        if run is None:
            st.info("暂无回测产物")
        else:
            st.subheader(f"最近运行: {run.name}")
            metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
            cols = st.columns(4)
            m = metrics.get("metrics", {})
            cards = [
                ("年化收益", m.get("annualized_return")),
                ("最大回撤", m.get("max_drawdown")),
                ("Sharpe", m.get("sharpe")),
                ("月度胜率", m.get("monthly_win_rate")),
            ]
            for col, (label, val) in zip(cols, cards):
                col.metric(label, f"{val:.2%}" if isinstance(val, float) else "-")
            equity = pd.read_parquet(run / "equity.parquet")
            st.line_chart(equity.set_index("date")[["portfolio_value", "benchmark_value"]])
            stress = metrics.get("stress", {})
            if stress:
                st.write("压力测试")
                st.dataframe(pd.DataFrame(stress).T, use_container_width=True)

    with tab4:
        st.subheader("最近调仓信号")
        sig_dir = root / "signals"
        if sig_dir.exists():
            files = sorted(sig_dir.glob("signals_*.csv"))
            if files:
                st.dataframe(pd.read_csv(files[-1]).head(50), use_container_width=True)
            else:
                st.info("暂无信号文件")
        else:
            st.info("暂无信号文件（先运行 --step live）")

    with tab5:
        st.subheader("纸面交易（每日模拟验证）")
        hist = root / "paper" / "history.csv"
        if not hist.exists():
            st.info("暂无纸面账户历史，请先运行 --step paper（或每日任务 --step daily）")
        else:
            df = pd.read_csv(hist, parse_dates=["date"]).sort_values("date")
            st.dataframe(df, use_container_width=True)
            cols = st.columns(4)
            last = df.iloc[-1]
            cols[0].metric("组合市值", f"{last['portfolio_value']:,.0f}")
            cols[1].metric("累计收益", f"{last['return_pct']:.2%}")
            cols[2].metric("累计费用", f"{last['total_fees']:,.0f}")
            cols[3].metric("最新交易日", str(last["date"].date()))
            chart = pd.DataFrame(
                {
                    "组合(归一)": df["portfolio_value"] / df["portfolio_value"].iloc[0],
                },
                index=df["date"],
            )
            if df["benchmark_close"].notna().any():
                bench0 = df["benchmark_close"].dropna().iloc[0]
                chart["基准(归一)"] = df["benchmark_close"] / bench0
            st.line_chart(chart)
            orders = root / "paper" / "orders.csv"
            if orders.exists():
                st.subheader("订单流水")
                st.dataframe(pd.read_csv(orders), use_container_width=True)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--data-root", default="data")
    args = ap.parse_args()
    main(args.artifacts, args.data_root)
