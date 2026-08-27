"""每日运营调度：增量同步 → 质量校验 → 因子 → 最新模型推理 → 信号 → 纸面调仓 → 通知。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from quant.config import Config
from quant.data.quality import check_benchmark, check_prices
from quant.data.storage import Storage
from quant.monitor.alerts import build_notifier
from quant.pipeline import live_signals, prepare_data
from quant.utils import setup_logging


def run_daily(cfg: Config, output_root: str = "artifacts") -> dict:
    """盘后任务：同步 → 质量校验 → 信号 → 纸面调仓 → 通知（幂等，周末/节假日自动跳过）。"""
    log = setup_logging(cfg.run.verbose)
    notifier = build_notifier()
    try:
        log.info("===== 每日任务开始 %s =====", datetime.now().isoformat(timespec="seconds"))
        bundle = prepare_data(cfg)
        rep_p = check_prices(bundle.prices)
        rep_b = check_benchmark(bundle.benchmark)
        if not rep_p.ok or not rep_b.ok:
            raise RuntimeError(f"数据质量不合格: {rep_p.summary()}; {rep_b.summary()}")

        latest = pd.Timestamp(bundle.prices["date"].max())
        # 每日分析师一致预期快照采集（累积成修正序列，供 consensus_revision 因子使用）
        try:
            from quant.data.consensus_ths import fetch_consensus_snapshot, save_snapshot

            top_syms = (
                bundle.prices[bundle.prices["date"] >= (latest - pd.Timedelta(days=365))]
                .groupby("symbol")["amount"].mean()
                .sort_values(ascending=False).head(200).index.tolist()
            )
            snapshot = fetch_consensus_snapshot(top_syms, verbose=False, pause_sec=0.1)
            if not snapshot.empty:
                save_snapshot(snapshot, cfg.data.root)
                log.info("一致预期快照更新: %d 只", snapshot["symbol"].nunique())
        except Exception as exc:  # noqa: BLE001
            log.warning("一致预期快照采集失败: %s", exc)

        paper_state = Path(output_root) / "paper" / "state.json"
        last_paper_date = None
        if paper_state.exists():
            last_paper_date = json.loads(paper_state.read_text(encoding="utf-8")).get("date")
        if last_paper_date == str(latest.date()):
            msg = f"最新数据日 {latest.date()} 已执行过，跳过重复运行（周末/节假日无需执行）"
            log.info(msg)
            notifier.send("每日任务：无新交易日", msg)
            return {"skipped": True, "latest": str(latest.date())}

        signals = live_signals(cfg, output_root)
        # 因子覆盖度哨兵：数据源失效导致因子静默退化为 0 时告警
        from quant.monitor.coverage import check_factor_coverage

        cov_alerts = check_factor_coverage(Path(output_root) / "signals", cfg)
        for alert in cov_alerts:
            notifier.send("因子覆盖度告警", alert)
        top = signals.head(min(cfg.portfolio.top_n, len(signals)))
        lines = "\n".join(
            f"{r['symbol']}: {r['score']:.4f}" for _, r in top.iterrows()
        )
        from quant.execution.paper_runner import run_paper

        paper = run_paper(cfg, output_root, signals=signals)
        notifier.send(
            f"每日信号 + 纸面调仓 {signals['date'].max().date()}",
            f"候选 {len(signals)} 只\nTop {len(top)}:\n{lines}\n\n"
            f"纸面账户: 市值 {paper['portfolio_value']:,.0f} | 现金 {paper['cash']:,.0f} | "
            f"持仓 {len(paper['positions'])} 只 | 订单 {paper['orders']} | "
            f"累计费用 {paper['total_fees']:,.2f}",
        )
        log.info("每日信号与纸面调仓完成")
        return {"signals": signals, "top": top, "paper": paper}
    except Exception as exc:  # noqa: BLE001
        log.exception("每日任务失败")
        notifier.send("每日任务失败", str(exc))
        raise
