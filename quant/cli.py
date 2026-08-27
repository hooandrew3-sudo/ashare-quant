"""命令行入口。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from quant.config import load_config
from quant.utils import process_lock


def _print_summary(res: dict) -> None:
    m = res.get("metrics", {})
    print("\n===== 回测摘要 =====")
    for k in ("annualized_return", "max_drawdown", "sharpe", "calmar",
              "monthly_win_rate", "excess_return", "information_ratio"):
        if k in m and m[k] is not None:
            v = m[k]
            if isinstance(v, float) and k in ("annualized_return", "max_drawdown",
                                              "monthly_win_rate", "excess_return"):
                v = f"{v:.2%}"
            print(f"{k:>22}: {v}")
    print(f"{'报告':>22}: {res.get('report', '')}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="A股多因子概率决策系统")
    parser.add_argument("--demo", action="store_true", help="用合成数据跑通全流程")
    parser.add_argument("--config", type=str, default=None, help="YAML 配置文件")
    parser.add_argument("--source", type=str, default=None, help="synthetic|baostock|parquet")
    parser.add_argument("--symbols", type=str, default=None, help="baostock 代码列表(逗号分隔)")
    parser.add_argument("--universe", type=str, default=None,
                        choices=["manual", "csi800", "all"], help="baostock 股票池")
    parser.add_argument("--start", type=str, default=None, help="同步起始日期 YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None, help="同步结束日期 YYYY-MM-DD")
    parser.add_argument("--full", action="store_true", help="强制全量同步（忽略增量）")
    parser.add_argument("--root", type=str, default=None, help="数据目录覆盖（默认 data）")
    parser.add_argument("--step", type=str, default="all",
                        choices=["all", "data", "sync", "daily", "factors", "model",
                                 "backtest", "report", "live", "paper", "fundamentals",
                                 "sentiment", "universe-history", "industry", "cashflow"],
                        help="执行阶段")
    parser.add_argument("--output", type=str, default="artifacts", help="产物根目录")
    parser.add_argument("--force", action="store_true", help="强制重算")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.source:
        cfg.data.source = args.source
    if args.universe:
        cfg.data.sync.universe = args.universe
    if args.start:
        cfg.data.sync.start = args.start
    if args.end:
        cfg.data.sync.end = args.end
    if args.full:
        cfg.data.sync.incremental = False
    if args.root:
        cfg.data.root = Path(args.root)
    symbols = args.symbols.split(",") if args.symbols else None

    # 进程级互斥锁：计划任务补跑 + 手工运行并发时会双写 paper state /
    # 并发增量同步，属生产事故源。锁粒度 = 数据目录。
    with process_lock(cfg.data.root):
        return _dispatch(args, cfg, symbols, parser)


def _dispatch(args, cfg, symbols, parser) -> int:
    from quant.data.storage import Storage
    from quant.pipeline import live_signals, prepare_data, run_research

    # --demo 必须最先处理：此前排在 step 分支之后，当真实数据已存在时
    # （data/processed/prices 非空）合成数据被静默跳过，直接用真实数据跑研究
    if args.demo:
        cfg.data.source = "synthetic"
        # demo 与真实数据隔离，避免合成数据覆盖 data/processed
        cfg.data.root = Path("artifacts/demo_data")
        storage = Storage(cfg.data.root)
        if not storage.has("prices") or args.force:
            prepare_data(cfg)
        bundle = storage.load_bundle()
        res = run_research(cfg, bundle, args.output, force=args.force)
        _print_summary(res)
        return 0

    if args.step == "live":
        signals = live_signals(cfg, args.output)
        print(signals.head(30).to_string(index=False))
        return 0

    if args.step == "paper":
        from quant.execution.paper_runner import run_paper

        state = run_paper(cfg, args.output)
        print(json.dumps(state, ensure_ascii=False, indent=2, default=str))
        return 0

    if args.step == "sync":
        prepare_data(cfg, symbols=symbols)
        print("数据同步完成")
        return 0

    if args.step == "daily":
        from quant.scheduler import run_daily

        run_daily(cfg, args.output)
        return 0

    if args.step == "fundamentals":
        from quant.pipeline import prepare_fundamentals

        fund = prepare_fundamentals(cfg)
        print(f"财务数据完成: {len(fund)} 条, {fund['symbol'].nunique()} 只")
        return 0

    if args.step == "cashflow":
        from quant.pipeline import prepare_cashflow

        cf = prepare_cashflow(cfg)
        print(f"现金流数据完成: {len(cf)} 条, {cf['symbol'].nunique()} 只")
        return 0

    if args.step == "industry":
        from quant.pipeline import prepare_industry

        ind = prepare_industry(cfg)
        print(f"行业数据完成: {len(ind)} 条, {ind['symbol'].nunique()} 只")
        return 0

    if args.step == "sentiment":
        from quant.pipeline import prepare_sentiment

        senti = prepare_sentiment(cfg)
        print(f"情绪数据完成: {len(senti)} 条, {senti['symbol'].nunique()} 只")
        return 0

    if args.step == "universe-history":
        from quant.pipeline import compute_history_members

        delta = compute_history_members(cfg)
        print(f"历史成员增量: {len(delta)} 只 → data/cache/history_members_delta.json")
        return 0

    if args.step in ("all", "data", "factors", "model", "backtest", "report"):
        if args.step == "data":
            prepare_data(cfg, symbols=symbols)
            print("数据准备完成")
            return 0
        storage = Storage(cfg.data.root)
        if args.step == "all" or (args.step in ("factors", "model", "backtest", "report")):
            # 全流程：数据(如缺失) → 研究
            if not storage.has("prices"):
                prepare_data(cfg, symbols=symbols)
            from quant.data.storage import DataBundle

            bundle = storage.load_bundle()
            res = run_research(cfg, bundle, args.output, force=args.force)
            _print_summary(res)
            if args.step == "report":
                print(f"报告: {res['report']}")
            return 0

    parser.print_help()
    # 参数组合不匹配任何分支时返回非零：此前返回 0，
    # 计划任务视角是"成功"，失败被静默吞掉
    return 2


if __name__ == "__main__":
    sys.exit(main())
