"""极速冒烟测试：synthetic 数据 + 最小配置，跑通全流程。"""

from __future__ import annotations

import sys
from pathlib import Path

# 确保项目根目录可导入
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quant.config import Config
from quant.pipeline import prepare_data, run_research


def main() -> None:
    cfg = Config()
    cfg.data.source = "synthetic"
    # 隔离：synthetic 演示数据写入独立目录，严禁覆盖 data/ 真实数据
    cfg.data.root = Path("data_demo")
    cfg.data.demo.n_stocks = 20
    cfg.data.demo.years = 2
    cfg.model.n_splits = 3
    cfg.model.horizons = [20]
    cfg.model.n_seeds = 1
    cfg.model.seeds = [42]
    cfg.portfolio.top_n = 8
    cfg.portfolio.rebalance_freq = "M"
    cfg.backtest.start = "2023-01-01"
    cfg.backtest.end = "2024-12-31"
    cfg.run.verbose = False

    print("=== 1. 准备数据 ===")
    bundle = prepare_data(cfg)
    print(f"   数据行数: {len(bundle.prices)}")

    print("=== 2. 运行研究流水线 ===")
    result = run_research(cfg, bundle)
    print("run_id:", result["run_id"])
    print("metrics:", result["metrics"])
    print("oos_model:", result["metrics"].get("oos_model"))
    assert "oos_model" in result["metrics"], "缺少 OOS 模型基线度量"
    print("=== 冒烟测试通过 ===")


if __name__ == "__main__":
    main()