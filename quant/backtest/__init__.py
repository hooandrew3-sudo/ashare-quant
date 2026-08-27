"""回测引擎：撮合规则、成本模型、绩效与压力测试。"""

from quant.backtest.cost import CostModel
from quant.backtest.engine import BacktestEngine, BacktestResult
from quant.backtest.stress import run_stress

__all__ = ["CostModel", "BacktestEngine", "BacktestResult", "run_stress"]
