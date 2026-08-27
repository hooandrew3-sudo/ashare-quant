"""因子层：定义、计算、预处理、IC 分析。"""

from quant.factors.analysis import factor_ic_report
from quant.factors.compute import build_panels, compute_all_factors
from quant.factors.definitions import FACTOR_SPECS

__all__ = ["FACTOR_SPECS", "build_panels", "compute_all_factors", "factor_ic_report"]
