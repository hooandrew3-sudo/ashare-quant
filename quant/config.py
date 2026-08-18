"""配置体系：dataclass + YAML 加载与校验。"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_type_hints

import yaml


@dataclass
class RunConfig:
    seed: int = 42
    verbose: bool = True


@dataclass
class UniverseConfig:
    min_avg_amount: float = 50_000_000
    min_price: float = 1.0      # 仙股过滤：收盘价中位数低于该值的股票不进因子横截面
    exclude_st: bool = True
    min_list_days: int = 60


@dataclass
class DemoConfig:
    n_stocks: int = 200
    years: int = 4
    start: str = "2021-01-01"


@dataclass
class SyncConfig:
    universe: str = "manual"  # manual | csi800 | all
    start: str = "2021-01-01"
    end: str = ""
    incremental: bool = True


@dataclass
class DataConfig:
    source: str = "synthetic"
    root: Path = Path("data")
    benchmark: str = "000906.SH"
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    demo: DemoConfig = field(default_factory=DemoConfig)
    sync: SyncConfig = field(default_factory=SyncConfig)


@dataclass
class FactorConfig:
    top_n: int = 10
    min_ic: float = 0.03
    min_icir: float = 0.3
    min_t_stat: float = 2.0
    winsor: float = 0.05
    neutralize_industry: bool = True
    neutralize_size: bool = True
    composite: bool = True
    composite_n: int = 5
    composite_corr_max: float = 0.6
    composite_require_decay: bool = True
    composite_weight: str = "icir"  # icir | t | equal


@dataclass
class ModelConfig:
    name: str = "auto"  # lightgbm | gbt | auto
    horizon: int = 20
    horizons: list[int] = field(default_factory=lambda: [20])  # 多周期标签集成
    n_seeds: int = 1             # 多种子集成（降模型方差）
    seeds: list[int] = field(default_factory=lambda: [42, 2024, 7])
    label_mode: str = "binary"
    top_quantile: float = 0.3
    train_years: int = 3
    test_months: int = 6
    n_splits: int = 5
    selection_mode: str = "model"  # model(ML概率) | composite(复合因子直接排序) | hybrid(两者融合)
    params: dict[str, Any] = field(
        default_factory=lambda: {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "min_child_samples": 30,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        }
    )
    gbt_params: dict[str, Any] = field(
        default_factory=lambda: {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 3,
        }
    )


@dataclass
class RegimeConfig:
    fast: int = 20
    slow: int = 120
    long: int = 250
    levels: list[float] = field(default_factory=lambda: [1.0, 0.5, 0.3])


@dataclass
class PortfolioConfig:
    top_n: int = 30
    max_weight: float = 0.05
    max_industry: float = 0.25
    target_vol: float = 0.10
    vol_window: int = 20
    cash_rate: float = 0.02
    turnover_cap: float = 0.20
    max_turnover_annual: float = 4.0  # 年化换手硬上限（超过则自动跳过调仓）
    stickiness: float = 0.80
    stop_loss: float = 0.12
    band: float = 0.25
    exposure_step: float = 0.25  # 状态机仓位每期最大变动，平滑换手
    min_overlap: float = 0.85    # 新组合与当前持仓重叠 ≥ 该比例时跳过调仓
    rebalance_freq: str = "M"    # M=月度 | 2M=双月 | Q=季度
    beta_target: float = 0.5     # 滚动 beta 目标（指数增强式暴露调节）
    beta_window: int = 60
    beta_scale_max: float = 1.5
    signal_health_enabled: bool = False    # 信号失效开关：近期选股超额为负时下调仓位
    signal_health_window: int = 126        # 信号健康度回看交易日数（约半年）
    signal_health_floor: float = 0.5       # 信号失效时最低目标仓位系数
    signal_health_scale: float = 0.10      # 近期超额 -scale 时健康度降到 floor
    smallcap_regime_enabled: bool = False  # 小盘辅助择时：中证1000 跌破 MA250 时降仓
    smallcap_index: str = "000852.SH"
    smallcap_long: int = 250
    smallcap_floor: float = 0.5
    weight_method: str = "equal"  # equal | risk_budget
    min_avg_amount: float = 50_000_000
    max_smallcap_ratio: float = 0.30  # 组合中小市值股票（成交额 bottom 30%）数量占比上限
    cvar_enabled: bool = False
    cvar_threshold: float = -0.005
    cvar_lookback: int = 252
    cvar_alpha: float = 0.05
    regime: RegimeConfig = field(default_factory=RegimeConfig)


@dataclass
class BacktestConfig:
    start: str = "2022-01-01"
    end: str = "2024-12-31"
    initial_cash: float = 1_000_000
    commission_bp: float = 2.5
    stamp_bp: float = 5.0
    transfer_bp: float = 0.1
    slippage_bp: float = 5.0
    min_commission: float = 5.0
    lot_size: int = 100
    postpone_max_days: int = 5
    slippage_model: str = "fixed"       # fixed | adaptive（按订单额/日均成交额动态冲击）
    slippage_cap_bp: float = 20.0
    slippage_impact_coef: float = 5.0
    circuit_breaker_enabled: bool = False   # 尾部熔断：单日亏损超阈值次日等比减仓
    circuit_breaker_daily_dd: float = 0.03
    circuit_breaker_scale: float = 0.5


@dataclass
class StressConfig:
    scenarios: dict[str, list[str]] = field(
        default_factory=lambda: {
            "crash_2015": ["2015-06-15", "2015-09-30"],
            "bear_2018": ["2018-01-01", "2018-12-31"],
            "smallcap_2024": ["2024-01-01", "2024-02-29"],
            "rally_2024": ["2024-09-24", "2025-02-28"],
        }
    )


@dataclass
class ReportConfig:
    output_dir: Path = Path("artifacts/reports")
    title: str = "A股多因子概率决策系统 · 回测报告"


@dataclass
class Config:
    run: RunConfig = field(default_factory=RunConfig)
    data: DataConfig = field(default_factory=DataConfig)
    factors: FactorConfig = field(default_factory=FactorConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    stress: StressConfig = field(default_factory=StressConfig)
    report: ReportConfig = field(default_factory=ReportConfig)

    def fingerprint(self) -> str:
        """配置内容哈希，用于 run_id 与产物追溯。"""
        import hashlib

        blob = yaml.safe_dump(to_dict(self), sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:12]

    def validate(self) -> None:
        """工业级配置校验：防止 synthetic 演示数据污染真实数据目录等事故。

        触发条件：data.source == 'synthetic' 且 data.root 不含 demo/test/tmp 标记。
        """
        root_str = str(self.data.root).lower()
        if self.data.source == "synthetic" and not any(
            marker in root_str for marker in ("demo", "test", "tmp", "temp")
        ):
            raise ValueError(
                "数据隔离约束：synthetic 演示数据严禁写入真实数据目录 "
                f"(root={self.data.root})。请将 data.root 改为含 demo/test 的路径，"
                "例如 data_demo。"
            )
        _validate_enum("data.source", self.data.source, {"synthetic", "baostock", "parquet"})
        _validate_enum(
            "data.sync.universe",
            self.data.sync.universe,
            {"manual", "csi800", "csi800_delisted", "all"},
        )
        _validate_enum(
            "model.selection_mode",
            self.model.selection_mode,
            {"model", "composite", "hybrid"},
        )
        _validate_enum(
            "model.label_mode",
            self.model.label_mode,
            {"binary", "top_quantile", "quantile_contrast", "regression"},
        )
        _validate_enum(
            "model.name", self.model.name, {"lightgbm", "gbt", "auto"}
        )
        _validate_enum(
            "factors.composite_weight",
            self.factors.composite_weight,
            {"icir", "t", "equal"},
        )
        _validate_enum(
            "portfolio.weight_method",
            self.portfolio.weight_method,
            {"equal", "risk_budget"},
        )
        _validate_enum(
            "portfolio.rebalance_freq",
            self.portfolio.rebalance_freq,
            {"M", "2M", "Q"},
        )
        _validate_enum(
            "backtest.slippage_model",
            self.backtest.slippage_model,
            {"fixed", "adaptive"},
        )
        for name, value in (
            ("portfolio.top_n", self.portfolio.top_n),
            ("model.n_splits", self.model.n_splits),
            ("model.train_years", self.model.train_years),
            ("model.test_months", self.model.test_months),
        ):
            if value <= 0:
                raise ValueError(f"配置 {name} 必须为正数，当前为 {value}")


def _build(section, data: dict[str, Any]):
    """按 dataclass 字段递归构造；未知键直接报错（防拼写错误静默生效）。"""
    kwargs: dict[str, Any] = {}
    hints = get_type_hints(section)
    unknown = [k for k in (data or {}) if k not in hints]
    if unknown:
        raise ValueError(
            f"配置段 {section.__name__} 包含未知键: {sorted(unknown)}，"
            "请检查拼写（此前版本会静默忽略，属生产隐患）。"
        )
    for f in dataclasses.fields(section):
        if f.name not in data:
            continue
        value = data[f.name]
        ftype = hints.get(f.name, f.type)
        if dataclasses.is_dataclass(ftype):
            kwargs[f.name] = _build(ftype, value)
        elif ftype is Path and isinstance(value, str):
            kwargs[f.name] = Path(value)
        else:
            kwargs[f.name] = value
    return section(**kwargs)


def load_config(path: str | Path | None = None) -> Config:
    """从 YAML 加载配置；未提供路径时使用内置默认值。"""
    if path is None:
        return Config()
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return _build(Config, raw)


def _validate_enum(name: str, value: Any, allowed: set[str]) -> None:
    if value not in allowed:
        raise ValueError(
            f"配置 {name}={value!r} 非法，允许值: {sorted(allowed)}"
        )


def to_dict(cfg: Config) -> dict[str, Any]:
    """将配置转回纯 dict（用于存档/打印）。"""
    return _convert(dataclasses.asdict(cfg))


def _convert(value: Any) -> Any:
    """递归将 Path 等非 YAML 可序列化对象转为基础类型。"""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _convert(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_convert(v) for v in value]
    return value
