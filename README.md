# A股多因子概率决策系统 · 生产级可执行版

基于「多因子 + 概率输出 + Walk-Forward 回测 + 严格风控」的个人可落地量化系统。
设计目标（终审重标定）：年化 ≥10%、最大回撤 ≤12%、信息比率 ≥0.7、月度正收益概率 ≥65%；
定位为「低 beta 防御型绝对收益增强器」。原「年化 12-16% / 回撤 6-8% / IR 1.2」目标经长窗口实证
不可达，完整结论见 [docs/FINAL_VERDICT.md](docs/FINAL_VERDICT.md)。

> ⚠️ 本系统仅供研究与个人投资决策辅助，不构成投资建议。合成数据回测收益不代表真实 Alpha。

## 功能概览

- **数据层**：合成数据（零依赖 demo）/ baostock 真实日线接入；Parquet 版本化存储 +
  manifest 指纹；14 项数据质量校验（OHLC、涨跌停一致性、停牌、幸存者偏差）；
- **因子层**：10 个核心因子（进攻 6 + 防御 4）；去极值/标准化/行业中性化；
  Rank IC / ICIR / 分层收益 / 衰减分析与因子准入/退出；
- **模型层**：LightGBM（或 sklearn GBDT 兜底）二分类，预测未来 20 日跑赢中证800；
  Walk-Forward（3 年训练 / 6 月测试，5 折）防过拟合；模型注册表（数据指纹 + 参数 +
  区间 + 指标全存档）；
- **组合与风控**：top-30 等权、单票 ≤5%、单行业 ≤25%、换手 ≤40%/期、
  市场状态机（MA20/MA120/MA250 三档仓位）、波动率目标、-12% 止损、涨停不买/跌停顺延；
- **回测**：事件驱动引擎，T+1 / 涨跌停 / 停牌 / 整手 / 双边成本 0.25%；
  压力测试（2015 股灾、2018 熊市、2024 小盘踩踏）；
- **执行层**：Broker 抽象 + OMS（幂等/重试/熔断）+ PaperBroker（纸面交易）+
  miniQMT(XtQuant) 实盘适配器（需券商开通）；
- **每日模拟验证**：每日增量同步 → 信号 → 纸面调仓（T+1、佣金/印花税/滑点与回测同口径）
  → 账户净值/订单流水落盘，可与基准对比；
- **监控**：Streamlit 看板（数据状态 / 因子 IC / 回测 / 今日信号 / 纸面账户）、HTML 自包含报告。

## 快速开始（零依赖 demo）

```bash
pip install -e .[model,report,dev]   # 或最小依赖：pip install pandas scikit-learn PyYAML fastparquet
python run_pipeline.py --demo --force
```

`--demo` 生成 200 只合成股票 × 4 年日线，跑通「数据 → 因子 → 模型 → Walk-Forward →
组合 → 回测 → 压力测试 → HTML 报告」全流程。报告输出到 `artifacts/reports/`。

```bash
python run_pipeline.py --step live    # 用已注册模型生成今日调仓信号
python run_pipeline.py --config configs/real.yaml --step paper  # 按最新信号执行纸面调仓（模拟验证）
python -m pytest tests                # 运行测试套件
streamlit run quant/monitor/dashboard.py -- --artifacts artifacts
```

每日自动运行（数据同步 + 信号 + 纸面调仓 + 通知）：
`powershell -ExecutionPolicy Bypass -File scripts/setup_windows_schedule.ps1`，
注册 Windows 计划任务 `AshareQuantDaily`（每交易日 18:30，日志写入 `daily.log`）。

## 接入真实数据（baostock）

```bash
pip install baostock
python run_pipeline.py --source baostock \
  --symbols sh.600519,sz.000001,sh.600000,sz.300750 \
  --config configs/default.yaml
```

`--source parquet` 可直接读取 `data/processed/` 中已存在的 Parquet 数据集
（支持 Tushare SQLite 导入后自建 Parquet）。

## 真实数据回测（CSI800 + 退市股）

已同步中证800 当前成分 + 2021 年后退市股票共 **1007 只**，并回填至 **2018-01**
（2018-01 ~ 2026-08，约 180 万行日线）。长窗口 Walk-Forward 样本外（2019-02 ~ 2026-08，
15 折 × 6 月）终审结果：

| 指标 | 终审结果 | 重标定验收 |
| :--- | :--- | :--- |
| 年化收益 / 最大回撤 | 24.6% / -9.85% | ✅（收益为偏差上限，不可外推） |
| Sharpe / Calmar | 2.13 / 2.50 | ✅ |
| 信息比率 | 0.94 | ✅（≥0.7） |
| 月度正收益概率 | 77.3% | ✅（≥65%） |
| 单边年换手 | ≈0.9x | ✅（≤2x） |
| 复合因子 | t=8.1 | ✅（t≥5） |
| 压力测试 | 2018/2022 熊市均盈利 | ✅ |

> ⚠️ **诚实结论**：这是低 beta 防御型风格策略（价值 + 低波 + 小市值 + 规避 MAX 的截面倾斜），
> 不是高 IR 预测型 alpha。模型 AUC≈0.5 无增量；信号 regime 依赖（2024-2025 失效）；
> 幸存者偏差残余未完全消除。完整 A/B 记录与验收基线重标定见
> [docs/FINAL_VERDICT.md](docs/FINAL_VERDICT.md)。

## 项目结构

```
ashare/
├── run_pipeline.py          # CLI 入口（python run_pipeline.py --demo）
├── configs/default.yaml     # 默认配置（可哈希存档）
├── quant/
│   ├── config.py            # 配置 dataclass + YAML 加载
│   ├── data/                # storage / synthetic / baostock_loader / quality
│   ├── factors/             # 10 因子定义 / 计算 / 预处理 / IC 分析
│   ├── model/               # 标签 / Walk-Forward / 模型注册表
│   ├── portfolio/           # 市场状态机 / 组合构建 / 风控
│   ├── backtest/            # 回测引擎 / 撮合 / 成本 / 压力测试
│   ├── metrics/             # 绩效指标
│   ├── report/              # 自包含 HTML 报告
│   ├── execution/           # Broker 抽象 / OMS / Paper / QMT 适配器
│   └── monitor/             # Streamlit 看板
├── tests/                   # 数据/因子/模型/回测/流水线测试
├── data/                    # raw / processed / manifest（gitignore）
├── artifacts/               # 模型 / 回测 / 报告
└── docs/PRODUCTION_SPEC.md  # 生产级设计说明书（含 GitHub 案例借鉴）
```

## 防过拟合纪律（比代码更重要）

1. 新因子先过 `factor_ic_report`：Rank IC ≥0.03 且 ICIR ≥0.3 才准入；
2. 因子总数控制在 15 以内；两两相关 >0.7 只留一个；
3. 所有结论以 Walk-Forward 样本外结果为准，样本内结果不用于决策；
4. 参数搜索组合 >20 组时，结论按多重检验打折；
5. 上线流程：demo → 真实数据 3 年 → 纸面 3 个月 → 小资金 6 个月 → 按验收基准放量。

## 主要参考项目

- [Microsoft Qlib](https://github.com/microsoft/qlib) — AI 量化全流程与 Alpha158 因子体系
- [ashare-lowfreq-research](https://github.com/cyecho-io/ashare-lowfreq-research) — A 股低频研究台（Parquet/分数回测/模拟执行）
- [VeighNa (vnpy)](https://github.com/vnpy/vnpy) — 事件驱动与交易网关抽象
- [EasyXT](https://github.com/quant-king299/EasyXT) — miniQMT(xtquant) 封装与多数据源降级
- [Hikyuu](https://github.com/fasiondog/hikyuu) — A 股适配的高性能回测
- [Freqtrade](https://github.com/freqtrade/freqtrade) — 生产部署与告警实践
- [Alphalens-reloaded](https://github.com/stefan-jansen/alphalens-reloaded) — 因子绩效分析

完整设计细节见 [docs/PRODUCTION_SPEC.md](docs/PRODUCTION_SPEC.md)。
