# 工业级生产修复记录（2026-08-18）

> 依据《A股量化项目 · 工业级生产审核意见》执行的第一批修复。
> 验证：`python -m pytest tests` 全绿；baostock 实连 6 只股票增量同步通过。

## P0

### P0-1 每日生产流水线恢复（baostock factor 字段）
- 问题：`DAILY_FIELDS` 含 `factor`，baostock 接口 2026-08 起返回
  `指标不存在:factor`，导致每日 18:30 增量同步对全部股票失败（122 ERROR）。
- 修复：[baostock_sync.py](../quant/data/baostock_sync.py) 移除 `factor`，
  `adj_factor` 保持常量 1.0 占位（schema 稳定）。
- 验证：实连 `600519/000001/300750/688981/002032/600000`，42 行、0 错误、
  2026-08-18 数据可拉取。

### P0-2 项目版本控制基线
- `git init` + 基线提交；`.gitignore` 覆盖 `data/`、`data_subset/`、
  `data_demo/`、`artifacts/`、`.env`、日志与 scratch 文件；
  `.gitattributes` 统一行尾，避免 CRLF 噪音。

## P1

### P1-1 回测撮合：幽灵现金 / 同日重复卖出
- 修复：[engine.py](../quant/backtest/engine.py) 卖出前校验持仓与可卖数量，
  不足则丢弃并记账；卖出同时扣减 `available`；同一标的因止损/熔断卖出后
  当日禁止回补（`no_buy_after_stop`）。
- 新增测试：`test_duplicate_sell_no_phantom_cash`、
  `test_no_buy_after_stop_same_day`。

### P1-3 撮合层：买入顺延 + 开盘一字板判定
- 修复：
  - [engine.py](../quant/backtest/engine.py) 停牌/无行情买入按
    `postpone_max_days` 顺延重试（此前静默丢弃），超期丢弃并记录；
  - [baostock_sync.py](../quant/data/baostock_sync.py) 新增
    `is_limit_up_open / is_limit_down_open`，按 open vs preclose 计算，
    并按板块 + 时间演进区分涨跌停比例（主板 10%、ST 5%、创业板 2020-08-24
    起 20%、科创板 20%、北交所 30%）；
  - [fills.py](../quant/backtest/fills.py) 撮合改用开盘一字板标记，
    缺失/NaN 安全回退，修复 `bool(np.nan)` 误判。
- 新增测试：`test_fill_open_limit_rules`、`test_buy_postponed_retries`。

### P1-5 risk_budget 权重不再被丢弃
- 修复：[risk_budget.py](../quant/portfolio/risk_budget.py) 行业约束改为在
  真实权重上投影（剔除超限行业权重最小成员后归一化），此前行业表存在时
  权重被等权覆盖导致 risk_budget 静默退化为等权。

### P1-6 研究 ↔ 实盘推理一致性
- 修复：
  - [pipeline.py](../quant/pipeline.py) `live_signals` 强制校验模型配置指纹，
    数据指纹变化告警；
  - [registry.py](../quant/model/registry.py) 持久化部署校准器
    `calibrator.joblib`，live 推理与研究流程一致地先校准再排序；
  - `run_research` 入口统一 `cfg.validate()`。

### P1-7 数据缓存 TTL
- 修复：[baostock_sync.py](../quant/data/baostock_sync.py) `stock_basic` 与
  `industry` 快照缓存 24h 过期重拉，退市/新上市/行业变更不再被无限期冻结。

### P1-9 测试套件恢复全绿
- 修复：[train.py](../quant/model/train.py) `ensemble_scores` 对缺失标签列
  容错；`tests/test_model.py::test_ensemble_scores` 恢复通过。

## P2

- **配置 fail-fast**：[config.py](../quant/config.py) 未知键直接报错（此前
  静默忽略），枚举字段（source/label_mode/selection_mode/weight_method/
  rebalance_freq/slippage_model/composite_weight）启动校验。
- **换手口径统一**：[performance.py](../quant/metrics/performance.py) 与
  [engine.py](../quant/backtest/engine.py) `_turnover_breached` 统一为
  “单边年化换手 = (买额+卖额)/2 / 平均净值 / 年数”（此前是全程双边累计，
  阈值实际放大 2 倍）。
- **压力测试门禁**：[stress.py](../quant/backtest/stress.py) 覆盖 <30 交易日
  的场景标记 `ok=False`（如 2015 场景在当前数据上为空跑，不再算“通过”），
  报告增加状态列。
- **consensus_revision 口径**：[definitions.py](../quant/factors/definitions.py)
  改为对齐交易日历后按 30 个交易日位移（此前为行位移，周频快照语义漂移）。
- **sentiment 走 HTTPS**、paper `settle` 统一 Timestamp、dashboard 路径修复、
  report 增加 OOS 指标与压力状态列、计划任务日志强制 UTF-8。

## 待办（下轮）

1. P1-2 数据口径：不复权价 + 复权因子存储、基准改全收益口径，重跑全量回测；
2. P1-4 paper 与回测共用同一决策管线（band/regime/止损/exposure）；
3. P1-8 标签对齐 T+1 开盘执行口径；
4. P1-10 OMS 盘前风控 + QMT 订单对账；
5. P1-7 行业中性化接入历史行业分类（或敏感性分析）。

---

## 当日故障处置补录（2026-08-18 晚）

### 18:30 计划任务失败复盘
- 直接原因：任务启动时（18:30）代码仍是修复前版本，800 只股票 × 3 次重试
  空转 7514s 后以"未返回任何数据"失败；修复在 20:00 后才落地。
- 处置：修复后用生产配置完整补跑：增量同步（800 只 / 4000 行 / 0 错误）→
  生产模型重训 → 信号 → 纸面调仓 → 通知，全部通过
  （`signals_2026-08-18.csv`、paper 账户 39 笔订单全部成交）。

### 连带修复
1. **策略指纹门禁**：[config.py](../quant/config.py) 新增
   `strategy_fingerprint()`（仅锁 factors/model/portfolio）；live 推理改用
   该指纹，每日任务的 `--source/--universe` 数据层覆盖不再误触发"需重训"；
   旧模型（无 strategy_fingerprint）降级为告警。
2. **consensus 透视崩溃**：[compute.py](../quant/factors/compute.py)
   `build_panels` 快照表按 `as_of_date` 透视（此前固定按 `date`，一旦
   consensus 数据累积即 KeyError）。
3. **flag 列类型崩溃**：[baostock_sync.py](../quant/data/baostock_sync.py) +
   [storage.py](../quant/data/storage.py) 增量合并/读取时把涨跌停/停牌/ST
   标记列规范化为 bool（旧数据缺新列导致 object 混合类型，fastparquet
   落盘失败）。
4. **训练性能**：[train.py](../quant/model/train.py) + [pipeline.py](../quant/pipeline.py)
   特征宽表只构建一次，多周期 × 多种子复用（此前 12 次重复 pivot 百万行）。
5. **产物完整性**：live_signals 只认含 meta.json 的完整模型（训练中断的
   半成品不得参与推理）。

### 生产模型基线（real.yaml 指纹 797ba5f8960f）
- 60 folds（4 周期 × 3 种子 × 5 折）；OOS AUC 0.5007、Rank IC 0.0136；
  部署校准器（isotonic）已随模型持久化，后续每日任务走严格策略指纹门禁。

---

## 第二轮整改（采纳外部审核意见，按顺序执行）

### 1. PaperBroker 日期上界（P1 前视）
- [paper.py](../quant/execution/paper.py) `_last_price` 按 `trade_date` 截断，
  数据碰脏（含未来行情）不再前视；新增回归测试。

### 2. consensus_revision PIP 掩码（P1 前视）
- [definitions.py](../quant/factors/definitions.py) 快照对齐交易日历后
  `shift(1)`：盘后采集的一致预期最早下一交易日参与决策；历史不足 30 日
  或无覆盖保持 NaN（走标准 winsorize→zscore→0 填充），原始覆盖度可被
  哨兵观测。新增 PIP 单测（采集日不可用、次日生效、历史不足为 NaN）。

### 3. 模型上线门禁（Model Gate）
- 新增 [gate.py](../quant/model/gate.py)：`n_folds ≥ 5` + 复合因子
  `t ≥ 5.0`（FINAL_VERDICT 重标定口径）为硬门槛，AUC/单因子 IC 为信息项
  （默认 min=0）；阈值经 `model.gate_*` 可配置，`gate_block_on_fail` 控制
  告警/硬失败。
- [registry.py](../quant/model/registry.py) 持久化 `composite_t /
  composite_rank_ic`；[pipeline.py](../quant/pipeline.py) live 出信号前评估
  并写 `signals/gate_<date>.json`，不达标默认硬失败。
- 策略指纹排除 `gate_* / coverage_*` 运维字段：调整门禁/哨兵阈值不会强制
  重训历史模型（新增指纹单测）。

### 4. 中性化向量化（性能）
- [compute.py](../quant/factors/compute.py) 回归设计矩阵（行业哑变量 + 规模）
  预计算一次、跨因子复用；多因子 RHS 单次 lstsq 批量求解（调用次数从
  n_factors × n_dates 降为 n_dates）。实测：1052 只 × 2129 日，中性化
  **55s → 6.5s**；因子+标签+IC 全链路约 91s。修复了 `neutralize_size=False`
  时哑变量行数与横截面不一致的潜在崩溃。
- 数值等价性单测：批量路径与旧逐日循环结果 allclose。

### 5. paper/backtest 对账
- 修复 PaperBroker 现金不足整单拒绝：改为按可负担数量缩量成交（与回测引擎
  语义一致）——此前二次调仓时 C 股买入被整体拒绝，导致 1/3 仓位现金拖累。
- 新增对账测试：同一目标组合下 paper 与 backtest 期末净值偏差 **0.068%**
  （容差 2%）。

### 6. 因子覆盖度哨兵
- 新增 [coverage.py](../quant/monitor/coverage.py)：扫描
  `signals/factor_coverage_*.json`，`coverage_watch` 因子连续 N 日原始覆盖度
  低于阈值触发告警（默认 consensus_revision、30%、5 日）。
- `compute_all_factors(report_coverage=True)` 输出原始值覆盖度；live 写
  `factor_coverage_<date>.json`，scheduler 每日检查并通知。
- 实测：当前 consensus_revision 覆盖度 0.0（快照历史不足 30 日），哨兵将
  如实告警直到数据累积，防止因子静默退化为 0。
