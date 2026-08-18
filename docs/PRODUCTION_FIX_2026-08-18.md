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
