# 分析师一致预期修正因子 · 落地指南

> 背景：审计发现当前策略在 point-in-time 条件下无 alpha（OOS IC≈0）。
> 方向：引入 A 股最强基本面 alpha 之一——分析师一致预期修正（EPS Forecast Revision）。

---

## 一、为什么是"一致预期修正"而非"业绩预告"

| 维度 | 业绩预告（公司自报） | 分析师一致预期修正（卖方共识） |
|------|---------------------|-------------------------------|
| 发布方 | 上市公司 | 卖方分析师（多机构共识） |
| 频率 | 每年 1-4 次（季报前） | 持续更新（月度频率） |
| 截面区分度 | 弱（实测 IC=-0.02, t=-0.61） | 强（下修趋势持续性强） |
| Alpha 机制 | 一次性事件 | **预期修正的持续性**（上修→继续上修） |

实测验证（`scripts/_test_forecast_alpha.py`）：现有业绩预告增速因子 **IC=-0.0207, t=-0.61 无效**，
证实必须引入真实的"分析师一致预期"数据，而非依赖现有业绩预告。

---

## 二、数据方案（工业级：自建快照累积）

**问题**：A 股免费数据源不提供历史一致预期修正序列（Wind/朝阳永续为付费）。

**方案**：每日拉取当前一致预期快照 → 累积成时间序列 → 计算修正因子。

```
每日: akshare.stock_profit_forecast_ths(symbol, '预测年报每股收益')
      → {symbol, year, n_institutions, eps_min, eps_mean, eps_max}
      → 追加到 data/processed/consensus/ (幂等)
```

**已实现模块**：
- `quant/data/consensus_ths.py` — 采集器（fetch/save/load，带重试）
- `quant/factors/definitions.py` — `consensus_revision` 因子
- `quant/scheduler.py` — 每日调度已接入
- `scripts/init_consensus.py` — 初始快照采集

---

## 三、因子定义

```python
consensus_revision = 当前一致预期EPS / 30日前一致预期EPS - 1
```

- **正值 = 上修**（看多信号）；**负值 = 下修**（看空）
- 数据不足（<30 日快照）时返回 0（中性，不引入错误信号）
- 已注册到 `FACTOR_SPECS`，direction=+1

**配套增强因子（后续可加）**：
- 机构数变化率：`n_institutions 增加 = 关注度提升`
- 一致预期离散度：`(max-min)/mean`，离散度收窄 = 共识加强

---

## 四、启用步骤

### 第 1 天（一次性）
```bash
python scripts/init_consensus.py 300   # 拉取 300 只核心池当前快照（约 5-10 分钟）
```

### 每日（自动）
- `python -m quant.scheduler --step daily` 已内置一致预期快照采集
- 或手动：`python scripts/init_consensus.py 300`（幂等，同日跳过）

### 30 天后
- consensus 序列累积 ≥30 个快照 → `consensus_revision` 因子开始产生有效值
- 运行 `python scripts/run_research.py` 全流程，IC 报告自动包含 consensus_revision

### 90 天后
- 累积 ≥90 日快照，进行因子稳定性验证：
  - Rank IC / ICIR / 分层收益
  - 与动量/小市值因子的相关性（防冗余）

---

## 五、验证指标（上线门槛）

| 指标 | 门槛 |
|------|------|
| Rank IC 均值 | > 0.03 |
| ICIR | > 0.5 |
| t 值 | > 3.0 |
| 与 size_proxy 相关性 | < 0.5（防与小市值因子冗余） |
| point-in-time 回测 OOS IC | > 0.02 |

---

## 六、注意事项

1. **接口稳定性**：akshare 的 `stock_profit_forecast_ths` 曾有接口变更（见 akshare issue #5748），采集器需容忍失败（已带重试）
2. **覆盖偏差**：免费数据只覆盖有机构跟踪的股票（约 1000-2000 只），无机构覆盖股票因子=0
3. **采集频率**：建议每日（修正信号周度级别有效），可降频至每周以减负载
4. **不要与业绩预告混用**：两者数据源、语义、时效均不同
5. **累积纪律**：快照一旦累积不可回填历史（免费源限制），尽早开始采集

---

## 七、与其他 alpha 的协同

一致预期修正与现有因子的组合建议：
- **动量 + 一致预期上修**：动量确认 + 基本面确认（经典双确认）
- **小市值 + 一致预期上修**：小市值风格 + 机构关注度提升（高弹性）
- **避免**：与 quality 因子（ROE）高度相关，需检查共线性
