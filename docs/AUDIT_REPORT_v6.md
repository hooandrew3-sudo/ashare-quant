# A股量化项目工业级生产审核报告 v6

> 审核: 顶级量化交易设计师视角
> 范围: 全量代码逐模块审查（pipeline / backtest / factors / model / portfolio / data / execution / monitor / cli / scheduler）
> 基线: FINAL_VERDICT.md、AUDIT_REPORT_v3/v4/v5、PRODUCTION_SPEC、PRODUCTION_FIX_2026-08-18
> 时间: 2026-08-18

---

## 〇、整体定位（先讲结论）

本项目**已是同类零售/研究型框架中成熟度较高的一档**，不是 toy。工业级实践已经落地一批：
- `Config` dataclass + YAML 加载 + `config_fingerprint` / `strategy_fingerprint` / `data_fingerprint` 三维指纹，实验可复现性有支撑；
- `data/storage.py` Parquet 按票分区 + manifest 指纹 + `load_bundle` 一次性加载 9 个数据集；
- PIT universe 修复（成分股动态进出，不再全程用最终成份）—— 已写入 v5；
- `model/gate.py` composite_t 硬门（≥5）+ AUC/rank_ic 软指标；
- `monitor/coverage.py` factor coverage sentinel；
- `execution/paper_runner.py` reconcile + 幂等追加 history/orders CSV；
- `data/quality.py` OHLC/dup/涨跌停标记/停牌无价检查；
- 仙股过滤、行业循环剔末位、小盘替换层。

这套骨架是**值得继续投入**的。下面所有问题都是在承认这批工业实践之上提出的"再上半级台阶"清单，分级 P0–P3。

---

## 一、P0 —— 影响策略真实性 / 会直接砸盘的项

### P0-1　`data/real.py:78` 硬抛导致每日调度全线崩溃（已知、未修，生产阻断）

**现场**：`daily.log 2026-08-18 20:35` → `RuntimeError: baostock 未返回任何数据` 抛在 `run_daily → prepare_data`。

```python
# quant/data/real.py
if df.empty:
    raise RuntimeError("baostock 未返回任何数据")   # ← 这里直接顶穿
```

**工业判断**：这是一个**只检不降**的脆弱点。baostock 本身网络抖、token 失效、个别交易日补数延迟都属常态。对一个**每日自动化调度**系统，"取不到数就 raise"=当天策略不跑 = 仓位裸露或错过信号，**比脏数据更危险**。

**生产级修法（按优先级）**：
1. 三级源降级：baostock 空 → 自动回退到本地最新 Parquet bundle（`storage.load_bundle`）的最近交易日，且把当日降级写进 `notifier` 与 `coverage` 哨兵，标 ` data_mode=degraded`；
2. 重试退避：网络类异常重试 3 次，指数退避 1s/2s/4s，仍败才走降级；
3. `assert tolerate_partial`：明确"当日缺失"与"全量缺失"的分级，前者用前一交易日截面后备，后者**才**raise 并告警；
4. scheduler 侧 `run_daily` 的 try/except 不能停留在"发个通知就结束"——要把 `data_mode` 状态写进 state.json 以便复盘；
5. 引入 `CALENDAR` 依赖（`tushare`/`akshare`/`baostock` 的交易日历），**避免非交易日仍在反复重试**。

> 注：scheduler 的"幂等跳过"以 `last_paper_date==latest` 字符串相等为准，本质正确，但 baostock 在非交易日返回空也会触发该崩溃——需要日历感知。

### P0-2　回测成交可行性缺口：`fills.py:63` 冗余裁剪 + 无量约束

```python
# quant/backtest/fills.py
# buy shares = min(shares, round_lot(shares))  # round_lot ≤ shares，恒等于 round_lot 自身
```

这行**不只是冗余**，它掩盖了一个更狠的缺口：**没有任何对当日成交量的参与率上限**。当前 `try_fill` 只做开盘价撮合 + 涨跌停判断，意味着回测可以"一笔吃掉标的 100% 当日量"，在流动性差的标的尤甚，导致回测净值系统性高估、换手与成本测算乐观。

**生产级修法**：
- 引入 `max_participation`（默认 5%–10%），`shares = min(shares, round_lot(volume * max_participation))`；
- `cost.effective_slippage_bp` 已有 `impact_coef*participation`，但 participation 目前**不是基于成交量算的真实参与率**，应改成 `order_shares / day_volume`；
- `paper.py` 的 `place_order` 用**昨收撮合**且无量纲约束，同一缺口，需同步补上限并对账到"回测假设 ≈ paper 假设 ≈ 线下真实抢单"，否则三套结果不可比；
- 在 HTML 报告里新增"流动性压力表"：每个持仓标的在回测内的平均参与率 p95，p95>20% 的标的单独列出。

### P0-3　幸存者偏差在模型训练里进、在因子中性里出（PIT 残留）

v5 修了 universe 的 PIT，但 **fundamental 与 industry 的 PIT 仍未完全闭合**：

```python
# quant/factors/compute.py  _attach_fundamentals
# 缺失 ROE/forecast → fillna(0)   # 0 被当成"中性值"喂进中性化设计矩阵
# industry_map.drop_duplicates(keep='last')  # keep last = 用最新行业分类回看历史
```

两个并发病灶:
1. **填 0 ≠ 中性**：缺失 ROE 的票往往是次新/ST/刚借壳，用 0 参与行业 demean 等于把"缺失"当成"行业平均"，扭曲同行业横截面排序；中性化设计矩阵没有 intercept 项,缺失哑变量本应单独成列.D+1 维 → 现在是 D 维乘以 0 向量,残差里**仍然带着"缺失"信号**。
2. **行业 keep last = 用 2026 年的行业归属去归一 2019 年的截面**：申万分类本身有历史调整,用最新分类对历史截面做中性,**等价于一类隐性前视**,会**低估真实横截面 IC**(因为行业归属变化带来的残差本来就属于当时可见信息),反过来也会让"事后幸存的标的"在历史截面的行业虚变量上获得稳定的中性后残差.

**生产级修法**：
- 行业序列走 PIT：`industry_map` 按 `(symbol, date)` 取 `as_of_date ≤ t` 的最后一条,而不是全局 `drop_last`；
- fundamental 缺失**显式建模**：加 `is_missing` 哑变量列，进入设计矩阵；缺失 ROE 用行业季节中位数或 `np.nan`→矩阵求解时按列掩码,而不是 0；
- 模型训练 `train.py` 的 dtype 准备里加一套 assert: 所有进 X 的列**不允许**含 0-only 缺失填充列(否则树模型会把"是否缺失"学成一棵几乎纯分裂的子树,泛化崩溃)。

---

## 二、P1 —— 数值/工程正确性、口径对不齐

### P1-1　`turnover_cap` 配置三处分歧 + 窗口口径不一致

| 位置 | 值/口径 |
|---|---|
| `configs/default.yaml` | `turnover_cap: 0.40` |
| `Config` dataclass default | `0.20` |
| `backtest/engine._turnover_breached` | 滚动 **365 天**窗口（日历） |
| `metrics/performance._turnover` | 全周期 `(buy+sell)/2 / avg_equity / years` |

后果:
- YAML 没写就用 0.20，写了 0.40；**新人切换配置极易踩**。
- breaker 用滚动 365,报告指标用全周期/年,**两者算的不是同一个东西**,会出现"报告显示换手 3.2 倍、但 breaker 就是没触发"的诡异局面.
- `max_turnover_annual=4.0` 与 `turnover_cap` 是**两套并行**的换手限制且语义重叠：团队共识里到底以哪个为准，代码看不出来.

**生产级修法**:
- 单一来源(SSOT): 删掉 dataclass 里的 default 重复值,改为"YAML 必填,缺失即报错";
- 统一口径:breaker 与 metrics 走同一个函数 `turnover_curve(equity_curve, trades) -> pd.Series`,窗口与除数都从该函数出;
- 用交易日历窗口(约 244/年)而非 365 日历天,或显式 `trading_days_per_year` 配置项统一;
- HTML 报告三处口径并排展示:**strategy 换手上限 / breaker 滚动窗口实测 / 全周期年化换手**,三者一致才发版.

### P1-2　`portfolio/risk_budget.py` 数值健壮性

```python
cov = cov + 1e-3 * np.eye(n)        # 1. 均匀抖动，对小票协方差阵信噪比偏低
np.nan_to_num(cov, nan=0.0, posinf=0.0, ...)
# 2. posinf → 0 会把"无限相关"当作"无关"，权重被错配
# 3. double-clip 后 renormalize: w = w / w.sum() 在存在硬 max_weight 截断时，
#    解空间非凸，clip+normalize 不等价于带约束优化，concentrated 解会漂
```

**生产级修法**:
- 抖动用对角主元缩放: `cov += 1e-6 * np.trace(cov)/n * np.eye(n)`,而不是 1e-3 常量;
- 对 `posinf`/`neginf` 引入 mask 而非置 0: 协方差该行该列对应的两个标的应**剔出本日预算**,而不是当 0;
- 权重约束用 `scipy.optimize.minimize(SLSQP)` 求解带 `0≤w≤max_weight, Σw=1` 的真实约束问题,或退一步用迭代 water-filling.当前 clip+normalize 在 `max_weight=0.10` + 20+ 候选票的集中场景会违反约束( normalization 后某些权会再次超 max_weight).
- 空/NaN 协方差日期回退到等权 + 锁定 `risk_budget_mode=fallback`,告警.

### P1-3　`qmt.py:87` 假定 `res==0` 即成交 / OMS 去重仅客户端

```python
# qmt.py  place_order
if res == 0:
    order.status = OrderStatus.SUBMITTED   # ← 仅表示"券商接口接受",非成交
```

- xtquant 异步下单 res==0 只代表**报单已被柜台接受**,不代表成交.当前没有对成交回报的轮询/订阅回调,`PaperBroker` 的 reconciliation 在 QMTBroker 上没有对应物.
- `OrderManager` 幂等靠客户端生成的 `order.id` 集合,**券商那侧的报单号(sys_id/filled_id)从未进入去重键**.一旦本地重启、state 恢复、order.id 重生,就有重复挂单风险.

**生产级修法**:
- 引入成交回报同步:`xt_connect.subscribe_quote` 或定时 `query_stock_orders` 回填 `fill_px/fill_qty`,与本地 OMS 对账,状态机 `SUBMITTED→PARTIALLY_FILLED→FILLED/CANCELED/REJECTED`;
- OMS 去重键 = `(client_id, broker_sys_id)` 二元组,broker_sys_id 为空时才回退到 `client_id`,并**在 OMS 状态机里禁止"无 ack 的同名重挂"**;
- `kill_threshold=3` 重试的退避应指数化,且对 `BrokerTimeout` 与 `BrokerReject` 区别对待:前者重试,后者立即转 killed 并通知;
- 在 paper_runner 的 reconcile 之外,**QMTBroker 同样需要日终 reconcile**(目前只有 paper 有,QMT 直连生产没有闭环).

### P1-4　`metrics/performance._turnover` 与 engine breaker 口径分叉

已并入 P1-1 说明，但单独点一根：这两个函数名都叫 turnover 系列，**实现各异且没有共享测试**。建议把 turnover 计算抽到 `metrics/turnover.py` 一处实现，回收两处调用方。

---

## 三、P2 —— 模型/信号层稳健性

### P2-1　`factors/analysis.py:31` `fillna(0.0)` 掩盖全 NaN 行偏置

```python
bucket = (factor_wide.rank(axis=1, pct=True) * n).fillna(0.0).astype(int).clip(0, n-1)
```

当年某日某因子全 NaN(如新因子冷启动期/某行业停牌潮),整行被塞进 bucket 0,**计算 quantile return 时 0 档多了一堆噪声样本**,IC 与分位收益被拉偏.历史上那次 `IntCastingNaNError` 崩溃已被 `fillna(0.0)` 硬止血,但**根因(全 NaN 行)未处理**.

**生产级修法**:
- 整行全 NaN → 直接 `dropna(axis=0, how='all')` 后再算 quantile return,不要把空人塞进最低档;
- `rank_ic_series` 同步:全 NaN 日记 NaN 并在前向报告里标注"因子覆盖率不足,当日 IC 不计入";
- coverage 哨兵要捕获"日内 NaN 比例"而不只是"有/没有",设阈值(如日内覆盖率<80% 触发降级)。

### P2-2　`portfolio/regime.py` 波动率不稳 + `beta_scale` 数值不安全

- `vol_scale` 用 `close.pct_change().std() * sqrt(252)`,这是**全样本口径**,且窗口随每日滚动扩,早期天数不足时数值发散.**未对**:应该用 ewm 半衰期 ~20d 或滚动 60d,并 clip 后做仓位缩放.
- `engine.compute_beta_scale` 用 `np.polyfit(beta_series, pf_ret, 1)`,当 beta_series 含 inf/nan 或样本<10 会返回退化系数,且返回值 `np.clip(scale, 1.0, pf.beta_scale_max)` **下界恒 ≥1** —— 这意味着**永远只会加杠杆,永不降杠杆**,与 regime 风控意图相反.

**生产级修法**:
- vol 改 ewm,样本不足回退到 `target_vol`(给定先验),并写状态 `vol_regime=uncertain`;
- beta_scale 下界改为 `clip(scale, 1/beta_scale_max, beta_scale_max)` 或更安全: `clip(scale, 0.5, 2.0)`; polyfit 前先 `np.isfinite` 过滤 + `len>=N_min` 检查,不然返回 1.0(不缩放);
- 总暴露 `target_exposure = min(1.0, target_exposure * beta_scale)` 这条**本身**就是"只增不减",建议改为 `target_exposure * beta_scale` 后对上限独立 clip.

### P2-3　`compute_beta_scale` 在 polyfit 失败时回退行为不明示

当前未对"polyfit 失败/样本不足"显式返回 1.0 之外的标记,建议返回 `(scale, is_degraded)` 元组,degraded 写入 regime_map/signal_health_map 以便 HTML 报告展示"哪些日子的 beta 缩放是凑合凑合".

---

## 四、P3 —— 工程/测试基础设施

### P3-1　测试在 DSH 文件沙箱下全量 `tmp_path` 失败 (~7 E)

约 7 个测试因 `tmp_path` 触发 `PermissionError [WinError 5]` 于 `os.scandir`,约 90 个纯逻辑测试通过.**这不是代码缺陷,是环境约束**,但反映一个真实生产风险:**测试与临时文件路径耦合过紧**,CI 换环境就挂.

**生产级修法**:
- 受影响测试改用 `monkeypatch.setattr` 指向 session workspace 下固定子目录(`tests/_tmp/`),并在 teardown 里手动 `shutil.rmtree`;
- 长期:把"文件依赖"测试收进独立 marker `@pytest.mark.fs`,在 CI 矩阵里单独跑 + 单独沙箱;
- 在 README/CONTRIBUTING 写明"运行测试的工作目录要求",避免本地新人卡这里.

### P3-2　`model/registry.py` 硬依赖 joblib

`joblib.dump` 在版本/环境迁移时偶尔背刺(scikit-learn 版本耦合).建议:
- 模型序列化加版本字段 `serializer_version`,加载侧 `try/except` 回退到 `lightgbm.Booster(model_file=model.model)` 原生加载;
- meta.json 里记 lightgbm 版本,部署前 guard `assert match_minor(lgbm_version)`.

### P3-3　scheduler / consensus 明细

- `run_daily` try/except 仅 `notifier.send`,**没有重试或挂起到下一交易日**;建议失败态写入 `state.json` 的 `pending_recover=True`,次日先补跑昨日再跑当日(PIT 隔离两日截面).
- consensus 拉取异常吞掉但只 warn,**生产期应该 fail-loud** 或至少进 alerts,因为 consensus 错会污染后续所有 cross-sectional 排序.

---

## 五、按交付物映射的"上半级"清单

| 交付物 | 当前状态 | 建议升级 |
|---|---|---|
| 交易日历 | baostock 取数即崩 | 三级降级 + 重试退避 + 日历感知调度 |
| 成交撮合 | 价+涨跌停,无量约束 | 参与率上限 + 流动性压力表 |
| 数据 PIT | universe 已修,fundamental/industry 残留 | industry PIT 滚动 + fundamental 缺失显式建模 |
| 换手限制 | 三处口径分叉 | SSOT + 单一 turnover_curve 函数 |
| 风险预算 | 1e-3 eye + clip+normalize | 对角主元抖动 + SLSQP 约束求解 |
| OMS | 客户端 id 去重 | (client_id, broker_sys_id) 二元去重 + 成交回报对账 |
| 因子分析 | fillna(0) 掩盖 | 全 NaN 行 dropna + 日内覆盖率阈值 |
| 波动/beta | 全样本 std + polyfit 只增不减 | ewm + 对称 clip + degraded 标记 |
| 测试 | tmp_path 全环境挂 | monkeypatch 工作区目录 + fs marker |
| 序列化 | joblib 裸用 | serializer_version + lgbm 原生回退 |

---

## 六、最小生效干预（建议本次合并)

建议**以一个 PR 打包 P0 三项**，其余分级排期:
1. `data/real.py` 改三级降级 + 重试退避（直接堵住 daily 崩溃);
2. `fills.py` / `paper.py` 加 `max_participation`（让回测可发表);
3. `factors/compute.py` industry PIT 滚动 + fundamental 缺失哑变量(让模型训练里 PIT 闭合).

P1 配置 SSOT + turnover 统一口径可作为下一个 PR，与 P0 并行评审.

---

## 七、结论

本项目在"研究型→生产型"的渡口**已经过半**，PIT universe、指纹、gate、coverage、reconcile 这批基础设施是真实存在的工业实践，值得肯定。但要把策略**从 backtest 可发表推到 live 可托管**，P0–P1 这几道关必须先过，尤其:

- **daily 不崩**(P0-1) 是 live 的最低门槛;
- **回测假设≈真实抢单**(P0-2) 是 backtest 可发表的前提;
- **PIT 全链闭合**(P0-3) 是任何 ML alpha 声明成立的前提——目前 ML 层 AUC≈0.5 的结论与"残存 PIT"耦合,真伪需在修完后重测.

按本报告清单推进后，建议重跑一轮 walk-forward 并在 v7 报告里对比 PIT 闭合前后的 OOS IC 与换手成本，**以数值证伪/证实**当前 "alpha=风格暴露、ML 无增量" 之结论是否仍成立。

—— 以工业级标准衡量，这次审核给出的不是"推翻重来"，而是"再上半级台阶"的目标清单。
