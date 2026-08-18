# Tushare 历史成分 + 申万行业 point-in-time 回填方案

> 目标：消除「中证800 当前成分 + 2021 后退市股」带来的幸存者偏差，以及静态行业分类的前视。
> 产出：point-in-time 宇宙（date → symbols）+ point-in-time 申万行业归属。

> ⚠️ **实测结论（2026-08-14）**：当前 token 为免费非 Pro 层，实测 `stock_basic` 可访问（限频），
> 但 `index_weight`（历史成分）、`index_classify`/`index_member`（申万行业）均**无权限**。
> 因此本方案中的「历史成分」与「申万行业 point-in-time」**无法用当前 token 完成**，需升级 Pro。
> 已完成的免费部分：通过 baostock 缓存的 `stock_basic` 回填了 2018-2021 退市的 35 只股票（见
> `scripts/backfill_delisted.py`），并加了「仙股过滤」（`UniverseConfig.min_price`）。

## 0. 前置条件

1. 注册 Tushare Pro，取得 token，写入 `.env`（`TUSHARE_TOKEN=...`，已 gitignore）；
2. 确认账户积分：`stock_basic`（基础）、`index_weight`（月频成分，门槛较高）、
   `index_classify` / `index_member`（申万行业）。积分不足时先用 `stock_basic` 补退市股，
   历史成分退而用「季度市值前 N 并集」代理（已有 `universe_history.py`，但会高估）。

## 1. 数据接口与用途

| 接口 | 用途 | 关键字段 |
| :--- | :--- | :--- |
| `stock_basic(list_status=L/D/P)` | 全量股票（含退市/暂停） | ts_code, list_date, delist_date |
| `index_weight(index_code=000300.SH/000905.SH)` | 中证800 月度成分 | con_code, trade_date |
| `index_classify(level=L1, src=SW2021)` | 申万行业当前快照 | 行业名/代码 |
| `index_member(index_code=801xxx.SI)` | 申万行业成分进出（point-in-time） | con_code, in_date, out_date |

说明：中证800 = 沪深300 + 中证500；`index_weight` 按月返回成分，天然 point-in-time。
申万行业 point-in-time 需要遍历一级行业指数调 `index_member`，拿每只股票的 in/out 日期。

## 2. 回填步骤

```bash
python scripts/fetch_tushare_universe.py --start 2018-01-01 --end 2026-08-12
```

产出到 `data/raw_tushare/`：
- `stock_basic.parquet`（全量股票，含退市，覆盖当前 1007 只之外的历史退市股）
- `csi800_members.parquet`（中证800 月度成分，point-in-time）
- `sw_industry.parquet`（申万行业，先静态，后按 index_member 升级为 point-in-time）

## 3. 存储 schema 变更

新增 `data/processed/universe_history.parquet`：`(date, symbol)`，表示该日在中证800 内。
新增/升级 `industry.parquet`：从静态 `(symbol, industry, as_of_date)` 升级为
`(symbol, industry, in_date, out_date)`，中性化时按「in_date ≤ 交易日 < out_date」取有效行业。

## 4. 集成改动（代码层）

1. `quant/data/storage.py`：`load_bundle()` 增加 `universe_history`；`DataBundle` 增加字段。
2. `quant/factors/compute.py`：`build_panels()` 的行业映射改为按交易日取有效行业
   （`in_date ≤ date < out_date`），替换 `drop_duplicates(keep="last")`。
3. `quant/portfolio/selection.py`：`_universe_ok()` 的股票池过滤改为用
   `universe_history`（当日成分）∩ 流动性/ST/停牌/涨跌停过滤，替换「当前成分全集」。
4. `quant/pipeline.py`：`prepare_data()` 增加 `--step universe-history` 的 Tushare 分支，
   并把 `universe_history` 写入 storage。

## 5. 验证

1. 对比补前/补后的宇宙覆盖：退市股、调出股的缺失比例应显著下降；
2. 行业中性化回归测试：`test_neutralize_keeps_all_symbols` 继续通过；
3. 重跑长窗口 `python run_pipeline.py --config configs/real_2018.yaml`，
   预期年化从 24.6% 向下修正，得到「偏差修正后的真实收益」。

## 6. 已知限制

- Tushare 积分不足时，历史成分仍可能不完整；此时退市股用 `stock_basic` 补齐是硬收益，
  历史成分用市值代理是次优替代；
- 申万行业 point-in-time 的接口门槛可能更高，先用静态 SW2021 + 文档标注该局限。
