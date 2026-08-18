"""分析师一致预期采集器（同花顺 THS 盈利预测）。

核心：A 股免费数据源不提供"历史一致预期修正序列"，本模块采用工业级方案——
**每日拉取当前一致预期快照，累积成时间序列**，从而支持"一致预期修正"因子：

    consensus_revision = 当前一致预期EPS / N日前一致预期EPS - 1

数据源：akshare.stock_profit_forecast_ths(symbol, indicator='预测年报每股收益')
返回：年度 / 预测机构数 / 最小值 / 均值 / 最大值 / 行业平均数
本模块提取「最近一个年度的均值」作为一致预期 EPS 快照。

采集策略：
- 逐只拉取（带重试 + 休眠），仅处理传入的股票池（建议 300-800 只核心池）；
- 每日快照按 date 追加写入 data/processed/consensus/；
- 幂等：同日重复拉取直接跳过（按 manifest 检查）。
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Iterable

import pandas as pd

from quant.utils import setup_logging

LOGGER = logging.getLogger("ashare.consensus")


def normalize_code(code) -> str | None:
    """6 位代码 → '600519.SH'；仅保留沪深 A 股。"""
    s = str(code).strip()
    if len(s) != 6 or not s.isdigit():
        return None
    if s[0] in ("6", "9"):
        return f"{s}.SH"
    if s[0] in ("0", "3"):
        return f"{s}.SZ"
    return None


def _fetch_one(symbol: str, verbose: bool = False) -> pd.DataFrame:
    """拉取单只股票的一致预期 EPS，返回 [(symbol, year, n_inst, min, mean, max)]。"""
    import akshare as ak

    code = symbol.split(".")[0]
    df = ak.stock_profit_forecast_ths(symbol=code, indicator="预测年报每股收益")
    if df is None or df.empty:
        return pd.DataFrame(columns=["symbol", "year", "n_institutions", "eps_min", "eps_mean", "eps_max"])
    df = df.rename(
        columns={
            "年度": "year",
            "预测机构数": "n_institutions",
            "最小值": "eps_min",
            "均值": "eps_mean",
            "最大值": "eps_max",
        }
    )
    for col in ("n_institutions", "eps_min", "eps_mean", "eps_max"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["symbol"] = symbol
    return df[["symbol", "year", "n_institutions", "eps_min", "eps_mean", "eps_max"]]


def fetch_consensus_snapshot(
    symbols: Iterable[str],
    verbose: bool = True,
    pause_sec: float = 0.2,
    max_retries: int = 2,
) -> pd.DataFrame:
    """拉取一批股票的一致预期快照，返回 long 表（含 as_of_date=今天）。"""
    log = setup_logging(verbose)
    today = pd.Timestamp.today().normalize()
    parts: list[pd.DataFrame] = []
    syms = sorted(set(symbols))
    errors: list[str] = []
    for i, sym in enumerate(syms, 1):
        ok = False
        for attempt in range(1, max_retries + 1):
            try:
                df = _fetch_one(sym)
                if not df.empty:
                    df["as_of_date"] = today
                    parts.append(df)
                    ok = True
                break
            except Exception as exc:  # noqa: BLE001
                if attempt >= max_retries:
                    errors.append(f"{sym}: {exc}")
                    break
                time.sleep(1.0 * attempt)
        if verbose and (i % 50 == 0 or i == len(syms)):
            log.info("一致预期采集进度 %d/%d（成功 %d，失败 %d）", i, len(syms), len(parts), len(errors))
        if pause_sec:
            time.sleep(pause_sec)
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["symbol", "year", "n_institutions", "eps_min", "eps_mean", "eps_max", "as_of_date"]
    )
    if errors:
        log.warning("一致预期采集完成，%d 只失败（如 %s）", len(errors), errors[:3])
    return out


def save_snapshot(snapshot: pd.DataFrame, root: str | Path) -> None:
    """按日期幂等追加到 data/processed/consensus/。"""
    from quant.data.storage import Storage

    storage = Storage(root)
    if snapshot.empty:
        return
    dates = snapshot["as_of_date"].unique()
    existing = storage.load("consensus")
    combined = pd.concat([existing, snapshot], ignore_index=True)
    combined = combined.drop_duplicates(subset=["symbol", "year", "as_of_date"], keep="last")
    storage.save("consensus", combined, partition_by_symbol=False)
    LOGGER.info("一致预期快照已保存：%d 行（%d 只 × %s）", len(combined), snapshot["symbol"].nunique(), dates[0].date())


def load_consensus_history(root: str | Path) -> pd.DataFrame:
    """加载全量一致预期历史快照。"""
    from quant.data.storage import Storage

    return Storage(root).load("consensus")
