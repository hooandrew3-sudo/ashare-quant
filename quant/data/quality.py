"""数据质量校验：任何关键项失败即抛 DataQualityError。"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


class DataQualityError(Exception):
    """数据质量不达标。"""


@dataclass
class QualityReport:
    ok: bool = True
    checks: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def add(self, name: str, passed: bool, detail: str = "") -> None:
        self.checks.append(name)
        if not passed:
            self.ok = False
            self.errors.append(f"{name}: {detail}" if detail else name)

    def summary(self) -> str:
        status = "PASS" if self.ok else "FAIL"
        return f"[{status}] {len(self.checks)} 项检查, {len(self.errors)} 项失败"


def _fmt(x) -> str:
    return "" if x is None else str(x)


def check_prices(df: pd.DataFrame, allow_empty: bool = False) -> QualityReport:
    """对 prices 长表执行关键质量检查。"""
    rep = QualityReport()
    if df.empty:
        if allow_empty:
            rep.add("empty", True, "数据为空(允许)")
            return rep
        rep.add("empty", False, "prices 为空")
        return rep

    required = ["date", "symbol", "open", "high", "low", "close", "volume"]
    missing_cols = [c for c in required if c not in df.columns]
    rep.add("required_columns", not missing_cols, f"缺少列: {missing_cols}")

    rep.add("dup_rows", not df.duplicated(["date", "symbol"]).any(), "存在重复 (date, symbol)")

    df = df.dropna(subset=["open", "high", "low", "close"])
    rep.add("ohlc_high_low", (df["high"] >= df["low"]).all(), "存在 high < low")
    rep.add("ohlc_inside", ((df["high"] >= df[["open", "close"]].max(axis=1)) &
                            (df["low"] <= df[["open", "close"]].min(axis=1))).all(),
            "存在 high/low 无法包含 open/close")
    rep.add("positive_price", (df[["open", "high", "low", "close"]] > 0).all().all(), "存在非正价格")
    rep.add("nonneg_volume", (df["volume"] >= 0).all(), "存在负成交量")

    if "is_limit_up" in df.columns and "is_limit_down" in df.columns:
        rep.add("limit_flag_consistency",
                not (df["is_limit_up"] & df["is_limit_down"]).any(),
                "同日既涨停又跌停")

    if "is_suspended" in df.columns:
        suspended = df["is_suspended"].astype(bool)
        if suspended.any():
            rep.add("suspended_no_price",
                    df.loc[suspended, ["open", "high", "low", "close"]].isna().all(axis=1).all(),
                    "停牌日存在价格")

    # 日期范围
    if "date" in df.columns:
        dmin, dmax = df["date"].min(), df["date"].max()
        rep.add("date_range", pd.Timestamp(dmax) > pd.Timestamp(dmin), f"范围 {dmin} ~ {dmax}")
    return rep


def check_benchmark(df: pd.DataFrame) -> QualityReport:
    rep = QualityReport()
    if df.empty or "date" not in df.columns or "close" not in df.columns:
        rep.add("benchmark_schema", False, "benchmark 缺少 date/close")
        return rep
    rep.add("benchmark_positive", (df["close"] > 0).all(), "基准存在非正收盘")
    rep.add("benchmark_dup", not df["date"].duplicated().any(), "基准日期重复")
    return rep
