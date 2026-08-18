"""Parquet 存储：版本化、增量写入、manifest 指纹。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

from quant.utils import ensure_dir, sha256_df


@dataclass
class DataBundle:
    """一次清洗后的完整数据快照。"""

    prices: pd.DataFrame          # date, symbol, open, high, low, close, volume, amount, ...
    benchmark: pd.DataFrame       # date, close
    meta: pd.DataFrame = field(default_factory=pd.DataFrame)          # symbol, list_date, delist_date
    industry: pd.DataFrame = field(default_factory=pd.DataFrame)      # symbol, industry, as_of_date
    fundamentals: pd.DataFrame = field(default_factory=pd.DataFrame)  # symbol, ann_date, roe, div_yield, ...
    sentiment: pd.DataFrame = field(default_factory=pd.DataFrame)    # symbol, date, sentiment
    forecast: pd.DataFrame = field(default_factory=pd.DataFrame)     # symbol, as_of_date, forecast_growth
    consensus: pd.DataFrame = field(default_factory=pd.DataFrame)    # symbol, as_of_date, year, n_institutions, eps_mean（分析师一致预期快照）
    smallcap: pd.DataFrame = field(default_factory=pd.DataFrame)     # date, close（中证1000 小盘基准）

    def validate(self) -> None:
        if self.prices.empty:
            raise ValueError("prices 为空")
        if self.benchmark.empty:
            raise ValueError("benchmark 为空")
        for col in ("date", "symbol", "open", "high", "low", "close", "volume"):
            if col not in self.prices.columns:
                raise ValueError(f"prices 缺少列: {col}")
        for col in ("date", "close"):
            if col not in self.benchmark.columns:
                raise ValueError(f"benchmark 缺少列: {col}")


class Storage:
    """本地 Parquet 仓库，支持按 symbol 分区与 manifest 追踪。"""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.processed = ensure_dir(self.root / "processed")
        self.manifest_path = self.root / "manifest.json"
        self._manifest: dict = self._load_manifest()

    # ---- manifest ----
    def _load_manifest(self) -> dict:
        if self.manifest_path.exists():
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"datasets": {}, "updated_at": None}

    def save_manifest(self) -> None:
        self._manifest["updated_at"] = pd.Timestamp.now().isoformat()
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(self._manifest, f, ensure_ascii=False, indent=2, default=str)

    def update_manifest(self, dataset: str, symbol: str, df: pd.DataFrame) -> None:
        entry = self._manifest["datasets"].setdefault(dataset, {})
        sym = str(symbol)
        dates = df["date"] if "date" in df.columns else pd.Series(dtype="datetime64[ns]")
        entry[sym] = {
            "rows": int(len(df)),
            "start": str(dates.min()) if len(dates) else None,
            "end": str(dates.max()) if len(dates) else None,
            "fingerprint": sha256_df(df)[:16],
            "updated_at": pd.Timestamp.now().isoformat(),
        }

    # ---- io ----
    def save(self, dataset: str, df: pd.DataFrame, partition_by_symbol: bool = True) -> None:
        """写入 processed/{dataset}/，可选按 symbol 分区。"""
        target = ensure_dir(self.processed / dataset)
        if df.empty:
            return
        # pandas 3 可能产生 ms/s 精度 datetime，混合精度写入后 fastparquet 无法读取；
        # 统一转为 datetime64[ns]
        datetime_cols = df.select_dtypes(include=["datetime64"]).columns
        if len(datetime_cols):
            df = df.copy()
            for col in datetime_cols:
                df[col] = pd.to_datetime(df[col]).astype("datetime64[ns]")
        if partition_by_symbol and "symbol" in df.columns:
            for symbol, group in df.groupby("symbol", sort=False):
                group.to_parquet(target / f"{symbol}.parquet", index=False)
                self.update_manifest(dataset, symbol, group)
        else:
            df.to_parquet(target / "all.parquet", index=False)
            self.update_manifest(dataset, "*", df)
        self.save_manifest()

    def load(self, dataset: str) -> pd.DataFrame:
        target = self.processed / dataset
        if not target.exists():
            return pd.DataFrame()
        parquet_files = sorted(target.glob("*.parquet"))
        if not parquet_files:
            return pd.DataFrame()
        frames = [pd.read_parquet(p) for p in parquet_files]
        df = pd.concat(frames, ignore_index=True)
        datetime_cols = df.select_dtypes(include=["datetime64"]).columns
        if len(datetime_cols):
            for col in datetime_cols:
                df[col] = pd.to_datetime(df[col]).astype("datetime64[ns]")
        return df

    def has(self, dataset: str) -> bool:
        return (self.processed / dataset).exists() and len(list((self.processed / dataset).glob("*.parquet"))) > 0

    def load_bundle(self) -> DataBundle:
        return DataBundle(
            prices=self.load("prices"),
            benchmark=self.load("benchmark"),
            meta=self.load("meta"),
            industry=self.load("industry"),
            fundamentals=self.load("fundamentals"),
            sentiment=self.load("sentiment"),
            forecast=self.load("forecast"),
            consensus=self.load("consensus"),
            smallcap=self.load("smallcap"),
        )
