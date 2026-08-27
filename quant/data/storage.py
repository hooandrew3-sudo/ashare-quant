"""Parquet 存储：版本化、增量写入、manifest 指纹。"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

from quant.data.baostock_sync import normalize_flag_columns
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
    constituents: pd.DataFrame = field(default_factory=pd.DataFrame) # snapshot_date, symbol（指数成分点内快照，消除幸存者偏差）
    cashflow: pd.DataFrame = field(default_factory=pd.DataFrame)     # symbol, as_of_date, report_period, ocf_ytd, eps, ocfps（现金流因子源数据）

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
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                # 写入中途崩溃会留下半截 JSON；回退 .bak，避免全库不可用
                bak = self.manifest_path.with_suffix(".json.bak")
                if bak.exists():
                    import logging

                    logging.getLogger("ashare.storage").warning(
                        "manifest.json 损坏，已从 %s 回退", bak.name
                    )
                    with open(bak, "r", encoding="utf-8") as f:
                        return json.load(f)
                raise
        return {"datasets": {}, "updated_at": None}

    def save_manifest(self) -> None:
        """原子写 manifest：先写临时文件再 os.replace，并保留上一版 .bak。

        此前为截断式直接写：进程中途崩溃 → 半截 JSON → 下次启动全库不可用。
        """
        self._manifest["updated_at"] = pd.Timestamp.now().isoformat()
        tmp = self.manifest_path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._manifest, f, ensure_ascii=False, indent=2, default=str)
        if self.manifest_path.exists():
            shutil.copyfile(self.manifest_path, self.manifest_path.with_suffix(".json.bak"))
        os.replace(tmp, self.manifest_path)

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
    def save(
        self,
        dataset: str,
        df: pd.DataFrame,
        partition_by_symbol: bool = True,
        allow_empty: bool = False,
    ) -> None:
        """写入 processed/{dataset}/，可选按 symbol 分区。

        空表默认抛错：上游拉取失败返回空表时，静默早退会保留磁盘上的过期
        数据，而调用方误以为已刷新。确认允许空表时显式传 allow_empty=True。
        """
        target = ensure_dir(self.processed / dataset)
        if df.empty:
            if allow_empty:
                return
            raise ValueError(
                f"拒绝写入空数据集 {dataset}：上游可能拉取失败。"
                "如确需写入空表请传 allow_empty=True"
            )
        # pandas 3 可能产生 ms/s 精度 datetime，混合精度写入后 fastparquet 无法读取；
        # 统一转为 datetime64[ns]
        datetime_cols = df.select_dtypes(include=["datetime64"]).columns
        if len(datetime_cols):
            df = df.copy()
            for col in datetime_cols:
                df[col] = pd.to_datetime(df[col]).astype("datetime64[ns]")
        if partition_by_symbol and "symbol" in df.columns:
            symbols: set[str] = set()
            for symbol, group in df.groupby("symbol", sort=False):
                fname = f"{symbol}.parquet"
                group.to_parquet(target / fname, index=False)
                self.update_manifest(dataset, symbol, group)
                symbols.add(fname)
            # 孤儿分区回收：universe 切换后旧 symbol 文件残留会被 load() 读入，
            # 造成新旧股票池数据混合污染
            for p in target.glob("*.parquet"):
                if p.name not in symbols:
                    p.unlink()
        else:
            tmp = target / "all.parquet.tmp"
            df.to_parquet(tmp, index=False)
            final = target / "all.parquet"
            if final.exists():
                final.unlink()
            os.replace(tmp, final)
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
        df = normalize_flag_columns(df)
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
            constituents=self.load("constituents"),
            cashflow=self.load("cashflow"),
        )
