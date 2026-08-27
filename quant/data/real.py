"""真实数据接入：baostock（日线/复权，走生产级同步器）与 CSV 兜底。"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from quant.data.storage import DataBundle

log = logging.getLogger("ashare.sync")


def _fmt_date(d) -> str:
    return pd.Timestamp(d).strftime("%Y-%m-%d")


def load_from_baostock(
    symbols: list[str],
    start: str,
    end: str,
    benchmark: str = "sh.000906",
    incremental: bool = False,
    manifest: dict | None = None,
    storage: "Storage | None" = None,
    universe: str = "manual",
) -> DataBundle:
    """从 baostock 下载日线（前复权），委托生产级同步器。

    symbols 支持 'sh.600519' 或 '600519.SH' 两种格式；
    incremental=True 时基于 manifest 只拉取增量区间并与已有数据合并。
    """
    from quant.data.baostock_sync import BaoStockSync, bs_to_symbol, merge_incremental, symbol_to_bs

    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    norm_symbols = [
        s if "." in s and len(s.split(".")[0]) == 6 else bs_to_symbol(s) for s in symbols
    ]
    cache_dir = storage.root / "cache" if storage is not None else None
    with BaoStockSync(cache_dir=cache_dir) as sync:
        basic = sync.stock_basic()
        industry = sync.industry()
        sync_start = _fmt_date(start)
        inc = incremental
        if inc and storage is not None and manifest is not None:
            from quant.data.baostock_sync import incremental_window

            meta = manifest.get("meta", {})
            if meta.get("universe") != universe or meta.get("source") != "baostock":
                log.warning("manifest 股票池与本次不一致（%s vs %s），改为全量同步",
                            meta.get("universe"), universe)
                inc = False
            else:
                sync_start = incremental_window(manifest, "prices", end)
        if inc:
            log.info("增量同步窗口: %s → %s", sync_start, _fmt_date(end))
        else:
            log.info("全量同步窗口: %s → %s", sync_start, _fmt_date(end))
        calendar = sync.trading_calendar(sync_start, _fmt_date(end))
        raw, _ = sync.fetch_daily(norm_symbols, sync_start, _fmt_date(end), calendar=calendar, basic=basic)
        bench_start = (
            incremental_window(manifest, "benchmark", end)
            if inc and manifest is not None
            else _fmt_date(start)
        )
        bench_raw = sync._query(
            sync._bs.query_history_k_data_plus,
            symbol_to_bs(benchmark),
            "date,close",
            start_date=bench_start,
            end_date=_fmt_date(end),
            frequency="d",
            adjustflag="3",
        )

    if raw.empty or bench_raw is None or bench_raw.empty:
        raise RuntimeError("baostock 未返回任何数据，请检查日期范围与代码格式")
    if inc and storage is not None and storage.has("prices"):
        existing = storage.load("prices")
        # ---- 复权接缝自愈（审计 P0-1 运行期防线）----
        # 前复权价以抓取时刻为锚：除权后增量拼接会在接缝处产生虚假跳空。
        # 检测 ①历史库内已有断裂 ②本次增量边界的锚点漂移，
        # 命中 symbol 从原始 start 全量重拉，使整条历史回到同一锚点。
        from quant.data.baostock_sync import detect_adjustment_breaks, detect_anchor_shift

        import json as _json

        quirks_path = storage.root / "cache" / "adjustment_quirks.json"
        known_quirks: set[str] = set()
        try:
            if quirks_path.exists():
                known_quirks = set(_json.loads(quirks_path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            pass

        affected = set(detect_adjustment_breaks(existing))
        boundary_shifted = set(
            detect_anchor_shift(existing, raw, sync_start)
        )
        if boundary_shifted:
            log.warning(
                "增量边界检测到 %d 只股票复权锚点漂移（除权事件）: %s...",
                len(boundary_shifted), sorted(boundary_shifted)[:5],
            )
            affected |= boundary_shifted
        # 已知数据源固有缺陷（退市整理期复牌日 baostock 复权序列自身不连续，
        # 全量重拉无法消除）：不再重复重拉，仅保留告警
        to_refetch = sorted(affected - known_quirks)
        if to_refetch:
            log.warning("全量重拉 %d 只股票以修复复权接缝…", len(to_refetch))
            cal_full = sync.trading_calendar(_fmt_date(start), _fmt_date(end))
            full_frames, _ = sync.fetch_daily(
                to_refetch,
                _fmt_date(start),
                _fmt_date(end),
                calendar=cal_full,
                basic=basic,
            )
            keep = existing[~existing["symbol"].isin(to_refetch)]
            raw = merge_incremental(
                merge_incremental(keep, full_frames), raw
            )
            # 重拉后仍断裂的 symbol 属数据源固有缺陷，记入白名单
            still = set(detect_adjustment_breaks(raw[raw["symbol"].isin(to_refetch)]))
            new_quirks = still - known_quirks
            if new_quirks:
                log.warning(
                    "%d 只股票为 baostock 数据源固有断裂（退市整理期），已记录白名单: %s",
                    len(new_quirks), sorted(new_quirks),
                )
                known_quirks |= new_quirks
                quirks_path.parent.mkdir(parents=True, exist_ok=True)
                quirks_path.write_text(
                    _json.dumps(sorted(known_quirks)), encoding="utf-8"
                )
        else:
            if affected:
                log.info("检测到 %d 只已知源缺陷股，跳过重拉（见 adjustment_quirks.json）", len(affected))
            raw = merge_incremental(existing, raw)

    benchmark_df = pd.DataFrame(
        {
            "date": pd.to_datetime(bench_raw["date"]),
            "close": pd.to_numeric(bench_raw["close"], errors="coerce"),
        }
    ).dropna()
    if inc and storage is not None and storage.has("benchmark"):
        existing_b = storage.load("benchmark")
        existing_b["date"] = pd.to_datetime(existing_b["date"])
        benchmark_df["date"] = pd.to_datetime(benchmark_df["date"])
        benchmark_df = (
            pd.concat([existing_b, benchmark_df], ignore_index=True)
            .drop_duplicates(subset=["date"], keep="last")
            .sort_values("date")
            .reset_index(drop=True)
        )
    meta = basic[basic["symbol"].isin(norm_symbols)][
        ["symbol", "name", "list_date", "delist_date"]
    ].copy()
    industry = industry[industry["symbol"].isin(norm_symbols)].copy()
    industry["as_of_date"] = pd.to_datetime(industry["as_of_date"], errors="coerce")

    # ---- PIT 成分快照（幸存者偏差消除，增量式）----
    # 用月末序列拉取历史成员并集；与当前快照 universe 的差异部分
    # （退市/调出股）也一并纳入，保证历史时点的横截面真实。
    # 增量语义：已有快照月份不重拉（省 4~5 分钟/日），仅补缺失月份。
    constituents_df = pd.DataFrame(columns=["snapshot_date", "symbol"])
    try:
        existing_pit = (
            storage.load("constituents")
            if storage is not None and storage.has("constituents")
            else pd.DataFrame(columns=["snapshot_date", "symbol"])
        )
        if not benchmark_df.empty:
            month_ends = [
                str(x)[:10]
                for x in benchmark_df.set_index("date")["close"]
                .resample("ME")
                .last()
                .index
            ]
            done: set[str] = set()
            if not existing_pit.empty:
                done = {
                    str(pd.Timestamp(d).date())
                    for d in existing_pit["snapshot_date"].unique()
                }
            todo = [d for d in month_ends if d[:10] not in done]
            if todo:
                log.info("PIT 成分快照需补拉 %d/%d 期", len(todo), len(month_ends))
                fresh = sync.constituents_pit(todo)
            else:
                fresh = pd.DataFrame(columns=["snapshot_date", "symbol"])
            if not existing_pit.empty or not fresh.empty:
                constituents_df = pd.concat(
                    [existing_pit, fresh], ignore_index=True
                ).drop_duplicates(subset=["snapshot_date", "symbol"])
                # concat 后可能混入 object 类型的 Timestamp（旧库 datetime64
                # + 新拉 Timestamp），统一规范化，否则 to_parquet 无法推断类型
                constituents_df["snapshot_date"] = pd.to_datetime(
                    constituents_df["snapshot_date"]
                ).astype("datetime64[ns]")
                constituents_df = constituents_df.sort_values(
                    ["snapshot_date", "symbol"]
                ).reset_index(drop=True)
            if not constituents_df.empty:
                log.info(
                    "PIT 成分快照: %d 期 / 覆盖股票 %d 只（本期新增 %d 期）",
                    constituents_df["snapshot_date"].nunique(),
                    constituents_df["symbol"].nunique(),
                    len(fresh),
                )
    except Exception as exc:  # noqa: BLE001
        log.warning("PIT 成分拉取失败（回退当前快照口径，存在幸存者偏差）: %s", exc)

    return DataBundle(
        prices=raw,
        benchmark=benchmark_df,
        meta=meta,
        industry=industry,
        fundamentals=pd.DataFrame(),
        constituents=constituents_df,
    )


def load_csv_bundle(root: str | Path, benchmark_file: str = "benchmark.csv") -> DataBundle:
    """从目录读取每只股票一个 CSV（date,open,high,low,close,volume,amount,turnover）。"""
    root = Path(root)
    frames = []
    for f in sorted(root.glob("*.csv")):
        if f.name == benchmark_file:
            continue
        df = pd.read_csv(f, parse_dates=["date"])
        df["symbol"] = f.stem
        frames.append(df)
    prices = pd.concat(frames, ignore_index=True)
    bench = pd.read_csv(root / benchmark_file, parse_dates=["date"])
    return DataBundle(prices=prices, benchmark=bench)
