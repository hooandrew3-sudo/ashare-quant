"""baostock 真实数据同步器（生产级）。

功能：
- pandas>=2 兼容垫片（baostock 0.9.x 内部仍使用 DataFrame.append）；
- 股票池：中证800 = 沪深300 + 中证500 当前成分快照（历史成分需 Tushare 积分）；
- 增量同步：仅拉取 manifest 记录末日后 5 个自然日的数据，去重合并；
- 停牌（tradestatus=0 或缺失交易日补行）、ST、退市（meta 记录 out_date）处理；
- 字段：前复权 OHLC、成交额、换手率、PE-TTM、PB-MRQ、涨跌停标记。
"""

from __future__ import annotations

import socket
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from quant.utils import ensure_dir, setup_logging


def ensure_pandas_compat() -> None:
    """baostock 内部调用 DataFrame.append，pandas>=2 已移除，运行时补回。"""
    if not hasattr(pd.DataFrame, "append"):

        def _append(self, other, ignore_index=False, **kwargs):
            return pd.concat([self, other], ignore_index=ignore_index, **kwargs)

        pd.DataFrame.append = _append


def bs_to_symbol(code: str) -> str:
    """'sh.600519' → '600519.SH'；'sz.000001' → '000001.SZ'。"""
    prefix, num = code.split(".")
    return f"{num}.{prefix.upper()}"


def symbol_to_bs(symbol: str) -> str:
    """'600519.SH' → 'sh.600519'。"""
    num, suffix = symbol.split(".")
    return f"{suffix.lower()}.{num}"


def _round_to_tick(px: "pd.Series") -> "pd.Series":
    """按交易所规则四舍五入到分（round half up）。

    注意不能用 Python round()/np.round：它们是银行家舍入
    （round(2.675, 2) == 2.67），会把涨停价算低半分导致封板漏判。
    """
    return np.floor(px.astype(float) * 100.0 + 0.5 + 1e-9) / 100.0


def _limit_ratio_series(
    dates: pd.Series, symbol: str, is_st: pd.Series
) -> pd.Series:
    """按板块 + 日期 + ST 状态返回逐行涨跌停比例。

    规则（A 股现行与历史）：
      - 科创板 688：20%（上市以来）
      - 创业板 300/301：2020-08-24 前 10%（ST 5%），之后 20%
      - 北交所 4/8/92 开头：30%
      - 主板/中小板：10%，ST 5%
    """
    num = str(symbol).split(".")[0]
    dt = pd.to_datetime(dates)
    is_st_arr = is_st.astype(bool)
    if num.startswith("688"):
        return pd.Series(0.20, index=dates.index)
    if num.startswith(("300", "301")):
        after_reform = dt >= pd.Timestamp("2020-08-24")
        return pd.Series(
            np.where(after_reform, 0.20, np.where(is_st_arr, 0.05, 0.10)),
            index=dates.index,
        )
    if num.startswith(("4", "8", "92")):
        return pd.Series(0.30, index=dates.index)
    return pd.Series(
        np.where(is_st_arr, 0.05, 0.10), index=dates.index
    )


@dataclass
class SyncStats:
    symbols: int = 0
    rows: int = 0
    errors: list[str] = None  # type: ignore[assignment]
    elapsed_sec: float = 0.0

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class BaoStockSync:
    """baostock 同步器：登录一次，复用会话；所有查询带重试。"""

    DAILY_FIELDS = (
        "date,code,open,high,low,close,preclose,volume,amount,turn,"
        "tradestatus,pctChg,peTTM,pbMRQ,isST"
    )

    def __init__(
        self,
        verbose: bool = True,
        retries: int = 3,
        pause_sec: float = 0.0,
        cache_dir: str | Path | None = None,
    ):
        ensure_pandas_compat()
        self.log = setup_logging(verbose)
        self.retries = retries
        self.pause_sec = pause_sec
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._bs = None

    # ---------- 会话 ----------
    def connect(self):
        # baostock 查询偶发挂起：给底层 socket 加超时，让挂起变成异常并触发重试/重登
        socket.setdefaulttimeout(30)
        try:
            import baostock as bs
        except ImportError as exc:
            raise RuntimeError("需要安装 baostock：pip install baostock") from exc
        self._bs = bs
        lg = bs.login()
        if lg.error_code != "0":
            raise RuntimeError(f"baostock 登录失败: {lg.error_msg}")
        self.log.info("baostock 登录成功")
        return self

    def close(self):
        if self._bs is not None:
            try:
                self._bs.logout()
            except Exception:  # noqa: BLE001
                pass

    def __enter__(self):
        return self.connect()

    def __exit__(self, *exc):
        self.close()

    def _query(self, fn, *args, **kwargs) -> pd.DataFrame:
        """带重试的查询：失败后重新登录再试。"""
        last_err: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                rs = fn(*args, **kwargs)
                if rs.error_code != "0":
                    raise RuntimeError(f"{rs.error_msg}")
                data = rs.get_data()
                if data is None:
                    return pd.DataFrame()
                return data
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                self.log.warning("baostock 查询失败(第 %d/%d 次): %s", attempt, self.retries, exc)
                if attempt < self.retries:
                    time.sleep(1.5 * attempt)
                    self._bs.login()
        raise RuntimeError(f"baostock 查询重试耗尽: {last_err}")

    def _cached(
        self, key: str, fn, *args, ttl_hours: float = 0.0, **kwargs
    ) -> pd.DataFrame:
        """本地缓存常用查询（股票列表/交易日历），避免每次全量拉取。

        ttl_hours > 0 时按文件 mtime 判断缓存是否过期：过期则重新拉取并覆盖。
        交易日历类数据（不可变）应保持 ttl=0 永久缓存；
        stock_basic / industry 等会漂移的快照必须设置 TTL，否则退市/新上市/
        行业变更会被无限期冻结。
        """
        if self.cache_dir is not None:
            path = self.cache_dir / f"{key}.parquet"
            if path.exists():
                age_h = (pd.Timestamp.now() - pd.Timestamp.fromtimestamp(path.stat().st_mtime)).total_seconds() / 3600.0
                if ttl_hours <= 0 or age_h < ttl_hours:
                    self.log.info("命中缓存 %s (age=%.1fh)", path.name, age_h)
                    return pd.read_parquet(path)
                self.log.info("缓存过期 %s (age=%.1fh > %.1fh)，重新拉取", path.name, age_h, ttl_hours)
        df = self._query(fn, *args, **kwargs)
        if self.cache_dir is not None and not df.empty:
            ensure_dir(self.cache_dir)
            df.to_parquet(self.cache_dir / f"{key}.parquet", index=False)
        return df

    # ---------- 股票池 ----------
    def stock_basic(self) -> pd.DataFrame:
        """全量股票基本信息（含指数与退市股票，由调用方过滤）。"""
        raw = self._cached("stock_basic", self._bs.query_stock_basic, ttl_hours=24.0)
        df = raw.copy()
        df["symbol"] = df["code"].map(bs_to_symbol)
        df = df.rename(
            columns={"code_name": "name", "ipoDate": "list_date", "outDate": "delist_date"}
        )
        df["list_date"] = pd.to_datetime(df["list_date"], errors="coerce")
        df["delist_date"] = pd.to_datetime(df["delist_date"], errors="coerce")
        return df

    def constituents(self, indexes: Iterable[str] = ("hs300", "zz500")) -> list[str]:
        """拉取指数成分（当前快照），返回统一格式 symbol 列表。"""
        symbols: list[str] = []
        for idx in indexes:
            fn = getattr(self._bs, f"query_{idx}_stocks", None)
            if fn is None:
                raise ValueError(f"baostock 不支持指数: {idx}")
            raw = self._query(fn)
            if not raw.empty:
                symbols.extend(raw["code"].map(bs_to_symbol).tolist())
        return sorted(set(symbols))

    def csi800_symbols(self) -> list[str]:
        """中证800 = 沪深300 + 中证500（当前成分快照）。"""
        return self.constituents(("hs300", "zz500"))

    def constituents_pit(self, dates) -> pd.DataFrame:
        """点内成分快照：按给定日期序列拉取 hs300+zz500 成员并集。

        返回 long 表 (snapshot_date, symbol)。指数成分的幸存者偏差（用今日
        名单回测历史）由此消除：每个历史时点只用当时真实的成员集合。
        月末采样即可覆盖半年度调仓（≤1 个月滞后），是业界标准做法。
        """
        rows: list[pd.DataFrame] = []
        date_list = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in dates]
        total = len(date_list)
        for i, ds in enumerate(date_list, start=1):
            codes: set[str] = set()
            for getter in (self._bs.query_hs300_stocks, self._bs.query_zz500_stocks):
                try:
                    df = self._query(getter, ds)
                except Exception as exc:  # noqa: BLE001
                    self.log.warning("成分快照 %s 拉取失败: %s", ds, exc)
                    continue
                if df.empty:
                    continue
                col = "code" if "code" in df.columns else df.columns[0]
                codes |= set(df[col].astype(str))
            if codes:
                rows.append(
                    pd.DataFrame(
                        {
                            "snapshot_date": pd.Timestamp(ds),
                            "symbol": sorted(bs_to_symbol(c) for c in codes),
                        }
                    )
                )
            if i % 12 == 0 or i == total:
                self.log.info("PIT 成分快照进度 %d/%d", i, total)
        if not rows:
            return pd.DataFrame(columns=["snapshot_date", "symbol"])
        out = pd.concat(rows, ignore_index=True).drop_duplicates(
            subset=["snapshot_date", "symbol"]
        )
        return out.sort_values(["snapshot_date", "symbol"]).reset_index(drop=True)

    def industry(self) -> pd.DataFrame:
        """全市场行业分类（证监会行业，当前快照，静态口径）。"""
        raw = self._cached("industry", self._bs.query_stock_industry, ttl_hours=24.0)
        if raw.empty:
            return pd.DataFrame(columns=["symbol", "industry", "as_of_date"])
        df = raw.copy()
        df["symbol"] = df["code"].map(bs_to_symbol)
        df = df.rename(columns={"updateDate": "as_of_date", "industry": "industry"})
        return df[["symbol", "industry", "as_of_date"]]

    # ---------- 交易日历 ----------
    def trading_calendar(self, start: str, end: str) -> pd.DatetimeIndex:
        raw = self._cached(
            f"trade_dates_{start}_{end}",
            self._bs.query_trade_dates, start_date=start, end_date=end
        )
        if raw.empty:
            raise RuntimeError("baostock 未返回交易日历")
        dates = raw.loc[raw["is_trading_day"] == "1", "calendar_date"]
        return pd.DatetimeIndex(sorted(pd.to_datetime(dates)))

    # ---------- 日线同步 ----------
    def fetch_daily(
        self,
        symbols: Iterable[str],
        start: str,
        end: str,
        calendar: pd.DatetimeIndex | None = None,
        basic: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, SyncStats]:
        """同步多只股票日线（前复权），统一清洗后返回长表。

        basic：stock_basic 结果（用于上市/退市日期过滤与停牌补行范围）。
        """
        stats = SyncStats()
        t0 = time.time()
        frames: list[pd.DataFrame] = []
        basic_by_sym = {}
        if basic is not None and not basic.empty:
            basic_by_sym = basic.set_index("symbol").to_dict("index")

        syms = sorted(set(symbols))
        for i, sym in enumerate(syms, 1):
            try:
                df = self._fetch_daily_one(sym, start, end)
                if df.empty:
                    self.log.info("[%d/%d] %s 无数据", i, len(syms), sym)
                    continue
                if calendar is not None:
                    df = self._fill_suspensions(df, sym, calendar, basic_by_sym.get(sym))
                frames.append(df)
                stats.rows += int(len(df))
                if i % 50 == 0 or i == len(syms):
                    self.log.info("同步进度 %d/%d（累计 %d 行）", i, len(syms), stats.rows)
            except Exception as exc:  # noqa: BLE001
                stats.errors.append(f"{sym}: {exc}")
                self.log.error("同步失败 %s: %s", sym, exc)
            if self.pause_sec:
                time.sleep(self.pause_sec)
        stats.symbols = len(syms)
        stats.elapsed_sec = round(time.time() - t0, 1)
        self.log.info(
            "同步完成: %d 只 / %d 行 / %.1fs / %d 个错误",
            stats.symbols, stats.rows, stats.elapsed_sec, len(stats.errors),
        )
        # 错误率阈值：大面积失败时缺失数据静默入库比失败本身更危险
        # （某只股票从某天起永远缺席，横截面因子每天悄悄少一只票）
        if syms and len(stats.errors) / len(syms) > 0.2:
            raise RuntimeError(
                f"同步错误率 {len(stats.errors) / len(syms):.0%} 超过 20% 阈值"
                f"（{len(stats.errors)}/{len(syms)} 只失败），拒绝入库。"
                f"首批错误: {stats.errors[:3]}"
            )
        if frames:
            out = pd.concat(frames, ignore_index=True)
            return out.sort_values(["symbol", "date"]).reset_index(drop=True), stats
        return pd.DataFrame(), stats

    def _fetch_daily_one(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        raw = self._query(
            self._bs.query_history_k_data_plus,
            symbol_to_bs(symbol),
            self.DAILY_FIELDS,
            start_date=start,
            end_date=end,
            frequency="d",
            adjustflag="2",  # 前复权
        )
        if raw.empty:
            return pd.DataFrame()
        raw = raw.replace("", pd.NA)
        traded = raw["tradestatus"].astype(str) == "1"
        prices = raw[traded].copy()
        if prices.empty:
            # 区间内全部停牌：仍保留日历补行
            return pd.DataFrame()
        for col in ("open", "high", "low", "close", "preclose"):
            prices[col] = pd.to_numeric(prices[col], errors="coerce")
        prices["volume"] = pd.to_numeric(prices["volume"], errors="coerce").fillna(0.0)
        prices["amount"] = pd.to_numeric(prices["amount"], errors="coerce").fillna(0.0)
        prices["turnover"] = pd.to_numeric(prices["turn"], errors="coerce").fillna(0.0)
        prices["pe"] = pd.to_numeric(prices["peTTM"], errors="coerce")
        prices["pb"] = pd.to_numeric(prices["pbMRQ"], errors="coerce")
        prices["is_st"] = prices["isST"].astype(str) == "1"
        # 涨跌停比例按板块 + 时间演进（创业板 2020-08-24 起 20%、科创板 20%、
        # 北交所 30%、主板/中小板 10%、ST 5%）。收盘封板判定：close 必须
        # ≥ 交易所规则涨停价 round_half_up(前收×(1+ratio), 2)。
        # 此前用 pctChg≥ratio*100-0.2 的宽容差，+9.85% 未封板会被误判涨停。
        ratio = _limit_ratio_series(prices["date"], symbol, prices["is_st"])
        limit_up_px = _round_to_tick(prices["preclose"] * (1.0 + ratio))
        limit_dn_px = _round_to_tick(prices["preclose"] * (1.0 - ratio))
        close_ok = prices["close"].notna() & prices["preclose"].notna()
        eps = 1e-4  # 浮点比较容差（半分以下）
        prices["is_limit_up"] = close_ok & (prices["close"] >= limit_up_px - eps)
        prices["is_limit_down"] = close_ok & (prices["close"] <= limit_dn_px + eps)
        pre = prices["preclose"]
        open_px = prices["open"]
        open_ok = pre.notna() & open_px.notna() & (pre > 0)
        prices["is_limit_up_open"] = open_ok & (open_px >= limit_up_px - eps)
        prices["is_limit_down_open"] = open_ok & (open_px <= limit_dn_px + eps)
        prices["is_suspended"] = False
        # 复权因子字段：baostock 日线接口已不支持 factor 字段（2026-08 起返回
        # “指标不存在:factor”）。当前存储为常量占位，保留 schema 稳定；
        # 彻底方案（不复权价 + query_adjust_factor 复权因子）见审计报告 P1-2。
        prices["adj_factor"] = 1.0
        # 落库 preclose：前复权序列中 preclose 恒等于库内前一交易日 close
        # （同锚点），据此可精确检测增量拼接的锚点漂移（复权接缝），
        # 替代"单日涨跌幅超限"启发式（后者会把退市整理期合法暴跌误判为断裂）
        return prices[
            [
                "date", "open", "high", "low", "close", "preclose", "volume", "amount",
                "turnover", "pe", "pb", "is_limit_up", "is_limit_down",
                "is_limit_up_open", "is_limit_down_open",
                "is_suspended", "is_st", "adj_factor",
            ]
        ].assign(symbol=symbol, date=pd.to_datetime(prices["date"]))

    def _fill_suspensions(
        self,
        df: pd.DataFrame,
        symbol: str,
        calendar: pd.DatetimeIndex,
        basic: dict | None,
    ) -> pd.DataFrame:
        """按交易日历补行：未上市/已退市区间裁剪，缺失交易日标记停牌。"""
        dmin, dmax = calendar.min(), calendar.max()
        if basic:
            listed = pd.Timestamp(basic.get("list_date")) if pd.notna(basic.get("list_date")) else dmin
            delisted = (
                pd.Timestamp(basic.get("delist_date"))
                if pd.notna(basic.get("delist_date"))
                else dmax
            )
            dmin = max(dmin, listed)
            dmax = min(dmax, delisted)
        if dmin > dmax:
            return df
        # 防御：补行范围不超过本次实际拉取的数据范围，避免用全量日历覆盖历史
        df_range_min = df["date"].min()
        df_range_max = df["date"].max()
        cal = calendar[
            (calendar >= dmin)
            & (calendar <= dmax)
            & (calendar >= df_range_min)
            & (calendar <= df_range_max)
        ]
        if cal.empty:
            return df
        df = df.set_index("date")
        df = df.reindex(cal)
        missing = df["close"].isna()
        if missing.any():
            for col in ("open", "high", "low", "close", "preclose"):
                if col in df.columns:
                    df.loc[missing, col] = pd.NA
            df.loc[missing, "volume"] = 0.0
            df.loc[missing, "amount"] = 0.0
            df.loc[missing, "turnover"] = 0.0
            df.loc[missing, "is_suspended"] = True
            df.loc[missing, "is_limit_up"] = False
            df.loc[missing, "is_limit_down"] = False
            df.loc[missing, "is_limit_up_open"] = False
            df.loc[missing, "is_limit_down_open"] = False
            if "is_st" in df.columns:
                df.loc[missing, "is_st"] = df.loc[missing, "is_st"].fillna(False)
            if "pe" in df.columns:
                df.loc[missing, "pe"] = pd.NA
                df.loc[missing, "pb"] = pd.NA
        df = df.reset_index().rename(columns={"index": "date"})
        df["symbol"] = symbol
        return df


def merge_incremental(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """增量合并：同 (date, symbol) 以新数据为准，去重后按时间排序。"""
    if existing is None or existing.empty:
        return normalize_flag_columns(new).sort_values(["symbol", "date"]).reset_index(drop=True)
    combined = pd.concat([existing, new], ignore_index=True)
    combined = combined.drop_duplicates(subset=["date", "symbol"], keep="last")
    combined = normalize_flag_columns(combined)
    return combined.sort_values(["symbol", "date"]).reset_index(drop=True)


def detect_adjustment_breaks(
    prices: pd.DataFrame,
    ret_threshold: float = 0.35,
    skip_head_bars: int = 5,
) -> list[str]:
    """检测已入库价格中的复权断裂（前视审计 P0-1 的运行期防线）。

    前复权价以"抓取时刻"为锚：两次同步之间若发生除权，接缝两侧比例尺
    不同。返回需要全量重拉的 symbol 列表（重拉后整条历史用同一锚点，
    接缝自愈）。

    两级判定：
    - 精确规则（行含 preclose）：同锚点前复权序列中 preclose 应等于库内
      前一交易日 close；不等即为锚点漂移（接缝）。对除权日、退市整理期
      暴跌等合法行情零误报。
    - 启发式（遗留行缺 preclose）：单日 |收益| > ret_threshold 才标记，
      可能包含退市整理期首日等合法行情——重拉幂等无害，仅多一次带宽。
    跳过每只股票头部 skip_head_bars 根 K 线（新股上市初期无涨跌幅限制）。
    """
    if prices is None or prices.empty or "close" not in prices.columns:
        return []
    d = prices.dropna(subset=["close"])
    d = d.sort_values(["symbol", "date"])
    # 跳过每只股票头部 skip_head_bars 根 K 线（新股上市初期无涨跌幅限制）
    d = d[d.groupby("symbol").cumcount() >= skip_head_bars]
    if d.empty:
        return []
    bad_syms: set[str] = set()

    has_preclose = "preclose" in d.columns and d["preclose"].notna().any()
    if has_preclose:
        # 按 symbol 划分（非按行）：只要该股任一行有 preclose 就走精确规则，
        # 其余行是停牌补行（close/preclose 均为 NaN）。若按行划分，启发式的
        # pct_change 会跨停牌缺口计算，把退市整理期首日暴跌误判为接缝。
        sym_has_pc = d.groupby("symbol")["preclose"].transform(
            lambda s: s.notna().any()
        ).astype(bool)
        exact = d[sym_has_pc]
        legacy = d[~sym_has_pc]
        prev_close = exact.groupby("symbol")["close"].shift(1)
        pc = pd.to_numeric(exact["preclose"], errors="coerce")
        # 锚点漂移判定：偏差超过 max(1.5 跳, 前收 2%)。容差依据：
        # ① 仙股前复权分位舍入噪声 ±0.01~0.02；② baostock 对个别
        # 退市整理股在停牌日前后的 preclose 有 ±1 跳不一致。
        # 真实接缝来自除权再缩放，偏差为百分比级，远高于该阈值。
        tol = np.maximum(0.015, prev_close * 0.02).fillna(0.015)
        mismatch = ((pc - prev_close).abs() > tol).fillna(False)
        seam_mask = (pc.notna() & prev_close.notna() & mismatch).fillna(False)
        if seam_mask.any():
            bad_syms |= set(exact.loc[seam_mask, "symbol"].unique())
    else:
        legacy = d

    if not legacy.empty:
        ret = legacy.groupby("symbol")["close"].pct_change()
        heur_bad = legacy[(ret.abs() > ret_threshold).fillna(False)]
        bad_syms |= set(heur_bad["symbol"].unique())
    return sorted(bad_syms)


def detect_anchor_shift(
    existing: pd.DataFrame,
    new_raw: pd.DataFrame,
    window_start: str,
    threshold: float = 0.35,
) -> list[str]:
    """检测增量边界处的复权锚点漂移。

    对比旧库窗口起点前的最后收盘价与新数据首个收盘价：同一锚点下该比值
    应为正常日度涨跌幅；超过阈值说明两次同步之间发生了除权再缩放，
    该 symbol 必须全量重拉，否则接缝将永久留在库中。
    """
    if existing is None or existing.empty or new_raw is None or new_raw.empty:
        return []
    start_ts = pd.Timestamp(window_start)
    old = existing[existing["date"] < start_ts]
    last_old = (
        old.sort_values("date").groupby("symbol")["close"].last().dropna()
    )
    first_new = (
        new_raw.sort_values("date").groupby("symbol")["close"].first().dropna()
    )
    joined = pd.concat([last_old.rename("prev"), first_new.rename("nxt")], axis=1).dropna()
    if joined.empty:
        return []
    mask = (joined["prev"] > 0) & ((joined["nxt"] / joined["prev"] - 1.0).abs() > threshold)
    return sorted(joined[mask].index.tolist())


FLAG_COLUMNS = (
    "is_limit_up",
    "is_limit_down",
    "is_limit_up_open",
    "is_limit_down_open",
    "is_suspended",
    "is_st",
)


def normalize_flag_columns(df: pd.DataFrame) -> pd.DataFrame:
    """把涨跌停/停牌/ST 标记列规范化为 bool，NaN（历史数据缺列）按 False 处理。

    增量合并时旧数据可能缺失新列（如 is_limit_up_open），pd.concat 会把整列
    抬升为 float64/object，导致 fastparquet 无法推断类型而落盘失败。
    """
    if df.empty:
        return df
    cols = [c for c in FLAG_COLUMNS if c in df.columns]
    if not cols:
        return df
    out = df.copy()
    for c in cols:
        out[c] = out[c].fillna(False).astype(bool)
    return out


def incremental_window(manifest: dict | None, dataset: str, end: str, buffer_days: int = 5) -> str:
    """根据 manifest 计算增量同步起点（含 buffer 覆盖上次尾部）。"""
    if manifest is None:
        return "1990-12-19"
    entries = manifest.get("datasets", {}).get(dataset, {})
    ends = [
        pd.Timestamp(v.get("end")) for v in entries.values() if v.get("end") is not None
    ]
    if not ends:
        return "1990-12-19"
    last = max(ends) - timedelta(days=buffer_days)
    return min(last, pd.Timestamp(end)).strftime("%Y-%m-%d")
