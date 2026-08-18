"""因子注册表：10 个核心因子（进攻 6 + 防御 4），含计算实现。"""

from __future__ import annotations

import numpy as np
import pandas as pd


class Panels:
    """宽表面板集：date × symbol。"""

    def __init__(
        self,
        close: pd.DataFrame,
        volume: pd.DataFrame,
        amount: pd.DataFrame,
        turnover: pd.DataFrame,
        pe: pd.DataFrame,
        pb: pd.DataFrame,
        roe: pd.DataFrame,
        gross_margin: pd.DataFrame,
        div_yield: pd.DataFrame,
        sentiment: pd.DataFrame | None = None,
        forecast: pd.DataFrame | None = None,
        industry: pd.Series | None = None,
        consensus: pd.DataFrame | None = None,
    ):
        self.close = close
        self.volume = volume
        self.amount = amount
        self.turnover = turnover
        self.pe = pe
        self.pb = pb
        self.roe = roe
        self.gross_margin = gross_margin
        self.div_yield = div_yield
        self.sentiment = sentiment
        self.forecast = forecast
        self.industry = industry
        self.consensus = consensus

    def get(self, name: str) -> pd.DataFrame:
        return getattr(self, name)


def _f_mom_12_1(p: Panels) -> pd.DataFrame:
    c = p.close
    return c.shift(21) / c.shift(252) - 1.0


def _f_rev_5(p: Panels) -> pd.DataFrame:
    return -(p.close / p.close.shift(5) - 1.0)


def _f_vol_ratio(p: Panels) -> pd.DataFrame:
    v = p.volume
    ma5 = v.rolling(5, min_periods=3).mean()
    ma120 = v.rolling(120, min_periods=20).mean()
    return np.log((ma5 + 1.0) / (ma120 + 1.0))


def _f_sentiment(p: Panels) -> pd.DataFrame:
    if p.sentiment is None or p.sentiment.empty:
        return pd.DataFrame(0.5, index=p.close.index, columns=p.close.columns)
    # 情绪已在拉取层按月末窗口聚合（公告日得分），这里直接使用
    return p.sentiment.fillna(0.0)


def _f_earnings_forecast(p: Panels) -> pd.DataFrame:
    """业绩预告净利增速（公告日点内时间，事件驱动 alpha）。"""
    if p.forecast is None or p.forecast.empty:
        return pd.DataFrame(0.0, index=p.close.index, columns=p.close.columns)
    return p.forecast.fillna(0.0)


def _f_industry_mom(p: Panels) -> pd.DataFrame:
    ret20 = p.close / p.close.shift(20) - 1.0
    if p.industry is None or p.industry.empty:
        return ret20
    long = ret20.stack().rename("value").reset_index()
    long["industry"] = long["symbol"].map(p.industry)
    long = long.dropna(subset=["value", "industry"])
    long["industry_mom"] = long.groupby(["date", "industry"])["value"].transform("median")
    return long.pivot(index="date", columns="symbol", values="industry_mom").reindex(
        index=ret20.index, columns=ret20.columns
    )


def _f_valuation(p: Panels) -> pd.DataFrame:
    # 盈利收益率 EP=1/PE 与账面市值比 BP=1/PB 的截面分位复合（越高越便宜）
    ep = (1.0 / p.pe.replace(0, pd.NA)).rank(axis=1, pct=True)
    bp = (1.0 / p.pb.replace(0, pd.NA)).rank(axis=1, pct=True)
    return 0.5 * (ep + bp)


def _f_low_vol(p: Panels) -> pd.DataFrame:
    ret = p.close.pct_change()
    return -ret.rolling(20, min_periods=10).std()


def _f_div_yield(p: Panels) -> pd.DataFrame:
    if p.div_yield is None or p.div_yield.empty:
        return pd.DataFrame(0.0, index=p.close.index, columns=p.close.columns)
    return p.div_yield.ffill()


def _f_quality(p: Panels) -> pd.DataFrame:
    if (p.roe is None or p.roe.empty) and (p.gross_margin is None or p.gross_margin.empty):
        return pd.DataFrame(0.0, index=p.close.index, columns=p.close.columns)
    roe = p.roe if p.roe is not None and not p.roe.empty else pd.DataFrame(0.0, index=p.close.index, columns=p.close.columns)
    gm = p.gross_margin if p.gross_margin is not None and not p.gross_margin.empty else pd.DataFrame(0.0, index=p.close.index, columns=p.close.columns)
    # 质量 = ROE 水平 + 毛利率水平（预处理层再做 z-score，尺度由标准化处理）。
    # 已实测「ROE 稳定性（滚动 8 期均值/标准差）」为负结果：把 quality IC 从 0.030/ICIR 0.364
    # 打到 0.003/ICIR 0.032，故保持纯水平口径，稳定性暂不纳入。
    return 0.5 * (roe.fillna(0.0) + gm.fillna(0.0))


def _f_crowding(p: Panels) -> pd.DataFrame:
    # 拥挤度 = 换手率水平 z-score + 换手加速度（60日/120日）
    t60 = p.turnover.rolling(60, min_periods=20).mean()
    t120 = p.turnover.rolling(120, min_periods=30).mean()
    z = t60.sub(t60.mean(axis=1), axis=0).div(t60.std(axis=1).replace(0, np.nan), axis=0)
    acc = np.log((t60 + 1.0) / (t120 + 1.0))
    return -(z + acc)


def _f_illiquidity(p: Panels) -> pd.DataFrame:
    """Amihud 非流动性：20 日平均 |收益|/成交额，取负（低非流动性更好）。"""
    ret = p.close.pct_change().abs()
    amt = p.amount.replace(0, np.nan)
    illiq = (ret / amt).rolling(20, min_periods=10).mean()
    return -illiq


def _f_max_ret(p: Panels) -> pd.DataFrame:
    """MAX 效应：20 日最大单日涨幅，取负（彩票偏好股表现差）。"""
    ret = p.close.pct_change()
    return -ret.rolling(20, min_periods=10).max()


def _f_consensus_revision(p: Panels) -> pd.DataFrame:
    """分析师一致预期 EPS 修正（30 日变化率）。

    数据来源：consensus 快照序列（每日采集累积）。
    revision = 当前一致预期EPS / 30个交易日前一致预期EPS - 1，正值=上修（看多信号）。
    数据不足（<2 个快照或缺失）时返回 0（中性，不引入错误信号）。
    """
    if p.consensus is None or p.consensus.empty:
        return pd.DataFrame(0.0, index=p.close.index, columns=p.close.columns)
    # 快照可能非日频：先对齐到交易日历（asof 前向取最近快照），
    # 再按 30 个交易日位移，避免“行位移”在周频/不规则快照下语义漂移。
    aligned = p.consensus.reindex(index=p.close.index, method="ffill")
    rev = aligned / aligned.shift(30) - 1.0
    rev = rev.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return rev.reindex(index=p.close.index, columns=p.close.columns).fillna(0.0)


def _f_size_proxy(p: Panels) -> pd.DataFrame:
    """规模代理：60 日均成交额取负（小市值溢价，注意 2024 踩踏尾部风险）。"""
    amt = p.amount.rolling(60, min_periods=20).mean()
    return -np.log(amt + 1.0)


FACTOR_SPECS: dict[str, dict] = {
    "mom_12_1": {"category": "momentum", "direction": +1, "horizon": 20, "compute": _f_mom_12_1},
    "rev_5": {"category": "reversal", "direction": +1, "horizon": 5, "compute": _f_rev_5},
    "vol_ratio": {"category": "volume", "direction": +1, "horizon": 20, "compute": _f_vol_ratio},
    "sentiment": {"category": "sentiment", "direction": +1, "horizon": 7, "compute": _f_sentiment},
    # 注：earnings_forecast 因子（业绩预告）实测为弱信号（IC≈0.02），且其加入改变了
    # composite 成分导致回测塌方（见 FINAL_VERDICT 负结果记录），暂从因子集移除。
    # "earnings_forecast": {"category": "event", "direction": +1, "horizon": 20, "compute": _f_earnings_forecast},
    "industry_mom": {"category": "industry", "direction": +1, "horizon": 20, "compute": _f_industry_mom},
    "valuation": {"category": "valuation", "direction": +1, "horizon": 20, "compute": _f_valuation},
    # direction 约定：对「已取负」的因子（低波/拥挤/非流动性/MAX/规模/反转）值为 +1，
    # 因为 compute 函数已经做了「高值 = 更优」的符号翻转；IC 报告会再用 direction 调整符号，
    # 若此处再写 -1 会造成双重取负。
    "low_vol": {"category": "defensive", "direction": +1, "horizon": 20, "compute": _f_low_vol},
    "div_yield": {"category": "defensive", "direction": +1, "horizon": 20, "compute": _f_div_yield},
    "quality": {"category": "defensive", "direction": +1, "horizon": 20, "compute": _f_quality},
    "crowding": {"category": "defensive", "direction": +1, "horizon": 20, "compute": _f_crowding},
    "illiquidity": {"category": "liquidity", "direction": +1, "horizon": 20, "compute": _f_illiquidity},
    "max_ret": {"category": "behavioral", "direction": +1, "horizon": 20, "compute": _f_max_ret},
    "size_proxy": {"category": "size", "direction": +1, "horizon": 20, "compute": _f_size_proxy},
    "consensus_revision": {
        "category": "event", "direction": +1, "horizon": 20,
        "compute": _f_consensus_revision,
        # 说明：需先运行一致预期采集器累积 ≥30 日快照；数据不足时返回 0（中性）
        "requires": "consensus_snapshot",
    },
}


def factor_direction(name: str) -> int:
    return FACTOR_SPECS[name]["direction"]
