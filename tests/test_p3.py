"""P3 前置功能测试：财务因子时点、带式调仓、纸面交易。"""

from __future__ import annotations

import pandas as pd

from quant.backtest.engine import BacktestEngine
from quant.config import Config
from quant.execution.paper import PaperBroker
from quant.factors.compute import build_panels
from quant.data.storage import DataBundle


def _mini_prices() -> pd.DataFrame:
    dates = pd.bdate_range("2023-01-02", periods=60)
    rows = []
    for sym in ["A.SH", "B.SH", "C.SH"]:
        close = 10.0 * (1 + pd.Series(range(len(dates))) * 0.002)
        rows.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "symbol": sym,
                    "open": close * 0.999,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "volume": 1e6,
                    "amount": 1e6 * close * 100,
                    "turnover": 1.0,
                    "is_limit_up": False,
                    "is_limit_down": False,
                    "is_suspended": False,
                    "is_st": False,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def test_fundamentals_point_in_time():
    """公告日在未来时不得使用（杜绝前视）。"""
    prices = _mini_prices()
    fundamentals = pd.DataFrame(
        {
            "symbol": ["A.SH"],
            "as_of_date": [pd.Timestamp("2023-03-01")],  # 数据 3 月才可用
            "roe": [0.20],
            "div_yield": [0.04],
        }
    )
    bundle = DataBundle(prices=prices, benchmark=prices[["date", "close"]].groupby("date").mean().reset_index())
    bundle.fundamentals = fundamentals
    cfg = Config()
    panels = build_panels(bundle, cfg)
    feb_vals = panels.roe.loc[:"2023-02-28", "A.SH"].dropna()
    # 2 月应无 ROE 数据（0 占位后为 0）
    assert (feb_vals == 0.0).all()
    mar_vals = panels.roe.loc["2023-03-01":, "A.SH"].dropna()
    assert (mar_vals == 0.20).any()


def test_band_reduces_turnover():
    dates = pd.bdate_range("2023-01-02", periods=60)
    rows = []
    for sym, drift in zip(["A.SH", "B.SH", "C.SH"], [0.03, 0.01, 0.0]):
        close = 10.0 * (1 + pd.Series(range(len(dates))) * drift)
        rows.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "symbol": sym,
                    "open": close * 0.999,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "volume": 1e6,
                    "amount": 1e6 * close * 100,
                    "turnover": 1.0,
                    "is_limit_up": False,
                    "is_limit_down": False,
                    "is_suspended": False,
                    "is_st": False,
                }
            )
        )
    prices = pd.concat(rows, ignore_index=True)
    bench = pd.DataFrame({"date": pd.unique(prices["date"]), "close": 1000.0})
    dates = pd.DatetimeIndex(pd.unique(prices["date"]))
    month_ends = dates.to_series().groupby(dates.to_period("M")).max()
    tw = pd.DataFrame(
        {
            "date": [d for d in month_ends.tolist() for _ in range(3)],
            "symbol": ["A.SH", "B.SH", "C.SH"] * len(month_ends),
            "weight": [1 / 3] * (3 * len(month_ends)),
        }
    )
    cfg = Config()
    cfg.backtest.start, cfg.backtest.end = str(prices["date"].min().date()), str(prices["date"].max().date())
    cfg.portfolio.top_n = 3
    cfg.portfolio.band = 0.0
    cfg.portfolio.min_overlap = 0.0
    res0 = BacktestEngine(prices, bench, cfg.backtest, cfg.portfolio).run(
        tw, start=cfg.backtest.start, end=cfg.backtest.end
    )
    cfg.portfolio.band = 0.5
    cfg.portfolio.min_overlap = 0.0
    res1 = BacktestEngine(prices, bench, cfg.backtest, cfg.portfolio).run(
        tw, start=cfg.backtest.start, end=cfg.backtest.end
    )
    trades0 = len(res0.trades[res0.trades["status"] == "filled"])
    trades1 = len(res1.trades[res1.trades["status"] == "filled"])
    assert trades1 < trades0


def test_paper_broker_flow():
    prices = _mini_prices()
    broker = PaperBroker(prices, initial_cash=100_000)
    from quant.execution.broker import Order

    broker.place_order(Order(id="1", symbol="A.SH", side="buy", shares=100, price=10.0))
    broker.place_order(Order(id="2", symbol="B.SH", side="buy", shares=200, price=10.0))
    assert broker.portfolio_value() > 0
    assert broker.get_positions()["A.SH"].available == 0  # T+1
    o = Order(id="3", symbol="A.SH", side="sell", shares=100, price=10.0)
    broker.place_order(o)
    assert o.status.value == "rejected"  # 当日买入不可卖（T+1）


def test_storage_datetime_roundtrip(tmp_path):
    """pandas3 混合精度 datetime 经 Parquet 往返不得报错。"""
    from quant.data.storage import Storage

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-06-06", "2024-06-07"]),  # 可能是 ms/s 精度
            "symbol": ["A.SH", "A.SH"],
            "value": [1.0, 2.0],
        }
    )
    st = Storage(tmp_path / "data")
    st.save("test_dt", df, partition_by_symbol=False)
    loaded = st.load("test_dt")
    assert len(loaded) == 2
    assert loaded["date"].dtype == "datetime64[ns]"


def test_exposure_smoothing():
    from quant.backtest.engine import _smooth_exposure

    assert _smooth_exposure(1.0, 0.5, 0.25) == 0.75
    assert _smooth_exposure(0.75, 0.5, 0.25) == 0.5
    assert _smooth_exposure(0.5, 1.0, 0.25) == 0.75
    assert _smooth_exposure(1.0, 0.9, 0.25) == 0.9


def test_sentiment_scoring():
    from quant.data.sentiment_cninfo import score_title

    assert score_title("2024年半年度业绩预增公告") > 0
    assert score_title("股东减持计划公告") < 0
    assert score_title("关于召开股东大会的通知") == 0


def test_overlap_gate_skips_rebalance():
    prices = _mini_prices()
    bench = pd.DataFrame({"date": pd.unique(prices["date"]), "close": 1000.0})
    dates = pd.DatetimeIndex(pd.unique(prices["date"]))
    month_ends = dates.to_series().groupby(dates.to_period("M")).max().tolist()
    cfg = Config()
    cfg.backtest.start, cfg.backtest.end = str(prices["date"].min().date()), str(prices["date"].max().date())
    cfg.portfolio.top_n = 3
    cfg.portfolio.band = 0.0
    # 两个调仓日组合完全一致但权重不同：重叠门槛应跳过第二次调仓
    tw = pd.DataFrame(
        {
            "date": [month_ends[0]] * 3 + [month_ends[1]] * 3,
            "symbol": ["A.SH", "B.SH", "C.SH"] * 2,
            "weight": [1 / 3] * 3 + [0.5, 0.3, 0.2],
        }
    )
    cfg.portfolio.min_overlap = 0.0
    res0 = BacktestEngine(prices, bench, cfg.backtest, cfg.portfolio).run(
        tw, start=cfg.backtest.start, end=cfg.backtest.end
    )
    cfg.portfolio.min_overlap = 0.9
    res1 = BacktestEngine(prices, bench, cfg.backtest, cfg.portfolio).run(
        tw, start=cfg.backtest.start, end=cfg.backtest.end
    )
    n0 = len(res0.trades[res0.trades["status"] == "filled"])
    n1 = len(res1.trades[res1.trades["status"] == "filled"])
    assert n1 < n0


def test_historical_members():
    from quant.data.universe_history import historical_largecap_members

    dates = pd.bdate_range("2023-01-02", periods=60)
    prices = pd.concat(
        [
            pd.DataFrame(
                {
                    "date": dates,
                    "symbol": sym,
                    "close": 10.0 + i,
                    "open": 10.0 + i,
                    "high": 11.0 + i,
                    "low": 9.0 + i,
                    "volume": 1e6,
                    "amount": 1e8,
                }
            )
            for i, sym in enumerate(["A.SH", "B.SH", "C.SH", "D.SH", "E.SH"])
        ],
        ignore_index=True,
    )
    fundamentals = pd.DataFrame(
        {
            "symbol": ["A.SH", "B.SH", "C.SH", "D.SH", "E.SH"],
            "as_of_date": [dates[10], dates[10], dates[10], dates[10], dates[10]],
            "total_share": [1e9, 9e8, 8e8, 7e8, 6e8],
        }
    )
    members = historical_largecap_members(prices, fundamentals, top=3)
    assert members == ["A.SH", "B.SH", "C.SH"]


def test_neutralize_keeps_all_symbols():
    """行业中性化不得截断横截面（历史 bug：部分行业映射导致其余股票全 NaN）。"""
    from quant.factors.compute import _neutralize

    df = pd.DataFrame(
        {"A.SH": [1.0, 2.0], "B.SH": [3.0, 4.0], "C.SH": [5.0, 6.0]},
        index=pd.date_range("2023-01-02", periods=2),
    )
    industry = pd.Series({"A.SH": "X"})
    out = _neutralize(df, industry, None)
    assert set(out.columns) == {"A.SH", "B.SH", "C.SH"}
    assert out.notna().all().all()


def test_composite_factor():
    from quant.factors.composite import build_composite_factor

    dates = pd.bdate_range("2023-01-02", periods=10)
    rows = []
    for f in ["mom_12_1", "low_vol", "sentiment"]:
        for i, d in enumerate(dates):
            for s in ["A.SH", "B.SH", "C.SH", "D.SH"]:
                # mom 看好 A/B，low_vol 看好 A/C（相关 -1/3），sentiment 中性
                mom_bias = 1.0 if s in ("A.SH", "B.SH") else 0.0
                vol_bias = 1.0 if s in ("A.SH", "C.SH") else 0.0
                v = {"mom_12_1": mom_bias, "low_vol": vol_bias, "sentiment": 0.5}.get(f, 0.5)
                rows.append({"date": d, "symbol": s, "factor": f, "value": v})
    factor_long = pd.DataFrame(rows)
    ic_report = {
        "factors": {
            "mom_12_1": {"rank_ic_mean": 0.05, "t_stat": 3.0,
                         "decay": {"5": 0.05, "10": 0.04, "20": 0.03, "40": 0.02}},
            "low_vol": {"rank_ic_mean": 0.04, "t_stat": 2.5,
                        "decay": {"5": 0.04, "10": 0.04, "20": 0.03, "40": 0.02}},
            "sentiment": {"rank_ic_mean": 0.08, "t_stat": 1.0,
                          "decay": {"5": 0.08, "10": 0.07, "20": 0.06, "40": 0.05}},  # t 不足
        }
    }
    comp = build_composite_factor(factor_long, ic_report, n=3, min_t=2.0, weight_by="equal")
    assert set(comp["factor"]) == {"composite"}
    comp_map = comp.set_index(["date", "symbol"])["value"]
    # B.SH：mom=1.0、low_vol=0.0（sentiment t 不足被排除）→ 等权均值 0.5
    assert abs(float(comp_map.loc[(dates[0], "B.SH")]) - 0.5) < 1e-9


def test_beta_scale():
    import numpy as np

    from quant.backtest.engine import compute_beta_scale
    from quant.config import PortfolioConfig

    dates = pd.bdate_range("2023-01-02", periods=80)
    bench_ret = np.linspace(0.0005, 0.002, 80)
    bench_close = 100 * (1 + np.cumsum(bench_ret))
    benchmark = pd.DataFrame({"date": dates, "close": bench_close})

    pf = PortfolioConfig()
    # beta ≈ 0.3 → scale = 0.5/0.3 = 1.67 → clip 1.5
    equity_rows = [
        {"date": d, "portfolio_value": 1e6 * (1 + np.cumsum(bench_ret * 0.3 + 0.0001))[i]}
        for i, d in enumerate(dates)
    ]
    s1 = compute_beta_scale(equity_rows, benchmark, pf)
    assert abs(s1 - pf.beta_scale_max) < 1e-9

    # beta ≈ 1.0 → scale = 0.5 → clip 下限 1.0
    equity_rows2 = [
        {"date": d, "portfolio_value": 1e6 * (1 + np.cumsum(bench_ret * 1.0))[i]}
        for i, d in enumerate(dates)
    ]
    s2 = compute_beta_scale(equity_rows2, benchmark, pf)
    assert abs(s2 - 1.0) < 1e-9


def test_signal_health_decreases_when_signal_fails():
    """top-N 组合近期跑输基准时，信号健康度应降到 floor；首段无历史应为 1.0。"""
    import numpy as np

    from quant.config import PortfolioConfig
    from quant.portfolio.signal_health import compute_signal_health

    cfg = PortfolioConfig()
    cfg.top_n = 5
    cfg.signal_health_window = 60
    cfg.signal_health_floor = 0.5
    cfg.signal_health_scale = 0.10

    dates = pd.bdate_range("2023-01-02", periods=180)
    rows = []
    for j in range(5):
        for i, d in enumerate(dates):
            rows.append(
                {
                    "date": d,
                    "symbol": f"S{j}.SH",
                    "score": 1.0 - 0.1 * j,
                    "close": 100.0 * (1 + 0.0001 * i),  # 全部跑输基准
                }
            )
    df = pd.DataFrame(rows)
    prices = df[["date", "symbol", "close"]]
    scores = df[["date", "symbol", "score"]]
    benchmark = pd.DataFrame({"date": dates, "close": 100.0 * (1 + 0.005 * np.arange(len(dates)))})
    health = compute_signal_health(scores, prices, benchmark, cfg)
    assert float(health.iloc[0]["signal_health"]) == 1.0
    assert float(health.iloc[-1]["signal_health"]) == cfg.signal_health_floor
    assert health["signal_health"].between(cfg.signal_health_floor, 1.0).all()


def test_effective_slippage_adaptive():
    """自适应滑点：按参与率放大冲击并封顶，fixed 模式保持基础滑点。"""
    from quant.backtest.cost import CostModel

    fixed = CostModel()
    assert fixed.effective_slippage_bp(100_000, 1_000_000) == 5.0
    c = CostModel(slippage_model="adaptive", slippage_bp=5.0, slippage_cap_bp=20.0, slippage_impact_coef=5.0)
    assert c.effective_slippage_bp(0, 0) == 5.0
    assert c.effective_slippage_bp(100, 1_000_000) == 10.0   # 0.01% 参与率 -> 冲击 5bp
    assert c.effective_slippage_bp(10_000, 1_000_000) == 20.0  # 1% 参与率 -> 封顶 20bp


def test_compute_smallcap_regime():
    """小盘辅助择时：中证1000 跌破 MA250 时降仓至 floor。"""
    import numpy as np

    from quant.config import PortfolioConfig
    from quant.portfolio.regime import compute_smallcap_regime

    cfg = PortfolioConfig()
    cfg.smallcap_long = 20
    cfg.smallcap_floor = 0.5
    idx = pd.bdate_range("2023-01-02", periods=60)
    up = pd.Series(np.linspace(100, 200, 60), index=idx)
    down = pd.Series(np.linspace(200, 100, 60), index=idx)
    ru = compute_smallcap_regime(up, cfg)
    rd = compute_smallcap_regime(down, cfg)
    assert set(ru["smallcap_level"].unique()) <= {0.5, 1.0}
    assert float(ru["smallcap_level"].iloc[-1]) == 1.0
    assert float(rd["smallcap_level"].iloc[-1]) == cfg.smallcap_floor
