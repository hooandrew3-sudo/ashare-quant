"""事件驱动月度调仓回测引擎（T+1 / 涨跌停 / 停牌 / 成本 / 止损）。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quant.backtest.cost import CostModel
from quant.backtest.fills import FillSimulator
from quant.config import BacktestConfig, PortfolioConfig
from quant.portfolio.risk import stop_hit
from quant.portfolio.cvar import check_cvar_limit
from quant.utils import setup_logging


@dataclass
class BacktestResult:
    equity: pd.DataFrame
    trades: pd.DataFrame
    monthly: pd.DataFrame
    metrics: dict = field(default_factory=dict)
    stress: dict = field(default_factory=dict)
    positions: dict = field(default_factory=dict)


class BacktestEngine:
    def __init__(
        self,
        prices: pd.DataFrame,
        benchmark: pd.DataFrame,
        bt_cfg: BacktestConfig,
        pf_cfg: PortfolioConfig,
        verbose: bool = True,
    ):
        self.prices = prices
        self.benchmark = benchmark.sort_values("date")
        self.bt = bt_cfg
        self.pf = pf_cfg
        self.log = setup_logging(verbose)
        self.cost = CostModel(
            commission_bp=bt_cfg.commission_bp,
            stamp_bp=bt_cfg.stamp_bp,
            transfer_bp=bt_cfg.transfer_bp,
            slippage_bp=bt_cfg.slippage_bp,
            min_commission=bt_cfg.min_commission,
            lot_size=bt_cfg.lot_size,
            slippage_model=bt_cfg.slippage_model,
            slippage_cap_bp=bt_cfg.slippage_cap_bp,
            slippage_impact_coef=bt_cfg.slippage_impact_coef,
        )
        self.fills = FillSimulator(prices, self.cost)
        cal = self.benchmark["date"] if not self.benchmark.empty else prices["date"]
        self.trading_dates = pd.DatetimeIndex(sorted(pd.unique(cal)))

    # ---------- 主流程 ----------
    def run(
        self,
        target_weights: pd.DataFrame,
        regime: pd.DataFrame | None = None,
        signal_health: pd.DataFrame | None = None,
        smallcap_regime: pd.DataFrame | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> BacktestResult:
        start = pd.Timestamp(start or self.bt.start)
        end = pd.Timestamp(end or self.bt.end)
        dates = self.trading_dates[(self.trading_dates >= start) & (self.trading_dates <= end)]
        if len(dates) == 0:
            raise ValueError("回测区间内无交易日")

        signals = (
            sorted(pd.unique(target_weights["date"]))
            if not target_weights.empty
            else []
        )
        signals = [pd.Timestamp(d) for d in signals if start <= pd.Timestamp(d) <= end]
        target_map: dict[pd.Timestamp, pd.DataFrame] = {}
        if not target_weights.empty:
            for d, g in target_weights.groupby("date"):
                target_map[pd.Timestamp(d)] = g.set_index("symbol")["weight"]
        target_map = {d: w for d, w in sorted(target_map.items())}
        last_target: pd.Series | None = None

        regime_map: dict[pd.Timestamp, float] = {}
        if regime is not None and not regime.empty:
            regime_map = {
                pd.Timestamp(d): float(r)
                for d, r in zip(regime["date"], regime["target_exposure"])
            }
        signal_health_map: dict[pd.Timestamp, float] = {}
        if signal_health is not None and not signal_health.empty:
            signal_health_map = {
                pd.Timestamp(d): float(h)
                for d, h in zip(signal_health["date"], signal_health["signal_health"])
            }
        smallcap_map: dict[pd.Timestamp, float] = {}
        if smallcap_regime is not None and not smallcap_regime.empty:
            smallcap_map = {
                pd.Timestamp(d): float(v)
                for d, v in zip(smallcap_regime["date"], smallcap_regime["smallcap_level"])
            }

        cash = float(self.bt.initial_cash)
        positions: dict[str, dict] = {}   # sym -> {shares, cost, entry}
        order_queue: list[dict] = []      # 待执行订单
        equity_rows: list[dict] = []
        trade_rows: list[dict] = []
        bench0 = float(self.benchmark.iloc[0]["close"])
        current_exposure = 1.0
        prev_equity = float(self.bt.initial_cash)
        no_buy_today: set[str] = set()  # 当日因止损/熔断卖出的标的，禁止同日回补

        for date in dates:
            no_buy_today = set()
            # ---- 1. T+1 解锁 ----
            for sym in positions:
                positions[sym]["available"] = positions[sym]["shares"]

            # ---- 2. 当日收盘估值 ----
            close_val = sum(
                int(p["shares"]) * self.fills.last_close(sym, date)
                for sym, p in positions.items()
            )
            equity = cash + close_val
            bench_close = float(self.benchmark.loc[self.benchmark["date"] == date, "close"].iloc[0])

            # ---- 3. 止损信号（收盘触发，次日开盘卖） ----
            for sym, p in list(positions.items()):
                if p["shares"] <= 0:
                    continue
                cur = self.fills.last_close(sym, date)
                if stop_hit(p["entry"], cur, self.pf.stop_loss):
                    self._queue_sell(
                        order_queue, sym, p["shares"], self._next_trading_day(date), "stop_loss"
                    )

            # ---- 4. 调仓信号（T 收盘决策，T+1 开盘执行） ----
            if date in target_map:
                target_exposure = regime_map.get(date, 1.0)
                # 滚动 beta 目标：低 beta 时提高暴露（指数增强式），封顶 100% 仓位
                target_exposure = min(
                    1.0, target_exposure * compute_beta_scale(equity_rows, self.benchmark, self.pf)
                )
                # 信号失效开关：近期选股未跑赢基准时下调目标仓位
                if self.pf.signal_health_enabled:
                    target_exposure = target_exposure * signal_health_map.get(date, 1.0)
                # 小盘辅助择时：中证1000 跌破 MA250 时降仓
                if self.pf.smallcap_regime_enabled:
                    target_exposure = target_exposure * smallcap_map.get(date, 1.0)
                # 仓位渐变：避免状态机跃变导致整体大额调仓
                exposure = _smooth_exposure(current_exposure, target_exposure, self.pf.exposure_step)
                current_exposure = exposure
                tw = target_map[date]
                # 换手率硬上限：近 252 日实际换手超过 max_turnover_annual 时跳过调仓
                if self._turnover_breached(trade_rows, equity_rows):
                    self.log.warning(
                        "换手率超限（>%.0fx/年），跳过 %s 调仓",
                        self.pf.max_turnover_annual, date.date(),
                    )
                    continue
                # 重叠度门槛：新组合与当前持仓高度重合时跳过调仓（降低噪声换手）
                if self.pf.min_overlap > 0 and last_target is not None:
                    overlap = len(set(tw.index) & set(last_target.index)) / max(
                        len(tw.index), 1
                    )
                    if overlap >= self.pf.min_overlap:
                        self.log.info(
                            "调仓跳过 %s: 组合重叠 %.0f%% ≥ %.0f%%",
                            date.date(), overlap * 100, self.pf.min_overlap * 100,
                        )
                        continue
                last_target = tw
                self._schedule_rebalance(
                    order_queue, positions, tw, equity, exposure, date
                )

            # ---- 5. 执行今日订单（开盘） ----
            orders_today = [o for o in order_queue if o["exec_date"] == date]
            order_queue = [o for o in order_queue if o["exec_date"] != date]
            for order in orders_today:
                cash = self._execute(
                    order, date, positions, cash, order_queue, trade_rows, no_buy_today
                )

            # ---- 6. 执行后收盘估值并记录净值 ----
            equity = cash + sum(
                int(p["shares"]) * self.fills.last_close(sym, date)
                for sym, p in positions.items()
            )
            equity_rows.append(
                {
                    "date": date,
                    "portfolio_value": equity,
                    "benchmark_value": self.bt.initial_cash * bench_close / bench0,
                    "cash": cash,
                    "position_count": sum(1 for p in positions.values() if p["shares"] > 0),
                }
            )
            # ---- 尾部熔断：单日亏损超阈值，次日等比减仓 ----
            if self.bt.circuit_breaker_enabled and prev_equity > 0:
                daily_ret = equity / prev_equity - 1.0
                if daily_ret <= -abs(self.bt.circuit_breaker_daily_dd):
                    next_day = self._next_trading_day(date)
                    reduce = 1.0 - self.bt.circuit_breaker_scale
                    for sym, p in positions.items():
                        if p["shares"] <= 0 or p["available"] <= 0:
                            continue
                        sell_shares = int(
                            p["available"] * reduce // self.bt.lot_size * self.bt.lot_size
                        )
                        if sell_shares > 0:
                            self._queue_sell(order_queue, sym, sell_shares, next_day, "circuit_breaker")
                    self.log.warning("尾部熔断触发 %s: 日亏损 %.2f%%", date.date(), daily_ret * 100)
            # ---- 尾部风险：CVaR 约束（历史模拟法） ----
            if self.pf.cvar_enabled and prev_equity > 0:
                cvar_info = check_cvar_limit(
                    positions,
                    self.prices,
                    cvar_threshold=self.pf.cvar_threshold,
                    lookback=self.pf.cvar_lookback,
                    alpha=self.pf.cvar_alpha,
                )
                if cvar_info["triggered"]:
                    next_day = self._next_trading_day(date)
                    for sym, p in list(positions.items()):
                        if p["shares"] <= 0 or p.get("available", 0) <= 0:
                            continue
                        sell_shares = int(
                            p["shares"] * 0.5 // self.bt.lot_size * self.bt.lot_size
                        )
                        if sell_shares > 0:
                            self._queue_sell(order_queue, sym, sell_shares, next_day, "cvar_limit")
                    self.log.warning(
                        "CVaR 约束触发 %s: cvar=%.4f < %.4f",
                        date.date(),
                        cvar_info["cvar"],
                        cvar_info["threshold"],
                    )
            prev_equity = equity

        equity = pd.DataFrame(equity_rows)
        trades = pd.DataFrame(trade_rows)
        monthly = self._monthly(equity)
        return BacktestResult(equity=equity, trades=trades, monthly=monthly, positions=positions)

    # ---------- 订单 ----------
    def _queue_sell(
        self,
        queue: list[dict],
        symbol: str,
        shares: int,
        date: pd.Timestamp,
        reason: str,
    ) -> None:
        if shares <= 0:
            return
        queue.append(
            {
                "symbol": symbol,
                "side": "sell",
                "shares": int(shares),
                "exec_date": date,
                "days_left": self.bt.postpone_max_days,
                "reason": reason,
            }
        )

    def _schedule_rebalance(
        self,
        queue: list[dict],
        positions: dict[str, dict],
        target: pd.Series,
        equity: float,
        exposure: float,
        signal_date: pd.Timestamp,
    ) -> None:
        """按目标权重生成次日订单。"""
        exec_date = self._next_trading_day(signal_date)
        investable = equity * exposure
        held = set(positions.keys())
        target_syms = set(target.index)

        # 调出：持仓但不在目标中 → 全卖
        for sym in held - target_syms:
            self._queue_sell(queue, sym, positions[sym]["shares"], exec_date, "rebalance_out")

        for sym, w in target.items():
            target_val = investable * float(w)
            if sym in positions and positions[sym]["shares"] > 0:
                cur_val = positions[sym]["shares"] * self.fills.last_close(sym, signal_date)
                diff = target_val - cur_val
                px = self.fills.last_close(sym, signal_date)
                if pd.isna(px) or px <= 0:
                    continue
                # 带式调仓：偏差小于 band×目标值 不交易，显著降低换手
                if abs(diff) < self.pf.band * target_val:
                    continue
                if diff > px * self.bt.lot_size:
                    shares = self.cost.round_lot(diff / px)
                    if shares > 0:
                        queue.append(
                            {
                                "symbol": sym, "side": "buy", "shares": shares,
                                "exec_date": exec_date,
                                "days_left": self.bt.postpone_max_days,
                                "reason": "rebalance",
                            }
                        )
                elif diff < -px * self.bt.lot_size:
                    shares = self.cost.round_lot(-diff / px)
                    shares = min(shares, positions[sym]["available"])
                    self._queue_sell(queue, sym, shares, exec_date, "rebalance")
            else:
                px = self.fills.last_close(sym, signal_date)
                if pd.isna(px) or px <= 0:
                    continue
                shares = self.cost.round_lot(target_val / px)
                if shares > 0:
                    queue.append(
                        {
                            "symbol": sym, "side": "buy", "shares": shares,
                            "exec_date": exec_date,
                            "days_left": self.bt.postpone_max_days,
                            "reason": "rebalance",
                        }
                    )

    def _turnover_breached(self, trade_rows: list[dict], equity_rows: list[dict]) -> bool:
        """近 365 日累计成交额 / 同期平均净值 > max_turnover_annual 时返回 True。"""
        if self.pf.max_turnover_annual <= 0:
            return False
        if len(trade_rows) < 20 or len(equity_rows) < 20:
            return False
        trades = pd.DataFrame(trade_rows[-5000:])
        filled = trades[trades["status"] == "filled"]
        if filled.empty or "date" not in filled.columns:
            return False
        filled = filled.copy()
        filled["date"] = pd.to_datetime(filled["date"])
        cutoff = filled["date"].max() - pd.Timedelta(days=365)
        recent_t = filled[filled["date"] >= cutoff]
        if recent_t.empty:
            return False
        buy_mask = recent_t["side"] == "buy"
        sell_mask = recent_t["side"] == "sell"
        buy_amt = float(
            (recent_t.loc[buy_mask, "price"] * recent_t.loc[buy_mask, "shares"]).sum()
        ) if buy_mask.any() else 0.0
        sell_amt = float(
            (recent_t.loc[sell_mask, "price"] * recent_t.loc[sell_mask, "shares"]).sum()
        ) if sell_mask.any() else 0.0
        # 单边口径：买卖金额取平均，与 performance.turnover_annual 统一
        amount = (buy_amt + sell_amt) / 2.0

        eq = pd.DataFrame(equity_rows[-5000:])
        if eq.empty or "date" not in eq.columns or "portfolio_value" not in eq.columns:
            return False
        eq = eq.copy()
        eq["date"] = pd.to_datetime(eq["date"])
        recent_eq = eq[eq["date"] >= cutoff]
        avg_equity = float(recent_eq["portfolio_value"].mean()) if len(recent_eq) else 0.0
        if avg_equity <= 0:
            return False
        annual_turnover = amount / avg_equity
        return annual_turnover > self.pf.max_turnover_annual

    def _next_trading_day(self, date: pd.Timestamp) -> pd.Timestamp:
        idx = self.trading_dates.searchsorted(date)
        if idx < len(self.trading_dates) - 1:
            return self.trading_dates[idx + 1]
        return date  # 无下一交易日：原地处理（边界）

    def _execute(
        self,
        order: dict,
        date: pd.Timestamp,
        positions: dict[str, dict],
        cash: float,
        queue: list[dict],
        trade_rows: list[dict],
        no_buy_today: set[str],
    ) -> float:
        sym, side, shares = order["symbol"], order["side"], order["shares"]
        fill = self.fills.try_fill(sym, date, side, shares, order["days_left"])

        if fill.status == "filled":
            if side == "buy":
                if sym in no_buy_today:
                    trade_rows.append(
                        {
                            "date": date, "symbol": sym, "side": side,
                            "shares": 0, "price": fill.price, "fee": 0.0,
                            "status": "skipped", "reason": "no_buy_after_stop",
                        }
                    )
                    return cash
                cost_total = fill.price * fill.shares + fill.fee
                if cost_total > cash + 1e-6:
                    # 现金不足：按现金 + 实际费用反解最大可买数量（替代固定 1.003 系数）
                    max_shares = self.cost.round_lot(cash / (fill.price * 1.005))
                    while max_shares > 0:
                        fee_est = self.cost.buy_fee(fill.price * max_shares)
                        if fill.price * max_shares + fee_est <= cash + 1e-6:
                            break
                        max_shares = self.cost.round_lot(max_shares - self.bt.lot_size)
                    if max_shares <= 0:
                        trade_rows.append(
                            {
                                "date": date, "symbol": sym, "side": side,
                                "shares": 0, "price": fill.price, "fee": 0.0,
                                "status": "skipped", "reason": "insufficient_cash",
                            }
                        )
                        return cash
                    fill.shares = max_shares
                    fill.fee = self.cost.buy_fee(fill.price * fill.shares)
                    cost_total = fill.price * fill.shares + fill.fee
                cash -= cost_total
                pos = positions.setdefault(sym, {"shares": 0, "cost": 0.0, "entry": 0.0, "available": 0})
                pos["shares"] += fill.shares
                pos["cost"] += cost_total
                pos["entry"] = pos["cost"] / pos["shares"]
                pos["available"] = 0  # T+1：当日买入不可卖
            else:
                pos = positions.get(sym)
                if pos is None or pos["available"] <= 0:
                    trade_rows.append(
                        {
                            "date": date, "symbol": sym, "side": side,
                            "shares": 0, "price": fill.price, "fee": 0.0,
                            "status": "dropped", "reason": "no_position_or_available",
                        }
                    )
                    return cash
                # 同日多笔卖出（止损+调仓+熔断并发）时按可卖数量截断，杜绝幽灵现金
                fill.shares = min(fill.shares, pos["available"])
                fill.fee = self.cost.sell_fee(fill.price * fill.shares)
                proceeds = fill.price * fill.shares - fill.fee
                cash += proceeds
                entry = pos["entry"]
                pos["available"] -= fill.shares
                pos["cost"] -= entry * fill.shares
                pos["shares"] -= fill.shares
                if pos["shares"] <= 0:
                    positions.pop(sym, None)
                entry_price = entry
                if order["reason"] in ("stop_loss", "circuit_breaker", "cvar_limit"):
                    no_buy_today.add(sym)
            trade_rows.append(
                {
                    "date": date, "symbol": sym, "side": side,
                    "shares": fill.shares, "price": round(fill.price, 4),
                    "entry_price": round(entry_price, 4) if side == "sell" else round(fill.price, 4),
                    "fee": round(fill.fee, 2), "status": "filled", "reason": order["reason"],
                }
            )
        elif fill.status == "postponed":
            # 停牌/跌停/无行情：买卖双方都按 postpone_max_days 顺延，超期丢弃
            order["days_left"] -= 1
            if order["days_left"] > 0:
                order["exec_date"] = self._next_trading_day(date)
                queue.append(order)
            else:
                trade_rows.append(
                    {
                        "date": date, "symbol": sym, "side": side, "shares": 0,
                        "price": 0.0, "fee": 0.0, "status": "dropped",
                        "reason": fill.reason,
                    }
                )
        else:
            trade_rows.append(
                {
                    "date": date, "symbol": sym, "side": side, "shares": 0,
                    "price": 0.0, "fee": 0.0, "status": fill.status,
                    "reason": fill.reason or order["reason"],
                }
            )
        return cash

    def _monthly(self, equity: pd.DataFrame) -> pd.DataFrame:
        if equity.empty:
            return pd.DataFrame(columns=["date", "return", "benchmark_return"])
        e = equity.set_index("date")
        m = e["portfolio_value"].resample("ME").last()
        b = e["benchmark_value"].resample("ME").last()
        return pd.DataFrame(
            {
                "date": m.index,
                "return": m.pct_change().fillna(0.0),
                "benchmark_return": b.pct_change().fillna(0.0),
            }
        ).reset_index(drop=True)


def _smooth_exposure(prev: float, target: float, step: float) -> float:
    """仓位渐变：每期最多朝目标移动 step。"""
    delta = target - prev
    if abs(delta) <= step:
        return target
    return prev + (step if delta > 0 else -step)


def compute_beta_scale(
    equity_rows: list[dict],
    benchmark: pd.DataFrame,
    pf,
) -> float:
    """按组合近 60 日滚动 beta 计算暴露系数：scale = beta_target / beta，clip [1, max]。"""
    if len(equity_rows) < 30:
        return 1.0
    eq = pd.DataFrame(equity_rows).set_index("date")["portfolio_value"]
    p = eq.pct_change().dropna().tail(pf.beta_window)
    b = benchmark.set_index("date")["close"].pct_change().reindex(p.index).dropna()
    df = pd.concat([p, b], axis=1, keys=["p", "b"]).dropna()
    if len(df) < 30:
        return 1.0
    if df["b"].std() < 1e-12 or df["p"].std() < 1e-12:
        return 1.0
    beta = float(np.polyfit(df["b"], df["p"], 1)[0])
    if beta <= 0.05:
        return float(pf.beta_scale_max)
    scale = pf.beta_target / beta
    return float(np.clip(scale, 1.0, pf.beta_scale_max))
