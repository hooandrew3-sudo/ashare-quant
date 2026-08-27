"""miniQMT(XtQuant) 交易适配器：懒加载，未安装时给出明确指引。

⚠️ 生产纪律：本适配器曾存在多处与 xtquant 官方契约不符的实现
（connect 返回值反转、order_type 字符串比较、结果码当委托号），
证明实盘链路从未连通。本次已按官方契约重写，但接入真实资金前必须：
1. 用模拟账户完成冒烟测试（connect → subscribe → 下单 → 查单 → 撤单）；
2. 核对所用券商 xtquant 版本的 xtconstant 常量取值；
3. 确认 OMS 幂等账本已启用（ledger_path），防止断线重发双倍仓位。

环境变量：
  QMT_ACCOUNT_ID / QMT_DATA_DIR / QMT_SESSION（可选，默认随机）
安装：将券商提供的 xtquant 目录加入 sys.path（不同券商版本不通用）。
"""

from __future__ import annotations

import os
import random

from quant.execution.broker import Order, OrderStatus, Position


class QMTBroker:
    """基于 xtquant 的 A 股实盘通道（需券商开通 miniQMT 权限）。"""

    def __init__(
        self,
        data_dir: str | None = None,
        account_id: str | None = None,
        lot_size: int = 100,
    ):
        self.data_dir = data_dir or os.getenv("QMT_DATA_DIR", "")
        self.account_id = account_id or os.getenv("QMT_ACCOUNT_ID", "")
        self.lot_size = int(lot_size)
        self._xt = None
        self._xtc = None
        self._trader = None
        self._account = None

    def connect(self) -> None:
        try:
            from xtquant import xtconstant, xtdata, xttrader  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "未找到 xtquant。请先开通券商 miniQMT 权限，并将券商提供的 "
                "xtquant 目录加入 Python 路径（参见 docs/PRODUCTION_SPEC.md §10）。"
            ) from exc
        from xtquant.xttrader import XtQuantTrader
        from xtquant.xttype import StockAccount

        self._xt = xtdata
        self._xtc = xtconstant
        # 官方契约：connect()/subscribe() 返回 0 表示成功（此前误判为真值，
        # 导致连接成功反而抛异常）；session 固定为 1 会与人工登录冲突，
        # 未显式配置时使用进程内随机 session。
        session_env = os.getenv("QMT_SESSION", "")
        session = int(session_env) if session_env else random.randint(1000, 999999)
        if not self.data_dir:
            raise RuntimeError("缺少 QMT_DATA_DIR（miniQMT userdata 路径）")
        if not self.account_id:
            raise RuntimeError("缺少 QMT_ACCOUNT_ID（资金账号）")
        self._trader = XtQuantTrader(self.data_dir, session)
        self._trader.start()
        rc = self._trader.connect()
        if rc != 0:
            raise RuntimeError(
                f"QMT 交易服务连接失败 (rc={rc})，请检查 QMT_DATA_DIR 与客户端登录状态"
            )
        self._account = StockAccount(self.account_id)
        rc_sub = self._trader.subscribe(self._account)
        if rc_sub != 0:
            raise RuntimeError(f"QMT 账户订阅失败: {self.account_id} (rc={rc_sub})")

    def get_cash(self) -> float:
        self._require()
        asset = self._trader.query_stock_asset(self._account)
        return float(getattr(asset, "cash", 0.0))

    def get_positions(self) -> dict[str, Position]:
        self._require()
        pos_list = self._trader.query_stock_positions(self._account)
        out = {}
        for p in pos_list:
            if getattr(p, "volume", 0) > 0:
                out[p.stock_code] = Position(
                    symbol=p.stock_code,
                    shares=int(p.volume),
                    available=int(getattr(p, "can_use_volume", 0)),
                    cost=float(getattr(p, "open_price", 0.0)),
                )
        return out

    def place_order(self, order: Order) -> str:
        """下单前守卫链 + 真实委托号返回。

        此前实现把结果码当 order_id 返回（无法撤单/查单），且对 0/负数/
        非整手数量照单全发。现要求：shares>0 且为整手、price>0；
        成功返回正数委托号（str），失败返回 ""。
        """
        self._require()
        xtc = self._xtc
        if order.shares <= 0 or order.shares % self.lot_size != 0:
            order.status = OrderStatus.REJECTED
            order.reason = f"invalid_shares={order.shares}"
            raise ValueError(f"非法下单数量 {order.symbol} x {order.shares}（须为正整手）")
        if order.price is None or float(order.price) <= 0:
            order.status = OrderStatus.REJECTED
            order.reason = f"invalid_price={order.price}"
            raise ValueError(f"非法下单价格 {order.symbol} @ {order.price}")
        direction = xtc.STOCK_BUY if order.side == "buy" else xtc.STOCK_SELL
        seq = self._trader.order_stock(
            self._account,
            order.symbol,
            direction,
            int(order.shares),
            xtc.FIX_PRICE,
            float(order.price),
            "ashare_quant",
            str(order.id)[:24],
        )
        if seq is None or int(seq) <= 0:
            # 券商明确拒绝（资金不足/涨跌停价外等）：不进入 OMS 熔断计数
            order.status = OrderStatus.REJECTED
            order.reason = f"xt_rejected_seq={seq}"
            self.logger_error(order.reason)
            return ""
        order.status = OrderStatus.SUBMITTED
        order.reason = ""
        return str(int(seq))

    def cancel_order(self, order_id: str) -> bool:
        self._require()
        rc = self._trader.cancel_order_stock(self._account, int(order_id))
        return rc == 0

    def get_orders(self) -> list[Order]:
        """查询当日委托；order_type 为 int 常量（此前按字符串比较永远失配）。"""
        self._require()
        xtc = self._xtc
        status_map = {
            xtc.ORDER_UNREPORTED: OrderStatus.SUBMITTED,
            xtc.ORDER_WAIT_REPORTING: OrderStatus.SUBMITTED,
            xtc.ORDER_REPORTED: OrderStatus.SUBMITTED,
            xtc.ORDER_PART_SUCC: OrderStatus.PARTIAL,
            xtc.ORDER_SUCCEEDED: OrderStatus.FILLED,
            xtc.ORDER_JUNK: OrderStatus.REJECTED,
            xtc.ORDER_CANCELED: OrderStatus.CANCELLED,
        }
        orders = []
        try:
            xt_orders = self._trader.query_stock_orders(self._account)
        except Exception:  # noqa: BLE001
            xt_orders = []
        for o in xt_orders:
            side = "buy" if o.order_type == xtc.STOCK_BUY else "sell"
            orders.append(
                Order(
                    id=str(o.order_id),
                    symbol=o.stock_code,
                    side=side,
                    shares=int(getattr(o, "order_volume", 0) or 0),
                    price=float(getattr(o, "price", 0.0) or 0.0),
                    status=status_map.get(
                        int(getattr(o, "order_status", -1)), OrderStatus.SUBMITTED
                    ),
                )
            )
        return orders

    def _require(self) -> None:
        if self._trader is None or self._account is None:
            raise RuntimeError("尚未 connect() 或连接失败")

    @staticmethod
    def logger_error(msg: str) -> None:
        import logging

        logging.getLogger("ashare.qmt").error("QMT 拒单: %s", msg)
