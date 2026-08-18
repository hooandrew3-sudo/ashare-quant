"""miniQMT(XtQuant) 交易适配器：懒加载，未安装时给出明确指引。"""

from __future__ import annotations

import os

from quant.execution.broker import Order, OrderStatus, Position


class QMTBroker:
    """基于 xtquant 的 A 股实盘通道（需券商开通 miniQMT 权限）。

    环境变量：
      QMT_USER_ID / QMT_PASSWORD / QMT_ACCOUNT_ID / QMT_DATA_DIR
    安装：将券商提供的 xtquant 目录加入 sys.path（不同券商版本不通用）。
    """

    def __init__(self, data_dir: str | None = None, account_id: str | None = None):
        self.data_dir = data_dir or os.getenv("QMT_DATA_DIR", "")
        self.account_id = account_id or os.getenv("QMT_ACCOUNT_ID", "")
        self._xt = None
        self._trader = None

    def connect(self) -> None:
        try:
            from xtquant import xtdata, xttrader  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "未找到 xtquant。请先开通券商 miniQMT 权限，并将券商提供的 "
                "xtquant 目录加入 Python 路径（参见 docs/PRODUCTION_SPEC.md §10）。"
            ) from exc
        from xtquant.xttrader import XtQuantTrader
        from xtquant.xttype import StockAccount

        self._xt = xtdata
        session = int(os.getenv("QMT_SESSION", "1"))
        self._trader = XtQuantTrader(self.data_dir, session)
        self._trader.start()
        ok = self._trader.connect()
        if not ok:
            raise RuntimeError("QMT 交易服务连接失败，请检查 QMT_DATA_DIR 与客户端登录状态")
        account = StockAccount(self.account_id)
        if not self._trader.subscribe(account):
            raise RuntimeError(f"QMT 账户订阅失败: {self.account_id}")

    def get_cash(self) -> float:
        self._require()
        from xtquant.xttype import StockAccount

        account = StockAccount(self.account_id)
        asset = self._trader.query_stock_asset(account)
        return float(getattr(asset, "cash", 0.0))

    def get_positions(self) -> dict[str, Position]:
        self._require()
        from xtquant.xttype import StockAccount

        account = StockAccount(self.account_id)
        pos_list = self._trader.query_stock_positions(account)
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
        self._require()
        from xtquant.xttrader import StockOrder

        order_type = StockOrder.STOCK_BUY if order.side == "buy" else StockOrder.STOCK_SELL
        price_type = StockOrder.FIX_PRICE
        xt_order = StockOrder(
            self.account_id,
            order.symbol,
            order_type,
            price_type,
            order.shares,
            order.price,
            str(order.id),
        )
        res = self._trader.order_stock(xt_order)
        order.status = OrderStatus.SUBMITTED if res == 0 else OrderStatus.REJECTED
        order.reason = f"xt_result={res}"
        return str(res)

    def cancel_order(self, order_id: str) -> None:
        self._require()
        self._trader.cancel_order_stock(self.account_id, order_id)

    def get_orders(self) -> list[Order]:
        self._require()
        orders = self._trader.query_stock_orders(self.account_id)
        return [
            Order(
                id=str(o.order_id),
                symbol=o.stock_code,
                side="buy" if o.order_type == "buy" else "sell",
                shares=int(o.order_volume),
                price=float(o.price),
                status=OrderStatus.SUBMITTED,
            )
            for o in orders
        ]

    def _require(self) -> None:
        if self._trader is None:
            raise RuntimeError("尚未 connect()")
