"""交易成本模型（A股：佣金/印花税/过户费/滑点）。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CostModel:
    commission_bp: float = 2.5
    stamp_bp: float = 5.0
    transfer_bp: float = 0.1
    slippage_bp: float = 5.0
    min_commission: float = 5.0
    lot_size: int = 100
    slippage_model: str = "fixed"
    slippage_cap_bp: float = 20.0
    slippage_impact_coef: float = 5.0

    def effective_slippage_bp(self, order_amount: float, daily_amount: float) -> float:
        """固定滑点，或按参与率（订单额/日均成交额）动态冲击，封顶 cap。"""
        if self.slippage_model != "adaptive" or daily_amount <= 0 or order_amount <= 0:
            return self.slippage_bp
        participation = order_amount / daily_amount
        impact = self.slippage_impact_coef * participation * 10_000
        return min(self.slippage_bp + impact, self.slippage_cap_bp)

    def buy_fee(self, amount: float) -> float:
        commission = max(amount * self.commission_bp / 10_000, self.min_commission)
        transfer = amount * self.transfer_bp / 10_000
        # slippage 已由撮合层反映到成交价中，此处不再重复计算
        return commission + transfer

    def sell_fee(self, amount: float) -> float:
        commission = max(amount * self.commission_bp / 10_000, self.min_commission)
        transfer = amount * self.transfer_bp / 10_000
        stamp = amount * self.stamp_bp / 10_000
        # slippage 已由撮合层反映到成交价中，此处不再重复计算
        return commission + transfer + stamp

    def round_lot(self, shares: float) -> int:
        return int(shares // self.lot_size * self.lot_size)
