"""Order and position lifecycle management."""

from __future__ import annotations

import time

from config.settings import RiskSettings
from core.models import MarketSnapshot, OrderResult, Position, RiskDecision, Side, TradeRecord


class OrderManager:
    """Tracks open positions and closes them at stop-loss or take-profit."""

    def __init__(self, risk_settings: RiskSettings):
        self.risk_settings = risk_settings
        self.positions: dict[str, Position] = {}
        self.closed_trades: list[TradeRecord] = []

    @property
    def open_position_count(self) -> int:
        return len(self.positions)

    def has_position(self, symbol: str) -> bool:
        return symbol in self.positions

    def open_from_fill(self, result: OrderResult, decision: RiskDecision) -> Position | None:
        if not result.success or result.filled_amount <= 0:
            return None
        position = Position(
            symbol=decision.symbol,
            side=decision.side,
            amount=result.filled_amount,
            entry_price=result.filled_price,
            stop_loss=decision.stop_loss,
            take_profit=decision.take_profit,
            opened_at=time.time(),
            strategy_name=decision.strategy_name,
            confidence=decision.confidence,
            leverage=decision.leverage,
            fees_paid=result.fee,
            metadata={"order_id": result.exchange_order_id, **decision.metadata},
        )
        self.positions[decision.symbol] = position
        return position

    def mark_to_market(self, snapshot: MarketSnapshot) -> list[TradeRecord]:
        position = self.positions.get(snapshot.symbol)
        if position is None:
            return []
        should_close, reason, exit_price = position.should_close(snapshot.high, snapshot.low)
        if not should_close or exit_price is None:
            return []
        return [self.close_position(snapshot.symbol, exit_price, reason)]

    def close_position(self, symbol: str, exit_price: float, reason: str) -> TradeRecord:
        position = self.positions.pop(symbol)
        adjusted_exit = self._apply_exit_slippage(position.side, exit_price)
        gross_pnl = (adjusted_exit - position.entry_price) * position.amount * position.side.direction
        exit_fee = abs(adjusted_exit * position.amount) * self.risk_settings.fee_bps / 10_000
        total_fees = position.fees_paid + exit_fee
        realized_pnl = gross_pnl - total_fees
        trade = TradeRecord(
            symbol=symbol,
            side=position.side,
            amount=position.amount,
            entry_price=position.entry_price,
            exit_price=adjusted_exit,
            opened_at=position.opened_at,
            closed_at=time.time(),
            realized_pnl=float(realized_pnl),
            fees=float(total_fees),
            reason=reason,
            strategy_name=position.strategy_name,
            confidence=position.confidence,
            metadata=position.metadata,
        )
        self.closed_trades.append(trade)
        return trade

    def unrealized_pnl(self, mark_prices: dict[str, float]) -> float:
        total = 0.0
        for symbol, position in self.positions.items():
            price = mark_prices.get(symbol)
            if price is not None:
                total += position.unrealized_pnl(price)
        return total

    def _apply_exit_slippage(self, side: Side, exit_price: float) -> float:
        slippage = exit_price * self.risk_settings.slippage_bps / 10_000
        if side == Side.BUY:
            return exit_price - slippage
        return exit_price + slippage

