"""Order and position lifecycle management."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
import time

from config.settings import RiskSettings
from core.models import MarketSnapshot, OrderResult, Position, RiskDecision, Side, TradeRecord

logger = logging.getLogger(__name__)


class OrderManager:
    """Tracks open positions and closes them at stop-loss or take-profit."""

    def __init__(self, risk_settings: RiskSettings, trade_history_path: str = "data/trade_history.csv"):
        self.risk_settings = risk_settings
        self.trade_history_path = Path(trade_history_path)
        self.positions: dict[str, Position] = {}
        self.closed_trades: list[TradeRecord] = []

    @property
    def open_position_count(self) -> int:
        return len(self.positions)

    def has_position(self, symbol: str) -> bool:
        return symbol in self.positions

    def open_from_fill(
        self,
        result: OrderResult,
        decision: RiskDecision,
        opened_iteration: int | None = None,
    ) -> Position | None:
        if not result.success or result.filled_amount <= 0:
            return None
        metadata = {"order_id": result.exchange_order_id, **decision.metadata}
        if opened_iteration is not None:
            metadata["opened_iteration"] = opened_iteration
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
            metadata=metadata,
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
        self._append_trade_history(trade)
        return trade

    def unrealized_pnl(self, mark_prices: dict[str, float]) -> float:
        return sum(self.unrealized_pnl_by_symbol(mark_prices).values())

    def unrealized_pnl_by_symbol(self, mark_prices: dict[str, float]) -> dict[str, float]:
        pnl_by_symbol: dict[str, float] = {}
        for symbol, position in self.positions.items():
            price = mark_prices.get(symbol)
            if price is not None:
                pnl_by_symbol[symbol] = self._net_unrealized_pnl(position, price)
        return pnl_by_symbol

    @property
    def realized_pnl(self) -> float:
        return sum(trade.realized_pnl for trade in self.closed_trades)

    def _apply_exit_slippage(self, side: Side, exit_price: float) -> float:
        slippage = exit_price * self.risk_settings.slippage_bps / 10_000
        if side == Side.BUY:
            return exit_price - slippage
        return exit_price + slippage

    def _net_unrealized_pnl(self, position: Position, mark_price: float) -> float:
        adjusted_exit = self._apply_exit_slippage(position.side, mark_price)
        gross_pnl = (adjusted_exit - position.entry_price) * position.amount * position.side.direction
        exit_fee = abs(adjusted_exit * position.amount) * self.risk_settings.fee_bps / 10_000
        return float(gross_pnl - position.fees_paid - exit_fee)

    def _append_trade_history(self, trade: TradeRecord) -> None:
        fieldnames = [
            "closed_at",
            "symbol",
            "side",
            "amount",
            "entry_price",
            "exit_price",
            "realized_pnl",
            "fees",
            "reason",
            "strategy_name",
            "confidence",
            "metadata",
        ]
        row = {
            "closed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(trade.closed_at)),
            "symbol": trade.symbol,
            "side": trade.side.value,
            "amount": f"{trade.amount:.12f}",
            "entry_price": f"{trade.entry_price:.8f}",
            "exit_price": f"{trade.exit_price:.8f}",
            "realized_pnl": f"{trade.realized_pnl:.8f}",
            "fees": f"{trade.fees:.8f}",
            "reason": trade.reason,
            "strategy_name": trade.strategy_name,
            "confidence": f"{trade.confidence:.6f}",
            "metadata": json.dumps(trade.metadata, sort_keys=True, default=str),
        }
        try:
            self.trade_history_path.parent.mkdir(parents=True, exist_ok=True)
            file_exists = self.trade_history_path.exists()
            with self.trade_history_path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)
        except OSError as exc:
            logger.warning("Could not write trade history to %s: %s", self.trade_history_path, exc)

