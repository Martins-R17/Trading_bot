"""Risk management and capital protection."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from config.settings import RiskSettings
from core.models import MarketSnapshot, RiskDecision, Side, StrategySignal


@dataclass
class RiskState:
    equity: float
    peak_equity: float
    daily_pnl: float = 0.0
    current_day: int = 0


class RiskManager:
    """Converts strategy signals into tightly controlled trade sizes."""

    def __init__(self, settings: RiskSettings):
        current_day = time.gmtime().tm_yday
        self.settings = settings
        self.state = RiskState(
            equity=settings.initial_equity,
            peak_equity=settings.initial_equity,
            daily_pnl=0.0,
            current_day=current_day,
        )

    def assess_trade(
        self,
        signal: StrategySignal,
        snapshot: MarketSnapshot,
        confidence: float,
        open_positions: int,
        max_open_positions: int,
    ) -> RiskDecision:
        """Return approved sizing or a rejection reason."""

        self._reset_daily_if_needed()
        if signal.side == Side.HOLD:
            return self._reject(signal, "hold_signal")
        if open_positions >= max_open_positions:
            return self._reject(signal, "max_open_positions")
        if self.daily_loss_limit_reached():
            return self._reject(signal, "daily_loss_limit_reached")
        if self.drawdown_limit_reached():
            return self._reject(signal, "drawdown_limit_reached")
        abnormal_reason = self.abnormal_market_reason(snapshot)
        if abnormal_reason:
            return self._reject(signal, abnormal_reason)

        entry = signal.entry_price or snapshot.close
        stop_loss, take_profit = self._ensure_exits(signal, entry)
        risk_per_unit = abs(entry - stop_loss)
        if risk_per_unit <= 0:
            return self._reject(signal, "invalid_stop_distance")

        sentiment_multiplier = self._sentiment_risk_multiplier(signal.side, snapshot.sentiment_score)
        confidence_multiplier = float(np.clip(confidence, 0.2, 1.0))
        risk_capital = self.state.equity * self.settings.max_risk_per_trade * sentiment_multiplier * confidence_multiplier
        raw_amount = risk_capital / risk_per_unit

        leverage = self.dynamic_leverage(snapshot.volatility)
        max_notional = self.state.equity * self.settings.max_position_notional_fraction * leverage
        amount = min(raw_amount, max_notional / entry)
        notional = amount * entry
        if amount <= 0 or notional <= 0:
            return self._reject(signal, "position_size_zero")

        return RiskDecision(
            approved=True,
            reason="approved",
            symbol=signal.symbol,
            side=signal.side,
            amount=float(amount),
            entry_price=float(entry),
            stop_loss=float(stop_loss),
            take_profit=float(take_profit),
            leverage=float(leverage),
            notional=float(notional),
            margin_required=float(notional / leverage),
            confidence=float(confidence),
            strategy_name=signal.strategy_name,
            metadata={
                **signal.metadata,
                "risk_capital": risk_capital,
                "sentiment_multiplier": sentiment_multiplier,
            },
        )

    def dynamic_leverage(self, volatility: float) -> float:
        """Reduce leverage as realized volatility rises."""

        volatility = max(float(volatility), 0.0)
        if volatility <= 0.006:
            target = self.settings.max_leverage
        elif volatility >= 0.03:
            target = self.settings.min_leverage
        else:
            slope = (volatility - 0.006) / (0.03 - 0.006)
            target = self.settings.max_leverage - slope * (self.settings.max_leverage - self.settings.min_leverage)
        return float(np.clip(target, self.settings.min_leverage, self.settings.max_leverage))

    def abnormal_market_reason(self, snapshot: MarketSnapshot) -> str:
        if snapshot.volatility >= self.settings.abnormal_volatility:
            return "abnormal_volatility"
        book = snapshot.order_book
        if book is not None and book.spread_bps > self.settings.max_spread_bps:
            return "spread_too_wide"
        if snapshot.close <= 0:
            return "invalid_price"
        return ""

    def record_trade(self, realized_pnl: float) -> None:
        self._reset_daily_if_needed()
        self.state.equity += realized_pnl
        self.state.daily_pnl += realized_pnl
        self.state.peak_equity = max(self.state.peak_equity, self.state.equity)

    def daily_loss_limit_reached(self) -> bool:
        max_loss = self.settings.initial_equity * self.settings.max_daily_loss
        return self.state.daily_pnl <= -abs(max_loss)

    def drawdown_limit_reached(self) -> bool:
        if self.state.peak_equity <= 0:
            return True
        drawdown = (self.state.peak_equity - self.state.equity) / self.state.peak_equity
        return drawdown >= self.settings.max_drawdown

    def _ensure_exits(self, signal: StrategySignal, entry: float) -> tuple[float, float]:
        stop_loss = signal.stop_loss
        take_profit = signal.take_profit
        if stop_loss is None:
            stop_loss = entry * (1 - signal.side.direction * self.settings.stop_loss_bps / 10_000)
        if take_profit is None:
            take_profit = entry * (1 + signal.side.direction * self.settings.take_profit_bps / 10_000)
        return float(stop_loss), float(take_profit)

    def _sentiment_risk_multiplier(self, side: Side, sentiment_score: float) -> float:
        aligned = side.direction * sentiment_score
        multiplier = 1.0 + aligned * self.settings.sentiment_risk_multiplier
        return float(np.clip(multiplier, 0.5, 1.25))

    def _reset_daily_if_needed(self) -> None:
        day = time.gmtime().tm_yday
        if day != self.state.current_day:
            self.state.current_day = day
            self.state.daily_pnl = 0.0

    def _reject(self, signal: StrategySignal, reason: str) -> RiskDecision:
        return RiskDecision(
            approved=False,
            reason=reason,
            symbol=signal.symbol,
            side=signal.side,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            strategy_name=signal.strategy_name,
        )
