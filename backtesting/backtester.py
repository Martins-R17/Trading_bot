"""Historical backtesting engine."""

from __future__ import annotations

import time
from dataclasses import dataclass

import pandas as pd

from ai.confidence_model import ConfidenceModel
from ai.strategy_selector import StrategySelector
from backtesting.metrics import summarize
from config.settings import RiskSettings, TradingSettings
from core.models import MarketSnapshot, Position, Side, TradeRecord
from data.preprocess import DataPreprocessor
from risk.risk_manager import RiskManager
from strategies.base_strategy import BaseStrategy


@dataclass
class BacktestResult:
    trades: list[TradeRecord]
    equity_curve: list[float]
    metrics: dict[str, float]


class Backtester:
    """Bar-by-bar simulator with dynamic strategy selection and risk sizing."""

    def __init__(
        self,
        strategies: list[BaseStrategy],
        risk_settings: RiskSettings,
        trading_settings: TradingSettings,
    ) -> None:
        self.strategies = strategies
        self.risk_settings = risk_settings
        self.trading_settings = trading_settings

    def run(self, symbol: str, historical_ohlcv: pd.DataFrame) -> BacktestResult:
        df = DataPreprocessor.add_features(DataPreprocessor.normalize_ohlcv(historical_ohlcv))
        risk = RiskManager(self.risk_settings)
        selector = StrategySelector(
            self.strategies,
            ConfidenceModel(),
            confidence_threshold=self.trading_settings.confidence_threshold,
        )
        equity_curve = [risk.state.equity]
        trades: list[TradeRecord] = []
        position: Position | None = None

        warmup = max(strategy.min_bars for strategy in self.strategies)
        for index in range(warmup, len(df)):
            window = df.iloc[: index + 1].copy()
            snapshot = MarketSnapshot(
                symbol=symbol,
                timestamp=float(window["timestamp"].iloc[-1]),
                ohlcv=window,
                volatility=DataPreprocessor.realized_volatility(window),
            )
            bar = window.iloc[-1]

            if position is not None:
                close_record = self._maybe_close_position(position, bar)
                if close_record is not None:
                    trades.append(close_record)
                    risk.record_trade(close_record.realized_pnl)
                    for strategy in self.strategies:
                        if strategy.name == close_record.strategy_name:
                            strategy.record_trade(close_record.realized_pnl)
                    position = None

            if position is None and not risk.daily_loss_limit_reached() and not risk.drawdown_limit_reached():
                selection = selector.select(snapshot)
                if selection.approved and selection.signal is not None:
                    decision = risk.assess_trade(
                        selection.signal,
                        snapshot,
                        selection.confidence,
                        open_positions=0,
                        max_open_positions=1,
                    )
                    if decision.approved:
                        position = Position(
                            symbol=symbol,
                            side=decision.side,
                            amount=decision.amount,
                            entry_price=decision.entry_price,
                            stop_loss=decision.stop_loss,
                            take_profit=decision.take_profit,
                            opened_at=float(bar["timestamp"]),
                            strategy_name=decision.strategy_name,
                            confidence=decision.confidence,
                            leverage=decision.leverage,
                            fees_paid=decision.notional * self.risk_settings.taker_fee_rate,
                        )

            mark_equity = risk.state.equity
            if position is not None:
                mark_equity += position.unrealized_pnl(float(bar["close"]))
            equity_curve.append(float(mark_equity))

        if position is not None:
            final_bar = df.iloc[-1]
            trades.append(self._close(position, float(final_bar["close"]), "end_of_backtest", float(final_bar["timestamp"])))
            risk.record_trade(trades[-1].realized_pnl)
            equity_curve.append(float(risk.state.equity))

        return BacktestResult(trades=trades, equity_curve=equity_curve, metrics=summarize(trades, equity_curve))

    def _maybe_close_position(self, position: Position, bar: pd.Series) -> TradeRecord | None:
        should_close, reason, exit_price = position.should_close(float(bar["high"]), float(bar["low"]))
        if should_close and exit_price is not None:
            return self._close(position, exit_price, reason, float(bar["timestamp"]))
        return None

    def _close(self, position: Position, exit_price: float, reason: str, closed_at: float) -> TradeRecord:
        if position.side == Side.BUY:
            adjusted_exit = exit_price * (1 - self.risk_settings.slippage_bps / 10_000)
        else:
            adjusted_exit = exit_price * (1 + self.risk_settings.slippage_bps / 10_000)
        gross_pnl = (exit_price - position.entry_price) * position.amount * position.side.direction
        exit_slippage_cost = abs(adjusted_exit - exit_price) * position.amount
        exit_fee = abs(adjusted_exit * position.amount) * self.risk_settings.taker_fee_rate
        total_fees = position.fees_paid + exit_fee
        total_costs = total_fees + exit_slippage_cost
        return TradeRecord(
            symbol=position.symbol,
            side=position.side,
            amount=position.amount,
            entry_price=position.entry_price,
            exit_price=adjusted_exit,
            opened_at=position.opened_at,
            closed_at=closed_at or time.time(),
            realized_pnl=float(gross_pnl - total_costs),
            gross_pnl=float(gross_pnl),
            fees=float(total_fees),
            slippage_costs=float(exit_slippage_cost),
            total_costs=float(total_costs),
            reason=reason,
            strategy_name=position.strategy_name,
            confidence=position.confidence,
            metadata=position.metadata,
        )

