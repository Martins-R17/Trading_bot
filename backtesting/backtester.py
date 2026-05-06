"""Historical backtesting engine."""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

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
    diagnostics: dict[str, Any] = field(default_factory=dict)


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
        total_candidates = 0
        total_signals_checked = 0
        rejection_counts: Counter[str] = Counter()
        best_rejected_by_strategy: dict[str, dict[str, Any]] = {}
        closest_to_approved: dict[str, Any] | None = None

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
                total_candidates += 1
                total_signals_checked += len(selection.candidate_diagnostics) or len(selection.strategy_scores)
                for row in self._selector_diagnostic_rows(selection):
                    closest_to_approved = self._record_rejected_candidate(
                        row,
                        rejection_counts,
                        best_rejected_by_strategy,
                        closest_to_approved,
                    )
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
                    else:
                        row = self._decision_diagnostic_row(decision)
                        closest_to_approved = self._record_rejected_candidate(
                            row,
                            rejection_counts,
                            best_rejected_by_strategy,
                            closest_to_approved,
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

        diagnostics = {
            "total_candidates": total_candidates,
            "total_signals_checked": total_signals_checked,
            "diagnostic_notional": max(
                self.risk_settings.min_position_notional,
                self.risk_settings.diagnostic_notional,
            ),
            "calibration_min_expected_net_profit": self.risk_settings.calibration_min_expected_net_profit_usd,
            "rejection_counts_by_detailed_reason": dict(rejection_counts),
            "best_rejected_by_strategy": best_rejected_by_strategy,
            "closest_to_approved": closest_to_approved or {},
        }

        return BacktestResult(
            trades=trades,
            equity_curve=equity_curve,
            metrics=summarize(trades, equity_curve),
            diagnostics=diagnostics,
        )

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

    def _selector_diagnostic_rows(self, selection: Any) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for diagnostic in selection.candidate_diagnostics:
            reason = str(diagnostic.detailed_rejection_reason or "")
            if not reason and diagnostic.rejection_reason:
                reason = self._detailed_reason_from_text(diagnostic.rejection_reason)
            row = {
                "symbol": diagnostic.symbol,
                "strategy_name": diagnostic.strategy_name,
                "side_considered": diagnostic.side_considered,
                "confidence": self._finite_number(diagnostic.confidence),
                "expected_gross_reward": self._finite_number(
                    diagnostic.expected_gross_reward
                ),
                "estimated_costs": self._finite_number(diagnostic.estimated_costs),
                "expected_net_profit": self._finite_number(
                    diagnostic.expected_net_profit
                ),
                "target_move_bps": self._finite_number(diagnostic.target_move_bps),
                "reward_cost_ratio": self._finite_number(diagnostic.reward_cost_ratio),
                "rejection_reason": diagnostic.rejection_reason,
                "detailed_rejection_reason": reason,
                "rsi_check": diagnostic.rsi_check,
                "ema_trend_check": diagnostic.ema_trend_check,
                "macd_check": diagnostic.macd_check,
                "volatility_atr_check": diagnostic.volatility_atr_check,
                "target_move_check": diagnostic.target_move_check,
                "reward_cost_check": diagnostic.reward_cost_check,
                "expected_net_profit_check": diagnostic.expected_net_profit_check,
                "has_economics": (
                    diagnostic.estimated_costs > 0
                    or diagnostic.reward_cost_ratio > 0
                    or diagnostic.target_move_bps > 0
                ),
            }
            self._populate_diagnostic_economics(row)
            rows.append(row)
        return rows

    def _decision_diagnostic_row(self, decision: Any) -> dict[str, Any]:
        metadata = decision.metadata or {}
        row = {
            "symbol": decision.symbol,
            "strategy_name": decision.strategy_name or "unknown",
            "side_considered": decision.side.value,
            "confidence": self._finite_number(decision.confidence),
            "expected_gross_reward": self._finite_number(metadata.get("expected_gross_reward")),
            "estimated_costs": self._finite_number(metadata.get("estimated_round_trip_cost")),
            "expected_net_profit": self._finite_number(metadata.get("expected_net_profit")),
            "target_move_bps": self._finite_number(metadata.get("target_move_bps")),
            "reward_cost_ratio": self._finite_number(metadata.get("reward_cost_ratio")),
            "rejection_reason": decision.reason,
            "detailed_rejection_reason": self._detailed_reason_from_text(decision.reason),
            "rsi_check": str(metadata.get("rsi_check", "not_checked")),
            "ema_trend_check": str(metadata.get("ema_trend_check", "not_checked")),
            "macd_check": str(metadata.get("macd_check", "not_checked")),
            "volatility_atr_check": str(metadata.get("volatility_atr_check", "not_checked")),
            "target_move_check": "fail"
            if decision.reason == "target_move_too_small_after_costs"
            else str(metadata.get("target_move_check", "not_checked")),
            "reward_cost_check": "fail"
            if decision.reason == "expected_reward_below_costs"
            else str(metadata.get("reward_cost_check", "not_checked")),
            "expected_net_profit_check": "fail"
            if decision.reason == "expected_net_profit_too_low"
            else str(metadata.get("expected_net_profit_check", "not_checked")),
            "has_economics": self._finite_number(metadata.get("estimated_round_trip_cost")) > 0,
        }
        self._populate_diagnostic_economics(row)
        return row

    def _populate_diagnostic_economics(self, row: dict[str, Any]) -> None:
        if self._finite_number(row.get("estimated_costs")) > 0:
            return
        target_move_bps = self._finite_number(row.get("target_move_bps"))
        if target_move_bps <= 0:
            return

        diagnostic_notional = max(
            self.risk_settings.min_position_notional,
            self.risk_settings.diagnostic_notional,
        )
        estimated_costs = diagnostic_notional * self.risk_settings.round_trip_taker_cost_rate
        expected_gross_reward = diagnostic_notional * target_move_bps / 10_000
        row["estimated_costs"] = self._finite_number(estimated_costs)
        row["expected_gross_reward"] = self._finite_number(expected_gross_reward)
        row["expected_net_profit"] = self._finite_number(expected_gross_reward - estimated_costs)
        row["has_economics"] = True

        if row.get("expected_net_profit_check") == "not_checked":
            row["expected_net_profit_check"] = (
                "pass"
                if row["expected_net_profit"] >= self.risk_settings.calibration_min_expected_net_profit_usd
                else "fail"
            )

    def _record_rejected_candidate(
        self,
        candidate: dict[str, Any],
        rejection_counts: Counter[str],
        best_rejected_by_strategy: dict[str, dict[str, Any]],
        closest_to_approved: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        reason = str(candidate.get("detailed_rejection_reason") or "")
        if not reason:
            reason = self._detailed_reason_from_text(str(candidate.get("rejection_reason") or ""))
        if not reason or reason in {"approved", "pending_risk_review", "pending_ai_review"}:
            return closest_to_approved

        rejection_counts[reason] += 1
        strategy_name = str(candidate.get("strategy_name") or "unknown")
        existing = best_rejected_by_strategy.get(strategy_name)
        if existing is None or self._candidate_rank(candidate) > self._candidate_rank(existing):
            best_rejected_by_strategy[strategy_name] = dict(candidate)

        if closest_to_approved is None or self._candidate_rank(candidate) > self._candidate_rank(closest_to_approved):
            return dict(candidate)
        return closest_to_approved

    def _detailed_reason_from_text(self, reason: str) -> str:
        reason = str(reason or "").strip()
        if not reason:
            return ""
        parts = reason.split(":")
        reason_key = parts[1] if parts[0] == "no_actionable_strategy" and len(parts) > 1 else parts[0]
        if reason_key in {
            "trend_not_confirmed",
            "ema_trend_filter",
            "ema_trend_too_strong",
            "counter_trend_short_blocked",
            "counter_trend_long_blocked",
        }:
            return "trend_not_confirmed"
        if reason_key in {
            "macd_not_confirmed",
            "macd_reversal_not_confirmed",
            "macd_hist_not_strong_enough",
        }:
            return "macd_not_confirmed"
        if reason_key in {
            "rsi_not_confirmed",
            "rsi_overextended",
            "neutral_rsi_requires_stronger_liquidity",
        }:
            return "rsi_not_confirmed"
        if reason_key in {
            "volatility_too_low",
            "range_expansion_not_confirmed",
            "breakout_not_confirmed",
            "volatility_target_too_small_after_costs",
            "breakout_range_too_small_after_costs",
            "zero_variance",
        }:
            return "volatility_too_low"
        if reason_key in {
            "target_move_too_small",
            "target_move_too_small_after_costs",
            "mean_distance_too_small_after_costs",
        }:
            return "target_move_too_small"
        if reason_key == "expected_reward_below_costs":
            return "reward_cost_ratio_too_low"
        if reason_key == "expected_net_profit_too_low":
            return "expected_net_profit_too_low"
        if reason_key == "confidence_below_threshold":
            return "confidence_below_threshold"
        if "spread" in reason_key or "cost" in reason_key:
            return "spread_cost_filter"
        return reason_key

    def _candidate_rank(self, candidate: dict[str, Any]) -> tuple[int, float, float, float, float, float]:
        has_economics = 1 if candidate.get("has_economics") else 0
        return (
            has_economics,
            self._finite_number(candidate.get("expected_net_profit")),
            self._finite_number(candidate.get("reward_cost_ratio")),
            self._finite_number(candidate.get("target_move_bps")),
            self._finite_number(candidate.get("expected_gross_reward")),
            self._finite_number(candidate.get("confidence")),
        )

    def _finite_number(self, value: Any, default: float = 0.0) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return default
        if pd.isna(number):
            return default
        return number

