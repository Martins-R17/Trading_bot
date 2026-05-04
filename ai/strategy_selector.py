"""Dynamic strategy selection engine."""

from __future__ import annotations

import logging

import numpy as np

from ai.confidence_model import ConfidenceModel
from core.models import CandidateDiagnostics, MarketSnapshot, SelectionResult, Side, StrategySignal
from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class StrategySelector:
    """Scores all strategies and returns the highest-confidence candidate."""

    def __init__(
        self,
        strategies: list[BaseStrategy],
        confidence_model: ConfidenceModel,
        confidence_threshold: float = 0.8,
        ema_trend_deadband_bps: float = 1.0,
        counter_trend_rsi_overbought: float = 70.0,
        counter_trend_rsi_oversold: float = 30.0,
        counter_trend_macd_hist_bps: float = 1.0,
    ) -> None:
        self.strategies = strategies
        self.confidence_model = confidence_model
        self.confidence_threshold = confidence_threshold
        self.ema_trend_deadband_bps = ema_trend_deadband_bps
        self.counter_trend_rsi_overbought = counter_trend_rsi_overbought
        self.counter_trend_rsi_oversold = counter_trend_rsi_oversold
        self.counter_trend_macd_hist_bps = counter_trend_macd_hist_bps

    def select(self, snapshot: MarketSnapshot) -> SelectionResult:
        best_signal: StrategySignal | None = None
        best_confidence = 0.0
        strategy_scores: dict[str, float] = {}
        rejections: dict[str, str] = {}
        candidate_diagnostics: list[CandidateDiagnostics] = []

        for strategy in self.strategies:
            market_score = strategy.score_market(snapshot)
            performance_score = strategy.performance_score()
            signal = strategy.generate_signal(snapshot)

            if not signal.is_actionable:
                strategy_scores[strategy.name] = 0.0
                candidate_diagnostics.append(
                    CandidateDiagnostics(
                        symbol=snapshot.symbol,
                        strategy_name=strategy.name,
                        side=signal.side,
                        market_score=float(market_score),
                        performance_score=float(performance_score),
                        rejection_reason=str(signal.metadata.get("reason", "not_actionable")),
                        metadata=dict(signal.metadata),
                    )
                )
                continue

            counter_trend_reason = self._counter_trend_rejection_reason(signal, snapshot)
            if counter_trend_reason:
                strategy_scores[strategy.name] = 0.0
                signal.metadata["rejection_reason"] = counter_trend_reason
                rejections[strategy.name] = counter_trend_reason
                candidate_diagnostics.append(
                    CandidateDiagnostics(
                        symbol=snapshot.symbol,
                        strategy_name=strategy.name,
                        side=signal.side,
                        market_score=float(market_score),
                        performance_score=float(performance_score),
                        actionable=True,
                        rejection_reason=counter_trend_reason,
                        metadata=dict(signal.metadata),
                    )
                )
                continue

            feature_snapshot = self.confidence_model.features(signal, snapshot, market_score, performance_score)
            confidence = self.confidence_model.predict(signal, snapshot, market_score, performance_score)
            liquidity_score = self._liquidity_score(snapshot)
            sentiment_score = self._sentiment_alignment(snapshot, signal)
            final_score = float(
                np.clip(
                    0.45 * confidence
                    + 0.22 * market_score
                    + 0.13 * performance_score
                    + 0.12 * liquidity_score
                    + 0.08 * sentiment_score,
                    0.0,
                    1.0,
                )
            )
            strategy_scores[strategy.name] = final_score
            signal.metadata["confidence_features"] = feature_snapshot
            signal.metadata["raw_model_confidence"] = confidence
            signal.metadata["market_score"] = market_score
            signal.metadata["performance_score"] = performance_score
            candidate_diagnostics.append(
                CandidateDiagnostics(
                    symbol=snapshot.symbol,
                    strategy_name=strategy.name,
                    side=signal.side,
                    confidence=final_score,
                    market_score=float(market_score),
                    performance_score=float(performance_score),
                    final_score=final_score,
                    actionable=True,
                    metadata=dict(signal.metadata),
                )
            )

            if final_score > best_confidence:
                best_confidence = final_score
                best_signal = signal

        if best_signal is None:
            return SelectionResult(
                None,
                0.0,
                strategy_scores,
                False,
                "no_actionable_strategy",
                rejections,
                candidate_diagnostics,
            )

        best_signal.metadata["selector_score"] = best_confidence
        if best_confidence < self.confidence_threshold:
            reason = f"confidence_below_threshold:{best_confidence:.3f}"
            rejections[best_signal.strategy_name] = reason
            for candidate in candidate_diagnostics:
                if candidate.strategy_name == best_signal.strategy_name:
                    candidate.rejection_reason = reason
                    break
            return SelectionResult(
                best_signal,
                best_confidence,
                strategy_scores,
                False,
                reason,
                rejections,
                candidate_diagnostics,
            )

        return SelectionResult(
            best_signal,
            best_confidence,
            strategy_scores,
            True,
            "approved",
            rejections,
            candidate_diagnostics,
        )

    def _liquidity_score(self, snapshot: MarketSnapshot) -> float:
        book = snapshot.order_book
        if book is None or book.mid_price is None:
            return 0.45
        spread_component = 1.0 - min(book.spread_bps / 20.0, 1.0)
        depth_component = min(book.total_depth_quote(levels=5) / 500_000.0, 1.0)
        return float(np.clip(0.65 * spread_component + 0.35 * depth_component, 0.0, 1.0))

    def _sentiment_alignment(self, snapshot: MarketSnapshot, signal: StrategySignal) -> float:
        if signal.side == Side.HOLD:
            return 0.0
        aligned = snapshot.sentiment_score * signal.side.direction
        return float(np.clip(0.5 + aligned / 2.0, 0.0, 1.0))

    def _counter_trend_rejection_reason(self, signal: StrategySignal, snapshot: MarketSnapshot) -> str:
        if snapshot.ohlcv is None or len(snapshot.ohlcv) == 0:
            return ""
        df = snapshot.ohlcv
        if not {"close", "ema_fast", "ema_slow", "rsi", "macd_hist"}.issubset(df.columns):
            return ""

        price = max(float(df["close"].iloc[-1]), 1e-9)
        ema_fast = float(df["ema_fast"].iloc[-1])
        ema_slow = float(df["ema_slow"].iloc[-1])
        rsi = float(df["rsi"].iloc[-1])
        macd_hist = float(df["macd_hist"].iloc[-1])
        ema_gap_bps = (ema_fast - ema_slow) / price * 10_000
        macd_hist_bps = macd_hist / price * 10_000

        ema_up = ema_gap_bps > self.ema_trend_deadband_bps
        ema_down = ema_gap_bps < -self.ema_trend_deadband_bps
        strong_negative_macd = macd_hist_bps <= -abs(self.counter_trend_macd_hist_bps)
        strong_positive_macd = macd_hist_bps >= abs(self.counter_trend_macd_hist_bps)

        if signal.side == Side.SELL and ema_up:
            if rsi >= self.counter_trend_rsi_overbought and strong_negative_macd:
                return ""
            return (
                "counter_trend_short_blocked:"
                f"ema_gap={ema_gap_bps:.2f}bps rsi={rsi:.1f} "
                f"macd_hist={macd_hist_bps:.2f}bps"
            )

        if signal.side == Side.BUY and ema_down:
            if rsi <= self.counter_trend_rsi_oversold and strong_positive_macd:
                return ""
            return (
                "counter_trend_long_blocked:"
                f"ema_gap={ema_gap_bps:.2f}bps rsi={rsi:.1f} "
                f"macd_hist={macd_hist_bps:.2f}bps"
            )

        return ""
