"""Dynamic strategy selection engine."""

from __future__ import annotations

import logging

import numpy as np

from ai.confidence_model import ConfidenceModel
from core.models import MarketSnapshot, SelectionResult, Side, StrategySignal
from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class StrategySelector:
    """Scores all strategies and returns the highest-confidence candidate."""

    def __init__(
        self,
        strategies: list[BaseStrategy],
        confidence_model: ConfidenceModel,
        confidence_threshold: float = 0.8,
    ) -> None:
        self.strategies = strategies
        self.confidence_model = confidence_model
        self.confidence_threshold = confidence_threshold

    def select(self, snapshot: MarketSnapshot) -> SelectionResult:
        best_signal: StrategySignal | None = None
        best_confidence = 0.0
        strategy_scores: dict[str, float] = {}

        for strategy in self.strategies:
            market_score = strategy.score_market(snapshot)
            performance_score = strategy.performance_score()
            signal = strategy.generate_signal(snapshot)

            if not signal.is_actionable:
                strategy_scores[strategy.name] = 0.0
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

            if final_score > best_confidence:
                best_confidence = final_score
                best_signal = signal

        if best_signal is None:
            return SelectionResult(None, 0.0, strategy_scores, False, "no_actionable_strategy")

        best_signal.metadata["selector_score"] = best_confidence
        if best_confidence < self.confidence_threshold:
            return SelectionResult(
                best_signal,
                best_confidence,
                strategy_scores,
                False,
                f"confidence_below_threshold:{best_confidence:.3f}",
            )

        return SelectionResult(best_signal, best_confidence, strategy_scores, True, "approved")

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
