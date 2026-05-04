"""Confidence scoring for candidate trades."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError:  # pragma: no cover - dependency is in requirements.
    LogisticRegression = None
    Pipeline = None
    StandardScaler = None

from core.models import MarketSnapshot, Side, StrategySignal
from data.preprocess import DataPreprocessor


FEATURE_NAMES = [
    "signal_strength",
    "confidence_hint",
    "strategy_market_score",
    "strategy_performance_score",
    "volatility",
    "spread_bps",
    "depth_imbalance_signed",
    "sentiment_alignment",
    "volume_zscore",
    "rsi",
    "ema_trend_alignment",
    "macd_alignment",
]


@dataclass
class ConfidenceModel:
    """Hybrid heuristic and online-trainable scikit-learn confidence model."""

    min_samples_to_fit: int = 30
    samples: list[list[float]] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.model = None
        if Pipeline is not None and StandardScaler is not None and LogisticRegression is not None:
            self.model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=500, class_weight="balanced")),
                ]
            )
        self.is_fitted = False

    def features(
        self,
        signal: StrategySignal,
        snapshot: MarketSnapshot,
        strategy_market_score: float,
        strategy_performance_score: float,
    ) -> dict[str, float]:
        book_features = DataPreprocessor.order_book_features(snapshot.order_book)
        sentiment_alignment = snapshot.sentiment_score * signal.side.direction
        depth_imbalance_signed = book_features["depth_imbalance"] * signal.side.direction
        volume_zscore = 0.0
        rsi = 50.0
        ema_trend_alignment = 0.0
        macd_alignment = 0.0
        if snapshot.ohlcv is not None and "volume_zscore" in snapshot.ohlcv.columns and len(snapshot.ohlcv) > 0:
            volume_zscore = float(snapshot.ohlcv["volume_zscore"].iloc[-1])
        if snapshot.ohlcv is not None and len(snapshot.ohlcv) > 0:
            df = snapshot.ohlcv
            close = max(float(df["close"].iloc[-1]), 1e-9)
            if "rsi" in df.columns:
                rsi = float(df["rsi"].iloc[-1])
            if "ema_fast" in df.columns and "ema_slow" in df.columns:
                ema_gap_bps = (float(df["ema_fast"].iloc[-1]) - float(df["ema_slow"].iloc[-1])) / close * 10_000
                ema_trend_alignment = ema_gap_bps * signal.side.direction
            if "macd" in df.columns and "macd_signal" in df.columns:
                macd_gap_bps = (float(df["macd"].iloc[-1]) - float(df["macd_signal"].iloc[-1])) / close * 10_000
                macd_alignment = macd_gap_bps * signal.side.direction
        return {
            "signal_strength": float(signal.strength),
            "confidence_hint": float(signal.confidence_hint),
            "strategy_market_score": float(strategy_market_score),
            "strategy_performance_score": float(strategy_performance_score),
            "volatility": float(snapshot.volatility),
            "spread_bps": float(book_features["spread_bps"]),
            "depth_imbalance_signed": float(depth_imbalance_signed),
            "sentiment_alignment": float(sentiment_alignment),
            "volume_zscore": float(volume_zscore),
            "rsi": float(rsi),
            "ema_trend_alignment": float(ema_trend_alignment),
            "macd_alignment": float(macd_alignment),
        }

    def predict(
        self,
        signal: StrategySignal,
        snapshot: MarketSnapshot,
        strategy_market_score: float,
        strategy_performance_score: float,
    ) -> float:
        feature_dict = self.features(signal, snapshot, strategy_market_score, strategy_performance_score)
        vector = self._vectorize(feature_dict)
        if self.is_fitted and self.model is not None:
            probability = float(self.model.predict_proba([vector])[0][1])
            return float(np.clip(probability, 0.0, 1.0))
        return self._heuristic_probability(feature_dict, signal.side)

    def update(
        self,
        feature_dict: dict[str, float] | list[float],
        profitable: bool,
    ) -> None:
        vector = feature_dict if isinstance(feature_dict, list) else self._vectorize(feature_dict)
        self.samples.append([float(value) for value in vector])
        self.labels.append(1 if profitable else 0)
        if self.model is not None and len(self.samples) >= self.min_samples_to_fit and len(set(self.labels)) == 2:
            self.model.fit(self.samples, self.labels)
            self.is_fitted = True

    def _vectorize(self, feature_dict: dict[str, float]) -> list[float]:
        return [float(feature_dict.get(name, 0.0)) for name in FEATURE_NAMES]

    def _heuristic_probability(self, feature_dict: dict[str, float], side: Side) -> float:
        spread_penalty = min(feature_dict["spread_bps"] / 20.0, 0.25)
        volatility_penalty = max(feature_dict["volatility"] - 0.025, 0.0) * 4.0
        sentiment_bonus = max(feature_dict["sentiment_alignment"], 0.0) * 0.08
        micro_bonus = max(feature_dict["depth_imbalance_signed"], 0.0) * 0.08 if side != Side.HOLD else 0.0
        trend_bonus = min(max(feature_dict["ema_trend_alignment"], 0.0) / 25.0, 1.0) * 0.06
        macd_bonus = min(max(feature_dict["macd_alignment"], 0.0) / 6.0, 1.0) * 0.05
        rsi = feature_dict["rsi"]
        if side == Side.BUY:
            rsi_penalty = max(rsi - 82.0, 0.0) / 100.0
        elif side == Side.SELL:
            rsi_penalty = max(18.0 - rsi, 0.0) / 100.0
        else:
            rsi_penalty = 0.0
        score = (
            0.34 * feature_dict["signal_strength"]
            + 0.2 * feature_dict["confidence_hint"]
            + 0.2 * feature_dict["strategy_market_score"]
            + 0.16 * feature_dict["strategy_performance_score"]
            + sentiment_bonus
            + micro_bonus
            + trend_bonus
            + macd_bonus
            - spread_penalty
            - volatility_penalty
            - rsi_penalty
        )
        return float(np.clip(0.35 + score, 0.0, 0.98))
