"""Lightweight NLP sentiment model for crypto news.

The default analyzer is self-contained: a small scikit-learn classifier trained
on seed phrases is blended with a transparent finance/crypto lexicon. This
avoids external model downloads while keeping the interface replaceable.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

import numpy as np

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
except ImportError:  # pragma: no cover - dependency is in requirements.
    TfidfVectorizer = None
    LogisticRegression = None
    Pipeline = Any

from data.news_data import NewsItem


class SentimentAnalyzer:
    """Scores text in [-1, 1], where positive favors long exposure."""

    POSITIVE_TERMS = {
        "adoption",
        "approval",
        "beat",
        "breakout",
        "bull",
        "bullish",
        "buy",
        "growth",
        "inflow",
        "partnership",
        "rally",
        "rebound",
        "record",
        "surge",
        "upgrade",
    }
    NEGATIVE_TERMS = {
        "ban",
        "bear",
        "bearish",
        "crackdown",
        "drop",
        "exploit",
        "hack",
        "lawsuit",
        "liquidation",
        "outflow",
        "rejection",
        "selloff",
        "slump",
        "warning",
        "withdrawal",
    }

    def __init__(self) -> None:
        self.model = self._train_seed_model()

    def score(self, text: str) -> float:
        if not text.strip():
            return 0.0
        ml_score = self._model_score(text)
        lexicon_score = self._lexicon_score(text)
        return float(np.clip(0.65 * ml_score + 0.35 * lexicon_score, -1.0, 1.0))

    def score_items(self, items: list[NewsItem], symbols: tuple[str, ...]) -> dict[str, float]:
        """Aggregate recent article sentiment per configured trading symbol."""

        if not items:
            return {symbol: 0.0 for symbol in symbols}

        scores: dict[str, list[float]] = {symbol: [] for symbol in symbols}
        for item in items:
            item_score = self.score(item.text)
            text_upper = item.text.upper()
            for symbol in symbols:
                base = symbol.split("/", 1)[0].upper()
                quote = symbol.split("/", 1)[1].upper() if "/" in symbol else ""
                symbol_match = base in item.symbols or base in text_upper
                market_match = "CRYPTO" in text_upper or "BITCOIN" in text_upper or "ETHEREUM" in text_upper
                if symbol_match or (quote == "USDT" and market_match):
                    scores[symbol].append(item_score)

        return {
            symbol: float(np.clip(np.mean(symbol_scores), -1.0, 1.0)) if symbol_scores else 0.0
            for symbol, symbol_scores in scores.items()
        }

    def _model_score(self, text: str) -> float:
        if self.model is None:
            return self._lexicon_score(text)
        probabilities = self.model.predict_proba([text])[0]
        classes = list(self.model.named_steps["clf"].classes_)
        positive = probabilities[classes.index(1)] if 1 in classes else 0.0
        negative = probabilities[classes.index(-1)] if -1 in classes else 0.0
        return float(positive - negative)

    def _lexicon_score(self, text: str) -> float:
        tokens = re.findall(r"[a-zA-Z]+", text.lower())
        counts = Counter(tokens)
        positive = sum(counts[token] for token in self.POSITIVE_TERMS)
        negative = sum(counts[token] for token in self.NEGATIVE_TERMS)
        total = positive + negative
        if total == 0:
            return 0.0
        return float((positive - negative) / total)

    def _train_seed_model(self) -> Pipeline | None:
        if TfidfVectorizer is None or LogisticRegression is None:
            return None
        samples = [
            ("bitcoin rallies after institutional inflows accelerate", 1),
            ("exchange approval sparks bullish crypto market breakout", 1),
            ("ethereum adoption grows after major partnership announcement", 1),
            ("crypto market rebounds as liquidity improves", 1),
            ("token suffers exploit and users rush to withdraw funds", -1),
            ("regulator announces crackdown and lawsuit against exchange", -1),
            ("bitcoin selloff deepens after liquidation wave", -1),
            ("bearish warning follows sharp outflows from crypto funds", -1),
            ("market trades sideways before central bank decision", 0),
            ("exchange publishes routine maintenance schedule", 0),
            ("analysts await fresh catalyst as volatility cools", 0),
            ("stablecoin issuer releases monthly reserve attestation", 0),
        ]
        model = Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
                ("clf", LogisticRegression(max_iter=500, class_weight="balanced")),
            ]
        )
        model.fit([text for text, _ in samples], [label for _, label in samples])
        return model
