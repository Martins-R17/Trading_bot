"""Strategy base class and shared technical helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque

import numpy as np
import pandas as pd

from core.models import MarketSnapshot, Side, StrategySignal
from data.preprocess import DataPreprocessor


class BaseStrategy(ABC):
    """Abstract interface every strategy must implement."""

    name = "base"
    min_bars = 50

    def __init__(self) -> None:
        self.recent_pnl: deque[float] = deque(maxlen=100)
        self.recent_wins: deque[int] = deque(maxlen=100)

    @abstractmethod
    def generate_signal(self, snapshot: MarketSnapshot) -> StrategySignal:
        """Return a directional signal or HOLD."""

    @abstractmethod
    def score_market(self, snapshot: MarketSnapshot) -> float:
        """Score how suitable the current market is for this strategy."""

    def record_trade(self, pnl: float) -> None:
        self.recent_pnl.append(float(pnl))
        self.recent_wins.append(1 if pnl > 0 else 0)

    def performance_score(self) -> float:
        """Convert recent realized performance to a bounded selector score."""

        if not self.recent_pnl:
            return 0.55
        pnl = np.array(self.recent_pnl, dtype=float)
        avg = float(np.mean(pnl))
        std = float(np.std(pnl)) or 1.0
        sharpe_like = avg / std
        win_rate = float(np.mean(self.recent_wins)) if self.recent_wins else 0.5
        return float(np.clip(0.5 + 0.18 * np.tanh(sharpe_like) + 0.25 * (win_rate - 0.5), 0.0, 1.0))

    def hold_signal(self, snapshot: MarketSnapshot, reason: str) -> StrategySignal:
        return StrategySignal(
            strategy_name=self.name,
            symbol=snapshot.symbol,
            side=Side.HOLD,
            strength=0.0,
            entry_price=snapshot.close,
            stop_loss=None,
            take_profit=None,
            confidence_hint=0.0,
            metadata={"reason": reason},
        )

    def has_enough_data(self, snapshot: MarketSnapshot) -> bool:
        return snapshot.ohlcv is not None and len(snapshot.ohlcv) >= self.min_bars

    def atr(self, df: pd.DataFrame) -> float:
        if "atr" in df.columns:
            return float(df["atr"].iloc[-1])
        enriched = DataPreprocessor.add_features(df)
        return float(enriched["atr"].iloc[-1])

    def rsi(self, close: pd.Series) -> float:
        return float(DataPreprocessor.rsi(close).iloc[-1])

    def clamp_strength(self, value: float) -> float:
        return float(np.clip(value, 0.0, 1.0))

