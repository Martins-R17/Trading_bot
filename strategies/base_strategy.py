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

    def latest_float(self, df: pd.DataFrame, column: str, default: float = 0.0) -> float:
        if column not in df.columns or len(df) == 0:
            return default
        value = float(df[column].iloc[-1])
        if np.isfinite(value):
            return value
        return default

    def ema_trend_confirms(self, df: pd.DataFrame, side: Side, tolerance_bps: float = 2.0) -> bool:
        if side == Side.HOLD or len(df) == 0:
            return False
        price = max(self.latest_float(df, "close", 0.0), 1e-9)
        ema_fast = self.latest_float(df, "ema_fast", price)
        ema_slow = self.latest_float(df, "ema_slow", price)
        tolerance = price * tolerance_bps / 10_000
        if side == Side.BUY:
            return ema_fast >= ema_slow - tolerance
        return ema_fast <= ema_slow + tolerance

    def macd_confirms(self, df: pd.DataFrame, side: Side, tolerance_bps: float = 0.2) -> bool:
        if side == Side.HOLD or len(df) == 0:
            return False
        price = max(self.latest_float(df, "close", 0.0), 1e-9)
        tolerance = price * tolerance_bps / 10_000
        macd = self.latest_float(df, "macd", 0.0)
        macd_signal = self.latest_float(df, "macd_signal", 0.0)
        macd_hist = self.latest_float(df, "macd_hist", macd - macd_signal)
        if side == Side.BUY:
            return macd >= macd_signal - tolerance and macd_hist >= -tolerance
        return macd <= macd_signal + tolerance and macd_hist <= tolerance

    def macd_reversal_confirms(self, df: pd.DataFrame, side: Side) -> bool:
        if side == Side.HOLD or "macd_hist" not in df.columns or len(df) < 2:
            return False
        macd_hist = float(df["macd_hist"].iloc[-1])
        previous_hist = float(df["macd_hist"].iloc[-2])
        if not np.isfinite(macd_hist) or not np.isfinite(previous_hist):
            return False
        if side == Side.BUY:
            return macd_hist >= previous_hist
        return macd_hist <= previous_hist

    def indicator_metadata(self, df: pd.DataFrame) -> dict[str, float]:
        return {
            "rsi": self.latest_float(df, "rsi", 50.0),
            "ema_fast": self.latest_float(df, "ema_fast", 0.0),
            "ema_slow": self.latest_float(df, "ema_slow", 0.0),
            "macd": self.latest_float(df, "macd", 0.0),
            "macd_signal": self.latest_float(df, "macd_signal", 0.0),
            "macd_hist": self.latest_float(df, "macd_hist", 0.0),
        }

    def clamp_strength(self, value: float) -> float:
        return float(np.clip(value, 0.0, 1.0))

