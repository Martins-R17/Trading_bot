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

    def __init__(
        self,
        min_target_move_bps: float = 75.0,
        atr_take_profit_multiplier: float = 3.0,
        atr_stop_loss_multiplier: float = 1.0,
        min_reward_to_cost_ratio: float = 3.0,
        round_trip_cost_bps: float = 24.0,
    ) -> None:
        self.recent_pnl: deque[float] = deque(maxlen=100)
        self.recent_wins: deque[int] = deque(maxlen=100)
        self.min_target_move_bps = float(min_target_move_bps)
        self.atr_take_profit_multiplier = float(atr_take_profit_multiplier)
        self.atr_stop_loss_multiplier = float(atr_stop_loss_multiplier)
        self.min_reward_to_cost_ratio = float(min_reward_to_cost_ratio)
        self.round_trip_cost_bps = float(round_trip_cost_bps)

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

    def hold_signal(
        self,
        snapshot: MarketSnapshot,
        reason: str,
        metadata: dict[str, float | str] | None = None,
    ) -> StrategySignal:
        return StrategySignal(
            strategy_name=self.name,
            symbol=snapshot.symbol,
            side=Side.HOLD,
            strength=0.0,
            entry_price=snapshot.close,
            stop_loss=None,
            take_profit=None,
            confidence_hint=0.0,
            metadata={"reason": reason, **(metadata or {})},
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

    def required_target_move_bps(self) -> float:
        return max(
            self.min_target_move_bps,
            self.round_trip_cost_bps * self.min_reward_to_cost_ratio,
        )

    def target_move_bps(self, entry: float, take_profit: float) -> float:
        return abs(float(take_profit) - float(entry)) / max(float(entry), 1e-9) * 10_000

    def stop_move_bps(self, entry: float, stop_loss: float) -> float:
        return abs(float(entry) - float(stop_loss)) / max(float(entry), 1e-9) * 10_000

    def edge_metadata(
        self,
        entry: float,
        stop_loss: float,
        take_profit: float,
        atr: float,
        extra: dict[str, float] | None = None,
    ) -> dict[str, float]:
        target_move_bps = self.target_move_bps(entry, take_profit)
        stop_move_bps = self.stop_move_bps(entry, stop_loss)
        round_trip_cost_bps = max(self.round_trip_cost_bps, 1e-9)
        return {
            "atr": float(atr),
            "atr_bps": float(atr / max(entry, 1e-9) * 10_000),
            "target_move_bps": float(target_move_bps),
            "stop_move_bps": float(stop_move_bps),
            "required_target_move_bps": float(self.required_target_move_bps()),
            "reward_cost_ratio": float(target_move_bps / round_trip_cost_bps),
            "round_trip_cost_bps": float(self.round_trip_cost_bps),
            "min_reward_to_cost_ratio": float(self.min_reward_to_cost_ratio),
            **(extra or {}),
        }

    def target_too_small_reason(self, edge: dict[str, float]) -> str:
        if edge["target_move_bps"] >= edge["required_target_move_bps"]:
            return ""
        return (
            "target_move_too_small_after_costs:"
            f"target={edge['target_move_bps']:.2f}bps "
            f"required={edge['required_target_move_bps']:.2f}bps "
            f"reward_cost={edge['reward_cost_ratio']:.2f}x"
        )

    def clamp_strength(self, value: float) -> float:
        return float(np.clip(value, 0.0, 1.0))

