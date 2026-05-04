"""Intraday mean-reversion scalping strategy."""

from __future__ import annotations

import numpy as np

from core.models import MarketSnapshot, Side, StrategySignal
from strategies.base_strategy import BaseStrategy


class MeanReversionStrategy(BaseStrategy):
    """Fades stretched moves in range-bound markets."""

    name = "mean_reversion"
    min_bars = 80

    def __init__(self, lookback: int = 40, entry_z: float = 1.65) -> None:
        super().__init__()
        self.lookback = lookback
        self.entry_z = entry_z

    def generate_signal(self, snapshot: MarketSnapshot) -> StrategySignal:
        if not self.has_enough_data(snapshot):
            return self.hold_signal(snapshot, "insufficient_data")

        df = snapshot.ohlcv
        close = df["close"]
        price = float(close.iloc[-1])
        rolling_mean = float(close.rolling(self.lookback).mean().iloc[-1])
        rolling_std = float(close.rolling(self.lookback).std().iloc[-1] or 0.0)
        if rolling_std <= 0:
            return self.hold_signal(snapshot, "zero_variance")

        zscore = (price - rolling_mean) / rolling_std
        rsi = float(df["rsi"].iloc[-1])
        atr = max(self.atr(df), price * 0.0007)

        if zscore <= -self.entry_z and rsi < 38:
            side = Side.BUY
        elif zscore >= self.entry_z and rsi > 62:
            side = Side.SELL
        else:
            return self.hold_signal(snapshot, "mean_reversion_not_confirmed")

        stretch_score = min(abs(zscore) / 3.0, 1.0)
        rsi_score = min(abs(rsi - 50) / 35.0, 1.0)
        strength = self.clamp_strength(0.65 * stretch_score + 0.35 * rsi_score)
        stop_loss = price - side.direction * atr * 0.85
        take_profit = price + side.direction * min(abs(price - rolling_mean), atr * 1.1)

        return StrategySignal(
            strategy_name=self.name,
            symbol=snapshot.symbol,
            side=side,
            strength=strength,
            entry_price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence_hint=self.clamp_strength(0.52 + strength * 0.33),
            metadata={"zscore": zscore, "rsi": rsi, "rolling_mean": rolling_mean},
        )

    def score_market(self, snapshot: MarketSnapshot) -> float:
        if not self.has_enough_data(snapshot):
            return 0.0

        df = snapshot.ohlcv
        price = snapshot.close
        trend_gap = abs(float(df["ema_fast"].iloc[-1]) - float(df["ema_slow"].iloc[-1])) / max(price, 1e-9)
        recent_range = (float(df["high"].tail(self.lookback).max()) - float(df["low"].tail(self.lookback).min())) / price
        volatility_score = 1.0 - min(snapshot.volatility / 0.02, 1.0)
        range_score = min(recent_range / 0.025, 1.0)
        trend_penalty = min(trend_gap * 900, 0.5)
        return float(np.clip(0.25 + 0.35 * volatility_score + 0.35 * range_score - trend_penalty, 0.0, 1.0))

