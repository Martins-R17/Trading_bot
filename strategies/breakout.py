"""Micro-breakout strategy."""

from __future__ import annotations

import numpy as np

from core.models import MarketSnapshot, Side, StrategySignal
from strategies.base_strategy import BaseStrategy


class BreakoutStrategy(BaseStrategy):
    """Trades tight range breakouts confirmed by volume expansion."""

    name = "breakout"
    min_bars = 80

    def __init__(self, lookback: int = 30, volume_multiplier: float = 1.15) -> None:
        super().__init__()
        self.lookback = lookback
        self.volume_multiplier = volume_multiplier

    def generate_signal(self, snapshot: MarketSnapshot) -> StrategySignal:
        if not self.has_enough_data(snapshot):
            return self.hold_signal(snapshot, "insufficient_data")

        df = snapshot.ohlcv
        price = snapshot.close
        previous = df.iloc[-self.lookback - 1 : -1]
        range_high = float(previous["high"].max())
        range_low = float(previous["low"].min())
        avg_volume = float(previous["volume"].mean())
        latest_volume = float(df["volume"].iloc[-1])
        atr = max(self.atr(df), price * 0.0009)

        volume_confirmed = latest_volume > avg_volume * self.volume_multiplier
        if price > range_high and volume_confirmed:
            side = Side.BUY
            breakout_distance = (price - range_high) / price
        elif price < range_low and volume_confirmed:
            side = Side.SELL
            breakout_distance = (range_low - price) / price
        else:
            return self.hold_signal(snapshot, "breakout_not_confirmed")

        volume_score = min(latest_volume / max(avg_volume * self.volume_multiplier, 1e-9), 2.0) / 2.0
        distance_score = min(breakout_distance / 0.003, 1.0)
        strength = self.clamp_strength(0.55 * distance_score + 0.45 * volume_score)
        stop_loss = price - side.direction * atr * 0.75
        take_profit = price + side.direction * atr * 1.25

        return StrategySignal(
            strategy_name=self.name,
            symbol=snapshot.symbol,
            side=side,
            strength=strength,
            entry_price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence_hint=self.clamp_strength(0.54 + strength * 0.36),
            metadata={"range_high": range_high, "range_low": range_low, "volume_ratio": latest_volume / avg_volume},
        )

    def score_market(self, snapshot: MarketSnapshot) -> float:
        if not self.has_enough_data(snapshot):
            return 0.0

        df = snapshot.ohlcv
        price = snapshot.close
        previous = df.iloc[-self.lookback - 1 : -1]
        range_width = (float(previous["high"].max()) - float(previous["low"].min())) / max(price, 1e-9)
        compression = 1.0 - min(range_width / 0.025, 1.0)
        volume_z = max(float(df["volume_zscore"].iloc[-1]), 0.0)
        volatility = min(snapshot.volatility / 0.018, 1.0)
        score = 0.25 + 0.35 * compression + 0.25 * volatility + 0.15 * min(volume_z / 3.0, 1.0)
        return float(np.clip(score, 0.0, 1.0))

