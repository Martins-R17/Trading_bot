"""Micro-breakout strategy."""

from __future__ import annotations

import numpy as np

from core.models import MarketSnapshot, Side, StrategySignal
from strategies.base_strategy import BaseStrategy


class BreakoutStrategy(BaseStrategy):
    """Trades tight range breakouts confirmed by volume expansion."""

    name = "breakout"
    min_bars = 80

    def __init__(
        self,
        lookback: int = 30,
        volume_multiplier: float = 1.15,
        min_target_move_bps: float = 75.0,
        atr_take_profit_multiplier: float = 3.0,
        atr_stop_loss_multiplier: float = 1.0,
        min_reward_to_cost_ratio: float = 3.0,
        round_trip_cost_bps: float = 24.0,
    ) -> None:
        super().__init__(
            min_target_move_bps=min_target_move_bps,
            atr_take_profit_multiplier=atr_take_profit_multiplier,
            atr_stop_loss_multiplier=atr_stop_loss_multiplier,
            min_reward_to_cost_ratio=min_reward_to_cost_ratio,
            round_trip_cost_bps=round_trip_cost_bps,
        )
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
        rsi = float(df["rsi"].iloc[-1])
        atr = max(self.atr(df), price * 0.0009)
        range_width = max(range_high - range_low, 0.0)
        range_width_bps = range_width / max(price, 1e-9) * 10_000
        atr_bps = atr / max(price, 1e-9) * 10_000
        base_metadata = {
            **self.indicator_metadata(df),
            "range_high": range_high,
            "range_low": range_low,
            "range_width_bps": range_width_bps,
            "atr": atr,
            "atr_bps": atr_bps,
        }

        volume_confirmed = latest_volume > avg_volume * self.volume_multiplier
        if price > range_high and volume_confirmed:
            side = Side.BUY
            breakout_distance = (price - range_high) / price
        elif price < range_low and volume_confirmed:
            side = Side.SELL
            breakout_distance = (range_low - price) / price
        else:
            return self.hold_signal(snapshot, "breakout_not_confirmed", base_metadata)

        breakout_distance_bps = breakout_distance * 10_000
        base_metadata = {
            **base_metadata,
            "breakout_distance_bps": breakout_distance_bps,
            "volume_ratio": latest_volume / max(avg_volume, 1e-9),
        }

        if not self.ema_trend_confirms(df, side, tolerance_bps=4.0):
            return self.hold_signal(snapshot, "ema_trend_filter", base_metadata)
        if not self.macd_confirms(df, side, tolerance_bps=0.5):
            return self.hold_signal(snapshot, "macd_not_confirmed", base_metadata)
        if side == Side.BUY and rsi > 82:
            return self.hold_signal(snapshot, "rsi_overextended", base_metadata)
        if side == Side.SELL and rsi < 18:
            return self.hold_signal(snapshot, "rsi_overextended", base_metadata)
        target_distance = max(atr * self.atr_take_profit_multiplier, range_width * 0.6)
        stop_loss = price - side.direction * atr * self.atr_stop_loss_multiplier
        take_profit = price + side.direction * target_distance
        edge = self.edge_metadata(
            price,
            stop_loss,
            take_profit,
            atr,
            {
                "range_width_bps": range_width_bps,
                "breakout_distance_bps": breakout_distance_bps,
                "volume_ratio": latest_volume / max(avg_volume, 1e-9),
            },
        )
        if range_width_bps < self.required_target_move_bps() * 0.6:
            return self.hold_signal(
                snapshot,
                "breakout_range_too_small_after_costs",
                {**base_metadata, **edge},
            )
        target_reason = self.target_too_small_reason(edge)
        if target_reason:
            return self.hold_signal(snapshot, target_reason, {**base_metadata, **edge})

        volume_score = min(latest_volume / max(avg_volume * self.volume_multiplier, 1e-9), 2.0) / 2.0
        distance_score = min(breakout_distance / 0.003, 1.0)
        strength = self.clamp_strength(0.55 * distance_score + 0.45 * volume_score)

        return StrategySignal(
            strategy_name=self.name,
            symbol=snapshot.symbol,
            side=side,
            strength=strength,
            entry_price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence_hint=self.clamp_strength(0.54 + strength * 0.36),
            metadata={
                **base_metadata,
                **edge,
            },
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

