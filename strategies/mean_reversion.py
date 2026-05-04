"""Intraday mean-reversion scalping strategy."""

from __future__ import annotations

import numpy as np

from core.models import MarketSnapshot, Side, StrategySignal
from strategies.base_strategy import BaseStrategy


class MeanReversionStrategy(BaseStrategy):
    """Fades stretched moves in range-bound markets."""

    name = "mean_reversion"
    min_bars = 80

    def __init__(
        self,
        lookback: int = 40,
        entry_z: float = 1.65,
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
        ema_fast = float(df["ema_fast"].iloc[-1])
        ema_slow = float(df["ema_slow"].iloc[-1])
        atr = max(self.atr(df), price * 0.0007)
        mean_distance = abs(price - rolling_mean)
        mean_distance_bps = mean_distance / max(price, 1e-9) * 10_000
        atr_bps = atr / max(price, 1e-9) * 10_000
        target_hint = self.target_hint_metadata(
            min(mean_distance_bps, atr_bps * self.atr_take_profit_multiplier)
        )
        target_hint_pass = target_hint["target_move_bps"] >= target_hint["required_target_move_bps"]
        raw_side = Side.HOLD
        if zscore <= -self.entry_z:
            raw_side = Side.BUY
        elif zscore >= self.entry_z:
            raw_side = Side.SELL
        rsi_ok = (
            (raw_side == Side.BUY and rsi < 38)
            or (raw_side == Side.SELL and rsi > 62)
        )
        base_metadata = {
            **self.indicator_metadata(df),
            **self.diagnostic_metadata(
                side_considered=raw_side,
                rsi_check="pass" if rsi_ok else "fail",
                volatility_atr_check="pass" if raw_side != Side.HOLD else "fail",
                target_move_check="pass" if target_hint_pass else "fail",
                reward_cost_check="pass" if target_hint_pass else "fail",
                detailed_rejection_reason=(
                    "target_move_too_small" if raw_side == Side.HOLD else "rsi_not_confirmed"
                ),
            ),
            **target_hint,
            "zscore": zscore,
            "rolling_mean": rolling_mean,
            "mean_distance_bps": mean_distance_bps,
            "atr": atr,
            "atr_bps": atr_bps,
        }

        side = raw_side if rsi_ok else Side.HOLD
        if side == Side.HOLD:
            return self.hold_signal(snapshot, base_metadata["detailed_rejection_reason"], base_metadata)

        ema_gap_bps = abs(ema_fast - ema_slow) / max(price, 1e-9) * 10_000
        base_metadata["ema_gap_bps"] = ema_gap_bps
        trend_side = Side.BUY if ema_fast >= ema_slow else Side.SELL
        if side != trend_side and ema_gap_bps > 35:
            return self.hold_signal(
                snapshot,
                "trend_not_confirmed",
                {**base_metadata, "ema_trend_check": "fail", "detailed_rejection_reason": "trend_not_confirmed"},
            )
        if not self.macd_reversal_confirms(df, side):
            return self.hold_signal(
                snapshot,
                "macd_not_confirmed",
                {
                    **base_metadata,
                    "ema_trend_check": "pass",
                    "macd_check": "fail",
                    "detailed_rejection_reason": "macd_not_confirmed",
                },
            )
        base_metadata = {
            **base_metadata,
            "ema_trend_check": "pass",
            "macd_check": "pass",
            "detailed_rejection_reason": "",
        }
        stop_loss = price - side.direction * atr * self.atr_stop_loss_multiplier
        target_distance = min(mean_distance, atr * self.atr_take_profit_multiplier)
        take_profit = price + side.direction * target_distance
        edge = self.edge_metadata(
            price,
            stop_loss,
            take_profit,
            atr,
            {
                "zscore": zscore,
                "rolling_mean": rolling_mean,
                "mean_distance_bps": mean_distance_bps,
                "ema_gap_bps": ema_gap_bps,
            },
        )
        if mean_distance_bps < self.required_target_move_bps():
            return self.hold_signal(
                snapshot,
                "target_move_too_small",
                {
                    **base_metadata,
                    **edge,
                    "target_move_check": "fail",
                    "reward_cost_check": "fail",
                    "detailed_rejection_reason": "target_move_too_small",
                },
            )
        target_reason = self.target_too_small_reason(edge)
        if target_reason:
            return self.hold_signal(
                snapshot,
                "target_move_too_small",
                {
                    **base_metadata,
                    **edge,
                    "target_move_check": "fail",
                    "reward_cost_check": "fail",
                    "detailed_rejection_reason": "target_move_too_small",
                },
            )
        edge = {
            **edge,
            "target_move_check": "pass",
            "reward_cost_check": "pass",
            "expected_net_profit_check": "pending_risk",
        }

        stretch_score = min(abs(zscore) / 3.0, 1.0)
        rsi_score = min(abs(rsi - 50) / 35.0, 1.0)
        strength = self.clamp_strength(0.65 * stretch_score + 0.35 * rsi_score)

        return StrategySignal(
            strategy_name=self.name,
            symbol=snapshot.symbol,
            side=side,
            strength=strength,
            entry_price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence_hint=self.clamp_strength(0.52 + strength * 0.33),
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
        trend_gap = abs(float(df["ema_fast"].iloc[-1]) - float(df["ema_slow"].iloc[-1])) / max(price, 1e-9)
        recent_range = (float(df["high"].tail(self.lookback).max()) - float(df["low"].tail(self.lookback).min())) / price
        volatility_score = 1.0 - min(snapshot.volatility / 0.02, 1.0)
        range_score = min(recent_range / 0.025, 1.0)
        trend_penalty = min(trend_gap * 900, 0.5)
        return float(np.clip(0.25 + 0.35 * volatility_score + 0.35 * range_score - trend_penalty, 0.0, 1.0))

