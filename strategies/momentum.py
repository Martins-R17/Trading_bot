"""Short-horizon momentum strategy."""

from __future__ import annotations

import numpy as np

from core.models import MarketSnapshot, Side, StrategySignal
from strategies.base_strategy import BaseStrategy


class MomentumStrategy(BaseStrategy):
    """Trades continuation when fast trend, returns, and RSI agree."""

    name = "momentum"
    min_bars = 60

    def __init__(
        self,
        return_threshold: float = 0.0012,
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
        self.return_threshold = return_threshold

    def generate_signal(self, snapshot: MarketSnapshot) -> StrategySignal:
        if not self.has_enough_data(snapshot):
            return self.hold_signal(snapshot, "insufficient_data")

        df = snapshot.ohlcv
        close = df["close"]
        price = float(close.iloc[-1])
        ema_fast = float(df["ema_fast"].iloc[-1])
        ema_slow = float(df["ema_slow"].iloc[-1])
        short_return = float(close.pct_change(5).iloc[-1])
        rsi = float(df["rsi"].iloc[-1])
        macd_hist = float(df["macd_hist"].iloc[-1])
        atr = max(self.atr(df), price * 0.0008)
        atr_bps = atr / max(price, 1e-9) * 10_000
        required_atr_bps = self.required_target_move_bps() / max(
            self.atr_take_profit_multiplier,
            1e-9,
        )
        target_hint = self.target_hint_metadata(atr_bps * self.atr_take_profit_multiplier)
        target_hint_pass = target_hint["target_move_bps"] >= target_hint["required_target_move_bps"]
        trend_side = Side.HOLD
        if ema_fast > ema_slow and short_return > self.return_threshold:
            trend_side = Side.BUY
        elif ema_fast < ema_slow and short_return < -self.return_threshold:
            trend_side = Side.SELL

        rsi_ok = (
            (trend_side == Side.BUY and 52 <= rsi <= 74)
            or (trend_side == Side.SELL and 26 <= rsi <= 48)
        )
        side = trend_side if rsi_ok else Side.HOLD
        base_metadata = {
            **self.indicator_metadata(df),
            **self.diagnostic_metadata(
                side_considered=trend_side,
                rsi_check="pass" if rsi_ok else "fail",
                ema_trend_check="pass" if trend_side != Side.HOLD else "fail",
                volatility_atr_check="pass" if atr_bps >= required_atr_bps else "fail",
                target_move_check="pass" if target_hint_pass else "fail",
                reward_cost_check="pass" if target_hint_pass else "fail",
                detailed_rejection_reason=(
                    "trend_not_confirmed" if trend_side == Side.HOLD else "rsi_not_confirmed"
                ),
            ),
            **target_hint,
            "return_5": short_return,
            "atr": atr,
            "atr_bps": atr_bps,
            "required_atr_bps": required_atr_bps,
        }

        if side == Side.HOLD:
            return self.hold_signal(snapshot, base_metadata["detailed_rejection_reason"], base_metadata)
        if not self.ema_trend_confirms(df, side):
            return self.hold_signal(
                snapshot,
                "trend_not_confirmed",
                {**base_metadata, "ema_trend_check": "fail", "detailed_rejection_reason": "trend_not_confirmed"},
            )
        if not self.macd_confirms(df, side):
            return self.hold_signal(
                snapshot,
                "macd_not_confirmed",
                {**base_metadata, "macd_check": "fail", "detailed_rejection_reason": "macd_not_confirmed"},
            )
        macd_hist_bps = macd_hist / max(price, 1e-9) * 10_000
        min_macd_hist_bps = max(0.5, self.round_trip_cost_bps * 0.02)
        if side == Side.BUY and macd_hist_bps < min_macd_hist_bps:
            return self.hold_signal(
                snapshot,
                "macd_not_confirmed",
                {
                    **base_metadata,
                    "macd_check": "fail",
                    "detailed_rejection_reason": "macd_not_confirmed",
                    "macd_hist_bps": macd_hist_bps,
                    "min_macd_hist_bps": min_macd_hist_bps,
                },
            )
        if side == Side.SELL and macd_hist_bps > -min_macd_hist_bps:
            return self.hold_signal(
                snapshot,
                "macd_not_confirmed",
                {
                    **base_metadata,
                    "macd_check": "fail",
                    "detailed_rejection_reason": "macd_not_confirmed",
                    "macd_hist_bps": macd_hist_bps,
                    "min_macd_hist_bps": min_macd_hist_bps,
                },
            )
        base_metadata = {
            **base_metadata,
            "macd_check": "pass",
            "detailed_rejection_reason": "",
        }
        trend_gap = abs(ema_fast - ema_slow) / price
        return_score = abs(short_return) / max(self.return_threshold * 3, 1e-9)
        rsi_score = 1.0 - abs(rsi - 55) / 55 if side == Side.BUY else 1.0 - abs(rsi - 45) / 55
        strength = self.clamp_strength(0.35 * return_score + 0.45 * min(trend_gap * 800, 1.0) + 0.2 * rsi_score)
        stop_loss = price - side.direction * atr * self.atr_stop_loss_multiplier
        take_profit = price + side.direction * atr * self.atr_take_profit_multiplier
        edge = self.edge_metadata(
            price,
            stop_loss,
            take_profit,
            atr,
            {"macd_hist_bps": macd_hist_bps, "return_5": short_return},
        )
        if atr_bps < required_atr_bps:
            return self.hold_signal(
                snapshot,
                "volatility_too_low",
                {
                    **base_metadata,
                    **edge,
                    "volatility_atr_check": "fail",
                    "target_move_check": "fail",
                    "detailed_rejection_reason": "volatility_too_low",
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
                    "volatility_atr_check": "pass",
                    "target_move_check": "fail",
                    "reward_cost_check": "fail",
                    "detailed_rejection_reason": "target_move_too_small",
                },
            )
        edge = {
            **edge,
            "volatility_atr_check": "pass",
            "target_move_check": "pass",
            "reward_cost_check": "pass",
            "expected_net_profit_check": "pending_risk",
        }

        return StrategySignal(
            strategy_name=self.name,
            symbol=snapshot.symbol,
            side=side,
            strength=strength,
            entry_price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence_hint=self.clamp_strength(0.55 + strength * 0.35),
            metadata={**base_metadata, **edge},
        )

    def score_market(self, snapshot: MarketSnapshot) -> float:
        if not self.has_enough_data(snapshot):
            return 0.0
        df = snapshot.ohlcv
        close = df["close"]
        trend_gap = abs(float(df["ema_fast"].iloc[-1]) - float(df["ema_slow"].iloc[-1])) / max(snapshot.close, 1e-9)
        vol = max(snapshot.volatility, float(df["rolling_volatility"].tail(30).mean()))
        volume_z = abs(float(df["volume_zscore"].iloc[-1]))
        score = 0.35 + min(trend_gap * 900, 0.35) + min(vol * 80, 0.2) + min(volume_z * 0.03, 0.1)
        if abs(float(close.pct_change(5).iloc[-1])) < self.return_threshold:
            score *= 0.7
        return float(np.clip(score, 0.0, 1.0))

