"""Short-horizon momentum strategy."""

from __future__ import annotations

import numpy as np

from core.models import MarketSnapshot, Side, StrategySignal
from strategies.base_strategy import BaseStrategy


class MomentumStrategy(BaseStrategy):
    """Trades continuation when fast trend, returns, and RSI agree."""

    name = "momentum"
    min_bars = 60

    def __init__(self, return_threshold: float = 0.0012) -> None:
        super().__init__()
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
        atr = max(self.atr(df), price * 0.0008)

        side = Side.HOLD
        if ema_fast > ema_slow and short_return > self.return_threshold and rsi < 76:
            side = Side.BUY
        elif ema_fast < ema_slow and short_return < -self.return_threshold and rsi > 24:
            side = Side.SELL

        if side == Side.HOLD:
            return self.hold_signal(snapshot, "momentum_not_confirmed")

        trend_gap = abs(ema_fast - ema_slow) / price
        return_score = abs(short_return) / max(self.return_threshold * 3, 1e-9)
        rsi_score = 1.0 - abs(rsi - 55) / 55 if side == Side.BUY else 1.0 - abs(rsi - 45) / 55
        strength = self.clamp_strength(0.35 * return_score + 0.45 * min(trend_gap * 800, 1.0) + 0.2 * rsi_score)
        stop_loss = price - side.direction * atr * 0.65
        take_profit = price + side.direction * atr * 1.05

        return StrategySignal(
            strategy_name=self.name,
            symbol=snapshot.symbol,
            side=side,
            strength=strength,
            entry_price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence_hint=self.clamp_strength(0.55 + strength * 0.35),
            metadata={"ema_fast": ema_fast, "ema_slow": ema_slow, "return_5": short_return, "rsi": rsi},
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

