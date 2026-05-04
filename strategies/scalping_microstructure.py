"""Order-book microstructure scalping strategy."""

from __future__ import annotations

import numpy as np

from core.models import MarketSnapshot, Side, StrategySignal
from strategies.base_strategy import BaseStrategy


class ScalpingMicrostructureStrategy(BaseStrategy):
    """Uses spread, depth imbalance, and liquidity to capture tiny moves."""

    name = "scalping_microstructure"
    min_bars = 30

    def __init__(
        self,
        imbalance_threshold: float = 0.35,
        max_spread_bps: float = 8.0,
        min_macd_hist_bps: float = 0.05,
        neutral_rsi_band: float = 5.0,
        strong_liquidity_imbalance: float = 0.60,
        strong_liquidity_depth_quote: float = 500_000.0,
    ) -> None:
        super().__init__()
        self.imbalance_threshold = imbalance_threshold
        self.max_spread_bps = max_spread_bps
        self.min_macd_hist_bps = min_macd_hist_bps
        self.neutral_rsi_band = neutral_rsi_band
        self.strong_liquidity_imbalance = strong_liquidity_imbalance
        self.strong_liquidity_depth_quote = strong_liquidity_depth_quote

    def generate_signal(self, snapshot: MarketSnapshot) -> StrategySignal:
        if not self.has_enough_data(snapshot):
            return self.hold_signal(snapshot, "insufficient_data")
        book = snapshot.order_book
        if book is None or book.mid_price is None:
            return self.hold_signal(snapshot, "missing_order_book")
        if book.spread_bps > self.max_spread_bps:
            return self.hold_signal(snapshot, "spread_too_wide")

        imbalance = book.depth_imbalance(levels=5)
        if imbalance >= self.imbalance_threshold:
            side = Side.BUY
            entry = float(book.best_ask or snapshot.close)
        elif imbalance <= -self.imbalance_threshold:
            side = Side.SELL
            entry = float(book.best_bid or snapshot.close)
        else:
            return self.hold_signal(snapshot, "imbalance_not_confirmed")

        df = snapshot.ohlcv
        rsi = float(df["rsi"].iloc[-1])
        macd_hist = float(df["macd_hist"].iloc[-1])
        macd_hist_bps = macd_hist / max(entry, 1e-9) * 10_000
        depth_quote = book.total_depth_quote(levels=5)
        if not self.ema_trend_confirms(df, side, tolerance_bps=3.0):
            return self.hold_signal(snapshot, "ema_trend_filter")
        if not self.macd_confirms(df, side, tolerance_bps=0.5):
            return self.hold_signal(snapshot, "macd_not_confirmed")
        if side.direction * macd_hist_bps < self.min_macd_hist_bps:
            return self.hold_signal(snapshot, "macd_hist_not_aligned")
        if side == Side.BUY and rsi > 80:
            return self.hold_signal(snapshot, "rsi_overextended")
        if side == Side.SELL and rsi < 20:
            return self.hold_signal(snapshot, "rsi_overextended")
        liquidity_very_strong = (
            abs(imbalance) >= self.strong_liquidity_imbalance
            and depth_quote >= self.strong_liquidity_depth_quote
        )
        if abs(rsi - 50.0) <= self.neutral_rsi_band and not liquidity_very_strong:
            return self.hold_signal(snapshot, "neutral_rsi_requires_stronger_liquidity")

        spread = max(book.spread or entry * 0.0001, entry * 0.00003)
        strength = self.clamp_strength((abs(imbalance) - self.imbalance_threshold) / (1 - self.imbalance_threshold))
        stop_distance = max(spread * 3.0, entry * 0.0012)
        take_distance = max(spread * 6.0, entry * 0.0030)
        stop_loss = entry - side.direction * stop_distance
        take_profit = entry + side.direction * take_distance

        return StrategySignal(
            strategy_name=self.name,
            symbol=snapshot.symbol,
            side=side,
            strength=strength,
            entry_price=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence_hint=self.clamp_strength(0.58 + strength * 0.34),
            metadata={
                **self.indicator_metadata(df),
                "imbalance": imbalance,
                "spread_bps": book.spread_bps,
                "depth_quote": depth_quote,
                "macd_hist_bps": macd_hist_bps,
            },
        )

    def score_market(self, snapshot: MarketSnapshot) -> float:
        book = snapshot.order_book
        if book is None or book.mid_price is None:
            return 0.0
        spread_score = 1.0 - min(book.spread_bps / self.max_spread_bps, 1.0)
        depth_score = min(book.total_depth_quote(levels=5) / 500_000, 1.0)
        imbalance_score = min(abs(book.depth_imbalance(levels=5)) / 0.6, 1.0)
        vol_score = 1.0 - min(snapshot.volatility / 0.035, 1.0)
        return float(np.clip(0.35 * spread_score + 0.25 * depth_score + 0.25 * imbalance_score + 0.15 * vol_score, 0.0, 1.0))

