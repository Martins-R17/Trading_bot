"""Causality and execution invariants for the backtesting-only search engine."""

from __future__ import annotations

import unittest
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from backtesting.scalping_search import (
    FeatureArrays,
    SearchSpec,
    SearchTrade,
    build_feature_arrays,
    daily_metrics,
    default_specs,
    simulate_trade,
)
from data.preprocess import DataPreprocessor


def feature_arrays(length: int = 100) -> FeatureArrays:
    base = np.full(length, 100.0)
    timestamps = np.arange(length, dtype=float) * 60_000 + 1_700_000_000_000
    return FeatureArrays(
        timestamp=timestamps,
        open=base.copy(), high=np.full(length, 100.2), low=np.full(length, 99.8), close=base.copy(),
        volume=np.full(length, 10.0), ema_fast=np.full(length, 100.1), ema_slow=np.full(length, 99.9),
        macd_hist=np.full(length, 0.1), rsi=np.full(length, 60.0), atr=np.full(length, 1.0),
        atr_bps=np.full(length, 100.0), atr_percentile=np.full(length, 0.5),
        candle_range_atr_ratio=np.full(length, 0.4), hour_utc=np.full(length, 12),
        volume_ratio=np.full(length, 1.5), close_position=np.full(length, 0.5), vwap=base.copy(),
        return_3_bps=np.zeros(length), return_5_bps=np.zeros(length), return_10_bps=np.zeros(length),
        trend_20_bps=np.full(length, 20.0), range_high_20=np.full(length, 101.0),
        range_low_20=np.full(length, 99.0), range_high_40=np.full(length, 102.0),
        range_low_40=np.full(length, 98.0), range_bps_20=np.full(length, 200.0),
        range_bps_40=np.full(length, 400.0),
    )


def trade(timestamp: float, exit_timestamp: float, net: float) -> SearchTrade:
    return SearchTrade(
        timestamp=timestamp, exit_timestamp=exit_timestamp, entry_index=1, exit_index=2, side="buy",
        entry_price=100.0, exit_price=100.0 + net, exit_reason="max_horizon_exit", hold_candles=1,
        gross_pnl=net, fees=0.0, slippage_costs=0.0, spread_costs=0.0, total_costs=0.0,
        net_pnl=net, position_notional=100.0, target_bps=100.0, stop_bps=100.0,
    )


class ScalpingSearchInvariantTests(unittest.TestCase):
    def test_feature_prefix_is_invariant_to_future_changes(self) -> None:
        length = 240
        timestamp = np.arange(length, dtype=float) * 60_000 + 1_700_000_000_000
        close = 100.0 + np.arange(length) * 0.03
        raw = pd.DataFrame({
            "timestamp": timestamp, "open": close - 0.02, "high": close + 0.1,
            "low": close - 0.1, "close": close, "volume": 10.0 + np.arange(length) % 11,
        })
        changed = raw.copy()
        changed.loc[180:, ["open", "high", "low", "close", "volume"]] *= 4.0
        original = build_feature_arrays(DataPreprocessor.add_features(DataPreprocessor.normalize_ohlcv(raw)))
        mutated = build_feature_arrays(DataPreprocessor.add_features(DataPreprocessor.normalize_ohlcv(changed)))
        for name in ("atr", "atr_bps", "atr_percentile", "ema_fast", "ema_slow", "macd_hist", "rsi", "vwap"):
            np.testing.assert_allclose(getattr(original, name)[:180], getattr(mutated, name)[:180], rtol=0, atol=1e-10)

    def test_entry_uses_next_candle_open(self) -> None:
        arrays = feature_arrays()
        arrays.close[60] = 95.0
        arrays.open[61] = 101.0
        arrays.atr_bps[60] = 0.0
        arrays.candle_range_atr_ratio[60] = 0.0
        spec = SearchSpec("test", "momentum_burst", "buy", 20, 500.0, 500.0, 2)
        result = simulate_trade(60, spec, arrays, 100.0, 1.0, 0.0005, 0.0, 0.0, 0, 0, 0.004, 1.0)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.entry_index, 61)
        self.assertAlmostEqual(result.entry_price, 101.0)

    def test_ambiguous_candle_resolves_stop_first(self) -> None:
        for side in ("buy", "sell"):
            arrays = feature_arrays()
            arrays.open[61] = 100.0
            arrays.high[61] = 102.0
            arrays.low[61] = 98.0
            spec = SearchSpec("test", "momentum_burst", side, 20, 100.0, 100.0, 2)
            result = simulate_trade(60, spec, arrays, 100.0, 1.0, 0.0005, 0.0, 0.0, 0, 0, 0.004, 1.0)
            self.assertIsNotNone(result)
            assert result is not None
            self.assertEqual(result.exit_reason, "stop_loss_hit")

    def test_daily_metrics_use_explicit_full_period(self) -> None:
        start = datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp() * 1000
        end = datetime(2026, 1, 10, tzinfo=timezone.utc).timestamp() * 1000
        trades = [trade(start, start + 86_400_000, 1.0), trade(start, start + 4 * 86_400_000, -0.5)]
        metrics = daily_metrics(trades, 100.0, start, end)
        self.assertEqual(metrics["calendar_days"], 10)
        self.assertEqual(metrics["zero_trade_days"], 8)
        self.assertAlmostEqual(metrics["trades_per_day"], 0.2)

    def test_legacy_scalping_grid_still_builds(self) -> None:
        self.assertGreater(len(default_specs("1m")), 0)


if __name__ == "__main__":
    unittest.main()
