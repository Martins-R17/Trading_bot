"""Causality tests for structural OHLC features."""

from __future__ import annotations

from dataclasses import fields
from datetime import datetime, timezone
import unittest

import numpy as np

from backtesting.structural_features import StructuralFeatures, build_structural_features


def minute_timestamps(length: int, start: datetime | None = None) -> np.ndarray:
    start = start or datetime(2026, 1, 2, tzinfo=timezone.utc)
    return start.timestamp() * 1_000 + np.arange(length) * 60_000


def sample_inputs(length: int = 360) -> dict[str, np.ndarray | int]:
    timestamps = minute_timestamps(length)
    close = 100.0 + np.arange(length) * 0.01
    open_ = close - 0.005
    return {
        "timestamps": timestamps,
        "open_": open_,
        "high": np.maximum(open_, close) + 0.05,
        "low": np.minimum(open_, close) - 0.05,
        "close": close,
        "atr_bps": np.full(length, 12.0),
        "range_bps_20": np.full(length, 80.0),
        "range_bps_40": np.full(length, 120.0),
        "timeframe_minutes": 1,
    }


class StructuralFeatureTests(unittest.TestCase):
    def test_all_outputs_are_aligned_and_future_invariant(self) -> None:
        original_inputs = sample_inputs()
        changed_inputs = {
            name: value.copy() if isinstance(value, np.ndarray) else value
            for name, value in original_inputs.items()
        }
        cut = 240
        for name in (
            "open_",
            "high",
            "low",
            "close",
            "atr_bps",
            "range_bps_20",
            "range_bps_40",
        ):
            changed_inputs[name][cut:] *= 4.0

        original = build_structural_features(**original_inputs)
        changed = build_structural_features(**changed_inputs)

        for field in fields(StructuralFeatures):
            first = getattr(original, field.name)
            second = getattr(changed, field.name)
            self.assertEqual(len(first), len(original_inputs["timestamps"]))
            if np.issubdtype(first.dtype, np.number):
                np.testing.assert_allclose(first[:cut], second[:cut], equal_nan=True)
            else:
                np.testing.assert_array_equal(first[:cut], second[:cut])

    def test_higher_timeframe_trend_uses_only_completed_bucket(self) -> None:
        inputs = sample_inputs(31)
        inputs["timestamps"] = np.arange(31) * 60_000
        features = build_structural_features(**inputs)

        self.assertTrue(np.isnan(features.htf_trend_bps[:15]).all())
        self.assertTrue(np.all(features.htf_trend_direction[:15] == 0))
        expected = (inputs["close"][14] - inputs["open_"][0]) / inputs["open_"][0] * 10_000
        self.assertAlmostEqual(features.htf_trend_bps[15], expected)
        self.assertEqual(features.htf_trend_direction[15], 1)
        self.assertTrue(np.all(features.htf_trend_bps[15:30] == features.htf_trend_bps[15]))

    def test_session_boundaries_publish_completed_asia_and_london_ranges(self) -> None:
        minute_offsets = np.asarray([418, 419, 420, 719, 720, 959, 960])
        day = datetime(2026, 1, 2, tzinfo=timezone.utc).timestamp() * 1_000
        timestamps = day + minute_offsets * 60_000
        close = np.asarray([100.0, 101.0, 102.0, 103.0, 102.5, 104.0, 103.5])
        open_ = close.copy()
        high = np.asarray([101.0, 103.0, 102.5, 105.0, 103.0, 104.5, 104.0])
        low = np.asarray([99.0, 100.0, 101.0, 101.5, 101.0, 102.0, 102.5])
        features = build_structural_features(
            timestamps, open_, high, low, close,
            np.full(7, 20.0), np.full(7, 80.0), np.full(7, 120.0), 1,
        )

        np.testing.assert_array_equal(
            features.session_label,
            ["asia", "asia", "london", "london", "overlap", "overlap", "new_york"],
        )
        self.assertEqual(features.prior_asia_high[2], 103.0)
        self.assertEqual(features.prior_asia_low[2], 99.0)
        self.assertEqual(features.prior_london_high[4], 105.0)
        self.assertEqual(features.prior_london_low[4], 101.0)
        self.assertEqual(features.session_code.tolist(), [0, 0, 1, 1, 2, 2, 3])

    def test_compression_and_expansion_transitions_are_causal_edges(self) -> None:
        inputs = sample_inputs(30)
        inputs["range_bps_20"][:] = 80.0
        inputs["range_bps_40"][:] = 100.0
        inputs["range_bps_20"][20:25] = 40.0
        inputs["atr_bps"][:] = 20.0
        inputs["open_"][25] = 100.0
        inputs["close"][25] = 101.0
        inputs["high"][25] = 102.0
        inputs["low"][25] = 98.0

        features = build_structural_features(**inputs)

        self.assertEqual(np.flatnonzero(features.compression_transition).tolist(), [20])
        self.assertEqual(np.flatnonzero(features.expansion_transition).tolist(), [25])
        self.assertIn("expansion", features.regime_interaction_label[25])


if __name__ == "__main__":
    unittest.main()
