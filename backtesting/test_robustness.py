"""Tests for reporting-only robustness helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import unittest

from backtesting.robustness import block_bootstrap_monte_carlo, monthly_performance


@dataclass
class TradeLike:
    exit_timestamp: float
    net_pnl: float


class MonthlyPerformanceTests(unittest.TestCase):
    def test_accepts_trade_objects_and_reports_monthly_expectancy(self) -> None:
        january = datetime(2026, 1, 3, tzinfo=timezone.utc).timestamp() * 1000
        march = datetime(2026, 3, 2, tzinfo=timezone.utc).timestamp() * 1000
        summary = monthly_performance(
            [TradeLike(january, 8.0), TradeLike(january + 86_400_000, 2.0), TradeLike(march, -5.0)],
            100.0,
        )

        self.assertEqual(summary["months"], 3)
        self.assertEqual(summary["positive_months"], 1)
        self.assertAlmostEqual(summary["positive_month_percentage"], 100 / 3)
        self.assertEqual(summary["worst_month"], "2026-03")
        self.assertEqual(summary["worst_month_net"], -5.0)
        self.assertEqual(summary["worst_month_return_pct"], -5.0)
        self.assertAlmostEqual(summary["monthly_expectancy"], 5 / 3)
        self.assertEqual(summary["monthly_results"][1]["net"], 0.0)

    def test_accepts_timestamp_net_pairs_and_mapping_rows(self) -> None:
        pairs = monthly_performance([("2026-01-01", 4.0), ("2026-02-01", -1.0)], 200.0)
        row = monthly_performance({"timestamp": "2026-01-01", "daily_net": 4.0}, 200.0)

        self.assertEqual(pairs["monthly_expectancy_pct"], 0.75)
        self.assertEqual(row["monthly_expectancy_pct"], 2.0)

    def test_empty_report_and_payload_are_json_safe(self) -> None:
        summary = monthly_performance([], 100.0)

        self.assertEqual(summary["status"], "insufficient_data")
        self.assertIsNone(summary["worst_month"])
        json.dumps(summary, allow_nan=False)


class BlockBootstrapMonteCarloTests(unittest.TestCase):
    def test_all_positive_daily_net_has_no_drawdown(self) -> None:
        summary = block_bootstrap_monte_carlo([1.0, 2.0, 0.5], 100.0, 250, 2, seed=7)

        self.assertEqual(summary["status"], "simulated")
        self.assertEqual(summary["survival_probability"], 100.0)
        self.assertEqual(summary["probability_positive_final_return"], 100.0)
        self.assertEqual(summary["p95_drawdown_pct"], 0.0)

    def test_seed_makes_bootstrap_report_deterministic(self) -> None:
        first = block_bootstrap_monte_carlo([3.0, -2.0, 1.0, -4.0, 2.0], 100.0, 500, 3, 1234)
        second = block_bootstrap_monte_carlo([3.0, -2.0, 1.0, -4.0, 2.0], 100.0, 500, 3, 1234)

        self.assertEqual(first, second)
        self.assertEqual(first["seed"], 1234)
        self.assertGreater(first["p95_drawdown_pct"], 0.0)
        self.assertEqual(first["p95_drawdown"], first["p95_drawdown_pct"])
        json.dumps(first, allow_nan=False)

    def test_equity_at_or_below_zero_does_not_survive(self) -> None:
        summary = block_bootstrap_monte_carlo([-100.0], 100.0, 25, 1, seed=99)

        self.assertEqual(summary["survival_probability"], 0.0)
        self.assertEqual(summary["positive_return_probability"], 0.0)
        self.assertEqual(summary["worst_case_drawdown_p95_pct"], 100.0)

    def test_empty_values_return_stable_schema(self) -> None:
        summary = block_bootstrap_monte_carlo([], 100.0, 100, seed=2)

        self.assertEqual(summary["status"], "insufficient_data")
        self.assertEqual(summary["iterations"], 100)
        self.assertEqual(summary["horizon_days"], 0)

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "diagnostic_notional"):
            block_bootstrap_monte_carlo([1.0], 0.0, 10)
        with self.assertRaisesRegex(ValueError, "block_size_days"):
            block_bootstrap_monte_carlo([1.0], 100.0, 10, 0)
        with self.assertRaisesRegex(ValueError, "finite"):
            block_bootstrap_monte_carlo([float("nan")], 100.0, 10)


if __name__ == "__main__":
    unittest.main()
