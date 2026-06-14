"""Reporting-only robustness helpers for historical net PnL.

The public helpers use duck typing instead of importing backtest record classes,
so this module cannot create a circular dependency with a search engine.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime, timezone
import math
from typing import Any

import numpy as np


DEFAULT_MONTE_CARLO_SEED = 20260614


def monthly_performance(
    trades_or_timestamp_net: Iterable[object] | Mapping[object, float],
    diagnostic_notional: float,
    period_start: object | None = None,
    period_end: object | None = None,
) -> dict[str, Any]:
    """Report monthly expectancy and stability from timestamped net PnL.

    Observations may be trade-like objects, dictionaries, or ``(timestamp,
    net)`` pairs. Trade-like inputs may expose ``exit_timestamp`` and
    ``net_pnl``; generic rows may use ``timestamp`` and ``daily_net`` or
    ``net``. Numeric timestamps may be Unix seconds or milliseconds.
    """

    notional = _positive_notional(diagnostic_notional)
    observations = _timestamp_net_observations(trades_or_timestamp_net)
    if not observations:
        return _empty_monthly_report(notional)

    net_by_month: dict[str, float] = {}
    for timestamp, net_value in observations:
        month = _coerce_datetime(timestamp).strftime("%Y-%m")
        net_by_month[month] = net_by_month.get(month, 0.0) + net_value

    first_month = _coerce_datetime(period_start).strftime("%Y-%m") if period_start is not None else min(net_by_month)
    last_month = _coerce_datetime(period_end).strftime("%Y-%m") if period_end is not None else max(net_by_month)
    months = _inclusive_months(first_month, last_month)
    rows = [
        {
            "month": month,
            "net": float(net_by_month.get(month, 0.0)),
            "return_pct": float(net_by_month.get(month, 0.0) / notional * 100.0),
        }
        for month in months
    ]
    positive_months = sum(row["net"] > 0.0 for row in rows)
    positive_month_pct = positive_months / len(rows) * 100.0
    worst = min(rows, key=lambda row: row["net"])
    expectancy = float(np.mean([row["net"] for row in rows]))
    return_std = float(np.std([row["return_pct"] for row in rows], ddof=0))

    return {
        "status": "calculated",
        "basis": "calendar_month_net_pnl_over_diagnostic_notional",
        "diagnostic_notional": notional,
        "months": len(rows),
        "positive_months": positive_months,
        "positive_month_percentage": positive_month_pct,
        "positive_month_pct": positive_month_pct,
        "worst_month": worst["month"],
        "worst_month_net": worst["net"],
        "worst_month_return_pct": worst["return_pct"],
        "monthly_expectancy": expectancy,
        "monthly_expectancy_pct": expectancy / notional * 100.0,
        "monthly_return_std_pct": return_std,
        "monthly_results": rows,
    }


def block_bootstrap_monte_carlo(
    daily_net_values: Sequence[float] | Iterable[float],
    diagnostic_notional: float,
    iterations: int,
    block_size_days: int = 5,
    seed: int = DEFAULT_MONTE_CARLO_SEED,
) -> dict[str, Any]:
    """Bootstrap contiguous daily-net blocks into deterministic equity paths.

    Blocks are sampled with replacement and wrap around the source series, so
    every observation can begin a full block. Each path has the same number of
    days as the source data.
    """

    notional = _positive_notional(diagnostic_notional)
    if block_size_days <= 0:
        raise ValueError("block_size_days must be positive")

    values = _finite_values(daily_net_values)
    completed_iterations = max(int(iterations), 0)
    base = {
        "iterations": completed_iterations,
        "block_size_days": int(block_size_days),
        "horizon_days": len(values),
        "seed": int(seed),
        "diagnostic_notional": notional,
        "survival_probability": 0.0,
        "survival_probability_pct": 0.0,
        "positive_return_probability": 0.0,
        "positive_return_probability_pct": 0.0,
        "probability_positive_final_return": 0.0,
        "robust_path_probability": 0.0,
        "expected_max_drawdown_pct": 0.0,
        "p95_drawdown": 0.0,
        "p95_drawdown_pct": 0.0,
        "worst_case_drawdown_p95_pct": 0.0,
    }
    if not values or completed_iterations == 0:
        return {"status": "insufficient_data", **base}

    sample = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    drawdowns: list[float] = []
    survived = 0
    positive = 0
    robust_paths = 0
    for _ in range(completed_iterations):
        path = _sample_block_path(sample, block_size_days, rng)
        equity = notional
        peak = notional
        max_drawdown = 0.0
        path_survived = True
        for daily_net in path:
            equity += float(daily_net)
            peak = max(peak, equity)
            max_drawdown = max(max_drawdown, peak - equity)
            if equity <= 0.0:
                path_survived = False
        survived += int(path_survived)
        positive += int(equity > notional)
        robust_paths += int(equity > notional and max_drawdown / notional * 100.0 <= 10.0)
        drawdowns.append(max_drawdown / notional * 100.0)

    survival_probability = survived / completed_iterations * 100.0
    positive_probability = positive / completed_iterations * 100.0
    expected_drawdown = float(np.mean(drawdowns))
    p95_drawdown = float(np.percentile(drawdowns, 95))
    return {
        "status": "simulated",
        **base,
        "survival_probability": survival_probability,
        "survival_probability_pct": survival_probability,
        "positive_return_probability": positive_probability,
        "positive_return_probability_pct": positive_probability,
        "probability_positive_final_return": positive_probability,
        "robust_path_probability": robust_paths / completed_iterations * 100.0,
        "expected_max_drawdown_pct": expected_drawdown,
        "p95_drawdown": p95_drawdown,
        "p95_drawdown_pct": p95_drawdown,
        "worst_case_drawdown_p95_pct": p95_drawdown,
    }


def _empty_monthly_report(notional: float) -> dict[str, Any]:
    return {
        "status": "insufficient_data",
        "basis": "calendar_month_net_pnl_over_diagnostic_notional",
        "diagnostic_notional": notional,
        "months": 0,
        "positive_months": 0,
        "positive_month_percentage": 0.0,
        "positive_month_pct": 0.0,
        "worst_month": None,
        "worst_month_net": 0.0,
        "worst_month_return_pct": 0.0,
        "monthly_expectancy": 0.0,
        "monthly_expectancy_pct": 0.0,
        "monthly_return_std_pct": 0.0,
        "monthly_results": [],
    }


def _timestamp_net_observations(
    source: Iterable[object] | Mapping[object, float],
) -> list[tuple[object, float]]:
    if isinstance(source, Mapping) and not _looks_like_observation(source):
        raw_items: Iterable[object] = source.items()
    elif isinstance(source, Mapping):
        raw_items = (source,)
    else:
        raw_items = source
    return [_extract_timestamp_net(item) for item in raw_items]


def _extract_timestamp_net(item: object) -> tuple[object, float]:
    if isinstance(item, Mapping):
        timestamp = _mapping_value(item, "exit_timestamp", "timestamp", "date", "day")
        net_value = _mapping_value(item, "net_pnl", "daily_net", "net")
    elif isinstance(item, (tuple, list)) and len(item) == 2:
        timestamp, net_value = item
    else:
        timestamp = _attribute_value(item, "exit_timestamp", "timestamp", "date", "day")
        net_value = _attribute_value(item, "net_pnl", "daily_net", "net")
    return timestamp, _finite_float(net_value, "net values")


def _looks_like_observation(value: Mapping[object, object]) -> bool:
    return bool(set(value).intersection({"exit_timestamp", "timestamp", "date", "day"}))


def _mapping_value(mapping: Mapping[object, object], *keys: str) -> object:
    for key in keys:
        if key in mapping:
            return mapping[key]
    raise ValueError(f"observation is missing one of: {', '.join(keys)}")


def _attribute_value(value: object, *names: str) -> object:
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
    raise ValueError(f"observation is missing one of: {', '.join(names)}")


def _coerce_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        timestamp = _finite_float(value, "timestamps")
        if abs(timestamp) >= 100_000_000_000:
            timestamp /= 1000.0
        return datetime.fromtimestamp(timestamp, tz=timezone.utc)
    text = str(value).strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"invalid timestamp: {value!r}") from exc


def _inclusive_months(first: str, last: str) -> list[str]:
    year, month = (int(part) for part in first.split("-"))
    last_year, last_month = (int(part) for part in last.split("-"))
    result: list[str] = []
    while (year, month) <= (last_year, last_month):
        result.append(f"{year:04d}-{month:02d}")
        month += 1
        if month == 13:
            year += 1
            month = 1
    return result


def _sample_block_path(
    sample: np.ndarray,
    block_size_days: int,
    rng: np.random.Generator,
) -> np.ndarray:
    blocks_needed = math.ceil(sample.size / block_size_days)
    starts = rng.integers(0, sample.size, size=blocks_needed)
    offsets = np.arange(block_size_days)
    indices = (starts[:, np.newaxis] + offsets) % sample.size
    return sample[indices].reshape(-1)[: sample.size]


def _finite_values(values: Sequence[float] | Iterable[float]) -> list[float]:
    return [_finite_float(value, "daily net values") for value in values]


def _finite_float(value: object, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _positive_notional(value: float) -> float:
    notional = _finite_float(value, "diagnostic_notional")
    if notional <= 0.0:
        raise ValueError("diagnostic_notional must be positive")
    return notional
