"""Causal NumPy structural features for historical OHLC data.

Features at row ``i`` are evaluated after row ``i`` closes and never inspect
rows after ``i``. This module is backtesting-only and has no execution hooks.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import re
from typing import Sequence

import numpy as np


_ASIA = "asia"
_LONDON = "london"
_OVERLAP = "overlap"
_NEW_YORK = "new_york"
_OFF_HOURS = "off_hours"
_SESSION_LABELS = (_ASIA, _LONDON, _OVERLAP, _NEW_YORK, _OFF_HOURS)

_MINUTE_NS = 60 * 1_000_000_000
_DAY_NS = 24 * 60 * _MINUTE_NS
_NAT_INT = np.datetime64("NaT", "ns").astype(np.int64)


@dataclass(frozen=True, slots=True)
class StructuralFeatures:
    """Integration-ready arrays aligned one-for-one with source OHLC rows.

    ``htf_trend_direction`` is ``-1`` for down, ``0`` for unavailable/flat,
    and ``1`` for up. Session codes are Asia=0, London=1, overlap=2,
    New York=3, and off-hours=4.
    """

    htf_trend_bps: np.ndarray
    htf_trend_direction: np.ndarray
    session_label: np.ndarray
    session_code: np.ndarray
    prior_asia_high: np.ndarray
    prior_asia_low: np.ndarray
    prior_london_high: np.ndarray
    prior_london_low: np.ndarray
    compression_transition: np.ndarray
    expansion_transition: np.ndarray
    regime_interaction_label: np.ndarray


def build_structural_features(
    timestamps: Sequence[object] | np.ndarray,
    open_: Sequence[float] | np.ndarray,
    high: Sequence[float] | np.ndarray,
    low: Sequence[float] | np.ndarray,
    close: Sequence[float] | np.ndarray,
    atr_bps: Sequence[float] | np.ndarray,
    range_bps_20: Sequence[float] | np.ndarray,
    range_bps_40: Sequence[float] | np.ndarray,
    timeframe_minutes: int,
) -> StructuralFeatures:
    """Build same-length, causal structural features from lower-timeframe data.

    ``timeframe_minutes`` is the source candle size. Completed higher-timeframe
    context uses 15m for 1m data, 1h for up to 5m, 4h for up to 15m, 1d for
    up to 1h, and four source bars otherwise. Initial unavailable values are
    represented by ``NaN`` or neutral numeric codes.
    """

    if timeframe_minutes <= 0:
        raise ValueError("timeframe_minutes must be positive")

    timestamp_ns = _timestamps_to_ns(timestamps, timeframe_minutes)
    length = len(timestamp_ns)
    open_values, high_values, low_values, close_values = _validated_ohlc(
        open_, high, low, close, length
    )
    atr_values = _feature_array("atr_bps", atr_bps, length)
    range_20 = _feature_array("range_bps_20", range_bps_20, length)
    range_40 = _feature_array("range_bps_40", range_bps_40, length)

    htf_trend_bps = _completed_htf_trend_bps(
        timestamp_ns,
        open_values,
        close_values,
        _higher_timeframe_minutes(timeframe_minutes),
    )
    htf_direction = np.zeros(length, dtype=np.int8)
    htf_direction[htf_trend_bps > 0.0] = 1
    htf_direction[htf_trend_bps < 0.0] = -1

    labels, session_code = _session_labels_and_codes(timestamp_ns)
    prior_asia_high, prior_asia_low, prior_london_high, prior_london_low = (
        _prior_named_session_ranges(timestamp_ns, labels, high_values, low_values)
    )

    compression_ratio = _safe_ratio(range_20, range_40)
    is_compressed = np.isfinite(compression_ratio) & (compression_ratio <= 0.55)
    prior_compressed = np.r_[False, is_compressed[:-1]] if length else is_compressed
    compression_transition = is_compressed & ~prior_compressed

    expansion_window = min(20, max(3, int(round(60 / timeframe_minutes))))
    prior_atr = _prior_rolling_mean(atr_values, expansion_window)
    candle_range_bps = _safe_bps(high_values - low_values, close_values)
    expansion_ratio = _safe_ratio(candle_range_bps, prior_atr)
    prior_high = _shifted_rolling_extreme(high_values, 20, maximum=True)
    prior_low = _shifted_rolling_extreme(low_values, 20, maximum=False)
    breakout = (high_values > prior_high) | (low_values < prior_low)
    is_expanding = breakout & np.isfinite(expansion_ratio) & (expansion_ratio >= 1.5)
    expansion_transition = prior_compressed & is_expanding

    trend_component = np.full(length, "flat", dtype="<U7")
    trend_component[htf_direction > 0] = "up"
    trend_component[htf_direction < 0] = "down"
    trend_component[~np.isfinite(htf_trend_bps)] = "unknown"
    structure_component = np.full(length, "normal", dtype="<U11")
    structure_component[is_compressed] = "compression"
    structure_component[expansion_transition] = "expansion"
    interactions = _interaction_labels(trend_component, structure_component, labels)

    return StructuralFeatures(
        htf_trend_bps=htf_trend_bps,
        htf_trend_direction=htf_direction,
        session_label=labels,
        session_code=session_code,
        prior_asia_high=prior_asia_high,
        prior_asia_low=prior_asia_low,
        prior_london_high=prior_london_high,
        prior_london_low=prior_london_low,
        compression_transition=compression_transition,
        expansion_transition=expansion_transition,
        regime_interaction_label=interactions,
    )


def _completed_htf_trend_bps(
    timestamp_ns: np.ndarray,
    open_values: np.ndarray,
    close_values: np.ndarray,
    timeframe_minutes: int,
) -> np.ndarray:
    """Align each row to the latest completed UTC higher-timeframe candle."""

    length = len(timestamp_ns)
    output = np.full(length, np.nan)
    if length == 0:
        return output

    width_ns = timeframe_minutes * _MINUTE_NS
    bucket_ids = np.floor_divide(timestamp_ns, width_ns)
    starts = np.r_[0, np.flatnonzero(np.diff(bucket_ids)) + 1]
    ends = np.r_[starts[1:], length]
    group_open = open_values[starts]
    group_close = close_values[ends - 1]
    group_trend = _safe_bps(group_close - group_open, group_open)
    row_groups = np.repeat(np.arange(len(starts)), ends - starts)
    completed_groups = row_groups - 1
    available = completed_groups >= 0
    output[available] = group_trend[completed_groups[available]]
    return output


def _session_labels_and_codes(timestamp_ns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    minute = np.floor_divide(np.mod(timestamp_ns, _DAY_NS), _MINUTE_NS)
    labels = np.full(len(timestamp_ns), _OFF_HOURS, dtype="<U9")
    labels[minute < 7 * 60] = _ASIA
    labels[(minute >= 7 * 60) & (minute < 12 * 60)] = _LONDON
    labels[(minute >= 12 * 60) & (minute < 16 * 60)] = _OVERLAP
    labels[(minute >= 16 * 60) & (minute < 21 * 60)] = _NEW_YORK
    codes = np.full(len(timestamp_ns), 4, dtype=np.int8)
    for code, label in enumerate(_SESSION_LABELS):
        codes[labels == label] = code
    return labels, codes


def _prior_named_session_ranges(
    timestamp_ns: np.ndarray,
    labels: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Carry the latest completed Asia and London ranges forward causally."""

    length = len(timestamp_ns)
    outputs = tuple(np.full(length, np.nan) for _ in range(4))
    asia_high, asia_low, london_high, london_low = outputs
    completed = {_ASIA: [np.nan, np.nan], _LONDON: [np.nan, np.nan]}
    current_key: tuple[int, str] | None = None
    current_high = np.nan
    current_low = np.nan

    for index in range(length):
        key = (int(timestamp_ns[index] // _DAY_NS), str(labels[index]))
        if key != current_key:
            if current_key is not None and current_key[1] in completed:
                completed[current_key[1]] = [current_high, current_low]
            current_key = key
            current_high = high[index]
            current_low = low[index]
        else:
            current_high = max(current_high, high[index])
            current_low = min(current_low, low[index])
        asia_high[index], asia_low[index] = completed[_ASIA]
        london_high[index], london_low[index] = completed[_LONDON]
    return outputs


def _shifted_rolling_extreme(
    values: np.ndarray,
    window: int,
    *,
    maximum: bool,
) -> np.ndarray:
    """Return the prior full-window extreme at every row."""

    rolling = np.full(len(values), np.nan)
    candidates: deque[int] = deque()
    for index, value in enumerate(values):
        while candidates and candidates[0] <= index - window:
            candidates.popleft()
        while candidates and (
            value >= values[candidates[-1]] if maximum else value <= values[candidates[-1]]
        ):
            candidates.pop()
        candidates.append(index)
        if index >= window - 1:
            rolling[index] = values[candidates[0]]
    shifted = np.full(len(values), np.nan)
    if len(values) > 1:
        shifted[1:] = rolling[:-1]
    return shifted


def _prior_rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    output = np.full(len(values), np.nan)
    if len(values) <= window:
        return output
    finite = np.isfinite(values)
    cumulative = np.r_[0.0, np.cumsum(np.where(finite, values, 0.0))]
    counts = np.r_[0, np.cumsum(finite)]
    indices = np.arange(window, len(values))
    valid = counts[indices] - counts[indices - window] == window
    valid_indices = indices[valid]
    output[valid_indices] = (
        cumulative[valid_indices] - cumulative[valid_indices - window]
    ) / window
    return output


def _higher_timeframe_minutes(source_minutes: int) -> int:
    if source_minutes <= 1:
        return 15
    if source_minutes <= 5:
        return 60
    if source_minutes <= 15:
        return 240
    if source_minutes <= 60:
        return 1_440
    return source_minutes * 4


def _timestamps_to_ns(
    timestamps: Sequence[object] | np.ndarray,
    expected_interval_minutes: int,
) -> np.ndarray:
    values = np.asarray(timestamps)
    if values.ndim != 1:
        raise ValueError("timestamps must be one-dimensional")
    if np.issubdtype(values.dtype, np.datetime64):
        timestamp_ns = values.astype("datetime64[ns]").astype(np.int64)
        if np.any(timestamp_ns == _NAT_INT):
            raise ValueError("timestamps must not contain NaT")
    else:
        try:
            numeric = values.astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError("timestamps must be numeric epochs or numpy datetimes") from exc
        if np.any(~np.isfinite(numeric)):
            raise ValueError("timestamps must be finite")
        max_abs = float(np.max(np.abs(numeric))) if len(numeric) else 0.0
        multiplier = 1_000_000_000
        if max_abs >= 1e17:
            multiplier = 1
        elif max_abs >= 1e14:
            multiplier = 1_000
        elif max_abs >= 1e11:
            multiplier = 1_000_000
        elif len(numeric) > 1:
            positive_deltas = np.diff(numeric)
            positive_deltas = positive_deltas[positive_deltas > 0.0]
            if len(positive_deltas):
                smallest_delta = float(np.min(positive_deltas))
                expected_ns = expected_interval_minutes * _MINUTE_NS
                candidates = np.asarray([1_000_000_000, 1_000_000, 1_000, 1])
                errors = np.abs(
                    np.log(np.maximum(smallest_delta * candidates / expected_ns, 1e-300))
                )
                multiplier = int(candidates[int(np.argmin(errors))])
        timestamp_ns = np.rint(numeric * multiplier).astype(np.int64)
    if len(timestamp_ns) > 1 and np.any(np.diff(timestamp_ns) < 0):
        raise ValueError("timestamps must be sorted in ascending order")
    return timestamp_ns


def _validated_ohlc(
    open_: Sequence[float] | np.ndarray,
    high: Sequence[float] | np.ndarray,
    low: Sequence[float] | np.ndarray,
    close: Sequence[float] | np.ndarray,
    length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arrays = tuple(
        _finite_array(name, values, length)
        for name, values in (
            ("open", open_),
            ("high", high),
            ("low", low),
            ("close", close),
        )
    )
    open_values, high_values, low_values, close_values = arrays
    if np.any(high_values < low_values):
        raise ValueError("high must be greater than or equal to low")
    if np.any(high_values < np.maximum(open_values, close_values)) or np.any(
        low_values > np.minimum(open_values, close_values)
    ):
        raise ValueError("OHLC values are inconsistent")
    return arrays


def _finite_array(
    name: str,
    values: Sequence[float] | np.ndarray,
    length: int,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) != length:
        raise ValueError(f"{name} must be one-dimensional with length {length}")
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array.copy()


def _feature_array(
    name: str,
    values: Sequence[float] | np.ndarray,
    length: int,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) != length:
        raise ValueError(f"{name} must be one-dimensional with length {length}")
    return array.copy()


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    shape = np.broadcast_shapes(numerator.shape, denominator.shape)
    output = np.full(shape, np.nan)
    np.divide(numerator, denominator, out=output, where=denominator != 0.0)
    return output


def _safe_bps(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return _safe_ratio(numerator, denominator) * 10_000.0


def _interaction_labels(*components: np.ndarray) -> np.ndarray:
    """Create normalized labels whose values do not depend on category order."""

    return np.asarray(
        [
            "|".join(_normalize_label(component[index]) for component in components)
            for index in range(len(components[0]))
        ],
        dtype=str,
    )


def _normalize_label(value: object) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    return normalized or "unknown"


__all__ = ["StructuralFeatures", "build_structural_features"]
