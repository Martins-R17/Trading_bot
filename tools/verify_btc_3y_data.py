"""Verify local BTCUSDT 3-year calibration CSV availability.

Reporting-only. This script reads local CSV metadata and does not download data,
place orders, or access private endpoints.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


EXPECTED = {
    "1m": ("data/historical_3y_1m/BTCUSDT_1m.csv", 1_576_800),
    "5m": ("data/historical_3y_5m/BTCUSDT_5m.csv", 315_360),
    "15m": ("data/historical_3y_15m/BTCUSDT_15m.csv", 105_120),
}


@dataclass(frozen=True)
class CsvProfile:
    timeframe: str
    path: Path
    exists: bool
    rows: int = 0
    expected_rows: int = 0
    first_timestamp: float | None = None
    last_timestamp: float | None = None

    @property
    def coverage_ratio(self) -> float:
        if self.expected_rows <= 0:
            return 0.0
        return self.rows / self.expected_rows

    @property
    def approx_days(self) -> float | None:
        if self.first_timestamp is None or self.last_timestamp is None:
            return None
        scale = 1000 if abs(self.first_timestamp) > 10_000_000_000 else 1
        return max(0.0, (self.last_timestamp - self.first_timestamp) / scale / 86_400)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify local BTCUSDT 3-year CSV data.")
    parser.add_argument(
        "--timeframes",
        default="15m,5m,1m",
        help="Comma-separated timeframes to verify. Default: 15m,5m,1m.",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.98,
        help="Minimum row-count coverage ratio considered OK. Default: 0.98.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timeframes = [item.strip() for item in args.timeframes.split(",") if item.strip()]
    profiles = [profile_timeframe(timeframe) for timeframe in timeframes]
    print("BTCUSDT 3-Year Data Verification")
    print("local files only; no downloads, no API keys, no trading")
    print(
        f"{'TF':<5} {'Status':<10} {'Rows':>10} {'Expected':>10} {'Coverage':>9} "
        f"{'Start':<25} {'End':<25} {'Days':>8} {'Path'}"
    )
    failed = False
    for profile in profiles:
        status = "missing"
        if profile.exists:
            status = "ok" if profile.coverage_ratio >= args.min_coverage else "short"
        if status != "ok":
            failed = True
        days = "n/a" if profile.approx_days is None else f"{profile.approx_days:.1f}"
        print(
            f"{profile.timeframe:<5} {status:<10} {profile.rows:>10} {profile.expected_rows:>10} "
            f"{profile.coverage_ratio:>8.1%} {format_timestamp(profile.first_timestamp):<25} "
            f"{format_timestamp(profile.last_timestamp):<25} {days:>8} {profile.path}"
        )
    if failed:
        raise SystemExit(1)


def profile_timeframe(timeframe: str) -> CsvProfile:
    if timeframe not in EXPECTED:
        raise SystemExit(f"Unsupported timeframe {timeframe!r}; expected one of {', '.join(EXPECTED)}")
    raw_path, expected_rows = EXPECTED[timeframe]
    path = Path(raw_path)
    if not path.exists():
        return CsvProfile(timeframe=timeframe, path=path, exists=False, expected_rows=expected_rows)
    rows, first_ts, last_ts = read_csv_profile(path)
    return CsvProfile(
        timeframe=timeframe,
        path=path,
        exists=True,
        rows=rows,
        expected_rows=expected_rows,
        first_timestamp=first_ts,
        last_timestamp=last_ts,
    )


def read_csv_profile(path: Path) -> tuple[int, float | None, float | None]:
    rows = 0
    first_ts: float | None = None
    last_ts: float | None = None
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            timestamp = parse_float(row.get("timestamp"))
            if timestamp is None:
                continue
            rows += 1
            if first_ts is None:
                first_ts = timestamp
            last_ts = timestamp
    return rows, first_ts, last_ts


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def format_timestamp(value: float | None) -> str:
    if value is None:
        return "n/a"
    seconds = value / 1000 if abs(value) > 10_000_000_000 else value
    return datetime.fromtimestamp(seconds, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


if __name__ == "__main__":
    main()
