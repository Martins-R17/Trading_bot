"""Download public Binance klines for calibration CSVs.

This tool uses only Binance public market-data endpoints. It does not require
API keys, account access, private endpoints, or trading permissions.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


PUBLIC_BINANCE_BASE_URL = "https://api.binance.com"
KLINES_PATH = "/api/v3/klines"
MAX_KLINES_PER_REQUEST = 1000
ALLOWED_CALIBRATION_INTERVALS = ("1m", "5m", "15m")
DEFAULT_BACKTEST_DAYS = 365 * 3
INTERVAL_ALIASES = {
    "1min": "1m",
    "5min": "5m",
    "15min": "15m",
}
INTERVAL_MS = {
    "1m": 60_000,
    "3m": 3 * 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "30m": 30 * 60_000,
    "1h": 60 * 60_000,
    "2h": 2 * 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "6h": 6 * 60 * 60_000,
    "8h": 8 * 60 * 60_000,
    "12h": 12 * 60 * 60_000,
    "1d": 24 * 60 * 60_000,
    "3d": 3 * 24 * 60 * 60_000,
    "1w": 7 * 24 * 60 * 60_000,
    "1M": 30 * 24 * 60 * 60_000,
}


@dataclass(slots=True)
class DownloadResult:
    symbol: str
    exchange_symbol: str
    interval: str
    path: Path
    rows: int
    first_timestamp: int | None
    last_timestamp: int | None


class PublicKlineDownloadError(RuntimeError):
    """Raised when public kline download fails in a user-actionable way."""


def normalize_symbol(symbol: str) -> str:
    cleaned = symbol.strip().upper().replace("-", "/")
    if not cleaned:
        raise ValueError("empty symbol")
    if "/" in cleaned:
        base, quote = cleaned.split("/", 1)
        if not base or not quote:
            raise ValueError(f"invalid symbol {symbol!r}")
        return f"{base}{quote}"
    return cleaned


def parse_symbols(raw: str) -> tuple[str, ...]:
    symbols = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not symbols:
        raise ValueError("at least one symbol is required")
    return symbols


def parse_intervals(raw: str) -> tuple[str, ...]:
    intervals = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not intervals:
        raise ValueError("at least one interval is required")
    unique: list[str] = []
    for interval in intervals:
        validated = validate_interval(interval)
        if validated not in unique:
            unique.append(validated)
    return tuple(unique)


def validate_interval(interval: str) -> str:
    interval = INTERVAL_ALIASES.get(interval.strip(), interval.strip())
    if interval not in ALLOWED_CALIBRATION_INTERVALS:
        valid = ", ".join(ALLOWED_CALIBRATION_INTERVALS)
        raise ValueError(
            f"invalid calibration interval {interval!r}; allowed intervals: {valid}"
        )
    if interval not in INTERVAL_MS:
        valid = ", ".join(INTERVAL_MS)
        raise ValueError(f"unsupported Binance interval {interval!r}; valid intervals: {valid}")
    return interval


def download_symbol(
    symbol: str,
    interval: str,
    days: float,
    output_dir: Path,
    base_url: str = PUBLIC_BINANCE_BASE_URL,
    request_sleep_seconds: float = 0.25,
    timeout_seconds: float = 20.0,
) -> DownloadResult:
    exchange_symbol = normalize_symbol(symbol)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{exchange_symbol}_{interval}.csv"

    end_ms = int(time.time() * 1000)
    start_ms = end_ms - int(days * 24 * 60 * 60 * 1000)
    rows = fetch_klines(
        exchange_symbol=exchange_symbol,
        interval=interval,
        start_ms=start_ms,
        end_ms=end_ms,
        base_url=base_url,
        request_sleep_seconds=request_sleep_seconds,
        timeout_seconds=timeout_seconds,
    )
    if not rows:
        raise PublicKlineDownloadError(
            f"empty response for {symbol} {interval}; no CSV written"
        )

    write_csv(output_path, rows)
    return DownloadResult(
        symbol=symbol,
        exchange_symbol=exchange_symbol,
        interval=interval,
        path=output_path,
        rows=len(rows),
        first_timestamp=int(rows[0][0]) if rows else None,
        last_timestamp=int(rows[-1][0]) if rows else None,
    )


def fetch_klines(
    exchange_symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    base_url: str,
    request_sleep_seconds: float,
    timeout_seconds: float,
) -> list[list[Any]]:
    rows: list[list[Any]] = []
    next_start = start_ms
    interval_ms = INTERVAL_MS[interval]

    while next_start < end_ms:
        batch = request_klines(
            base_url=base_url,
            exchange_symbol=exchange_symbol,
            interval=interval,
            start_ms=next_start,
            end_ms=end_ms,
            timeout_seconds=timeout_seconds,
        )
        if not batch:
            break

        rows.extend(batch)
        last_open_time = int(batch[-1][0])
        next_start = last_open_time + interval_ms
        if len(batch) < MAX_KLINES_PER_REQUEST:
            break
        time.sleep(max(request_sleep_seconds, 0.0))

    deduped: dict[int, list[Any]] = {}
    for row in rows:
        deduped[int(row[0])] = row
    return [deduped[key] for key in sorted(deduped)]


def request_klines(
    base_url: str,
    exchange_symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    timeout_seconds: float,
) -> list[list[Any]]:
    params = urlencode(
        {
            "symbol": exchange_symbol,
            "interval": interval,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": MAX_KLINES_PER_REQUEST,
        }
    )
    url = f"{base_url.rstrip('/')}{KLINES_PATH}?{params}"
    request = Request(url, headers={"User-Agent": "TradingBotCalibration/1.0"})
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            payload = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = read_error_detail(exc)
        raise PublicKlineDownloadError(
            f"Binance public kline request failed for {exchange_symbol}: "
            f"HTTP {exc.code} {detail}"
        ) from exc
    except URLError as exc:
        raise PublicKlineDownloadError(
            f"network error downloading {exchange_symbol}: {exc.reason}"
        ) from exc
    except TimeoutError as exc:
        raise PublicKlineDownloadError(
            f"network timeout downloading {exchange_symbol}"
        ) from exc

    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise PublicKlineDownloadError(
            f"invalid JSON response for {exchange_symbol}"
        ) from exc

    if isinstance(data, dict) and "code" in data:
        message = data.get("msg", "unknown Binance error")
        raise PublicKlineDownloadError(
            f"Binance rejected {exchange_symbol} {interval}: {message}"
        )
    if not isinstance(data, list):
        raise PublicKlineDownloadError(
            f"unexpected response for {exchange_symbol}: {type(data).__name__}"
        )
    return data


def read_error_detail(exc: HTTPError) -> str:
    try:
        payload = exc.read().decode("utf-8")
        data = json.loads(payload)
    except Exception:
        return ""
    if isinstance(data, dict):
        code = data.get("code", "")
        message = data.get("msg", "")
        return f"code={code} msg={message}"
    return ""


def write_csv(path: Path, rows: list[list[Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "candle_time", "open", "high", "low", "close", "volume"])
        for row in rows:
            open_time_ms = int(row[0])
            writer.writerow(
                [
                    open_time_ms,
                    format_utc(open_time_ms),
                    row[1],
                    row[2],
                    row[3],
                    row[4],
                    row[5],
                ]
            )


def format_utc(timestamp_ms: int | None) -> str:
    if timestamp_ms is None:
        return "-"
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).isoformat()


def calibration_limit_for_days(interval: str, days: float) -> int:
    interval_ms = INTERVAL_MS[validate_interval(interval)]
    return max(1, int(days * 24 * 60 * 60 * 1000 / interval_ms))


def default_output_dir(interval: str, days: float) -> Path:
    years = round(days / 365)
    if years == 3 and abs(days - DEFAULT_BACKTEST_DAYS) < 0.001:
        return Path(f"data/historical_3y_{interval}")
    safe_days = int(round(days))
    return Path(f"data/historical_{safe_days}d_{interval}")


def output_dir_for_interval(base_output_dir: Path | None, interval: str, days: float) -> Path:
    return base_output_dir if base_output_dir is not None else default_output_dir(interval, days)


def calibration_command(
    output_dir: Path,
    symbols: tuple[str, ...],
    interval: str,
    days: float,
    limit: int | None,
) -> str:
    command = (
        "python -m backtesting.calibration "
        f"--data-dir {output_dir} "
        f"--symbols {','.join(symbols)} "
        f"--timeframe {interval} "
        f"--years {days / 365:.2f}"
    )
    if limit is not None:
        command += f" --limit {limit}"
    return command


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download public Binance candles for calibration CSVs only."
    )
    parser.add_argument(
        "--symbols",
        required=True,
        help="Comma-separated symbols, e.g. BTC/USDT,ETH/USDT.",
    )
    parser.add_argument(
        "--interval",
        help="Single calibration interval. Allowed: 1m,5m,15m. Overrides --intervals when provided.",
    )
    parser.add_argument(
        "--intervals",
        default=",".join(ALLOWED_CALIBRATION_INTERVALS),
        help="Comma-separated calibration intervals. Allowed/default: 1m,5m,15m.",
    )
    parser.add_argument(
        "--days",
        type=float,
        default=float(DEFAULT_BACKTEST_DAYS),
        help="Number of recent days of public candles to download. Default: 1095 (~3 years).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Directory for downloaded calibration CSVs. If omitted, uses clear "
            "per-timeframe directories such as data/historical_3y_15m."
        ),
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.25,
        help="Small sleep between paged public requests.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="Network timeout in seconds.",
    )
    parser.add_argument(
        "--calibration-limit",
        type=int,
        help=(
            "Optional limit value shown in printed calibration commands. Omit for full "
            "downloaded history; --limit intentionally truncates calibration input."
        ),
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    symbols = parse_symbols(args.symbols)
    intervals = (validate_interval(args.interval),) if args.interval else parse_intervals(args.intervals)
    if args.days <= 0:
        raise SystemExit("--days must be greater than 0")

    print("Public Binance kline downloader")
    print("calibration only")
    print("no API key, no account access, no private endpoints, no trading")
    print(f"days={args.days:g} intervals={','.join(intervals)}")
    if "1m" in intervals and args.days >= DEFAULT_BACKTEST_DAYS:
        print("warning=1m 3-year data is large and slow; test 15m first, then 5m, then 1m")
    print("note=printed calibration commands omit --limit by default so full downloaded history is used")

    results: list[DownloadResult] = []
    failures: list[str] = []
    for interval in intervals:
        output_dir = output_dir_for_interval(args.output_dir, interval, args.days)
        for symbol in symbols:
            try:
                result = download_symbol(
                    symbol=symbol,
                    interval=interval,
                    days=args.days,
                    output_dir=output_dir,
                    request_sleep_seconds=args.sleep,
                    timeout_seconds=args.timeout,
                )
            except (ValueError, PublicKlineDownloadError) as exc:
                failures.append(f"{symbol} {interval}: {exc}")
                continue

            results.append(result)
            print(
                f"saved {result.exchange_symbol} {result.interval} rows={result.rows} "
                f"first={format_utc(result.first_timestamp)} "
                f"last={format_utc(result.last_timestamp)} path={result.path}"
            )

    if failures:
        print("Download errors")
        for failure in failures:
            print(failure)

    if results:
        print("Calibration commands")
        for interval in intervals:
            output_dir = output_dir_for_interval(args.output_dir, interval, args.days)
            print(calibration_command(output_dir, symbols, interval, args.days, args.calibration_limit))

    if failures and not results:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
