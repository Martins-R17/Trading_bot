"""Compare saved realized sweep summary logs.

This tool is reporting-only. It reads JSONL records written by
backtesting.calibration --save-summary-log and prints a compact comparison
table. It does not run backtests, download data, or touch trading execution.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_LOG_PATH = Path("data/backtest_logs/realized_sweep_summary.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare realized sweep summary JSONL logs.")
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_LOG_PATH,
        help=f"Path to realized sweep summary JSONL log. Default: {DEFAULT_LOG_PATH}.",
    )
    parser.add_argument(
        "--last",
        type=int,
        help="Show only the last N records after sorting newest last.",
    )
    parser.add_argument(
        "--symbol-filter",
        default="BTC/USDT",
        help="Only show records whose compact summary symbols exactly match this symbol. Default: BTC/USDT.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_records = load_records(args.path)
    records = filter_records_by_symbol(all_records, args.symbol_filter)
    if args.last is not None:
        records = records[-max(args.last, 0) :]
    print_table(records, args.path, ignored_count=max(len(all_records) - len(filter_records_by_symbol(all_records, args.symbol_filter)), 0), symbol_filter=args.symbol_filter)


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                records.append(
                    {
                        "logged_at_utc": "",
                        "run_label": f"invalid_json_line_{line_number}",
                        "summary": {"verdict": f"json_error:{exc.msg}"},
                    }
                )
                continue
            if isinstance(record, dict):
                record["_line_number"] = line_number
                records.append(record)

    return sorted(records, key=record_sort_key)


def filter_records_by_symbol(records: list[dict[str, Any]], symbol: str) -> list[dict[str, Any]]:
    if not symbol:
        return records
    filtered: list[dict[str, Any]] = []
    for record in records:
        summary = safe_dict(record.get("summary"))
        symbols = format_list(summary.get("symbols")).split(",")
        if [item.strip() for item in symbols if item.strip() and item.strip() != "n/a"] == [symbol]:
            filtered.append(record)
    return filtered


def record_sort_key(record: dict[str, Any]) -> tuple[datetime, int]:
    timestamp = parse_timestamp(str(record.get("logged_at_utc", "")))
    line_number = int(record.get("_line_number", 0) or 0)
    return timestamp, line_number


def parse_timestamp(raw: str) -> datetime:
    if not raw:
        return datetime.min
    normalized = raw.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return datetime.min
    return parsed.replace(tzinfo=None)


def print_table(records: list[dict[str, Any]], path: Path, ignored_count: int = 0, symbol_filter: str = "") -> None:
    print(f"Realized Sweep Summary Log: {path}")
    if symbol_filter:
        print(f"symbol_filter={symbol_filter} ignored_non_matching_records={ignored_count}")
    if not records:
        print("No records found.")
        return

    columns = (
        ("RunLabel", 24),
        ("Timestamp", 19),
        ("Symbols", 10),
        ("Tf", 8),
        ("Profile", 12),
        ("Candles", 9),
        ("Window", 7),
        ("Notional", 10),
        ("CalibNet", 10),
        ("Agent", 16),
        ("Lev", 6),
        ("Liq", 5),
        ("TPD", 8),
        ("5-20/day", 14),
        ("MedDay%", 9),
        ("FeeDrag%", 9),
        ("100/day", 16),
        ("5%day", 16),
        ("WF", 24),
        ("SoftLate", 9),
        ("Total", 7),
        ("Pos", 5),
        ("Pos30", 7),
        ("FreqRows", 8),
        ("BestOverall", 48),
        ("Best30", 48),
        ("Best5-20/day", 48),
        ("Verdict", 34),
    )
    print(" ".join(f"{name:<{width}}" for name, width in columns))
    print(" ".join("-" * width for _, width in columns))
    for record in records:
        summary = safe_dict(record.get("summary"))
        values = (
            truncate(str(record.get("run_label") or "n/a"), 24),
            truncate(str(record.get("logged_at_utc") or "n/a"), 19),
            truncate(format_list(summary.get("symbols")), 10),
            truncate(format_list(summary.get("timeframes")), 8),
            truncate(str(summary.get("quality_profile") or summary.get("mode") or "n/a"), 12),
            str(summary.get("total_candles", "n/a")),
            str(summary.get("signal_window_bars", "n/a")),
            format_money(summary.get("diagnostic_notional")),
            format_money(summary.get("calibration_min_expected_net_profit")),
            truncate(str(summary.get("agent_name") or "n/a"), 16),
            format_number(summary.get("leverage_used")),
            str(summary.get("liquidation_events", "n/a")),
            format_number(summary.get("trades_per_day")),
            truncate(str(summary.get("verdict_5_to_20_trades_per_day") or "n/a"), 14),
            format_percent(summary.get("median_daily_return_pct")),
            format_percent(summary.get("fee_drag_pct")),
            truncate(str(summary.get("verdict_100_trades_per_day") or "n/a"), 16),
            truncate(str(summary.get("verdict_5pct_daily_target") or "n/a"), 16),
            truncate(str(summary.get("walk_forward_verdict") or "n/a"), 24),
            truncate(str(summary.get("reject_soft_late_momentum") or "n/a"), 9),
            str(summary.get("total_combinations", "n/a")),
            str(summary.get("positive_combinations", "n/a")),
            str(summary.get("positive_combinations_with_at_least_30_trades", "n/a")),
            str(summary.get("combinations_in_frequency_band", "n/a")),
            truncate(format_best(summary.get("best_overall")), 48),
            truncate(format_best(summary.get("best_at_least_30")), 48),
            truncate(format_best(summary.get("best_in_5_to_20_trades_per_day")), 48),
            truncate(str(summary.get("verdict") or "n/a"), 34),
        )
        print(" ".join(f"{value:<{width}}" for value, (_, width) in zip(values, columns)))


def safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def format_best(value: Any) -> str:
    row = safe_dict(value)
    if not row:
        return "n/a"
    symbol = row.get("symbol", "n/a")
    strategy = row.get("strategy", "n/a")
    trades = row.get("trades", "n/a")
    net = format_money(row.get("net"))
    pf = format_number(row.get("pf"))
    return f"{symbol} {strategy} trades={trades} net={net} pf={pf}"


def format_list(value: Any) -> str:
    if isinstance(value, list):
        return ",".join(str(item) for item in value) or "n/a"
    if isinstance(value, tuple):
        return ",".join(str(item) for item in value) or "n/a"
    if value:
        return str(value)
    return "n/a"


def format_money(value: Any) -> str:
    number = to_float(value)
    if number is None:
        return "n/a"
    return f"${number:.2f}"


def format_number(value: Any) -> str:
    number = to_float(value)
    if number is None:
        return "n/a"
    if number == float("inf"):
        return "inf"
    return f"{number:.2f}"


def format_percent(value: Any) -> str:
    number = to_float(value)
    if number is None:
        return "n/a"
    return f"{number:.2f}%"


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def truncate(value: str, width: int) -> str:
    if len(value) <= width:
        return value
    if width <= 1:
        return value[:width]
    return value[: width - 1] + "~"


if __name__ == "__main__":
    main()
