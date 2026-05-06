"""Fast BTCUSDT futures scalping research search.

This module is backtesting-only. It does not talk to exchanges, place orders,
read private account state, or change live/paper execution. It scans local
historical OHLCV CSVs with precomputed numpy arrays and writes compact summary
records for the static dashboard.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data.preprocess import DataPreprocessor


DEFAULT_SYMBOL = "BTC/USDT"
DEFAULT_SUMMARY_LOG_PATH = Path("data/backtest_logs/realized_sweep_summary.jsonl")
DEFAULT_DATA_DIR_TEMPLATE = "data/historical_3y_{timeframe}"
TIMEFRAME_ORDER = ("1m", "5m", "15m")
TIMEFRAME_MINUTES = {"1m": 1, "5m": 5, "15m": 15}
DEFAULT_FUTURES_MAKER_FEE_RATE = 0.0002
DEFAULT_FUTURES_TAKER_FEE_RATE = 0.0005
DEFAULT_SLIPPAGE_BPS = 2.0
DEFAULT_MAINTENANCE_MARGIN_RATE = 0.004
TARGET_TRADES_PER_DAY = 100.0
TARGET_AVG_DAILY_RETURN_PCT = 5.0
TARGET_DAYS_ABOVE_5PCT_PCT = 75.0


@dataclass(frozen=True)
class SearchSpec:
    agent_name: str
    strategy_name: str
    side: str
    lookback: int
    target_bps: float
    stop_bps: float
    max_hold: int
    min_return_bps: float = 0.0
    min_volume_ratio: float = 1.0
    min_atr_bps: float = 0.0
    min_ema_gap_bps: float = 0.0
    min_close_position: float = 0.0
    max_close_position: float = 1.0
    min_rsi: float = 0.0
    max_rsi: float = 100.0
    min_macd_bps: float = 0.0
    min_range_bps: float = 0.0
    vwap_side_required: bool = False
    min_spacing: int = 1


@dataclass
class SearchTrade:
    timestamp: float
    entry_index: int
    exit_index: int
    side: str
    entry_price: float
    exit_price: float
    exit_reason: str
    hold_candles: int
    gross_pnl: float
    fees: float
    slippage_costs: float
    total_costs: float
    net_pnl: float
    liquidation_event: bool = False


@dataclass
class SearchRow:
    symbol: str
    timeframe: str
    agent_name: str
    strategy: str
    side: str
    target_bps: float
    stop_bps: float
    max_hold: int
    parameter_set: str
    candles_tested: int
    signals_considered: int
    trades: int = 0
    wins: int = 0
    losses: int = 0
    gross: float = 0.0
    costs: float = 0.0
    net: float = 0.0
    avg_net: float = 0.0
    pf: float | None = None
    win_rate: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    trades_per_day: float = 0.0
    avg_daily_return_pct: float = 0.0
    median_daily_return_pct: float = 0.0
    best_daily_return_pct: float = 0.0
    worst_daily_return_pct: float = 0.0
    days_profitable_pct: float = 0.0
    days_above_5pct: int = 0
    days_above_5pct_pct: float = 0.0
    max_daily_drawdown_pct: float = 0.0
    fee_drag_pct: float = 0.0
    liquidation_events: int = 0
    liquidation_risk_flag: bool = False
    leverage_used: float = 1.0
    walk_forward: list[dict[str, Any]] = field(default_factory=list)
    overfit_warning: bool = True
    verdict: str = "too_few_trades"
    failure_reasons: list[str] = field(default_factory=list)
    exit_reason_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class FeatureArrays:
    timestamp: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    ema_fast: np.ndarray
    ema_slow: np.ndarray
    macd_hist: np.ndarray
    rsi: np.ndarray
    atr: np.ndarray
    atr_bps: np.ndarray
    volume_ratio: np.ndarray
    close_position: np.ndarray
    vwap: np.ndarray
    return_3_bps: np.ndarray
    return_5_bps: np.ndarray
    return_10_bps: np.ndarray
    trend_20_bps: np.ndarray
    range_high_20: np.ndarray
    range_low_20: np.ndarray
    range_high_40: np.ndarray
    range_low_40: np.ndarray
    range_bps_20: np.ndarray
    range_bps_40: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast BTCUSDT futures scalping research search.")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL, help="Backtest symbol. Default: BTC/USDT.")
    parser.add_argument("--timeframe", choices=TIMEFRAME_ORDER, default="1m")
    parser.add_argument("--data-dir", type=Path, help="Historical CSV folder. Default: data/historical_3y_<timeframe>.")
    parser.add_argument("--csv", type=Path, help="Optional explicit BTCUSDT CSV path.")
    parser.add_argument("--limit", type=int, help="Optional most-recent candle limit for feasibility tests.")
    parser.add_argument("--diagnostic-notional", type=float, default=100.0)
    parser.add_argument("--simulated-leverage", type=float, default=1.0)
    parser.add_argument("--futures-maker-fee-rate", type=float, default=DEFAULT_FUTURES_MAKER_FEE_RATE)
    parser.add_argument("--futures-taker-fee-rate", type=float, default=DEFAULT_FUTURES_TAKER_FEE_RATE)
    parser.add_argument("--slippage-bps", type=float, default=DEFAULT_SLIPPAGE_BPS)
    parser.add_argument("--maintenance-margin-rate", type=float, default=DEFAULT_MAINTENANCE_MARGIN_RATE)
    parser.add_argument("--max-parameter-sets", type=int, default=0, help="0 means all default internal-agent specs.")
    parser.add_argument("--save-summary-log", action="store_true")
    parser.add_argument("--summary-log-path", type=Path, default=DEFAULT_SUMMARY_LOG_PATH)
    parser.add_argument("--run-label", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.symbol != DEFAULT_SYMBOL:
        raise SystemExit("This research mode is BTCUSDT-only. Use --symbol BTC/USDT.")
    if args.simulated_leverage < 1.0:
        raise SystemExit("--simulated-leverage must be >= 1.0")
    result = run_search(args)
    print_report(result)
    if args.save_summary_log:
        append_summary(result, args.summary_log_path, args.run_label)


def run_search(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    path = args.csv or find_csv(args.data_dir or Path(DEFAULT_DATA_DIR_TEMPLATE.format(timeframe=args.timeframe)), args.symbol, args.timeframe)
    raw_df = load_csv(path, args.limit)
    feature_started = time.perf_counter()
    df = DataPreprocessor.add_features(DataPreprocessor.normalize_ohlcv(raw_df))
    arrays = build_feature_arrays(df)
    feature_seconds = max(time.perf_counter() - feature_started, 0.0)
    specs = default_specs(args.timeframe)
    if args.max_parameter_sets and args.max_parameter_sets > 0:
        specs = specs[: args.max_parameter_sets]

    rows: list[SearchRow] = []
    total_candles_scanned = 0
    for spec in specs:
        row = evaluate_spec(
            spec=spec,
            arrays=arrays,
            symbol=args.symbol,
            timeframe=args.timeframe,
            diagnostic_notional=args.diagnostic_notional,
            leverage=args.simulated_leverage,
            taker_fee_rate=args.futures_taker_fee_rate,
            slippage_bps=args.slippage_bps,
            maintenance_margin_rate=args.maintenance_margin_rate,
        )
        rows.append(row)
        total_candles_scanned += row.candles_tested

    elapsed = max(time.perf_counter() - started, 0.0)
    best_overall = best_row(rows, min_trades=1)
    best_30 = best_row(rows, min_trades=30)
    worst = worst_row(rows)
    primary = best_30 or best_overall
    agent_comparison = build_agent_comparison(rows)
    strategy_leaderboard = [row_to_dict(row) for row in sorted(rows, key=leaderboard_key, reverse=True)[:20]]
    data_profile = build_data_profile(args.symbol, args.timeframe, arrays.timestamp)
    target_verdicts = evaluate_targets(primary)
    summary = {
        "summary_version": 5,
        "mode": "fast_futures_scalping_search",
        "symbols": [args.symbol],
        "timeframes": [args.timeframe],
        "timeframe": args.timeframe,
        "data_profiles": [data_profile],
        "total_candles": int(len(arrays.close)),
        "candle_count": int(len(arrays.close)),
        "data_period_start": data_profile["start_utc"],
        "data_period_end": data_profile["end_utc"],
        "approx_days": data_profile["approx_days"],
        "data_years": data_profile["data_years"],
        "btc_only": True,
        "contract_type": "BTCUSDT USDT-M futures simulation",
        "diagnostic_notional": args.diagnostic_notional,
        "leverage_used": args.simulated_leverage,
        "liquidation_risk_flag": bool(args.simulated_leverage > 1.0),
        "futures_maker_fee_rate": args.futures_maker_fee_rate,
        "futures_taker_fee_rate": args.futures_taker_fee_rate,
        "slippage_bps": args.slippage_bps,
        "maintenance_margin_rate": args.maintenance_margin_rate,
        "runtime_seconds": elapsed,
        "feature_precompute_seconds": feature_seconds,
        "candles_per_second": total_candles_scanned / elapsed if elapsed > 0 else 0.0,
        "strategy_evaluations_per_second": total_candles_scanned / elapsed if elapsed > 0 else 0.0,
        "parameter_sets": len(specs),
        "parameter_sets_per_minute": len(specs) / elapsed * 60 if elapsed > 0 else 0.0,
        "total_combinations": len(rows),
        "positive_combinations": sum(1 for row in rows if row.net > 0),
        "positive_combinations_with_at_least_30_trades": sum(1 for row in rows if row.trades >= 30 and row.net > 0),
        "best_overall": row_to_dict(best_overall),
        "best_at_least_30": row_to_dict(best_30),
        "worst_overall": row_to_dict(worst),
        "agent_comparison": agent_comparison,
        "strategy_leaderboard": strategy_leaderboard,
        "agent_name": primary.agent_name if primary else "n/a",
        "strategy_name": primary.strategy if primary else "n/a",
        "trades_per_day": primary.trades_per_day if primary else 0.0,
        "avg_daily_return_pct": primary.avg_daily_return_pct if primary else 0.0,
        "median_daily_return_pct": primary.median_daily_return_pct if primary else 0.0,
        "days_profitable_pct": primary.days_profitable_pct if primary else 0.0,
        "days_above_5pct": primary.days_above_5pct if primary else 0,
        "days_above_5pct_pct": primary.days_above_5pct_pct if primary else 0.0,
        "max_daily_drawdown_pct": primary.max_daily_drawdown_pct if primary else 0.0,
        "fee_drag_pct": primary.fee_drag_pct if primary else 0.0,
        "liquidation_events": primary.liquidation_events if primary else 0,
        "overfit_warning": primary.overfit_warning if primary else True,
        "walk_forward_verdict": primary.verdict if primary else "too_few_trades",
        "verdict": primary.verdict if primary else "too_few_trades",
        "overfit_warning_reasons": primary.failure_reasons if primary else ["too_few_trades"],
        "target_a_100_trades_per_day": target_verdicts["target_a"],
        "target_b_5pct_avg_daily_return": target_verdicts["target_b"],
        "target_c_75pct_days_above_5pct": target_verdicts["target_c"],
        "verdict_100_trades_per_day": target_verdicts["target_a"]["verdict"],
        "verdict_5pct_daily_target": target_verdicts["target_b"]["verdict"],
        "verdict_75pct_consistency_target": target_verdicts["target_c"]["verdict"],
        "system_status": "PROFITABLE_CANDIDATE" if primary and primary.verdict in {"robust_candidate", "potentially_promising_needs_more_testing"} else "NOT_PROFITABLE",
        "primary_failure": primary_failure(primary),
        "daily_scalping_metrics": daily_summary_dict(primary),
    }
    return {"summary": summary, "rows": rows, "path": str(path)}


def load_csv(path: Path, limit: int | None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        columns = ["timestamp", "open", "high", "low", "close", "volume"]
        df = pd.read_csv(path, names=columns, header=0)
    if limit is not None and limit > 0:
        df = df.tail(limit)
    return df[["timestamp", "open", "high", "low", "close", "volume"]].copy()


def find_csv(data_dir: Path, symbol: str, timeframe: str) -> Path:
    compact = symbol.replace("/", "")
    candidates = [
        data_dir / f"{compact}_{timeframe}.csv",
        data_dir / f"{compact}.csv",
        data_dir / f"{symbol.replace('/', '_')}_{timeframe}.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No BTCUSDT {timeframe} CSV found in {data_dir}")


def build_feature_arrays(df: pd.DataFrame) -> FeatureArrays:
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)
    typical = (high + low + close) / 3
    rolling_volume = volume.rolling(50).sum().replace(0.0, np.nan)
    vwap = (typical * volume).rolling(50).sum() / rolling_volume
    volume_ratio = volume / volume.rolling(30).mean().replace(0.0, np.nan)
    close_position = (close - low) / (high - low).replace(0.0, np.nan)
    range_high_20 = high.shift(1).rolling(20).max()
    range_low_20 = low.shift(1).rolling(20).min()
    range_high_40 = high.shift(1).rolling(40).max()
    range_low_40 = low.shift(1).rolling(40).min()
    return FeatureArrays(
        timestamp=df["timestamp"].to_numpy(dtype=float, copy=False),
        open=df["open"].to_numpy(dtype=float, copy=False),
        high=high.to_numpy(dtype=float, copy=False),
        low=low.to_numpy(dtype=float, copy=False),
        close=close.to_numpy(dtype=float, copy=False),
        volume=volume.to_numpy(dtype=float, copy=False),
        ema_fast=df["ema_fast"].to_numpy(dtype=float, copy=False),
        ema_slow=df["ema_slow"].to_numpy(dtype=float, copy=False),
        macd_hist=df["macd_hist"].to_numpy(dtype=float, copy=False),
        rsi=df["rsi"].to_numpy(dtype=float, copy=False),
        atr=df["atr"].to_numpy(dtype=float, copy=False),
        atr_bps=(df["atr"] / close * 10_000).replace([np.inf, -np.inf], 0.0).fillna(0.0).to_numpy(dtype=float, copy=False),
        volume_ratio=volume_ratio.replace([np.inf, -np.inf], 1.0).fillna(1.0).to_numpy(dtype=float, copy=False),
        close_position=close_position.replace([np.inf, -np.inf], 0.5).fillna(0.5).to_numpy(dtype=float, copy=False),
        vwap=vwap.bfill().fillna(close).to_numpy(dtype=float, copy=False),
        return_3_bps=(close / close.shift(3) - 1).replace([np.inf, -np.inf], 0.0).fillna(0.0).mul(10_000).to_numpy(dtype=float, copy=False),
        return_5_bps=(close / close.shift(5) - 1).replace([np.inf, -np.inf], 0.0).fillna(0.0).mul(10_000).to_numpy(dtype=float, copy=False),
        return_10_bps=(close / close.shift(10) - 1).replace([np.inf, -np.inf], 0.0).fillna(0.0).mul(10_000).to_numpy(dtype=float, copy=False),
        trend_20_bps=(close / close.shift(20) - 1).replace([np.inf, -np.inf], 0.0).fillna(0.0).mul(10_000).to_numpy(dtype=float, copy=False),
        range_high_20=range_high_20.bfill().fillna(close).to_numpy(dtype=float, copy=False),
        range_low_20=range_low_20.bfill().fillna(close).to_numpy(dtype=float, copy=False),
        range_high_40=range_high_40.bfill().fillna(close).to_numpy(dtype=float, copy=False),
        range_low_40=range_low_40.bfill().fillna(close).to_numpy(dtype=float, copy=False),
        range_bps_20=((range_high_20 - range_low_20) / close * 10_000).replace([np.inf, -np.inf], 0.0).fillna(0.0).to_numpy(dtype=float, copy=False),
        range_bps_40=((range_high_40 - range_low_40) / close * 10_000).replace([np.inf, -np.inf], 0.0).fillna(0.0).to_numpy(dtype=float, copy=False),
    )


def default_specs(timeframe: str) -> list[SearchSpec]:
    hold_fast = 12 if timeframe == "1m" else 16
    hold_slow = 24 if timeframe == "1m" else 32
    target_values = (18.0, 24.0, 32.0) if timeframe == "1m" else (35.0, 50.0, 75.0)
    stop_values = (10.0, 14.0, 18.0) if timeframe == "1m" else (18.0, 24.0, 32.0)
    specs: list[SearchSpec] = []

    for target_bps, stop_bps, ret_bps, volume_ratio in product(target_values, stop_values[:2], (6.0, 10.0, 14.0), (1.05, 1.20)):
        if target_bps <= stop_bps:
            continue
        specs.append(SearchSpec("agent_1_momentum_breakout", "momentum_burst", "buy", 20, target_bps, stop_bps, hold_fast, ret_bps, volume_ratio, 8.0, 3.0, 0.58, 0.92, 50.0, 72.0, 0.0))
        specs.append(SearchSpec("agent_1_momentum_breakout", "momentum_burst", "sell", 20, target_bps, stop_bps, hold_fast, ret_bps, volume_ratio, 8.0, 3.0, 0.08, 0.42, 28.0, 50.0, 0.0))

    for target_bps, stop_bps, volume_ratio in product(target_values, stop_values[:2], (1.10, 1.30, 1.60)):
        if target_bps <= stop_bps:
            continue
        specs.append(SearchSpec("agent_1_momentum_breakout", "range_breakout", "buy", 20, target_bps, stop_bps, hold_slow, 0.0, volume_ratio, 10.0, 2.0, 0.55, 0.95, 48.0, 78.0, 0.0, min_range_bps=target_bps * 0.55))
        specs.append(SearchSpec("agent_1_momentum_breakout", "range_breakout", "sell", 20, target_bps, stop_bps, hold_slow, 0.0, volume_ratio, 10.0, 2.0, 0.05, 0.45, 22.0, 52.0, 0.0, min_range_bps=target_bps * 0.55))

    for target_bps, stop_bps, volume_ratio in product((12.0, 18.0, 24.0), (8.0, 12.0, 16.0), (0.90, 1.05, 1.20)):
        if target_bps <= stop_bps:
            continue
        specs.append(SearchSpec("agent_2_microstructure_scalping", "ema_vwap_continuation", "buy", 20, target_bps, stop_bps, hold_fast, 2.0, volume_ratio, 5.0, 1.5, 0.35, 0.78, 45.0, 68.0, 0.0, vwap_side_required=True))
        specs.append(SearchSpec("agent_2_microstructure_scalping", "ema_vwap_continuation", "sell", 20, target_bps, stop_bps, hold_fast, 2.0, volume_ratio, 5.0, 1.5, 0.22, 0.65, 32.0, 55.0, 0.0, vwap_side_required=True))

    for target_bps, stop_bps, ret_bps, atr_bps in product((16.0, 24.0, 32.0, 45.0), (10.0, 14.0, 20.0), (4.0, 8.0, 12.0), (5.0, 12.0, 20.0)):
        if target_bps <= stop_bps:
            continue
        specs.append(SearchSpec("agent_3_adaptive_experimental", "adaptive_hybrid", "buy", 40, target_bps, stop_bps, hold_slow, ret_bps, 1.0, atr_bps, 1.0, 0.48, 0.88, 42.0, 74.0, 0.0, min_range_bps=target_bps * 0.4))
        specs.append(SearchSpec("agent_3_adaptive_experimental", "adaptive_hybrid", "sell", 40, target_bps, stop_bps, hold_slow, ret_bps, 1.0, atr_bps, 1.0, 0.12, 0.52, 26.0, 58.0, 0.0, min_range_bps=target_bps * 0.4))

    return interleave_by_agent(specs)


def interleave_by_agent(specs: list[SearchSpec]) -> list[SearchSpec]:
    grouped: dict[str, list[SearchSpec]] = defaultdict(list)
    for spec in specs:
        grouped[spec.agent_name].append(spec)
    ordered: list[SearchSpec] = []
    agent_names = sorted(grouped)
    cursor = 0
    while True:
        added = False
        for agent_name in agent_names:
            items = grouped[agent_name]
            if cursor < len(items):
                ordered.append(items[cursor])
                added = True
        if not added:
            break
        cursor += 1
    return ordered


def evaluate_spec(
    spec: SearchSpec,
    arrays: FeatureArrays,
    symbol: str,
    timeframe: str,
    diagnostic_notional: float,
    leverage: float,
    taker_fee_rate: float,
    slippage_bps: float,
    maintenance_margin_rate: float,
) -> SearchRow:
    mask = signal_mask(spec, arrays)
    max_hold = max(1, int(spec.max_hold))
    valid = np.arange(len(arrays.close)) >= max(spec.lookback, 60)
    valid &= np.arange(len(arrays.close)) < len(arrays.close) - max_hold - 1
    indices = np.flatnonzero(mask & valid)
    if spec.min_spacing > 1 and len(indices) > 1:
        indices = spaced_indices(indices, spec.min_spacing)

    trades = [
        simulate_trade(
            index=int(index),
            spec=spec,
            arrays=arrays,
            diagnostic_notional=diagnostic_notional,
            leverage=leverage,
            taker_fee_rate=taker_fee_rate,
            slippage_bps=slippage_bps,
            maintenance_margin_rate=maintenance_margin_rate,
        )
        for index in indices
    ]
    return summarize_trades(
        symbol=symbol,
        timeframe=timeframe,
        spec=spec,
        trades=trades,
        candles_tested=max(int(len(arrays.close) - max(spec.lookback, 60)), 0),
        signals_considered=int(len(indices)),
        diagnostic_notional=diagnostic_notional,
        leverage=leverage,
    )


def signal_mask(spec: SearchSpec, arrays: FeatureArrays) -> np.ndarray:
    close = arrays.close
    side_direction = 1 if spec.side == "buy" else -1
    trend = (arrays.ema_fast > arrays.ema_slow) if spec.side == "buy" else (arrays.ema_fast < arrays.ema_slow)
    macd = (arrays.macd_hist * side_direction / np.maximum(close, 1e-9) * 10_000) >= spec.min_macd_bps
    rsi = (arrays.rsi >= spec.min_rsi) & (arrays.rsi <= spec.max_rsi)
    volume = arrays.volume_ratio >= spec.min_volume_ratio
    atr = arrays.atr_bps >= spec.min_atr_bps
    ema_gap = np.abs(arrays.ema_fast - arrays.ema_slow) / np.maximum(close, 1e-9) * 10_000 >= spec.min_ema_gap_bps
    close_position = (arrays.close_position >= spec.min_close_position) & (arrays.close_position <= spec.max_close_position)

    if spec.strategy_name == "range_breakout":
        range_high = arrays.range_high_20 if spec.lookback <= 20 else arrays.range_high_40
        range_low = arrays.range_low_20 if spec.lookback <= 20 else arrays.range_low_40
        range_bps = arrays.range_bps_20 if spec.lookback <= 20 else arrays.range_bps_40
        breakout = (close > range_high) if spec.side == "buy" else (close < range_low)
        return trend & macd & rsi & volume & atr & ema_gap & close_position & breakout & (range_bps >= spec.min_range_bps)

    returns = arrays.return_3_bps if spec.strategy_name == "ema_vwap_continuation" else arrays.return_5_bps
    momentum = returns * side_direction >= spec.min_return_bps
    vwap_ok = np.ones_like(close, dtype=bool)
    if spec.vwap_side_required:
        vwap_ok = close >= arrays.vwap if spec.side == "buy" else close <= arrays.vwap

    if spec.strategy_name == "adaptive_hybrid":
        range_ok = arrays.range_bps_40 >= spec.min_range_bps
        trend_strength = arrays.trend_20_bps * side_direction >= spec.min_return_bps * 0.8
        return trend & macd & rsi & volume & atr & ema_gap & close_position & vwap_ok & range_ok & (momentum | trend_strength)

    return trend & macd & rsi & volume & atr & ema_gap & close_position & vwap_ok & momentum


def spaced_indices(indices: np.ndarray, spacing: int) -> np.ndarray:
    selected: list[int] = []
    last = -10**12
    for index in indices:
        if int(index) - last >= spacing:
            selected.append(int(index))
            last = int(index)
    return np.asarray(selected, dtype=int)


def simulate_trade(
    index: int,
    spec: SearchSpec,
    arrays: FeatureArrays,
    diagnostic_notional: float,
    leverage: float,
    taker_fee_rate: float,
    slippage_bps: float,
    maintenance_margin_rate: float,
) -> SearchTrade:
    direction = 1 if spec.side == "buy" else -1
    entry = float(arrays.close[index])
    target = entry * (1 + direction * spec.target_bps / 10_000)
    stop = entry * (1 - direction * spec.stop_bps / 10_000)
    liq = liquidation_price(entry, spec.side, leverage, maintenance_margin_rate)
    exit_price = float(arrays.close[min(index + spec.max_hold, len(arrays.close) - 1)])
    exit_index = min(index + spec.max_hold, len(arrays.close) - 1)
    exit_reason = "max_horizon_exit"
    liquidation_event = False
    for offset in range(1, spec.max_hold + 1):
        cursor = index + offset
        if cursor >= len(arrays.close):
            break
        high = float(arrays.high[cursor])
        low = float(arrays.low[cursor])
        if spec.side == "buy":
            if leverage > 1.0 and low <= liq:
                exit_price = liq
                exit_reason = "liquidation_event"
                liquidation_event = True
            elif low <= stop:
                exit_price = stop
                exit_reason = "stop_loss_hit"
            elif high >= target:
                exit_price = target
                exit_reason = "take_profit_hit"
            else:
                continue
        else:
            if leverage > 1.0 and high >= liq:
                exit_price = liq
                exit_reason = "liquidation_event"
                liquidation_event = True
            elif high >= stop:
                exit_price = stop
                exit_reason = "stop_loss_hit"
            elif low <= target:
                exit_price = target
                exit_reason = "take_profit_hit"
            else:
                continue
        exit_index = cursor
        break

    gross = (exit_price - entry) / max(entry, 1e-9) * diagnostic_notional * direction
    if liquidation_event:
        margin = diagnostic_notional / max(leverage, 1e-9)
        gross = max(gross, -margin)
    exit_notional = diagnostic_notional * max(exit_price / max(entry, 1e-9), 0.0)
    fees = (diagnostic_notional + exit_notional) * taker_fee_rate
    slippage = (diagnostic_notional + exit_notional) * slippage_bps / 10_000
    costs = fees + slippage
    return SearchTrade(
        timestamp=float(arrays.timestamp[index]),
        entry_index=index,
        exit_index=exit_index,
        side=spec.side,
        entry_price=entry,
        exit_price=exit_price,
        exit_reason=exit_reason,
        hold_candles=max(exit_index - index, 0),
        gross_pnl=float(gross),
        fees=float(fees),
        slippage_costs=float(slippage),
        total_costs=float(costs),
        net_pnl=float(gross - costs),
        liquidation_event=liquidation_event,
    )


def liquidation_price(entry: float, side: str, leverage: float, maintenance_margin_rate: float) -> float:
    if leverage <= 1.0:
        return 0.0 if side == "buy" else float("inf")
    if side == "buy":
        return entry * max(0.0, 1 - 1 / leverage + maintenance_margin_rate)
    return entry * (1 + 1 / leverage - maintenance_margin_rate)


def summarize_trades(
    symbol: str,
    timeframe: str,
    spec: SearchSpec,
    trades: list[SearchTrade],
    candles_tested: int,
    signals_considered: int,
    diagnostic_notional: float,
    leverage: float,
) -> SearchRow:
    net_values = [trade.net_pnl for trade in trades]
    gross_profit = sum(value for value in net_values if value > 0)
    gross_loss = abs(sum(value for value in net_values if value < 0))
    pf = None if gross_loss <= 0 and gross_profit <= 0 else (float("inf") if gross_loss <= 0 else gross_profit / gross_loss)
    wins = sum(1 for value in net_values if value > 0)
    losses = len(trades) - wins
    daily = daily_metrics(trades, diagnostic_notional)
    walk_forward = walk_forward_splits(trades)
    overfit_warning = len(trades) < 30 or any(split["net"] <= 0 or split["pf"] is None or split["pf"] <= 1.0 for split in walk_forward if split["trades"] > 0)
    failure_reasons = failure_reasons_for(
        trades=trades,
        net=sum(net_values),
        pf=pf,
        daily=daily,
        walk_forward=walk_forward,
    )
    verdict = verdict_for(trades, sum(net_values), pf, overfit_warning)
    exit_counts = Counter(trade.exit_reason for trade in trades)
    return SearchRow(
        symbol=symbol,
        timeframe=timeframe,
        agent_name=spec.agent_name,
        strategy=spec.strategy_name,
        side=spec.side,
        target_bps=spec.target_bps,
        stop_bps=spec.stop_bps,
        max_hold=spec.max_hold,
        parameter_set=parameter_label(spec),
        candles_tested=candles_tested,
        signals_considered=signals_considered,
        trades=len(trades),
        wins=wins,
        losses=losses,
        gross=sum(trade.gross_pnl for trade in trades),
        costs=sum(trade.total_costs for trade in trades),
        net=sum(net_values),
        avg_net=sum(net_values) / len(trades) if trades else 0.0,
        pf=pf,
        win_rate=wins / len(trades) * 100 if trades else 0.0,
        max_drawdown=max_drawdown(net_values),
        max_drawdown_pct=max_drawdown(net_values) / max(diagnostic_notional, 1e-9) * 100,
        trades_per_day=daily["trades_per_day"],
        avg_daily_return_pct=daily["avg_daily_return_pct"],
        median_daily_return_pct=daily["median_daily_return_pct"],
        best_daily_return_pct=daily["best_daily_return_pct"],
        worst_daily_return_pct=daily["worst_daily_return_pct"],
        days_profitable_pct=daily["days_profitable_pct"],
        days_above_5pct=daily["days_above_5pct"],
        days_above_5pct_pct=daily["days_above_5pct_pct"],
        max_daily_drawdown_pct=daily["max_daily_drawdown_pct"],
        fee_drag_pct=daily["fee_drag_pct"],
        liquidation_events=sum(1 for trade in trades if trade.liquidation_event),
        liquidation_risk_flag=bool(leverage > 1.0),
        leverage_used=leverage,
        walk_forward=walk_forward,
        overfit_warning=overfit_warning,
        verdict=verdict,
        failure_reasons=failure_reasons,
        exit_reason_counts=dict(exit_counts),
    )


def daily_metrics(trades: list[SearchTrade], diagnostic_notional: float) -> dict[str, Any]:
    if not trades:
        return empty_daily_metrics()
    by_day: dict[str, list[SearchTrade]] = defaultdict(list)
    for trade in trades:
        by_day[day_key(trade.timestamp)].append(trade)
    first_day = min(by_day)
    last_day = max(by_day)
    calendar_days = max((datetime.fromisoformat(last_day) - datetime.fromisoformat(first_day)).days + 1, len(by_day))
    daily_returns: list[float] = []
    daily_drawdowns: list[float] = []
    daily_fee_drag: list[float] = []
    for day in sorted(by_day):
        values = [trade.net_pnl for trade in by_day[day]]
        daily_returns.append(sum(values) / max(diagnostic_notional, 1e-9) * 100)
        daily_drawdowns.append(max_drawdown(values) / max(diagnostic_notional, 1e-9) * 100)
        daily_fee_drag.append(sum(trade.total_costs for trade in by_day[day]) / max(diagnostic_notional, 1e-9) * 100)
    zero_days = max(calendar_days - len(daily_returns), 0)
    calendar_returns = daily_returns + [0.0] * zero_days
    return {
        "basis": "full_data_period_including_zero_trade_days",
        "calendar_days": calendar_days,
        "active_trade_days": len(daily_returns),
        "zero_trade_days": zero_days,
        "trades_per_day": len(trades) / calendar_days if calendar_days else 0.0,
        "avg_daily_return_pct": sum(calendar_returns) / len(calendar_returns) if calendar_returns else 0.0,
        "median_daily_return_pct": median(calendar_returns),
        "best_daily_return_pct": max(calendar_returns) if calendar_returns else 0.0,
        "worst_daily_return_pct": min(calendar_returns) if calendar_returns else 0.0,
        "days_profitable_pct": sum(1 for value in calendar_returns if value > 0) / len(calendar_returns) * 100 if calendar_returns else 0.0,
        "days_above_5pct": sum(1 for value in calendar_returns if value >= 5.0),
        "days_above_5pct_pct": sum(1 for value in calendar_returns if value >= 5.0) / len(calendar_returns) * 100 if calendar_returns else 0.0,
        "max_daily_drawdown_pct": max(daily_drawdowns) if daily_drawdowns else 0.0,
        "fee_drag_pct": sum(daily_fee_drag) / calendar_days if calendar_days else 0.0,
    }


def walk_forward_splits(trades: list[SearchTrade]) -> list[dict[str, Any]]:
    labels = ("train", "validation", "test")
    if not trades:
        return [split_stats(label, []) for label in labels]
    ordered = sorted(trades, key=lambda trade: trade.timestamp)
    chunks = [
        ordered[round(i * len(ordered) / 3) : round((i + 1) * len(ordered) / 3)]
        for i in range(3)
    ]
    return [split_stats(label, chunk) for label, chunk in zip(labels, chunks)]


def split_stats(label: str, trades: list[SearchTrade]) -> dict[str, Any]:
    values = [trade.net_pnl for trade in trades]
    profit = sum(value for value in values if value > 0)
    loss = abs(sum(value for value in values if value < 0))
    pf = None if loss <= 0 and profit <= 0 else (float("inf") if loss <= 0 else profit / loss)
    return {
        "split": label,
        "trades": len(trades),
        "net": sum(values),
        "avg_net": sum(values) / len(values) if values else 0.0,
        "pf": pf,
        "win_rate": sum(1 for value in values if value > 0) / len(values) * 100 if values else 0.0,
        "max_drawdown": max_drawdown(values),
    }


def failure_reasons_for(
    trades: list[SearchTrade],
    net: float,
    pf: float | None,
    daily: dict[str, Any],
    walk_forward: list[dict[str, Any]],
) -> list[str]:
    reasons: list[str] = []
    if len(trades) < 30:
        reasons.append("too_few_trades")
    if net <= 0 or pf is None or pf < 1.1:
        reasons.append("insufficient_edge")
    total_costs = sum(trade.total_costs for trade in trades)
    gross_profit = sum(trade.gross_pnl for trade in trades if trade.gross_pnl > 0)
    if total_costs >= max(gross_profit * 0.35, 1e-9):
        reasons.append("fee_drag")
    if sum(trade.slippage_costs for trade in trades) > max(abs(net), 1e-9) * 0.25:
        reasons.append("slippage")
    if daily.get("trades_per_day", 0.0) < TARGET_TRADES_PER_DAY:
        reasons.append("volatility_constraints")
    if any(split["trades"] > 0 and (split["net"] <= 0 or split["pf"] is None or split["pf"] <= 1.0) for split in walk_forward):
        reasons.append("overfitting")
    if any(trade.liquidation_event for trade in trades):
        reasons.append("liquidation_risk")
    return sorted(set(reasons))


def verdict_for(trades: list[SearchTrade], net: float, pf: float | None, overfit_warning: bool) -> str:
    if len(trades) < 30:
        return "too_few_trades"
    if overfit_warning:
        return "not_profitable_out_of_sample"
    if net <= 0 or pf is None or pf <= 1.0:
        return "not_profitable"
    if len(trades) >= 100 and pf >= 1.2:
        return "robust_candidate"
    return "potentially_promising_needs_more_testing"


def evaluate_targets(row: SearchRow | None) -> dict[str, dict[str, Any]]:
    if row is None:
        return {
            "target_a": target_result("not_achieved", ["too_few_trades"]),
            "target_b": target_result("not_achieved", ["too_few_trades"]),
            "target_c": target_result("not_achieved", ["too_few_trades"]),
        }
    base_reasons = row.failure_reasons or ["insufficient_edge"]
    return {
        "target_a": target_result("achieved" if row.trades_per_day >= TARGET_TRADES_PER_DAY and row.net > 0 and not row.overfit_warning else target_failure_verdict(row), base_reasons),
        "target_b": target_result("achieved" if row.avg_daily_return_pct >= TARGET_AVG_DAILY_RETURN_PCT and row.net > 0 and not row.overfit_warning else target_failure_verdict(row), base_reasons),
        "target_c": target_result("achieved" if row.days_above_5pct_pct >= TARGET_DAYS_ABOVE_5PCT_PCT and row.net > 0 and not row.overfit_warning else target_failure_verdict(row), base_reasons),
    }


def target_failure_verdict(row: SearchRow) -> str:
    if row.overfit_warning or row.pf is None or row.pf < 0.8 or row.net <= 0:
        return "unrealistic_given_data"
    return "not_achieved"


def target_result(verdict: str, reasons: list[str]) -> dict[str, Any]:
    return {"verdict": verdict, "reasons": sorted(set(reasons))}


def build_agent_comparison(rows: list[SearchRow]) -> list[dict[str, Any]]:
    grouped: dict[str, list[SearchRow]] = defaultdict(list)
    for row in rows:
        grouped[row.agent_name].append(row)
    comparison: list[dict[str, Any]] = []
    for agent_name, items in sorted(grouped.items()):
        best = best_row(items, min_trades=1)
        comparison.append(
            {
                "agent_name": agent_name,
                "parameter_sets": len(items),
                "positive_rows": sum(1 for row in items if row.net > 0),
                "best": row_to_dict(best),
                "best_30": row_to_dict(best_row(items, min_trades=30)),
                "worst": row_to_dict(worst_row(items)),
            }
        )
    return comparison


def best_row(rows: list[SearchRow], min_trades: int) -> SearchRow | None:
    candidates = [row for row in rows if row.trades >= min_trades]
    return max(candidates, key=leaderboard_key, default=None)


def worst_row(rows: list[SearchRow]) -> SearchRow | None:
    return min(rows, key=lambda row: (row.net, row.avg_net), default=None)


def leaderboard_key(row: SearchRow) -> tuple[float, float, float, float, float]:
    pf_value = 0.0 if row.pf is None or row.pf == float("inf") else row.pf
    return (
        1.0 if row.verdict in {"robust_candidate", "potentially_promising_needs_more_testing"} else 0.0,
        row.net,
        pf_value,
        row.trades_per_day,
        -row.max_drawdown,
    )


def row_to_dict(row: SearchRow | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "symbol": row.symbol,
        "timeframe": row.timeframe,
        "agent_name": row.agent_name,
        "strategy": row.strategy,
        "side": row.side,
        "target_bps": row.target_bps,
        "stop_bps": row.stop_bps,
        "hold": row.max_hold,
        "parameter_set": row.parameter_set,
        "candles_tested": row.candles_tested,
        "signals_considered": row.signals_considered,
        "trades": row.trades,
        "wins": row.wins,
        "losses": row.losses,
        "win_rate": row.win_rate,
        "gross": row.gross,
        "costs": row.costs,
        "net": row.net,
        "avg_net": row.avg_net,
        "pf": row.pf,
        "max_drawdown": row.max_drawdown,
        "max_drawdown_pct": row.max_drawdown_pct,
        "trades_per_day": row.trades_per_day,
        "avg_daily_return_pct": row.avg_daily_return_pct,
        "median_daily_return_pct": row.median_daily_return_pct,
        "days_profitable_pct": row.days_profitable_pct,
        "days_above_5pct": row.days_above_5pct,
        "days_above_5pct_pct": row.days_above_5pct_pct,
        "max_daily_drawdown_pct": row.max_daily_drawdown_pct,
        "fee_drag_pct": row.fee_drag_pct,
        "liquidation_events": row.liquidation_events,
        "liquidation_risk_flag": row.liquidation_risk_flag,
        "leverage_used": row.leverage_used,
        "walk_forward": row.walk_forward,
        "overfit_warning": row.overfit_warning,
        "verdict": row.verdict,
        "failure_reasons": row.failure_reasons,
        "exit_reason_counts": row.exit_reason_counts,
        "daily_metrics": daily_summary_dict(row),
    }


def daily_summary_dict(row: SearchRow | None) -> dict[str, Any]:
    if row is None:
        return empty_daily_metrics()
    return {
        "basis": "full_data_period_including_zero_trade_days",
        "trades_per_day": row.trades_per_day,
        "avg_daily_return_pct": row.avg_daily_return_pct,
        "median_daily_return_pct": row.median_daily_return_pct,
        "best_daily_return_pct": row.best_daily_return_pct,
        "worst_daily_return_pct": row.worst_daily_return_pct,
        "days_profitable_pct": row.days_profitable_pct,
        "days_above_5pct": row.days_above_5pct,
        "days_above_5pct_pct": row.days_above_5pct_pct,
        "max_daily_drawdown_pct": row.max_daily_drawdown_pct,
        "fee_drag_pct": row.fee_drag_pct,
        "liquidation_events": row.liquidation_events,
        "verdict_100_trades_per_day": "achieved" if row.trades_per_day >= TARGET_TRADES_PER_DAY and row.net > 0 else "not_achieved",
        "verdict_5pct_daily_target": "achieved" if row.avg_daily_return_pct >= TARGET_AVG_DAILY_RETURN_PCT and row.net > 0 else "not_achieved",
    }


def empty_daily_metrics() -> dict[str, Any]:
    return {
        "basis": "full_data_period_including_zero_trade_days",
        "trades_per_day": 0.0,
        "avg_daily_return_pct": 0.0,
        "median_daily_return_pct": 0.0,
        "best_daily_return_pct": 0.0,
        "worst_daily_return_pct": 0.0,
        "days_profitable_pct": 0.0,
        "days_above_5pct": 0,
        "days_above_5pct_pct": 0.0,
        "max_daily_drawdown_pct": 0.0,
        "fee_drag_pct": 0.0,
        "liquidation_events": 0,
    }


def build_data_profile(symbol: str, timeframe: str, timestamps: np.ndarray) -> dict[str, Any]:
    first = float(timestamps[0]) if len(timestamps) else None
    last = float(timestamps[-1]) if len(timestamps) else None
    approx = approximate_days(first, last)
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "candles": int(len(timestamps)),
        "first_timestamp": first,
        "last_timestamp": last,
        "start_utc": timestamp_to_utc(first),
        "end_utc": timestamp_to_utc(last),
        "approx_days": approx,
        "data_years": approx / 365.25 if approx is not None else None,
    }


def append_summary(result: dict[str, Any], path: Path, run_label: str) -> None:
    record = {
        "logged_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_label": run_label or f"btc_{result['summary']['timeframe']}_fast_scalping_search",
        "mode": "fast_futures_scalping_search",
        "summary": result["summary"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_safe(record), sort_keys=True, allow_nan=False) + "\n")


def print_report(result: dict[str, Any]) -> None:
    summary = result["summary"]
    print("Fast BTCUSDT Futures Scalping Search")
    print("backtesting only; no live trading; no orders; no leverage unless simulated via CLI")
    print(f"csv={result['path']}")
    print(f"timeframe={summary['timeframe']} candles={summary['total_candles']} data_years={summary['data_years']:.2f}")
    print(f"futures_maker_fee_rate={summary['futures_maker_fee_rate']:.6f} futures_taker_fee_rate={summary['futures_taker_fee_rate']:.6f} slippage_bps={summary['slippage_bps']:.2f}")
    print(f"runtime_seconds={summary['runtime_seconds']:.2f} candles_per_second={summary['candles_per_second']:.2f} parameter_sets_per_minute={summary['parameter_sets_per_minute']:.2f}")
    print(f"system_status={summary['system_status']}")
    print(f"primary_failure={summary['primary_failure']}")
    print(f"best_at_least_30={format_row(summary.get('best_at_least_30'))}")
    print(f"best_overall={format_row(summary.get('best_overall'))}")
    print(f"target_100_trades_per_day={summary['target_a_100_trades_per_day']}")
    print(f"target_5pct_avg_daily_return={summary['target_b_5pct_avg_daily_return']}")
    print(f"target_75pct_days_above_5pct={summary['target_c_75pct_days_above_5pct']}")
    print("Agent Comparison")
    for agent in summary["agent_comparison"]:
        print(f"{agent['agent_name']}: best={format_row(agent.get('best'))} best30={format_row(agent.get('best_30'))}")
    print("Top Leaderboard")
    for row in summary["strategy_leaderboard"][:8]:
        print(format_row(row))


def format_row(row: dict[str, Any] | None) -> str:
    if not row:
        return "n/a"
    return (
        f"{row.get('agent_name', 'n/a')} {row.get('strategy', 'n/a')} {row.get('side', 'n/a')} "
        f"tf={row.get('timeframe', 'n/a')} target={float(row.get('target_bps') or 0):.1f} "
        f"hold={row.get('hold', 'n/a')} trades={row.get('trades', 'n/a')} "
        f"tpd={float(row.get('trades_per_day') or 0):.2f} net=${float(row.get('net') or 0):.2f} "
        f"pf={format_pf(row.get('pf'))} verdict={row.get('verdict', 'n/a')}"
    )


def format_pf(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if math.isinf(number):
        return "inf"
    return f"{number:.2f}"


def primary_failure(row: SearchRow | None) -> str:
    if row is None:
        return "NO_ACCEPTED_STRATEGY"
    reasons = set(row.failure_reasons)
    if {"fee_drag", "insufficient_edge"}.issubset(reasons):
        return "FEE_DRAG + LOW_EDGE"
    if "overfitting" in reasons:
        return "OUT_OF_SAMPLE_FAILURE"
    if "volatility_constraints" in reasons:
        return "LOW_TRADE_FREQUENCY"
    return " + ".join(row.failure_reasons[:2]) if row.failure_reasons else "NONE"


def max_drawdown(values: list[float]) -> float:
    equity = 0.0
    peak = 0.0
    drawdown = 0.0
    for value in values:
        equity += value
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    return drawdown


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def day_key(timestamp: float) -> str:
    seconds = timestamp / 1000 if abs(timestamp) > 10_000_000_000 else timestamp
    return datetime.fromtimestamp(seconds, tz=timezone.utc).date().isoformat()


def timestamp_to_utc(value: float | None) -> str:
    if value is None:
        return "n/a"
    seconds = value / 1000 if abs(value) > 10_000_000_000 else value
    return datetime.fromtimestamp(seconds, tz=timezone.utc).isoformat()


def approximate_days(start: float | None, end: float | None) -> float | None:
    if start is None or end is None:
        return None
    scale = 1000 if abs(start) > 10_000_000_000 or abs(end) > 10_000_000_000 else 1
    days = (end - start) / scale / 86_400
    return days if days >= 0 else None


def parameter_label(spec: SearchSpec) -> str:
    return (
        f"agent={spec.agent_name}|strategy={spec.strategy_name}|side={spec.side}|"
        f"target={spec.target_bps}|stop={spec.stop_bps}|hold={spec.max_hold}|"
        f"ret={spec.min_return_bps}|vol={spec.min_volume_ratio}|atr={spec.min_atr_bps}"
    )


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


if __name__ == "__main__":
    main()
