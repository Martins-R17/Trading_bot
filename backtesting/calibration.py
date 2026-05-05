"""Historical strategy edge calibration.

This module is intentionally separate from the live/paper trading loop. It scans
historical OHLCV candles and reports whether existing strategies can produce
fee-aware candidates under production thresholds and under calibration sweeps.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from config.settings import Settings, load_settings
from core.models import MarketSnapshot, Side, StrategySignal
from data.preprocess import DataPreprocessor
from strategies import (
    BreakoutStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    ScalpingMicrostructureStrategy,
)
from strategies.base_strategy import BaseStrategy


DEFAULT_TARGET_SWEEP = (25.0, 50.0, 75.0, 100.0)
DEFAULT_REWARD_COST_SWEEP = (1.5, 2.0, 3.0)
CORE_CHECKS = ("rsi_check", "ema_trend_check", "macd_check", "volatility_atr_check")


def evaluate_thresholds(
    target_move_bps: float,
    reward_cost_ratio: float,
    expected_net_profit: float,
    min_target_move_bps: float,
    min_reward_cost_ratio: float,
    min_expected_net_profit: float,
) -> "ThresholdGateResult":
    """Evaluate threshold gates independently."""

    return ThresholdGateResult(
        target_pass=target_move_bps >= min_target_move_bps,
        reward_cost_pass=reward_cost_ratio >= min_reward_cost_ratio,
        expected_net_pass=expected_net_profit >= min_expected_net_profit,
    )


def validate_threshold_gate(
    gate: "ThresholdGateResult",
    target_move_bps: float,
    reward_cost_ratio: float,
    expected_net_profit: float,
    min_target_move_bps: float,
    min_reward_cost_ratio: float,
    min_expected_net_profit: float,
    label: str,
) -> None:
    """Fail fast if a counted gate would violate its threshold."""

    if gate.target_pass and target_move_bps < min_target_move_bps:
        raise AssertionError(
            f"{label}: target gate violation target={target_move_bps:.4f} "
            f"threshold={min_target_move_bps:.4f}"
        )
    if gate.reward_cost_pass and reward_cost_ratio < min_reward_cost_ratio:
        raise AssertionError(
            f"{label}: reward/cost gate violation ratio={reward_cost_ratio:.4f} "
            f"threshold={min_reward_cost_ratio:.4f}"
        )
    if gate.expected_net_pass and expected_net_profit < min_expected_net_profit:
        raise AssertionError(
            f"{label}: expected net gate violation net={expected_net_profit:.4f} "
            f"threshold={min_expected_net_profit:.4f}"
        )


@dataclass
class StrategyCalibrationStats:
    """Per-symbol, per-strategy production-threshold funnel."""

    symbol: str
    strategy_name: str
    candles_tested: int = 0
    signals_considered: int = 0
    passing_core_filters: int = 0
    passing_target_move_bps: int = 0
    passing_reward_cost_ratio: int = 0
    passing_expected_net_profit: int = 0
    would_be_trades: int = 0
    target_move_sum: float = 0.0
    reward_cost_sum: float = 0.0
    expected_net_profit_sum: float = 0.0

    def record_considered(
        self,
        target_move_bps: float,
        reward_cost_ratio: float,
        expected_net_profit: float,
    ) -> None:
        self.signals_considered += 1
        self.target_move_sum += target_move_bps
        self.reward_cost_sum += reward_cost_ratio
        self.expected_net_profit_sum += expected_net_profit

    @property
    def average_target_move_bps(self) -> float:
        return self._average(self.target_move_sum)

    @property
    def average_reward_cost_ratio(self) -> float:
        return self._average(self.reward_cost_sum)

    @property
    def average_expected_net_profit(self) -> float:
        return self._average(self.expected_net_profit_sum)

    def _average(self, total: float) -> float:
        if self.signals_considered <= 0:
            return 0.0
        return total / self.signals_considered


@dataclass
class SweepStats:
    """Threshold-sweep result for one symbol/strategy/threshold combination."""

    symbol: str
    strategy_name: str
    min_target_move_bps: float
    min_reward_cost_ratio: float
    pass_count: int = 0
    expected_net_profit_sum: float = 0.0

    @property
    def average_expected_net_profit(self) -> float:
        if self.pass_count <= 0:
            return 0.0
        return self.expected_net_profit_sum / self.pass_count

    def record_if_passes(
        self,
        target_move_bps: float,
        reward_cost_ratio: float,
        expected_net_profit: float,
        min_expected_net_profit: float,
    ) -> None:
        gate = evaluate_thresholds(
            target_move_bps=target_move_bps,
            reward_cost_ratio=reward_cost_ratio,
            expected_net_profit=expected_net_profit,
            min_target_move_bps=self.min_target_move_bps,
            min_reward_cost_ratio=self.min_reward_cost_ratio,
            min_expected_net_profit=min_expected_net_profit,
        )
        validate_threshold_gate(
            gate,
            target_move_bps=target_move_bps,
            reward_cost_ratio=reward_cost_ratio,
            expected_net_profit=expected_net_profit,
            min_target_move_bps=self.min_target_move_bps,
            min_reward_cost_ratio=self.min_reward_cost_ratio,
            min_expected_net_profit=min_expected_net_profit,
            label=f"{self.symbol}:{self.strategy_name}:sweep",
        )
        if not gate.all_pass:
            return
        self.pass_count += 1
        self.expected_net_profit_sum += expected_net_profit


@dataclass
class SimulationTrade:
    """Realized historical exit result for one production-approved candidate."""

    symbol: str
    strategy_name: str
    entry_index: int
    entry_timestamp: float
    entry_price: float
    side: Side
    stop_loss: float
    take_profit: float
    exit_timestamp: float
    exit_price: float
    exit_reason: str
    hold_candles: int
    future_high: float
    future_low: float
    future_high_low_path: list[dict[str, float]]
    realized_gross_pnl: float
    fees: float
    slippage_costs: float
    total_costs: float
    realized_net_pnl: float


@dataclass
class SimulationSummary:
    """Aggregated realized simulation stats for one symbol/strategy pair."""

    symbol: str
    strategy_name: str
    trades: int = 0
    wins: int = 0
    losses: int = 0
    gross_pnl: float = 0.0
    costs: float = 0.0
    net_pnl: float = 0.0
    best_trade: SimulationTrade | None = None
    worst_trade: SimulationTrade | None = None
    exit_reason_counts: dict[str, int] = field(default_factory=dict)

    @property
    def win_rate(self) -> float:
        if self.trades <= 0:
            return 0.0
        return self.wins / self.trades * 100

    @property
    def average_net_per_trade(self) -> float:
        if self.trades <= 0:
            return 0.0
        return self.net_pnl / self.trades


@dataclass(frozen=True)
class ThresholdGateResult:
    """Pass/fail state for target, reward/cost, and expected-net gates."""

    target_pass: bool
    reward_cost_pass: bool
    expected_net_pass: bool

    @property
    def all_pass(self) -> bool:
        return self.target_pass and self.reward_cost_pass and self.expected_net_pass


@dataclass
class CalibrationResult:
    """Complete calibration report data."""

    production_target_move_bps: float
    production_reward_cost_ratio: float
    production_min_expected_net_profit: float
    diagnostic_notional: float
    max_hold_candles: int = 60
    rows: list[StrategyCalibrationStats] = field(default_factory=list)
    sweep_rows: list[SweepStats] = field(default_factory=list)
    simulation_rows: list[SimulationSummary] = field(default_factory=list)
    simulation_trades: list[SimulationTrade] = field(default_factory=list)


class HistoricalStrategyCalibrator:
    """Fast strategy-edge calibration over historical candles."""

    def __init__(
        self,
        settings: Settings,
        target_sweep: tuple[float, ...] = DEFAULT_TARGET_SWEEP,
        reward_cost_sweep: tuple[float, ...] = DEFAULT_REWARD_COST_SWEEP,
        diagnostic_notional: float | None = None,
        max_hold_candles: int = 60,
    ) -> None:
        self.settings = settings
        self.target_sweep = target_sweep
        self.reward_cost_sweep = reward_cost_sweep
        self.diagnostic_notional = diagnostic_notional or self._default_diagnostic_notional()
        self.max_hold_candles = max(1, int(max_hold_candles))

    def run(self, historical_by_symbol: dict[str, pd.DataFrame]) -> CalibrationResult:
        rows: list[StrategyCalibrationStats] = []
        sweep_rows: list[SweepStats] = []
        simulation_trades: list[SimulationTrade] = []

        for symbol, raw_df in historical_by_symbol.items():
            df = DataPreprocessor.add_features(DataPreprocessor.normalize_ohlcv(raw_df))
            for strategy in self._build_strategies():
                stats, sweeps, trades = self._calibrate_strategy(symbol, df, strategy)
                rows.append(stats)
                sweep_rows.extend(sweeps)
                simulation_trades.extend(trades)

        return CalibrationResult(
            production_target_move_bps=self.settings.risk.min_target_move_bps,
            production_reward_cost_ratio=self.settings.risk.min_reward_to_cost_ratio,
            production_min_expected_net_profit=self.settings.risk.min_expected_net_profit_usd,
            diagnostic_notional=self.diagnostic_notional,
            max_hold_candles=self.max_hold_candles,
            rows=rows,
            sweep_rows=sweep_rows,
            simulation_rows=self._summarize_simulation_trades(simulation_trades),
            simulation_trades=simulation_trades,
        )

    def _calibrate_strategy(
        self,
        symbol: str,
        df: pd.DataFrame,
        strategy: BaseStrategy,
    ) -> tuple[StrategyCalibrationStats, list[SweepStats], list[SimulationTrade]]:
        stats = StrategyCalibrationStats(symbol=symbol, strategy_name=strategy.name)
        simulation_trades: list[SimulationTrade] = []
        sweep_lookup = {
            (target, reward): SweepStats(
                symbol=symbol,
                strategy_name=strategy.name,
                min_target_move_bps=target,
                min_reward_cost_ratio=reward,
            )
            for target in self.target_sweep
            for reward in self.reward_cost_sweep
        }

        for index in range(strategy.min_bars, len(df)):
            stats.candles_tested += 1
            window = df.iloc[: index + 1]
            snapshot = MarketSnapshot(
                symbol=symbol,
                timestamp=self._finite_number(window["timestamp"].iloc[-1]),
                ohlcv=window,
                volatility=DataPreprocessor.realized_volatility(window),
            )
            signal = strategy.generate_signal(snapshot)
            metadata = dict(signal.metadata)
            side_considered = str(metadata.get("side_considered", signal.side.value)).lower()
            has_directional_setup = side_considered in {"buy", "sell"} or signal.side in {Side.BUY, Side.SELL}
            if not has_directional_setup:
                continue

            target_move_bps = self._finite_number(metadata.get("target_move_bps"))
            reward_cost_ratio = self._finite_number(metadata.get("reward_cost_ratio"))
            expected_net_profit = self._expected_net_profit(target_move_bps)
            stats.record_considered(target_move_bps, reward_cost_ratio, expected_net_profit)

            core_filters_pass = self._core_filters_pass(metadata)
            if not core_filters_pass:
                continue
            stats.passing_core_filters += 1

            production_gate = evaluate_thresholds(
                target_move_bps=target_move_bps,
                reward_cost_ratio=reward_cost_ratio,
                expected_net_profit=expected_net_profit,
                min_target_move_bps=self.settings.risk.min_target_move_bps,
                min_reward_cost_ratio=self.settings.risk.min_reward_to_cost_ratio,
                min_expected_net_profit=self.settings.risk.min_expected_net_profit_usd,
            )
            validate_threshold_gate(
                production_gate,
                target_move_bps=target_move_bps,
                reward_cost_ratio=reward_cost_ratio,
                expected_net_profit=expected_net_profit,
                min_target_move_bps=self.settings.risk.min_target_move_bps,
                min_reward_cost_ratio=self.settings.risk.min_reward_to_cost_ratio,
                min_expected_net_profit=self.settings.risk.min_expected_net_profit_usd,
                label=f"{symbol}:{strategy.name}:production",
            )
            if production_gate.target_pass:
                stats.passing_target_move_bps += 1
            if production_gate.reward_cost_pass:
                stats.passing_reward_cost_ratio += 1
            if production_gate.expected_net_pass:
                stats.passing_expected_net_profit += 1
            if production_gate.all_pass and self._is_simulatable_signal(signal):
                stats.would_be_trades += 1
                simulation_trades.append(
                    self._simulate_exit(
                        symbol=symbol,
                        strategy_name=strategy.name,
                        df=df,
                        entry_index=index,
                        signal=signal,
                    )
                )

            self._record_sweeps(sweep_lookup, target_move_bps, reward_cost_ratio, expected_net_profit)

        return stats, list(sweep_lookup.values()), simulation_trades

    def _is_simulatable_signal(self, signal: StrategySignal) -> bool:
        if not signal.is_actionable or signal.side not in {Side.BUY, Side.SELL}:
            return False
        entry = self._finite_number(signal.entry_price)
        stop_loss = self._finite_number(signal.stop_loss)
        take_profit = self._finite_number(signal.take_profit)
        if min(entry, stop_loss, take_profit) <= 0:
            return False
        if signal.side == Side.BUY:
            return stop_loss < entry < take_profit
        return take_profit < entry < stop_loss

    def _simulate_exit(
        self,
        symbol: str,
        strategy_name: str,
        df: pd.DataFrame,
        entry_index: int,
        signal: StrategySignal,
    ) -> SimulationTrade:
        entry_price = self._finite_number(signal.entry_price)
        stop_loss = self._finite_number(signal.stop_loss)
        take_profit = self._finite_number(signal.take_profit)
        entry_timestamp = self._finite_number(df["timestamp"].iloc[entry_index])
        horizon_end = min(len(df), entry_index + 1 + self.max_hold_candles)
        future = df.iloc[entry_index + 1 : horizon_end]

        exit_price = entry_price
        exit_timestamp = entry_timestamp
        exit_reason = "max_horizon_exit"
        hold_candles = 0
        future_high = entry_price
        future_low = entry_price
        future_high_low_path: list[dict[str, float]] = []

        if len(future) > 0:
            future_high = self._finite_number(future["high"].max(), entry_price)
            future_low = self._finite_number(future["low"].min(), entry_price)
            last_row = future.iloc[-1]
            exit_price = self._finite_number(last_row["close"], entry_price)
            exit_timestamp = self._finite_number(last_row["timestamp"], entry_timestamp)
            hold_candles = len(future)
            future_high_low_path = [
                {
                    "timestamp": self._finite_number(getattr(row, "timestamp"), entry_timestamp),
                    "high": self._finite_number(getattr(row, "high"), entry_price),
                    "low": self._finite_number(getattr(row, "low"), entry_price),
                }
                for row in future.itertuples(index=False)
            ]

            for offset, candle in enumerate(future_high_low_path, start=1):
                high = candle["high"]
                low = candle["low"]
                timestamp = candle["timestamp"]

                if signal.side == Side.BUY:
                    stop_hit = low <= stop_loss
                    target_hit = high >= take_profit
                else:
                    stop_hit = high >= stop_loss
                    target_hit = low <= take_profit

                if stop_hit:
                    exit_price = stop_loss
                    exit_reason = "stop_loss_hit"
                elif target_hit:
                    exit_price = take_profit
                    exit_reason = "take_profit_hit"
                else:
                    continue

                exit_timestamp = timestamp
                hold_candles = offset
                break

        gross_pnl, fees, slippage_costs, total_costs, net_pnl = self._realized_pnl_components(
            side=signal.side,
            entry_price=entry_price,
            exit_price=exit_price,
        )
        return SimulationTrade(
            symbol=symbol,
            strategy_name=strategy_name,
            entry_index=entry_index,
            entry_timestamp=entry_timestamp,
            entry_price=entry_price,
            side=signal.side,
            stop_loss=stop_loss,
            take_profit=take_profit,
            exit_timestamp=exit_timestamp,
            exit_price=exit_price,
            exit_reason=exit_reason,
            hold_candles=hold_candles,
            future_high=future_high,
            future_low=future_low,
            future_high_low_path=future_high_low_path,
            realized_gross_pnl=gross_pnl,
            fees=fees,
            slippage_costs=slippage_costs,
            total_costs=total_costs,
            realized_net_pnl=net_pnl,
        )

    def _realized_pnl_components(
        self,
        side: Side,
        entry_price: float,
        exit_price: float,
    ) -> tuple[float, float, float, float, float]:
        amount = self.diagnostic_notional / max(entry_price, 1e-9)
        gross_pnl = (exit_price - entry_price) * amount * side.direction
        entry_notional = abs(entry_price * amount)
        exit_notional = abs(exit_price * amount)
        fees = (entry_notional + exit_notional) * self.settings.risk.taker_fee_rate
        slippage_costs = (entry_notional + exit_notional) * self.settings.risk.slippage_rate
        total_costs = fees + slippage_costs
        return (
            float(gross_pnl),
            float(fees),
            float(slippage_costs),
            float(total_costs),
            float(gross_pnl - total_costs),
        )

    def _summarize_simulation_trades(
        self,
        trades: list[SimulationTrade],
    ) -> list[SimulationSummary]:
        grouped: dict[tuple[str, str], list[SimulationTrade]] = defaultdict(list)
        for trade in trades:
            grouped[(trade.symbol, trade.strategy_name)].append(trade)

        summaries: list[SimulationSummary] = []
        for (symbol, strategy_name), items in grouped.items():
            best_trade = max(items, key=lambda trade: trade.realized_net_pnl)
            worst_trade = min(items, key=lambda trade: trade.realized_net_pnl)
            exit_counts = Counter(trade.exit_reason for trade in items)
            summaries.append(
                SimulationSummary(
                    symbol=symbol,
                    strategy_name=strategy_name,
                    trades=len(items),
                    wins=sum(1 for trade in items if trade.realized_net_pnl > 0),
                    losses=sum(1 for trade in items if trade.realized_net_pnl <= 0),
                    gross_pnl=sum(trade.realized_gross_pnl for trade in items),
                    costs=sum(trade.total_costs for trade in items),
                    net_pnl=sum(trade.realized_net_pnl for trade in items),
                    best_trade=best_trade,
                    worst_trade=worst_trade,
                    exit_reason_counts=dict(exit_counts),
                )
            )
        return sorted(summaries, key=lambda row: (row.symbol, row.strategy_name))

    def _record_sweeps(
        self,
        sweep_lookup: dict[tuple[float, float], SweepStats],
        target_move_bps: float,
        reward_cost_ratio: float,
        expected_net_profit: float,
    ) -> None:
        for sweep in sweep_lookup.values():
            sweep.record_if_passes(
                target_move_bps=target_move_bps,
                reward_cost_ratio=reward_cost_ratio,
                expected_net_profit=expected_net_profit,
                min_expected_net_profit=self.settings.risk.min_expected_net_profit_usd,
            )

    def _core_filters_pass(self, metadata: dict[str, Any]) -> bool:
        return all(str(metadata.get(check, "not_checked")) == "pass" for check in CORE_CHECKS)

    def _expected_net_profit(self, target_move_bps: float) -> float:
        expected_gross_reward = self.diagnostic_notional * max(target_move_bps, 0.0) / 10_000
        estimated_costs = self.diagnostic_notional * self.settings.risk.round_trip_taker_cost_rate
        return expected_gross_reward - estimated_costs

    def _default_diagnostic_notional(self) -> float:
        return max(
            self.settings.risk.min_position_notional,
            self.settings.risk.initial_equity * self.settings.risk.max_position_notional_fraction,
        )

    def _build_strategies(self) -> list[BaseStrategy]:
        edge_config = {
            "min_target_move_bps": self.settings.risk.min_target_move_bps,
            "atr_take_profit_multiplier": self.settings.risk.atr_take_profit_multiplier,
            "atr_stop_loss_multiplier": self.settings.risk.atr_stop_loss_multiplier,
            "min_reward_to_cost_ratio": self.settings.risk.min_reward_to_cost_ratio,
            "round_trip_cost_bps": self.settings.risk.round_trip_taker_cost_bps,
        }
        strategies: list[BaseStrategy] = [
            MomentumStrategy(**edge_config),
            MeanReversionStrategy(**edge_config),
            BreakoutStrategy(**edge_config),
        ]
        if self.settings.trading.enable_scalping_microstructure:
            strategies.append(
                ScalpingMicrostructureStrategy(max_spread_bps=self.settings.risk.max_spread_bps)
            )
        return strategies

    def _finite_number(self, value: Any, default: float = 0.0) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return default
        if pd.isna(number):
            return default
        return number


def load_csv(path: Path, limit: int | None = None) -> pd.DataFrame:
    """Load OHLCV CSV data with either named or first-six ccxt-style columns."""

    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        for alternate in ("time", "datetime", "date"):
            if alternate in df.columns:
                df = df.rename(columns={alternate: "timestamp"})
                break

    if "timestamp" not in df.columns and len(df.columns) >= 6:
        renamed = df.iloc[:, :6].copy()
        renamed.columns = ["timestamp", "open", "high", "low", "close", "volume"]
        df = renamed

    if "timestamp" in df.columns and not pd.api.types.is_numeric_dtype(df["timestamp"]):
        parsed = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df["timestamp"] = parsed.map(
            lambda value: value.timestamp() * 1000 if pd.notna(value) else pd.NA
        )

    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{path} is missing OHLCV columns: {missing}")

    if limit is not None and limit > 0:
        df = df.tail(limit)
    return df[["timestamp", "open", "high", "low", "close", "volume"]].copy()


def load_historical_inputs(
    symbols: tuple[str, ...],
    csv_mappings: list[str],
    data_dir: Path | None,
    timeframe: str,
    limit: int | None,
) -> dict[str, pd.DataFrame]:
    mapping = parse_csv_mappings(csv_mappings)
    data: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        path = mapping.get(symbol)
        if path is None and data_dir is not None:
            path = find_symbol_csv(data_dir, symbol, timeframe)
        if path is None:
            raise FileNotFoundError(
                f"No historical CSV provided for {symbol}. Use --csv {symbol}=path.csv or --data-dir."
            )
        data[symbol] = load_csv(path, limit=limit)
    return data


def parse_csv_mappings(items: list[str]) -> dict[str, Path]:
    mappings: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--csv must use SYMBOL=PATH format, got {item!r}")
        symbol, raw_path = item.split("=", 1)
        mappings[symbol.strip()] = Path(raw_path.strip())
    return mappings


def find_symbol_csv(data_dir: Path, symbol: str, timeframe: str) -> Path | None:
    compact = symbol.replace("/", "")
    underscore = symbol.replace("/", "_")
    dash = symbol.replace("/", "-")
    candidates = [
        f"{compact}.csv",
        f"{underscore}.csv",
        f"{dash}.csv",
        f"{compact}_{timeframe}.csv",
        f"{underscore}_{timeframe}.csv",
        f"{dash}_{timeframe}.csv",
    ]
    for name in candidates:
        path = data_dir / name
        if path.exists():
            return path
    return None


def print_report(result: CalibrationResult) -> None:
    print("Historical Strategy Edge Calibration")
    print("calibration only")
    print("production thresholds unchanged")
    print("candidate_target_move_bps is the strategy target estimate")
    print("expected-only section estimates profit from candidate target and configured costs")
    print("realized simulation section walks future candle high/low paths after WouldTrade candidates")
    print("same-candle stop/target ambiguity is resolved conservatively as stop_loss_hit")
    print("averages are over all directional candidates, not only WouldTrade candidates")
    print(
        f"production_min_target_move_bps={result.production_target_move_bps:.2f} | "
        f"production_min_reward_cost_ratio={result.production_reward_cost_ratio:.2f}x | "
        f"production_min_expected_net_profit=${result.production_min_expected_net_profit:.2f} | "
        f"diagnostic_notional=${result.diagnostic_notional:,.2f}"
    )
    print()
    print("Expected-Only Candidate Funnel")
    print(
        f"{'Symbol':<10} {'Strategy':<18} {'Candles':>8} {'Signals':>8} "
        f"{'CoreOK':>8} {'TargetOK':>8} {'RewardOK':>8} {'NetOK':>8} "
        f"{'WouldTrade':>10} {'AvgCandTgt':>10} {'AvgR/C':>8} {'AvgExpNet':>10}"
    )
    for row in result.rows:
        print(
            f"{row.symbol:<10} {row.strategy_name:<18} {row.candles_tested:>8} "
            f"{row.signals_considered:>8} {row.passing_core_filters:>8} "
            f"{row.passing_target_move_bps:>8} {row.passing_reward_cost_ratio:>8} "
            f"{row.passing_expected_net_profit:>8} {row.would_be_trades:>10} "
            f"{row.average_target_move_bps:>10.2f} {row.average_reward_cost_ratio:>8.2f} "
            f"${row.average_expected_net_profit:>9.2f}"
        )

    print()
    print("Threshold Sweep (expected-only; counts require core filters, target, reward/cost, and expected net)")
    print(
        f"{'Symbol':<10} {'Strategy':<18} {'Target':>8} {'R/C':>8} "
        f"{'WouldTrade':>10} {'AvgExpNet':>10} {'TotalExpNet':>12}"
    )
    for row in result.sweep_rows:
        print(
            f"{row.symbol:<10} {row.strategy_name:<18} {row.min_target_move_bps:>8.2f} "
            f"{row.min_reward_cost_ratio:>8.2f} {row.pass_count:>10} "
            f"${row.average_expected_net_profit:>9.2f} ${row.expected_net_profit_sum:>11.2f}"
        )

    print()
    print("Best Threshold Combinations By Trade Count")
    for line in best_threshold_lines(result.sweep_rows, key="count"):
        print(line)

    print()
    print("Best Threshold Combinations By Expected Net Profit")
    for line in best_threshold_lines(result.sweep_rows, key="net"):
        print(line)

    print()
    print("Realized Exit Simulation")
    print(f"max_hold_candles={result.max_hold_candles}")
    if not result.simulation_rows:
        print("no simulated trades passed production gates")
        return
    print(
        f"{'Symbol':<10} {'Strategy':<18} {'Trades':>7} {'Wins':>6} {'Losses':>7} "
        f"{'WinRate':>8} {'Gross':>11} {'Costs':>11} {'Net':>11} {'AvgNet':>11} "
        f"{'Best':>11} {'Worst':>11} {'ExitReasons':<36}"
    )
    for row in result.simulation_rows:
        print(
            f"{row.symbol:<10} {row.strategy_name:<18} {row.trades:>7} "
            f"{row.wins:>6} {row.losses:>7} {row.win_rate:>7.1f}% "
            f"${row.gross_pnl:>10.2f} ${row.costs:>10.2f} ${row.net_pnl:>10.2f} "
            f"${row.average_net_per_trade:>10.2f} "
            f"{format_trade_net(row.best_trade):>11} {format_trade_net(row.worst_trade):>11} "
            f"{format_exit_reasons(row.exit_reason_counts):<36}"
        )


def format_trade_net(trade: SimulationTrade | None) -> str:
    if trade is None:
        return "n/a"
    return f"${trade.realized_net_pnl:.2f}"


def format_exit_reasons(exit_reason_counts: dict[str, int]) -> str:
    if not exit_reason_counts:
        return "n/a"
    return ", ".join(
        f"{reason}:{count}" for reason, count in sorted(exit_reason_counts.items())
    )


def best_threshold_lines(rows: list[SweepStats], key: str) -> list[str]:
    grouped: dict[tuple[float, float], dict[str, float]] = defaultdict(
        lambda: {"count": 0.0, "net": 0.0}
    )
    for row in rows:
        bucket = grouped[(row.min_target_move_bps, row.min_reward_cost_ratio)]
        bucket["count"] += row.pass_count
        bucket["net"] += row.expected_net_profit_sum

    if key == "count":
        sorted_items = sorted(grouped.items(), key=lambda item: (item[1]["count"], item[1]["net"]), reverse=True)
    else:
        sorted_items = sorted(grouped.items(), key=lambda item: (item[1]["net"], item[1]["count"]), reverse=True)

    if not sorted_items:
        return ["none"]
    return [
        f"target={target:.2f}bps reward_cost={reward:.2f}x "
        f"would_trade={int(values['count'])} expected_net=${values['net']:.2f}"
        for (target, reward), values in sorted_items[:5]
    ]


def parse_float_list(raw: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in raw.split(",") if item.strip())


def run_internal_self_test() -> None:
    """Validate that threshold gates cannot pass below configured thresholds."""

    failing_target = evaluate_thresholds(
        target_move_bps=74.99,
        reward_cost_ratio=3.5,
        expected_net_profit=10.0,
        min_target_move_bps=75.0,
        min_reward_cost_ratio=3.0,
        min_expected_net_profit=1.0,
    )
    assert not failing_target.target_pass
    assert not failing_target.all_pass

    failing_reward = evaluate_thresholds(
        target_move_bps=80.0,
        reward_cost_ratio=2.99,
        expected_net_profit=10.0,
        min_target_move_bps=75.0,
        min_reward_cost_ratio=3.0,
        min_expected_net_profit=1.0,
    )
    assert failing_reward.target_pass
    assert not failing_reward.reward_cost_pass
    assert not failing_reward.all_pass

    failing_net = evaluate_thresholds(
        target_move_bps=80.0,
        reward_cost_ratio=3.1,
        expected_net_profit=0.99,
        min_target_move_bps=75.0,
        min_reward_cost_ratio=3.0,
        min_expected_net_profit=1.0,
    )
    assert failing_net.target_pass
    assert failing_net.reward_cost_pass
    assert not failing_net.expected_net_pass
    assert not failing_net.all_pass

    passing = evaluate_thresholds(
        target_move_bps=75.0,
        reward_cost_ratio=3.0,
        expected_net_profit=1.0,
        min_target_move_bps=75.0,
        min_reward_cost_ratio=3.0,
        min_expected_net_profit=1.0,
    )
    assert passing.all_pass
    validate_threshold_gate(
        passing,
        target_move_bps=75.0,
        reward_cost_ratio=3.0,
        expected_net_profit=1.0,
        min_target_move_bps=75.0,
        min_reward_cost_ratio=3.0,
        min_expected_net_profit=1.0,
        label="self_test",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate strategy edge on historical OHLCV candles.")
    parser.add_argument(
        "--symbols",
        default=",".join(load_settings().trading.symbols),
        help="Comma-separated symbols to calibrate, e.g. BTC/USDT,ETH/USDT.",
    )
    parser.add_argument(
        "--csv",
        action="append",
        default=[],
        help="Historical CSV mapping in SYMBOL=PATH format. Can be repeated.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Directory containing symbol CSVs like BTCUSDT.csv or BTC_USDT_1m.csv.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional number of most recent candles to use from each CSV.",
    )
    parser.add_argument(
        "--target-sweep",
        default="25,50,75,100",
        help="Comma-separated MIN_TARGET_MOVE_BPS sweep values.",
    )
    parser.add_argument(
        "--reward-cost-sweep",
        default="1.5,2.0,3.0",
        help="Comma-separated reward/cost ratio sweep values.",
    )
    parser.add_argument(
        "--diagnostic-notional",
        type=float,
        help="Optional notional used only for calibration expected net profit.",
    )
    parser.add_argument(
        "--max-hold-candles",
        type=int,
        default=60,
        help="Maximum future candles used for calibration exit simulation.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run internal threshold-gate sanity checks and exit.",
    )
    return parser


def main() -> None:
    settings = load_settings()
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.self_test:
        run_internal_self_test()
        print("calibration threshold self-test passed")
        return
    symbols = tuple(symbol.strip() for symbol in args.symbols.split(",") if symbol.strip())
    historical = load_historical_inputs(
        symbols=symbols,
        csv_mappings=args.csv,
        data_dir=args.data_dir,
        timeframe=settings.trading.timeframe,
        limit=args.limit,
    )
    calibrator = HistoricalStrategyCalibrator(
        settings=settings,
        target_sweep=parse_float_list(args.target_sweep),
        reward_cost_sweep=parse_float_list(args.reward_cost_sweep),
        diagnostic_notional=args.diagnostic_notional,
        max_hold_candles=args.max_hold_candles,
    )
    print_report(calibrator.run(historical))


if __name__ == "__main__":
    main()
