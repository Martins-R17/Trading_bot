"""Historical strategy edge calibration.

This module is intentionally separate from the live/paper trading loop. It scans
historical OHLCV candles and reports whether existing strategies can produce
fee-aware candidates under production thresholds and under calibration sweeps.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from itertools import product
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
DEFAULT_REALIZED_TARGET_SWEEP = (75.0, 100.0, 125.0, 150.0, 200.0)
DEFAULT_REALIZED_REWARD_COST_SWEEP = (2.0, 3.0, 4.0)
DEFAULT_REALIZED_MAX_HOLD_SWEEP = (16, 32, 48, 96)
DEFAULT_REALIZED_ATR_TP_SWEEP = (2.0, 3.0, 4.0, 5.0)
DEFAULT_REALIZED_ATR_STOP_SWEEP = (0.75, 1.0, 1.5, 2.0)
DEFAULT_EXHAUSTION_RSI_HIGH = 68.0
DEFAULT_EXHAUSTION_RSI_LOW = 32.0
DEFAULT_EXHAUSTION_CLOSE_POSITION_HIGH = 0.90
DEFAULT_EXHAUSTION_CLOSE_POSITION_LOW = 0.10
DEFAULT_EXHAUSTION_CANDLE_ATR_MULTIPLIER = 1.5
DEFAULT_EXTREME_RSI_HIGH = 72.0
DEFAULT_EXTREME_RSI_LOW = 28.0
DEFAULT_SOFT_RSI_LOW_SHORT = 35.0
DEFAULT_SOFT_CLOSE_POSITION_LOW_SHORT = 0.30
DEFAULT_SOFT_RSI_HIGH_LONG = 65.0
DEFAULT_SOFT_CLOSE_POSITION_HIGH_LONG = 0.70
DEFAULT_REJECT_SOFT_LATE_MOMENTUM = False
DEFAULT_SOFT_RSI_HIGH_LONG_SWEEP = (65.0, 67.0, 69.0)
DEFAULT_SOFT_CLOSE_POSITION_HIGH_LONG_SWEEP = (0.70, 0.80, 0.90)
DEFAULT_SOFT_RSI_LOW_SHORT_SWEEP = (31.0, 33.0, 35.0)
DEFAULT_SOFT_CLOSE_POSITION_LOW_SHORT_SWEEP = (0.10, 0.20, 0.30)
DEFAULT_SUMMARY_LOG_PATH = Path("data/backtest_logs/realized_sweep_summary.jsonl")
ALLOWED_BACKTEST_TIMEFRAMES = ("1m", "5m", "15m")
DEFAULT_BACKTEST_YEARS = 3.0
DEFAULT_SIGNAL_WINDOW_BARS = 500
TIMEFRAME_MINUTES = {"1m": 1, "5m": 5, "15m": 15}
TIMEFRAME_ALIASES = {
    "1min": "1m",
    "5min": "5m",
    "15min": "15m",
}
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
    quality_rejection_counts: dict[str, int] = field(default_factory=dict)
    quality_rejected_losing_count: int = 0

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
    metadata: dict[str, Any] = field(default_factory=dict)


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
    hold_candles_sum: int = 0
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

    @property
    def average_hold_candles(self) -> float:
        if self.trades <= 0:
            return 0.0
        return self.hold_candles_sum / self.trades

    @property
    def stop_loss_hit_rate(self) -> float:
        return self._exit_rate(("stop_loss_hit", "trailing_stop_hit", "breakeven_stop_hit"))

    @property
    def take_profit_hit_rate(self) -> float:
        return self._exit_rate(("take_profit_hit",))

    @property
    def max_horizon_exit_rate(self) -> float:
        return self._exit_rate(("max_horizon_exit",))

    def _exit_rate(self, reasons: tuple[str, ...]) -> float:
        if self.trades <= 0:
            return 0.0
        return sum(self.exit_reason_counts.get(reason, 0) for reason in reasons) / self.trades * 100


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
    calibration_min_expected_net_profit: float
    diagnostic_notional: float
    signal_window_bars: int = DEFAULT_SIGNAL_WINDOW_BARS
    data_profiles: list[dict[str, Any]] = field(default_factory=list)
    max_hold_candles: int = 60
    quality_filter_enabled: bool = False
    trailing_exits_enabled: bool = False
    quality_config: BacktestQualityConfig = field(default_factory=lambda: BacktestQualityConfig())
    rows: list[StrategyCalibrationStats] = field(default_factory=list)
    sweep_rows: list[SweepStats] = field(default_factory=list)
    simulation_rows: list[SimulationSummary] = field(default_factory=list)
    simulation_trades: list[SimulationTrade] = field(default_factory=list)
    quality_rejection_counts: dict[str, int] = field(default_factory=dict)
    quality_rejected_simulations: list[SimulationTrade] = field(default_factory=list)
    quality_rejected_losing_count: int = 0
    accepted_loser_clusters: list["AcceptedLoserCluster"] = field(default_factory=list)


@dataclass(frozen=True)
class RealizedSweepConfig:
    """One calibration-only realized optimization combination."""

    timeframe: str
    min_target_move_bps: float
    min_reward_cost_ratio: float
    max_hold_candles: int
    atr_take_profit_multiplier: float
    atr_stop_loss_multiplier: float
    soft_rsi_high_long: float
    soft_close_position_high_long: float
    soft_rsi_low_short: float
    soft_close_position_low_short: float


@dataclass
class RealizedOptimizationRow:
    """Realized simulation stats for one symbol/strategy/sweep combination."""

    timeframe: str
    symbol: str
    strategy_name: str
    min_target_move_bps: float
    min_reward_cost_ratio: float
    max_hold_candles: int
    atr_take_profit_multiplier: float
    atr_stop_loss_multiplier: float
    soft_rsi_high_long: float
    soft_close_position_high_long: float
    soft_rsi_low_short: float
    soft_close_position_low_short: float
    trades: int = 0
    wins: int = 0
    losses: int = 0
    gross_pnl: float = 0.0
    costs: float = 0.0
    net_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    hold_candles_sum: int = 0
    max_drawdown: float = 0.0
    exit_reason_counts: dict[str, int] = field(default_factory=dict)
    quality_rejection_counts: dict[str, int] = field(default_factory=dict)
    quality_rejected_losing_count: int = 0

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

    @property
    def profit_factor(self) -> float | None:
        if self.trades <= 0:
            return None
        if self.gross_loss <= 0:
            return None if self.gross_profit <= 0 else float("inf")
        return self.gross_profit / self.gross_loss

    @property
    def average_hold_candles(self) -> float:
        if self.trades <= 0:
            return 0.0
        return self.hold_candles_sum / self.trades

    @property
    def stop_loss_hit_rate(self) -> float:
        return self._exit_rate(("stop_loss_hit", "trailing_stop_hit", "breakeven_stop_hit"))

    @property
    def take_profit_hit_rate(self) -> float:
        return self._exit_rate(("take_profit_hit",))

    @property
    def max_horizon_exit_rate(self) -> float:
        return self._exit_rate(("max_horizon_exit",))

    def _exit_rate(self, reasons: tuple[str, ...]) -> float:
        if self.trades <= 0:
            return 0.0
        return sum(self.exit_reason_counts.get(reason, 0) for reason in reasons) / self.trades * 100

    @property
    def quality_rejected_count(self) -> int:
        return sum(self.quality_rejection_counts.values())


@dataclass
class RealizedOptimizationResult:
    """Complete realized optimization report data."""

    production_target_move_bps: float
    production_reward_cost_ratio: float
    production_min_expected_net_profit: float
    calibration_min_expected_net_profit: float
    diagnostic_notional: float
    signal_window_bars: int = DEFAULT_SIGNAL_WINDOW_BARS
    data_profiles: list[dict[str, Any]] = field(default_factory=list)
    quality_filter_enabled: bool = False
    trailing_exits_enabled: bool = False
    quality_config: BacktestQualityConfig = field(default_factory=lambda: BacktestQualityConfig())
    rows: list[RealizedOptimizationRow] = field(default_factory=list)
    losing_examples: list[SimulationTrade] = field(default_factory=list)
    quality_rejection_counts: dict[str, int] = field(default_factory=dict)
    quality_rejected_simulations: list[SimulationTrade] = field(default_factory=list)
    quality_rejected_losing_count: int = 0
    accepted_loser_clusters: list["AcceptedLoserCluster"] = field(default_factory=list)
    momentum_entry_clusters: list["MomentumEntryCluster"] = field(default_factory=list)
    momentum_entry_only_clusters: list["MomentumEntryOnlyCluster"] = field(default_factory=list)


@dataclass(frozen=True)
class BacktestQualityConfig:
    """Calibration-only exhaustion and late-entry thresholds."""

    exhaustion_rsi_high: float = DEFAULT_EXHAUSTION_RSI_HIGH
    exhaustion_rsi_low: float = DEFAULT_EXHAUSTION_RSI_LOW
    exhaustion_close_position_high: float = DEFAULT_EXHAUSTION_CLOSE_POSITION_HIGH
    exhaustion_close_position_low: float = DEFAULT_EXHAUSTION_CLOSE_POSITION_LOW
    exhaustion_candle_atr_multiplier: float = DEFAULT_EXHAUSTION_CANDLE_ATR_MULTIPLIER
    extreme_rsi_high: float = DEFAULT_EXTREME_RSI_HIGH
    extreme_rsi_low: float = DEFAULT_EXTREME_RSI_LOW
    soft_rsi_low_short: float = DEFAULT_SOFT_RSI_LOW_SHORT
    soft_close_position_low_short: float = DEFAULT_SOFT_CLOSE_POSITION_LOW_SHORT
    soft_rsi_high_long: float = DEFAULT_SOFT_RSI_HIGH_LONG
    soft_close_position_high_long: float = DEFAULT_SOFT_CLOSE_POSITION_HIGH_LONG
    reject_soft_late_momentum: bool = DEFAULT_REJECT_SOFT_LATE_MOMENTUM


@dataclass
class AcceptedLoserCluster:
    """Cluster of accepted losing trades for diagnostics only."""

    side: str
    strategy_name: str
    exit_reason: str
    rsi_band: str
    close_position_band: str
    hold_band: str
    soft_label: str
    count: int = 0
    net_pnl: float = 0.0
    rsi_sum: float = 0.0
    rsi_count: int = 0
    close_position_sum: float = 0.0
    close_position_count: int = 0
    hold_sum: int = 0
    stop_loss_count: int = 0

    @property
    def average_rsi(self) -> float:
        if self.rsi_count <= 0:
            return 0.0
        return self.rsi_sum / self.rsi_count

    @property
    def average_close_position(self) -> float:
        if self.close_position_count <= 0:
            return 0.0
        return self.close_position_sum / self.close_position_count

    @property
    def average_hold(self) -> float:
        if self.count <= 0:
            return 0.0
        return self.hold_sum / self.count


@dataclass
class MomentumEntryCluster:
    """Accepted momentum trade cluster for entry-feature diagnostics."""

    side: str
    rsi_band: str
    macd_band: str
    trend_regime: str
    volume_band: str
    atr_bps_band: str
    close_position_band: str
    exit_reason: str
    count: int = 0
    wins: int = 0
    losses: int = 0
    gross_pnl: float = 0.0
    costs: float = 0.0
    net_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    hold_sum: int = 0

    def record(self, trade: SimulationTrade) -> None:
        self.count += 1
        self.gross_pnl += trade.realized_gross_pnl
        self.costs += trade.total_costs
        self.net_pnl += trade.realized_net_pnl
        self.hold_sum += trade.hold_candles
        if trade.realized_net_pnl > 0:
            self.wins += 1
            self.gross_profit += trade.realized_net_pnl
        else:
            self.losses += 1
            self.gross_loss += abs(trade.realized_net_pnl)

    @property
    def win_rate(self) -> float:
        if self.count <= 0:
            return 0.0
        return self.wins / self.count * 100

    @property
    def average_net_per_trade(self) -> float:
        if self.count <= 0:
            return 0.0
        return self.net_pnl / self.count

    @property
    def profit_factor(self) -> float | None:
        if self.count <= 0:
            return None
        if self.gross_loss <= 0:
            return None if self.gross_profit <= 0 else float("inf")
        return self.gross_profit / self.gross_loss

    @property
    def average_hold(self) -> float:
        if self.count <= 0:
            return 0.0
        return self.hold_sum / self.count


@dataclass
class MomentumEntryOnlyCluster:
    """Accepted momentum trade cluster keyed only by entry-time features."""

    side: str
    rsi_band: str
    macd_band: str
    trend_regime: str
    volume_band: str
    atr_bps_band: str
    close_position_band: str
    count: int = 0
    wins: int = 0
    losses: int = 0
    gross_pnl: float = 0.0
    costs: float = 0.0
    net_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    hold_sum: int = 0
    exit_reason_counts: dict[str, int] = field(default_factory=dict)

    def record(self, trade: SimulationTrade) -> None:
        self.count += 1
        self.gross_pnl += trade.realized_gross_pnl
        self.costs += trade.total_costs
        self.net_pnl += trade.realized_net_pnl
        self.hold_sum += trade.hold_candles
        self.exit_reason_counts[trade.exit_reason] = self.exit_reason_counts.get(trade.exit_reason, 0) + 1
        if trade.realized_net_pnl > 0:
            self.wins += 1
            self.gross_profit += trade.realized_net_pnl
        else:
            self.losses += 1
            self.gross_loss += abs(trade.realized_net_pnl)

    @property
    def win_rate(self) -> float:
        if self.count <= 0:
            return 0.0
        return self.wins / self.count * 100

    @property
    def average_net_per_trade(self) -> float:
        if self.count <= 0:
            return 0.0
        return self.net_pnl / self.count

    @property
    def profit_factor(self) -> float | None:
        if self.count <= 0:
            return None
        if self.gross_loss <= 0:
            return None if self.gross_profit <= 0 else float("inf")
        return self.gross_profit / self.gross_loss

    @property
    def average_hold(self) -> float:
        if self.count <= 0:
            return 0.0
        return self.hold_sum / self.count

    @property
    def stop_loss_hit_count(self) -> int:
        return self.exit_reason_counts.get("stop_loss_hit", 0)

    @property
    def take_profit_hit_count(self) -> int:
        return self.exit_reason_counts.get("take_profit_hit", 0)

    @property
    def max_horizon_exit_count(self) -> int:
        return self.exit_reason_counts.get("max_horizon_exit", 0)


@dataclass(frozen=True)
class BacktestQualityDecision:
    """Backtest-only candidate quality verdict."""

    approved: bool
    reason: str
    metadata: dict[str, float | str]


class BacktestQualityFilter:
    """Experimental calibration-only filters from realized backtest failures."""

    def __init__(
        self,
        settings: Settings,
        config: BacktestQualityConfig | None = None,
    ) -> None:
        self.settings = settings
        self.config = config or BacktestQualityConfig()

    def evaluate(
        self,
        strategy_name: str,
        df: pd.DataFrame,
        signal: StrategySignal,
    ) -> BacktestQualityDecision:
        if len(df) < 60:
            return self._reject("quality_filter_insufficient_data", {})

        side = signal.side
        price = self._latest(df, "close", signal.entry_price)
        atr = max(self._latest(df, "atr", price * 0.001), price * 0.0001)
        atr_bps = atr / max(price, 1e-9) * 10_000
        ema_fast = self._latest(df, "ema_fast", price)
        ema_slow = self._latest(df, "ema_slow", price)
        ema_gap_bps = abs(ema_fast - ema_slow) / max(price, 1e-9) * 10_000
        trend_slope_bps = self._trend_slope_bps(df, side, price)
        atr_expansion = self._atr_expansion(df)
        volume_ratio = self._volume_ratio(df)
        close_position = self._close_position(df)
        body_bps = self._body_bps(df, price)
        candle_range_bps = self._candle_range_bps(df, price)
        candle_body_direction = self._candle_body_direction(df)
        macd_hist = self._latest(df, "macd_hist", 0.0)
        previous_macd_hist = self._previous(df, "macd_hist", macd_hist)
        macd_hist_bps = macd_hist / max(price, 1e-9) * 10_000
        rsi = self._latest(df, "rsi", 50.0)
        stop_move_bps = abs(signal.entry_price - float(signal.stop_loss or signal.entry_price)) / max(signal.entry_price, 1e-9) * 10_000
        target_move_bps = abs(float(signal.take_profit or signal.entry_price) - signal.entry_price) / max(signal.entry_price, 1e-9) * 10_000
        target_stop_ratio = target_move_bps / max(stop_move_bps, 1e-9)
        stop_atr_multiple = stop_move_bps / max(atr_bps, 1e-9)
        body_atr_multiple = body_bps / max(atr_bps, 1e-9)
        range_atr_multiple = candle_range_bps / max(atr_bps, 1e-9)
        common_metadata = {
            "backtest_quality_filter": "checked",
            "backtest_quality_reason": "",
            "exhaustion_rsi_high": self.config.exhaustion_rsi_high,
            "exhaustion_rsi_low": self.config.exhaustion_rsi_low,
            "exhaustion_close_position_high": self.config.exhaustion_close_position_high,
            "exhaustion_close_position_low": self.config.exhaustion_close_position_low,
            "exhaustion_candle_atr_multiplier": self.config.exhaustion_candle_atr_multiplier,
            "extreme_rsi_high": self.config.extreme_rsi_high,
            "extreme_rsi_low": self.config.extreme_rsi_low,
            "soft_rsi_low_short": self.config.soft_rsi_low_short,
            "soft_close_position_low_short": self.config.soft_close_position_low_short,
            "soft_rsi_high_long": self.config.soft_rsi_high_long,
            "soft_close_position_high_long": self.config.soft_close_position_high_long,
            "reject_soft_late_momentum": "enabled" if self.config.reject_soft_late_momentum else "disabled",
            "ema_gap_bps": ema_gap_bps,
            "trend_slope_bps": trend_slope_bps,
            "atr_expansion": atr_expansion,
            "volume_ratio": volume_ratio,
            "close_position": close_position,
            "body_bps": body_bps,
            "candle_range_bps": candle_range_bps,
            "candle_body_direction": candle_body_direction,
            "body_atr_multiple": body_atr_multiple,
            "range_atr_multiple": range_atr_multiple,
            "macd_hist_bps": macd_hist_bps,
            "rsi": rsi,
            "stop_move_bps": stop_move_bps,
            "target_stop_ratio": target_stop_ratio,
            "stop_atr_multiple": stop_atr_multiple,
        }

        late_long = (
            side == Side.BUY
            and rsi >= self.config.exhaustion_rsi_high
            and close_position >= self.config.exhaustion_close_position_high
        )
        late_short = (
            side == Side.SELL
            and rsi <= self.config.exhaustion_rsi_low
            and close_position <= self.config.exhaustion_close_position_low
        )
        large_candle = (
            body_atr_multiple >= self.config.exhaustion_candle_atr_multiplier
            or range_atr_multiple >= self.config.exhaustion_candle_atr_multiplier
        )
        trend_strategy = strategy_name in {"momentum", "breakout"}
        if trend_strategy and side == Side.BUY and rsi > self.config.extreme_rsi_high:
            return self._reject("extreme_rsi_long", common_metadata)
        if trend_strategy and side == Side.SELL and rsi < self.config.extreme_rsi_low:
            return self._reject("extreme_rsi_short", common_metadata)
        if self.config.reject_soft_late_momentum and strategy_name == "momentum":
            if (
                side == Side.BUY
                and rsi >= self.config.soft_rsi_high_long
                and close_position >= self.config.soft_close_position_high_long
            ):
                return self._reject("rejected_soft_late_long", common_metadata)
            if (
                side == Side.SELL
                and rsi <= self.config.soft_rsi_low_short
                and close_position <= self.config.soft_close_position_low_short
            ):
                return self._reject("rejected_soft_late_short", common_metadata)
        if late_long and large_candle and candle_body_direction > 0:
            return self._reject("exhausted_long", common_metadata)
        if late_short and large_candle and candle_body_direction < 0:
            return self._reject("exhausted_short", common_metadata)
        if late_long or late_short:
            return self._reject("late_entry", common_metadata)
        if large_candle:
            if side == Side.BUY and candle_body_direction > 0:
                return self._reject("exhausted_long", common_metadata)
            if side == Side.SELL and candle_body_direction < 0:
                return self._reject("exhausted_short", common_metadata)
            return self._reject("late_entry", common_metadata)
        if stop_atr_multiple < 0.75:
            return self._reject("stop_too_tight_for_atr", common_metadata)
        if target_stop_ratio < max(1.35, self.settings.risk.min_reward_risk_ratio):
            return self._reject("target_not_large_enough_vs_stop", common_metadata)
        if target_move_bps < stop_move_bps + self.settings.risk.round_trip_taker_cost_bps * 0.75:
            return self._reject("target_not_large_enough_after_costs", common_metadata)

        if strategy_name in {"momentum", "breakout"}:
            decision = self._trend_quality_decision(
                side=side,
                ema_fast=ema_fast,
                ema_slow=ema_slow,
                ema_gap_bps=ema_gap_bps,
                trend_slope_bps=trend_slope_bps,
                atr_expansion=atr_expansion,
                volume_ratio=volume_ratio,
                close_position=close_position,
                macd_hist_bps=macd_hist_bps,
                rsi=rsi,
                common_metadata=common_metadata,
            )
            if not decision.approved:
                return decision
        elif strategy_name == "mean_reversion":
            decision = self._range_quality_decision(
                side=side,
                ema_gap_bps=ema_gap_bps,
                trend_slope_bps=trend_slope_bps,
                atr_expansion=atr_expansion,
                volume_ratio=volume_ratio,
                close_position=close_position,
                macd_hist=macd_hist,
                previous_macd_hist=previous_macd_hist,
                rsi=rsi,
                common_metadata=common_metadata,
            )
            if not decision.approved:
                return decision

        return BacktestQualityDecision(
            approved=True,
            reason="pass",
            metadata={
                **common_metadata,
                "backtest_quality_filter": "pass",
                "backtest_quality_reason": "pass",
            },
        )

    def _trend_quality_decision(
        self,
        side: Side,
        ema_fast: float,
        ema_slow: float,
        ema_gap_bps: float,
        trend_slope_bps: float,
        atr_expansion: float,
        volume_ratio: float,
        close_position: float,
        macd_hist_bps: float,
        rsi: float,
        common_metadata: dict[str, float | str],
    ) -> BacktestQualityDecision:
        trend_aligned = (
            (side == Side.BUY and ema_fast > ema_slow and trend_slope_bps > 18)
            or (side == Side.SELL and ema_fast < ema_slow and trend_slope_bps > 18)
        )
        if not trend_aligned or ema_gap_bps < 8:
            return self._reject("trend_regime_not_clear", common_metadata)
        if atr_expansion < 1.03:
            return self._reject("volatility_expansion_not_confirmed", common_metadata)
        if volume_ratio < 1.12:
            return self._reject("volume_expansion_not_confirmed", common_metadata)
        if side == Side.BUY and close_position < 0.62:
            return self._reject("bullish_close_not_confirmed", common_metadata)
        if side == Side.SELL and close_position > 0.38:
            return self._reject("bearish_close_not_confirmed", common_metadata)
        if side == Side.BUY and (macd_hist_bps <= 0.0 or rsi < 54 or rsi > 76):
            return self._reject("rsi_macd_conflict", common_metadata)
        if side == Side.SELL and (macd_hist_bps >= 0.0 or rsi > 46 or rsi < 24):
            return self._reject("rsi_macd_conflict", common_metadata)
        return BacktestQualityDecision(True, "pass", common_metadata)

    def _range_quality_decision(
        self,
        side: Side,
        ema_gap_bps: float,
        trend_slope_bps: float,
        atr_expansion: float,
        volume_ratio: float,
        close_position: float,
        macd_hist: float,
        previous_macd_hist: float,
        rsi: float,
        common_metadata: dict[str, float | str],
    ) -> BacktestQualityDecision:
        if ema_gap_bps > 28 or abs(trend_slope_bps) > 55:
            return self._reject("range_regime_not_confirmed", common_metadata)
        if atr_expansion > 1.35:
            return self._reject("range_break_risk_too_high", common_metadata)
        if volume_ratio > 2.4:
            return self._reject("volume_spike_against_reversion", common_metadata)
        if side == Side.BUY and close_position < 0.28:
            return self._reject("reversion_close_not_confirmed", common_metadata)
        if side == Side.SELL and close_position > 0.72:
            return self._reject("reversion_close_not_confirmed", common_metadata)
        if side == Side.BUY and (rsi > 42 or macd_hist < previous_macd_hist):
            return self._reject("rsi_macd_conflict", common_metadata)
        if side == Side.SELL and (rsi < 58 or macd_hist > previous_macd_hist):
            return self._reject("rsi_macd_conflict", common_metadata)
        return BacktestQualityDecision(True, "pass", common_metadata)

    def _reject(
        self,
        reason: str,
        metadata: dict[str, float | str],
    ) -> BacktestQualityDecision:
        return BacktestQualityDecision(
            approved=False,
            reason=reason,
            metadata={
                **metadata,
                "backtest_quality_filter": "reject",
                "backtest_quality_reason": reason,
            },
        )

    def _trend_slope_bps(self, df: pd.DataFrame, side: Side, price: float) -> float:
        if len(df) < 21:
            return 0.0
        previous = self._float(df["close"].iloc[-21], price)
        raw_slope = (price - previous) / max(price, 1e-9) * 10_000
        return raw_slope * side.direction

    def _atr_expansion(self, df: pd.DataFrame) -> float:
        latest = self._latest(df, "atr", 0.0)
        if "atr" not in df.columns or len(df) < 50:
            return 1.0
        baseline = self._float(df["atr"].tail(50).mean(), latest)
        return latest / max(baseline, 1e-9)

    def _volume_ratio(self, df: pd.DataFrame) -> float:
        latest = self._latest(df, "volume", 0.0)
        if len(df) < 31:
            return 1.0
        baseline = self._float(df["volume"].iloc[-31:-1].mean(), latest)
        return latest / max(baseline, 1e-9)

    def _close_position(self, df: pd.DataFrame) -> float:
        high = self._latest(df, "high", 0.0)
        low = self._latest(df, "low", high)
        close = self._latest(df, "close", low)
        return (close - low) / max(high - low, 1e-9)

    def _body_bps(self, df: pd.DataFrame, price: float) -> float:
        return abs(self._latest(df, "close", price) - self._latest(df, "open", price)) / max(price, 1e-9) * 10_000

    def _candle_range_bps(self, df: pd.DataFrame, price: float) -> float:
        return (self._latest(df, "high", price) - self._latest(df, "low", price)) / max(price, 1e-9) * 10_000

    def _candle_body_direction(self, df: pd.DataFrame) -> float:
        close = self._latest(df, "close", 0.0)
        open_price = self._latest(df, "open", close)
        if close > open_price:
            return 1.0
        if close < open_price:
            return -1.0
        return 0.0

    def _latest(self, df: pd.DataFrame, column: str, default: float) -> float:
        if column not in df.columns or len(df) == 0:
            return default
        return self._float(df[column].iloc[-1], default)

    def _previous(self, df: pd.DataFrame, column: str, default: float) -> float:
        if column not in df.columns or len(df) < 2:
            return default
        return self._float(df[column].iloc[-2], default)

    def _float(self, value: Any, default: float = 0.0) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return default
        if pd.isna(number):
            return default
        return number


class HistoricalStrategyCalibrator:
    """Fast strategy-edge calibration over historical candles."""

    def __init__(
        self,
        settings: Settings,
        target_sweep: tuple[float, ...] = DEFAULT_TARGET_SWEEP,
        reward_cost_sweep: tuple[float, ...] = DEFAULT_REWARD_COST_SWEEP,
        diagnostic_notional: float | None = None,
        calibration_min_expected_net_profit: float | None = None,
        timeframe: str | None = None,
        signal_window_bars: int = DEFAULT_SIGNAL_WINDOW_BARS,
        max_hold_candles: int = 60,
        quality_filter_enabled: bool = False,
        trailing_exits_enabled: bool = False,
        breakeven_trigger_r: float = 1.0,
        trailing_atr_multiplier: float = 1.0,
        quality_config: BacktestQualityConfig | None = None,
    ) -> None:
        self.settings = settings
        self.target_sweep = target_sweep
        self.reward_cost_sweep = reward_cost_sweep
        self.diagnostic_notional = diagnostic_notional or self._default_diagnostic_notional()
        self.timeframe = timeframe or settings.trading.timeframe
        self.signal_window_bars = max(100, int(signal_window_bars))
        self.calibration_min_expected_net_profit = max(
            0.0,
            float(
                calibration_min_expected_net_profit
                if calibration_min_expected_net_profit is not None
                else self.settings.risk.calibration_min_expected_net_profit_usd
            ),
        )
        self.max_hold_candles = max(1, int(max_hold_candles))
        self.quality_filter_enabled = quality_filter_enabled
        self.quality_config = quality_config or BacktestQualityConfig()
        self.quality_filter = BacktestQualityFilter(settings, self.quality_config)
        self.trailing_exits_enabled = trailing_exits_enabled
        self.breakeven_trigger_r = max(float(breakeven_trigger_r), 0.0)
        self.trailing_atr_multiplier = max(float(trailing_atr_multiplier), 0.0)

    def run(self, historical_by_symbol: dict[str, pd.DataFrame]) -> CalibrationResult:
        rows: list[StrategyCalibrationStats] = []
        sweep_rows: list[SweepStats] = []
        simulation_trades: list[SimulationTrade] = []
        quality_rejection_counts: Counter[str] = Counter()
        quality_rejected_simulations: list[SimulationTrade] = []
        quality_rejected_losing_count = 0

        for symbol, raw_df in historical_by_symbol.items():
            df = DataPreprocessor.add_features(DataPreprocessor.normalize_ohlcv(raw_df))
            for strategy in self._build_strategies():
                stats, sweeps, trades, rejections, rejected_trades, rejected_losing_count = self._calibrate_strategy(symbol, df, strategy)
                rows.append(stats)
                sweep_rows.extend(sweeps)
                simulation_trades.extend(trades)
                quality_rejection_counts.update(rejections)
                quality_rejected_simulations.extend(rejected_trades)
                quality_rejected_losing_count += rejected_losing_count

        return CalibrationResult(
            production_target_move_bps=self.settings.risk.min_target_move_bps,
            production_reward_cost_ratio=self.settings.risk.min_reward_to_cost_ratio,
            production_min_expected_net_profit=self.settings.risk.min_expected_net_profit_usd,
            calibration_min_expected_net_profit=self.calibration_min_expected_net_profit,
            diagnostic_notional=self.diagnostic_notional,
            signal_window_bars=self.signal_window_bars,
            data_profiles=build_data_profiles(historical_by_symbol, self.timeframe),
            max_hold_candles=self.max_hold_candles,
            quality_filter_enabled=self.quality_filter_enabled,
            trailing_exits_enabled=self.trailing_exits_enabled,
            quality_config=self.quality_config,
            rows=rows,
            sweep_rows=sweep_rows,
            simulation_rows=self._summarize_simulation_trades(simulation_trades),
            simulation_trades=simulation_trades,
            quality_rejection_counts=dict(quality_rejection_counts),
            quality_rejected_simulations=quality_rejected_simulations,
            quality_rejected_losing_count=quality_rejected_losing_count,
            accepted_loser_clusters=build_accepted_loser_clusters(simulation_trades, self.quality_config),
        )

    def _calibrate_strategy(
        self,
        symbol: str,
        df: pd.DataFrame,
        strategy: BaseStrategy,
    ) -> tuple[
        StrategyCalibrationStats,
        list[SweepStats],
        list[SimulationTrade],
        Counter[str],
        list[SimulationTrade],
        int,
    ]:
        stats = StrategyCalibrationStats(symbol=symbol, strategy_name=strategy.name)
        simulation_trades: list[SimulationTrade] = []
        quality_rejections: Counter[str] = Counter()
        quality_rejected_simulations: list[SimulationTrade] = []
        quality_rejected_losing_count = 0
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
            window_start = max(0, index + 1 - self.signal_window_bars)
            window = df.iloc[window_start : index + 1]
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

            calibration_gate = evaluate_thresholds(
                target_move_bps=target_move_bps,
                reward_cost_ratio=reward_cost_ratio,
                expected_net_profit=expected_net_profit,
                min_target_move_bps=self.settings.risk.min_target_move_bps,
                min_reward_cost_ratio=self.settings.risk.min_reward_to_cost_ratio,
                min_expected_net_profit=self.calibration_min_expected_net_profit,
            )
            validate_threshold_gate(
                calibration_gate,
                target_move_bps=target_move_bps,
                reward_cost_ratio=reward_cost_ratio,
                expected_net_profit=expected_net_profit,
                min_target_move_bps=self.settings.risk.min_target_move_bps,
                min_reward_cost_ratio=self.settings.risk.min_reward_to_cost_ratio,
                min_expected_net_profit=self.calibration_min_expected_net_profit,
                label=f"{symbol}:{strategy.name}:calibration",
            )
            if calibration_gate.target_pass:
                stats.passing_target_move_bps += 1
            if calibration_gate.reward_cost_pass:
                stats.passing_reward_cost_ratio += 1
            if calibration_gate.expected_net_pass:
                stats.passing_expected_net_profit += 1
            backtest_quality_passes = True
            if calibration_gate.all_pass and self._is_simulatable_signal(signal):
                if self.quality_filter_enabled:
                    quality_decision = self.quality_filter.evaluate(strategy.name, window, signal)
                    if not quality_decision.approved:
                        backtest_quality_passes = False
                        signal.metadata.update(quality_decision.metadata)
                        quality_rejections[quality_decision.reason] += 1
                        rejected_simulation = self._simulate_exit(
                            symbol=symbol,
                            strategy_name=strategy.name,
                            df=df,
                            entry_index=index,
                            signal=signal,
                        )
                        if rejected_simulation.realized_net_pnl < 0:
                            quality_rejected_losing_count += 1
                            if len(quality_rejected_simulations) < 8:
                                quality_rejected_simulations.append(rejected_simulation)
                    else:
                        signal.metadata.update(quality_decision.metadata)
                if backtest_quality_passes:
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

        stats.quality_rejection_counts = dict(quality_rejections)
        stats.quality_rejected_losing_count = quality_rejected_losing_count
        return (
            stats,
            list(sweep_lookup.values()),
            simulation_trades,
            quality_rejections,
            quality_rejected_simulations,
            quality_rejected_losing_count,
        )

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
        entry_atr = self._finite_number(signal.metadata.get("atr"), 0.0)
        if entry_atr <= 0 and "atr" in df.columns:
            entry_atr = self._finite_number(df["atr"].iloc[entry_index], 0.0)
        dynamic_stop = stop_loss
        dynamic_stop_reason = "stop_loss_hit"
        risk_per_unit = abs(entry_price - stop_loss)

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
                    stop_hit = low <= dynamic_stop
                    target_hit = high >= take_profit
                else:
                    stop_hit = high >= dynamic_stop
                    target_hit = low <= take_profit

                if stop_hit:
                    exit_price = dynamic_stop
                    exit_reason = dynamic_stop_reason
                elif target_hit:
                    exit_price = take_profit
                    exit_reason = "take_profit_hit"
                else:
                    if self.trailing_exits_enabled and risk_per_unit > 0:
                        dynamic_stop, dynamic_stop_reason = self._updated_dynamic_stop(
                            side=signal.side,
                            entry_price=entry_price,
                            current_stop=dynamic_stop,
                            current_reason=dynamic_stop_reason,
                            high=high,
                            low=low,
                            take_profit=take_profit,
                            atr=entry_atr,
                            risk_per_unit=risk_per_unit,
                        )
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
            metadata=dict(signal.metadata),
        )

    def _updated_dynamic_stop(
        self,
        side: Side,
        entry_price: float,
        current_stop: float,
        current_reason: str,
        high: float,
        low: float,
        take_profit: float,
        atr: float,
        risk_per_unit: float,
    ) -> tuple[float, str]:
        if atr <= 0:
            return current_stop, current_reason

        updated_stop = current_stop
        updated_reason = current_reason
        if side == Side.BUY:
            favorable_move = high - entry_price
            if favorable_move >= risk_per_unit * self.breakeven_trigger_r and entry_price > updated_stop:
                updated_stop = entry_price
                updated_reason = "breakeven_stop_hit"
            trailing_stop = high - atr * self.trailing_atr_multiplier
            if trailing_stop > updated_stop and trailing_stop < take_profit:
                updated_stop = trailing_stop
                updated_reason = "trailing_stop_hit"
        elif side == Side.SELL:
            favorable_move = entry_price - low
            if favorable_move >= risk_per_unit * self.breakeven_trigger_r and entry_price < updated_stop:
                updated_stop = entry_price
                updated_reason = "breakeven_stop_hit"
            trailing_stop = low + atr * self.trailing_atr_multiplier
            if trailing_stop < updated_stop and trailing_stop > take_profit:
                updated_stop = trailing_stop
                updated_reason = "trailing_stop_hit"
        return updated_stop, updated_reason

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
                    hold_candles_sum=sum(trade.hold_candles for trade in items),
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
                min_expected_net_profit=self.calibration_min_expected_net_profit,
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
            self.settings.risk.diagnostic_notional,
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


class RealizedSweepOptimizer:
    """Calibration-only realized sweep over strategy edge and exit settings."""

    def __init__(
        self,
        settings: Settings,
        timeframe: str,
        target_sweep: tuple[float, ...],
        reward_cost_sweep: tuple[float, ...],
        max_hold_sweep: tuple[int, ...],
        atr_tp_sweep: tuple[float, ...],
        atr_stop_sweep: tuple[float, ...],
        signal_window_bars: int = DEFAULT_SIGNAL_WINDOW_BARS,
        soft_rsi_high_long_sweep: tuple[float, ...] | None = None,
        soft_close_position_high_long_sweep: tuple[float, ...] | None = None,
        soft_rsi_low_short_sweep: tuple[float, ...] | None = None,
        soft_close_position_low_short_sweep: tuple[float, ...] | None = None,
        diagnostic_notional: float | None = None,
        calibration_min_expected_net_profit: float | None = None,
        quality_filter_enabled: bool = True,
        trailing_exits_enabled: bool = False,
        breakeven_trigger_r: float = 1.0,
        trailing_atr_multiplier: float = 1.0,
        quality_config: BacktestQualityConfig | None = None,
    ) -> None:
        self.settings = settings
        self.timeframe = timeframe
        self.target_sweep = target_sweep
        self.reward_cost_sweep = reward_cost_sweep
        self.max_hold_sweep = max_hold_sweep
        self.atr_tp_sweep = atr_tp_sweep
        self.atr_stop_sweep = atr_stop_sweep
        self.signal_window_bars = max(100, int(signal_window_bars))
        self.diagnostic_notional = diagnostic_notional
        self.calibration_min_expected_net_profit = calibration_min_expected_net_profit
        self.quality_filter_enabled = quality_filter_enabled
        self.trailing_exits_enabled = trailing_exits_enabled
        self.breakeven_trigger_r = breakeven_trigger_r
        self.trailing_atr_multiplier = trailing_atr_multiplier
        self.quality_config = quality_config or BacktestQualityConfig()
        self.soft_rsi_high_long_sweep = soft_rsi_high_long_sweep or (self.quality_config.soft_rsi_high_long,)
        self.soft_close_position_high_long_sweep = soft_close_position_high_long_sweep or (self.quality_config.soft_close_position_high_long,)
        self.soft_rsi_low_short_sweep = soft_rsi_low_short_sweep or (self.quality_config.soft_rsi_low_short,)
        self.soft_close_position_low_short_sweep = soft_close_position_low_short_sweep or (self.quality_config.soft_close_position_low_short,)

    def run(self, historical_by_symbol: dict[str, pd.DataFrame]) -> RealizedOptimizationResult:
        rows: list[RealizedOptimizationRow] = []
        all_trades: list[SimulationTrade] = []
        quality_rejection_counts: Counter[str] = Counter()
        quality_rejected_simulations: list[SimulationTrade] = []
        quality_rejected_losing_count = 0
        for config in self._configs():
            calibration_settings = self._settings_for_config(config)
            config_quality = self._quality_config_for_config(config)
            calibrator = HistoricalStrategyCalibrator(
                settings=calibration_settings,
                target_sweep=(config.min_target_move_bps,),
                reward_cost_sweep=(config.min_reward_cost_ratio,),
                diagnostic_notional=self.diagnostic_notional,
                calibration_min_expected_net_profit=self.calibration_min_expected_net_profit,
                timeframe=config.timeframe,
                signal_window_bars=self.signal_window_bars,
                max_hold_candles=config.max_hold_candles,
                quality_filter_enabled=self.quality_filter_enabled,
                trailing_exits_enabled=self.trailing_exits_enabled,
                breakeven_trigger_r=self.breakeven_trigger_r,
                trailing_atr_multiplier=self.trailing_atr_multiplier,
                quality_config=config_quality,
            )
            result = calibrator.run(historical_by_symbol)
            all_trades.extend(result.simulation_trades)
            quality_rejection_counts.update(result.quality_rejection_counts)
            quality_rejected_losing_count += result.quality_rejected_losing_count
            for rejected_simulation in result.quality_rejected_simulations:
                if len(quality_rejected_simulations) < 8:
                    quality_rejected_simulations.append(rejected_simulation)
            trades_by_key = self._group_trades(result.simulation_trades)
            for stats in result.rows:
                trades = trades_by_key.get((stats.symbol, stats.strategy_name), [])
                rows.append(self._row_from_trades(config, stats, trades))

        diagnostic_notional = (
            self.diagnostic_notional
            if self.diagnostic_notional is not None
            else HistoricalStrategyCalibrator(self.settings)._default_diagnostic_notional()
        )
        calibration_min_expected_net_profit = (
            self.calibration_min_expected_net_profit
            if self.calibration_min_expected_net_profit is not None
            else self.settings.risk.calibration_min_expected_net_profit_usd
        )
        return RealizedOptimizationResult(
            production_target_move_bps=self.settings.risk.min_target_move_bps,
            production_reward_cost_ratio=self.settings.risk.min_reward_to_cost_ratio,
            production_min_expected_net_profit=self.settings.risk.min_expected_net_profit_usd,
            calibration_min_expected_net_profit=calibration_min_expected_net_profit,
            diagnostic_notional=diagnostic_notional,
            signal_window_bars=self.signal_window_bars,
            data_profiles=build_data_profiles(historical_by_symbol, self.timeframe),
            quality_filter_enabled=self.quality_filter_enabled,
            trailing_exits_enabled=self.trailing_exits_enabled,
            quality_config=self.quality_config,
            rows=rows,
            losing_examples=self._losing_examples(all_trades),
            quality_rejection_counts=dict(quality_rejection_counts),
            quality_rejected_simulations=quality_rejected_simulations,
            quality_rejected_losing_count=quality_rejected_losing_count,
            accepted_loser_clusters=build_accepted_loser_clusters(all_trades, self.quality_config),
            momentum_entry_clusters=build_momentum_entry_clusters(all_trades),
            momentum_entry_only_clusters=build_momentum_entry_only_clusters(all_trades),
        )

    def _configs(self) -> list[RealizedSweepConfig]:
        return [
            RealizedSweepConfig(
                timeframe=self.timeframe,
                min_target_move_bps=float(target),
                min_reward_cost_ratio=float(reward_cost),
                max_hold_candles=max(1, int(max_hold)),
                atr_take_profit_multiplier=float(atr_tp),
                atr_stop_loss_multiplier=float(atr_stop),
                soft_rsi_high_long=float(soft_rsi_high_long),
                soft_close_position_high_long=float(soft_close_position_high_long),
                soft_rsi_low_short=float(soft_rsi_low_short),
                soft_close_position_low_short=float(soft_close_position_low_short),
            )
            for (
                target,
                reward_cost,
                max_hold,
                atr_tp,
                atr_stop,
                soft_rsi_high_long,
                soft_close_position_high_long,
                soft_rsi_low_short,
                soft_close_position_low_short,
            ) in product(
                self.target_sweep,
                self.reward_cost_sweep,
                self.max_hold_sweep,
                self.atr_tp_sweep,
                self.atr_stop_sweep,
                self.soft_rsi_high_long_sweep,
                self.soft_close_position_high_long_sweep,
                self.soft_rsi_low_short_sweep,
                self.soft_close_position_low_short_sweep,
            )
        ]

    def _quality_config_for_config(self, config: RealizedSweepConfig) -> BacktestQualityConfig:
        return replace(
            self.quality_config,
            soft_rsi_high_long=config.soft_rsi_high_long,
            soft_close_position_high_long=config.soft_close_position_high_long,
            soft_rsi_low_short=config.soft_rsi_low_short,
            soft_close_position_low_short=config.soft_close_position_low_short,
        )

    def _settings_for_config(self, config: RealizedSweepConfig) -> Settings:
        risk = replace(
            self.settings.risk,
            min_target_move_bps=config.min_target_move_bps,
            min_reward_to_cost_ratio=config.min_reward_cost_ratio,
            min_reward_cost_multiple=config.min_reward_cost_ratio,
            atr_take_profit_multiplier=config.atr_take_profit_multiplier,
            atr_stop_loss_multiplier=config.atr_stop_loss_multiplier,
        )
        return replace(self.settings, risk=risk)

    def _group_trades(
        self,
        trades: list[SimulationTrade],
    ) -> dict[tuple[str, str], list[SimulationTrade]]:
        grouped: dict[tuple[str, str], list[SimulationTrade]] = defaultdict(list)
        for trade in trades:
            grouped[(trade.symbol, trade.strategy_name)].append(trade)
        return grouped

    def _row_from_trades(
        self,
        config: RealizedSweepConfig,
        stats: StrategyCalibrationStats,
        trades: list[SimulationTrade],
    ) -> RealizedOptimizationRow:
        sorted_trades = sorted(trades, key=lambda trade: (trade.entry_index, trade.exit_timestamp))
        exit_counts = Counter(trade.exit_reason for trade in sorted_trades)
        net_values = [trade.realized_net_pnl for trade in sorted_trades]
        gross_profit = sum(value for value in net_values if value > 0)
        gross_loss = abs(sum(value for value in net_values if value < 0))
        return RealizedOptimizationRow(
            timeframe=config.timeframe,
            symbol=stats.symbol,
            strategy_name=stats.strategy_name,
            min_target_move_bps=config.min_target_move_bps,
            min_reward_cost_ratio=config.min_reward_cost_ratio,
            max_hold_candles=config.max_hold_candles,
            atr_take_profit_multiplier=config.atr_take_profit_multiplier,
            atr_stop_loss_multiplier=config.atr_stop_loss_multiplier,
            soft_rsi_high_long=config.soft_rsi_high_long,
            soft_close_position_high_long=config.soft_close_position_high_long,
            soft_rsi_low_short=config.soft_rsi_low_short,
            soft_close_position_low_short=config.soft_close_position_low_short,
            trades=len(sorted_trades),
            wins=sum(1 for value in net_values if value > 0),
            losses=sum(1 for value in net_values if value <= 0),
            gross_pnl=sum(trade.realized_gross_pnl for trade in sorted_trades),
            costs=sum(trade.total_costs for trade in sorted_trades),
            net_pnl=sum(net_values),
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            hold_candles_sum=sum(trade.hold_candles for trade in sorted_trades),
            max_drawdown=self._max_drawdown(net_values),
            exit_reason_counts=dict(exit_counts),
            quality_rejection_counts=stats.quality_rejection_counts,
            quality_rejected_losing_count=stats.quality_rejected_losing_count,
        )

    def _max_drawdown(self, net_values: list[float]) -> float:
        equity = 0.0
        peak = 0.0
        max_drawdown = 0.0
        for value in net_values:
            equity += value
            peak = max(peak, equity)
            max_drawdown = max(max_drawdown, peak - equity)
        return max_drawdown

    def _losing_examples(self, trades: list[SimulationTrade]) -> list[SimulationTrade]:
        return sorted(
            [trade for trade in trades if trade.realized_net_pnl < 0],
            key=lambda trade: trade.realized_net_pnl,
        )[:8]


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


def build_data_profiles(
    historical_by_symbol: dict[str, pd.DataFrame],
    timeframe: str,
) -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    for symbol, df in sorted(historical_by_symbol.items()):
        candles = int(len(df))
        first_timestamp = timestamp_value(df["timestamp"].iloc[0]) if candles > 0 and "timestamp" in df.columns else None
        last_timestamp = timestamp_value(df["timestamp"].iloc[-1]) if candles > 0 and "timestamp" in df.columns else None
        profiles.append(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "candles": candles,
                "first_timestamp": first_timestamp,
                "last_timestamp": last_timestamp,
                "start_utc": timestamp_to_utc(first_timestamp),
                "end_utc": timestamp_to_utc(last_timestamp),
                "approx_days": approximate_days(first_timestamp, last_timestamp),
            }
        )
    return profiles


def timestamp_value(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    return number


def timestamp_to_utc(value: float | None) -> str:
    if value is None:
        return "n/a"
    seconds = value / 1000 if abs(value) > 10_000_000_000 else value
    try:
        return datetime.fromtimestamp(seconds, tz=timezone.utc).isoformat()
    except (OSError, OverflowError, ValueError):
        return "n/a"


def approximate_days(start: float | None, end: float | None) -> float | None:
    if start is None or end is None:
        return None
    scale = 1000 if abs(start) > 10_000_000_000 or abs(end) > 10_000_000_000 else 1
    days = (end - start) / scale / 86_400
    if days < 0:
        return None
    return days


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
    print("NetOK/WouldTrade use calibration_min_expected_net_profit; production min remains unchanged")
    print(f"signal_window_bars={result.signal_window_bars}")
    print_data_profiles(result.data_profiles)
    print(f"backtest_quality_filter={'enabled' if result.quality_filter_enabled else 'disabled'}")
    print(f"backtest_trailing_exits={'enabled' if result.trailing_exits_enabled else 'disabled'}")
    print_quality_config(result.quality_config)
    print("averages are over all directional candidates, not only WouldTrade candidates")
    print(
        f"production_min_target_move_bps={result.production_target_move_bps:.2f} | "
        f"production_min_reward_cost_ratio={result.production_reward_cost_ratio:.2f}x | "
        f"production_min_expected_net_profit=${result.production_min_expected_net_profit:.2f} | "
        f"calibration_min_expected_net_profit=${result.calibration_min_expected_net_profit:.2f} | "
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
        print_quality_rejection_diagnostics(
            result.quality_rejection_counts,
            result.quality_rejected_simulations,
            result.quality_rejected_losing_count,
        )
        return
    print(
        f"{'Symbol':<10} {'Strategy':<18} {'Trades':>7} {'Wins':>6} {'Losses':>7} "
        f"{'WinRate':>8} {'Gross':>11} {'Costs':>11} {'Net':>11} {'AvgNet':>11} "
        f"{'Stop%':>7} {'TP%':>7} {'Hor%':>7} {'AvgHold':>8} "
        f"{'Best':>11} {'Worst':>11} {'ExitReasons':<36}"
    )
    for row in result.simulation_rows:
        print(
            f"{row.symbol:<10} {row.strategy_name:<18} {row.trades:>7} "
            f"{row.wins:>6} {row.losses:>7} {row.win_rate:>7.1f}% "
            f"${row.gross_pnl:>10.2f} ${row.costs:>10.2f} ${row.net_pnl:>10.2f} "
            f"${row.average_net_per_trade:>10.2f} "
            f"{row.stop_loss_hit_rate:>6.1f}% {row.take_profit_hit_rate:>6.1f}% "
            f"{row.max_horizon_exit_rate:>6.1f}% {row.average_hold_candles:>8.1f} "
            f"{format_trade_net(row.best_trade):>11} {format_trade_net(row.worst_trade):>11} "
            f"{format_exit_reasons(row.exit_reason_counts):<36}"
        )
    print_quality_rejection_diagnostics(
        result.quality_rejection_counts,
        result.quality_rejected_simulations,
        result.quality_rejected_losing_count,
    )
    print_accepted_loser_clusters(
        build_accepted_loser_clusters(result.simulation_trades, result.quality_config),
        "Accepted Loser Clusters",
        repeated_combinations=False,
    )
    print_losing_setup_examples(result.simulation_trades, result.quality_config)


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


def print_quality_config(config: BacktestQualityConfig) -> None:
    print(
        "exhaustion_thresholds="
        f"rsi_high={config.exhaustion_rsi_high:.2f}, "
        f"rsi_low={config.exhaustion_rsi_low:.2f}, "
        f"close_high={config.exhaustion_close_position_high:.2f}, "
        f"close_low={config.exhaustion_close_position_low:.2f}, "
        f"candle_atr={config.exhaustion_candle_atr_multiplier:.2f}, "
        f"extreme_rsi_high={config.extreme_rsi_high:.2f}, "
        f"extreme_rsi_low={config.extreme_rsi_low:.2f}, "
        f"soft_short_rsi={config.soft_rsi_low_short:.2f}, "
        f"soft_short_close={config.soft_close_position_low_short:.2f}, "
        f"soft_long_rsi={config.soft_rsi_high_long:.2f}, "
        f"soft_long_close={config.soft_close_position_high_long:.2f}, "
        f"reject_soft_late_momentum={'enabled' if config.reject_soft_late_momentum else 'disabled'}"
    )


def print_realized_optimization_report(result: RealizedOptimizationResult) -> None:
    print("Realized Backtest Optimization Report")
    print("calibration only")
    print("realized historical simulation")
    print("production thresholds unchanged")
    print("temporary sweep settings do not change main.py, settings/.env, or live/paper bot behavior")
    print("NetOK/WouldTrade use calibration_min_expected_net_profit; production min remains unchanged")
    print(f"signal_window_bars={result.signal_window_bars}")
    print_data_profiles(result.data_profiles)
    print(f"backtest_quality_filter={'enabled' if result.quality_filter_enabled else 'disabled'}")
    print(f"backtest_trailing_exits={'enabled' if result.trailing_exits_enabled else 'disabled'}")
    print_quality_config(result.quality_config)
    print(
        f"production_min_target_move_bps={result.production_target_move_bps:.2f} | "
        f"production_min_reward_cost_ratio={result.production_reward_cost_ratio:.2f}x | "
        f"production_min_expected_net_profit=${result.production_min_expected_net_profit:.2f} | "
        f"calibration_min_expected_net_profit=${result.calibration_min_expected_net_profit:.2f} | "
        f"diagnostic_notional=${result.diagnostic_notional:,.2f}"
    )
    print()
    print_realized_optimization_rows("All Realized Sweep Combinations", result.rows)

    traded_rows = [row for row in result.rows if row.trades > 0]
    print()
    print_realized_optimization_rows(
        "Best Combinations By Realized Net PnL",
        sorted(traded_rows, key=lambda row: (row.net_pnl, row.average_net_per_trade), reverse=True)[:10],
    )
    print()
    print_realized_optimization_rows(
        "Best Combinations By Avg Net Per Trade",
        sorted(traded_rows, key=lambda row: (row.average_net_per_trade, row.net_pnl), reverse=True)[:10],
    )
    print()
    print_realized_optimization_rows(
        "Best Combinations With At Least 30 Trades",
        sorted(
            [row for row in traded_rows if row.trades >= 30],
            key=lambda row: (row.net_pnl, row.average_net_per_trade),
            reverse=True,
        )[:10],
    )
    print()
    print_realized_optimization_rows(
        "Worst Combinations",
        sorted(traded_rows, key=lambda row: (row.net_pnl, row.average_net_per_trade))[:10],
    )
    print_quality_rejection_diagnostics(
        result.quality_rejection_counts,
        result.quality_rejected_simulations,
        result.quality_rejected_losing_count,
    )
    print_accepted_loser_clusters(
        result.accepted_loser_clusters,
        "Accepted Loser Clusters (Global Across Evaluated Combinations)",
        repeated_combinations=True,
    )
    print_momentum_entry_diagnostics(result.momentum_entry_clusters)
    print_momentum_entry_only_diagnostics(result.momentum_entry_only_clusters)
    print_losing_setup_examples(result.losing_examples, result.quality_config)
    print_compact_realized_sweep_summary(result)


def print_realized_optimization_rows(
    title: str,
    rows: list[RealizedOptimizationRow],
) -> None:
    print(title)
    if not rows:
        print("none")
        return
    print(
        f"{'TF':<5} {'Symbol':<10} {'Strategy':<18} {'Tgt':>7} {'R/C':>6} "
        f"{'Hold':>5} {'ATRTP':>6} {'ATRSL':>6} {'Trades':>7} {'Wins':>6} "
        f"{'Loss':>6} {'Win%':>7} {'Gross':>11} {'Costs':>11} {'Net':>11} "
        f"{'AvgNet':>11} {'PF':>7} {'MaxDD':>10} {'Stop%':>7} {'TP%':>7} "
        f"{'Hor%':>7} {'AvgHold':>8} {'QRej':>7} {'QRejLoss':>9} "
        f"{'ExitReasons':<36} {'QualityRejections':<42}"
    )
    for row in rows:
        print(
            f"{row.timeframe:<5} {row.symbol:<10} {row.strategy_name:<18} "
            f"{row.min_target_move_bps:>7.2f} {row.min_reward_cost_ratio:>6.2f} "
            f"{row.max_hold_candles:>5} {row.atr_take_profit_multiplier:>6.2f} "
            f"{row.atr_stop_loss_multiplier:>6.2f} {row.trades:>7} {row.wins:>6} "
            f"{row.losses:>6} {row.win_rate:>6.1f}% ${row.gross_pnl:>10.2f} "
            f"${row.costs:>10.2f} ${row.net_pnl:>10.2f} "
            f"${row.average_net_per_trade:>10.2f} {format_profit_factor(row.profit_factor):>7} "
            f"${row.max_drawdown:>9.2f} {row.stop_loss_hit_rate:>6.1f}% "
            f"{row.take_profit_hit_rate:>6.1f}% {row.max_horizon_exit_rate:>6.1f}% "
            f"{row.average_hold_candles:>8.1f} {row.quality_rejected_count:>7} "
            f"{row.quality_rejected_losing_count:>9} "
            f"{format_exit_reasons(row.exit_reason_counts):<36} "
            f"{format_rejection_counts(row.quality_rejection_counts):<42}"
        )


def format_profit_factor(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value == float("inf"):
        return "inf"
    return f"{value:.2f}"


def print_data_profiles(profiles: list[dict[str, Any]]) -> None:
    if not profiles:
        print("data_profiles=n/a")
        return
    print("Data Profiles")
    for profile in profiles:
        approx_days = profile.get("approx_days")
        approx_text = "n/a" if approx_days is None else f"{float(approx_days):.1f}"
        print(
            f"data_profile symbol={profile.get('symbol', 'n/a')} "
            f"timeframe={profile.get('timeframe', 'n/a')} "
            f"candles={profile.get('candles', 'n/a')} "
            f"start={profile.get('start_utc', 'n/a')} "
            f"end={profile.get('end_utc', 'n/a')} "
            f"approx_days={approx_text}"
        )
    print()


def print_compact_realized_sweep_summary(result: RealizedOptimizationResult) -> None:
    summary = build_compact_realized_sweep_summary(result)

    print()
    print("=== Compact Realized Sweep Summary ===")
    print(f"diagnostic_notional=${summary['diagnostic_notional']:,.2f}")
    print(f"signal_window_bars={summary['signal_window_bars']}")
    print(f"production_min_target_move_bps={summary['production_min_target_move_bps']:.2f}")
    print(f"production_min_reward_cost_ratio={summary['production_min_reward_cost_ratio']:.2f}")
    print(f"production_min_expected_net_profit=${summary['production_min_expected_net_profit']:,.2f}")
    print(f"calibration_min_expected_net_profit=${summary['calibration_min_expected_net_profit']:,.2f}")
    print(f"symbols={','.join(summary['symbols']) if summary['symbols'] else 'n/a'}")
    print(f"timeframes={','.join(summary['timeframes']) if summary['timeframes'] else 'n/a'}")
    print(f"total_candles={summary['total_candles']}")
    print(f"data_period_start={summary['data_period_start']}")
    print(f"data_period_end={summary['data_period_end']}")
    approx_days = summary.get("approx_days")
    print(f"approx_days={approx_days:.1f}" if isinstance(approx_days, (int, float)) else "approx_days=n/a")
    print(f"btc_only={summary['btc_only']}")
    print(f"reject_soft_late_momentum={summary['reject_soft_late_momentum']}")
    print(f"total_combinations={summary['total_combinations']}")
    print(f"positive_combinations={summary['positive_combinations']}")
    print(
        "positive_combinations_with_at_least_30_trades="
        f"{summary['positive_combinations_with_at_least_30_trades']}"
    )
    print(f"best_overall={format_compact_row_summary(summary['best_overall'])}")
    print(f"best_at_least_30={format_compact_row_summary(summary['best_at_least_30'])}")
    print(f"worst_overall={format_compact_row_summary(summary['worst_overall'])}")
    print(f"best_soft_threshold_set={format_compact_threshold_summary(summary['best_soft_threshold_set'])}")
    print(f"top_quality_rejections={format_top_quality_rejection_summary(summary['top_quality_rejections'])}")
    for key in ("rejected_soft_late_long", "rejected_soft_late_short"):
        count = summary["soft_late_rejections"].get(key, 0)
        if count > 0:
            print(f"{key}={count}")
    print(f"top_accepted_loser_cluster={format_compact_cluster_summary(summary['top_accepted_loser_cluster'])}")
    print(f"best_momentum_cluster={format_compact_momentum_cluster_summary(summary['best_momentum_cluster'])}")
    print(f"worst_momentum_cluster={format_compact_momentum_cluster_summary(summary['worst_momentum_cluster'])}")
    print(f"buy_momentum={format_compact_momentum_side_summary(summary['buy_momentum'])}")
    print(f"sell_momentum={format_compact_momentum_side_summary(summary['sell_momentum'])}")
    print(f"best_entry_momentum_cluster={format_compact_entry_momentum_cluster_summary(summary['best_entry_momentum_cluster'])}")
    print(f"worst_entry_momentum_cluster={format_compact_entry_momentum_cluster_summary(summary['worst_entry_momentum_cluster'])}")
    print(f"best_buy_entry_momentum_cluster={format_compact_entry_momentum_cluster_summary(summary['best_buy_entry_momentum_cluster'])}")
    print(f"worst_buy_entry_momentum_cluster={format_compact_entry_momentum_cluster_summary(summary['worst_buy_entry_momentum_cluster'])}")
    print(f"best_sell_entry_momentum_cluster={format_compact_entry_momentum_cluster_summary(summary['best_sell_entry_momentum_cluster'])}")
    print(f"worst_sell_entry_momentum_cluster={format_compact_entry_momentum_cluster_summary(summary['worst_sell_entry_momentum_cluster'])}")
    print(f"best_entry_momentum_cluster_at_least_20={format_compact_entry_momentum_cluster_summary(summary['best_entry_momentum_cluster_at_least_20'])}")
    print(f"best_entry_momentum_cluster_at_least_30={format_compact_entry_momentum_cluster_summary(summary['best_entry_momentum_cluster_at_least_30'])}")
    print(f"verdict={summary['verdict']}")
    print("=== End Compact Realized Sweep Summary ===")


def build_compact_realized_sweep_summary(result: RealizedOptimizationResult) -> dict[str, Any]:
    traded_rows = [row for row in result.rows if row.trades > 0]
    rows_at_least_30 = [row for row in traded_rows if row.trades >= 30]
    data_summary = compact_data_summary(result.data_profiles)
    best_overall = max(
        traded_rows,
        key=lambda row: (row.net_pnl, row.average_net_per_trade),
        default=None,
    )
    best_at_least_30 = max(
        rows_at_least_30,
        key=lambda row: (row.net_pnl, row.average_net_per_trade),
        default=None,
    )
    worst_overall = min(
        traded_rows,
        key=lambda row: (row.net_pnl, row.average_net_per_trade),
        default=None,
    )
    return {
        "summary_version": 2,
        "diagnostic_notional": result.diagnostic_notional,
        "signal_window_bars": result.signal_window_bars,
        "production_min_target_move_bps": result.production_target_move_bps,
        "production_min_reward_cost_ratio": result.production_reward_cost_ratio,
        "production_min_expected_net_profit": result.production_min_expected_net_profit,
        "calibration_min_expected_net_profit": result.calibration_min_expected_net_profit,
        "data_profiles": result.data_profiles,
        "symbols": data_summary["symbols"],
        "timeframes": data_summary["timeframes"],
        "total_candles": data_summary["total_candles"],
        "data_period_start": data_summary["data_period_start"],
        "data_period_end": data_summary["data_period_end"],
        "approx_days": data_summary["approx_days"],
        "btc_only": data_summary["btc_only"],
        "reject_soft_late_momentum": "enabled" if result.quality_config.reject_soft_late_momentum else "disabled",
        "total_combinations": len(result.rows),
        "positive_combinations": sum(1 for row in traded_rows if row.net_pnl > 0),
        "positive_combinations_with_at_least_30_trades": sum(
            1 for row in rows_at_least_30 if row.net_pnl > 0
        ),
        "best_overall": compact_realized_row_data(best_overall),
        "best_at_least_30": compact_realized_row_data(best_at_least_30),
        "worst_overall": compact_realized_row_data(worst_overall),
        "best_soft_threshold_set": compact_soft_threshold_data(best_at_least_30 or best_overall),
        "top_quality_rejections": top_quality_rejection_data(result.quality_rejection_counts),
        "soft_late_rejections": {
            "rejected_soft_late_long": result.quality_rejection_counts.get("rejected_soft_late_long", 0),
            "rejected_soft_late_short": result.quality_rejection_counts.get("rejected_soft_late_short", 0),
        },
        "top_accepted_loser_cluster": compact_accepted_loser_cluster_data(result.accepted_loser_clusters),
        "best_momentum_cluster": compact_momentum_cluster_data(best_momentum_cluster(result.momentum_entry_clusters)),
        "worst_momentum_cluster": compact_momentum_cluster_data(worst_momentum_cluster(result.momentum_entry_clusters)),
        "buy_momentum": compact_momentum_side_summary(result.momentum_entry_clusters, "buy"),
        "sell_momentum": compact_momentum_side_summary(result.momentum_entry_clusters, "sell"),
        "best_entry_momentum_cluster": compact_entry_momentum_cluster_data(
            best_entry_momentum_cluster(result.momentum_entry_only_clusters)
        ),
        "worst_entry_momentum_cluster": compact_entry_momentum_cluster_data(
            worst_entry_momentum_cluster(result.momentum_entry_only_clusters)
        ),
        "best_buy_entry_momentum_cluster": compact_entry_momentum_cluster_data(
            best_entry_momentum_cluster(result.momentum_entry_only_clusters, side="buy")
        ),
        "worst_buy_entry_momentum_cluster": compact_entry_momentum_cluster_data(
            worst_entry_momentum_cluster(result.momentum_entry_only_clusters, side="buy")
        ),
        "best_sell_entry_momentum_cluster": compact_entry_momentum_cluster_data(
            best_entry_momentum_cluster(result.momentum_entry_only_clusters, side="sell")
        ),
        "worst_sell_entry_momentum_cluster": compact_entry_momentum_cluster_data(
            worst_entry_momentum_cluster(result.momentum_entry_only_clusters, side="sell")
        ),
        "best_entry_momentum_cluster_at_least_20": compact_entry_momentum_cluster_data(
            best_entry_momentum_cluster(result.momentum_entry_only_clusters, min_trades=20)
        ),
        "best_entry_momentum_cluster_at_least_30": compact_entry_momentum_cluster_data(
            best_entry_momentum_cluster(result.momentum_entry_only_clusters, min_trades=30)
        ),
        "verdict": compact_realized_verdict(best_at_least_30),
    }


def compact_data_summary(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    symbols = sorted({str(profile.get("symbol", "")) for profile in profiles if profile.get("symbol")})
    timeframes = sorted({str(profile.get("timeframe", "")) for profile in profiles if profile.get("timeframe")})
    starts = [str(profile.get("start_utc")) for profile in profiles if profile.get("start_utc") not in {None, "n/a"}]
    ends = [str(profile.get("end_utc")) for profile in profiles if profile.get("end_utc") not in {None, "n/a"}]
    total_candles = sum(int(profile.get("candles") or 0) for profile in profiles)
    day_values = [float(profile["approx_days"]) for profile in profiles if profile.get("approx_days") is not None]
    return {
        "symbols": symbols,
        "timeframes": timeframes,
        "total_candles": total_candles,
        "data_period_start": min(starts) if starts else "n/a",
        "data_period_end": max(ends) if ends else "n/a",
        "approx_days": min(day_values) if day_values else None,
        "btc_only": bool(symbols) and all(symbol == "BTC/USDT" for symbol in symbols),
    }


def append_realized_summary_log(
    result: RealizedOptimizationResult,
    path: Path,
    run_label: str | None = None,
) -> None:
    summary = build_compact_realized_sweep_summary(result)
    record = {
        "logged_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_label": run_label or "",
        "mode": "realized_sweep",
        "summary": summary,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def compact_realized_row_data(row: RealizedOptimizationRow | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "symbol": row.symbol,
        "strategy": row.strategy_name,
        "target_bps": row.min_target_move_bps,
        "reward_cost": row.min_reward_cost_ratio,
        "hold": row.max_hold_candles,
        "atrtp": row.atr_take_profit_multiplier,
        "atrsl": row.atr_stop_loss_multiplier,
        "trades": row.trades,
        "wins": row.wins,
        "losses": row.losses,
        "win_rate": row.win_rate,
        "gross": row.gross_pnl,
        "costs": row.costs,
        "net": row.net_pnl,
        "avg_net": row.average_net_per_trade,
        "pf": row.profit_factor,
        "max_drawdown": row.max_drawdown,
        "stop_loss_hit_rate": row.stop_loss_hit_rate,
        "take_profit_hit_rate": row.take_profit_hit_rate,
        "max_horizon_exit_rate": row.max_horizon_exit_rate,
        "soft_thresholds": compact_soft_threshold_data(row),
    }


def compact_soft_threshold_data(row: RealizedOptimizationRow | None) -> dict[str, float] | None:
    if row is None:
        return None
    return {
        "soft_rsi_high_long": row.soft_rsi_high_long,
        "soft_close_position_high_long": row.soft_close_position_high_long,
        "soft_rsi_low_short": row.soft_rsi_low_short,
        "soft_close_position_low_short": row.soft_close_position_low_short,
    }


def top_quality_rejection_data(counts: dict[str, int], limit: int = 8) -> list[dict[str, int | str]]:
    sorted_counts = sorted(
        ((reason, count) for reason, count in counts.items() if count > 0),
        key=lambda item: (-item[1], item[0]),
    )
    return [{"reason": reason, "count": count} for reason, count in sorted_counts[:limit]]


def compact_accepted_loser_cluster_data(clusters: list[AcceptedLoserCluster]) -> dict[str, Any] | None:
    if not clusters:
        return None
    cluster = clusters[0]
    return {
        "side": cluster.side,
        "strategy": cluster.strategy_name,
        "exit_reason": cluster.exit_reason,
        "rsi_band": cluster.rsi_band,
        "close_position_band": cluster.close_position_band,
        "hold_band": cluster.hold_band,
        "soft_label": cluster.soft_label,
        "count": cluster.count,
        "net": cluster.net_pnl,
        "avg_rsi": cluster.average_rsi if cluster.rsi_count > 0 else None,
        "avg_close": cluster.average_close_position if cluster.close_position_count > 0 else None,
        "avg_hold": cluster.average_hold,
        "stop_loss_count": cluster.stop_loss_count,
    }


def format_compact_row_summary(row: dict[str, Any] | None) -> str:
    if row is None:
        return "n/a"
    return (
        f"{row['symbol']} {row['strategy']} target={row['target_bps']:.2f}bps "
        f"hold={row['hold']} atrtp={row['atrtp']:.2f} atrsl={row['atrsl']:.2f} "
        f"trades={row['trades']} net=${row['net']:.2f} avg_net=${row['avg_net']:.2f} "
        f"pf={format_profit_factor(row['pf'])} "
        f"{format_compact_threshold_summary(row['soft_thresholds'])}"
    )


def format_compact_threshold_summary(thresholds: dict[str, float] | None) -> str:
    if thresholds is None:
        return "n/a"
    return (
        f"soft_rsi_high_long={thresholds['soft_rsi_high_long']:.2f} "
        f"soft_close_position_high_long={thresholds['soft_close_position_high_long']:.2f} "
        f"soft_rsi_low_short={thresholds['soft_rsi_low_short']:.2f} "
        f"soft_close_position_low_short={thresholds['soft_close_position_low_short']:.2f}"
    )


def format_top_quality_rejection_summary(rejections: list[dict[str, int | str]]) -> str:
    if not rejections:
        return "none"
    return ", ".join(f"{item['reason']}:{item['count']}" for item in rejections)


def format_compact_cluster_summary(cluster: dict[str, Any] | None) -> str:
    if cluster is None:
        return "none"
    return (
        f"{cluster['side']} {cluster['strategy']} {cluster['exit_reason']} "
        f"rsi={cluster['rsi_band']} close_position={cluster['close_position_band']} "
        f"hold={cluster['hold_band']} soft={cluster['soft_label']} count={cluster['count']} "
        f"net=${cluster['net']:.2f} avg_rsi={format_optional_value(cluster['avg_rsi'])} "
        f"avg_close={format_optional_value(cluster['avg_close'])} "
        f"avg_hold={cluster['avg_hold']:.1f} stop_loss_count={cluster['stop_loss_count']}"
    )


def format_compact_realized_row(row: RealizedOptimizationRow | None) -> str:
    if row is None:
        return "n/a"
    return (
        f"{row.symbol} {row.strategy_name} target={row.min_target_move_bps:.2f}bps "
        f"hold={row.max_hold_candles} atrtp={row.atr_take_profit_multiplier:.2f} "
        f"atrsl={row.atr_stop_loss_multiplier:.2f} trades={row.trades} "
        f"net=${row.net_pnl:.2f} avg_net=${row.average_net_per_trade:.2f} "
        f"pf={format_profit_factor(row.profit_factor)} "
        f"{format_compact_soft_thresholds(row)}"
    )


def format_compact_soft_thresholds(row: RealizedOptimizationRow | None) -> str:
    if row is None:
        return "n/a"
    return (
        f"soft_rsi_high_long={row.soft_rsi_high_long:.2f} "
        f"soft_close_position_high_long={row.soft_close_position_high_long:.2f} "
        f"soft_rsi_low_short={row.soft_rsi_low_short:.2f} "
        f"soft_close_position_low_short={row.soft_close_position_low_short:.2f}"
    )


def format_top_quality_rejections(counts: dict[str, int]) -> str:
    if not counts:
        return "none"
    sorted_counts = sorted(
        ((reason, count) for reason, count in counts.items() if count > 0),
        key=lambda item: (-item[1], item[0]),
    )
    if not sorted_counts:
        return "none"
    return ", ".join(f"{reason}:{count}" for reason, count in sorted_counts[:8])


def format_compact_accepted_loser_cluster(clusters: list[AcceptedLoserCluster]) -> str:
    if not clusters:
        return "none"
    cluster = clusters[0]
    return (
        f"{cluster.side} {cluster.strategy_name} {cluster.exit_reason} "
        f"rsi={cluster.rsi_band} close_position={cluster.close_position_band} "
        f"hold={cluster.hold_band} soft={cluster.soft_label} count={cluster.count} "
        f"net=${cluster.net_pnl:.2f} avg_rsi={format_optional_float(cluster.average_rsi, cluster.rsi_count)} "
        f"avg_close={format_optional_float(cluster.average_close_position, cluster.close_position_count)} "
        f"avg_hold={cluster.average_hold:.1f} stop_loss_count={cluster.stop_loss_count}"
    )


def compact_realized_verdict(best_at_least_30: RealizedOptimizationRow | None) -> str:
    if best_at_least_30 is None:
        return "too_few_trades"
    profit_factor = best_at_least_30.profit_factor
    if best_at_least_30.net_pnl <= 0 or profit_factor is None or profit_factor <= 1.0:
        return "not_profitable_at_30_trades"
    return "potentially_promising_needs_more_testing"


def print_quality_rejection_diagnostics(
    rejection_counts: dict[str, int],
    rejected_simulations: list[SimulationTrade],
    rejected_losing_count: int,
) -> None:
    print()
    print("Backtest Quality Filter Rejections (Global Across Evaluated Combinations)")
    if not rejection_counts:
        print("none")
        return
    keys = (
        "extreme_rsi_long",
        "extreme_rsi_short",
        "rejected_soft_late_long",
        "rejected_soft_late_short",
        "exhausted_long",
        "exhausted_short",
        "late_entry",
    )
    for key in keys:
        print(f"{key}={rejection_counts.get(key, 0)}")
    other_counts = {
        key: count
        for key, count in sorted(rejection_counts.items())
        if key not in keys and count > 0
    }
    if other_counts:
        print("other_rejections=" + format_rejection_counts(other_counts))
    print(f"rejected_candidates_that_simulated_losing_global={rejected_losing_count}")
    print("note=global rejection totals may include repeated sweep parameter combinations")


def build_accepted_loser_clusters(
    trades: list[SimulationTrade] | None,
    config: BacktestQualityConfig,
    limit: int = 12,
) -> list[AcceptedLoserCluster]:
    if not trades:
        return []

    clusters: dict[tuple[str, str, str, str, str, str, str], AcceptedLoserCluster] = {}
    for trade in trades:
        if trade.realized_net_pnl >= 0:
            continue
        metadata = trade.metadata
        rsi = metadata_float_or_none(metadata, "rsi")
        close_position = metadata_float_or_none(metadata, "close_position")
        soft_label = soft_late_label(trade, rsi, close_position, config)
        key = (
            trade.side.value,
            trade.strategy_name,
            trade.exit_reason,
            rsi_band(rsi),
            close_position_band(close_position),
            hold_band(trade.hold_candles),
            soft_label,
        )
        cluster = clusters.get(key)
        if cluster is None:
            cluster = AcceptedLoserCluster(
                side=key[0],
                strategy_name=key[1],
                exit_reason=key[2],
                rsi_band=key[3],
                close_position_band=key[4],
                hold_band=key[5],
                soft_label=key[6],
            )
            clusters[key] = cluster
        cluster.count += 1
        cluster.net_pnl += trade.realized_net_pnl
        if rsi is not None:
            cluster.rsi_sum += rsi
            cluster.rsi_count += 1
        if close_position is not None:
            cluster.close_position_sum += close_position
            cluster.close_position_count += 1
        cluster.hold_sum += trade.hold_candles
        if "stop_loss" in trade.exit_reason:
            cluster.stop_loss_count += 1
    return sorted(
        clusters.values(),
        key=lambda cluster: (cluster.net_pnl, -cluster.count),
    )[:limit]


def print_accepted_loser_clusters(
    clusters: list[AcceptedLoserCluster],
    title: str,
    repeated_combinations: bool,
) -> None:
    print()
    print(title)
    if repeated_combinations:
        print("note=clusters may include repeated historical setups across sweep parameter combinations")
    if not clusters:
        print("none")
        return
    print(
        f"{'Side':<5} {'Strategy':<18} {'Exit':<18} {'RSI':<8} {'ClosePos':<11} "
        f"{'Hold':<8} {'SoftLabel':<26} {'Count':>6} {'Net':>11} {'AvgRSI':>8} "
        f"{'AvgClose':>9} {'AvgHold':>8} {'StopCnt':>8}"
    )
    for cluster in clusters:
        print(
            f"{cluster.side:<5} {cluster.strategy_name:<18} {cluster.exit_reason:<18} "
            f"{cluster.rsi_band:<8} {cluster.close_position_band:<11} {cluster.hold_band:<8} "
            f"{cluster.soft_label:<26} {cluster.count:>6} ${cluster.net_pnl:>10.2f} "
            f"{format_optional_float(cluster.average_rsi, cluster.rsi_count):>8} "
            f"{format_optional_float(cluster.average_close_position, cluster.close_position_count):>9} "
            f"{cluster.average_hold:>8.1f} {cluster.stop_loss_count:>8}"
        )


def build_momentum_entry_clusters(
    trades: list[SimulationTrade] | None,
) -> list[MomentumEntryCluster]:
    if not trades:
        return []

    clusters: dict[tuple[str, str, str, str, str, str, str, str], MomentumEntryCluster] = {}
    for trade in trades:
        if trade.strategy_name != "momentum":
            continue
        metadata = trade.metadata
        key = (
            trade.side.value,
            rsi_band(metadata_float_or_none(metadata, "rsi")),
            macd_strength_band(metadata_float_or_none(metadata, "macd_hist_bps")),
            trend_regime_band(
                metadata_float_or_none(metadata, "trend_slope_bps"),
                metadata_float_or_none(metadata, "ema_gap_bps"),
            ),
            volume_ratio_band(metadata_float_or_none(metadata, "volume_ratio")),
            atr_bps_band(metadata_float_or_none(metadata, "atr_bps")),
            close_position_band(metadata_float_or_none(metadata, "close_position")),
            trade.exit_reason,
        )
        cluster = clusters.get(key)
        if cluster is None:
            cluster = MomentumEntryCluster(
                side=key[0],
                rsi_band=key[1],
                macd_band=key[2],
                trend_regime=key[3],
                volume_band=key[4],
                atr_bps_band=key[5],
                close_position_band=key[6],
                exit_reason=key[7],
            )
            clusters[key] = cluster
        cluster.record(trade)
    return sorted(
        clusters.values(),
        key=lambda cluster: (cluster.net_pnl, cluster.count),
        reverse=True,
    )


def print_momentum_entry_diagnostics(clusters: list[MomentumEntryCluster]) -> None:
    print()
    print("Momentum Entry Diagnostics (Accepted Trades, Global Across Evaluated Combinations)")
    print("note=clusters may include repeated historical setups across sweep parameter combinations")
    if not clusters:
        print("none")
        return
    print_momentum_side_summary(clusters)
    print()
    print_momentum_cluster_table(
        "Best Momentum Entry Clusters By Net PnL",
        sorted(clusters, key=lambda cluster: (cluster.net_pnl, cluster.average_net_per_trade), reverse=True)[:10],
    )
    print()
    print_momentum_cluster_table(
        "Worst Momentum Entry Clusters By Net PnL",
        sorted(clusters, key=lambda cluster: (cluster.net_pnl, cluster.average_net_per_trade))[:10],
    )


def print_momentum_side_summary(clusters: list[MomentumEntryCluster]) -> None:
    print(f"{'Side':<5} {'Trades':>7} {'Wins':>6} {'Loss':>6} {'Win%':>7} {'Net':>11} {'AvgNet':>11} {'PF':>7}")
    for side in ("buy", "sell"):
        summary = compact_momentum_side_summary(clusters, side)
        print(
            f"{side:<5} {summary['trades']:>7} {summary['wins']:>6} {summary['losses']:>6} "
            f"{summary['win_rate']:>6.1f}% ${summary['net']:>10.2f} "
            f"${summary['avg_net']:>10.2f} {format_profit_factor(summary['pf']):>7}"
        )


def print_momentum_cluster_table(title: str, clusters: list[MomentumEntryCluster]) -> None:
    print(title)
    if not clusters:
        print("none")
        return
    print(
        f"{'Side':<5} {'RSI':<8} {'MACD':<14} {'Trend':<18} {'Volume':<10} "
        f"{'ATRbps':<10} {'ClosePos':<11} {'Exit':<18} {'Trades':>7} "
        f"{'Net':>11} {'AvgNet':>11} {'PF':>7} {'Win%':>7} {'AvgHold':>8}"
    )
    for cluster in clusters:
        print(
            f"{cluster.side:<5} {cluster.rsi_band:<8} {cluster.macd_band:<14} "
            f"{cluster.trend_regime:<18} {cluster.volume_band:<10} "
            f"{cluster.atr_bps_band:<10} {cluster.close_position_band:<11} "
            f"{cluster.exit_reason:<18} {cluster.count:>7} ${cluster.net_pnl:>10.2f} "
            f"${cluster.average_net_per_trade:>10.2f} {format_profit_factor(cluster.profit_factor):>7} "
            f"{cluster.win_rate:>6.1f}% {cluster.average_hold:>8.1f}"
        )


def best_momentum_cluster(clusters: list[MomentumEntryCluster]) -> MomentumEntryCluster | None:
    if not clusters:
        return None
    return max(clusters, key=lambda cluster: (cluster.net_pnl, cluster.average_net_per_trade))


def worst_momentum_cluster(clusters: list[MomentumEntryCluster]) -> MomentumEntryCluster | None:
    if not clusters:
        return None
    return min(clusters, key=lambda cluster: (cluster.net_pnl, cluster.average_net_per_trade))


def compact_momentum_cluster_data(cluster: MomentumEntryCluster | None) -> dict[str, Any] | None:
    if cluster is None:
        return None
    return {
        "side": cluster.side,
        "rsi_band": cluster.rsi_band,
        "macd_band": cluster.macd_band,
        "trend_regime": cluster.trend_regime,
        "volume_band": cluster.volume_band,
        "atr_bps_band": cluster.atr_bps_band,
        "close_position_band": cluster.close_position_band,
        "exit_reason": cluster.exit_reason,
        "trades": cluster.count,
        "wins": cluster.wins,
        "losses": cluster.losses,
        "win_rate": cluster.win_rate,
        "net": cluster.net_pnl,
        "avg_net": cluster.average_net_per_trade,
        "pf": cluster.profit_factor,
        "avg_hold": cluster.average_hold,
    }


def compact_momentum_side_summary(
    clusters: list[MomentumEntryCluster],
    side: str,
) -> dict[str, Any]:
    selected = [cluster for cluster in clusters if cluster.side == side]
    trades = sum(cluster.count for cluster in selected)
    wins = sum(cluster.wins for cluster in selected)
    losses = sum(cluster.losses for cluster in selected)
    net = sum(cluster.net_pnl for cluster in selected)
    gross_profit = sum(cluster.gross_profit for cluster in selected)
    gross_loss = sum(cluster.gross_loss for cluster in selected)
    if trades <= 0:
        profit_factor = None
        win_rate = 0.0
        avg_net = 0.0
    else:
        profit_factor = None if gross_loss <= 0 and gross_profit <= 0 else (float("inf") if gross_loss <= 0 else gross_profit / gross_loss)
        win_rate = wins / trades * 100
        avg_net = net / trades
    return {
        "side": side,
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "net": net,
        "avg_net": avg_net,
        "pf": profit_factor,
    }


def format_compact_momentum_cluster_summary(cluster: dict[str, Any] | None) -> str:
    if cluster is None:
        return "n/a"
    return (
        f"{cluster['side']} rsi={cluster['rsi_band']} macd={cluster['macd_band']} "
        f"trend={cluster['trend_regime']} volume={cluster['volume_band']} "
        f"atr={cluster['atr_bps_band']} close_position={cluster['close_position_band']} "
        f"exit={cluster['exit_reason']} trades={cluster['trades']} "
        f"net=${cluster['net']:.2f} avg_net=${cluster['avg_net']:.2f} "
        f"pf={format_profit_factor(cluster['pf'])}"
    )


def format_compact_momentum_side_summary(summary: dict[str, Any]) -> str:
    return (
        f"trades={summary['trades']} net=${summary['net']:.2f} "
        f"avg_net=${summary['avg_net']:.2f} pf={format_profit_factor(summary['pf'])}"
    )


def build_momentum_entry_only_clusters(
    trades: list[SimulationTrade] | None,
) -> list[MomentumEntryOnlyCluster]:
    if not trades:
        return []

    clusters: dict[tuple[str, str, str, str, str, str, str], MomentumEntryOnlyCluster] = {}
    for trade in trades:
        if trade.strategy_name != "momentum":
            continue
        metadata = trade.metadata
        key = (
            trade.side.value,
            rsi_band(metadata_float_or_none(metadata, "rsi")),
            macd_strength_band(metadata_float_or_none(metadata, "macd_hist_bps")),
            trend_regime_band(
                metadata_float_or_none(metadata, "trend_slope_bps"),
                metadata_float_or_none(metadata, "ema_gap_bps"),
            ),
            volume_ratio_band(metadata_float_or_none(metadata, "volume_ratio")),
            atr_bps_band(metadata_float_or_none(metadata, "atr_bps")),
            close_position_band(metadata_float_or_none(metadata, "close_position")),
        )
        cluster = clusters.get(key)
        if cluster is None:
            cluster = MomentumEntryOnlyCluster(
                side=key[0],
                rsi_band=key[1],
                macd_band=key[2],
                trend_regime=key[3],
                volume_band=key[4],
                atr_bps_band=key[5],
                close_position_band=key[6],
            )
            clusters[key] = cluster
        cluster.record(trade)
    return sorted(
        clusters.values(),
        key=lambda cluster: (cluster.net_pnl, cluster.count),
        reverse=True,
    )


def print_momentum_entry_only_diagnostics(clusters: list[MomentumEntryOnlyCluster]) -> None:
    print()
    print("Momentum Entry-Only Diagnostics (Accepted Trades, Global Across Evaluated Combinations)")
    print("note=cluster keys use entry-time features only; outcome columns are realized after entry")
    print("note=clusters may include repeated historical setups across sweep parameter combinations")
    if not clusters:
        print("none")
        return
    print_entry_momentum_cluster_table(
        "Best Entry-Only Momentum Clusters By Net PnL",
        sorted(clusters, key=lambda cluster: (cluster.net_pnl, cluster.average_net_per_trade), reverse=True)[:10],
    )
    print()
    print_entry_momentum_cluster_table(
        "Best Entry-Only Momentum Clusters With At Least 20 Trades",
        sorted(
            [cluster for cluster in clusters if cluster.count >= 20],
            key=lambda cluster: (cluster.net_pnl, cluster.average_net_per_trade),
            reverse=True,
        )[:10],
    )
    print()
    print_entry_momentum_cluster_table(
        "Best Entry-Only Momentum Clusters With At Least 30 Trades",
        sorted(
            [cluster for cluster in clusters if cluster.count >= 30],
            key=lambda cluster: (cluster.net_pnl, cluster.average_net_per_trade),
            reverse=True,
        )[:10],
    )
    print()
    print_entry_momentum_cluster_table(
        "Worst Entry-Only Momentum Clusters By Net PnL",
        sorted(clusters, key=lambda cluster: (cluster.net_pnl, cluster.average_net_per_trade))[:10],
    )


def print_entry_momentum_cluster_table(
    title: str,
    clusters: list[MomentumEntryOnlyCluster],
) -> None:
    print(title)
    if not clusters:
        print("none")
        return
    print(
        f"{'Side':<5} {'RSI':<8} {'MACD':<14} {'Trend':<18} {'Volume':<10} "
        f"{'ATRbps':<10} {'ClosePos':<11} {'Trades':>7} {'Wins':>6} {'Loss':>6} "
        f"{'Win%':>7} {'Net':>11} {'AvgNet':>11} {'PF':>7} {'Stop':>6} "
        f"{'TP':>6} {'Horizon':>8} {'AvgHold':>8}"
    )
    for cluster in clusters:
        print(
            f"{cluster.side:<5} {cluster.rsi_band:<8} {cluster.macd_band:<14} "
            f"{cluster.trend_regime:<18} {cluster.volume_band:<10} "
            f"{cluster.atr_bps_band:<10} {cluster.close_position_band:<11} "
            f"{cluster.count:>7} {cluster.wins:>6} {cluster.losses:>6} "
            f"{cluster.win_rate:>6.1f}% ${cluster.net_pnl:>10.2f} "
            f"${cluster.average_net_per_trade:>10.2f} {format_profit_factor(cluster.profit_factor):>7} "
            f"{cluster.stop_loss_hit_count:>6} {cluster.take_profit_hit_count:>6} "
            f"{cluster.max_horizon_exit_count:>8} {cluster.average_hold:>8.1f}"
        )


def best_entry_momentum_cluster(
    clusters: list[MomentumEntryOnlyCluster],
    side: str | None = None,
    min_trades: int = 1,
) -> MomentumEntryOnlyCluster | None:
    selected = [
        cluster
        for cluster in clusters
        if cluster.count >= min_trades and (side is None or cluster.side == side)
    ]
    if not selected:
        return None
    return max(selected, key=lambda cluster: (cluster.net_pnl, cluster.average_net_per_trade))


def worst_entry_momentum_cluster(
    clusters: list[MomentumEntryOnlyCluster],
    side: str | None = None,
    min_trades: int = 1,
) -> MomentumEntryOnlyCluster | None:
    selected = [
        cluster
        for cluster in clusters
        if cluster.count >= min_trades and (side is None or cluster.side == side)
    ]
    if not selected:
        return None
    return min(selected, key=lambda cluster: (cluster.net_pnl, cluster.average_net_per_trade))


def compact_entry_momentum_cluster_data(
    cluster: MomentumEntryOnlyCluster | None,
) -> dict[str, Any] | None:
    if cluster is None:
        return None
    return {
        "side": cluster.side,
        "rsi_band": cluster.rsi_band,
        "macd_band": cluster.macd_band,
        "trend_regime": cluster.trend_regime,
        "volume_band": cluster.volume_band,
        "atr_bps_band": cluster.atr_bps_band,
        "close_position_band": cluster.close_position_band,
        "trades": cluster.count,
        "wins": cluster.wins,
        "losses": cluster.losses,
        "win_rate": cluster.win_rate,
        "net": cluster.net_pnl,
        "avg_net": cluster.average_net_per_trade,
        "pf": cluster.profit_factor,
        "stop_loss_hit_count": cluster.stop_loss_hit_count,
        "take_profit_hit_count": cluster.take_profit_hit_count,
        "max_horizon_exit_count": cluster.max_horizon_exit_count,
        "avg_hold": cluster.average_hold,
    }


def format_compact_entry_momentum_cluster_summary(cluster: dict[str, Any] | None) -> str:
    if cluster is None:
        return "n/a"
    return (
        f"{cluster['side']} rsi={cluster['rsi_band']} macd={cluster['macd_band']} "
        f"trend={cluster['trend_regime']} volume={cluster['volume_band']} "
        f"atr={cluster['atr_bps_band']} close_position={cluster['close_position_band']} "
        f"trades={cluster['trades']} net=${cluster['net']:.2f} "
        f"avg_net=${cluster['avg_net']:.2f} pf={format_profit_factor(cluster['pf'])} "
        f"stop={cluster['stop_loss_hit_count']} tp={cluster['take_profit_hit_count']} "
        f"horizon={cluster['max_horizon_exit_count']} avg_hold={cluster['avg_hold']:.1f}"
    )


def rsi_band(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value < 28:
        return "<28"
    if value < 32:
        return "28-32"
    if value < 36:
        return "32-36"
    if value < 45:
        return "36-45"
    if value < 55:
        return "45-55"
    if value < 64:
        return "55-64"
    if value < 68:
        return "64-68"
    if value <= 72:
        return "68-72"
    return ">72"


def close_position_band(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value < 0.10:
        return "<0.10"
    if value < 0.25:
        return "0.10-0.25"
    if value < 0.40:
        return "0.25-0.40"
    if value < 0.60:
        return "0.40-0.60"
    if value < 0.75:
        return "0.60-0.75"
    if value <= 0.90:
        return "0.75-0.90"
    return ">0.90"


def macd_strength_band(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value <= -5:
        return "neg_strong"
    if value <= -1:
        return "neg_medium"
    if value < 0:
        return "neg_weak"
    if value == 0:
        return "zero"
    if value < 1:
        return "pos_weak"
    if value < 5:
        return "pos_medium"
    return "pos_strong"


def trend_regime_band(trend_slope_bps: float | None, ema_gap_bps: float | None) -> str:
    if trend_slope_bps is None:
        return "n/a"
    if trend_slope_bps < 0:
        return "countertrend"
    if trend_slope_bps < 18:
        return "weak_or_flat"
    if ema_gap_bps is not None and ema_gap_bps < 8:
        return "low_ema_gap"
    if trend_slope_bps >= 55 or (ema_gap_bps is not None and ema_gap_bps >= 28):
        return "strong_extended"
    return "clear_aligned"


def volume_ratio_band(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value < 0.80:
        return "<0.80"
    if value < 1.00:
        return "0.80-1.00"
    if value < 1.12:
        return "1.00-1.12"
    if value < 1.50:
        return "1.12-1.50"
    if value < 2.00:
        return "1.50-2.00"
    return ">2.00"


def atr_bps_band(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value < 10:
        return "<10"
    if value < 25:
        return "10-25"
    if value < 50:
        return "25-50"
    if value < 75:
        return "50-75"
    if value < 100:
        return "75-100"
    return ">100"


def hold_band(hold_candles: int) -> str:
    if hold_candles <= 2:
        return "<=2"
    if hold_candles <= 5:
        return "3-5"
    if hold_candles <= 10:
        return "6-10"
    if hold_candles <= 20:
        return "11-20"
    return ">20"


def soft_late_label(
    trade: SimulationTrade,
    rsi: float | None,
    close_position: float | None,
    config: BacktestQualityConfig,
) -> str:
    if rsi is None or close_position is None:
        return "n/a"
    metadata = trade.metadata
    soft_rsi_low_short = metadata_float_or_none(metadata, "soft_rsi_low_short")
    soft_close_position_low_short = metadata_float_or_none(metadata, "soft_close_position_low_short")
    soft_rsi_high_long = metadata_float_or_none(metadata, "soft_rsi_high_long")
    soft_close_position_high_long = metadata_float_or_none(metadata, "soft_close_position_high_long")
    if soft_rsi_low_short is None:
        soft_rsi_low_short = config.soft_rsi_low_short
    if soft_close_position_low_short is None:
        soft_close_position_low_short = config.soft_close_position_low_short
    if soft_rsi_high_long is None:
        soft_rsi_high_long = config.soft_rsi_high_long
    if soft_close_position_high_long is None:
        soft_close_position_high_long = config.soft_close_position_high_long
    if (
        trade.side == Side.SELL
        and rsi <= soft_rsi_low_short
        and close_position <= soft_close_position_low_short
    ):
        return "soft_late_short_candidate"
    if (
        trade.side == Side.BUY
        and rsi >= soft_rsi_high_long
        and close_position >= soft_close_position_high_long
    ):
        return "soft_late_long_candidate"
    return "none"


def metadata_float_or_none(metadata: dict[str, Any], key: str) -> float | None:
    value = metadata.get(key)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    return number


def format_optional_float(value: float, count: int) -> str:
    if count <= 0:
        return "n/a"
    return f"{value:.2f}"


def format_optional_value(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def format_rejection_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "none"
    return ", ".join(f"{reason}:{count}" for reason, count in counts.items())


def print_losing_setup_examples(
    trades: list[SimulationTrade],
    config: BacktestQualityConfig,
) -> None:
    losing_trades = sorted(
        [trade for trade in trades if trade.realized_net_pnl < 0],
        key=lambda trade: trade.realized_net_pnl,
    )[:8]
    print()
    print("Losing Setup Examples")
    if not losing_trades:
        print("none")
        return
    displayed_rejected = sum(
        1
        for trade in losing_trades
        if str(trade.metadata.get("backtest_quality_reason", "n/a")) not in {"", "pass", "n/a"}
    )
    print(f"losing_setup_examples_would_now_be_rejected={displayed_rejected}")
    print("note=these rows are accepted losing trades unless QualityReason is a reject reason")
    print(
        f"{'Symbol':<10} {'Strategy':<18} {'Side':<5} {'Exit':<20} {'Hold':>5} "
        f"{'Net':>10} {'RSI':>7} {'MACD':>8} {'ATRbps':>8} {'Vol':>6} "
        f"{'Close':>7} {'StopBps':>8} {'Tgt/Stop':>8} {'SoftLabel':<26} "
        f"{'QualityReason':<32}"
    )
    for trade in losing_trades:
        metadata = trade.metadata
        rsi = metadata_float_or_none(metadata, "rsi")
        close_position = metadata_float_or_none(metadata, "close_position")
        print(
            f"{trade.symbol:<10} {trade.strategy_name:<18} {trade.side.value:<5} "
            f"{trade.exit_reason:<20} {trade.hold_candles:>5} "
            f"${trade.realized_net_pnl:>9.2f} "
            f"{format_metadata_float(metadata, 'rsi'):>7} "
            f"{format_metadata_float(metadata, 'macd_hist_bps'):>8} "
            f"{format_metadata_float(metadata, 'atr_bps'):>8} "
            f"{format_metadata_float(metadata, 'volume_ratio'):>6} "
            f"{format_metadata_float(metadata, 'close_position'):>7} "
            f"{format_metadata_float(metadata, 'stop_move_bps'):>8} "
            f"{format_metadata_float(metadata, 'target_stop_ratio'):>8} "
            f"{soft_late_label(trade, rsi, close_position, config):<26} "
            f"{str(metadata.get('backtest_quality_reason', 'n/a')):<32}"
        )


def format_metadata_float(metadata: dict[str, Any], key: str) -> str:
    value = metadata.get(key)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if pd.isna(number):
        return "n/a"
    return f"{number:.2f}"


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


def parse_int_list(raw: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in raw.split(",") if item.strip())


def normalize_timeframe(raw: str) -> str:
    timeframe = TIMEFRAME_ALIASES.get(raw.strip(), raw.strip())
    if timeframe not in ALLOWED_BACKTEST_TIMEFRAMES:
        valid = ", ".join(ALLOWED_BACKTEST_TIMEFRAMES)
        raise argparse.ArgumentTypeError(
            f"unsupported calibration timeframe {raw!r}; allowed: {valid}"
        )
    return timeframe


def calibration_limit_for_years(timeframe: str, years: float) -> int:
    timeframe = normalize_timeframe(timeframe)
    minutes = TIMEFRAME_MINUTES.get(timeframe)
    if minutes is None:
        valid = ", ".join(ALLOWED_BACKTEST_TIMEFRAMES)
        raise ValueError(f"unsupported calibration timeframe {timeframe!r}; allowed: {valid}")
    return max(1, int(years * 365 * 24 * 60 / minutes))


def calibration_env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return default if raw is None or raw == "" else float(raw)


def calibration_env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def format_float_list(values: tuple[float, ...]) -> str:
    return ",".join(f"{value:g}" for value in values)


def format_int_list(values: tuple[int, ...]) -> str:
    return ",".join(str(value) for value in values)


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
    loaded_settings = load_settings()
    parser.add_argument(
        "--symbols",
        default=",".join(loaded_settings.trading.symbols),
        help="Comma-separated symbols to calibrate, e.g. BTC/USDT,ETH/USDT.",
    )
    parser.add_argument(
        "--timeframe",
        type=normalize_timeframe,
        default=normalize_timeframe(loaded_settings.trading.timeframe),
        choices=ALLOWED_BACKTEST_TIMEFRAMES,
        help="Timeframe label used for CSV discovery and reporting. Allowed: 1m, 5m, 15m.",
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
        help="Optional number of most recent candles to use from each CSV. Overrides --years.",
    )
    parser.add_argument(
        "--years",
        type=float,
        default=DEFAULT_BACKTEST_YEARS,
        help="Backtesting lookback used to derive --limit when --limit is omitted. Default: 3 years.",
    )
    parser.add_argument(
        "--target-sweep",
        help=(
            "Comma-separated MIN_TARGET_MOVE_BPS sweep values. "
            f"Default: {format_float_list(DEFAULT_TARGET_SWEEP)}; "
            f"with --realized-sweep: {format_float_list(DEFAULT_REALIZED_TARGET_SWEEP)}."
        ),
    )
    parser.add_argument(
        "--reward-cost-sweep",
        help=(
            "Comma-separated reward/cost ratio sweep values. "
            f"Default: {format_float_list(DEFAULT_REWARD_COST_SWEEP)}; "
            f"with --realized-sweep: {format_float_list(DEFAULT_REALIZED_REWARD_COST_SWEEP)}."
        ),
    )
    parser.add_argument(
        "--diagnostic-notional",
        type=float,
        help="Optional notional used only for calibration expected net profit. Default comes from DIAGNOSTIC_NOTIONAL, currently $100 unless configured.",
    )
    parser.add_argument(
        "--calibration-min-expected-net-profit",
        type=float,
        help=(
            "Calibration/backtest-only expected net profit gate in USD. "
            "Default comes from CALIBRATION_MIN_EXPECTED_NET_PROFIT_USD, "
            "or 25bps of DIAGNOSTIC_NOTIONAL."
        ),
    )
    parser.add_argument(
        "--max-hold-candles",
        type=int,
        default=60,
        help="Maximum future candles used for calibration exit simulation.",
    )
    parser.add_argument(
        "--signal-window-bars",
        type=int,
        default=DEFAULT_SIGNAL_WINDOW_BARS,
        help=(
            "Calibration/backtest-only recent candle window passed to strategy signal generation. "
            "Keeps long-history runs tractable and does not change live/paper execution."
        ),
    )
    parser.add_argument(
        "--realized-sweep",
        action="store_true",
        help="Run calibration-only realized optimization sweeps over thresholds, ATR exits, and max hold.",
    )
    parser.add_argument(
        "--backtest-quality-filter",
        action="store_true",
        help="Enable experimental calibration-only regime and entry-quality filters for normal calibration.",
    )
    parser.add_argument(
        "--disable-backtest-quality-filter",
        action="store_true",
        help="Disable the experimental quality filter that is enabled by default for --realized-sweep.",
    )
    parser.add_argument(
        "--backtest-trailing-exits",
        action="store_true",
        help="Enable optional calibration-only breakeven/trailing stop simulation.",
    )
    parser.add_argument(
        "--breakeven-trigger-r",
        type=float,
        default=1.0,
        help="Favorable move in R before simulated breakeven stop is allowed.",
    )
    parser.add_argument(
        "--trailing-atr-multiplier",
        type=float,
        default=1.0,
        help="ATR multiple for optional calibration-only trailing stop simulation.",
    )
    parser.add_argument(
        "--exhaustion-rsi-high",
        type=float,
        default=calibration_env_float("EXHAUSTION_RSI_HIGH", DEFAULT_EXHAUSTION_RSI_HIGH),
        help="Calibration-only late-long RSI threshold. Env: EXHAUSTION_RSI_HIGH.",
    )
    parser.add_argument(
        "--exhaustion-rsi-low",
        type=float,
        default=calibration_env_float("EXHAUSTION_RSI_LOW", DEFAULT_EXHAUSTION_RSI_LOW),
        help="Calibration-only late-short RSI threshold. Env: EXHAUSTION_RSI_LOW.",
    )
    parser.add_argument(
        "--exhaustion-close-position-high",
        type=float,
        default=calibration_env_float(
            "EXHAUSTION_CLOSE_POSITION_HIGH",
            DEFAULT_EXHAUSTION_CLOSE_POSITION_HIGH,
        ),
        help="Calibration-only close-near-high threshold. Env: EXHAUSTION_CLOSE_POSITION_HIGH.",
    )
    parser.add_argument(
        "--exhaustion-close-position-low",
        type=float,
        default=calibration_env_float(
            "EXHAUSTION_CLOSE_POSITION_LOW",
            DEFAULT_EXHAUSTION_CLOSE_POSITION_LOW,
        ),
        help="Calibration-only close-near-low threshold. Env: EXHAUSTION_CLOSE_POSITION_LOW.",
    )
    parser.add_argument(
        "--exhaustion-candle-atr-multiplier",
        type=float,
        default=calibration_env_float(
            "EXHAUSTION_CANDLE_ATR_MULTIPLIER",
            DEFAULT_EXHAUSTION_CANDLE_ATR_MULTIPLIER,
        ),
        help="Calibration-only body/range ATR multiple for exhaustion. Env: EXHAUSTION_CANDLE_ATR_MULTIPLIER.",
    )
    parser.add_argument(
        "--extreme-rsi-high",
        type=float,
        default=calibration_env_float("EXTREME_RSI_HIGH", DEFAULT_EXTREME_RSI_HIGH),
        help="Calibration-only hard RSI ceiling for long momentum/breakout entries. Env: EXTREME_RSI_HIGH.",
    )
    parser.add_argument(
        "--extreme-rsi-low",
        type=float,
        default=calibration_env_float("EXTREME_RSI_LOW", DEFAULT_EXTREME_RSI_LOW),
        help="Calibration-only hard RSI floor for short momentum/breakout entries. Env: EXTREME_RSI_LOW.",
    )
    parser.add_argument(
        "--soft-rsi-low-short",
        type=float,
        default=calibration_env_float("SOFT_RSI_LOW_SHORT", DEFAULT_SOFT_RSI_LOW_SHORT),
        help="Diagnostics-only soft late-short RSI label threshold. Env: SOFT_RSI_LOW_SHORT.",
    )
    parser.add_argument(
        "--soft-close-position-low-short",
        type=float,
        default=calibration_env_float(
            "SOFT_CLOSE_POSITION_LOW_SHORT",
            DEFAULT_SOFT_CLOSE_POSITION_LOW_SHORT,
        ),
        help="Diagnostics-only soft late-short close-position label threshold. Env: SOFT_CLOSE_POSITION_LOW_SHORT.",
    )
    parser.add_argument(
        "--soft-rsi-high-long",
        type=float,
        default=calibration_env_float("SOFT_RSI_HIGH_LONG", DEFAULT_SOFT_RSI_HIGH_LONG),
        help="Diagnostics-only soft late-long RSI label threshold. Env: SOFT_RSI_HIGH_LONG.",
    )
    parser.add_argument(
        "--soft-close-position-high-long",
        type=float,
        default=calibration_env_float(
            "SOFT_CLOSE_POSITION_HIGH_LONG",
            DEFAULT_SOFT_CLOSE_POSITION_HIGH_LONG,
        ),
        help="Diagnostics-only soft late-long close-position label threshold. Env: SOFT_CLOSE_POSITION_HIGH_LONG.",
    )
    parser.add_argument(
        "--soft-rsi-high-long-sweep",
        help=(
            "Comma-separated calibration-only sweep for soft late-long RSI threshold. "
            f"Example: {format_float_list(DEFAULT_SOFT_RSI_HIGH_LONG_SWEEP)}."
        ),
    )
    parser.add_argument(
        "--soft-close-position-high-long-sweep",
        help=(
            "Comma-separated calibration-only sweep for soft late-long close-position threshold. "
            f"Example: {format_float_list(DEFAULT_SOFT_CLOSE_POSITION_HIGH_LONG_SWEEP)}."
        ),
    )
    parser.add_argument(
        "--soft-rsi-low-short-sweep",
        help=(
            "Comma-separated calibration-only sweep for soft late-short RSI threshold. "
            f"Example: {format_float_list(DEFAULT_SOFT_RSI_LOW_SHORT_SWEEP)}."
        ),
    )
    parser.add_argument(
        "--soft-close-position-low-short-sweep",
        help=(
            "Comma-separated calibration-only sweep for soft late-short close-position threshold. "
            f"Example: {format_float_list(DEFAULT_SOFT_CLOSE_POSITION_LOW_SHORT_SWEEP)}."
        ),
    )
    parser.add_argument(
        "--reject-soft-late-momentum",
        action="store_true",
        default=calibration_env_bool(
            "REJECT_SOFT_LATE_MOMENTUM",
            DEFAULT_REJECT_SOFT_LATE_MOMENTUM,
        ),
        help=(
            "Calibration-only: reject momentum entries matching soft late-entry labels. "
            "Default off. Env: REJECT_SOFT_LATE_MOMENTUM."
        ),
    )
    parser.add_argument(
        "--max-hold-sweep",
        default=format_int_list(DEFAULT_REALIZED_MAX_HOLD_SWEEP),
        help="Comma-separated max-hold candle values for --realized-sweep.",
    )
    parser.add_argument(
        "--atr-tp-sweep",
        default=format_float_list(DEFAULT_REALIZED_ATR_TP_SWEEP),
        help="Comma-separated ATR take-profit multipliers for --realized-sweep.",
    )
    parser.add_argument(
        "--atr-stop-sweep",
        default=format_float_list(DEFAULT_REALIZED_ATR_STOP_SWEEP),
        help="Comma-separated ATR stop-loss multipliers for --realized-sweep.",
    )
    parser.add_argument(
        "--save-summary-log",
        action="store_true",
        help=(
            "Append a compact realized-sweep summary JSONL record for later comparison. "
            "Only writes when this flag is provided."
        ),
    )
    parser.add_argument(
        "--summary-log-path",
        type=Path,
        default=DEFAULT_SUMMARY_LOG_PATH,
        help=f"JSONL summary log path used with --save-summary-log. Default: {DEFAULT_SUMMARY_LOG_PATH}.",
    )
    parser.add_argument(
        "--run-label",
        help="Optional label stored in the summary log, e.g. soft_late_sweep_15m.",
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
    if args.years <= 0:
        raise SystemExit("--years must be greater than 0")
    effective_limit = (
        args.limit
        if args.limit is not None
        else calibration_limit_for_years(args.timeframe, args.years)
    )
    print("Backtest data selection")
    print("calibration only")
    print(
        f"timeframe={args.timeframe} years={args.years:g} "
        f"candle_limit={effective_limit if effective_limit and effective_limit > 0 else 'unlimited'}"
    )
    if args.limit is not None:
        print("note=--limit overrides --years; omit --limit to use the full configured lookback")
    if args.timeframe == "1m" and (effective_limit is None or effective_limit >= calibration_limit_for_years("1m", DEFAULT_BACKTEST_YEARS)):
        print("warning=1m 3-year data is large and slow; test 15m first, then 5m, then 1m")
    calibration_min_expected_net_profit = (
        args.calibration_min_expected_net_profit
        if args.calibration_min_expected_net_profit is not None
        else settings.risk.calibration_min_expected_net_profit_usd
    )
    if calibration_min_expected_net_profit < 0:
        raise SystemExit("--calibration-min-expected-net-profit must be greater than or equal to 0")
    target_sweep = parse_float_list(
        args.target_sweep
        or format_float_list(
            DEFAULT_REALIZED_TARGET_SWEEP if args.realized_sweep else DEFAULT_TARGET_SWEEP
        )
    )
    reward_cost_sweep = parse_float_list(
        args.reward_cost_sweep
        or format_float_list(
            DEFAULT_REALIZED_REWARD_COST_SWEEP
            if args.realized_sweep
            else DEFAULT_REWARD_COST_SWEEP
        )
    )
    quality_filter_enabled = (
        args.backtest_quality_filter or args.realized_sweep
    ) and not args.disable_backtest_quality_filter
    quality_config = BacktestQualityConfig(
        exhaustion_rsi_high=args.exhaustion_rsi_high,
        exhaustion_rsi_low=args.exhaustion_rsi_low,
        exhaustion_close_position_high=args.exhaustion_close_position_high,
        exhaustion_close_position_low=args.exhaustion_close_position_low,
        exhaustion_candle_atr_multiplier=args.exhaustion_candle_atr_multiplier,
        extreme_rsi_high=args.extreme_rsi_high,
        extreme_rsi_low=args.extreme_rsi_low,
        soft_rsi_low_short=args.soft_rsi_low_short,
        soft_close_position_low_short=args.soft_close_position_low_short,
        soft_rsi_high_long=args.soft_rsi_high_long,
        soft_close_position_high_long=args.soft_close_position_high_long,
        reject_soft_late_momentum=args.reject_soft_late_momentum,
    )
    soft_rsi_high_long_sweep = (
        parse_float_list(args.soft_rsi_high_long_sweep)
        if args.soft_rsi_high_long_sweep
        else (quality_config.soft_rsi_high_long,)
    )
    soft_close_position_high_long_sweep = (
        parse_float_list(args.soft_close_position_high_long_sweep)
        if args.soft_close_position_high_long_sweep
        else (quality_config.soft_close_position_high_long,)
    )
    soft_rsi_low_short_sweep = (
        parse_float_list(args.soft_rsi_low_short_sweep)
        if args.soft_rsi_low_short_sweep
        else (quality_config.soft_rsi_low_short,)
    )
    soft_close_position_low_short_sweep = (
        parse_float_list(args.soft_close_position_low_short_sweep)
        if args.soft_close_position_low_short_sweep
        else (quality_config.soft_close_position_low_short,)
    )
    symbols = tuple(symbol.strip() for symbol in args.symbols.split(",") if symbol.strip())
    historical = load_historical_inputs(
        symbols=symbols,
        csv_mappings=args.csv,
        data_dir=args.data_dir,
        timeframe=args.timeframe,
        limit=effective_limit,
    )
    if args.realized_sweep:
        optimizer = RealizedSweepOptimizer(
            settings=settings,
            timeframe=args.timeframe,
            target_sweep=target_sweep,
            reward_cost_sweep=reward_cost_sweep,
            max_hold_sweep=parse_int_list(args.max_hold_sweep),
            atr_tp_sweep=parse_float_list(args.atr_tp_sweep),
            atr_stop_sweep=parse_float_list(args.atr_stop_sweep),
            soft_rsi_high_long_sweep=soft_rsi_high_long_sweep,
            soft_close_position_high_long_sweep=soft_close_position_high_long_sweep,
            soft_rsi_low_short_sweep=soft_rsi_low_short_sweep,
            soft_close_position_low_short_sweep=soft_close_position_low_short_sweep,
            signal_window_bars=args.signal_window_bars,
            diagnostic_notional=args.diagnostic_notional,
            calibration_min_expected_net_profit=calibration_min_expected_net_profit,
            quality_filter_enabled=quality_filter_enabled,
            trailing_exits_enabled=args.backtest_trailing_exits,
            breakeven_trigger_r=args.breakeven_trigger_r,
            trailing_atr_multiplier=args.trailing_atr_multiplier,
            quality_config=quality_config,
        )
        result = optimizer.run(historical)
        if args.save_summary_log:
            append_realized_summary_log(result, args.summary_log_path, args.run_label)
        print_realized_optimization_report(result)
        return
    calibrator = HistoricalStrategyCalibrator(
        settings=settings,
        timeframe=args.timeframe,
        signal_window_bars=args.signal_window_bars,
        target_sweep=target_sweep,
        reward_cost_sweep=reward_cost_sweep,
        diagnostic_notional=args.diagnostic_notional,
        calibration_min_expected_net_profit=calibration_min_expected_net_profit,
        max_hold_candles=args.max_hold_candles,
        quality_filter_enabled=quality_filter_enabled,
        trailing_exits_enabled=args.backtest_trailing_exits,
        breakeven_trigger_r=args.breakeven_trigger_r,
        trailing_atr_multiplier=args.trailing_atr_multiplier,
        quality_config=quality_config,
    )
    print_report(calibrator.run(historical))


if __name__ == "__main__":
    main()
