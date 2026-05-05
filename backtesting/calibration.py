"""Historical strategy edge calibration.

This module is intentionally separate from the live/paper trading loop. It scans
historical OHLCV candles and reports whether existing strategies can produce
fee-aware candidates under production thresholds and under calibration sweeps.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
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
    diagnostic_notional: float
    max_hold_candles: int = 60
    quality_filter_enabled: bool = False
    trailing_exits_enabled: bool = False
    rows: list[StrategyCalibrationStats] = field(default_factory=list)
    sweep_rows: list[SweepStats] = field(default_factory=list)
    simulation_rows: list[SimulationSummary] = field(default_factory=list)
    simulation_trades: list[SimulationTrade] = field(default_factory=list)


@dataclass(frozen=True)
class RealizedSweepConfig:
    """One calibration-only realized optimization combination."""

    timeframe: str
    min_target_move_bps: float
    min_reward_cost_ratio: float
    max_hold_candles: int
    atr_take_profit_multiplier: float
    atr_stop_loss_multiplier: float


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


@dataclass
class RealizedOptimizationResult:
    """Complete realized optimization report data."""

    production_target_move_bps: float
    production_reward_cost_ratio: float
    production_min_expected_net_profit: float
    diagnostic_notional: float
    quality_filter_enabled: bool = False
    trailing_exits_enabled: bool = False
    rows: list[RealizedOptimizationRow] = field(default_factory=list)
    losing_examples: list[SimulationTrade] = field(default_factory=list)


@dataclass(frozen=True)
class BacktestQualityDecision:
    """Backtest-only candidate quality verdict."""

    approved: bool
    reason: str
    metadata: dict[str, float | str]


class BacktestQualityFilter:
    """Experimental calibration-only filters from realized backtest failures."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

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
        macd_hist = self._latest(df, "macd_hist", 0.0)
        previous_macd_hist = self._previous(df, "macd_hist", macd_hist)
        macd_hist_bps = macd_hist / max(price, 1e-9) * 10_000
        rsi = self._latest(df, "rsi", 50.0)
        stop_move_bps = abs(signal.entry_price - float(signal.stop_loss or signal.entry_price)) / max(signal.entry_price, 1e-9) * 10_000
        target_move_bps = abs(float(signal.take_profit or signal.entry_price) - signal.entry_price) / max(signal.entry_price, 1e-9) * 10_000
        target_stop_ratio = target_move_bps / max(stop_move_bps, 1e-9)
        stop_atr_multiple = stop_move_bps / max(atr_bps, 1e-9)
        common_metadata = {
            "backtest_quality_filter": "checked",
            "backtest_quality_reason": "",
            "ema_gap_bps": ema_gap_bps,
            "trend_slope_bps": trend_slope_bps,
            "atr_expansion": atr_expansion,
            "volume_ratio": volume_ratio,
            "close_position": close_position,
            "body_bps": body_bps,
            "candle_range_bps": candle_range_bps,
            "macd_hist_bps": macd_hist_bps,
            "rsi": rsi,
            "stop_move_bps": stop_move_bps,
            "target_stop_ratio": target_stop_ratio,
            "stop_atr_multiple": stop_atr_multiple,
        }

        if candle_range_bps > max(atr_bps * 2.4, self.settings.risk.round_trip_taker_cost_bps * 2.0):
            return self._reject("exhaustion_candle", common_metadata)
        if body_bps > max(atr_bps * 1.8, self.settings.risk.round_trip_taker_cost_bps * 1.5):
            return self._reject("exhaustion_body", common_metadata)
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
        max_hold_candles: int = 60,
        quality_filter_enabled: bool = False,
        trailing_exits_enabled: bool = False,
        breakeven_trigger_r: float = 1.0,
        trailing_atr_multiplier: float = 1.0,
    ) -> None:
        self.settings = settings
        self.target_sweep = target_sweep
        self.reward_cost_sweep = reward_cost_sweep
        self.diagnostic_notional = diagnostic_notional or self._default_diagnostic_notional()
        self.max_hold_candles = max(1, int(max_hold_candles))
        self.quality_filter_enabled = quality_filter_enabled
        self.quality_filter = BacktestQualityFilter(settings)
        self.trailing_exits_enabled = trailing_exits_enabled
        self.breakeven_trigger_r = max(float(breakeven_trigger_r), 0.0)
        self.trailing_atr_multiplier = max(float(trailing_atr_multiplier), 0.0)

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
            quality_filter_enabled=self.quality_filter_enabled,
            trailing_exits_enabled=self.trailing_exits_enabled,
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
            backtest_quality_passes = True
            if production_gate.all_pass and self._is_simulatable_signal(signal):
                if self.quality_filter_enabled:
                    quality_decision = self.quality_filter.evaluate(strategy.name, window, signal)
                    if not quality_decision.approved:
                        backtest_quality_passes = False
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
        diagnostic_notional: float | None = None,
        quality_filter_enabled: bool = True,
        trailing_exits_enabled: bool = False,
        breakeven_trigger_r: float = 1.0,
        trailing_atr_multiplier: float = 1.0,
    ) -> None:
        self.settings = settings
        self.timeframe = timeframe
        self.target_sweep = target_sweep
        self.reward_cost_sweep = reward_cost_sweep
        self.max_hold_sweep = max_hold_sweep
        self.atr_tp_sweep = atr_tp_sweep
        self.atr_stop_sweep = atr_stop_sweep
        self.diagnostic_notional = diagnostic_notional
        self.quality_filter_enabled = quality_filter_enabled
        self.trailing_exits_enabled = trailing_exits_enabled
        self.breakeven_trigger_r = breakeven_trigger_r
        self.trailing_atr_multiplier = trailing_atr_multiplier

    def run(self, historical_by_symbol: dict[str, pd.DataFrame]) -> RealizedOptimizationResult:
        rows: list[RealizedOptimizationRow] = []
        all_trades: list[SimulationTrade] = []
        for config in self._configs():
            calibration_settings = self._settings_for_config(config)
            calibrator = HistoricalStrategyCalibrator(
                settings=calibration_settings,
                target_sweep=(config.min_target_move_bps,),
                reward_cost_sweep=(config.min_reward_cost_ratio,),
                diagnostic_notional=self.diagnostic_notional,
                max_hold_candles=config.max_hold_candles,
                quality_filter_enabled=self.quality_filter_enabled,
                trailing_exits_enabled=self.trailing_exits_enabled,
                breakeven_trigger_r=self.breakeven_trigger_r,
                trailing_atr_multiplier=self.trailing_atr_multiplier,
            )
            result = calibrator.run(historical_by_symbol)
            all_trades.extend(result.simulation_trades)
            trades_by_key = self._group_trades(result.simulation_trades)
            for stats in result.rows:
                trades = trades_by_key.get((stats.symbol, stats.strategy_name), [])
                rows.append(self._row_from_trades(config, stats.symbol, stats.strategy_name, trades))

        diagnostic_notional = (
            self.diagnostic_notional
            if self.diagnostic_notional is not None
            else HistoricalStrategyCalibrator(self.settings)._default_diagnostic_notional()
        )
        return RealizedOptimizationResult(
            production_target_move_bps=self.settings.risk.min_target_move_bps,
            production_reward_cost_ratio=self.settings.risk.min_reward_to_cost_ratio,
            production_min_expected_net_profit=self.settings.risk.min_expected_net_profit_usd,
            diagnostic_notional=diagnostic_notional,
            quality_filter_enabled=self.quality_filter_enabled,
            trailing_exits_enabled=self.trailing_exits_enabled,
            rows=rows,
            losing_examples=self._losing_examples(all_trades),
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
            )
            for target, reward_cost, max_hold, atr_tp, atr_stop in product(
                self.target_sweep,
                self.reward_cost_sweep,
                self.max_hold_sweep,
                self.atr_tp_sweep,
                self.atr_stop_sweep,
            )
        ]

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
        symbol: str,
        strategy_name: str,
        trades: list[SimulationTrade],
    ) -> RealizedOptimizationRow:
        sorted_trades = sorted(trades, key=lambda trade: (trade.entry_index, trade.exit_timestamp))
        exit_counts = Counter(trade.exit_reason for trade in sorted_trades)
        net_values = [trade.realized_net_pnl for trade in sorted_trades]
        gross_profit = sum(value for value in net_values if value > 0)
        gross_loss = abs(sum(value for value in net_values if value < 0))
        return RealizedOptimizationRow(
            timeframe=config.timeframe,
            symbol=symbol,
            strategy_name=strategy_name,
            min_target_move_bps=config.min_target_move_bps,
            min_reward_cost_ratio=config.min_reward_cost_ratio,
            max_hold_candles=config.max_hold_candles,
            atr_take_profit_multiplier=config.atr_take_profit_multiplier,
            atr_stop_loss_multiplier=config.atr_stop_loss_multiplier,
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
    print(f"backtest_quality_filter={'enabled' if result.quality_filter_enabled else 'disabled'}")
    print(f"backtest_trailing_exits={'enabled' if result.trailing_exits_enabled else 'disabled'}")
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
    print_losing_setup_examples(result.simulation_trades)


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


def print_realized_optimization_report(result: RealizedOptimizationResult) -> None:
    print("Realized Backtest Optimization Report")
    print("calibration only")
    print("realized historical simulation")
    print("production thresholds unchanged")
    print("temporary sweep settings do not change main.py, settings/.env, or live/paper bot behavior")
    print(f"backtest_quality_filter={'enabled' if result.quality_filter_enabled else 'disabled'}")
    print(f"backtest_trailing_exits={'enabled' if result.trailing_exits_enabled else 'disabled'}")
    print(
        f"production_min_target_move_bps={result.production_target_move_bps:.2f} | "
        f"production_min_reward_cost_ratio={result.production_reward_cost_ratio:.2f}x | "
        f"production_min_expected_net_profit=${result.production_min_expected_net_profit:.2f} | "
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
    print_losing_setup_examples(result.losing_examples)


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
        f"{'Hor%':>7} {'AvgHold':>8} {'ExitReasons':<36}"
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
            f"{row.average_hold_candles:>8.1f} {format_exit_reasons(row.exit_reason_counts):<36}"
        )


def format_profit_factor(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value == float("inf"):
        return "inf"
    return f"{value:.2f}"


def print_losing_setup_examples(trades: list[SimulationTrade]) -> None:
    losing_trades = sorted(
        [trade for trade in trades if trade.realized_net_pnl < 0],
        key=lambda trade: trade.realized_net_pnl,
    )[:8]
    print()
    print("Losing Setup Examples")
    if not losing_trades:
        print("none")
        return
    print(
        f"{'Symbol':<10} {'Strategy':<18} {'Side':<5} {'Exit':<20} {'Hold':>5} "
        f"{'Net':>10} {'RSI':>7} {'MACD':>8} {'ATRbps':>8} {'Vol':>6} "
        f"{'Close':>7} {'StopBps':>8} {'Tgt/Stop':>8} {'QualityReason':<32}"
    )
    for trade in losing_trades:
        metadata = trade.metadata
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
        default=loaded_settings.trading.timeframe,
        help="Timeframe label used for CSV discovery and reporting, e.g. 1m, 5m, 15m.",
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
        help="Optional notional used only for calibration expected net profit.",
    )
    parser.add_argument(
        "--max-hold-candles",
        type=int,
        default=60,
        help="Maximum future candles used for calibration exit simulation.",
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
    symbols = tuple(symbol.strip() for symbol in args.symbols.split(",") if symbol.strip())
    historical = load_historical_inputs(
        symbols=symbols,
        csv_mappings=args.csv,
        data_dir=args.data_dir,
        timeframe=args.timeframe,
        limit=args.limit,
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
            diagnostic_notional=args.diagnostic_notional,
            quality_filter_enabled=quality_filter_enabled,
            trailing_exits_enabled=args.backtest_trailing_exits,
            breakeven_trigger_r=args.breakeven_trigger_r,
            trailing_atr_multiplier=args.trailing_atr_multiplier,
        )
        print_realized_optimization_report(optimizer.run(historical))
        return
    calibrator = HistoricalStrategyCalibrator(
        settings=settings,
        target_sweep=target_sweep,
        reward_cost_sweep=reward_cost_sweep,
        diagnostic_notional=args.diagnostic_notional,
        max_hold_candles=args.max_hold_candles,
        quality_filter_enabled=quality_filter_enabled,
        trailing_exits_enabled=args.backtest_trailing_exits,
        breakeven_trigger_r=args.breakeven_trigger_r,
        trailing_atr_multiplier=args.trailing_atr_multiplier,
    )
    print_report(calibrator.run(historical))


if __name__ == "__main__":
    main()
