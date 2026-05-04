"""Backtesting and live-trading performance metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.models import TradeRecord


def sharpe_ratio(returns: pd.Series | list[float], periods_per_year: int = 365 * 24 * 60) -> float:
    series = pd.Series(returns, dtype=float).dropna()
    if len(series) < 2:
        return 0.0
    std = float(series.std())
    if std == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * series.mean() / std)


def max_drawdown(equity_curve: pd.Series | list[float]) -> float:
    equity = pd.Series(equity_curve, dtype=float).dropna()
    if len(equity) == 0:
        return 0.0
    running_max = equity.cummax()
    drawdown = (running_max - equity) / running_max.replace(0.0, np.nan)
    return float(drawdown.max() or 0.0)


def win_rate(trades: list[TradeRecord]) -> float:
    if not trades:
        return 0.0
    wins = sum(1 for trade in trades if trade.realized_pnl > 0)
    return wins / len(trades)


def profit_factor(trades: list[TradeRecord]) -> float:
    gross_profit = sum(trade.realized_pnl for trade in trades if trade.realized_pnl > 0)
    gross_loss = abs(sum(trade.realized_pnl for trade in trades if trade.realized_pnl < 0))
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return float(gross_profit / gross_loss)


def summarize(trades: list[TradeRecord], equity_curve: list[float]) -> dict[str, float]:
    equity = pd.Series(equity_curve, dtype=float)
    returns = equity.pct_change().fillna(0.0)
    return {
        "trades": float(len(trades)),
        "total_pnl": float(sum(trade.realized_pnl for trade in trades)),
        "win_rate": float(win_rate(trades)),
        "profit_factor": float(profit_factor(trades)),
        "sharpe_ratio": float(sharpe_ratio(returns)),
        "max_drawdown": float(max_drawdown(equity)),
        "ending_equity": float(equity.iloc[-1]) if len(equity) else 0.0,
    }

