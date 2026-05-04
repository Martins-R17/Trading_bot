"""Data normalization and feature engineering utilities."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from core.models import OrderBookSnapshot


OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


class DataPreprocessor:
    """Converts raw exchange payloads into model-ready dataframes."""

    @staticmethod
    def normalize_ohlcv(raw: Any) -> pd.DataFrame:
        if isinstance(raw, pd.DataFrame):
            df = raw.copy()
        else:
            df = pd.DataFrame(raw, columns=OHLCV_COLUMNS)

        missing = [column for column in OHLCV_COLUMNS if column not in df.columns]
        if missing:
            raise ValueError(f"OHLCV data missing columns: {missing}")

        df = df[OHLCV_COLUMNS].copy()
        for column in OHLCV_COLUMNS:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        df = df.dropna().drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        df = df.reset_index(drop=True)
        return df

    @staticmethod
    def add_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add lightweight scalping features used by strategies and models."""

        if len(df) == 0:
            return df

        enriched = df.copy()
        close = enriched["close"]
        high = enriched["high"]
        low = enriched["low"]
        volume = enriched["volume"]

        enriched["return_1"] = close.pct_change().fillna(0.0)
        enriched["log_return"] = np.log(close / close.shift(1)).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        enriched["ema_fast"] = close.ewm(span=8, adjust=False).mean()
        enriched["ema_slow"] = close.ewm(span=21, adjust=False).mean()
        enriched["rolling_volatility"] = enriched["log_return"].rolling(30).std().fillna(0.0)
        enriched["volume_zscore"] = DataPreprocessor._zscore(volume, window=30)

        true_range = pd.concat(
            [
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        enriched["atr"] = true_range.rolling(14).mean().bfill().fillna(0.0)
        enriched["rsi"] = DataPreprocessor.rsi(close)
        return enriched

    @staticmethod
    def realized_volatility(df: pd.DataFrame, window: int = 30) -> float:
        if df is None or len(df) < 2:
            return 0.0
        returns = np.log(df["close"] / df["close"].shift(1)).dropna()
        if len(returns) == 0:
            return 0.0
        return float(returns.tail(window).std() or 0.0)

    @staticmethod
    def rsi(close: pd.Series, window: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50.0).clip(0, 100)

    @staticmethod
    def order_book_features(order_book: OrderBookSnapshot | None) -> dict[str, float]:
        if order_book is None:
            return {
                "spread_bps": 0.0,
                "depth_imbalance": 0.0,
                "depth_quote": 0.0,
                "mid_price": 0.0,
            }
        return {
            "spread_bps": float(order_book.spread_bps),
            "depth_imbalance": float(order_book.depth_imbalance()),
            "depth_quote": float(order_book.total_depth_quote()),
            "mid_price": float(order_book.mid_price or 0.0),
        }

    @staticmethod
    def _zscore(series: pd.Series, window: int) -> pd.Series:
        rolling_mean = series.rolling(window).mean()
        rolling_std = series.rolling(window).std().replace(0.0, np.nan)
        return ((series - rolling_mean) / rolling_std).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    @staticmethod
    def safe_float(value: Any, default: float = 0.0) -> float:
        try:
            result = float(value)
            if math.isfinite(result):
                return result
        except (TypeError, ValueError):
            pass
        return default

