"""Async market data access.

The polling service uses ccxt for public OHLCV and order book data. The
WebSocket feed is Binance-compatible and can be swapped for exchange-specific
adapters later without changing strategy or risk modules.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from collections.abc import AsyncIterator
from typing import Any

import numpy as np
import pandas as pd

try:
    import ccxt.async_support as ccxt
except ImportError:  # pragma: no cover - dependency is in requirements.
    ccxt = None

try:
    import websockets
except ImportError:  # pragma: no cover - dependency is in requirements.
    websockets = None

from config.settings import Settings
from core.models import MarketSnapshot, OrderBookSnapshot
from data.preprocess import DataPreprocessor

logger = logging.getLogger(__name__)


class MarketDataService:
    """Fetches normalized OHLCV and order book snapshots."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.preprocessor = DataPreprocessor()
        self.exchange = self._create_exchange()

    def _create_exchange(self) -> Any | None:
        if ccxt is None:
            logger.warning("ccxt is unavailable; market data will use synthetic fallback.")
            return None
        exchange_cls = getattr(ccxt, self.settings.exchange.exchange_id, None)
        if exchange_cls is None:
            logger.warning("Unknown exchange %s; using synthetic fallback.", self.settings.exchange.exchange_id)
            return None

        exchange = exchange_cls(
            {
                "apiKey": self.settings.exchange.api_key,
                "secret": self.settings.exchange.secret,
                "password": self.settings.exchange.password,
                "enableRateLimit": True,
                "timeout": int(self.settings.market_data.request_timeout_seconds * 1000),
            }
        )
        if self.settings.exchange.sandbox and hasattr(exchange, "set_sandbox_mode"):
            exchange.set_sandbox_mode(True)
        return exchange

    async def get_snapshot(self, symbol: str, sentiment_score: float = 0.0) -> MarketSnapshot:
        """Return a current market snapshot for one symbol."""

        try:
            raw_ohlcv, raw_order_book = await asyncio.gather(
                self._fetch_ohlcv(symbol),
                self._fetch_order_book(symbol),
            )
        except Exception as exc:
            if not self.settings.market_data.synthetic_data_on_error:
                raise
            logger.warning("Market data error for %s: %s. Falling back to synthetic data.", symbol, exc)
            raw_ohlcv = self._synthetic_ohlcv(symbol, self.settings.trading.ohlcv_limit)
            raw_order_book = self._synthetic_order_book(symbol, raw_ohlcv[-1][4])

        ohlcv = self.preprocessor.add_features(self.preprocessor.normalize_ohlcv(raw_ohlcv))
        order_book = self._normalize_order_book(symbol, raw_order_book)
        volatility = self.preprocessor.realized_volatility(ohlcv)
        return MarketSnapshot(
            symbol=symbol,
            timestamp=time.time(),
            ohlcv=ohlcv,
            order_book=order_book,
            sentiment_score=sentiment_score,
            volatility=volatility,
        )

    async def snapshots(self, symbols: tuple[str, ...], sentiment_by_symbol: dict[str, float]) -> list[MarketSnapshot]:
        tasks = [self.get_snapshot(symbol, sentiment_by_symbol.get(symbol, 0.0)) for symbol in symbols]
        return await asyncio.gather(*tasks)

    async def _fetch_ohlcv(self, symbol: str) -> list[list[float]]:
        if self.exchange is None:
            return self._synthetic_ohlcv(symbol, self.settings.trading.ohlcv_limit)
        return await self.exchange.fetch_ohlcv(
            symbol,
            timeframe=self.settings.trading.timeframe,
            limit=self.settings.trading.ohlcv_limit,
        )

    async def _fetch_order_book(self, symbol: str) -> dict[str, Any]:
        if self.exchange is None:
            latest_close = self._synthetic_ohlcv(symbol, 2)[-1][4]
            return self._synthetic_order_book(symbol, latest_close)
        return await self.exchange.fetch_order_book(symbol, limit=self.settings.market_data.order_book_limit)

    def _normalize_order_book(self, symbol: str, raw: dict[str, Any]) -> OrderBookSnapshot:
        bids = [(float(price), float(amount)) for price, amount in raw.get("bids", [])]
        asks = [(float(price), float(amount)) for price, amount in raw.get("asks", [])]
        return OrderBookSnapshot(
            symbol=symbol,
            bids=bids,
            asks=asks,
            timestamp=float(raw.get("timestamp") or time.time()),
            sequence=raw.get("nonce"),
        )

    def _synthetic_ohlcv(self, symbol: str, limit: int) -> list[list[float]]:
        """Generate stable paper-mode data when network dependencies are absent."""

        seed = abs(hash(symbol)) % 10_000
        rng = np.random.default_rng(seed + int(time.time() // 60))
        base_price = 70_000 if "BTC" in symbol else 3_500 if "ETH" in symbol else 100
        returns = rng.normal(loc=0.00003, scale=0.0009, size=limit)
        prices = base_price * np.exp(np.cumsum(returns))
        now_ms = int(time.time() * 1000)
        minute_ms = 60_000
        rows: list[list[float]] = []
        for i, close in enumerate(prices):
            open_ = prices[i - 1] if i else close * (1 - returns[i])
            high = max(open_, close) * (1 + abs(rng.normal(0.0004, 0.0002)))
            low = min(open_, close) * (1 - abs(rng.normal(0.0004, 0.0002)))
            volume = max(1.0, rng.normal(150, 30))
            rows.append([now_ms - (limit - i) * minute_ms, open_, high, low, close, volume])
        return rows

    def _synthetic_order_book(self, symbol: str, mid_price: float) -> dict[str, Any]:
        spread_bps = random.uniform(1.5, 6.0)
        spread = mid_price * spread_bps / 10_000
        best_bid = mid_price - spread / 2
        best_ask = mid_price + spread / 2
        bids = [[best_bid * (1 - i * 0.00005), random.uniform(0.2, 3.0)] for i in range(20)]
        asks = [[best_ask * (1 + i * 0.00005), random.uniform(0.2, 3.0)] for i in range(20)]
        return {"symbol": symbol, "bids": bids, "asks": asks, "timestamp": time.time()}

    async def close(self) -> None:
        if self.exchange is not None and hasattr(self.exchange, "close"):
            await self.exchange.close()


class BinanceWebSocketFeed:
    """Binance public WebSocket feed for real-time kline and depth snapshots."""

    def __init__(self, settings: Settings):
        if websockets is None:
            raise RuntimeError("websockets package is required for WebSocket market data")
        self.settings = settings
        self.preprocessor = DataPreprocessor()
        self._bars: dict[str, pd.DataFrame] = {}
        self._books: dict[str, OrderBookSnapshot] = {}

    async def stream(self, sentiment_by_symbol: dict[str, float]) -> AsyncIterator[MarketSnapshot]:
        url = self.settings.market_data.websocket_url or self._combined_stream_url()
        async with websockets.connect(url, ping_interval=20, ping_timeout=20) as websocket:
            async for message in websocket:
                snapshot = self._handle_message(message, sentiment_by_symbol)
                if snapshot is not None:
                    yield snapshot

    def _combined_stream_url(self) -> str:
        streams: list[str] = []
        interval = self.settings.trading.timeframe
        for symbol in self.settings.trading.symbols:
            stream_symbol = symbol.replace("/", "").lower()
            streams.append(f"{stream_symbol}@kline_{interval}")
            streams.append(f"{stream_symbol}@depth20@100ms")
        return "wss://stream.binance.com:9443/stream?streams=" + "/".join(streams)

    def _handle_message(self, message: str, sentiment_by_symbol: dict[str, float]) -> MarketSnapshot | None:
        payload = json.loads(message)
        stream = payload.get("stream", "")
        data = payload.get("data", payload)
        symbol = self._symbol_from_stream(stream, data)
        if not symbol:
            return None

        if "k" in data:
            self._update_kline(symbol, data["k"])
        elif "bids" in data and "asks" in data:
            self._books[symbol] = OrderBookSnapshot(
                symbol=symbol,
                bids=[(float(price), float(amount)) for price, amount in data["bids"]],
                asks=[(float(price), float(amount)) for price, amount in data["asks"]],
                timestamp=time.time(),
                sequence=data.get("lastUpdateId"),
            )

        bars = self._bars.get(symbol)
        if bars is None or len(bars) < 30:
            return None

        ohlcv = self.preprocessor.add_features(bars.tail(self.settings.trading.ohlcv_limit).copy())
        order_book = self._books.get(symbol)
        return MarketSnapshot(
            symbol=symbol,
            timestamp=time.time(),
            ohlcv=ohlcv,
            order_book=order_book,
            sentiment_score=sentiment_by_symbol.get(symbol, 0.0),
            volatility=self.preprocessor.realized_volatility(ohlcv),
        )

    def _update_kline(self, symbol: str, kline: dict[str, Any]) -> None:
        row = {
            "timestamp": float(kline["t"]),
            "open": float(kline["o"]),
            "high": float(kline["h"]),
            "low": float(kline["l"]),
            "close": float(kline["c"]),
            "volume": float(kline["v"]),
        }
        existing = self._bars.get(symbol)
        if existing is None:
            existing = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        existing = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
        existing = existing.drop_duplicates(subset=["timestamp"], keep="last").tail(self.settings.trading.ohlcv_limit)
        self._bars[symbol] = existing.reset_index(drop=True)

    def _symbol_from_stream(self, stream: str, data: dict[str, Any]) -> str:
        if "s" in data:
            raw = str(data["s"]).upper()
        elif stream:
            raw = stream.split("@", 1)[0].upper()
        else:
            return ""
        for symbol in self.settings.trading.symbols:
            if symbol.replace("/", "").upper() == raw:
                return symbol
        if raw.endswith("USDT"):
            return f"{raw[:-4]}/USDT"
        return raw

