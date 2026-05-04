"""Environment-backed application settings.

The bot is intentionally paper-trading first. Live execution requires two
separate environment switches so accidental real orders are hard to trigger.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dependency is in requirements.
    load_dotenv = None


def _load_env_file() -> None:
    if load_dotenv is not None:
        load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return default if raw is None or raw == "" else float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return default if raw is None or raw == "" else int(raw)


def _env_tuple(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return tuple(item.strip() for item in raw.split(",") if item.strip())


@dataclass(frozen=True)
class AppSettings:
    log_level: str = "INFO"
    loop_interval_seconds: float = 2.0
    max_iterations: int = 0
    enable_monitoring_api: bool = False
    monitoring_host: str = "127.0.0.1"
    monitoring_port: int = 8080


@dataclass(frozen=True)
class ExchangeSettings:
    exchange_id: str = "binance"
    api_key: str = ""
    secret: str = ""
    password: str = ""
    sandbox: bool = True


@dataclass(frozen=True)
class TradingSettings:
    paper_trade: bool = True
    enable_live_trading: bool = False
    symbols: tuple[str, ...] = ("BTC/USDT", "ETH/USDT")
    timeframe: str = "1m"
    ohlcv_limit: int = 200
    confidence_threshold: float = 0.8
    max_open_positions: int = 3


@dataclass(frozen=True)
class MarketDataSettings:
    use_websocket: bool = False
    websocket_url: str = ""
    order_book_limit: int = 20
    request_timeout_seconds: float = 10.0
    synthetic_data_on_error: bool = True


@dataclass(frozen=True)
class RiskSettings:
    initial_equity: float = 10_000.0
    max_risk_per_trade: float = 0.01
    max_daily_loss: float = 0.03
    max_drawdown: float = 0.08
    max_position_notional_fraction: float = 0.25
    min_leverage: float = 1.0
    max_leverage: float = 5.0
    abnormal_volatility: float = 0.06
    max_spread_bps: float = 15.0
    stop_loss_bps: float = 12.0
    take_profit_bps: float = 18.0
    fee_bps: float = 4.0
    slippage_bps: float = 2.0
    sentiment_risk_multiplier: float = 0.25


@dataclass(frozen=True)
class NewsSettings:
    api_url: str = ""
    api_key: str = ""
    refresh_seconds: float = 300.0
    lookback_minutes: int = 120


@dataclass(frozen=True)
class Settings:
    app: AppSettings
    exchange: ExchangeSettings
    trading: TradingSettings
    market_data: MarketDataSettings
    risk: RiskSettings
    news: NewsSettings


def load_settings() -> Settings:
    """Load all settings from environment variables and optional .env."""

    _load_env_file()
    return Settings(
        app=AppSettings(
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            loop_interval_seconds=_env_float("LOOP_INTERVAL_SECONDS", 2.0),
            max_iterations=_env_int("MAX_ITERATIONS", 0),
            enable_monitoring_api=_env_bool("ENABLE_MONITORING_API", False),
            monitoring_host=os.getenv("MONITORING_HOST", "127.0.0.1"),
            monitoring_port=_env_int("MONITORING_PORT", 8080),
        ),
        exchange=ExchangeSettings(
            exchange_id=os.getenv("EXCHANGE_ID", "binance"),
            api_key=os.getenv("EXCHANGE_API_KEY", ""),
            secret=os.getenv("EXCHANGE_SECRET", ""),
            password=os.getenv("EXCHANGE_PASSWORD", ""),
            sandbox=_env_bool("EXCHANGE_SANDBOX", True),
        ),
        trading=TradingSettings(
            paper_trade=_env_bool("PAPER_TRADE", True),
            enable_live_trading=_env_bool("ENABLE_LIVE_TRADING", False),
            symbols=_env_tuple("SYMBOLS", ("BTC/USDT", "ETH/USDT")),
            timeframe=os.getenv("TIMEFRAME", "1m"),
            ohlcv_limit=_env_int("OHLCV_LIMIT", 200),
            confidence_threshold=_env_float("CONFIDENCE_THRESHOLD", 0.8),
            max_open_positions=_env_int("MAX_OPEN_POSITIONS", 3),
        ),
        market_data=MarketDataSettings(
            use_websocket=_env_bool("USE_WEBSOCKET", False),
            websocket_url=os.getenv("WEBSOCKET_URL", ""),
            order_book_limit=_env_int("ORDER_BOOK_LIMIT", 20),
            request_timeout_seconds=_env_float("REQUEST_TIMEOUT_SECONDS", 10.0),
            synthetic_data_on_error=_env_bool("SYNTHETIC_DATA_ON_ERROR", True),
        ),
        risk=RiskSettings(
            initial_equity=_env_float("INITIAL_EQUITY", 10_000.0),
            max_risk_per_trade=_env_float("MAX_RISK_PER_TRADE", 0.01),
            max_daily_loss=_env_float("MAX_DAILY_LOSS", 0.03),
            max_drawdown=_env_float("MAX_DRAWDOWN", 0.08),
            max_position_notional_fraction=_env_float("MAX_POSITION_NOTIONAL_FRACTION", 0.25),
            min_leverage=_env_float("MIN_LEVERAGE", 1.0),
            max_leverage=_env_float("MAX_LEVERAGE", 5.0),
            abnormal_volatility=_env_float("ABNORMAL_VOLATILITY", 0.06),
            max_spread_bps=_env_float("MAX_SPREAD_BPS", 15.0),
            stop_loss_bps=_env_float("STOP_LOSS_BPS", 12.0),
            take_profit_bps=_env_float("TAKE_PROFIT_BPS", 18.0),
            fee_bps=_env_float("FEE_BPS", 4.0),
            slippage_bps=_env_float("SLIPPAGE_BPS", 2.0),
            sentiment_risk_multiplier=_env_float("SENTIMENT_RISK_MULTIPLIER", 0.25),
        ),
        news=NewsSettings(
            api_url=os.getenv("NEWS_API_URL", ""),
            api_key=os.getenv("NEWS_API_KEY", ""),
            refresh_seconds=_env_float("NEWS_REFRESH_SECONDS", 300.0),
            lookback_minutes=_env_int("NEWS_LOOKBACK_MINUTES", 120),
        ),
    )
