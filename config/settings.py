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


def _env_fee_rate(name: str, legacy_bps_name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is not None and raw != "":
        return float(raw)
    legacy_bps = os.getenv(legacy_bps_name)
    if legacy_bps is not None and legacy_bps != "":
        return float(legacy_bps) / 10_000
    return default


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
    enable_scalping_microstructure: bool = False
    symbols: tuple[str, ...] = ("BTC/USDT", "ETH/USDT")
    timeframe: str = "1m"
    ohlcv_limit: int = 200
    confidence_threshold: float = 0.8
    max_open_positions: int = 3
    paper_max_holding_iterations: int = 20
    skip_duplicate_market_snapshots: bool = True
    ema_trend_deadband_bps: float = 1.0
    counter_trend_rsi_overbought: float = 70.0
    counter_trend_rsi_oversold: float = 30.0
    counter_trend_macd_hist_bps: float = 1.0
    trade_history_path: str = "data/trade_history.csv"


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
    min_position_notional: float = 10.0
    min_reward_risk_ratio: float = 1.05
    min_reward_to_cost_ratio: float = 3.0
    min_reward_cost_multiple: float = 3.0
    min_expected_net_profit_usd: float = 1.0
    min_expected_net_profit: float = 1.0
    min_target_move_bps: float = 75.0
    atr_take_profit_multiplier: float = 3.0
    atr_stop_loss_multiplier: float = 1.0
    scalping_min_net_cost_multiple: float = 3.0
    scalping_min_expected_net_profit: float = 5.0
    scalping_target_cost_buffer: float = 1.5
    max_order_size_fraction_of_depth: float = 0.2
    min_leverage: float = 1.0
    max_leverage: float = 5.0
    abnormal_volatility: float = 0.06
    max_spread_bps: float = 15.0
    stop_loss_bps: float = 12.0
    take_profit_bps: float = 18.0
    maker_fee_rate: float = 0.001
    taker_fee_rate: float = 0.001
    fee_bps: float = 10.0
    slippage_bps: float = 2.0
    sentiment_risk_multiplier: float = 0.25

    @property
    def slippage_rate(self) -> float:
        return self.slippage_bps / 10_000

    @property
    def round_trip_taker_cost_rate(self) -> float:
        return 2 * self.taker_fee_rate + 2 * self.slippage_rate

    @property
    def round_trip_taker_cost_bps(self) -> float:
        return self.round_trip_taker_cost_rate * 10_000


@dataclass(frozen=True)
class NewsSettings:
    api_url: str = ""
    api_key: str = ""
    refresh_seconds: float = 300.0
    lookback_minutes: int = 120


@dataclass(frozen=True)
class AITradeReviewSettings:
    enabled: bool = False
    paper_only: bool = True
    openai_model: str = "gpt-4o-mini"
    openai_api_key: str = ""
    timeout_seconds: float = 8.0
    max_tokens: int = 256
    min_confidence: float = 0.7


@dataclass(frozen=True)
class Settings:
    app: AppSettings
    exchange: ExchangeSettings
    trading: TradingSettings
    market_data: MarketDataSettings
    risk: RiskSettings
    news: NewsSettings
    ai_trade_review: AITradeReviewSettings


def load_settings() -> Settings:
    """Load all settings from environment variables and optional .env."""

    _load_env_file()
    maker_fee_rate = _env_fee_rate("MAKER_FEE_RATE", "FEE_BPS", 0.001)
    taker_fee_rate = _env_fee_rate("TAKER_FEE_RATE", "FEE_BPS", 0.001)
    min_reward_to_cost_ratio = _env_float(
        "MIN_REWARD_TO_COST_RATIO",
        _env_float("MIN_REWARD_COST_MULTIPLE", 3.0),
    )
    min_expected_net_profit_usd = _env_float(
        "MIN_EXPECTED_NET_PROFIT_USD",
        _env_float("MIN_EXPECTED_NET_PROFIT", 1.0),
    )
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
            enable_scalping_microstructure=_env_bool("ENABLE_SCALPING_MICROSTRUCTURE", False),
            symbols=_env_tuple("SYMBOLS", ("BTC/USDT", "ETH/USDT")),
            timeframe=os.getenv("TIMEFRAME", "1m"),
            ohlcv_limit=_env_int("OHLCV_LIMIT", 200),
            confidence_threshold=_env_float("CONFIDENCE_THRESHOLD", 0.8),
            max_open_positions=_env_int("MAX_OPEN_POSITIONS", 3),
            paper_max_holding_iterations=_env_int("PAPER_MAX_HOLDING_ITERATIONS", 20),
            skip_duplicate_market_snapshots=_env_bool("SKIP_DUPLICATE_MARKET_SNAPSHOTS", True),
            ema_trend_deadband_bps=_env_float("EMA_TREND_DEADBAND_BPS", 1.0),
            counter_trend_rsi_overbought=_env_float("COUNTER_TREND_RSI_OVERBOUGHT", 70.0),
            counter_trend_rsi_oversold=_env_float("COUNTER_TREND_RSI_OVERSOLD", 30.0),
            counter_trend_macd_hist_bps=_env_float("COUNTER_TREND_MACD_HIST_BPS", 1.0),
            trade_history_path=os.getenv("TRADE_HISTORY_PATH", "data/trade_history.csv"),
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
            min_position_notional=_env_float("MIN_POSITION_NOTIONAL", 10.0),
            min_reward_risk_ratio=_env_float("MIN_REWARD_RISK_RATIO", 1.05),
            min_reward_to_cost_ratio=min_reward_to_cost_ratio,
            min_reward_cost_multiple=min_reward_to_cost_ratio,
            min_expected_net_profit_usd=min_expected_net_profit_usd,
            min_expected_net_profit=min_expected_net_profit_usd,
            min_target_move_bps=_env_float("MIN_TARGET_MOVE_BPS", 75.0),
            atr_take_profit_multiplier=_env_float("ATR_TAKE_PROFIT_MULTIPLIER", 3.0),
            atr_stop_loss_multiplier=_env_float("ATR_STOP_LOSS_MULTIPLIER", 1.0),
            scalping_min_net_cost_multiple=_env_float("SCALPING_MIN_NET_COST_MULTIPLE", 3.0),
            scalping_min_expected_net_profit=_env_float("SCALPING_MIN_EXPECTED_NET_PROFIT", 5.0),
            scalping_target_cost_buffer=_env_float("SCALPING_TARGET_COST_BUFFER", 1.5),
            max_order_size_fraction_of_depth=_env_float("MAX_ORDER_SIZE_FRACTION_OF_DEPTH", 0.2),
            min_leverage=_env_float("MIN_LEVERAGE", 1.0),
            max_leverage=_env_float("MAX_LEVERAGE", 5.0),
            abnormal_volatility=_env_float("ABNORMAL_VOLATILITY", 0.06),
            max_spread_bps=_env_float("MAX_SPREAD_BPS", 15.0),
            stop_loss_bps=_env_float("STOP_LOSS_BPS", 12.0),
            take_profit_bps=_env_float("TAKE_PROFIT_BPS", 18.0),
            maker_fee_rate=maker_fee_rate,
            taker_fee_rate=taker_fee_rate,
            fee_bps=_env_float("FEE_BPS", taker_fee_rate * 10_000),
            slippage_bps=_env_float("SLIPPAGE_BPS", 2.0),
            sentiment_risk_multiplier=_env_float("SENTIMENT_RISK_MULTIPLIER", 0.25),
        ),
        news=NewsSettings(
            api_url=os.getenv("NEWS_API_URL", ""),
            api_key=os.getenv("NEWS_API_KEY", ""),
            refresh_seconds=_env_float("NEWS_REFRESH_SECONDS", 300.0),
            lookback_minutes=_env_int("NEWS_LOOKBACK_MINUTES", 120),
        ),
        ai_trade_review=AITradeReviewSettings(
            enabled=_env_bool("ENABLE_AI_TRADE_REVIEW", False),
            paper_only=_env_bool("AI_TRADE_REVIEW_PAPER_ONLY", True),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            timeout_seconds=_env_float("AI_TRADE_REVIEW_TIMEOUT_SECONDS", 8.0),
            max_tokens=_env_int("AI_TRADE_REVIEW_MAX_TOKENS", 256),
            min_confidence=_env_float("AI_TRADE_REVIEW_MIN_CONFIDENCE", 0.7),
        ),
    )
