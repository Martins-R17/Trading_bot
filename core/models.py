"""Core trading domain objects.

These dataclasses keep module boundaries explicit: strategies emit signals,
risk converts signals into trade decisions, execution returns fills, and the
order manager turns fills into positions and trade records.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4


class Side(str, Enum):
    """Canonical trade side values."""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"

    @property
    def direction(self) -> int:
        if self == Side.BUY:
            return 1
        if self == Side.SELL:
            return -1
        return 0


@dataclass(slots=True)
class OrderBookSnapshot:
    """Top-of-book and depth information for one symbol."""

    symbol: str
    bids: list[tuple[float, float]]
    asks: list[tuple[float, float]]
    timestamp: float
    sequence: int | None = None

    @property
    def best_bid(self) -> float | None:
        return self.bids[0][0] if self.bids else None

    @property
    def best_ask(self) -> float | None:
        return self.asks[0][0] if self.asks else None

    @property
    def mid_price(self) -> float | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return (self.best_bid + self.best_ask) / 2.0

    @property
    def spread(self) -> float | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return max(self.best_ask - self.best_bid, 0.0)

    @property
    def spread_bps(self) -> float:
        mid = self.mid_price
        if not mid:
            return 0.0
        return ((self.spread or 0.0) / mid) * 10_000

    def depth_imbalance(self, levels: int = 5) -> float:
        """Return bid-vs-ask depth imbalance in [-1, 1]."""

        bid_depth = sum(amount for _, amount in self.bids[:levels])
        ask_depth = sum(amount for _, amount in self.asks[:levels])
        total = bid_depth + ask_depth
        if total <= 0:
            return 0.0
        return (bid_depth - ask_depth) / total

    def total_depth_quote(self, levels: int = 5) -> float:
        bid_quote = sum(price * amount for price, amount in self.bids[:levels])
        ask_quote = sum(price * amount for price, amount in self.asks[:levels])
        return bid_quote + ask_quote


@dataclass(slots=True)
class MarketSnapshot:
    """Normalized market state passed to strategies and AI models."""

    symbol: str
    timestamp: float
    ohlcv: Any
    order_book: OrderBookSnapshot | None = None
    sentiment_score: float = 0.0
    volatility: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def close(self) -> float:
        if self.ohlcv is None or len(self.ohlcv) == 0:
            return 0.0
        return float(self.ohlcv["close"].iloc[-1])

    @property
    def high(self) -> float:
        if self.ohlcv is None or len(self.ohlcv) == 0:
            return self.close
        return float(self.ohlcv["high"].iloc[-1])

    @property
    def low(self) -> float:
        if self.ohlcv is None or len(self.ohlcv) == 0:
            return self.close
        return float(self.ohlcv["low"].iloc[-1])


@dataclass(slots=True)
class StrategySignal:
    """A strategy's proposed directional trade."""

    strategy_name: str
    symbol: str
    side: Side
    strength: float
    entry_price: float
    stop_loss: float | None
    take_profit: float | None
    confidence_hint: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_actionable(self) -> bool:
        return self.side != Side.HOLD and self.entry_price > 0 and self.strength > 0


@dataclass(slots=True)
class CandidateDiagnostics:
    """Per-strategy candidate details used for missed-opportunity reporting."""

    symbol: str
    strategy_name: str
    side: Side
    side_considered: str = "hold"
    confidence: float = 0.0
    market_score: float = 0.0
    performance_score: float = 0.0
    final_score: float = 0.0
    actionable: bool = False
    rejection_reason: str = ""
    detailed_rejection_reason: str = ""
    rsi_check: str = "not_checked"
    ema_trend_check: str = "not_checked"
    macd_check: str = "not_checked"
    volatility_atr_check: str = "not_checked"
    target_move_check: str = "not_checked"
    reward_cost_check: str = "not_checked"
    expected_net_profit_check: str = "not_checked"
    expected_gross_reward: float = 0.0
    estimated_costs: float = 0.0
    expected_net_profit: float = 0.0
    target_move_bps: float = 0.0
    reward_cost_ratio: float = 0.0
    required_target_move_bps: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SelectionResult:
    """Best strategy signal selected by the AI selector."""

    signal: StrategySignal | None
    confidence: float
    strategy_scores: dict[str, float]
    approved: bool
    reason: str
    rejections: dict[str, str] = field(default_factory=dict)
    candidate_diagnostics: list[CandidateDiagnostics] = field(default_factory=list)


@dataclass(slots=True)
class RiskDecision:
    """Risk-approved trade parameters."""

    approved: bool
    reason: str
    symbol: str
    side: Side = Side.HOLD
    amount: float = 0.0
    entry_price: float = 0.0
    stop_loss: float | None = None
    take_profit: float | None = None
    leverage: float = 1.0
    notional: float = 0.0
    margin_required: float = 0.0
    confidence: float = 0.0
    strategy_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OrderRequest:
    """Exchange order request."""

    symbol: str
    side: Side
    amount: float
    order_type: str = "market"
    price: float | None = None
    reduce_only: bool = False
    client_order_id: str = field(default_factory=lambda: uuid4().hex)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OrderResult:
    """Exchange or paper execution result."""

    request: OrderRequest
    success: bool
    filled_price: float
    filled_amount: float
    fee: float
    slippage: float
    exchange_order_id: str | None = None
    message: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def notional(self) -> float:
        return abs(self.filled_price * self.filled_amount)


@dataclass(slots=True)
class Position:
    """Open trading position."""

    symbol: str
    side: Side
    amount: float
    entry_price: float
    stop_loss: float | None
    take_profit: float | None
    opened_at: float
    strategy_name: str
    confidence: float
    leverage: float = 1.0
    fees_paid: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def unrealized_pnl(self, current_price: float) -> float:
        return (current_price - self.entry_price) * self.amount * self.side.direction

    def should_close(self, high: float, low: float) -> tuple[bool, str, float | None]:
        if self.side == Side.BUY:
            if self.stop_loss is not None and low <= self.stop_loss:
                return True, "stop_loss", self.stop_loss
            if self.take_profit is not None and high >= self.take_profit:
                return True, "take_profit", self.take_profit
        elif self.side == Side.SELL:
            if self.stop_loss is not None and high >= self.stop_loss:
                return True, "stop_loss", self.stop_loss
            if self.take_profit is not None and low <= self.take_profit:
                return True, "take_profit", self.take_profit
        return False, "", None


@dataclass(slots=True)
class TradeRecord:
    """Completed trade record used for metrics and online feedback."""

    symbol: str
    side: Side
    amount: float
    entry_price: float
    exit_price: float
    opened_at: float
    closed_at: float
    realized_pnl: float
    gross_pnl: float
    fees: float
    slippage_costs: float
    total_costs: float
    reason: str
    strategy_name: str
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)

