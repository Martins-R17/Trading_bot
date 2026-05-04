"""Exchange and paper execution adapter."""

from __future__ import annotations

import logging
from typing import Any

try:
    import ccxt.async_support as ccxt
except ImportError:  # pragma: no cover - dependency is in requirements.
    ccxt = None

from config.settings import Settings
from core.models import OrderRequest, OrderResult, RiskDecision, Side

logger = logging.getLogger(__name__)


class TradeExecutor:
    """Executes approved risk decisions through paper or live exchange mode."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.exchange = self._create_exchange() if self.live_enabled else None

    @property
    def live_enabled(self) -> bool:
        return self.settings.trading.enable_live_trading and not self.settings.trading.paper_trade

    async def execute(self, decision: RiskDecision) -> OrderResult:
        request = OrderRequest(
            symbol=decision.symbol,
            side=decision.side,
            amount=decision.amount,
            order_type="market",
            price=decision.entry_price,
            metadata={"strategy": decision.strategy_name, "confidence": decision.confidence},
        )

        if not decision.approved:
            return OrderResult(request, False, 0.0, 0.0, 0.0, 0.0, message=decision.reason)
        if not self.live_enabled:
            return self._paper_fill(request, decision.entry_price)
        return await self._live_order(request)

    def _paper_fill(self, request: OrderRequest, reference_price: float) -> OrderResult:
        slippage = reference_price * self.settings.risk.slippage_bps / 10_000
        if request.side == Side.BUY:
            fill_price = reference_price + slippage
        else:
            fill_price = reference_price - slippage
        fee = abs(fill_price * request.amount) * self.settings.risk.taker_fee_rate
        return OrderResult(
            request=request,
            success=True,
            filled_price=float(fill_price),
            filled_amount=float(request.amount),
            fee=float(fee),
            slippage=float(abs(fill_price - reference_price)),
            exchange_order_id=f"paper-{request.client_order_id}",
            message="paper_fill",
        )

    async def _live_order(self, request: OrderRequest) -> OrderResult:
        if self.exchange is None:
            raise RuntimeError("Live trading requested but ccxt exchange is not available")
        raw = await self.exchange.create_order(
            request.symbol,
            request.order_type,
            request.side.value,
            request.amount,
            request.price if request.order_type == "limit" else None,
            {"clientOrderId": request.client_order_id},
        )
        filled_price = float(raw.get("average") or raw.get("price") or request.price or 0.0)
        filled_amount = float(raw.get("filled") or request.amount)
        fee = self._extract_fee(raw)
        return OrderResult(
            request=request,
            success=True,
            filled_price=filled_price,
            filled_amount=filled_amount,
            fee=fee,
            slippage=abs(filled_price - float(request.price or filled_price)),
            exchange_order_id=str(raw.get("id", "")),
            message=str(raw.get("status", "submitted")),
            raw=raw,
        )

    def _extract_fee(self, raw: dict[str, Any]) -> float:
        fee = raw.get("fee") or {}
        if isinstance(fee, dict) and fee.get("cost") is not None:
            return float(fee["cost"])
        fees = raw.get("fees") or []
        return float(sum(float(item.get("cost") or 0.0) for item in fees if isinstance(item, dict)))

    def _create_exchange(self) -> Any:
        if ccxt is None:
            raise RuntimeError("ccxt is required for live exchange execution")
        exchange_cls = getattr(ccxt, self.settings.exchange.exchange_id, None)
        if exchange_cls is None:
            raise RuntimeError(f"Unsupported exchange: {self.settings.exchange.exchange_id}")
        exchange = exchange_cls(
            {
                "apiKey": self.settings.exchange.api_key,
                "secret": self.settings.exchange.secret,
                "password": self.settings.exchange.password,
                "enableRateLimit": True,
            }
        )
        if self.settings.exchange.sandbox and hasattr(exchange, "set_sandbox_mode"):
            exchange.set_sandbox_mode(True)
        return exchange

    async def close(self) -> None:
        if self.exchange is not None and hasattr(self.exchange, "close"):
            await self.exchange.close()

