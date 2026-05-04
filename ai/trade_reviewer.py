"""Optional OpenAI-backed veto review for risk-approved trade candidates."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from math import isfinite
from typing import Any

from config.settings import AITradeReviewSettings

logger = logging.getLogger(__name__)


REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "approve": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "reason": {"type": "string"},
        "risks": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 8,
        },
        "suggested_action": {"type": "string", "enum": ["approve", "reject"]},
    },
    "required": ["approve", "confidence", "reason", "risks", "suggested_action"],
}


@dataclass(slots=True)
class AITradeReviewResult:
    """Validated AI review outcome.

    Disabled reviews are represented as skipped approvals so paper trading keeps
    running exactly as it did before unless the user explicitly enables review.
    """

    approved: bool
    enabled: bool
    skipped: bool
    confidence: float
    reason: str
    risks: list[str] = field(default_factory=list)
    suggested_action: str = "reject"
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def action_label(self) -> str:
        if self.skipped:
            return "skipped"
        return "approved" if self.approved else "rejected"

    def to_metadata(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "skipped": self.skipped,
            "approved": self.approved,
            "confidence": self.confidence,
            "reason": self.reason,
            "risks": self.risks,
            "suggested_action": self.suggested_action,
        }


class OpenAITradeReviewer:
    """Veto-only OpenAI reviewer for already-approved candidate trades."""

    def __init__(self, settings: AITradeReviewSettings):
        self.settings = settings
        self._client: Any | None = None
        self._client_error = ""

    def status_message(self, paper_trade: bool) -> str:
        if not self.settings.openai_api_key:
            return "AI trade review disabled: missing OPENAI_API_KEY"
        if not self.settings.enabled:
            return "AI trade review disabled"
        if self.settings.paper_only and not paper_trade:
            return "AI trade review disabled: AI_TRADE_REVIEW_PAPER_ONLY"
        if self._client_error:
            return f"AI trade review disabled: {self._client_error}"
        return (
            "AI trade review enabled | "
            f"model={self.settings.openai_model} | "
            f"min_confidence={self.settings.min_confidence:.2f}"
        )

    def is_active(self, paper_trade: bool) -> bool:
        return (
            self.settings.enabled
            and bool(self.settings.openai_api_key)
            and (paper_trade or not self.settings.paper_only)
        )

    async def review(
        self,
        trade_summary: dict[str, Any],
        paper_trade: bool,
    ) -> AITradeReviewResult:
        if not self.is_active(paper_trade):
            return AITradeReviewResult(
                approved=True,
                enabled=False,
                skipped=True,
                confidence=1.0,
                reason=self.status_message(paper_trade),
                suggested_action="approve",
            )

        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(self._review_sync, trade_summary),
                timeout=self.settings.timeout_seconds + 1.0,
            )
        except TimeoutError:
            return self._safe_reject("ai_review_timeout")
        except Exception as exc:  # pragma: no cover - exercised in live integration.
            logger.warning("AI trade review failed safely: %s", exc)
            return self._safe_reject(f"ai_review_error:{type(exc).__name__}:{exc}")

        if result.confidence < self.settings.min_confidence:
            result.approved = False
            result.reason = (
                "ai_review_confidence_below_threshold:"
                f"{result.confidence:.3f}<{self.settings.min_confidence:.3f}; {result.reason}"
            )
            result.suggested_action = "reject"
        return result

    def _review_sync(self, trade_summary: dict[str, Any]) -> AITradeReviewResult:
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.settings.openai_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a veto-only trade risk reviewer for a paper crypto bot. "
                        "Review only the supplied candidate summary. You may approve or reject. "
                        "Do not invent trades, alter sizing, stops, targets, credentials, live flags, "
                        "or risk limits. If uncertain, reject. Return strict JSON matching the schema."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(trade_summary, sort_keys=True, separators=(",", ":")),
                },
            ],
            temperature=0,
            max_tokens=self.settings.max_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "trade_review",
                    "strict": True,
                    "schema": REVIEW_SCHEMA,
                },
            },
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("empty_ai_review_response")
        payload = json.loads(content)
        return self._validate_payload(payload)

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError as exc:
            self._client_error = "openai dependency missing"
            raise RuntimeError(self._client_error) from exc
        self._client = OpenAI(
            api_key=self.settings.openai_api_key,
            timeout=self.settings.timeout_seconds,
        )
        return self._client

    def _validate_payload(self, payload: Any) -> AITradeReviewResult:
        if not isinstance(payload, dict):
            raise ValueError("ai_review_invalid_json_object")

        approve = payload.get("approve")
        confidence = payload.get("confidence")
        reason = payload.get("reason")
        risks = payload.get("risks")
        suggested_action = payload.get("suggested_action")

        if not isinstance(approve, bool):
            raise ValueError("ai_review_invalid_approve")
        if not isinstance(confidence, (int, float)) or not isfinite(float(confidence)):
            raise ValueError("ai_review_invalid_confidence")
        confidence_value = float(confidence)
        if confidence_value < 0.0 or confidence_value > 1.0:
            raise ValueError("ai_review_confidence_out_of_range")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("ai_review_invalid_reason")
        if not isinstance(risks, list) or not all(isinstance(item, str) for item in risks):
            raise ValueError("ai_review_invalid_risks")
        if suggested_action not in {"approve", "reject"}:
            raise ValueError("ai_review_invalid_suggested_action")
        if approve and suggested_action != "approve":
            raise ValueError("ai_review_approve_action_mismatch")

        return AITradeReviewResult(
            approved=approve,
            enabled=True,
            skipped=False,
            confidence=confidence_value,
            reason=reason.strip(),
            risks=[item.strip() for item in risks if item.strip()],
            suggested_action=suggested_action,
            raw=payload,
        )

    def _safe_reject(self, reason: str) -> AITradeReviewResult:
        return AITradeReviewResult(
            approved=False,
            enabled=True,
            skipped=False,
            confidence=0.0,
            reason=reason,
            risks=["ai_review_unavailable"],
            suggested_action="reject",
        )
