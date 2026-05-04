"""Crypto news ingestion.

The provider accepts common JSON news formats such as NewsAPI (`articles`) and
CryptoPanic-style payloads (`results`). If no URL is configured it returns an
empty list, allowing the bot to run without external news credentials.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from config.settings import NewsSettings

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class NewsItem:
    title: str
    source: str = ""
    url: str = ""
    summary: str = ""
    published_at: float = 0.0
    symbols: tuple[str, ...] = ()

    @property
    def text(self) -> str:
        return " ".join(part for part in [self.title, self.summary] if part)


class NewsDataProvider:
    """Fetches and normalizes news articles for sentiment scoring."""

    def __init__(self, settings: NewsSettings):
        self.settings = settings

    async def fetch_latest(self) -> list[NewsItem]:
        if not self.settings.api_url:
            return []

        try:
            payload = await asyncio.to_thread(self._fetch_json)
        except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            logger.warning("Unable to fetch news: %s", exc)
            return []

        return self._parse_payload(payload)

    def _fetch_json(self) -> dict[str, Any] | list[Any]:
        headers = {"User-Agent": "crypto-scalping-bot/0.1"}
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"
            headers["X-Api-Key"] = self.settings.api_key
        request = Request(self.settings.api_url, headers=headers)
        with urlopen(request, timeout=10) as response:  # nosec - user-configured endpoint.
            return json.loads(response.read().decode("utf-8"))

    def _parse_payload(self, payload: dict[str, Any] | list[Any]) -> list[NewsItem]:
        if isinstance(payload, list):
            raw_items = payload
        else:
            raw_items = payload.get("articles") or payload.get("results") or payload.get("data") or []

        cutoff = time.time() - self.settings.lookback_minutes * 60
        items: list[NewsItem] = []
        for raw in raw_items:
            item = self._parse_item(raw)
            if item and (item.published_at == 0.0 or item.published_at >= cutoff):
                items.append(item)
        return items

    def _parse_item(self, raw: dict[str, Any]) -> NewsItem | None:
        title = raw.get("title") or raw.get("headline") or ""
        if not title:
            return None
        source = raw.get("source", "")
        if isinstance(source, dict):
            source = source.get("name", "")
        published_raw = raw.get("publishedAt") or raw.get("published_at") or raw.get("created_at") or ""
        return NewsItem(
            title=title,
            source=str(source),
            url=raw.get("url") or raw.get("link") or "",
            summary=raw.get("description") or raw.get("summary") or raw.get("body") or "",
            published_at=self._parse_time(published_raw),
            symbols=self._extract_symbols(raw),
        )

    def _parse_time(self, value: Any) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        if not value:
            return 0.0
        try:
            normalized = str(value).replace("Z", "+00:00")
            return datetime.fromisoformat(normalized).astimezone(timezone.utc).timestamp()
        except ValueError:
            return 0.0

    def _extract_symbols(self, raw: dict[str, Any]) -> tuple[str, ...]:
        currencies = raw.get("currencies") or raw.get("symbols") or []
        symbols: list[str] = []
        for item in currencies:
            if isinstance(item, dict):
                code = item.get("code") or item.get("symbol") or item.get("slug")
            else:
                code = str(item)
            if code:
                symbols.append(str(code).upper())
        return tuple(symbols)

