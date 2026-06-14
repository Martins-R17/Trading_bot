"""Backtesting-only access to optional crypto-native cached data.

The providers in this module never perform network requests and never require
API credentials. They return empty data with status metadata when a cache is
disabled, missing, unreadable, or malformed so backtests can continue safely.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

CryptoNativeMetric = Literal[
    "funding_rate",
    "open_interest",
    "liquidation_spikes",
    "long_short_ratio",
]
ProviderState = Literal[
    "ready",
    "disabled",
    "cache_missing",
    "cache_unreadable",
    "cache_invalid",
]

_CACHE_FILENAMES: dict[CryptoNativeMetric, str] = {
    "funding_rate": "funding_rate.json",
    "open_interest": "open_interest.json",
    "liquidation_spikes": "liquidation_spikes.json",
    "long_short_ratio": "long_short_ratio.json",
}


@dataclass(frozen=True, slots=True)
class ProviderStatus:
    """Serializable status for one optional provider."""

    metric: CryptoNativeMetric
    status: ProviderState
    enabled: bool
    available: bool
    source: str
    cache_path: str = ""
    record_count: int = 0
    message: str = ""
    requires_api_key: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "enabled": self.enabled,
            "available": self.available,
            "source": self.source,
            "requires_api_key": self.requires_api_key,
            "cache_path": self.cache_path,
            "record_count": self.record_count,
            "message": self.message,
        }


class CryptoNativeProvider(Protocol):
    """Common interface for backtesting crypto-native data providers."""

    metric: CryptoNativeMetric

    def load(self, symbol: str | None = None) -> list[dict[str, Any]]:
        """Return cached records, optionally filtered by symbol."""

    def status_metadata(self) -> dict[str, Any]:
        """Return non-sensitive provider diagnostics."""


class NoOpCryptoNativeProvider:
    """Disabled provider that always returns an empty result."""

    def __init__(self, metric: CryptoNativeMetric, message: str = "optional cache not configured"):
        self.metric = metric
        self._message = message

    def load(self, symbol: str | None = None) -> list[dict[str, Any]]:
        del symbol
        return []

    def status_metadata(self) -> dict[str, Any]:
        return ProviderStatus(
            metric=self.metric,
            status="disabled",
            enabled=False,
            available=False,
            source="no_op",
            message=self._message,
        ).as_dict()


class CachedJsonCryptoNativeProvider:
    """Lazy, in-memory view of a local JSON cache."""

    def __init__(
        self,
        metric: CryptoNativeMetric,
        cache_path: str | Path,
        *,
        cache_keys: tuple[str, ...] = (),
    ):
        self.metric = metric
        self.cache_path = Path(cache_path)
        self.cache_keys = (metric, *cache_keys, "records", "data", "items")
        self._loaded = False
        self._records: tuple[dict[str, Any], ...] = ()
        self._status: ProviderStatus | None = None

    def load(self, symbol: str | None = None) -> list[dict[str, Any]]:
        self._ensure_loaded()
        records = self._records
        if symbol:
            records = tuple(record for record in records if _matches_symbol(record, symbol))
        return [dict(record) for record in records]

    def status_metadata(self) -> dict[str, Any]:
        self._ensure_loaded()
        assert self._status is not None
        return self._status.as_dict()

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        cache_path = str(self.cache_path)

        if not self.cache_path.is_file():
            self._status = ProviderStatus(
                metric=self.metric,
                status="cache_missing",
                enabled=True,
                available=False,
                source="local_cache",
                cache_path=cache_path,
                message="cache file not found; provider is using an empty fallback",
            )
            return

        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            self._status = ProviderStatus(
                metric=self.metric,
                status="cache_unreadable",
                enabled=True,
                available=False,
                source="local_cache",
                cache_path=cache_path,
                message=f"cache could not be loaded; provider is using an empty fallback: {exc}",
            )
            return

        records, recognized = self._extract_records(payload)
        if not recognized:
            self._status = ProviderStatus(
                metric=self.metric,
                status="cache_invalid",
                enabled=True,
                available=False,
                source="local_cache",
                cache_path=cache_path,
                message="cache payload has no supported record collection",
            )
            return

        self._records = tuple(records)
        self._status = ProviderStatus(
            metric=self.metric,
            status="ready",
            enabled=True,
            available=True,
            source="local_cache",
            cache_path=cache_path,
            record_count=len(records),
            message="local cache loaded",
        )

    def _extract_records(self, payload: Any) -> tuple[list[dict[str, Any]], bool]:
        if isinstance(payload, list):
            return _normalize_records(payload), True
        if not isinstance(payload, Mapping):
            return [], False

        for key in self.cache_keys:
            if key in payload:
                return _normalize_collection(payload[key])

        # A root object keyed by symbols is also a supported cache shape.
        return _normalize_symbol_map(payload)


class FundingRateProvider(CachedJsonCryptoNativeProvider):
    def __init__(self, cache_path: str | Path):
        super().__init__("funding_rate", cache_path, cache_keys=("funding_rates",))


class OpenInterestProvider(CachedJsonCryptoNativeProvider):
    def __init__(self, cache_path: str | Path):
        super().__init__("open_interest", cache_path, cache_keys=("open_interests",))


class LiquidationSpikesProvider(CachedJsonCryptoNativeProvider):
    def __init__(self, cache_path: str | Path):
        super().__init__(
            "liquidation_spikes",
            cache_path,
            cache_keys=("liquidations", "liquidation_events"),
        )


class LongShortRatioProvider(CachedJsonCryptoNativeProvider):
    def __init__(self, cache_path: str | Path):
        super().__init__("long_short_ratio", cache_path, cache_keys=("long_short_ratios",))


@dataclass(slots=True)
class CryptoNativeProviders:
    """Container for all optional backtesting-only providers."""

    funding_rate: CryptoNativeProvider = field(
        default_factory=lambda: NoOpCryptoNativeProvider("funding_rate")
    )
    open_interest: CryptoNativeProvider = field(
        default_factory=lambda: NoOpCryptoNativeProvider("open_interest")
    )
    liquidation_spikes: CryptoNativeProvider = field(
        default_factory=lambda: NoOpCryptoNativeProvider("liquidation_spikes")
    )
    long_short_ratio: CryptoNativeProvider = field(
        default_factory=lambda: NoOpCryptoNativeProvider("long_short_ratio")
    )

    def load(self, symbol: str | None = None) -> dict[str, list[dict[str, Any]]]:
        return {
            "funding_rate": self.funding_rate.load(symbol),
            "open_interest": self.open_interest.load(symbol),
            "liquidation_spikes": self.liquidation_spikes.load(symbol),
            "long_short_ratio": self.long_short_ratio.load(symbol),
        }

    def status_metadata(self) -> dict[str, Any]:
        return {
            "funding_rate": self.funding_rate.status_metadata(),
            "open_interest": self.open_interest.status_metadata(),
            "liquidation_spikes": self.liquidation_spikes.status_metadata(),
            "long_short_ratio": self.long_short_ratio.status_metadata(),
            "note": "Backtesting-only local caches; missing crypto-native data is nonfatal.",
        }


def build_crypto_native_providers(
    cache_dir: str | Path | None = None,
    *,
    funding_rate_cache: str | Path | None = None,
    open_interest_cache: str | Path | None = None,
    liquidation_spikes_cache: str | Path | None = None,
    long_short_ratio_cache: str | Path | None = None,
) -> CryptoNativeProviders:
    """Build providers from explicit cache files or conventional cache names."""

    directory = Path(cache_dir) if cache_dir is not None else None
    explicit_paths: dict[CryptoNativeMetric, str | Path | None] = {
        "funding_rate": funding_rate_cache,
        "open_interest": open_interest_cache,
        "liquidation_spikes": liquidation_spikes_cache,
        "long_short_ratio": long_short_ratio_cache,
    }

    paths: dict[CryptoNativeMetric, Path | None] = {}
    for metric, explicit_path in explicit_paths.items():
        if explicit_path is not None:
            paths[metric] = Path(explicit_path)
        elif directory is not None:
            paths[metric] = directory / _CACHE_FILENAMES[metric]
        else:
            paths[metric] = None

    return CryptoNativeProviders(
        funding_rate=_provider_or_no_op("funding_rate", paths["funding_rate"]),
        open_interest=_provider_or_no_op("open_interest", paths["open_interest"]),
        liquidation_spikes=_provider_or_no_op(
            "liquidation_spikes", paths["liquidation_spikes"]
        ),
        long_short_ratio=_provider_or_no_op("long_short_ratio", paths["long_short_ratio"]),
    )


def crypto_native_status_metadata(
    providers: CryptoNativeProviders | None = None,
) -> dict[str, Any]:
    """Return status metadata without requiring callers to configure providers."""

    return (providers or CryptoNativeProviders()).status_metadata()


def _provider_or_no_op(
    metric: CryptoNativeMetric,
    cache_path: Path | None,
) -> CryptoNativeProvider:
    if cache_path is None:
        return NoOpCryptoNativeProvider(metric)
    provider_types = {
        "funding_rate": FundingRateProvider,
        "open_interest": OpenInterestProvider,
        "liquidation_spikes": LiquidationSpikesProvider,
        "long_short_ratio": LongShortRatioProvider,
    }
    return provider_types[metric](cache_path)


def _normalize_collection(value: Any) -> tuple[list[dict[str, Any]], bool]:
    if isinstance(value, list):
        return _normalize_records(value), True
    if isinstance(value, Mapping):
        return _normalize_symbol_map(value)
    return [], False


def _normalize_symbol_map(value: Mapping[Any, Any]) -> tuple[list[dict[str, Any]], bool]:
    records: list[dict[str, Any]] = []
    recognized = False
    for symbol, rows in value.items():
        if not isinstance(rows, list):
            continue
        recognized = True
        for row in _normalize_records(rows):
            row.setdefault("symbol", str(symbol))
            records.append(row)
    return records, recognized


def _normalize_records(rows: list[Any]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def _matches_symbol(record: Mapping[str, Any], requested_symbol: str) -> bool:
    record_symbol = record.get("symbol") or record.get("market") or record.get("pair")
    if not record_symbol:
        return True
    return _canonical_symbol(str(record_symbol)) == _canonical_symbol(requested_symbol)


def _canonical_symbol(symbol: str) -> str:
    return "".join(character for character in symbol.upper() if character.isalnum())
