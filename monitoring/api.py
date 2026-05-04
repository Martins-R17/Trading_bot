"""FastAPI monitoring app for bot state."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

try:
    from fastapi import FastAPI
except ImportError:  # pragma: no cover - dependency is in requirements.
    FastAPI = None


def build_monitoring_app(state_provider: Callable[[], dict[str, Any]]) -> Any:
    if FastAPI is None:
        raise RuntimeError("fastapi is required for monitoring API")

    app = FastAPI(title="Crypto Scalping Bot Monitor", version="0.1.0")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/state")
    async def state() -> dict[str, Any]:
        return state_provider()

    return app

