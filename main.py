"""Live trading orchestration loop."""

from __future__ import annotations

import argparse
import asyncio
import logging
import threading
import time
from typing import Any

from ai.confidence_model import ConfidenceModel
from ai.sentiment_analysis import SentimentAnalyzer
from ai.strategy_selector import StrategySelector
from config.settings import Settings, load_settings
from core.models import MarketSnapshot, TradeRecord
from data.market_data import BinanceWebSocketFeed, MarketDataService
from data.news_data import NewsDataProvider
from execution.executor import TradeExecutor
from execution.order_manager import OrderManager
from risk.risk_manager import RiskManager
from strategies import (
    BreakoutStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    ScalpingMicrostructureStrategy,
)
from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class TradingBot:
    """Coordinates data, AI selection, risk, execution, and position management."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.market_data = MarketDataService(settings)
        self.news_provider = NewsDataProvider(settings.news)
        self.sentiment_analyzer = SentimentAnalyzer()
        self.strategies: list[BaseStrategy] = [
            MomentumStrategy(),
            MeanReversionStrategy(),
            BreakoutStrategy(),
            ScalpingMicrostructureStrategy(max_spread_bps=settings.risk.max_spread_bps),
        ]
        self.confidence_model = ConfidenceModel()
        self.selector = StrategySelector(
            self.strategies,
            self.confidence_model,
            confidence_threshold=settings.trading.confidence_threshold,
        )
        self.risk_manager = RiskManager(settings.risk)
        self.order_manager = OrderManager(settings.risk)
        self.executor = TradeExecutor(settings)
        self.sentiment_by_symbol = {symbol: 0.0 for symbol in settings.trading.symbols}
        self.last_news_refresh = 0.0
        self.iterations = 0
        self.last_snapshots: dict[str, MarketSnapshot] = {}

    async def run(self) -> None:
        self._validate_live_safety()
        self._start_monitoring_if_enabled()
        try:
            if self.settings.market_data.use_websocket:
                await self._run_websocket_loop()
            else:
                await self._run_polling_loop()
        finally:
            await self.close()

    async def _run_polling_loop(self) -> None:
        while not self._max_iterations_reached():
            print("Running iteration...")

            await self._refresh_sentiment_if_due()
            snapshots = await self.market_data.snapshots(self.settings.trading.symbols, self.sentiment_by_symbol)
            for snapshot in snapshots:
                await self._process_snapshot(snapshot)
            self.iterations += 1
            await asyncio.sleep(self.settings.app.loop_interval_seconds)

    async def _run_websocket_loop(self) -> None:
        feed = BinanceWebSocketFeed(self.settings)
        async for snapshot in feed.stream(self.sentiment_by_symbol):
            print("Running iteration (websocket)...")

            await self._refresh_sentiment_if_due()
            await self._process_snapshot(snapshot)
            self.iterations += 1

            if self._max_iterations_reached():
                break

    async def _process_snapshot(self, snapshot: MarketSnapshot) -> None:
        self.last_snapshots[snapshot.symbol] = snapshot
        closed_trades = self.order_manager.mark_to_market(snapshot)
        self._record_closed_trades(closed_trades)

        if self.order_manager.has_position(snapshot.symbol):
            return
        if self.risk_manager.daily_loss_limit_reached() or self.risk_manager.drawdown_limit_reached():
            logger.warning("Risk stop active; skipping new entries.")
            return

        selection = self.selector.select(snapshot)
        logger.info("Selection %s scores=%s reason=%s", snapshot.symbol, selection.strategy_scores, selection.reason)
        if not selection.approved or selection.signal is None:
            return

        decision = self.risk_manager.assess_trade(
            selection.signal,
            snapshot,
            selection.confidence,
            open_positions=self.order_manager.open_position_count,
            max_open_positions=self.settings.trading.max_open_positions,
        )
        if not decision.approved:
            logger.info("Risk rejected %s %s: %s", selection.signal.strategy_name, snapshot.symbol, decision.reason)
            return

        result = await self.executor.execute(decision)
        if result.success:
            position = self.order_manager.open_from_fill(result, decision)
            logger.info(
                "Opened %s %s amount=%.8f entry=%.4f stop=%.4f take=%.4f confidence=%.3f",
                decision.side.value,
                decision.symbol,
                decision.amount,
                result.filled_price,
                decision.stop_loss or 0.0,
                decision.take_profit or 0.0,
                decision.confidence,
            )
            if position is None:
                logger.warning("Execution succeeded but no position was opened: %s", result.message)
        else:
            logger.warning("Execution failed for %s: %s", decision.symbol, result.message)

    async def _refresh_sentiment_if_due(self) -> None:
        now = time.time()
        if now - self.last_news_refresh < self.settings.news.refresh_seconds:
            return
        items = await self.news_provider.fetch_latest()
        self.sentiment_by_symbol = self.sentiment_analyzer.score_items(items, self.settings.trading.symbols)
        self.last_news_refresh = now
        logger.info("Updated sentiment: %s", self.sentiment_by_symbol)

    def _record_closed_trades(self, trades: list[TradeRecord]) -> None:
        for trade in trades:
            self.risk_manager.record_trade(trade.realized_pnl)
            for strategy in self.strategies:
                if strategy.name == trade.strategy_name:
                    strategy.record_trade(trade.realized_pnl)
            feature_snapshot = trade.metadata.get("confidence_features")
            if feature_snapshot:
                self.confidence_model.update(feature_snapshot, trade.realized_pnl > 0)
            logger.info(
                "Closed %s %s pnl=%.4f reason=%s equity=%.2f",
                trade.side.value,
                trade.symbol,
                trade.realized_pnl,
                trade.reason,
                self.risk_manager.state.equity,
            )

    def state(self) -> dict[str, Any]:
        return {
            "equity": self.risk_manager.state.equity,
            "daily_pnl": self.risk_manager.state.daily_pnl,
            "open_positions": {
                symbol: {
                    "side": position.side.value,
                    "amount": position.amount,
                    "entry_price": position.entry_price,
                    "stop_loss": position.stop_loss,
                    "take_profit": position.take_profit,
                    "strategy": position.strategy_name,
                    "confidence": position.confidence,
                }
                for symbol, position in self.order_manager.positions.items()
            },
            "sentiment": self.sentiment_by_symbol,
            "iterations": self.iterations,
            "paper_trade": self.settings.trading.paper_trade,
        }

    def _start_monitoring_if_enabled(self) -> None:
        if not self.settings.app.enable_monitoring_api:
            return

        def run_api() -> None:
            import uvicorn

            from monitoring.api import build_monitoring_app

            app = build_monitoring_app(self.state)
            uvicorn.run(
                app,
                host=self.settings.app.monitoring_host,
                port=self.settings.app.monitoring_port,
                log_level=self.settings.app.log_level.lower(),
            )

        thread = threading.Thread(target=run_api, name="monitoring-api", daemon=True)
        thread.start()

    def _validate_live_safety(self) -> None:
        if self.settings.trading.paper_trade:
            return
        if not self.settings.trading.enable_live_trading:
            raise RuntimeError("PAPER_TRADE=false requires ENABLE_LIVE_TRADING=true")
        if not self.settings.exchange.api_key or not self.settings.exchange.secret:
            raise RuntimeError("Live trading requires exchange API credentials")

    def _max_iterations_reached(self) -> bool:
        max_iterations = self.settings.app.max_iterations
        return max_iterations > 0 and self.iterations >= max_iterations

    async def close(self) -> None:
        await self.market_data.close()
        await self.executor.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the AI crypto scalping bot.")
    parser.add_argument("--paper", action="store_true", help="Force paper trading for this run.")
    parser.add_argument("--symbols", type=str, help="Comma-separated symbols, e.g. BTC/USDT,ETH/USDT.")
    parser.add_argument("--max-iterations", type=int, help="Stop after N loop iterations. Useful for smoke tests.")
    return parser


def apply_cli_overrides(settings: Settings, args: argparse.Namespace) -> Settings:
    """Return a settings copy with safe CLI overrides."""

    from dataclasses import replace

    trading = settings.trading
    app = settings.app
    if args.paper:
        trading = replace(trading, paper_trade=True, enable_live_trading=False)
    if args.symbols:
        trading = replace(trading, symbols=tuple(item.strip() for item in args.symbols.split(",") if item.strip()))
    if args.max_iterations is not None:
        app = replace(app, max_iterations=args.max_iterations)
    return replace(settings, trading=trading, app=app)


async def async_main() -> None:
    print("=== BOT STARTING ===")

    args = build_arg_parser().parse_args()
    settings = apply_cli_overrides(load_settings(), args)
    logging.basicConfig(
        level=getattr(logging, settings.app.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    bot = TradingBot(settings)
    await bot.run()


def cli() -> None:
    asyncio.run(async_main())

if __name__ == "__main__":
    cli()

