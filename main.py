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
        self.order_manager = OrderManager(
            settings.risk,
            trade_history_path=settings.trading.trade_history_path,
        )
        self.executor = TradeExecutor(settings)

        self.sentiment_by_symbol = {
            symbol: 0.0 for symbol in settings.trading.symbols
        }
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
            await self._refresh_sentiment_if_due()

            snapshots = await self.market_data.snapshots(
                self.settings.trading.symbols,
                self.sentiment_by_symbol,
            )

            for snapshot in snapshots:
                await self._process_snapshot(snapshot)

            self._print_dashboard()

            self.iterations += 1
            await asyncio.sleep(self.settings.app.loop_interval_seconds)

    async def _run_websocket_loop(self) -> None:
        feed = BinanceWebSocketFeed(self.settings)

        async for snapshot in feed.stream(self.sentiment_by_symbol):
            await self._refresh_sentiment_if_due()
            await self._process_snapshot(snapshot)

            self._print_dashboard()

            self.iterations += 1

            if self._max_iterations_reached():
                break

    def _print_dashboard(self) -> None:
        mark_prices = self._mark_prices()
        unrealized_by_symbol = self.order_manager.unrealized_pnl_by_symbol(mark_prices)
        unrealized_pnl = sum(unrealized_by_symbol.values())
        realized_equity = self.risk_manager.state.equity
        total_equity = realized_equity + unrealized_pnl
        max_iterations = self.settings.app.max_iterations
        iteration_label = f"{self.iterations + 1}/{max_iterations}" if max_iterations > 0 else f"{self.iterations + 1}"
        mode = "PAPER" if self.settings.trading.paper_trade else "LIVE"

        print()
        print("=" * 96)
        print(
            f"{mode} | Iteration {iteration_label} | "
            f"Balance {self._money(realized_equity)} | "
            f"Equity {self._money(total_equity)} | "
            f"Realized {self._signed_money(self.risk_manager.state.realized_pnl)} | "
            f"Unrealized {self._signed_money(unrealized_pnl)} | "
            f"Daily {self._signed_money(self.risk_manager.state.daily_pnl)}"
        )
        print("-" * 96)

        self._print_market_dashboard()
        self._print_position_dashboard(mark_prices, unrealized_by_symbol)
        self._print_trade_dashboard()
        print("=" * 96)

    def _print_market_dashboard(self) -> None:
        if not self.last_snapshots:
            print("Market: waiting for snapshots")
            return

        print("Market")
        print(
            f"{'Symbol':<10} {'Close':>12} {'Spread bps':>10} {'RSI':>7} "
            f"{'EMA':>6} {'MACD Hist':>12} {'Vol':>8}"
        )
        for symbol in self.settings.trading.symbols:
            snapshot = self.last_snapshots.get(symbol)
            if snapshot is None:
                continue
            spread_bps = snapshot.order_book.spread_bps if snapshot.order_book is not None else 0.0
            rsi = self._latest_column(snapshot, "rsi", 50.0)
            ema_fast = self._latest_column(snapshot, "ema_fast", snapshot.close)
            ema_slow = self._latest_column(snapshot, "ema_slow", snapshot.close)
            ema_trend = "up" if ema_fast > ema_slow else "down" if ema_fast < ema_slow else "flat"
            macd_hist = self._latest_column(snapshot, "macd_hist", 0.0)
            print(
                f"{symbol:<10} {snapshot.close:>12.4f} {spread_bps:>10.2f} "
                f"{rsi:>7.2f} {ema_trend:>6} {macd_hist:>12.6f} {snapshot.volatility:>8.4f}"
            )

    def _print_position_dashboard(
        self,
        mark_prices: dict[str, float],
        unrealized_by_symbol: dict[str, float],
    ) -> None:
        print("Positions")
        if not self.order_manager.positions:
            print("none")
            return

        print(
            f"{'Symbol':<10} {'Side':<5} {'Amount':>12} {'Entry':>12} "
            f"{'Mark':>12} {'U-PnL':>12} {'Held':>6} {'Stop':>12} {'Take':>12}"
        )
        for position in self.order_manager.positions.values():
            mark_price = mark_prices.get(position.symbol, position.entry_price)
            unrealized = unrealized_by_symbol.get(position.symbol, 0.0)
            held_iterations = self._held_iterations(position.metadata)
            print(
                f"{position.symbol:<10} {position.side.value:<5} {position.amount:>12.8f} "
                f"{position.entry_price:>12.4f} {mark_price:>12.4f} "
                f"{self._signed_money(unrealized):>12} {held_iterations:>6} "
                f"{self._price_or_dash(position.stop_loss):>12} {self._price_or_dash(position.take_profit):>12}"
            )

    def _print_trade_dashboard(self) -> None:
        closed_count = len(self.order_manager.closed_trades)
        wins = sum(1 for trade in self.order_manager.closed_trades if trade.realized_pnl > 0)
        win_rate = wins / closed_count * 100 if closed_count else 0.0
        print("Trades")
        print(
            f"closed={closed_count} | realized={self._signed_money(self.risk_manager.state.realized_pnl)} | "
            f"win_rate={win_rate:.1f}% | "
            f"history={self.order_manager.trade_history_path}"
        )
        recent_trades = self.order_manager.closed_trades[-3:]
        for trade in recent_trades:
            print(
                f"closed {trade.symbol} {trade.side.value} "
                f"pnl={self._signed_money(trade.realized_pnl)} "
                f"reason={trade.reason} strategy={trade.strategy_name}"
            )

    def _mark_prices(self) -> dict[str, float]:
        return {
            symbol: snapshot.close
            for symbol, snapshot in self.last_snapshots.items()
            if snapshot.close > 0
        }

    def _latest_column(self, snapshot: MarketSnapshot, column: str, default: float) -> float:
        if snapshot.ohlcv is None or column not in snapshot.ohlcv.columns or len(snapshot.ohlcv) == 0:
            return default
        try:
            return float(snapshot.ohlcv[column].iloc[-1])
        except (TypeError, ValueError):
            return default

    def _money(self, value: float) -> str:
        return f"${value:,.2f}"

    def _signed_money(self, value: float) -> str:
        sign = "+" if value >= 0 else "-"
        return f"{sign}${abs(value):,.2f}"

    def _price_or_dash(self, value: float | None) -> str:
        if value is None:
            return "-"
        return f"{value:.4f}"

    def _held_iterations(self, metadata: dict[str, Any]) -> int:
        opened_iteration = metadata.get("opened_iteration")
        if opened_iteration is None:
            return 0
        try:
            return max(self.iterations - int(opened_iteration), 0)
        except (TypeError, ValueError):
            return 0

    async def _process_snapshot(self, snapshot: MarketSnapshot) -> None:
        self.last_snapshots[snapshot.symbol] = snapshot

        closed_trades = self.order_manager.mark_to_market(snapshot)
        self._record_closed_trades(closed_trades)

        paper_max_hold_trade = self._close_paper_position_if_max_held(snapshot)
        if paper_max_hold_trade is not None:
            self._record_closed_trades([paper_max_hold_trade])

        if self.order_manager.has_position(snapshot.symbol):
            return

        if self.risk_manager.daily_loss_limit_reached() or self.risk_manager.drawdown_limit_reached():
            logger.warning("Risk stop active; skipping new entries.")
            return

        selection = self.selector.select(snapshot)

        logger.debug(
            "Selection %s scores=%s reason=%s",
            snapshot.symbol,
            selection.strategy_scores,
            selection.reason,
        )

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
            logger.debug(
                "Risk rejected %s %s: %s",
                selection.signal.strategy_name,
                snapshot.symbol,
                decision.reason,
            )
            return

        result = await self.executor.execute(decision)

        if result.success:
            position = self.order_manager.open_from_fill(
                result,
                decision,
                opened_iteration=self.iterations if self.settings.trading.paper_trade else None,
            )

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
                logger.warning(
                    "Execution succeeded but no position was opened: %s",
                    result.message,
                )
        else:
            logger.warning(
                "Execution failed for %s: %s",
                decision.symbol,
                result.message,
            )

    async def _refresh_sentiment_if_due(self) -> None:
        now = time.time()

        if now - self.last_news_refresh < self.settings.news.refresh_seconds:
            return

        items = await self.news_provider.fetch_latest()
        self.sentiment_by_symbol = self.sentiment_analyzer.score_items(
            items,
            self.settings.trading.symbols,
        )

        self.last_news_refresh = now
        logger.debug("Updated sentiment: %s", self.sentiment_by_symbol)

    def _record_closed_trades(self, trades: list[TradeRecord]) -> None:
        for trade in trades:
            self.risk_manager.record_trade(trade.realized_pnl)

            for strategy in self.strategies:
                if strategy.name == trade.strategy_name:
                    strategy.record_trade(trade.realized_pnl)

            feature_snapshot = trade.metadata.get("confidence_features")

            if feature_snapshot:
                self.confidence_model.update(
                    feature_snapshot,
                    trade.realized_pnl > 0,
                )

            logger.info(
                "Closed %s %s pnl=%.4f reason=%s equity=%.2f",
                trade.side.value,
                trade.symbol,
                trade.realized_pnl,
                trade.reason,
                self.risk_manager.state.equity,
            )

    def _close_paper_position_if_max_held(self, snapshot: MarketSnapshot) -> TradeRecord | None:
        if not self.settings.trading.paper_trade:
            return None

        max_holding_iterations = self.settings.trading.paper_max_holding_iterations
        if max_holding_iterations <= 0:
            return None

        position = self.order_manager.positions.get(snapshot.symbol)
        if position is None:
            return None

        held_iterations = self._held_iterations(position.metadata)
        if held_iterations < max_holding_iterations:
            return None

        logger.info(
            "Paper max-hold exit %s held=%s max=%s mark=%.4f",
            snapshot.symbol,
            held_iterations,
            max_holding_iterations,
            snapshot.close,
        )
        return self.order_manager.close_position(
            snapshot.symbol,
            snapshot.close,
            "max_hold_exit",
        )

    def state(self) -> dict[str, Any]:
        mark_prices = self._mark_prices()
        unrealized_by_symbol = self.order_manager.unrealized_pnl_by_symbol(mark_prices)
        unrealized_pnl = sum(unrealized_by_symbol.values())
        realized_equity = self.risk_manager.state.equity
        return {
            "balance": realized_equity,
            "equity": realized_equity + unrealized_pnl,
            "realized_pnl": self.risk_manager.state.realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "daily_pnl": self.risk_manager.state.daily_pnl,
            "closed_trade_count": self.risk_manager.state.closed_trade_count,
            "open_positions": {
                symbol: {
                    "side": position.side.value,
                    "amount": position.amount,
                    "entry_price": position.entry_price,
                    "mark_price": mark_prices.get(symbol),
                    "unrealized_pnl": unrealized_by_symbol.get(symbol, 0.0),
                    "held_iterations": self._held_iterations(position.metadata),
                    "stop_loss": position.stop_loss,
                    "take_profit": position.take_profit,
                    "strategy": position.strategy_name,
                    "confidence": position.confidence,
                }
                for symbol, position in self.order_manager.positions.items()
            },
            "recent_trades": [
                {
                    "symbol": trade.symbol,
                    "side": trade.side.value,
                    "amount": trade.amount,
                    "entry_price": trade.entry_price,
                    "exit_price": trade.exit_price,
                    "realized_pnl": trade.realized_pnl,
                    "fees": trade.fees,
                    "reason": trade.reason,
                    "strategy": trade.strategy_name,
                    "confidence": trade.confidence,
                    "closed_at": trade.closed_at,
                }
                for trade in self.order_manager.closed_trades[-20:]
            ],
            "trade_history_path": str(self.order_manager.trade_history_path),
            "sentiment": self.sentiment_by_symbol,
            "iterations": self.iterations,
            "paper_max_holding_iterations": self.settings.trading.paper_max_holding_iterations,
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

        thread = threading.Thread(
            target=run_api,
            name="monitoring-api",
            daemon=True,
        )
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
    parser.add_argument("--paper", action="store_true", help="Force paper trading.")
    parser.add_argument(
        "--symbols",
        type=str,
        help="Comma-separated symbols, e.g. BTC/USDT,ETH/USDT.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        help="Stop after N loop iterations.",
    )
    return parser


def apply_cli_overrides(settings: Settings, args: argparse.Namespace) -> Settings:
    from dataclasses import replace

    trading = settings.trading
    app = settings.app

    if args.paper:
        trading = replace(
            trading,
            paper_trade=True,
            enable_live_trading=False,
        )

    if args.symbols:
        trading = replace(
            trading,
            symbols=tuple(
                item.strip()
                for item in args.symbols.split(",")
                if item.strip()
            ),
        )

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
