"""Generate the public static research dashboard from compact summary logs.

This is reporting-only. It reads ignored local JSONL summary records and writes
docs/index.html. It does not run backtests, download data, place orders, or
read credentials.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any


DEFAULT_LOG_PATH = Path("data/backtest_logs/realized_sweep_summary.jsonl")
DEFAULT_OUTPUT_PATH = Path("docs/index.html")
LATEST_LIMIT = 10
DEFAULT_SYMBOL_FILTER = "BTC/USDT"
DAILY_TRADE_TARGET = 100.0
DAILY_RETURN_TARGET_PCT = 5.0
MIN_ACCEPTABLE_PF = 1.1
REPORT_VERSION = "vNext Execution Simulator Build"
DISPLAY_LABELS = {
    "achieved": "Achieved",
    "not_achieved": "Not achieved",
    "unrealistic_given_data": "Unrealistic",
    "not_profitable_out_of_sample": "Failed OOS",
    "not_profitable_frequency_target_not_met": "Too sparse",
    "not_profitable_at_30_trades": "Not profitable",
    "too_few_trades": "Too sparse",
    "potentially_promising_needs_more_testing": "Promising / thin sample",
    "robust_candidate": "Robust candidate",
    "weak_overfit_risk": "Overfit risk",
    "PROFITABLE_CANDIDATE": "Candidate",
    "NOT_PROFITABLE": "Not profitable",
    "NO_ACCEPTED_STRATEGY": "No accepted strategy",
    "FEE_DRAG + LOW_EDGE": "Fee drag + weak edge",
    "OUT_OF_SAMPLE_FAILURE": "Out-of-sample failure",
    "OVERTRADING_FOR_QUALITY_PROFILE": "Overtrading for quality profile",
    "TOO_FEW_HIGH_QUALITY_SETUPS": "Too few high-quality setups",
    "LOW_TRADE_FREQUENCY": "Low trade frequency",
    "full_3_year_dataset": "Full 3-year dataset",
    "limited_30_day_dataset": "Limited 30-day dataset",
    "partial_dataset": "Partial dataset",
    "below_30_trades": "Below 30 trades",
    "simulated": "Simulated",
    "insufficient_trade_count": "Insufficient trade count",
    "placeholder_optional_cache": "Optional cached placeholder",
    "available_from_ohlcv_only": "Available from OHLCV only",
    "True": "Yes",
    "False": "No",
    "LOW": "Low",
    "MEDIUM": "Medium",
    "HIGH": "High",
}


@dataclass(frozen=True)
class SummaryRecord:
    logged_at_utc: str
    run_label: str
    summary: dict[str, Any]
    line_number: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate docs/index.html from compact backtest summaries.")
    parser.add_argument(
        "--log-path",
        type=Path,
        default=DEFAULT_LOG_PATH,
        help=f"Compact JSONL summary log path. Default: {DEFAULT_LOG_PATH}.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Dashboard HTML output path. Default: {DEFAULT_OUTPUT_PATH}.",
    )
    parser.add_argument(
        "--latest",
        type=int,
        default=LATEST_LIMIT,
        help=f"Number of latest BTC-only runs to show. Default: {LATEST_LIMIT}.",
    )
    parser.add_argument(
        "--symbol-filter",
        default=DEFAULT_SYMBOL_FILTER,
        help="Only show compact summaries whose symbols are exactly this symbol. Default: BTC/USDT.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_records(args.log_path)
    filtered = filter_symbol_records(records, args.symbol_filter)
    latest = filtered[-max(args.latest, 0) :]
    html = render_dashboard(
        records=filtered,
        all_records=records,
        latest=latest,
        symbol_filter=args.symbol_filter,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(strip_trailing_whitespace(html), encoding="utf-8")
    print(f"Dashboard written: {args.output}")
    print(f"Summary records read: {len(records)}")
    print(f"{args.symbol_filter} records shown: {len(filtered)}")


def load_records(path: Path) -> list[SummaryRecord]:
    if not path.exists():
        return []

    records: list[SummaryRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            summary = payload.get("summary")
            if not isinstance(summary, dict):
                continue
            records.append(
                SummaryRecord(
                    logged_at_utc=str(payload.get("logged_at_utc") or ""),
                    run_label=str(payload.get("run_label") or "unlabeled"),
                    summary=summary,
                    line_number=line_number,
                )
            )
    return sorted(records, key=record_sort_key)


def get_git_info() -> dict[str, str]:
    remote_url = run_git_value(["remote", "get-url", "origin"])
    return {
        "commit": run_git_value(["rev-parse", "--short", "HEAD"]),
        "branch": run_git_value(["branch", "--show-current"]),
        "worktree": "dirty" if run_git_value(["status", "--porcelain"]) not in {"", "n/a"} else "clean",
        "repo_slug": github_repo_slug(remote_url),
    }


def run_git_value(args: list[str]) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            capture_output=True,
            check=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "n/a"
    return completed.stdout.strip() or "n/a"


def github_repo_slug(remote_url: str) -> str:
    if not remote_url or remote_url == "n/a":
        return "n/a"
    value = remote_url.strip()
    if value.endswith(".git"):
        value = value[:-4]
    marker = "github.com"
    if marker not in value:
        return "n/a"
    if value.startswith("git@github.com:"):
        return value.split("git@github.com:", 1)[1]
    if "github.com/" in value:
        return value.split("github.com/", 1)[1]
    return "n/a"


def filter_symbol_records(records: list[SummaryRecord], symbol: str) -> list[SummaryRecord]:
    filtered: list[SummaryRecord] = []
    for record in records:
        symbols = normalize_list(record.summary.get("symbols"))
        if symbols == [symbol]:
            filtered.append(record)
    return filtered


def record_sort_key(record: SummaryRecord) -> tuple[datetime, int]:
    return parse_timestamp(record.logged_at_utc), record.line_number


def parse_timestamp(raw: str) -> datetime:
    if not raw:
        return datetime.min
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return datetime.min


def render_dashboard(
    records: list[SummaryRecord],
    all_records: list[SummaryRecord],
    latest: list[SummaryRecord],
    symbol_filter: str,
) -> str:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    latest_record = latest[-1] if latest else None
    latest_summary = latest_record.summary if latest_record else {}
    git_info = get_git_info()
    best_overall = best_row(records, "best_overall")
    best_30 = best_row(records, "best_at_least_30")
    worst_overall = worst_row(records, "worst_overall")
    verdict_counts = count_values(records, "verdict")
    tf_rows = timeframe_rows(records)
    tf_completion = timeframe_completion(records)
    strategy_rows = strategy_rows_from_records(records)
    leaderboard_rows = scalping_leaderboard_rows(records)
    daily_metrics = daily_metrics_from_summary(latest_summary)
    legacy_records = len(all_records) - len(records)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BTC Scalping Research Dashboard</title>
  <meta name="description" content="BTC-only paper and backtesting research dashboard.">
  <link rel="stylesheet" href="assets/styles.css">
  <script src="assets/app.js" defer></script>
</head>
<body>
  <div id="research-shell" class="site-content">
  {render_update_banner(generated_at, latest_record, latest_summary, git_info, symbol_filter)}
  <header class="hero">
    <div class="shell hero-grid">
      <div>
        <p class="eyebrow">BTCUSDT futures paper/backtesting research</p>
        <h1>BTC Scalping Research Dashboard</h1>
        <p class="hero-copy">Fee-aware, liquidation-aware calibration for BTCUSDT futures-style short-term strategies across 1m, 5m, and 15m data. Results are research diagnostics only.</p>
        <div class="status-strip">
          <span class="pill ok">Paper mode default</span>
          <span class="pill ok">Live trading disabled</span>
          <span class="pill ok">Simulated leverage only</span>
          <span class="pill ok">BTC-only default</span>
          <span class="pill warn">Backtesting only</span>
        </div>
      </div>
      <div class="hero-card">
        <span>Latest verdict</span>
        <strong>{escape(display_label(latest_summary.get("verdict")))}</strong>
        <small>Generated {escape(generated_at)}</small>
      </div>
    </div>
  </header>

  <main class="shell layout">
    <section class="panel span-12 quick-summary-panel">
      {render_quick_summary(latest_summary)}
    </section>

    <section class="panel span-12 system-status-panel">
      <div class="section-head">
        <div>
          <p class="eyebrow">Executive Summary</p>
          <h2>SYSTEM STATUS: {escape(system_status(latest_summary))}</h2>
        </div>
        <span class="tag {status_class(latest_summary)}">PRIMARY FAILURE: {escape(primary_failure_text(latest_summary))}</span>
      </div>
      <p class="muted">Targets are evaluated only from historical, fee-aware, slippage-aware BTCUSDT research summaries. If a target is not supported out-of-sample, the dashboard marks it as not achieved or unrealistic given data.</p>
    </section>

    <section class="panel span-12">
      <div class="section-head">
        <div>
          <p class="eyebrow">Safety status</p>
          <h2>Research Guardrails</h2>
        </div>
        <span class="tag neutral">Not financial advice</span>
      </div>
      <div class="metric-grid five">
        {metric_card("Mode", "Paper default", "good")}
        {metric_card("Live trading", "Disabled", "good")}
        {metric_card("Leverage", "None", "good")}
        {metric_card("Universe", f"{symbol_filter} futures", "good")}
        {metric_card("Timeframes", "1m / 5m / 15m", "neutral")}
      </div>
      <p class="muted">This public page contains compact summaries only. Raw text outputs, JSONL logs, historical CSVs, API keys, and local environment files stay ignored and local.</p>
    </section>

    <section class="panel span-12">
      <div class="section-head">
        <div>
          <p class="eyebrow">Latest run</p>
          <h2>BTC Compact Summary</h2>
        </div>
        <span class="tag {verdict_class(latest_summary.get("verdict"))} verdict-badge">{escape(display_label(latest_summary.get("verdict")))}</span>
      </div>
      <div class="metric-grid">
        {metric_card("BTC records", str(len(records)), "neutral")}
        {metric_card("Ignored non-BTC/legacy", str(max(legacy_records, 0)), "neutral")}
        {metric_card("Diagnostic notional", money(latest_summary.get("diagnostic_notional")), "neutral")}
        {metric_card("Calibration min net", money(latest_summary.get("calibration_min_expected_net_profit")), "neutral")}
        {metric_card("Total candles", whole(latest_summary.get("total_candles")), "neutral")}
        {metric_card("Signal window", whole(latest_summary.get("signal_window_bars")), "neutral")}
        {metric_card("Data days", days(latest_summary.get("approx_days")), "neutral")}
        {metric_card("Data coverage", display_label(latest_summary.get("data_coverage")), "good" if latest_summary.get("uses_full_3_year_dataset") else "warn")}
      </div>
      {render_data_window(latest_summary)}
    </section>

    <section class="panel span-12">
      <div class="section-head">
        <div>
          <p class="eyebrow">Backtest period</p>
          <h2>Data Coverage</h2>
        </div>
        <span class="tag {('good' if latest_summary.get("uses_full_3_year_dataset") else 'warn')} verdict-badge">{escape(display_label(latest_summary.get("data_coverage")))}</span>
      </div>
      {render_data_coverage_block(latest_summary)}
    </section>

    <section class="panel span-12 terminal-panel">
      <div class="section-head">
        <div>
          <p class="eyebrow">Scalping targets</p>
          <h2>100 Trades/Day and 5% Daily Target</h2>
        </div>
        <span class="tag {target_class(daily_metrics.get("verdict_100_trades_per_day"))} verdict-badge">{escape(display_label(daily_metrics.get("verdict_100_trades_per_day")))}</span>
      </div>
      {render_daily_target_tape(daily_metrics)}
      <p class="muted">Targets are research objectives only. The dashboard reports whether historical, fee-aware BTCUSDT tests achieved them; it does not claim future profitability.</p>
    </section>

    <section class="panel span-12">
      <div class="section-head">
        <div>
          <p class="eyebrow">Performance</p>
          <h2>Latest BTC Runs</h2>
        </div>
      </div>
      {render_latest_table(latest)}
    </section>

    <section class="panel span-12">
      <div class="section-head">
        <div>
          <p class="eyebrow">3-year workflow</p>
          <h2>Timeframe Completion</h2>
        </div>
      </div>
      {render_timeframe_completion(tf_completion)}
    </section>

    <section class="panel span-4">
      <h2>Aggregate Best Overall</h2>
      {render_result_box(best_overall)}
    </section>

    <section class="panel span-4">
      <h2>Aggregate Best With 30+ Trades</h2>
      {render_result_box(best_30)}
    </section>

    <section class="panel span-4">
      <h2>Aggregate Worst Overall</h2>
      {render_result_box(worst_overall)}
    </section>

    <section class="panel span-6">
      <h2>Timeframe Comparison</h2>
      {render_rank_table(tf_rows, ("Timeframe", "Runs", "Best net", "Best PF", "Best 30+"))}
    </section>

    <section class="panel span-6">
      <h2>Strategy Leaderboard</h2>
      {render_strategy_leaderboard(leaderboard_rows)}
    </section>

    <section class="panel span-6">
      <h2>Agent Comparison</h2>
      {render_agent_comparison(latest_summary)}
    </section>

    <section class="panel span-6">
      <h2>Latest Run Performance Summary</h2>
      {render_performance_table(latest_summary)}
    </section>

    <section class="panel span-6">
      <h2>Fee / Slippage Drag</h2>
      {render_fee_drag_block(latest_summary)}
    </section>

    <section class="panel span-6">
      <h2>Execution Drag</h2>
      {render_execution_drag_block(latest_summary)}
    </section>

    <section class="panel span-6">
      <h2>Monte Carlo Robustness</h2>
      {render_monte_carlo_block(latest_summary)}
    </section>

    <section class="panel span-6">
      <h2>Bigger-Move Strategy Scope</h2>
      {render_bigger_move_block(latest_summary)}
    </section>

    <section class="panel span-6">
      <h2>Macro / News Filter</h2>
      {render_macro_filter_block(latest_summary)}
    </section>

    <section class="panel span-6">
      <h2>Crypto-Native Data Hooks</h2>
      {render_crypto_hooks_block(latest_summary)}
    </section>

    <section class="panel span-6">
      <h2>Market Regime Performance</h2>
      {render_group_performance_block(latest_summary, "regime_performance")}
    </section>

    <section class="panel span-6">
      <h2>Trading Session Performance</h2>
      {render_group_performance_block(latest_summary, "session_performance")}
    </section>

    <section class="panel span-6">
      <h2>Daily PnL Distribution</h2>
      {render_daily_distribution_block(daily_metrics)}
    </section>

    <section class="panel span-6">
      <h2>Daily Return Distribution Chart</h2>
      {render_daily_return_chart(daily_metrics)}
    </section>

    <section class="panel span-6">
      <h2>Trades/Day vs PF Chart</h2>
      {render_trades_pf_chart(leaderboard_rows)}
    </section>

    <section class="panel span-6">
      <h2>Cluster Diagnostics</h2>
      {render_cluster_block(latest_summary)}
    </section>

    <section class="panel span-6">
      <h2>Momentum Diagnostics</h2>
      {render_momentum_block(latest_summary)}
    </section>

    <section class="panel span-6">
      <h2>Quality Diagnostics</h2>
      {render_quality_block(latest_summary)}
    </section>

    <section class="panel span-6">
      <h2>Walk-Forward Validation</h2>
      {render_walk_forward_block(latest_summary)}
    </section>

    <section class="panel span-6">
      <h2>Strategy Interpretation</h2>
      {render_strategy_interpretation(latest_summary)}
    </section>

    <section class="panel span-6">
      <h2>Known Issues</h2>
      <ul class="clean">
        <li>Recent realized sweeps have not shown robust profitable 30+ trade settings.</li>
        <li>Expected-only edge can look positive while realized exit simulation remains negative.</li>
        <li>Short momentum has been weaker than buy momentum in prior diagnostics.</li>
        <li>1m 3-year data can be large and should run after 15m and 5m checks.</li>
      </ul>
    </section>

    <section class="panel span-6">
      <h2>Next Tasks</h2>
      {render_next_tasks(records)}
    </section>

    <section class="panel span-6">
      <h2>Ideas Backlog</h2>
      <ul class="clean">
        <li>Fee and slippage sensitivity tables for BTC-only candidates.</li>
        <li>Entry-only clusters by session/time-of-day if timestamp diagnostics justify it.</li>
        <li>Drawdown and equity-curve snapshots from compact summaries.</li>
        <li>Walk-forward stability score for candidate ranking.</li>
      </ul>
    </section>

    <section class="panel span-6">
      <h2>Local Workflow</h2>
      <div class="code">tools\\download_btc_3y_data.bat<br>tools\\verify_btc_3y_data.bat<br>tools\\run_btc_15m_3y_backtest.bat<br>tools\\run_btc_5m_3y_backtest.bat<br>tools\\run_btc_1m_3y_backtest.bat<br>tools\\run_btc_optimization_suite.bat<br>tools\\compare_logs.bat<br>tools\\update_dashboard.bat<br>tools\\safe_git_status.bat</div>
      <p class="muted">Backtest helpers write ignored local files under data/. The dashboard generator publishes compact, non-secret HTML under docs/.</p>
    </section>

    <section class="panel span-6">
      <h2>GitHub Workflow</h2>
      <ol>
        <li>Run BTC-only backtests locally.</li>
        <li>Regenerate the dashboard.</li>
        <li>Review changed files with <span class="code-inline">tools\\safe_git_status.bat</span>.</li>
        <li>Commit only code, docs, and tooling.</li>
        <li>Never commit data/*.txt, data/backtest_logs/, historical CSVs, .env, .venv, cache, or pyc files.</li>
      </ol>
    </section>

    <section class="panel span-12 summary-final">
      {render_final_summary(latest_summary)}
    </section>
  </main>

  <footer class="shell footer">
    Generated at {escape(generated_at)} from compact local summary records only.
  </footer>
  </div>
</body>
</html>
"""


def strip_trailing_whitespace(value: str) -> str:
    return "\n".join(line.rstrip() for line in value.splitlines()) + "\n"


def render_update_banner(
    generated_at: str,
    latest_record: SummaryRecord | None,
    latest_summary: dict[str, Any],
    git_info: dict[str, str],
    symbol_filter: str,
) -> str:
    data_start = text_value(latest_summary.get("data_start") or latest_summary.get("data_period_start"))
    data_end = text_value(latest_summary.get("data_end") or latest_summary.get("data_period_end"))
    data_range = f"{data_start} -> {data_end}" if data_start != "n/a" and data_end != "n/a" else "n/a"
    data_generated = short_timestamp(latest_record.logged_at_utc) if latest_record else "n/a"
    return f"""
  <section class="update-banner">
    <div class="shell update-grid">
      <div class="update-primary">
        <span>LAST UPDATED</span>
        <strong>{escape(generated_at)}</strong>
        <small>Dashboard generation timestamp</small>
      </div>
      <div class="update-item">
        <span>COMMIT</span>
        <strong id="latest-commit-live" data-repo="{escape(git_info.get("repo_slug", "n/a"))}" data-branch="{escape(git_info.get("branch", "n/a"))}">{escape(git_info.get("commit", "n/a"))}</strong>
        <small>Build hash shown; public GitHub API refreshes latest branch hash when available. Worktree at build: {escape(git_info.get("worktree", "n/a"))}</small>
      </div>
      <div class="update-item">
        <span>BRANCH</span>
        <strong>{escape(git_info.get("branch", "n/a"))}</strong>
      </div>
      <div class="update-item wide">
        <span>DATA</span>
        <strong>{escape(symbol_filter)} {escape(data_range)}</strong>
        <small>{escape(display_label(latest_summary.get("data_coverage")))} | candles {whole(latest_summary.get("candle_count") or latest_summary.get("total_candles"))}</small>
      </div>
      <div class="update-item">
        <span>DATA GENERATED</span>
        <strong>{escape(data_generated)}</strong>
      </div>
      <div class="update-item">
        <span>REPORT VERSION</span>
        <strong>{escape(text_value(latest_summary.get("report_version") or REPORT_VERSION))}</strong>
      </div>
      <div class="update-item">
        <span>BTCUSDT ONLY</span>
        <strong>{escape(display_label(latest_summary.get("btc_only")))}</strong>
      </div>
      <div class="update-item">
        <span>GITHUB PAGES</span>
        <strong>branch /docs</strong>
      </div>
    </div>
  </section>
"""


def render_latest_table(records: list[SummaryRecord]) -> str:
    if not records:
        return """
      <div class="empty-state">
        <strong>No BTC-only compact logs yet.</strong>
        <span>Run a BTC-focused backtest with --save-summary-log, then run tools\\update_dashboard.bat.</span>
      </div>
"""

    rows = "\n".join(render_latest_row(record) for record in records)
    return f"""
      <div class="table-wrap latest-table-wrap">
        <table class="data-table latest-runs-table">
          <thead>
            <tr>
              <th>Run label</th>
              <th>Timestamp</th>
              <th>Timeframe</th>
              <th>Profile</th>
              <th>Candles</th>
              <th>Notional</th>
              <th>Calib min net</th>
              <th>Agent</th>
              <th>Lev</th>
              <th>Liq</th>
              <th>Trades/day</th>
              <th>Edge Confidence</th>
              <th>5-20/day</th>
              <th>Median day</th>
              <th>Fee drag</th>
              <th>Exec drag</th>
              <th>Missed fills</th>
              <th>100/day</th>
              <th>5% day</th>
              <th>WF verdict</th>
              <th>Soft late</th>
              <th>Pos 30+</th>
              <th>Freq rows</th>
              <th>Best overall</th>
              <th>Best 30+</th>
              <th>Best 5-20/day</th>
              <th class="verdict-col">Verdict</th>
            </tr>
          </thead>
          <tbody>{rows}</tbody>
        </table>
      </div>
"""


def render_latest_row(record: SummaryRecord) -> str:
    summary = record.summary
    return f"""
            <tr>
              <td>{escape(record.run_label)}</td>
              <td>{escape(short_timestamp(record.logged_at_utc))}</td>
              <td>{escape(format_list(summary.get("timeframes")))}</td>
              <td>{escape(text_value(summary.get("quality_profile") or summary.get("mode")))}</td>
              <td>{escape(whole(summary.get("total_candles")))}</td>
              <td>{money(summary.get("diagnostic_notional"))}</td>
              <td>{money(summary.get("calibration_min_expected_net_profit"))}</td>
              <td>{escape(text_value(summary.get("agent_name")))}</td>
              <td>{number(summary.get("leverage_used"))}</td>
              <td>{escape(text_value(summary.get("liquidation_events")))}</td>
              <td>{number(summary.get("trades_per_day"))}</td>
              <td>{edge_confidence_badge(summary.get("edge_confidence"))}</td>
              <td>{escape(display_label(summary.get("verdict_5_to_20_trades_per_day")))}</td>
              <td>{percent(summary.get("median_daily_return_pct"))}</td>
              <td>{percent(summary.get("fee_drag_pct"))}</td>
              <td>{percent(summary.get("total_execution_drag_pct"))}</td>
              <td>{percent(summary.get("missed_fill_rate"))}</td>
              <td>{escape(display_label(summary.get("verdict_100_trades_per_day")))}</td>
              <td>{escape(display_label(summary.get("verdict_5pct_daily_target")))}</td>
              <td>{escape(display_label(summary.get("walk_forward_verdict")))}</td>
              <td>{escape(text_value(summary.get("reject_soft_late_momentum")))}</td>
              <td>{escape(text_value(summary.get("positive_combinations_with_at_least_30_trades")))}</td>
              <td>{escape(text_value(summary.get("combinations_in_frequency_band")))}</td>
              <td class="wide-text">{escape(format_row(summary.get("best_overall")))}</td>
              <td class="wide-text">{escape(format_row(summary.get("best_at_least_30")))}</td>
              <td class="wide-text">{escape(format_row(summary.get("best_in_5_to_20_trades_per_day")))}</td>
              <td>{verdict_tag(summary.get("verdict"))}</td>
            </tr>"""


def render_data_window(summary: dict[str, Any]) -> str:
    if not summary:
        return '<p class="muted">No BTC-only data profile has been logged yet.</p>'
    return (
        '<div class="data-window">'
        f'<span>Symbols: {escape(format_list(summary.get("symbols")))}</span>'
        f'<span>Timeframes: {escape(format_list(summary.get("timeframes")))}</span>'
        f'<span>Profile: {escape(text_value(summary.get("quality_profile") or summary.get("mode")))}</span>'
        f'<span>Frequency target: {number(summary.get("target_trades_per_day_min"))}-{number(summary.get("target_trades_per_day_max"))}/day</span>'
        f'<span>Start: {escape(text_value(summary.get("data_period_start")))}</span>'
        f'<span>End: {escape(text_value(summary.get("data_period_end")))}</span>'
        f'<span>Production target: {number(summary.get("production_min_target_move_bps"))} bps</span>'
        f'<span>Production reward/cost: {number(summary.get("production_min_reward_cost_ratio"))}x</span>'
        "</div>"
    )


def render_quick_summary(summary: dict[str, Any]) -> str:
    row = safe_dict(summary.get("best_at_least_30")) or safe_dict(summary.get("best_overall"))
    status = system_status(summary)
    pf = number(row.get("pf") if row else None)
    avg_day = percent(summary.get("avg_daily_return_pct"))
    trades_day = number(summary.get("trades_per_day"))
    issue = primary_failure_text(summary)
    edge = display_label(summary.get("edge_confidence"))
    text = f"{status} | PF: {pf} | {avg_day}/day | {trades_day} trades/day | Edge: {edge} | Issue: {issue}"
    return (
        '<div class="quick-summary">'
        '<span>Quick glance</span>'
        f'<strong>{escape(text)}</strong>'
        '</div>'
    )


def render_data_coverage_block(summary: dict[str, Any]) -> str:
    if not summary:
        return '<p class="muted">No data coverage has been logged yet.</p>'
    warning = text_value(summary.get("data_coverage_warning"))
    rows = [
        ("Backtest Period", display_label(summary.get("data_coverage"))),
        ("Data Start", text_value(summary.get("data_start") or summary.get("data_period_start"))),
        ("Data End", text_value(summary.get("data_end") or summary.get("data_period_end"))),
        ("Calendar Days", days(summary.get("backtest_days") or summary.get("approx_days"))),
        ("Years Covered", number(summary.get("data_years"))),
        ("Candle Count", whole(summary.get("candle_count") or summary.get("total_candles"))),
        ("Full 3-Year Dataset", text_value(summary.get("uses_full_3_year_dataset"))),
    ]
    warning_html = "" if warning in {"", "n/a"} else f'<p class="warning-line">{escape(warning)}</p>'
    return warning_html + '<div class="stacked">' + "".join(
        f'<div><strong>{escape(label)}</strong><span>{escape(value)}</span></div>' for label, value in rows
    ) + "</div>"


def render_macro_filter_block(summary: dict[str, Any]) -> str:
    macro = safe_dict(summary.get("macro_news_filter"))
    if not macro:
        return '<p class="muted">Macro/news filter status is not present in this log.</p>'
    events = macro.get("events_nearby")
    event_text = "none"
    if isinstance(events, list) and events:
        event_text = "; ".join(
            f"{safe_dict(event).get('time', 'n/a')} {safe_dict(event).get('impact', 'n/a')} {safe_dict(event).get('event', 'n/a')}"
            for event in events[:4]
        )
    rows = [
        ("Enabled", text_value(macro.get("enabled"))),
        ("Paper only", text_value(macro.get("paper_only"))),
        ("Allow trade", text_value(macro.get("allow_trade"))),
        ("Risk level", text_value(macro.get("risk_level"))),
        ("Source", text_value(macro.get("source"))),
        ("Reason", text_value(macro.get("reason"))),
        ("Nearby high-impact events", event_text),
    ]
    return '<div class="stacked">' + "".join(
        f'<div><strong>{escape(label)}</strong><span>{escape(value)}</span></div>' for label, value in rows
    ) + "</div>"


def render_crypto_hooks_block(summary: dict[str, Any]) -> str:
    hooks = safe_dict(summary.get("crypto_native_data_hooks"))
    if not hooks:
        return '<p class="muted">Crypto-native hooks are not present in this log yet.</p>'
    rows = []
    for key, value in hooks.items():
        if key == "note":
            continue
        data = safe_dict(value)
        if not data:
            continue
        rows.append((key.replace("_", " ").title(), f"{display_label(data.get('status'))} | API key required: {display_label(data.get('requires_api_key'))}"))
    note = text_value(hooks.get("note"))
    return (
        '<div class="stacked">'
        + "".join(f'<div><strong>{escape(label)}</strong><span>{escape(value)}</span></div>' for label, value in rows)
        + f'<div><strong>Note</strong><span>{escape(note)}</span></div>'
        + "</div>"
    )


def render_group_performance_block(summary: dict[str, Any], field: str) -> str:
    rows = summary.get(field)
    if not isinstance(rows, list) or not rows:
        return '<p class="muted">No regime/session performance has been logged yet. Re-run BTC search with the latest backtester.</p>'
    body = ""
    for row in rows[:8]:
        data = safe_dict(row)
        if not data:
            continue
        body += (
            "<tr>"
            f"<td>{escape(text_value(data.get('label')))}</td>"
            f"<td>{whole(data.get('trades'))}</td>"
            f"<td>{money(data.get('net'))}</td>"
            f"<td>{money(data.get('avg_net'))}</td>"
            f"<td>{number(data.get('pf'))}</td>"
            f"<td>{percent(data.get('win_rate'))}</td>"
            f"<td>{percent(data.get('max_drawdown_pct'))}</td>"
            f"<td>{percent(data.get('fee_drag_pct'))}</td>"
            f"<td>{escape(display_label(data.get('sample_warning')))}</td>"
            "</tr>"
        )
    if not body:
        return '<p class="muted">No rows available.</p>'
    return (
        '<div class="table-wrap compact"><table><thead><tr>'
        '<th>Bucket</th><th>Trades</th><th>Net</th><th>Avg</th><th>PF</th><th>Win</th><th>DD</th><th>Fee drag</th><th>Sample</th>'
        f'</tr></thead><tbody>{body}</tbody></table></div>'
    )


def render_daily_target_tape(metrics: dict[str, Any]) -> str:
    if not metrics:
        metrics = {}
    return (
        '<div class="target-tape">'
        + target_card(
            "Trades/day",
            number(metrics.get("trades_per_day")),
            DAILY_TRADE_TARGET,
            to_float(metrics.get("trades_per_day")),
            display_label(metrics.get("verdict_100_trades_per_day")),
        )
        + target_card(
            "Median daily return",
            percent(metrics.get("median_daily_return_pct")),
            DAILY_RETURN_TARGET_PCT,
            to_float(metrics.get("median_daily_return_pct")),
            display_label(metrics.get("verdict_5pct_daily_target")),
        )
        + target_card(
            "Avg daily return",
            percent(metrics.get("avg_daily_return_pct")),
            DAILY_RETURN_TARGET_PCT,
            to_float(metrics.get("avg_daily_return_pct")),
            "research metric",
        )
        + target_card(
            "Profitable days",
            percent(metrics.get("days_profitable_pct")),
            50.0,
            to_float(metrics.get("days_profitable_pct")),
            "calendar basis",
        )
        + target_card(
            "Days above 1%",
            whole(metrics.get("days_above_1pct")),
            1.0,
            to_float(metrics.get("days_above_1pct")),
            "count",
        )
        + target_card(
            "Days above 2%",
            whole(metrics.get("days_above_2pct")),
            1.0,
            to_float(metrics.get("days_above_2pct")),
            "count",
        )
        + target_card(
            "Days above 5%",
            whole(metrics.get("days_above_5pct")),
            1.0,
            to_float(metrics.get("days_above_5pct")),
            "count",
        )
        + target_card(
            "Max daily drawdown",
            percent(metrics.get("max_daily_drawdown_pct")),
            0.0,
            abs(to_float(metrics.get("max_daily_drawdown_pct")) or 0.0),
            "lower is better",
            inverse=True,
        )
        + target_card(
            "Fee drag/day",
            percent(metrics.get("fee_drag_pct")),
            0.0,
            to_float(metrics.get("fee_drag_pct")),
            "cost visibility",
            inverse=True,
        )
        + "</div>"
    )


def target_card(
    label: str,
    value: str,
    target: float,
    numeric_value: float | None,
    verdict: str,
    inverse: bool = False,
) -> str:
    numeric = numeric_value if numeric_value is not None else 0.0
    progress = 100.0 if target <= 0 and numeric <= 0 else min(abs(numeric) / max(abs(target), 1e-9) * 100, 100.0)
    state = target_class(verdict)
    if inverse:
        state = "good" if numeric <= target else "warn"
    return f"""
        <div class="target-card">
          <div class="label">{escape(label)}</div>
          <div class="target-value {state}">{escape(value)}</div>
          <div class="progress" data-progress="{progress:.2f}"><span></span></div>
          <small>{escape(verdict)}</small>
        </div>"""


def render_fee_drag_block(summary: dict[str, Any]) -> str:
    row = safe_dict(summary.get("best_at_least_30")) or safe_dict(summary.get("best_overall"))
    if not row:
        return '<p class="muted">No realized candidate available yet.</p>'
    gross = abs(to_float(row.get("gross")) or 0.0)
    costs = to_float(row.get("costs")) or 0.0
    trades = int(to_float(row.get("trades")) or 0)
    cost_per_trade = costs / trades if trades else None
    cost_vs_gross = costs / gross * 100 if gross else None
    return (
        '<div class="metric-grid">'
        f'{metric_card("Gross PnL", money(row.get("gross")), "neutral")}'
        f'{metric_card("Costs", money(row.get("costs")), "warn")}'
        f'{metric_card("Net PnL", money(row.get("net")), "bad" if (to_float(row.get("net")) or 0) < 0 else "good")}'
        f'{metric_card("Cost/trade", money(cost_per_trade), "warn")}'
        f'{metric_card("Costs/gross", percent(cost_vs_gross), "warn")}'
        f'{metric_card("Fee drag/day", percent(summary.get("fee_drag_pct")), "warn")}'
        '</div>'
    )


def render_execution_drag_block(summary: dict[str, Any]) -> str:
    if not summary:
        return '<p class="muted">Execution simulation metrics are not available yet.</p>'
    latency_avg = f"{number(summary.get('execution_latency_ms_avg'))} ms"
    latency_p95 = f"{number(summary.get('execution_latency_ms_p95'))} ms"
    drag_class = "bad" if (to_float(summary.get("total_execution_drag_pct")) or 0) > 1 else "warn"
    return (
        '<div class="metric-grid">'
        f'{metric_card("Latency avg", latency_avg, "neutral")}'
        f'{metric_card("Latency p95", latency_p95, "neutral")}'
        f'{metric_card("Spread cost", percent(summary.get("spread_cost_pct")), "warn")}'
        f'{metric_card("Slippage cost", percent(summary.get("slippage_cost_pct")), "warn")}'
        f'{metric_card("Total drag", percent(summary.get("total_execution_drag_pct")), drag_class)}'
        f'{metric_card("Missed fills", percent(summary.get("missed_fill_rate")), "warn")}'
        '</div>'
    )


def render_monte_carlo_block(summary: dict[str, Any]) -> str:
    monte = safe_dict(summary.get("monte_carlo"))
    if not monte:
        return '<p class="muted">Monte Carlo metrics are not available yet.</p>'
    rows = [
        ("Status", display_label(monte.get("status"))),
        ("Iterations", whole(monte.get("iterations"))),
        ("Positive final return probability", percent(monte.get("probability_positive_final_return"))),
        ("Survival probability", percent(monte.get("survival_probability"))),
        ("Expected max drawdown", percent(monte.get("expected_max_drawdown_pct"))),
        ("P95 worst drawdown", percent(monte.get("worst_case_drawdown_p95_pct"))),
        ("Robustness score", number(monte.get("robustness_score"))),
        ("Edge confidence", display_label(summary.get("edge_confidence"))),
    ]
    return '<div class="stacked">' + "".join(
        f'<div><strong>{escape(label)}</strong><span>{escape(value)}</span></div>' for label, value in rows
    ) + "</div>"


def render_bigger_move_block(summary: dict[str, Any]) -> str:
    rows = [
        ("Max hold", f"{whole(summary.get('max_hold_minutes'))} minutes"),
        ("Target move range", "0.30% - 2.00%"),
        ("Research note", text_value(summary.get("bigger_move_research_note"))),
        ("Reason", "Tiny high-frequency scalps remain structurally weak after realistic fees, spread, latency, and slippage."),
    ]
    return '<div class="stacked">' + "".join(
        f'<div><strong>{escape(label)}</strong><span>{escape(value)}</span></div>' for label, value in rows
    ) + "</div>"


def render_daily_distribution_block(metrics: dict[str, Any]) -> str:
    if not metrics:
        return '<p class="muted">Daily metrics are not present in this log. Re-run calibration with the latest code.</p>'
    rows = [
        ("Basis", text_value(metrics.get("basis"))),
        ("Calendar days", whole(metrics.get("calendar_days"))),
        ("Active trade days", whole(metrics.get("active_trade_days"))),
        ("Zero-trade days", whole(metrics.get("zero_trade_days"))),
        ("Best daily return", percent(metrics.get("best_daily_return_pct"))),
        ("Worst daily return", percent(metrics.get("worst_daily_return_pct"))),
        ("Days profitable", percent(metrics.get("days_profitable_pct"))),
        ("Days above 5%", whole(metrics.get("days_above_5pct"))),
    ]
    return '<div class="stacked">' + "".join(
        f'<div><strong>{escape(label)}</strong><span>{escape(value)}</span></div>' for label, value in rows
    ) + "</div>"


def render_daily_return_chart(metrics: dict[str, Any]) -> str:
    if not metrics:
        return '<p class="muted">No daily distribution data yet.</p>'
    values = [
        ("Avg", to_float(metrics.get("avg_daily_return_pct")) or 0.0),
        ("Median", to_float(metrics.get("median_daily_return_pct")) or 0.0),
        ("Best", to_float(metrics.get("best_daily_return_pct")) or 0.0),
        ("Worst", to_float(metrics.get("worst_daily_return_pct")) or 0.0),
        ("Fee drag", -(to_float(metrics.get("fee_drag_pct")) or 0.0)),
    ]
    return render_bar_chart(values, "%")


def render_trades_pf_chart(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return '<p class="muted">No strategy rows yet.</p>'
    points = []
    for row in rows[:10]:
        label = f"{row.get('timeframe', 'n/a')} {row.get('strategy', 'n/a')}"
        tpd = to_float(row.get("trades_per_day")) or 0.0
        pf_value = to_float(row.get("pf")) or 0.0
        points.append((label, min(tpd / DAILY_TRADE_TARGET * 100, 100.0), pf_value))
    body = "".join(
        f'<div class="scatter-row"><span>{escape(label)}</span><div class="scatter-track"><i style="left:{x:.2f}%"></i></div><strong>TPD {x / 100 * DAILY_TRADE_TARGET:.2f} | PF {pf_value:.2f}</strong></div>'
        for label, x, pf_value in points
    )
    return f'<div class="scatter">{body}</div>'


def render_bar_chart(values: list[tuple[str, float]], suffix: str) -> str:
    if not values:
        return '<p class="muted">n/a</p>'
    max_abs = max(abs(value) for _, value in values) or 1.0
    body = ""
    for label, value in values:
        width = min(abs(value) / max_abs * 100, 100.0)
        class_name = "good" if value > 0 else "bad" if value < 0 else "neutral"
        body += (
            f'<div class="bar-row"><span>{escape(label)}</span>'
            f'<div class="bar-shell"><i class="{class_name}" style="width:{width:.2f}%"></i></div>'
            f'<strong>{value:.2f}{escape(suffix)}</strong></div>'
        )
    return f'<div class="bar-chart">{body}</div>'


def render_agent_comparison(summary: dict[str, Any]) -> str:
    agents = summary.get("agent_comparison")
    if not isinstance(agents, list) or not agents:
        return '<p class="muted">Agent comparison appears after a fast futures scalping-agent search run.</p>'
    body = ""
    for agent in agents:
        if not isinstance(agent, dict):
            continue
        best = safe_dict(agent.get("best"))
        best30 = safe_dict(agent.get("best_30"))
        body += (
            "<tr>"
            f"<td>{escape(text_value(agent.get('agent_name')))}</td>"
            f"<td>{escape(text_value(agent.get('parameter_sets')))}</td>"
            f"<td>{escape(text_value(agent.get('positive_rows')))}</td>"
            f"<td>{escape(format_row(best))}</td>"
            f"<td>{escape(format_row(best30))}</td>"
            "</tr>"
        )
    return (
        '<div class="table-wrap compact"><table><thead><tr>'
        '<th>Agent</th><th>Sets</th><th>Positive</th><th>Best</th><th>Best 30+</th>'
        f'</tr></thead><tbody>{body}</tbody></table></div>'
    )


def render_timeframe_completion(rows: list[dict[str, Any]]) -> str:
    body = "".join(
        "<tr>"
        f"<td>{escape(row['timeframe'])}</td>"
        f"<td>{escape(row['data_status'])}</td>"
        f"<td>{escape(row['run_status'])}</td>"
        f"<td>{escape(row['candles'])}</td>"
        f"<td>{escape(row['days'])}</td>"
        f"<td>{escape(row.get('start', 'n/a'))}</td>"
        f"<td>{escape(row.get('end', 'n/a'))}</td>"
        f"<td>{escape(row.get('coverage', 'n/a'))}</td>"
        f"<td>{escape(row['latest_run'])}</td>"
        f"<td>{verdict_tag(row['verdict'])}</td>"
        "</tr>"
        for row in rows
    )
    return (
        '<div class="table-wrap compact"><table><thead><tr>'
        '<th>Timeframe</th><th>Data</th><th>Backtest</th><th>Candles</th>'
        '<th>Days</th><th>Start</th><th>End</th><th>Coverage</th><th>Latest run</th><th>Verdict</th>'
        f'</tr></thead><tbody>{body}</tbody></table></div>'
    )


def render_result_box(row: dict[str, Any] | None) -> str:
    if not row:
        return '<p class="muted">n/a</p>'
    return f"""
      <div class="result-line">
        <strong>{escape(str(row.get("symbol", "n/a")))} {escape(str(row.get("strategy", "n/a")))}</strong>
        <span>Trades {escape(text_value(row.get("trades")))} | Net {money(row.get("net"))} | Avg {money(row.get("avg_net"))} | PF {number(row.get("pf"))}</span>
        <span>Target {number(row.get("target_bps"))} bps | Reward/cost {number(row.get("reward_cost"))}x | Hold {escape(text_value(row.get("hold")))}</span>
        <span>ATR TP {number(row.get("atrtp"))} | ATR SL {number(row.get("atrsl"))}</span>
        <small>{escape(format_soft_thresholds(row.get("soft_thresholds")))}</small>
      </div>
"""


def render_performance_table(summary: dict[str, Any]) -> str:
    rows = [
        ("Best overall", safe_dict(summary.get("best_overall"))),
        ("Best 30+", safe_dict(summary.get("best_at_least_30"))),
        ("Worst overall", safe_dict(summary.get("worst_overall"))),
    ]
    body = ""
    for label, row in rows:
        if not row:
            body += f"<tr><td>{escape(label)}</td><td colspan=\"9\">n/a</td></tr>"
            continue
        body += (
            "<tr>"
            f"<td>{escape(label)}</td>"
            f"<td>{escape(str(row.get('strategy', 'n/a')))}</td>"
            f"<td>{escape(text_value(row.get('trades')))}</td>"
            f"<td>{money(row.get('net'))}</td>"
            f"<td>{money(row.get('avg_net'))}</td>"
            f"<td>{number(row.get('pf'))}</td>"
            f"<td>{number(row.get('win_rate'))}%</td>"
            f"<td>{money(row.get('costs'))}</td>"
            f"<td>{money(row.get('max_drawdown'))}</td>"
            f"<td>{number(row.get('stop_loss_hit_rate'))}% / {number(row.get('take_profit_hit_rate'))}%</td>"
            "</tr>"
        )
    return (
        '<div class="table-wrap compact"><table><thead><tr>'
        '<th>Row</th><th>Strategy</th><th>Trades</th><th>Net</th><th>Avg</th>'
        '<th>PF</th><th>Win</th><th>Costs</th><th>Max DD</th><th>SL / TP</th>'
        f'</tr></thead><tbody>{body}</tbody></table></div>'
    )


def render_quality_block(summary: dict[str, Any]) -> str:
    if not summary:
        return '<p class="muted">No quality diagnostics available yet.</p>'
    return "\n".join(
        [
            f"<h3>Top quality rejections</h3>{render_rejection_list(summary.get('top_quality_rejections'))}",
            f"<h3>Top accepted loser cluster</h3><p>{escape(format_cluster(summary.get('top_accepted_loser_cluster')))}</p>",
            f"<h3>Soft-late rejections</h3><p>{escape(format_soft_rejections(summary.get('soft_late_rejections')))}</p>",
        ]
    )


def render_cluster_block(summary: dict[str, Any]) -> str:
    if not summary:
        return '<p class="muted">No cluster diagnostics available yet.</p>'
    rows = [
        ("Best momentum", format_momentum_cluster(summary.get("best_momentum_cluster"))),
        ("Worst momentum", format_momentum_cluster(summary.get("worst_momentum_cluster"))),
        ("Best buy entry", format_entry_cluster(summary.get("best_buy_entry_momentum_cluster"))),
        ("Worst buy entry", format_entry_cluster(summary.get("worst_buy_entry_momentum_cluster"))),
        ("Best sell entry", format_entry_cluster(summary.get("best_sell_entry_momentum_cluster"))),
        ("Worst sell entry", format_entry_cluster(summary.get("worst_sell_entry_momentum_cluster"))),
    ]
    warning = '<p class="muted">Cluster rows can have small sample sizes; do not promote filters from tiny clusters.</p>'
    return warning + '<div class="stacked">' + "".join(
        f'<div><strong>{escape(label)}</strong><span>{escape(value)}</span></div>' for label, value in rows
    ) + "</div>"


def render_momentum_block(summary: dict[str, Any]) -> str:
    if not summary:
        return '<p class="muted">No momentum diagnostics available yet.</p>'
    rows = [
        ("Buy momentum", format_side_summary(summary.get("buy_momentum"))),
        ("Sell momentum", format_side_summary(summary.get("sell_momentum"))),
        ("Best entry-only cluster", format_entry_cluster(summary.get("best_entry_momentum_cluster"))),
        ("Best 30+ entry-only cluster", format_entry_cluster(summary.get("best_entry_momentum_cluster_at_least_30"))),
        ("Worst entry-only cluster", format_entry_cluster(summary.get("worst_entry_momentum_cluster"))),
    ]
    return '<div class="stacked">' + "".join(
        f'<div><strong>{escape(label)}</strong><span>{escape(value)}</span></div>' for label, value in rows
    ) + "</div>"


def render_strategy_interpretation(summary: dict[str, Any]) -> str:
    row = safe_dict(summary.get("best_at_least_30")) or safe_dict(summary.get("best_overall"))
    if not row:
        return '<p class="muted">No accepted BTC candidate has been logged yet.</p>'
    net = to_float(row.get("net")) or 0.0
    pf = to_float(row.get("pf")) or 0.0
    trades = int(to_float(row.get("trades")) or 0)
    best_regime = safe_dict(summary.get("best_regime"))
    best_session = safe_dict(summary.get("best_session"))
    strength = (
        f"Best logged pocket: {text_value(best_regime.get('label'))} / {text_value(best_session.get('label'))}"
        if best_regime or best_session
        else "No regime/session pocket has enough evidence yet"
    )
    weakness = "Net remains negative after costs" if net <= 0 else "Positive result is still sparse and needs more out-of-sample evidence"
    if trades < 100:
        weakness = "Trade sample is below 100, so robustness remains weak"
    risk = "Overfit risk remains active" if summary.get("overfit_warning") else "Out-of-sample split did not flag overfit in this run"
    if pf < MIN_ACCEPTABLE_PF:
        risk = "PF is below the minimum acceptable threshold"
    rows = [
        ("Strength", strength),
        ("Weakness", weakness),
        ("Risk", risk),
    ]
    return '<div class="stacked">' + "".join(
        f'<div><strong>{escape(label)}</strong><span>{escape(value)}</span></div>' for label, value in rows
    ) + "</div>"


def render_next_tasks(records: list[SummaryRecord]) -> str:
    tested = {
        timeframe
        for record in records
        for timeframe in normalize_list(record.summary.get("timeframes"))
    }
    items: list[str] = []
    if "15m" not in tested:
        items.append("Run BTC 15m full 3-year single-baseline test first.")
    if "15m" in tested and "5m" not in tested:
        items.append("Run BTC 5m full 3-year single-baseline test next.")
    if "15m" in tested and "5m" in tested and "1m" not in tested:
        items.append("Do not run BTC 1m until runtime is acceptable; 1m is much larger than 5m.")
    items.extend(
        [
            "Add rolling walk-forward optimization after the current chronological split diagnostics.",
            "Investigate breakout and momentum failures by entry-only clusters, fees, and stop-loss hit rate.",
            "Keep scalping microstructure disabled by default unless backtest-only evidence improves.",
        ]
    )
    return '<ul class="clean">' + "".join(f"<li>{escape(item)}</li>" for item in items) + "</ul>"


def render_final_summary(summary: dict[str, Any]) -> str:
    row = safe_dict(summary.get("best_at_least_30")) or safe_dict(summary.get("best_overall"))
    status = system_status(summary)
    issue = primary_failure_text(summary)
    strategy = "n/a"
    if row:
        strategy = (
            f"{row.get('agent_name', 'n/a')} / {row.get('strategy', 'n/a')} / {row.get('side', 'n/a')} "
            f"trades={row.get('trades', 'n/a')} net={money(row.get('net'))} PF={number(row.get('pf'))} "
            f"DD={percent(row.get('max_drawdown_pct'))}"
        )
    profitable = status.upper().startswith("PROFITABLE") or "CANDIDATE" in status.upper()
    conclusion = (
        "A candidate requires positive net PnL after fees/slippage, PF above 1.1, drawdown at or below 10%, and positive out-of-sample validation."
        if profitable
        else "Over the logged BTCUSDT futures data, the current strategy set has not proven a valid edge after fees, slippage, drawdown, and out-of-sample validation."
    )
    rows = [
        ("SYSTEM STATUS", status),
        ("CORE METRICS", f"avg daily {percent(summary.get('avg_daily_return_pct'))} | PF {number(row.get('pf') if row else None)} | trades/day {number(summary.get('trades_per_day'))}"),
        ("EDGE CONFIDENCE", f"{display_label(summary.get('edge_confidence'))} | MC survival {percent(safe_dict(summary.get('monte_carlo')).get('survival_probability'))} | execution drag {percent(summary.get('total_execution_drag_pct'))}"),
        ("TARGET VERDICT", f"100/day {display_label(summary.get('verdict_100_trades_per_day'))} | 5% daily {display_label(summary.get('verdict_5pct_daily_target'))} | 5-20/day {display_label(summary.get('verdict_5_to_20_trades_per_day'))}"),
        ("MAIN FAILURE", issue),
        ("BEST STRATEGY", strategy),
        ("FINAL CONCLUSION", conclusion),
    ]
    return (
        '<h2 class="summary-title">SUMMARY</h2>'
        '<div class="summary-grid">'
        + "".join(f'<div><strong>{escape(label)}</strong><span>{escape(value)}</span></div>' for label, value in rows)
        + "</div>"
    )


def render_walk_forward_block(summary: dict[str, Any]) -> str:
    row = safe_dict(summary.get("best_at_least_30"))
    if not row:
        return '<p class="muted">No 30+ trade candidate available for chronological split validation.</p>'
    splits = row.get("walk_forward")
    if not isinstance(splits, list) or not splits:
        return '<p class="muted">Walk-forward split data is not present in this log. Re-run calibration with the latest code.</p>'
    body = "".join(
        "<tr>"
        f"<td>{escape(text_value(split.get('split')))}</td>"
        f"<td>{escape(text_value(split.get('trades')))}</td>"
        f"<td>{money(split.get('net'))}</td>"
        f"<td>{money(split.get('avg_net'))}</td>"
        f"<td>{number(split.get('pf'))}</td>"
        f"<td>{money(split.get('max_drawdown'))}</td>"
        "</tr>"
        for split in splits
        if isinstance(split, dict)
    )
    return (
        f"<p class=\"muted\">Verdict: {escape(display_label(row.get('walk_forward_verdict')))}. "
        "This is a chronological train/validation/test split, not a rolling optimizer.</p>"
        '<div class="table-wrap compact"><table><thead><tr>'
        '<th>Split</th><th>Trades</th><th>Net</th><th>Avg</th><th>PF</th><th>Max DD</th>'
        f'</tr></thead><tbody>{body}</tbody></table></div>'
        f"<p class=\"muted\">Overfit warning: {escape(text_value(summary.get('overfit_warning')))}</p>"
    )


def render_rejection_list(value: Any) -> str:
    if not isinstance(value, list) or not value:
        return '<p class="muted">n/a</p>'
    items = []
    for item in value[:8]:
        if not isinstance(item, dict):
            continue
        items.append(f"<li>{escape(str(item.get('reason', 'n/a')))}: {escape(str(item.get('count', 'n/a')))}</li>")
    return f'<ul class="clean">{"".join(items)}</ul>' if items else '<p class="muted">n/a</p>'


def render_rank_table(rows: list[tuple[str, str, str, str, str]], headings: tuple[str, ...]) -> str:
    if not rows:
        return '<p class="muted">No BTC-only summaries yet.</p>'
    header = "".join(f"<th>{escape(heading)}</th>" for heading in headings)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(cell)}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return f'<div class="table-wrap compact"><table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div>'


def render_strategy_leaderboard(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return '<p class="muted">No BTC-only strategy rows yet.</p>'
    body = ""
    for index, row in enumerate(rows[:12], start=1):
        body += (
            "<tr>"
            f"<td><span class=\"rank\">{index}</span></td>"
            f"<td>{escape(text_value(row.get('timeframe')))}</td>"
            f"<td>{escape(text_value(row.get('agent_name')))}</td>"
            f"<td>{escape(text_value(row.get('strategy')))}</td>"
            f"<td>{whole(row.get('trades'))}</td>"
            f"<td>{number(row.get('trades_per_day'))}</td>"
            f"<td>{money(row.get('net'))}</td>"
            f"<td>{number(row.get('pf'))}</td>"
            f"<td>{edge_confidence_badge(row.get('edge_confidence'))}</td>"
            f"<td>{percent(row.get('median_daily_return_pct'))}</td>"
            f"<td>{percent(row.get('fee_drag_pct'))}</td>"
            f"<td>{percent(row.get('total_execution_drag_pct'))}</td>"
            f"<td>{verdict_tag(row.get('walk_forward_verdict'))}</td>"
            "</tr>"
        )
    return (
        '<div class="table-wrap compact leaderboard-wrap"><table class="leaderboard data-table"><thead><tr>'
        '<th>#</th><th>Tf</th><th>Agent</th><th>Strategy</th><th>Trades</th><th>TPD</th>'
        '<th>Net</th><th>PF</th><th>Edge</th><th>Med day</th><th>Fee drag</th><th>Exec drag</th><th class="verdict-col">WF</th>'
        f'</tr></thead><tbody>{body}</tbody></table></div>'
    )


def metric_card(label: str, value: str, value_class: str = "neutral") -> str:
    return f"""
        <div class="metric">
          <div class="label">{escape(label)}</div>
          <div class="value {escape(value_class)}" data-count="{escape(counter_value(value))}">{escape(value)}</div>
        </div>"""


def verdict_tag(value: Any) -> str:
    verdict = text_value(value)
    class_name = verdict_class(verdict)
    return f'<span class="tag {class_name} verdict-badge">{escape(display_label(verdict))}</span>'


def edge_confidence_badge(value: Any) -> str:
    label = display_label(value)
    raw = text_value(value).upper()
    class_name = "good" if raw == "HIGH" else "warn" if raw == "MEDIUM" else "bad"
    return f'<span class="tag {class_name} edge-badge">{escape(label)}</span>'


def verdict_class(value: Any) -> str:
    verdict = str(value or "")
    if "robust" in verdict or "promising" in verdict:
        return "good"
    if "not_profitable" in verdict:
        return "bad"
    if "too_few" in verdict or "overfit" in verdict:
        return "warn"
    return "neutral"


def system_status(summary: dict[str, Any]) -> str:
    explicit = str(summary.get("system_status") or "")
    if explicit:
        if explicit == "NOT_PROFITABLE":
            return "NOT PROFITABLE - insufficient edge after fees"
        return display_label(explicit)
    verdict = str(summary.get("verdict") or "")
    if "robust" in verdict or "promising" in verdict:
        return "RESEARCH CANDIDATE FOUND"
    return "NOT PROFITABLE - insufficient edge after fees"


def primary_failure_text(summary: dict[str, Any]) -> str:
    explicit = str(summary.get("primary_failure") or "")
    if explicit:
        return display_label(explicit)
    reasons = summary.get("overfit_warning_reasons")
    if isinstance(reasons, list) and reasons:
        joined = " + ".join(display_label(item) for item in reasons[:2])
        return joined
    return "Fee drag + weak edge"


def status_class(summary: dict[str, Any]) -> str:
    status = system_status(summary).upper()
    if "CANDIDATE" in status:
        return "good"
    return "bad"


def target_class(value: Any) -> str:
    verdict = str(value or "")
    if verdict == "achieved":
        return "good"
    if "unrealistic" in verdict:
        return "bad"
    if "not achieved" in verdict or "not_achieved" in verdict:
        return "warn"
    return verdict_class(verdict)


def best_row(records: list[SummaryRecord], field: str) -> dict[str, Any] | None:
    rows = [row for row in (safe_dict(record.summary.get(field)) for record in records) if row]
    return max(
        rows,
        key=lambda row: (
            score_float(row.get("net"), missing=float("-inf")),
            score_float(row.get("avg_net"), missing=float("-inf")),
        ),
        default=None,
    )


def worst_row(records: list[SummaryRecord], field: str) -> dict[str, Any] | None:
    rows = [row for row in (safe_dict(record.summary.get(field)) for record in records) if row]
    return min(
        rows,
        key=lambda row: (
            score_float(row.get("net"), missing=float("inf")),
            score_float(row.get("avg_net"), missing=float("inf")),
        ),
        default=None,
    )


def timeframe_rows(records: list[SummaryRecord]) -> list[tuple[str, str, str, str, str]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        timeframes = normalize_list(record.summary.get("timeframes")) or ["n/a"]
        for timeframe in timeframes:
            buckets.setdefault(timeframe, []).append(record.summary)
    rows: list[tuple[str, str, str, str, str]] = []
    for timeframe, summaries in sorted(buckets.items()):
        best = best_summary_row(summaries, "best_overall")
        best_30 = best_summary_row(summaries, "best_at_least_30")
        rows.append(
            (
                timeframe,
                str(len(summaries)),
                money(best.get("net") if best else None),
                number(best.get("pf") if best else None),
                format_row(best_30),
            )
        )
    return rows


def timeframe_completion(records: list[SummaryRecord]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for timeframe in ("15m", "5m", "1m"):
        matching = [
            record for record in records
            if timeframe in normalize_list(record.summary.get("timeframes"))
        ]
        latest = matching[-1] if matching else None
        path = Path(f"data/historical_3y_{timeframe}/BTCUSDT_{timeframe}.csv")
        summary = latest.summary if latest else {}
        rows.append(
            {
                "timeframe": timeframe,
                "data_status": "present" if path.exists() else "missing",
                "run_status": "complete" if latest else "pending",
                "candles": whole(summary.get("total_candles")),
                "days": days(summary.get("approx_days")),
                "start": text_value(summary.get("data_start") or summary.get("data_period_start")),
                "end": text_value(summary.get("data_end") or summary.get("data_period_end")),
                "coverage": display_label(summary.get("data_coverage")),
                "latest_run": latest.run_label if latest else "n/a",
                "verdict": text_value(summary.get("verdict")),
            }
        )
    return rows


def strategy_rows_from_records(records: list[SummaryRecord]) -> list[tuple[str, str, str, str, str]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        for field in ("best_overall", "best_at_least_30", "worst_overall"):
            row = safe_dict(record.summary.get(field))
            strategy = str(row.get("strategy") or "")
            if strategy:
                buckets.setdefault(strategy, []).append(row)
    rows: list[tuple[str, str, str, str, str]] = []
    for strategy, result_rows in sorted(buckets.items()):
        best = max(result_rows, key=lambda row: score_float(row.get("net"), missing=float("-inf")), default=None)
        best_30_rows = [row for row in result_rows if int(to_float(row.get("trades")) or 0) >= 30]
        best_30 = max(best_30_rows, key=lambda row: score_float(row.get("net"), missing=float("-inf")), default=None)
        rows.append((strategy, str(len(result_rows)), money(best.get("net") if best else None), number(best.get("pf") if best else None), format_row(best_30)))
    return rows


def scalping_leaderboard_rows(records: list[SummaryRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str, str, str]] = set()
    for record in records:
        logged_rows = record.summary.get("strategy_leaderboard")
        if isinstance(logged_rows, list):
            for item in logged_rows:
                row = safe_dict(item)
                if row:
                    candidate = {
                        "run_label": record.run_label,
                        "timeframe": row.get("timeframe") or format_list(record.summary.get("timeframes")),
                        "agent_name": row.get("agent_name"),
                        "strategy": row.get("strategy"),
                        "trades": row.get("trades"),
                        "trades_per_day": row.get("trades_per_day"),
                        "net": row.get("net"),
                        "pf": row.get("pf"),
                        "median_daily_return_pct": row.get("median_daily_return_pct"),
                        "fee_drag_pct": row.get("fee_drag_pct"),
                        "total_execution_drag_pct": row.get("total_execution_drag_pct"),
                        "edge_confidence": row.get("edge_confidence"),
                        "walk_forward_verdict": row.get("verdict") or row.get("walk_forward_verdict"),
                    }
                    dedupe_key = leaderboard_dedupe_key(candidate)
                    if dedupe_key not in seen:
                        seen.add(dedupe_key)
                        rows.append(candidate)
        for field in ("best_at_least_30", "best_overall"):
            row = safe_dict(record.summary.get(field))
            if not row:
                continue
            daily = daily_metrics_from_summary(record.summary, row)
            leaderboard_row = {
                "run_label": record.run_label,
                "timeframe": row.get("timeframe") or format_list(record.summary.get("timeframes")),
                "strategy": row.get("strategy"),
                "trades": row.get("trades"),
                "trades_per_day": daily.get("trades_per_day"),
                "net": row.get("net"),
                "pf": row.get("pf"),
                "median_daily_return_pct": daily.get("median_daily_return_pct"),
                "fee_drag_pct": daily.get("fee_drag_pct"),
                "total_execution_drag_pct": row.get("total_execution_drag_pct") or record.summary.get("total_execution_drag_pct"),
                "edge_confidence": row.get("edge_confidence") or record.summary.get("edge_confidence"),
                "walk_forward_verdict": row.get("walk_forward_verdict") or record.summary.get("walk_forward_verdict"),
            }
            dedupe_key = leaderboard_dedupe_key(leaderboard_row)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            rows.append(leaderboard_row)
    return sorted(
        rows,
        key=lambda row: (
            score_float(row.get("net"), missing=float("-inf")),
            score_float(row.get("pf"), missing=0.0),
            target_score(row.get("trades_per_day"), DAILY_TRADE_TARGET),
            -score_float(row.get("fee_drag_pct"), missing=0.0),
        ),
        reverse=True,
    )


def leaderboard_dedupe_key(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(row.get("timeframe")),
        str(row.get("agent_name")),
        str(row.get("strategy")),
        str(row.get("trades")),
        f"{to_float(row.get('net')) or 0.0:.4f}",
    )


def daily_metrics_from_summary(summary: dict[str, Any], row: dict[str, Any] | None = None) -> dict[str, Any]:
    if row:
        row_metrics = safe_dict(row.get("daily_metrics"))
        if row_metrics:
            return row_metrics
    metrics = safe_dict(summary.get("daily_scalping_metrics"))
    if metrics:
        return metrics
    return {}


def target_score(value: Any, target: float) -> float:
    numeric = to_float(value)
    if numeric is None:
        return 0.0
    return min(numeric / max(target, 1e-9), 1.0)


def score_float(value: Any, missing: float) -> float:
    numeric = to_float(value)
    return missing if numeric is None else numeric


def best_summary_row(summaries: list[dict[str, Any]], field: str) -> dict[str, Any] | None:
    rows = [safe_dict(summary.get(field)) for summary in summaries]
    rows = [row for row in rows if row]
    return max(rows, key=lambda row: score_float(row.get("net"), missing=float("-inf")), default=None)


def count_values(records: list[SummaryRecord], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = text_value(record.summary.get(field))
        if value != "n/a":
            counts[value] = counts.get(value, 0) + 1
    return counts


def format_row(value: Any) -> str:
    row = safe_dict(value)
    if not row:
        return "n/a"
    return (
        f"{row.get('symbol', 'n/a')} {row.get('strategy', 'n/a')} "
        f"target={number(row.get('target_bps'))}bps hold={row.get('hold', 'n/a')} "
        f"trades={row.get('trades', 'n/a')} net={money(row.get('net'))} pf={number(row.get('pf'))}"
    )


def format_soft_thresholds(value: Any) -> str:
    thresholds = safe_dict(value)
    if not thresholds:
        return "soft thresholds: n/a"
    return (
        "soft thresholds: "
        f"long_rsi={number(thresholds.get('soft_rsi_high_long'))}, "
        f"long_close={number(thresholds.get('soft_close_position_high_long'))}, "
        f"short_rsi={number(thresholds.get('soft_rsi_low_short'))}, "
        f"short_close={number(thresholds.get('soft_close_position_low_short'))}"
    )


def format_cluster(value: Any) -> str:
    cluster = safe_dict(value)
    if not cluster:
        return "n/a"
    return (
        f"{cluster.get('side', 'n/a')} {cluster.get('strategy', 'n/a')} "
        f"exit={cluster.get('exit_reason', 'n/a')} count={cluster.get('count', 'n/a')} "
        f"net={money(cluster.get('net'))} rsi={cluster.get('rsi_band', 'n/a')} "
        f"close={cluster.get('close_position_band', 'n/a')} hold={cluster.get('hold_band', 'n/a')} "
        f"label={cluster.get('soft_label', 'n/a')}"
    )


def format_soft_rejections(value: Any) -> str:
    row = safe_dict(value)
    if not row:
        return "n/a"
    return (
        f"long={row.get('rejected_soft_late_long', 0)} | "
        f"short={row.get('rejected_soft_late_short', 0)}"
    )


def format_side_summary(value: Any) -> str:
    row = safe_dict(value)
    if not row:
        return "n/a"
    return f"trades={row.get('trades', 'n/a')} net={money(row.get('net'))} avg={money(row.get('avg_net'))} pf={number(row.get('pf'))}"


def format_entry_cluster(value: Any) -> str:
    row = safe_dict(value)
    if not row:
        return "n/a"
    return (
        f"{row.get('side', 'n/a')} trades={row.get('trades', 'n/a')} wins={row.get('wins', 'n/a')} "
        f"net={money(row.get('net'))} avg={money(row.get('avg_net'))} pf={number(row.get('pf'))} "
        f"rsi={row.get('rsi_band', 'n/a')} macd={row.get('macd_band', 'n/a')} "
        f"trend={row.get('trend_regime', 'n/a')} close={row.get('close_position_band', 'n/a')}"
    )


def format_momentum_cluster(value: Any) -> str:
    row = safe_dict(value)
    if not row:
        return "n/a"
    return (
        f"{row.get('side', 'n/a')} exit={row.get('exit_reason', 'n/a')} "
        f"trades={row.get('trades', 'n/a')} net={money(row.get('net'))} "
        f"avg={money(row.get('avg_net'))} pf={number(row.get('pf'))} "
        f"rsi={row.get('rsi_band', 'n/a')} macd={row.get('macd_band', 'n/a')} "
        f"trend={row.get('trend_regime', 'n/a')} volume={row.get('volume_band', 'n/a')}"
    )


def money(value: Any) -> str:
    numeric = to_float(value)
    if numeric is None:
        return "n/a"
    return f"${numeric:,.2f}"


def number(value: Any) -> str:
    numeric = to_float(value)
    if numeric is None:
        return "n/a"
    if numeric == float("inf"):
        return "inf"
    return f"{numeric:.2f}"


def whole(value: Any) -> str:
    numeric = to_float(value)
    if numeric is None:
        return "n/a"
    return f"{int(numeric):,}"


def days(value: Any) -> str:
    numeric = to_float(value)
    if numeric is None:
        return "n/a"
    return f"{numeric:,.1f}"


def percent(value: Any) -> str:
    numeric = to_float(value)
    if numeric is None:
        return "n/a"
    return f"{numeric:.2f}%"


def display_label(value: Any) -> str:
    text = text_value(value)
    if text in DISPLAY_LABELS:
        return DISPLAY_LABELS[text]
    if text == "n/a":
        return text
    if "_" in text:
        return text.replace("_", " ").capitalize()
    return text


def text_value(value: Any) -> str:
    if value is None or value == "":
        return "n/a"
    return str(value)


def safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def normalize_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item)]
    if isinstance(value, str) and value:
        return [part.strip() for part in value.split(",") if part.strip()]
    return []


def format_list(value: Any) -> str:
    items = normalize_list(value)
    return ", ".join(items) if items else "n/a"


def short_timestamp(value: str) -> str:
    if not value:
        return "n/a"
    return value.replace("T", " ")[:19]


def counter_value(value: str) -> str:
    clean = value.replace("$", "").replace(",", "")
    try:
        float(clean)
    except ValueError:
        return ""
    return clean


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()
