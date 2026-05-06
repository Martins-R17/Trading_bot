"""Generate the public static research dashboard from compact summary logs.

This is reporting-only. It reads ignored local JSONL summary records and writes
docs/index.html. It does not run backtests, download data, place orders, or
read credentials.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any


DEFAULT_LOG_PATH = Path("data/backtest_logs/realized_sweep_summary.jsonl")
DEFAULT_OUTPUT_PATH = Path("docs/index.html")
LATEST_LIMIT = 10
DEFAULT_SYMBOL_FILTER = "BTC/USDT"


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
    args.output.write_text(html, encoding="utf-8")
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
    best_overall = best_row(records, "best_overall")
    best_30 = best_row(records, "best_at_least_30")
    worst_overall = worst_row(records, "worst_overall")
    verdict_counts = count_values(records, "verdict")
    tf_rows = timeframe_rows(records)
    tf_completion = timeframe_completion(records)
    strategy_rows = strategy_rows_from_records(records)
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
  <header class="hero">
    <div class="shell hero-grid">
      <div>
        <p class="eyebrow">BTCUSDT paper/backtesting research</p>
        <h1>BTC Scalping Research Dashboard</h1>
        <p class="hero-copy">Fee-aware, risk-aware calibration for BTC/USDT short-term strategies across 1m, 5m, and 15m data. Results are research diagnostics only.</p>
        <div class="status-strip">
          <span class="pill ok">Paper mode default</span>
          <span class="pill ok">Live trading disabled</span>
          <span class="pill ok">No leverage</span>
          <span class="pill ok">BTC-only default</span>
          <span class="pill warn">Backtesting only</span>
        </div>
      </div>
      <div class="hero-card">
        <span>Latest verdict</span>
        <strong>{escape(text_value(latest_summary.get("verdict")))}</strong>
        <small>Generated {escape(generated_at)}</small>
      </div>
    </div>
  </header>

  <main class="shell layout">
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
        {metric_card("Universe", symbol_filter, "good")}
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
        <span class="tag {verdict_class(latest_summary.get("verdict"))}">{escape(text_value(latest_summary.get("verdict")))}</span>
      </div>
      <div class="metric-grid">
        {metric_card("BTC records", str(len(records)), "neutral")}
        {metric_card("Ignored non-BTC/legacy", str(max(legacy_records, 0)), "neutral")}
        {metric_card("Diagnostic notional", money(latest_summary.get("diagnostic_notional")), "neutral")}
        {metric_card("Calibration min net", money(latest_summary.get("calibration_min_expected_net_profit")), "neutral")}
        {metric_card("Total candles", whole(latest_summary.get("total_candles")), "neutral")}
        {metric_card("Signal window", whole(latest_summary.get("signal_window_bars")), "neutral")}
        {metric_card("Data days", days(latest_summary.get("approx_days")), "neutral")}
      </div>
      {render_data_window(latest_summary)}
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
      <h2>Best Overall</h2>
      {render_result_box(best_overall)}
    </section>

    <section class="panel span-4">
      <h2>Best With 30+ Trades</h2>
      {render_result_box(best_30)}
    </section>

    <section class="panel span-4">
      <h2>Worst Overall</h2>
      {render_result_box(worst_overall)}
    </section>

    <section class="panel span-6">
      <h2>Timeframe Comparison</h2>
      {render_rank_table(tf_rows, ("Timeframe", "Runs", "Best net", "Best PF", "Best 30+"))}
    </section>

    <section class="panel span-6">
      <h2>Strategy Comparison</h2>
      {render_rank_table(strategy_rows, ("Strategy", "Rows", "Best net", "Best PF", "Best 30+"))}
    </section>

    <section class="panel span-6">
      <h2>Performance Summary</h2>
      {render_performance_table(latest_summary)}
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
  </main>

  <footer class="shell footer">
    Generated at {escape(generated_at)} from compact local summary records only.
  </footer>
</body>
</html>
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
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Run label</th>
              <th>Timestamp</th>
              <th>Timeframe</th>
              <th>Candles</th>
              <th>Notional</th>
              <th>Calib min net</th>
              <th>WF verdict</th>
              <th>Soft late</th>
              <th>Pos 30+</th>
              <th>Best overall</th>
              <th>Best 30+</th>
              <th>Verdict</th>
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
              <td>{escape(whole(summary.get("total_candles")))}</td>
              <td>{money(summary.get("diagnostic_notional"))}</td>
              <td>{money(summary.get("calibration_min_expected_net_profit"))}</td>
              <td>{escape(text_value(summary.get("walk_forward_verdict")))}</td>
              <td>{escape(text_value(summary.get("reject_soft_late_momentum")))}</td>
              <td>{escape(text_value(summary.get("positive_combinations_with_at_least_30_trades")))}</td>
              <td>{escape(format_row(summary.get("best_overall")))}</td>
              <td>{escape(format_row(summary.get("best_at_least_30")))}</td>
              <td>{verdict_tag(summary.get("verdict"))}</td>
            </tr>"""


def render_data_window(summary: dict[str, Any]) -> str:
    if not summary:
        return '<p class="muted">No BTC-only data profile has been logged yet.</p>'
    return (
        '<div class="data-window">'
        f'<span>Symbols: {escape(format_list(summary.get("symbols")))}</span>'
        f'<span>Timeframes: {escape(format_list(summary.get("timeframes")))}</span>'
        f'<span>Start: {escape(text_value(summary.get("data_period_start")))}</span>'
        f'<span>End: {escape(text_value(summary.get("data_period_end")))}</span>'
        f'<span>Production target: {number(summary.get("production_min_target_move_bps"))} bps</span>'
        f'<span>Production reward/cost: {number(summary.get("production_min_reward_cost_ratio"))}x</span>'
        "</div>"
    )


def render_timeframe_completion(rows: list[dict[str, Any]]) -> str:
    body = "".join(
        "<tr>"
        f"<td>{escape(row['timeframe'])}</td>"
        f"<td>{escape(row['data_status'])}</td>"
        f"<td>{escape(row['run_status'])}</td>"
        f"<td>{escape(row['candles'])}</td>"
        f"<td>{escape(row['days'])}</td>"
        f"<td>{escape(row['latest_run'])}</td>"
        f"<td>{verdict_tag(row['verdict'])}</td>"
        "</tr>"
        for row in rows
    )
    return (
        '<div class="table-wrap compact"><table><thead><tr>'
        '<th>Timeframe</th><th>Data</th><th>Backtest</th><th>Candles</th>'
        '<th>Days</th><th>Latest run</th><th>Verdict</th>'
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
        f"<p class=\"muted\">Verdict: {escape(text_value(row.get('walk_forward_verdict')))}. "
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


def metric_card(label: str, value: str, value_class: str = "neutral") -> str:
    return f"""
        <div class="metric">
          <div class="label">{escape(label)}</div>
          <div class="value {escape(value_class)}" data-count="{escape(counter_value(value))}">{escape(value)}</div>
        </div>"""


def verdict_tag(value: Any) -> str:
    verdict = text_value(value)
    class_name = verdict_class(verdict)
    return f'<span class="tag {class_name}">{escape(verdict)}</span>'


def verdict_class(value: Any) -> str:
    verdict = str(value or "")
    if "robust" in verdict or "promising" in verdict:
        return "good"
    if "not_profitable" in verdict:
        return "bad"
    if "too_few" in verdict or "overfit" in verdict:
        return "warn"
    return "neutral"


def best_row(records: list[SummaryRecord], field: str) -> dict[str, Any] | None:
    rows = [row for row in (safe_dict(record.summary.get(field)) for record in records) if row]
    return max(
        rows,
        key=lambda row: (to_float(row.get("net")) or float("-inf"), to_float(row.get("avg_net")) or float("-inf")),
        default=None,
    )


def worst_row(records: list[SummaryRecord], field: str) -> dict[str, Any] | None:
    rows = [row for row in (safe_dict(record.summary.get(field)) for record in records) if row]
    return min(
        rows,
        key=lambda row: (to_float(row.get("net")) or float("inf"), to_float(row.get("avg_net")) or float("inf")),
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
        best = max(result_rows, key=lambda row: to_float(row.get("net")) or float("-inf"), default=None)
        best_30_rows = [row for row in result_rows if int(to_float(row.get("trades")) or 0) >= 30]
        best_30 = max(best_30_rows, key=lambda row: to_float(row.get("net")) or float("-inf"), default=None)
        rows.append((strategy, str(len(result_rows)), money(best.get("net") if best else None), number(best.get("pf") if best else None), format_row(best_30)))
    return rows


def best_summary_row(summaries: list[dict[str, Any]], field: str) -> dict[str, Any] | None:
    rows = [safe_dict(summary.get(field)) for summary in summaries]
    rows = [row for row in rows if row]
    return max(rows, key=lambda row: to_float(row.get("net")) or float("-inf"), default=None)


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
