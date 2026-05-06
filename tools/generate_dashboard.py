"""Generate the public static dashboard from compact realized sweep summaries.

This is reporting-only. It reads the ignored local JSONL summary log and writes
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


@dataclass(frozen=True)
class SummaryRecord:
    logged_at_utc: str
    run_label: str
    summary: dict[str, Any]
    line_number: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate docs/index.html from compact backtest summary logs.")
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
        help=f"Number of latest runs to show. Default: {LATEST_LIMIT}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_records(args.log_path)
    latest = records[-max(args.latest, 0) :]
    html = render_dashboard(records=records, latest=latest, log_path=args.log_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(f"Dashboard written: {args.output}")
    print(f"Summary records read: {len(records)}")


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


def record_sort_key(record: SummaryRecord) -> tuple[datetime, int]:
    parsed = parse_timestamp(record.logged_at_utc)
    return parsed, record.line_number


def parse_timestamp(raw: str) -> datetime:
    if not raw:
        return datetime.min
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return datetime.min


def render_dashboard(records: list[SummaryRecord], latest: list[SummaryRecord], log_path: Path) -> str:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    latest_record = latest[-1] if latest else None
    latest_summary = latest_record.summary if latest_record else {}
    best_overall = best_row(records, "best_overall")
    best_30 = best_row(records, "best_at_least_30")
    worst_overall = worst_row(records, "worst_overall")
    verdict_counts = count_values(records, "verdict")

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Trading Bot Research Dashboard</title>
  <link rel="stylesheet" href="assets/styles.css">
</head>
<body>
  <header class="hero">
    <div class="shell">
      <h1>Trading Bot Research Dashboard</h1>
      <p>Local calibration and backtest dashboard for the Binance crypto paper trading bot. This page contains compact summaries only and is safe for GitHub Pages review.</p>
      <div class="status-strip">
        <span class="pill ok">Paper mode default</span>
        <span class="pill ok">Live trading disabled</span>
        <span class="pill ok">No leverage workflow</span>
        <span class="pill warn">Research/backtesting only</span>
      </div>
    </div>
  </header>

  <main class="shell grid">
    <section class="panel">
      <h2>Current Focus</h2>
      <p>Improve calibration diagnostics, compare realized sweep summaries, and determine whether any backtest-only quality filters produce robust 30+ trade results before promotion to paper execution is considered.</p>
      <div class="notice">This dashboard is not financial advice. Results are historical research diagnostics and may not predict future performance.</div>
    </section>

    <section class="panel">
      <h2>Run Overview</h2>
      <div class="metric-grid">
        {metric_card("Summary records", str(len(records)))}
        {metric_card("Latest notional", money(latest_summary.get("diagnostic_notional")))}
        {metric_card("Latest verdict", text_value(latest_summary.get("verdict")), verdict_class(latest_summary.get("verdict")))}
        {metric_card("Generated", generated_at)}
      </div>
      <p class="muted">Source log: {escape(str(log_path))}. Raw JSONL logs remain local and ignored by git.</p>
    </section>

    <section class="panel">
      <h2>Latest Compact Backtest Summaries</h2>
      {render_latest_table(latest)}
    </section>

    <section class="panel third">
      <h2>Best Overall</h2>
      {render_result_box(best_overall)}
    </section>

    <section class="panel third">
      <h2>Best With 30+ Trades</h2>
      {render_result_box(best_30)}
    </section>

    <section class="panel third">
      <h2>Worst Overall</h2>
      {render_result_box(worst_overall)}
    </section>

    <section class="panel half">
      <h2>Latest Quality Diagnostics</h2>
      {render_quality_block(latest_summary)}
    </section>

    <section class="panel half">
      <h2>Verdicts</h2>
      {render_verdict_counts(verdict_counts)}
    </section>

    <section class="panel half">
      <h2>Known Issues</h2>
      <ul class="clean">
        <li>Realized historical simulations have not produced robust profitable 30+ trade settings yet.</li>
        <li>Momentum sells have been weaker than buys in recent diagnostics.</li>
        <li>Fixed dollar expected-net thresholds can over-filter small diagnostic notional runs.</li>
        <li>Raw result text files, raw JSONL logs, and historical CSVs must remain local and ignored.</li>
      </ul>
    </section>

    <section class="panel half">
      <h2>Future Tasks</h2>
      <ul class="clean">
        <li>Review a calibration-only expected-net threshold that scales with diagnostic notional.</li>
        <li>Compare soft-late threshold sweeps with at least 30 trades.</li>
        <li>Study entry-only momentum clusters before promoting any filter.</li>
        <li>Test longer history in order: 15m first, then 5m, then 1m.</li>
      </ul>
    </section>

    <section class="panel half">
      <h2>Ideas Backlog</h2>
      <ul class="clean">
        <li>Add chart snapshots from compact summaries only.</li>
        <li>Add per-run notes without exposing raw logs.</li>
        <li>Track BTC-only default behavior separately from explicit manual ETH backtests.</li>
        <li>Separate production thresholds from calibration diagnostics in reports.</li>
      </ul>
    </section>

    <section class="panel half">
      <h2>Local Workflow</h2>
      <div class="code">tools\\run_focused_backtest.bat<br>tools\\compare_logs.bat<br>tools\\update_dashboard.bat<br>tools\\safe_git_status.bat</div>
      <p class="muted">The backtest helper writes ignored local files under data/. The dashboard generator reads compact JSONL summaries and updates docs/index.html.</p>
    </section>

    <section class="panel">
      <h2>GitHub Workflow</h2>
      <ol>
        <li>Run local backtests manually with the helper script.</li>
        <li>Regenerate the dashboard locally.</li>
        <li>Review changed files with <span class="code">tools\\safe_git_status.bat</span>.</li>
        <li>Commit only code and docs. Keep data/*.txt, data/backtest_logs/, historical CSVs, .env, .venv, and caches local.</li>
        <li>Push only after explicit approval.</li>
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
      <p class="muted">No logs yet. Run a focused backtest with --save-summary-log, then run tools\\update_dashboard.bat.</p>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Status</th></tr></thead>
          <tbody><tr><td>No compact summary logs found.</td></tr></tbody>
        </table>
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
              <th>Notional</th>
              <th>Soft late</th>
              <th>Total</th>
              <th>Pos</th>
              <th>Pos 30+</th>
              <th>Best overall</th>
              <th>Best 30+</th>
              <th>Verdict</th>
            </tr>
          </thead>
          <tbody>
            {rows}
          </tbody>
        </table>
      </div>
"""


def render_latest_row(record: SummaryRecord) -> str:
    summary = record.summary
    return f"""
            <tr>
              <td>{escape(record.run_label)}</td>
              <td>{escape(record.logged_at_utc or "n/a")}</td>
              <td>{money(summary.get("diagnostic_notional"))}</td>
              <td>{escape(text_value(summary.get("reject_soft_late_momentum")))}</td>
              <td>{escape(text_value(summary.get("total_combinations")))}</td>
              <td>{escape(text_value(summary.get("positive_combinations")))}</td>
              <td>{escape(text_value(summary.get("positive_combinations_with_at_least_30_trades")))}</td>
              <td>{escape(format_row(summary.get("best_overall")))}</td>
              <td>{escape(format_row(summary.get("best_at_least_30")))}</td>
              <td>{verdict_tag(summary.get("verdict"))}</td>
            </tr>"""


def render_result_box(row: dict[str, Any] | None) -> str:
    if not row:
        return '<p class="muted">n/a</p>'
    return f"""
      <div class="result-line">
        <strong>{escape(str(row.get("symbol", "n/a")))} {escape(str(row.get("strategy", "n/a")))}</strong>
        <span>Trades: {escape(text_value(row.get("trades")))} | Net: {money(row.get("net"))} | Avg: {money(row.get("avg_net"))} | PF: {number(row.get("pf"))}</span>
        <span>Target: {number(row.get("target_bps"))}bps | Reward/cost: {number(row.get("reward_cost"))}x | Hold: {escape(text_value(row.get("hold")))}</span>
        <span>ATR TP: {number(row.get("atrtp"))} | ATR SL: {number(row.get("atrsl"))}</span>
        <span>{escape(format_soft_thresholds(row.get("soft_thresholds")))}</span>
      </div>
"""


def render_quality_block(summary: dict[str, Any]) -> str:
    if not summary:
        return '<p class="muted">No diagnostics available yet.</p>'
    parts = [
        f"<h3>Top quality rejections</h3>{render_rejection_list(summary.get('top_quality_rejections'))}",
        f"<h3>Top accepted loser cluster</h3><p>{escape(format_cluster(summary.get('top_accepted_loser_cluster')))}</p>",
        f"<h3>Momentum side summary</h3><p>Buy: {escape(format_side_summary(summary.get('buy_momentum')))}<br>Sell: {escape(format_side_summary(summary.get('sell_momentum')))}</p>",
        f"<h3>Best entry-only momentum cluster at 30+</h3><p>{escape(format_entry_cluster(summary.get('best_entry_momentum_cluster_at_least_30')))}</p>",
    ]
    return "\n".join(parts)


def render_rejection_list(value: Any) -> str:
    if not isinstance(value, list) or not value:
        return '<p class="muted">n/a</p>'
    items = []
    for item in value[:8]:
        if not isinstance(item, dict):
            continue
        reason = escape(str(item.get("reason", "n/a")))
        count = escape(str(item.get("count", "n/a")))
        items.append(f"<li>{reason}: {count}</li>")
    if not items:
        return '<p class="muted">n/a</p>'
    return f'<ul class="clean">{"".join(items)}</ul>'


def render_verdict_counts(counts: dict[str, int]) -> str:
    if not counts:
        return '<p class="muted">No verdicts yet.</p>'
    items = [f"<li>{escape(verdict)}: {count}</li>" for verdict, count in sorted(counts.items())]
    return f'<ul class="clean">{"".join(items)}</ul>'


def metric_card(label: str, value: str, value_class: str = "") -> str:
    class_attr = f" {value_class}" if value_class else ""
    return f"""
        <div class="metric">
          <div class="label">{escape(label)}</div>
          <div class="value{class_attr}">{escape(value)}</div>
        </div>"""


def verdict_tag(value: Any) -> str:
    verdict = text_value(value)
    class_name = verdict_class(verdict)
    tag_class = f"tag {class_name}" if class_name else "tag"
    return f'<span class="{tag_class}">{escape(verdict)}</span>'


def verdict_class(value: Any) -> str:
    verdict = str(value or "")
    if "promising" in verdict:
        return "good"
    if "not_profitable" in verdict:
        return "bad"
    if "too_few" in verdict:
        return "warn"
    return ""


def best_row(records: list[SummaryRecord], field: str) -> dict[str, Any] | None:
    rows = [row for row in (safe_dict(record.summary.get(field)) for record in records) if row]
    return max(rows, key=lambda row: (to_float(row.get("net")) or float("-inf"), to_float(row.get("avg_net")) or float("-inf")), default=None)


def worst_row(records: list[SummaryRecord], field: str) -> dict[str, Any] | None:
    rows = [row for row in (safe_dict(record.summary.get(field)) for record in records) if row]
    return min(rows, key=lambda row: (to_float(row.get("net")) or float("inf"), to_float(row.get("avg_net")) or float("inf")), default=None)


def count_values(records: list[SummaryRecord], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = text_value(record.summary.get(field))
        if value == "n/a":
            continue
        counts[value] = counts.get(value, 0) + 1
    return counts


def format_row(value: Any) -> str:
    row = safe_dict(value)
    if not row:
        return "n/a"
    return (
        f"{row.get('symbol', 'n/a')} {row.get('strategy', 'n/a')} "
        f"target={number(row.get('target_bps'))}bps hold={row.get('hold', 'n/a')} "
        f"trades={row.get('trades', 'n/a')} net={money(row.get('net'))} "
        f"avg={money(row.get('avg_net'))} pf={number(row.get('pf'))}"
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
        f"{row.get('side', 'n/a')} trades={row.get('trades', 'n/a')} "
        f"net={money(row.get('net'))} avg={money(row.get('avg_net'))} pf={number(row.get('pf'))} "
        f"rsi={row.get('rsi_band', 'n/a')} macd={row.get('macd_band', 'n/a')} "
        f"trend={row.get('trend_regime', 'n/a')} close={row.get('close_position_band', 'n/a')}"
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


def text_value(value: Any) -> str:
    if value is None or value == "":
        return "n/a"
    return str(value)


def safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()
