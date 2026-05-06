# Trading Bot Research System

BTC-only, paper/backtesting-first research workflow for BTCUSDT short-term strategy diagnostics.

## Safety Rules

- Live trading remains disabled by default.
- Paper mode remains the default runtime mode.
- No leverage is used in this workflow.
- No API keys or secrets are hardcoded.
- Generated files stay local: `data/*.txt`, `data/backtest_logs/`, historical CSVs, `.env`, `.venv`, cache folders, and `*.pyc`.
- Results are research diagnostics only and are not financial advice.

## Current Scope

- Symbol: `BTC/USDT`
- Timeframes: `15m`, `5m`, `1m`
- Historical target: 3 years
- Primary workflow: download public BTC klines, run realized calibration, compare compact summaries, regenerate the GitHub Pages dashboard.

## Data Setup

Download BTC public klines for 15m and 5m first:

```bat
tools\download_btc_3y_data.bat
```

Download the larger 1m dataset only when ready:

```bat
tools\download_btc_3y_data.bat --include-1m
```

Verify local data coverage:

```bat
tools\verify_btc_3y_data.bat
```

The expected approximate row counts are:

- `15m`: `105,120`
- `5m`: `315,360`
- `1m`: `1,576,800`

## Backtest Order

Run progressively:

```bat
tools\run_btc_15m_3y_backtest.bat
tools\run_btc_5m_3y_backtest.bat
tools\run_btc_1m_3y_backtest.bat --confirm-large-1m
```

The 1m script is intentionally guarded because full 3-year 1m data is large.

Scalping-focused aliases are also available:

```bat
tools\run_btc_scalping_15m.bat
tools\run_btc_scalping_5m.bat
tools\run_btc_scalping_1m.bat --confirm-large-1m
tools\run_btc_scalping_suite.bat
```

The suite runs 15m and 5m only. Run 1m separately after runtime and strategy evidence justify it.

## Reporting

Compare BTC-only compact logs:

```bat
tools\compare_logs.bat
```

Regenerate the static dashboard:

```bat
tools\update_dashboard.bat
```

Open:

```text
docs\index.html
```

Dashboard metrics include diagnostic notional, calibration-only minimum expected net profit, trades/day, median daily return, profitable days, days above 5%, max daily drawdown, fee drag, walk-forward verdict, and compact best/worst candidates.

## Acceptance Criteria

A candidate is not considered promising unless it has:

- At least 30 trades, preferably 100+.
- For high-frequency scalping research, a separate objective is 100+ trades/day. This is a target to measure, not a promise.
- The 5% daily return target is reported as a diagnostic objective only and must be validated after fees, slippage, and walk-forward splits.
- Positive realized net PnL after fees and slippage.
- Profit factor above 1.1 minimum, preferably above 1.2.
- Acceptable max drawdown.
- Positive chronological train/validation/test behavior.
- No dependence on one tiny cluster or one lucky trade.

Current verdict labels include:

- `too_few_trades`
- `not_profitable_at_30_trades`
- `not_profitable_out_of_sample`
- `weak_overfit_risk`
- `potentially_promising_needs_more_testing`
- `robust_candidate`

## Git Safety

Before committing:

```bat
tools\safe_git_status.bat
tools\pre_commit_safety_check.bat
```

Commit only safe code/docs/tooling files. Do not commit generated data, logs, secrets, virtual environments, or caches.
