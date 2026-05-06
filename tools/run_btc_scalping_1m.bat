@echo off
setlocal
cd /d "%~dp0.."

echo BTCUSDT 1m scalping research workflow.
echo Backtesting only. Live trading disabled. No leverage. BTC only.
echo 1m 3-year data is large; the underlying script requires --confirm-large-1m.
echo Targets are diagnostics: 100 trades/day and 5%% median daily return.

call "%~dp0run_btc_1m_3y_backtest.bat" %*
exit /b %ERRORLEVEL%
