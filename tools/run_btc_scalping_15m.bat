@echo off
setlocal
cd /d "%~dp0.."

echo BTCUSDT 15m scalping research workflow.
echo Backtesting only. Live trading disabled. No leverage. BTC only.
echo Targets are diagnostics: 100 trades/day and 5%% median daily return.

call "%~dp0run_btc_15m_3y_backtest.bat" %*
exit /b %ERRORLEVEL%
