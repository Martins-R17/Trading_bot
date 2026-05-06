@echo off
setlocal
cd /d "%~dp0.."

echo Focused backtest now uses the BTC-only 15m 3-year workflow.
echo This is calibration/backtesting only. No live trading, no orders, no leverage.

call "%~dp0run_btc_15m_3y_backtest.bat" %*
exit /b %ERRORLEVEL%
