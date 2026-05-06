@echo off
setlocal
cd /d "%~dp0.."

set "PAUSE_ON_EXIT=1"
set "INCLUDE_1M=0"

if /i "%~1"=="--no-pause" set "PAUSE_ON_EXIT=0"
if /i "%~2"=="--no-pause" set "PAUSE_ON_EXIT=0"
if /i "%~1"=="--include-1m" set "INCLUDE_1M=1"
if /i "%~2"=="--include-1m" set "INCLUDE_1M=1"

echo Running BTC optimization suite: 15m, then 5m.
echo 1m is skipped unless --include-1m is passed.
echo This is calibration/backtesting only. No live trading, no orders, no leverage.

call "%~dp0run_btc_15m_3y_backtest.bat" --no-pause
if not "%ERRORLEVEL%"=="0" goto fail

call "%~dp0run_btc_5m_3y_backtest.bat" --no-pause
if not "%ERRORLEVEL%"=="0" goto fail

if "%INCLUDE_1M%"=="1" (
  call "%~dp0run_btc_1m_3y_backtest.bat" --no-pause --confirm-large-1m
  if not "%ERRORLEVEL%"=="0" goto fail
)

echo BTC optimization suite finished.
if "%PAUSE_ON_EXIT%"=="1" pause
exit /b 0

:fail
echo BTC optimization suite stopped because a step failed.
if "%PAUSE_ON_EXIT%"=="1" pause
exit /b 1
