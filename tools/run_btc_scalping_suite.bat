@echo off
setlocal
cd /d "%~dp0.."

set "PAUSE_ON_EXIT=1"
if /i "%~1"=="--no-pause" set "PAUSE_ON_EXIT=0"

echo BTCUSDT high-quality futures strategy research suite.
echo Backtesting only. Live trading disabled. No leverage. BTC only.
echo Runs 15m first, then 5m. Use run_btc_scalping_1m.bat --confirm-large-1m for the larger 1m run.

call "%~dp0run_btc_scalping_15m.bat" --no-pause
if not "%ERRORLEVEL%"=="0" (
  echo 15m scalping workflow failed.
  if "%PAUSE_ON_EXIT%"=="1" pause
  exit /b 1
)

call "%~dp0run_btc_scalping_5m.bat" --no-pause
if not "%ERRORLEVEL%"=="0" (
  echo 5m scalping workflow failed.
  if "%PAUSE_ON_EXIT%"=="1" pause
  exit /b 2
)

call "%~dp0compare_logs.bat" --no-pause
if not "%ERRORLEVEL%"=="0" (
  echo Log comparison failed.
  if "%PAUSE_ON_EXIT%"=="1" pause
  exit /b 3
)
call "%~dp0update_dashboard.bat" --no-pause
if not "%ERRORLEVEL%"=="0" (
  echo Dashboard update failed.
  if "%PAUSE_ON_EXIT%"=="1" pause
  exit /b 4
)

echo Suite complete. Review logs and dashboard before considering 1m.
if "%PAUSE_ON_EXIT%"=="1" pause
exit /b 0
