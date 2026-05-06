@echo off
setlocal
cd /d "%~dp0.."

set "PAUSE_ON_EXIT=1"
if /i "%~1"=="--no-pause" set "PAUSE_ON_EXIT=0"

if not exist ".\.venv\Scripts\python.exe" (
  echo Missing .\.venv\Scripts\python.exe
  if "%PAUSE_ON_EXIT%"=="1" pause
  exit /b 1
)

if not exist "data\historical_3y_5m\BTCUSDT_5m.csv" (
  echo Missing data\historical_3y_5m\BTCUSDT_5m.csv
  echo Download BTC 5m data first:
  echo .\.venv\Scripts\python.exe -m backtesting.download_klines --symbols BTC/USDT --interval 5m --days 1095
  if "%PAUSE_ON_EXIT%"=="1" pause
  exit /b 1
)

if not exist "data" mkdir "data"
if not exist "data\backtest_logs" mkdir "data\backtest_logs"

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "RUN_TS=%%i"
set "RUN_LABEL=btc_5m_3y_single_baseline_%RUN_TS%"
set "OUTPUT_FILE=data\btc_5m_3y_single_baseline_%RUN_TS%.txt"

echo Running BTC 5m 3-year realized single-baseline test.
echo Run label: %RUN_LABEL%
echo Output file: %OUTPUT_FILE%
echo This is calibration/backtesting only. No live trading, no orders, no leverage.

".\.venv\Scripts\python.exe" -m backtesting.calibration ^
  --symbols BTC/USDT ^
  --data-dir data\historical_3y_5m ^
  --years 3 ^
  --realized-sweep ^
  --timeframe 5m ^
  --target-sweep 75 ^
  --reward-cost-sweep 3.0 ^
  --max-hold-sweep 16 ^
  --atr-tp-sweep 3.0 ^
  --atr-stop-sweep 1.5 ^
  --reject-soft-late-momentum ^
  --save-summary-log ^
  --run-label "%RUN_LABEL%" ^
  > "%OUTPUT_FILE%"

set "EXIT_CODE=%ERRORLEVEL%"
if "%EXIT_CODE%"=="0" (
  echo BTC 5m backtest finished.
  call "%~dp0update_dashboard.bat" --no-pause
) else (
  echo BTC 5m backtest failed with exit code %EXIT_CODE%.
)

if "%PAUSE_ON_EXIT%"=="1" pause
exit /b %EXIT_CODE%
