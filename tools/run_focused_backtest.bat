@echo off
setlocal
cd /d "%~dp0.."

set "PAUSE_ON_EXIT=1"
if /i "%~1"=="--no-pause" set "PAUSE_ON_EXIT=0"

if not exist ".\.venv\Scripts\python.exe" (
  echo Missing .\.venv\Scripts\python.exe
  echo Create or activate the project virtual environment first.
  if "%PAUSE_ON_EXIT%"=="1" pause
  exit /b 1
)

if not exist "data\historical_90d_15m\ETHUSDT_15m.csv" (
  echo Missing data\historical_90d_15m\ETHUSDT_15m.csv
  echo Download or place the historical CSV locally before running this focused sweep.
  if "%PAUSE_ON_EXIT%"=="1" pause
  exit /b 1
)

if not exist "data" mkdir "data"
if not exist "data\backtest_logs" mkdir "data\backtest_logs"

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "RUN_TS=%%i"
set "RUN_LABEL=focused_eth15m_soft_thresholds_%RUN_TS%"
set "OUTPUT_FILE=data\focused_backtest_%RUN_TS%.txt"

echo Running focused ETH 15m realized sweep.
echo Run label: %RUN_LABEL%
echo Output file: %OUTPUT_FILE%
echo This is calibration/backtesting only. No live trading, no orders, no leverage.

".\.venv\Scripts\python.exe" -m backtesting.calibration ^
  --symbols ETH/USDT ^
  --csv "ETH/USDT=data\historical_90d_15m\ETHUSDT_15m.csv" ^
  --limit 3000 ^
  --realized-sweep ^
  --timeframe 15m ^
  --target-sweep 75,150 ^
  --reward-cost-sweep 3.0 ^
  --max-hold-sweep 16,32 ^
  --atr-tp-sweep 3.0,4.0 ^
  --atr-stop-sweep 1.0,1.5 ^
  --soft-rsi-high-long-sweep 65,67,69 ^
  --soft-close-position-high-long-sweep 0.70,0.80,0.90 ^
  --soft-rsi-low-short-sweep 31,33,35 ^
  --soft-close-position-low-short-sweep 0.10,0.20,0.30 ^
  --reject-soft-late-momentum ^
  --save-summary-log ^
  --run-label "%RUN_LABEL%" ^
  > "%OUTPUT_FILE%"

set "EXIT_CODE=%ERRORLEVEL%"
if "%EXIT_CODE%"=="0" (
  echo Focused backtest finished.
  echo Next: tools\compare_logs.bat
  echo Next: tools\update_dashboard.bat
) else (
  echo Focused backtest failed with exit code %EXIT_CODE%.
)

if "%PAUSE_ON_EXIT%"=="1" pause
exit /b %EXIT_CODE%
