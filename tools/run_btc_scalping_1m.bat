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

if not exist "data\historical_3y_1m\BTCUSDT_1m.csv" (
  echo Missing data\historical_3y_1m\BTCUSDT_1m.csv
  if "%PAUSE_ON_EXIT%"=="1" pause
  exit /b 1
)

if /i not "%~2"=="--confirm-large-1m" if /i not "%~1"=="--confirm-large-1m" (
  echo BTC 1m 3-year agent search is large.
  echo Pass --confirm-large-1m to run full 1m. For quick feasibility, pass --limit manually with python -m backtesting.scalping_search.
  if "%PAUSE_ON_EXIT%"=="1" pause
  exit /b 2
)

if not exist "data" mkdir "data"
if not exist "data\backtest_logs" mkdir "data\backtest_logs"

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "RUN_TS=%%i"
set "RUN_LABEL=btc_1m_high_quality_futures_search_%RUN_TS%"
set "OUTPUT_FILE=data\btc_1m_high_quality_futures_search_%RUN_TS%.txt"

echo BTCUSDT 1m high-quality lower-frequency futures strategy search.
echo Backtesting only. Live trading disabled. No real orders. Simulated leverage only.
echo Uses Binance futures-style fee defaults: maker 0.02%%, taker 0.05%%.
echo Targets are diagnostics: 5-20 trades/day, 0.3%%-2%% target moves, max 60 minute holds, realistic execution drag.

".\.venv\Scripts\python.exe" -m backtesting.scalping_search ^
  --symbol BTC/USDT ^
  --timeframe 1m ^
  --quality-profile high_quality ^
  --data-dir data\historical_3y_1m ^
  --simulated-leverage 1 ^
  --max-parameter-sets 0 ^
  --save-summary-log ^
  --run-label "%RUN_LABEL%" ^
  > "%OUTPUT_FILE%"

set "EXIT_CODE=%ERRORLEVEL%"
if "%EXIT_CODE%"=="0" (
  call "%~dp0update_dashboard.bat" --no-pause
  if not "%ERRORLEVEL%"=="0" exit /b 3
) else (
  echo BTC 1m fast scalping search failed with exit code %EXIT_CODE%.
)

if "%PAUSE_ON_EXIT%"=="1" pause
exit /b %EXIT_CODE%
