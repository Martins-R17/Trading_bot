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
  if "%PAUSE_ON_EXIT%"=="1" pause
  exit /b 1
)

if not exist "data" mkdir "data"
if not exist "data\backtest_logs" mkdir "data\backtest_logs"

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "RUN_TS=%%i"
set "RUN_LABEL=btc_5m_fast_futures_scalping_agents_%RUN_TS%"
set "OUTPUT_FILE=data\btc_5m_fast_futures_scalping_agents_%RUN_TS%.txt"

echo BTCUSDT 5m fast futures scalping agent search.
echo Backtesting only. Live trading disabled. No real orders. Simulated leverage only.
echo Uses Binance futures-style fee defaults: maker 0.02%%, taker 0.05%%.
echo Targets are diagnostics: 100 trades/day, 5%% avg daily return, 75%% days above 5%%.

".\.venv\Scripts\python.exe" -m backtesting.scalping_search ^
  --symbol BTC/USDT ^
  --timeframe 5m ^
  --data-dir data\historical_3y_5m ^
  --simulated-leverage 1 ^
  --max-parameter-sets 120 ^
  --save-summary-log ^
  --run-label "%RUN_LABEL%" ^
  > "%OUTPUT_FILE%"

set "EXIT_CODE=%ERRORLEVEL%"
if "%EXIT_CODE%"=="0" (
  call "%~dp0update_dashboard.bat" --no-pause
  if not "%ERRORLEVEL%"=="0" exit /b 3
) else (
  echo BTC 5m fast scalping search failed with exit code %EXIT_CODE%.
)

if "%PAUSE_ON_EXIT%"=="1" pause
exit /b %EXIT_CODE%
