@echo off
setlocal
cd /d "%~dp0.."

set "PAUSE_ON_EXIT=1"
set "INCLUDE_1M=0"
if /i "%~1"=="--no-pause" set "PAUSE_ON_EXIT=0"
if /i "%~2"=="--no-pause" set "PAUSE_ON_EXIT=0"
if /i "%~1"=="--include-1m" set "INCLUDE_1M=1"
if /i "%~2"=="--include-1m" set "INCLUDE_1M=1"

if not exist ".\.venv\Scripts\python.exe" (
  echo Missing .\.venv\Scripts\python.exe
  if "%PAUSE_ON_EXIT%"=="1" pause
  exit /b 1
)

echo Downloading BTCUSDT 3-year public klines for 15m then 5m.
echo This uses public market data only. No API keys, no account access, no trading.

".\.venv\Scripts\python.exe" -m backtesting.download_klines --symbols BTC/USDT --interval 15m --days 1095 --sleep 0.08
if not "%ERRORLEVEL%"=="0" goto fail

".\.venv\Scripts\python.exe" -m backtesting.download_klines --symbols BTC/USDT --interval 5m --days 1095 --sleep 0.08
if not "%ERRORLEVEL%"=="0" goto fail

if "%INCLUDE_1M%"=="1" (
  echo Downloading BTCUSDT 1m 3-year data. This can be large and slow.
  ".\.venv\Scripts\python.exe" -m backtesting.download_klines --symbols BTC/USDT --interval 1m --days 1095 --sleep 0.08
  if not "%ERRORLEVEL%"=="0" goto fail
) else (
  echo Skipping 1m. Pass --include-1m when you explicitly want the large 1m dataset.
)

call "%~dp0verify_btc_3y_data.bat" --no-pause
if not "%ERRORLEVEL%"=="0" goto fail

if "%PAUSE_ON_EXIT%"=="1" pause
exit /b 0

:fail
echo BTC 3-year data download or verification failed.
if "%PAUSE_ON_EXIT%"=="1" pause
exit /b 1
