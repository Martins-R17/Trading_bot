@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0.."

if not exist ".\.venv\Scripts\python.exe" (
  echo Missing .\.venv\Scripts\python.exe
  exit /b 1
)

for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "STAMP=%%I"

for %%T in (15m 5m 1m) do (
  set "LABEL=btc_%%T_structural_validation_!STAMP!"
  echo Running BTCUSDT %%T structural validation...
  ".\.venv\Scripts\python.exe" -m backtesting.scalping_search ^
    --timeframe %%T ^
    --quality-profile structural ^
    --data-dir "data\historical_3y_%%T" ^
    --max-hold-minutes 60 ^
    --monte-carlo-iterations 1000 ^
    --save-summary-log ^
    --run-label "!LABEL!" > "data\!LABEL!.txt"
  if errorlevel 1 exit /b !errorlevel!
)

call tools\update_dashboard.bat --no-pause
exit /b %errorlevel%
