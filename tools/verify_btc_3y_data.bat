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

".\.venv\Scripts\python.exe" tools\verify_btc_3y_data.py
set "EXIT_CODE=%ERRORLEVEL%"

if "%PAUSE_ON_EXIT%"=="1" pause
exit /b %EXIT_CODE%
