@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0.."

set "PAUSE_ON_EXIT=1"
if /i "%~1"=="--no-pause" set "PAUSE_ON_EXIT=0"

echo === Git Status ===
git status --short
echo.

echo === Changed Files ===
git diff --name-only
echo.

echo === Staged Files ===
git diff --cached --name-only
echo.

set "FORBIDDEN_STAGED=0"
for /f "delims=" %%f in ('git diff --cached --name-only') do (
  echo %%f | findstr /R /I "^data/ ^\\.env$ ^\\.venv/ .*\\.csv$ .*\\.txt$ .*__pycache__/ .*\\.pyc$" >nul
  if not errorlevel 1 (
    if "!FORBIDDEN_STAGED!"=="0" echo === Forbidden Staged Files ===
    set "FORBIDDEN_STAGED=1"
    echo %%f
  )
)
if "%FORBIDDEN_STAGED%"=="0" echo No forbidden staged files detected.
echo.

echo === Ignored Local Data / Result Files ===
git status --short --ignored data .env .venv __pycache__ 2>nul
echo.

echo === Safety Reminder ===
echo Do not commit or push:
echo   data\*.txt
echo   data\backtest_logs\
echo   downloaded historical CSV folders
echo   .env or any API keys/secrets
echo   .venv
echo   __pycache__ or *.pyc
echo.
echo Suggested manual review commands:
echo   git diff --stat
echo   git diff -- docs tools .gitignore
echo   git status --short --ignored
echo.
echo This script does not stage, commit, or push.
if "%FORBIDDEN_STAGED%"=="1" (
  echo ERROR: forbidden generated data/log/secret/cache files are staged.
  if "%PAUSE_ON_EXIT%"=="1" pause
  exit /b 1
)
if "%PAUSE_ON_EXIT%"=="1" pause
