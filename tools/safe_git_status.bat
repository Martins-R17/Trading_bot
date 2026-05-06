@echo off
setlocal
cd /d "%~dp0.."

echo === Git Status ===
git status --short
echo.

echo === Changed Files ===
git diff --name-only
echo.

echo === Staged Files ===
git diff --cached --name-only
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
pause
