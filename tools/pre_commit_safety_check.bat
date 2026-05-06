@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0.."

set "FORBIDDEN_STAGED=0"
echo Checking staged files before commit...
for /f "delims=" %%f in ('git diff --cached --name-only') do (
  echo %%f | findstr /R /I "^data/ ^\\.env$ ^\\.venv/ .*\\.csv$ .*\\.txt$ .*__pycache__/ .*\\.pyc$" >nul
  if not errorlevel 1 (
    if "!FORBIDDEN_STAGED!"=="0" echo Forbidden staged files:
    set "FORBIDDEN_STAGED=1"
    echo %%f
  )
)

if "%FORBIDDEN_STAGED%"=="1" (
  echo ERROR: unstage forbidden generated data/log/secret/cache files before committing.
  exit /b 1
)

echo OK: staged files do not include data logs, historical CSVs, .env, .venv, cache, or pyc files.
exit /b 0
