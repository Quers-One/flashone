@echo off
rem Flash UI -- start the training dashboard of THIS repository and open it.
rem Double-click this file, or run:  "Flash UI.cmd"
rem
rem Requirements: PowerShell 7 (pwsh) and Python 3.13 with torch installed.
rem   - powershell 5.1 (built into Windows) will NOT work; install pwsh 7.
rem   - python must be on PATH, because serve_dashboard.py is started with it.
rem
rem This file is pure ASCII with CRLF on purpose: cmd.exe reads .cmd files in the
rem OEM codepage and needs CRLF, so non-ASCII here would break parsing.
chcp 65001 >nul
setlocal
set "HERE=%~dp0"
set "PS1=%HERE%dashboard.ps1"

if not exist "%PS1%" (
  echo [Flash UI] dashboard.ps1 was not found next to this launcher.
  echo Looked in: %PS1%
  pause
  exit /b 1
)

where pwsh >nul 2>nul
if errorlevel 1 (
  echo [Flash UI] PowerShell 7 ^(pwsh^) is not in PATH.
  echo Install it from https://aka.ms/powershell and double-click again.
  pause
  exit /b 1
)

where python >nul 2>nul
if errorlevel 1 (
  echo [Flash UI] python is not in PATH. Install Python 3.13 and run: pip install torch
  pause
  exit /b 1
)

pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%PS1%" %*
set "RC=%ERRORLEVEL%"

if not "%RC%"=="0" (
  echo.
  echo [Flash UI] failed with exit code %RC%
  echo Try:  pwsh -NoProfile -File "%PS1%"
  pause
)
exit /b %RC%
