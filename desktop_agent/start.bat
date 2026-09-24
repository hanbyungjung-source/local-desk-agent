@echo off
setlocal
title Local Desk
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1" -Detached %*
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
    echo.
    echo Local Desk could not start. See the message above.
    pause
)
exit /b %EXIT_CODE%