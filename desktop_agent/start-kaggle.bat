@echo off
setlocal
call "%~dp0start.bat" -Kaggle %*
exit /b %ERRORLEVEL%