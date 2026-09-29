@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0l0xre.ps1" %*
exit /b %errorlevel%
