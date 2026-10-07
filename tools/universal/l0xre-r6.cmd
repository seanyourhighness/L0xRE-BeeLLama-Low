@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\universal\r6-launch.ps1" %*
exit /b %errorlevel%
