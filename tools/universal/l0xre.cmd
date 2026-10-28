@echo off
"%~dp0runtime\python.exe" -E -s -X utf8 "%~dp0tools\universal\v01-windows-launch.py" %*
exit /b %errorlevel%
