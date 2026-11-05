@echo off
rem L0xRE installer for Windows. Batch files are not subject to PowerShell execution policy,
rem so this runs install.ps1 without "-ExecutionPolicy Bypass" being typed. Same arguments as
rem install.ps1, e.g.  install.cmd -Vision   install.cmd -Update   install.cmd -Rollback
setlocal
set "L0XRE_PS1=%~dp0install.ps1"
set "L0XRE_TMP="
if exist "%L0XRE_PS1%" goto run
set "L0XRE_TMP=%TEMP%\l0xre-install-%RANDOM%%RANDOM%"
mkdir "%L0XRE_TMP%" || exit /b 1
set "L0XRE_PS1=%L0XRE_TMP%\install.ps1"
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -UseBasicParsing https://raw.githubusercontent.com/seanyourhighness/L0xRE-BeeLLama-Low/main/install/install.ps1 -OutFile $env:L0XRE_PS1"
if errorlevel 1 (rmdir /s /q "%L0XRE_TMP%" & exit /b 1)
:run
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%L0XRE_PS1%" %*
set "L0XRE_RC=%ERRORLEVEL%"
if defined L0XRE_TMP rmdir /s /q "%L0XRE_TMP%"
exit /b %L0XRE_RC%
