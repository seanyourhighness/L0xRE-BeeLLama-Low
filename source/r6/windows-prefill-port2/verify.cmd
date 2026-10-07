@echo off
call "D:\VS\BuildTools2022\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b %errorlevel%
cd /d C:\work\l0xre-r6-gdn-port-20261007
cl /nologo /O2 /EHsc /MD abi-check.cpp /Fe:gdn-abi-check.exe
if errorlevel 1 exit /b %errorlevel%
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin;%PATH%"
dumpbin /exports escha_r6_gdn_masked.dll > gdn-exports.txt
if errorlevel 1 exit /b %errorlevel%
dumpbin /dependents escha_r6_gdn_masked.dll > gdn-imports.txt
if errorlevel 1 exit /b %errorlevel%
cuobjdump --list-elf escha_r6_gdn_masked.dll > gdn-device-elf.txt
if errorlevel 1 exit /b %errorlevel%
gdn-abi-check.exe > gdn-abi-check.json
exit /b %errorlevel%
