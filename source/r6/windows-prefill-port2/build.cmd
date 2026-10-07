@echo off
call "D:\VS\BuildTools2022\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b %errorlevel%
cd /d C:\work\l0xre-r6-gdn-port-20261007
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\nvcc.exe" -O3 -std=c++17 -shared -Xcompiler /MD -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -Xlinker /DEF:gdn-masked.def -Xlinker /NODEFAULTLIB:LIBCMT -I. bridge.cu -o escha_r6_gdn_masked.dll -lcuda
exit /b %errorlevel%
