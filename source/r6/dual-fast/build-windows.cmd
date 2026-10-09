@echo off
call "D:\VS\BuildTools2022\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b %errorlevel%
set "CUDA_PATH=C:\work\l0xre-sm120-cuda130-windows-20261008\cuda"
set "PATH=%CUDA_PATH%\bin;%PATH%"
cd /d C:\work\l0xre-dual-fast-r6-20261009\bridge
"%CUDA_PATH%\bin\nvcc.exe" -O3 -std=c++17 -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_89,code=sm_89 -gencode=arch=compute_120,code=sm_120 -gencode=arch=compute_120a,code=sm_120a -shared -Xcompiler /MD -Xlinker /DEF:bridge-packed.def -Xlinker /NODEFAULTLIB:LIBCMT -I. -I"C:\work\l0xre-sm89-r6-20261009\source\ggml\src\ggml-cuda" -I"C:\work\l0xre-sm89-r6-20261009\source\ggml\src" -I"C:\work\l0xre-sm89-r6-20261009\source\ggml\include" bridge-packed.cu -lcublas -lcuda -o bridge-dual.dll
if errorlevel 1 exit /b %errorlevel%
cd /d C:\work\l0xre-dual-fast-r6-20261009\sm86
"%CUDA_PATH%\bin\nvcc.exe" -O3 -std=c++17 -shared -Xcompiler /MD -gencode=arch=compute_86,code=sm_86 -Xlinker /DEF:gdn-masked.def -Xlinker /NODEFAULTLIB:LIBCMT -I. gdn-masked.cu -lcuda -o gdn-dual.dll
if errorlevel 1 exit /b %errorlevel%
cd /d C:\work\l0xre-dual-fast-r6-20261009\sm89
"%CUDA_PATH%\bin\nvcc.exe" -O3 -std=c++17 -shared -Xcompiler /MD -gencode=arch=compute_89,code=sm_89 -Xlinker /DEF:gdn-masked.def -Xlinker /NODEFAULTLIB:LIBCMT -I. gdn-masked.cu -lcuda -o gdn-dual.dll
if errorlevel 1 exit /b %errorlevel%
cd /d C:\work\l0xre-dual-fast-r6-20261009\sm120
"%CUDA_PATH%\bin\nvcc.exe" -O3 -std=c++17 -shared -Xcompiler /MD -gencode=arch=compute_120,code=sm_120 -Xlinker /DEF:gdn-masked.def -Xlinker /NODEFAULTLIB:LIBCMT -I. gdn-masked.cu -lcuda -o gdn-dual.dll
if errorlevel 1 exit /b %errorlevel%
exit /b 0
