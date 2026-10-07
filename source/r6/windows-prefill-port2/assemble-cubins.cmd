@echo off
cd /d C:\work\l0xre-r6-gdn-port-20261007
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\ptxas.exe" -O3 --gpu-name=sm_86 gdn-cubins-sm86\cumsum.ptx -o gdn-cubins-sm86\cumsum.cubin
if errorlevel 1 exit /b %errorlevel%
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\ptxas.exe" -O3 --gpu-name=sm_86 gdn-cubins-sm86\kkt.ptx -o gdn-cubins-sm86\kkt.cubin
if errorlevel 1 exit /b %errorlevel%
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\ptxas.exe" -O3 --gpu-name=sm_86 gdn-cubins-sm86\merge.ptx -o gdn-cubins-sm86\merge.cubin
if errorlevel 1 exit /b %errorlevel%
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\ptxas.exe" -O3 --gpu-name=sm_86 gdn-cubins-sm86\output.ptx -o gdn-cubins-sm86\output.cubin
if errorlevel 1 exit /b %errorlevel%
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\ptxas.exe" -O3 --gpu-name=sm_86 gdn-cubins-sm86\recompute.ptx -o gdn-cubins-sm86\recompute.cubin
if errorlevel 1 exit /b %errorlevel%
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\ptxas.exe" -O3 --gpu-name=sm_86 gdn-cubins-sm86\solve.ptx -o gdn-cubins-sm86\solve.cubin
if errorlevel 1 exit /b %errorlevel%
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\ptxas.exe" -O3 --gpu-name=sm_86 gdn-cubins-sm86\state.ptx -o gdn-cubins-sm86\state.cubin
if errorlevel 1 exit /b %errorlevel%
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\ptxas.exe" -O3 --gpu-name=sm_89 gdn-cubins-sm89\cumsum.ptx -o gdn-cubins-sm89\cumsum.cubin
if errorlevel 1 exit /b %errorlevel%
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\ptxas.exe" -O3 --gpu-name=sm_89 gdn-cubins-sm89\kkt.ptx -o gdn-cubins-sm89\kkt.cubin
if errorlevel 1 exit /b %errorlevel%
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\ptxas.exe" -O3 --gpu-name=sm_89 gdn-cubins-sm89\merge.ptx -o gdn-cubins-sm89\merge.cubin
if errorlevel 1 exit /b %errorlevel%
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\ptxas.exe" -O3 --gpu-name=sm_89 gdn-cubins-sm89\output.ptx -o gdn-cubins-sm89\output.cubin
if errorlevel 1 exit /b %errorlevel%
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\ptxas.exe" -O3 --gpu-name=sm_89 gdn-cubins-sm89\recompute.ptx -o gdn-cubins-sm89\recompute.cubin
if errorlevel 1 exit /b %errorlevel%
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\ptxas.exe" -O3 --gpu-name=sm_89 gdn-cubins-sm89\solve.ptx -o gdn-cubins-sm89\solve.cubin
if errorlevel 1 exit /b %errorlevel%
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\ptxas.exe" -O3 --gpu-name=sm_89 gdn-cubins-sm89\state.ptx -o gdn-cubins-sm89\state.cubin
if errorlevel 1 exit /b %errorlevel%
exit /b 0
