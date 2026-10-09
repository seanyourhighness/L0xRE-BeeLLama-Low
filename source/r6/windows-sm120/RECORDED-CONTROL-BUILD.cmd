@echo off
call "D:\VS\BuildTools2022\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b %errorlevel%
set "CUDA_PATH=C:\work\l0xre-sm120-cuda130-windows-20261008\cuda"
set "PATH=%CUDA_PATH%\bin;%PATH%"
cmake.exe -S "C:\work\l0xre-sm120-cuda130-windows-20261008\source" -B "C:\work\l0xre-sm120-cuda130-windows-20261008\build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_CUDA_FA=ON -DGGML_CUDA_KVARN=ON -DGGML_CUDA_FA_ALL_QUANTS=OFF -DGGML_CUDA_COMPRESSION_MODE=size -DGGML_CUDA_GRAPHS=ON -DGGML_CCACHE=OFF -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=ON -DBUILD_SHARED_LIBS=ON -DGGML_RPC=OFF -DGGML_CUDA_NCCL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_UI=OFF -DCMAKE_CUDA_ARCHITECTURES=120 -DLLAMA_BUILD_COMMIT=47708f863027c0f3162ad59b7a2c9d7f2018a758 -DLLAMA_BUILD_NUMBER=6 -DLLAMA_BUILD_DIRTY=1 "-DCUDAToolkit_ROOT=%CUDA_PATH%" "-DCMAKE_CUDA_COMPILER=%CUDA_PATH%\bin\nvcc.exe" "-DCMAKE_MAKE_PROGRAM=D:\VS\BuildTools2022\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
if errorlevel 1 exit /b %errorlevel%
cmake.exe --build "C:\work\l0xre-sm120-cuda130-windows-20261008\build" --parallel 6 --target llama-server llama-cli llama-bench
if errorlevel 1 exit /b %errorlevel%
cd /d "C:\work\l0xre-sm120-cuda130-windows-20261008\bridge"
"%CUDA_PATH%\bin\nvcc.exe" -O3 -std=c++17 -gencode=arch=compute_120,code=sm_120 -gencode=arch=compute_120a,code=sm_120a -shared -Xcompiler /MD -Xlinker /DEF:bridge-packed.def -Xlinker /NODEFAULTLIB:LIBCMT -I. -I"C:\work\l0xre-sm120-cuda130-windows-20261008\source\ggml\src\ggml-cuda" -I"C:\work\l0xre-sm120-cuda130-windows-20261008\source\ggml\src" -I"C:\work\l0xre-sm120-cuda130-windows-20261008\source\ggml\include" bridge-packed.cu -lcublas -lcuda -o bridge-packed.dll
if errorlevel 1 exit /b %errorlevel%
cd /d "C:\work\l0xre-sm120-cuda130-windows-20261008\companions"
"%CUDA_PATH%\bin\nvcc.exe" -O3 -std=c++17 -shared -Xcompiler /MD -gencode=arch=compute_120,code=sm_120 -gencode=arch=compute_120a,code=sm_120a -Xlinker /DEF:gdn512.def -Xlinker /NODEFAULTLIB:LIBCMT -I. gdn512.cu -lcuda -o gdn512.dll
if errorlevel 1 exit /b %errorlevel%
"%CUDA_PATH%\bin\nvcc.exe" -O3 -std=c++17 -shared -Xcompiler /MD -gencode=arch=compute_120,code=sm_120 -gencode=arch=compute_120a,code=sm_120a -Xlinker /DEF:gdn-chunk.def -Xlinker /NODEFAULTLIB:LIBCMT -I. gdn-chunk.cu -lcuda -o gdn-chunk.dll
exit /b %errorlevel%
