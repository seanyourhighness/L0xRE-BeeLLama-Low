$ErrorActionPreference='Stop'
$build='C:\work\l0xre-sm89-r6-20261009'
$pkg=Join-Path $build 'consumer'
foreach ($e in @(Get-ChildItem Env:)) { if ($e.Name -match '^(ESCHA_|L0XRE_|GGML_|LLAMA_ARG_)') { Remove-Item ('Env:'+$e.Name) } }
$env:CUDA_VISIBLE_DEVICES='-1'
$env:L0XRE_INT8_PREFILL='0'
$env:PATH=(Join-Path $pkg 'bin')+';'+$env:PATH
$env:GGML_BACKEND_PATH=Join-Path $pkg 'bin'
$versions=@{}
foreach ($name in @('llama-server.exe','llama-cli.exe','llama-bench.exe')) {
    $argument=if ($name -eq 'llama-bench.exe') { '--help' } else { '--version' }
    $log=Join-Path $build ($name+'.'+$argument.Substring(2)+'.txt')
    $process=Start-Process -FilePath (Join-Path $pkg ('bin\'+$name)) -ArgumentList $argument -NoNewWindow -PassThru -Wait -RedirectStandardOutput $log -RedirectStandardError ($log+'.stderr')
    if ($process.ExitCode -ne 0) { throw "$name version check failed: $($process.ExitCode)" }
    $versions[$name]=@{argument=$argument;exit_code=$process.ExitCode;stdout=([IO.File]::ReadAllText($log));stderr=([IO.File]::ReadAllText($log+'.stderr'))}
}
# Load the exact packaged bridges against the bundled CUDA runtime, with GPU work disabled.
. 'C:\work\l0xre-dual-fast-r6-20261009\check-abi.ps1' | Out-Null
$env:PATH=(Join-Path $pkg 'bin')+';'+$env:PATH
[DualAbi]::Check((Join-Path $pkg 'bridge\sm89\escha_r6_packed_bridge.dll'),(Join-Path $pkg 'bridge\sm89\escha_r6_gdn_masked.dll'))
$chunk=[DualAbi]::LoadLibrary((Join-Path $pkg 'bridge\escha_gdn_chunk_bridge.dll'))
if ($chunk -eq [IntPtr]::Zero) { throw ('Chunk DLL load failed: '+[Runtime.InteropServices.Marshal]::GetLastWin32Error()) }
[DualAbi]::FreeLibrary($chunk) | Out-Null
$dump='C:\work\l0xre-sm120-cuda130-windows-20261008\cuda\bin\cuobjdump.exe'
$arches=@{}
foreach ($relative in @('bin\ggml-cuda.dll','bridge\sm89\escha_r6_packed_bridge.dll','bridge\sm89\escha_r6_gdn_masked.dll')) {
    $listed=@(& $dump --list-elf (Join-Path $pkg $relative))
    if ($LASTEXITCODE -ne 0) { throw 'Device code inspection failed' }
    $matches=@([regex]::Matches(($listed -join "`n"),'sm_[0-9]+[a-z]?') | ForEach-Object { $_.Value } | Sort-Object -Unique)
    if ($matches.Count -ne 1 -or $matches[0] -ne 'sm_89') { throw "Wrong device code: $relative / $matches" }
    $arches[$relative]=$matches
}
@{status='pass';hardware_qualified=$false;gpu_inference_run=$false;cuda_visible_devices='-1';native_cli_smoke=$versions;bridge_abi=1;chunk_dll_load='pass';device_architectures=$arches} | ConvertTo-Json -Depth 8
