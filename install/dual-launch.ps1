param([switch]$DryRun)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2.0
$root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$receipt = Get-Content -Raw -LiteralPath (Join-Path $root 'INSTALLATION.json') | ConvertFrom-Json
$plan = $receipt.selection
if ($plan.mode -ne 'dual-fast-experimental') { throw 'This installation is not configured for dual fast layer split.' }
foreach ($entry in $plan.launcher_files.PSObject.Properties) {
    if ((Get-FileHash -LiteralPath (Join-Path $root $entry.Name) -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.Value) { throw 'Dual launcher integrity mismatch.' }
}
$profilePath = Join-Path $receipt.runtime 'profile.json'
if (-not (Test-Path -LiteralPath $profilePath)) { $profilePath = Join-Path $receipt.runtime 'tools\universal\r6-windows-profile.json' }
$profile = Get-Content -Raw -LiteralPath $profilePath | ConvertFrom-Json
foreach ($package in @($receipt.runtime,$plan.dual_addon_root)) {
foreach ($line in Get-Content -LiteralPath (Join-Path $package 'SHA256SUMS')) {
    if ($line -notmatch '^([a-fA-F0-9]{64})  (.+)$') { throw 'Invalid runtime checksum catalog.' }
    $expected = $Matches[1].ToLowerInvariant(); $relative = $Matches[2]
    if ([IO.Path]::IsPathRooted($relative) -or $relative.Contains(':') -or ($relative -split '[/\\]') -contains '..') { throw 'Invalid runtime checksum path.' }
    if ((Get-FileHash -LiteralPath (Join-Path $package $relative) -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected) { throw ('Runtime integrity mismatch: ' + $relative) }
}
}
foreach ($gpu in $plan.gpus) {
    $raw = @(& nvidia-smi "--id=$($gpu.uuid)" '--query-gpu=uuid,name,compute_cap,memory.total' '--format=csv,noheader,nounits')
    if ($LASTEXITCODE -ne 0 -or $raw.Count -ne 1) { throw 'Selected GPU unavailable.' }
    $row = $raw[0] | ConvertFrom-Csv -Header uuid,name,cc,memory
    if ($row.uuid.Trim() -ne $gpu.uuid -or $row.name.Trim() -ne $gpu.name -or ('sm' + $row.cc.Trim().Replace('.','')) -ne $gpu.arch -or [int]$row.memory.Trim() -lt 12000) { throw 'Selected matched GPU is unavailable or changed.' }
}
$models = @{ target = 'L0xRE-27b-Low.gguf'; draft = 'Qwen3.8-27B-DFlash2-Q4_K_M.gguf' }
if ($receipt.vision) { $models.vision = 'mmproj-Qwen3.8-27B-Q8_0.gguf' }
foreach ($key in $models.Keys) {
    if ((Get-FileHash -LiteralPath (Join-Path $receipt.models_dir $models[$key]) -Algorithm SHA256).Hash.ToLowerInvariant() -ne $profile.model_sha256.$key) { throw ('Pinned model mismatch: ' + $key) }
}
function Expand-DualValue([string]$Value) {
    $Value.Replace('@PACKAGE_ROOT@',$receipt.runtime).Replace('@ARCH_ROOT@',(Join-Path $receipt.runtime ('bridge\'+$plan.gpu.arch))).Replace('@ARCH@',$plan.gpu.arch).Replace('/','\')
}
foreach ($entry in @(Get-ChildItem Env:)) {
    if ($entry.Name -match '^(L0XRE_|ESCHA_|GGML_|LLAMA_ARG_)' -or $entry.Name -eq 'CUDA_LAUNCH_BLOCKING') { Remove-Item ('Env:' + $entry.Name) }
}
$resolvedEnv=@{}
foreach ($entry in $profile.env.PSObject.Properties) {
    $value=Expand-DualValue $entry.Value
    [Environment]::SetEnvironmentVariable($entry.Name,$value,'Process')
    $resolvedEnv[$entry.Name]=$value
}
$addon=Join-Path $plan.dual_addon_root $plan.gpu.arch
$env:CUDA_VISIBLE_DEVICES = ($plan.gpus | ForEach-Object { $_.uuid }) -join ','
$env:GGML_BACKEND_PATH = Join-Path $receipt.runtime 'bin'
$env:PATH = $env:GGML_BACKEND_PATH + ';' + $addon + ';' + $env:PATH
$env:ESCHA_OFFICIAL_BRIDGE_LIBRARY=Join-Path $addon 'bridge-dual.dll'
$env:L0XRE_GDN_MASKED_LIBRARY=Join-Path $addon 'gdn-dual.dll'
$env:L0XRE_DUAL_FAST_EXPERIMENT='1'
if ($plan.gpu.arch -eq 'sm86') { $env:L0XRE_SM86_Q4_MMQ='1'; $env:ESCHA_E3_HEAD_RT_BLOCK128='1' }
if ($profile.PSObject.Properties['argv']) { $argv=@($profile.argv | Select-Object -Skip 1 | ForEach-Object { Expand-DualValue $_ }) }
else { $argv=@($profile.server_args) }
$argv[[Array]::IndexOf($argv,'--port')+1] = [string]$receipt.port
$argv += @('--split-mode','layer','--tensor-split','1,1','--device','CUDA0,CUDA1','--spec-draft-device','CUDA0')
$argv += @('-m',(Join-Path $receipt.models_dir $models.target),'-md',(Join-Path $receipt.models_dir $models.draft))
if ($receipt.vision) { $argv += @('--mmproj',(Join-Path $receipt.models_dir $models.vision)) }
$server = Join-Path $receipt.runtime 'bin\llama-server.exe'
if ($DryRun) {
    foreach ($name in @('CUDA_VISIBLE_DEVICES','GGML_BACKEND_PATH','ESCHA_OFFICIAL_BRIDGE_LIBRARY','L0XRE_GDN_MASKED_LIBRARY','L0XRE_DUAL_FAST_EXPERIMENT')) { $resolvedEnv[$name]=[Environment]::GetEnvironmentVariable($name,'Process') }
    @{ mode='dual-fast-experimental'; qualification_status='experimental / untested / uncertified'; hardware_qualified=$false; argv=@($server)+$argv; external_fast_bridges=$true; env=$resolvedEnv } | ConvertTo-Json -Depth 8
    exit 0
}
Write-Host 'EXPERIMENTAL DUAL GPU / UNTESTED / UNCERTIFIED: quality and performance parity are unmeasured.'
[Diagnostics.Process]::GetCurrentProcess().ProcessorAffinity = [IntPtr]255
& $server @argv
exit $LASTEXITCODE
