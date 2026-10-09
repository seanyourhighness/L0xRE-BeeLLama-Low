param(
    [Parameter(Mandatory=$true)][string]$Model,
    [Parameter(Mandatory=$true)][string]$Draft,
    [string]$Mmproj,
    [int]$Port=30172,
    [string]$BindAddress='127.0.0.1',
    [int]$Context=81920,
    [switch]$DryRun,
    [switch]$QualificationProbe,
    [Parameter(ValueFromRemainingArguments=$true)][string[]]$ServerArgs
)
$ErrorActionPreference='Stop'
$packageRoot=$PSScriptRoot
$profile=Get-Content -Raw (Join-Path $packageRoot 'profile.json') | ConvertFrom-Json
$manifestPath=Join-Path $packageRoot 'MANIFEST.json'
if (!(Test-Path $manifestPath)) { throw 'Windows payload manifest is not prepared' }
$manifest=Get-Content -Raw $manifestPath | ConvertFrom-Json
if (!$manifest.hardware_qualified -and !$QualificationProbe) { throw 'Windows SM120 hardware qualification is pending' }
function Get-L0xreSha256([string]$literalPath) {
    $stream=[System.IO.File]::OpenRead($literalPath)
    $hasher=[System.Security.Cryptography.SHA256]::Create()
    try { return ([BitConverter]::ToString($hasher.ComputeHash($stream))).Replace('-','').ToLowerInvariant() }
    finally { $hasher.Dispose(); $stream.Dispose() }
}
foreach ($entry in $manifest.files.PSObject.Properties) {
    $file=Join-Path $packageRoot $entry.Name
    if (!(Test-Path $file) -or (Get-L0xreSha256 $file) -ne $entry.Value) {
        throw ('Payload integrity mismatch: '+$entry.Name)
    }
}
$modelFile=(Resolve-Path $Model).Path
$draftFile=(Resolve-Path $Draft).Path
if ((Get-L0xreSha256 $modelFile) -ne $profile.model_sha256.target) { throw 'Target model differs from reference certificate' }
if ((Get-L0xreSha256 $draftFile) -ne $profile.model_sha256.draft) { throw 'Draft model differs from reference certificate' }
if ($Mmproj) {
    $visionFile=(Resolve-Path $Mmproj).Path
    if ((Get-L0xreSha256 $visionFile) -ne $profile.model_sha256.vision) { throw 'Projector differs from reference certificate' }
}
$gpuRecords=@(& nvidia-smi.exe --query-gpu=uuid,compute_cap --format=csv,noheader)
$gpuQueryExitCode=$LASTEXITCODE
if ($gpuQueryExitCode -ne 0 -or $gpuRecords.Count -eq 0) { throw 'Selected NVIDIA GPU is unavailable' }
$gpuRecord=$gpuRecords[0]
$gpuFields=$gpuRecord -split ','
if ($gpuFields[1].Trim() -ne '12.0') { throw 'This payload targets SM120' }
function Expand-PackageValue([string]$value) {
    $value.Replace('@PACKAGE_ROOT@',$packageRoot).Replace('@ARCH@','sm120').Replace('/','\')
}
foreach ($entry in @(Get-ChildItem Env:)) {
    if ($entry.Name -match '^(L0XRE_|ESCHA_|GGML_)') { Remove-Item ('Env:'+$entry.Name) }
}
$resolvedEnv=@{}
foreach ($entry in $profile.env.PSObject.Properties) {
    $value=Expand-PackageValue $entry.Value
    [Environment]::SetEnvironmentVariable($entry.Name,$value,'Process')
    $resolvedEnv[$entry.Name]=$value
}
$env:CUDA_VISIBLE_DEVICES=$gpuFields[0].Trim()
$env:GGML_BACKEND_PATH=Join-Path $packageRoot 'bin'
$env:PATH=(Join-Path $packageRoot 'bin')+';'+(Join-Path $packageRoot 'bridge')+';'+$env:PATH
$selfProc=[System.Diagnostics.Process]::GetCurrentProcess()
$selfProc.ProcessorAffinity=[IntPtr]255
$resolvedArgv=@($profile.argv | ForEach-Object { Expand-PackageValue $_ })
if ($Context -lt 2048 -or $Context -gt 81920) { throw 'Context must be within2048..81920' }
$resolvedArgv[[Array]::IndexOf($resolvedArgv,'-c')+1]=[string]$Context
$resolvedArgv[[Array]::IndexOf($resolvedArgv,'--port')+1]=[string]$Port
$resolvedArgv[[Array]::IndexOf($resolvedArgv,'--host')+1]=$BindAddress
$resolvedArgv+=@('-m',$modelFile,'-md',$draftFile)
if ($Mmproj) { $resolvedArgv+=@('--mmproj',$visionFile) }
if ($ServerArgs) { $resolvedArgv+=$ServerArgs }
if ($DryRun) {
    @{argv=$resolvedArgv;env=$resolvedEnv;hardware_qualified=$manifest.hardware_qualified;gpu_uuid=$env:CUDA_VISIBLE_DEVICES;cpu_affinity=255} | ConvertTo-Json -Depth 8
    exit 0
}
$serverExe=$resolvedArgv[0]
$nativeArgs=@($resolvedArgv | Select-Object -Skip 1)
& $serverExe @nativeArgs
exit $LASTEXITCODE
