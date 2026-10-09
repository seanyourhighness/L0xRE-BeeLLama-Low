param([Parameter(ValueFromRemainingArguments=$true)][string[]]$Rest)
$ErrorActionPreference = "Stop"
$ROOT = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path

function Show-Help {
    Write-Host "L0xRE BeeLLama R6 Windows launcher"
    Write-Host '.\l0xre-r6.cmd serve -m models\L0xRE-27b-Low.gguf -md models\Qwen3.8-27B-DFlash2-Q4_K_M.gguf'
    Write-Host "Pins the R6 model identities, one 81,920-token slot, KVarN4/4, Q4 draft, N7 and medium reasoning."
    Write-Host "--qualification-probe permits an unqualified architecture payload for testing. --dry-run prints the sealed command."
    Write-Host "SM89's Q4 MMQ path is opt-in with --sm89-q4-mmq and remains unqualified."
    Write-Host "--profile r6-sm89-4070ti selects the measured Windows refresh configuration; quality certification is pending."
}

if (-not $Rest -or $Rest[0] -in @("--help", "-h", "help")) { Show-Help; exit 0 }

$DryRun = $false
$QualificationProbe = $false
$Sm89Q4Mmq = $false
$ProfileName = "r6"
$Model = $null
$Draft = $null
$Mmproj = $null
$ServerExtra = [System.Collections.Generic.List[string]]::new()
$i = 0
while ($i -lt $Rest.Count) {
    $arg = $Rest[$i]
    if ($arg -eq "serve") { $i++; continue }
    if ($arg -eq "--dry-run") { $DryRun = $true; $i++; continue }
    if ($arg -eq "--qualification-probe") { $QualificationProbe = $true; $i++; continue }
    if ($arg -eq "--sm89-q4-mmq") { $Sm89Q4Mmq = $true; $i++; continue }
    if ($arg -eq "--profile") {
        if (($i + 1) -ge $Rest.Count -or $Rest[$i + 1] -notin @("r6", "r6-sm89-4070ti")) { throw "Supported profiles: r6, r6-sm89-4070ti." }
        $ProfileName = $Rest[$i + 1]
        $i += 2; continue
    }
    if ($arg -in @("-m", "--model", "-md", "--draft-model", "--spec-draft-model", "--mmproj")) {
        if (($i + 1) -ge $Rest.Count) { throw "Missing value after $arg." }
        $value = $Rest[$i + 1]
        if ($arg -in @("-m", "--model")) { $Model = $value }
        elseif ($arg -in @("-md", "--draft-model", "--spec-draft-model")) { $Draft = $value }
        else { $Mmproj = $value }
        $i += 2; continue
    }
    if ($arg -in @("-np", "-t", "-tb", "-ngl", "-fa", "-c", "-b", "-ub", "-ctk", "-ctv", "--kv-tail-tokens", "--spec-type", "--spec-draft-n-max", "--spec-draft-ngl", "--spec-draft-ubatch-size", "--spec-draft-type-k", "--spec-draft-type-v", "--spec-draft-threads", "--spec-draft-threads-batch", "--reasoning-effort", "--reasoning-budget", "--temp", "--top-p", "--top-k", "--min-p")) {
        throw "R6 profile option '$arg' is sealed; launch the package's R6 settings without overriding it."
    }
    $ServerExtra.Add($arg)
    $i++
}
if (-not $Model -or -not $Draft) { throw "Supply the pinned target with -m and Q4 draft with -md." }

$profileFile = if ($ProfileName -eq "r6-sm89-4070ti") { "r6-windows-sm89-4070ti-profile.json" } else { "r6-windows-profile.json" }
$Profile = Get-Content -Raw (Join-Path $PSScriptRoot $profileFile) | ConvertFrom-Json
$query = @("--query-gpu=uuid,compute_cap", "--format=csv,noheader")
if (Test-Path Env:CUDA_VISIBLE_DEVICES) {
    $selected = ($env:CUDA_VISIBLE_DEVICES -split ',')[0]
    if (-not $selected -or $selected -eq "-1") { throw "CUDA_VISIBLE_DEVICES exposes no GPU." }
    $query += @("-i", $selected)
}
$gpuRows = @(& nvidia-smi @query 2>$null)
$gpuExitCode = $LASTEXITCODE
$gpu = $gpuRows | Select-Object -First 1
if ($gpuExitCode -ne 0 -or -not $gpu) { throw "GPU detection failed. Check the NVIDIA driver and nvidia-smi." }
$fields = @($gpu -split ',' | ForEach-Object { $_.Trim() })
if ($fields.Count -lt 2) { throw "Could not parse nvidia-smi GPU identity: $gpu" }
$uuid = $fields[0]
$cc = $fields[1] -replace '\.', ''
$ARCH = "sm$cc"
if ($ARCH -notin @("sm86", "sm89", "sm120")) { throw "Unsupported compute capability: $($fields[1])." }
if ($ProfileName -eq "r6-sm89-4070ti" -and $ARCH -ne "sm89") { throw "The r6-sm89-4070ti profile requires SM89." }
if ($Sm89Q4Mmq -and $ARCH -ne "sm89") { throw "--sm89-q4-mmq is only valid on SM89." }

$archRoot = Join-Path $ROOT "architectures\$ARCH"
$manifestPath = Join-Path $archRoot "R6-MANIFEST.json"
if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) { throw "$ARCH R6 payload is not installed." }
$Manifest = Get-Content -Raw $manifestPath | ConvertFrom-Json
if (-not $Manifest.built) { throw "$ARCH R6 payload has not been built." }
if (-not $Manifest.hardware_qualified -and -not $QualificationProbe) { throw "$ARCH R6 hardware qualification is pending." }
foreach ($entry in $Manifest.runtime_sha256.PSObject.Properties) {
    $file = Join-Path $ROOT $entry.Name
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) { throw "R6 payload component missing: $($entry.Name)" }
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $file).Hash.ToLowerInvariant()
    if ($actual -ne $entry.Value) { throw "R6 payload integrity check failed: $($entry.Name)" }
}
if ($ProfileName -eq "r6-sm89-4070ti") {
    foreach ($entry in $Profile.required_runtime_sha256.PSObject.Properties) {
        $file = Join-Path $ROOT $entry.Name
        if (-not (Test-Path -LiteralPath $file -PathType Leaf) -or (Get-FileHash -Algorithm SHA256 -LiteralPath $file).Hash.ToLowerInvariant() -ne $entry.Value) {
            throw "The measured profile requires the pinned Windows SM89 refresh payload: $($entry.Name)"
        }
    }
}

$Model = (Resolve-Path -LiteralPath $Model).Path
$actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $Model).Hash.ToLowerInvariant()
if ($actual -ne $Profile.model_sha256.target) { throw "target differs from the R6 qualified model." }
$Draft = (Resolve-Path -LiteralPath $Draft).Path
$actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $Draft).Hash.ToLowerInvariant()
if ($actual -ne $Profile.model_sha256.draft) { throw "draft differs from the R6 qualified model." }
if ($Mmproj) {
    $Mmproj = (Resolve-Path -LiteralPath $Mmproj).Path
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $Mmproj).Hash.ToLowerInvariant()
    if ($actual -ne $Profile.model_sha256.vision) { throw "Vision projector differs from the R6 qualified file." }
}

foreach ($item in @(Get-ChildItem Env: | Where-Object { $_.Name -match '^(L0XRE_|ESCHA_|GGML_)' })) {
    Remove-Item "Env:$($item.Name)" -ErrorAction SilentlyContinue
}
foreach ($entry in $Profile.env.PSObject.Properties) {
    $value = [string]$entry.Value
    $value = $value.Replace('@PACKAGE_ROOT@', $ROOT).Replace('@ARCH_ROOT@', (Join-Path $ROOT "bridge\$ARCH")).Replace('@ARCH@', $ARCH)
    [Environment]::SetEnvironmentVariable($entry.Name, $value, "Process")
}
if ($ARCH -eq "sm86") { $env:L0XRE_SM86_Q4_MMQ = "1" }
if ($ARCH -eq "sm86") { $env:ESCHA_E3_HEAD_RT_BLOCK128 = "1" }
if ($Sm89Q4Mmq) { $env:L0XRE_SM89_Q4_MMQ = "1" }
$env:PATH = "$(Join-Path $ROOT 'bin');$(Join-Path $ROOT 'bridge');$env:PATH"
$env:CUDA_VISIBLE_DEVICES = $uuid
$affinity = $null
if ($ProfileName -eq "r6-sm89-4070ti") {
    $logicalProcessors = [Environment]::ProcessorCount
    if ($logicalProcessors -gt 63) { throw "The measured profile supports at most 63 logical processors in one processor group." }
    $affinity = if ($logicalProcessors -eq 63) { [long]::MaxValue } else { ([long]1 -shl $logicalProcessors) - 1 }
}

$argv = @($Profile.server_args) + @("-m", $Model, "-md", $Draft)
if ($Mmproj) { $argv += @("--mmproj", $Mmproj) }
$argv += $ServerExtra.ToArray()
$server = Join-Path $ROOT "bin\llama-server.exe"
if (-not (Test-Path -LiteralPath $server -PathType Leaf)) { throw "R6 server is missing: $server" }
if ($DryRun) {
    $visibleEnv = @{}
    foreach ($entry in $Profile.env.PSObject.Properties) { $visibleEnv[$entry.Name] = [Environment]::GetEnvironmentVariable($entry.Name, "Process") }
    $visibleEnv["L0XRE_ARCH"] = $ARCH
    if ($Sm89Q4Mmq) { $visibleEnv["L0XRE_SM89_Q4_MMQ"] = "1" }
    $visibleEnv["CUDA_VISIBLE_DEVICES"] = $uuid
    [PSCustomObject]@{ arch=$ARCH; uuid=$uuid; profile=$ProfileName; affinity=$affinity; hardware_qualified=[bool]$Manifest.hardware_qualified; argv=@($server) + $argv; env=$visibleEnv } | ConvertTo-Json -Depth 5
    exit 0
}
$launcherProcess = Get-Process -Id $PID
$originalAffinity = $launcherProcess.ProcessorAffinity
try {
    if ($null -ne $affinity) { $launcherProcess.ProcessorAffinity = [IntPtr]$affinity }
    & $server @argv
    $serverExitCode = $LASTEXITCODE
} finally {
    if ($null -ne $affinity) { $launcherProcess.ProcessorAffinity = $originalAffinity }
}
exit $serverExitCode
