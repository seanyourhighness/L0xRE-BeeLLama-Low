<#
.SYNOPSIS
    Build bridge-b74-k3-vector.dll for the L0xRE universal Windows runtime.

.DESCRIPTION
    Compiles official_bridge_b74_k3_vector.cu with MSVC + CUDA into an isolated
    output directory and verifies the artifact before returning success.

    The CUDA source is byte-identical to the reviewed Linux source
    (sha256 40b162dd48d41ca585968043a492d632ec0395be2029f3535c797474b3ce59c8).
    The per-platform artifacts are this script and bridge-b74-k3-vector.def.

    The B74 source adds a device kernel (round_f32_via_f16) next to the existing
    convert_f32_to_f16, so the DLL must carry device code for every architecture
    that can load it. The default is 86;89;120, matching the universal package.

    Verification performed before the script reports success:
      - every requested architecture is actually supported by the toolkit,
        checked against 'nvcc --list-gpu-code' rather than a version guess
        (CUDA 12.8 is the first toolkit with sm_120)
      - the DLL exports all 10 bridge ABI symbols
      - 'cuobjdump -lelf' lists every requested architecture
      - the import table and any compiler warnings are printed for review

    CRT note: the generated host object carries /DEFAULTLIB:MSVCRT (dynamic CRT,
    consistent with -Xcompiler /MD), but a CUDA static library still requests
    LIBCMT, which raised LNK4098. The DLL's real imports are the dynamic CRT set,
    so the link passes /NODEFAULTLIB:LIBCMT and the result is verified by the
    reported import list. ('--cudart shared' is deprecated in CUDA 13.3 and was
    measured to change neither the imports nor the warning, so it is not used.)

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File build.ps1

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File build.ps1 -CudaArch 86
#>
[CmdletBinding()]
param(
    [ValidateSet('86', '89', '120')]
    [string[]] $CudaArch = @('86', '89', '120'),
    [string]   $OutDir   = (Join-Path $PSScriptRoot 'out'),
    [string]   $CudaRoot,
    [string]   $VcVarsAll,
    [switch]   $KeepBat
)

$ErrorActionPreference = 'Stop'

$src = $PSScriptRoot
$cu  = Join-Path $src 'official_bridge_b74_k3_vector.cu'
$def = Join-Path $src 'bridge-b74-k3-vector.def'
$hdr = Join-Path $src 'escha_official_bridge_v1.h'
$dll = Join-Path $OutDir 'bridge-b74-k3-vector.dll'

foreach ($f in @($cu, $def, $hdr)) {
    if (-not (Test-Path $f)) { throw "missing required source file: $f" }
}

# --- CUDA toolkit ----------------------------------------------------------
if (-not $CudaRoot) {
    $candidates = @()
    $root = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA'
    if (Test-Path $root) {
        $candidates += (Get-ChildItem $root -Directory -ErrorAction SilentlyContinue |
                        Where-Object { $_.Name -match '^v(\d+)\.(\d+)$' } |
                        Sort-Object { [version]($_.Name.TrimStart('v')) } -Descending |
                        ForEach-Object { $_.FullName })
    }
    # CUDA_PATH is only a fallback: it often points at an older toolkit that
    # cannot target the newer architectures.
    if ($env:CUDA_PATH) { $candidates += $env:CUDA_PATH }
    foreach ($c in $candidates) {
        if ($c -and (Test-Path (Join-Path $c 'bin\nvcc.exe'))) { $CudaRoot = $c; break }
    }
}
if (-not $CudaRoot -or -not (Test-Path (Join-Path $CudaRoot 'bin\nvcc.exe'))) {
    throw 'CUDA toolkit not found. Pass -CudaRoot with the toolkit directory.'
}
$nvcc = Join-Path $CudaRoot 'bin\nvcc.exe'

$nvccText = (& $nvcc --version 2>&1 | Out-String)
$cudaVer  = 'unknown'
if ($nvccText -match 'release\s+(\d+)\.(\d+)') { $cudaVer = "$($Matches[1]).$($Matches[2])" }

# Capability check: ask the toolkit what it can generate instead of assuming a
# version threshold. CUDA 12.8 introduced sm_120; 12.9 is not required.
$gpuCodes = @(& $nvcc --list-gpu-code 2>&1 |
              ForEach-Object { $_.Trim() } |
              Where-Object { $_ -match '^sm_\d+$' })
if ($gpuCodes.Count -eq 0) {
    throw 'nvcc --list-gpu-code returned no target codes; cannot verify architecture support.'
}
$unsupported = @($CudaArch | Where-Object { $gpuCodes -notcontains "sm_$_" })
if ($unsupported.Count -gt 0) {
    throw ('CUDA {0} at {1} cannot target sm_{2}. Supported codes: {3}' -f $cudaVer, $CudaRoot, ($unsupported -join ', sm_'), ($gpuCodes -join ' '))
}

# --- MSVC ------------------------------------------------------------------
if (-not $VcVarsAll) {
    $pf86 = [Environment]::GetEnvironmentVariable('ProgramFiles(x86)')
    $vswhere = $null
    if ($pf86) { $vswhere = Join-Path $pf86 'Microsoft Visual Studio\Installer\vswhere.exe' }
    if ($vswhere -and (Test-Path $vswhere)) {
        $inst = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
        if ($inst) {
            $cand = Join-Path $inst 'VC\Auxiliary\Build\vcvarsall.bat'
            if (Test-Path $cand) { $VcVarsAll = $cand }
        }
    }
    if (-not $VcVarsAll) {
        $known = @(
            'D:\VS\BuildTools2022\VC\Auxiliary\Build\vcvarsall.bat',
            'C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat',
            'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat'
        )
        foreach ($cand in $known) { if (Test-Path $cand) { $VcVarsAll = $cand; break } }
    }
}
if (-not $VcVarsAll -or -not (Test-Path $VcVarsAll)) {
    throw 'vcvarsall.bat not found. Pass -VcVarsAll with the full path to vcvarsall.bat.'
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$gencode  = ($CudaArch | ForEach-Object { '-gencode arch=compute_{0},code=sm_{0}' -f $_ }) -join ' '
$archList = (($CudaArch | ForEach-Object { "sm_$_" }) -join ' ')

Write-Host "source : $cu"
Write-Host "header : $hdr"
Write-Host "def    : $def"
Write-Host "cuda   : $CudaRoot (nvcc $cudaVer)"
Write-Host "msvc   : $VcVarsAll"
Write-Host "arch   : $archList  (verified supported by nvcc --list-gpu-code)"
Write-Host "out    : $dll"

# --- compile ---------------------------------------------------------------
$bat = Join-Path $OutDir 'build-bridge.bat'
$bl = @(
    '@echo off',
    ('call "{0}" x64 >nul || exit /b 1' -f $VcVarsAll),
    ('cd /d "{0}"' -f $OutDir),
    ('"{0}" -O2 -shared -Xcompiler /MD {1} -Xlinker /DEF:"{2}" -Xlinker /NODEFAULTLIB:LIBCMT -I"{3}\include" -I"{4}" "{5}" -o "{6}" -lcuda' -f $nvcc, $gencode, $def, $CudaRoot, $src, $cu, $dll),
    'exit /b %ERRORLEVEL%'
)
Set-Content -Path $bat -Value $bl -Encoding ASCII

$prevEap = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
$buildOut = (& cmd.exe /c $bat 2>&1 | Out-String)
$buildRc  = $LASTEXITCODE
$ErrorActionPreference = $prevEap
Write-Host $buildOut

if ($buildRc -ne 0) { throw "bridge build failed (rc=$buildRc)" }
if (-not (Test-Path $dll)) { throw "build reported success but $dll is missing" }

$buildWarnings = @($buildOut -split "\r?\n" | Where-Object { $_ -match '(?i)warning' })
if ($buildWarnings.Count -eq 0) {
    Write-Host 'compile warnings: none'
} else {
    Write-Host ('compile warnings: {0}' -f $buildWarnings.Count)
    $buildWarnings | ForEach-Object { Write-Host ('  {0}' -f $_.Trim()) }
}

# --- verify ----------------------------------------------------------------
$vbat = Join-Path $OutDir 'verify-bridge.bat'
$vl = @(
    '@echo off',
    ('call "{0}" x64 >nul || exit /b 1' -f $VcVarsAll),
    ('cd /d "{0}"' -f $OutDir),
    'echo ==== ARCH ENTRIES ====',
    ('cuobjdump -lelf "{0}"' -f $dll),
    'echo ==== EXPORTS ====',
    ('dumpbin /nologo /exports "{0}"' -f $dll),
    'echo ==== IMPORTS ====',
    ('dumpbin /nologo /imports "{0}"' -f $dll),
    'exit /b %ERRORLEVEL%'
)
Set-Content -Path $vbat -Value $vl -Encoding ASCII

$ErrorActionPreference = 'Continue'
$verifyOut = (& cmd.exe /c $vbat 2>&1 | Out-String)
$verifyRc  = $LASTEXITCODE
$ErrorActionPreference = $prevEap
Write-Host $verifyOut

if ($verifyRc -ne 0) { throw "PE verification step returned rc=$verifyRc" }

$symbols = @(
    'escha_official_bridge_abi_version', 'escha_official_bridge_error', 'escha_official_bridge_probe',
    'escha_official_code_gemm', 'escha_official_code_gemm_pretransformed',
    'escha_official_decode_gemv_main_batch_v1', 'escha_official_decode_gemv_main_desc_v1',
    'escha_official_decode_gemv_main_f32_raw', 'escha_official_decode_gemv_main_raw',
    'escha_official_decode_gemv_raw'
)
$missingExports = @($symbols | Where-Object { $verifyOut -notmatch [regex]::Escape($_) })
if ($missingExports.Count -gt 0) {
    throw ('missing exports in built DLL: ' + ($missingExports -join ', '))
}
Write-Host 'exports: 10/10 present'

$missingArch = @()
foreach ($a in $CudaArch) {
    $hits = ([regex]::Matches($verifyOut, ('sm_' + [regex]::Escape($a) + '(?!\d)'))).Count
    Write-Host ('arch {0}: {1} ELF entries' -f ("sm_$a"), $hits)
    if ($hits -eq 0) { $missingArch += "sm_$a" }
}
if ($missingArch.Count -gt 0) {
    throw ('built DLL has no device code for: ' + ($missingArch -join ', ') + '. Rebuild with -CudaArch covering every shipped architecture.')
}

# dumpbin lists each imported module as an indented "<name>.dll" line.
$imports = @([regex]::Matches($verifyOut, '(?im)^\s{2,}([A-Za-z0-9_.\-]+\.dll)\s*$') | ForEach-Object { $_.Groups[1].Value } | Sort-Object -Unique)
Write-Host ('imports: ' + ($imports -join ', '))

$hash = (Get-FileHash $dll -Algorithm SHA256).Hash.ToLower()
$len  = (Get-Item $dll).Length

Write-Host ''
Write-Host "dll    : $dll"
Write-Host "bytes  : $len"
Write-Host "sha256 : $hash"
Write-Host "arch   : $($CudaArch -join ';')"

if (-not $KeepBat) { Remove-Item $bat, $vbat -ErrorAction SilentlyContinue }

Write-Host 'PE verification: PASS (exports, architectures and imports reported above)'
