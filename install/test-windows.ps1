$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'install.ps1')
$fixtureRoot = Join-Path ([IO.Path]::GetTempPath()) ("l0xre install fixtures '" + [guid]::NewGuid())
[IO.Directory]::CreateDirectory($fixtureRoot) | Out-Null
$script:checks = 0
function Assert-Install($Condition, [string]$Message) {
    if (-not $Condition) { throw "FAIL: $Message" }
    $script:checks++
}
function Assert-Rejected([scriptblock]$Action, [string]$Message) {
    $failed = $false
    try { & $Action | Out-Null } catch { $failed = $true }
    Assert-Install $failed $Message
}
function New-FixtureRecord([string]$Path) {
    return [pscustomobject]@{ filename = [IO.Path]::GetFileName($Path); bytes = (Get-Item -LiteralPath $Path).Length
        sha256 = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant(); url = 'https://example.invalid/fixture' }
}
try {
    $catalog = Get-InstallCatalog ''
    $blackwell = [pscustomobject]@{ index = 0; uuid = 'GPU-fixture'; name = 'NVIDIA GeForce RTX 5090'; arch = 'sm120'; memory_mib = 32607 }
    $ada = [pscustomobject]@{ index = 1; uuid = 'GPU-ada'; name = 'NVIDIA GeForce RTX 4090'; arch = 'sm89'; memory_mib = 24576 }
    $ampere = [pscustomobject]@{ index = 2; uuid = 'GPU-ampere'; name = 'NVIDIA GeForce RTX 3060'; arch = 'sm86'; memory_mib = 12288 }
    Assert-Install ((Get-InstallPlan $catalog $blackwell $false '').package.status -eq 'certified') 'Wrong certified Windows route'
    Assert-Rejected { Get-InstallPlan $catalog $ada $false '' } 'SM89 candidate silently selected'
    Assert-Rejected { Get-InstallPlan $catalog $ampere $false '' } 'SM86 candidate silently selected'
    Assert-Install ((Get-InstallPlan $catalog $ada $true '').package.status -eq 'candidate') 'Explicit candidate unavailable'
    $low = [pscustomobject]@{ name = 'RTX 5080'; arch = 'sm120'; memory_mib = 16384 }
    Assert-Rejected { Get-InstallPlan $catalog $low $true '' } 'Insufficient VRAM accepted'
    $other = [pscustomobject]@{ name = 'RTX 5080'; arch = 'sm120'; memory_mib = 32768 }
    Assert-Rejected { Get-InstallPlan $catalog $other $true '' } 'Wrong SM120 GPU scope accepted'

    $originalVisible = $env:CUDA_VISIBLE_DEVICES
    function nvidia-smi { $global:LASTEXITCODE = 0; return '1, GPU-second, NVIDIA GeForce RTX 4090, 8.9, 24576' }
    $env:CUDA_VISIBLE_DEVICES = 'GPU-second,GPU-first'
    Assert-Install ((Get-InstallGpu '').uuid -eq 'GPU-second') 'Visible GPU detection failed'
    $env:CUDA_VISIBLE_DEVICES = ''
    # PowerShell removes an environment variable when assigned an empty string.
    Assert-Rejected { Get-InstallGpu '-1' } 'Disabled GPU accepted'
    $env:CUDA_VISIBLE_DEVICES = $originalVisible

    $existing = Join-Path $fixtureRoot 'download.bin'
    [IO.File]::WriteAllText($existing, 'fixture')
    $expected = New-FixtureRecord $existing
    Assert-Install ((Get-InstallArtifact $expected $fixtureRoot) -eq $existing) 'Existing artifact not reused'
    [IO.File]::Move($existing, $existing + '.partial')
    Assert-Install ((Get-InstallArtifact $expected $fixtureRoot) -eq $existing) 'Complete partial not reused'
    [IO.File]::WriteAllText($existing, 'wrong')
    Assert-Rejected { Get-InstallArtifact $expected $fixtureRoot } 'Wrong model overwritten or accepted'
    Assert-Install ([IO.File]::ReadAllText($existing) -eq 'wrong') 'Existing mismatched file changed'
    Assert-Rejected { Get-PackagePath $fixtureRoot '../outside' } 'Path traversal accepted'
    Assert-Rejected { Get-PackagePath $fixtureRoot 'file:stream' } 'Alternate data stream accepted'
    Push-Location -LiteralPath $fixtureRoot
    try { Assert-Install ((Get-InstallFullPath '.\relative') -eq (Join-Path $fixtureRoot 'relative')) 'Relative path ignores PowerShell location' }
    finally { Pop-Location }

    $source = Join-Path $fixtureRoot 'source'
    $packageRoot = Join-Path $source 'package'
    [IO.Directory]::CreateDirectory($packageRoot) | Out-Null
    $stub = @'
param([Parameter(Mandatory=$true)][string]$Model, [Parameter(Mandatory=$true)][string]$Draft,
      [string]$Mmproj, [string]$BindAddress, [int]$Port, [switch]$DryRun)
@{ model = $Model; draft = $Draft; vision = $Mmproj; host = $BindAddress; port = $Port;
   dry = [bool]$DryRun; gpu = $env:CUDA_VISIBLE_DEVICES } | ConvertTo-Json -Compress
'@
    $launcher = Join-Path $packageRoot 'l0xre.ps1'
    [IO.File]::WriteAllText($launcher, $stub)
    $checksum = (Get-FileHash -LiteralPath $launcher -Algorithm SHA256).Hash.ToLowerInvariant()
    [IO.File]::WriteAllText((Join-Path $packageRoot 'SHA256SUMS'), "$checksum  l0xre.ps1`n")
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zipPath = Join-Path $fixtureRoot 'fixture.zip'
    [IO.Compression.ZipFile]::CreateFromDirectory($source, $zipPath)
    $fixturePackage = $catalog.packages.windows.sm120
    $archiveRecord = New-FixtureRecord $zipPath
    foreach ($key in @('filename', 'bytes', 'sha256', 'url')) { $fixturePackage.$key = $archiveRecord.$key }
    $fixturePackage.install_reserve_bytes = 1000
    $modelRoot = Join-Path $fixtureRoot "models with spaces '"
    [IO.Directory]::CreateDirectory($modelRoot) | Out-Null
    foreach ($key in @('target', 'draft', 'vision')) {
        $path = Join-Path $modelRoot $catalog.models.$key.filename
        [IO.File]::WriteAllText($path, $key)
        $catalog.models.$key = New-FixtureRecord $path
    }
    $fixtureCatalog = Join-Path $fixtureRoot 'catalog.json'
    [IO.File]::WriteAllText($fixtureCatalog, ($catalog | ConvertTo-Json -Depth 12))
    $script:CatalogPath = $fixtureCatalog
    $script:InstallDir = Join-Path $fixtureRoot 'installation'
    $script:ModelsDir = $modelRoot
    $script:RuntimeArchive = $zipPath
    $script:Yes = $true
    $script:Vision = $true
    $script:Port = 8190
    function Get-InstallGpu { return $blackwell }
    function Test-InstallCpu { } # Fixture host need not have eight cores; no inference is executed.
    $script:DryRun = $true
    Invoke-L0xreInstall | Out-Null
    Assert-Install (-not (Test-Path -LiteralPath $InstallDir)) 'Dry run wrote installation files'
    $script:DryRun = $false
    Invoke-L0xreInstall
    Invoke-L0xreInstall
    $start = Join-Path $InstallDir 'start.ps1'
    $native = Join-Path $PSHOME 'powershell.exe'
    $result = & $native -NoProfile -File $start -DryRun | ConvertFrom-Json
    Assert-Install ($LASTEXITCODE -eq 0) 'Generated Windows launcher failed'
    Assert-Install ($result.model -eq (Join-Path $modelRoot 'L0xRE-27b-Low.gguf')) 'Model path quoting failed'
    Assert-Install ($result.draft -eq (Join-Path $modelRoot 'Qwen3.8-27B-DFlash2-Q4_K_M.gguf')) 'Draft path quoting failed'
    Assert-Install ($result.vision -eq (Join-Path $modelRoot 'mmproj-Qwen3.8-27B-Q8_0.gguf')) 'CPU vision path missing'
    Assert-Install ($result.gpu -eq 'GPU-fixture' -and $result.port -eq 8190 -and $result.dry) 'GPU, port or dry-run forwarding failed'

    $candidateDir = Join-Path $fixtureRoot 'candidate'
    [IO.Directory]::CreateDirectory($candidateDir) | Out-Null
    $candidateStub = 'param([Parameter(ValueFromRemainingArguments=$true)][string[]]$Rest); @{ argv = @($Rest) } | ConvertTo-Json -Compress'
    [IO.File]::WriteAllText((Join-Path $candidateDir 'fixture.ps1'), $candidateStub)
    $plan = Get-InstallPlan $catalog $ada $true ''
    $plan.package.launcher = 'fixture.ps1'
    $candidateStart = Write-InstallLauncher $candidateDir $candidateDir $modelRoot $plan $false 8080
    $candidateResponse = & $native -NoProfile -File $candidateStart -DryRun | ConvertFrom-Json
    $arguments = @($candidateResponse.argv)
    Assert-Install ($arguments -contains '--qualification-probe' -and $arguments -contains '--dry-run') ("Candidate launcher flags not forwarded: " + ($arguments | ConvertTo-Json -Depth 4 -Compress))
    Assert-Install ($arguments -contains 'r6') 'Candidate sealed profile missing'

    $badZip = Join-Path $fixtureRoot 'unsafe.zip'
    $zip = [IO.Compression.ZipFile]::Open($badZip, [IO.Compression.ZipArchiveMode]::Create)
    try { $zip.CreateEntry('../escaped.txt') | Out-Null } finally { $zip.Dispose() }
    Assert-Rejected { Expand-InstallPackage $badZip (Join-Path $fixtureRoot 'unsafe-runtime') $fixturePackage } 'ZIP traversal accepted'
    Assert-Install (-not (Test-Path -LiteralPath (Join-Path $fixtureRoot 'escaped.txt'))) 'ZIP escaped extraction root'
    $receipt = Get-Content -Raw -LiteralPath (Join-Path $InstallDir 'INSTALLATION.json') | ConvertFrom-Json
    [IO.File]::WriteAllText((Join-Path $receipt.runtime 'l0xre.ps1'), 'changed')
    Assert-Rejected { Invoke-L0xreInstall } 'Tampered installed payload reused'
    Write-Host "WINDOWS_INSTALLER_PASS: $checks checks; native PowerShell $($PSVersionTable.PSVersion)"
} finally {
    Remove-Item -LiteralPath $fixtureRoot -Recurse -Force
}
