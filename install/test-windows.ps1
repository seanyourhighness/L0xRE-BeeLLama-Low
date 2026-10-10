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
    $namedCatalog = $catalog | ConvertTo-Json -Depth 20 | ConvertFrom-Json
    $namedCatalog.packages.windows.sm120 | Add-Member NoteProperty certified_gpu_names @('NVIDIA GeForce RTX 5090') -Force
    $variant5090 = [pscustomobject]@{index=0;uuid='GPU-variant';name='NVIDIA GeForce RTX 5090 D';arch='sm120';memory_mib=32607}
    Assert-Rejected { Get-InstallPlan $namedCatalog $variant5090 $false '' } 'Similarly named GPU silently inherited certification'
    Assert-Install (-not (Get-InstallPlan $namedCatalog $variant5090 $true '').hardware_qualified_for_gpu) 'Variant GPU incorrectly certified'
    Assert-Rejected { Get-InstallPlan $catalog $ada $false '' } 'SM89 candidate silently selected'
    Assert-Rejected { Get-InstallPlan $catalog $ampere $false '' } 'SM86 candidate silently selected'
    Assert-Install ((Get-InstallPlan $catalog $ada $true '').package.status -eq 'candidate') 'Explicit candidate unavailable'
    $low = [pscustomobject]@{ name = 'RTX 5080'; arch = 'sm120'; memory_mib = 16384 }
    Assert-Rejected { Get-InstallPlan $catalog $low $false '' } 'Low VRAM candidate selected without opt-in'
    Assert-Install ((Get-InstallPlan $catalog $low $true '').package.filename -match '12gb') 'SM120 low VRAM profile not selected'
    $other = [pscustomobject]@{ name = 'RTX 5080'; arch = 'sm120'; memory_mib = 32768 }
    Assert-Install (-not (Get-InstallPlan $catalog $other $true '').hardware_qualified_for_gpu) '5080 incorrectly inherits 5090 certification'
    $smallAda = [pscustomobject]@{ index=1; uuid='GPU-small'; name='NVIDIA GeForce RTX 4070 Ti'; arch='sm89'; memory_mib=12288 }
    Assert-Install ((Get-InstallPlan $catalog $smallAda $true '').package.status -eq 'candidate') '4070 Ti unavailable'
    $eightGb = [pscustomobject]@{ name='NVIDIA GeForce RTX 5060'; arch='sm120'; memory_mib=8192 }
    Assert-Rejected { Get-InstallPlan $catalog $eightGb $true '' } '8GB card accepted'
    $savedGpuFunction = (Get-Item Function:Get-InstallGpu).ScriptBlock
    function Get-InstallGpu([string]$Selector) { return [pscustomobject]@{ index=[int]$Selector; uuid=('GPU-'+$Selector); name='NVIDIA GeForce RTX 3060'; arch='sm86'; memory_mib=12288 } }
    Assert-Rejected { Get-DualInstallPlan $catalog '0,1' $false '' } 'Dual mode selected without opt-in'
    Assert-Rejected { Get-DualInstallPlan $catalog '0,0' $true '' } 'Duplicate dual GPU accepted'
    $pair = Get-DualInstallPlan $catalog '0,1' $true ''
    Assert-Install ($pair.mode -eq 'dual-fast-experimental' -and -not $pair.hardware_qualified_for_gpu) 'Dual candidate mislabeled'
    Set-Item Function:Get-InstallGpu $savedGpuFunction

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
    $fixturePackage.gpu_selector_adapter = $false
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
    Assert-Install (Test-Path -LiteralPath (Join-Path $InstallDir 'update.ps1')) 'Update entry point missing'
    $oldReceipt = Get-Content -Raw -LiteralPath (Join-Path $InstallDir 'INSTALLATION.json') | ConvertFrom-Json
    $nextRuntime = Join-Path $InstallDir 'fixture-next'
    Copy-Item -LiteralPath $oldReceipt.runtime -Destination $nextRuntime -Recurse
    $currentPlan = Get-InstallPlan $catalog $blackwell $false ''
    Write-InstallLauncher $InstallDir $nextRuntime $modelRoot $currentPlan $true 8190 | Out-Null
    $script:Rollback = $true
    $script:DryRun = $true
    Invoke-L0xreInstall
    $afterPreview = Get-Content -Raw -LiteralPath (Join-Path $InstallDir 'INSTALLATION.json') | ConvertFrom-Json
    Assert-Install ($afterPreview.runtime -eq $nextRuntime) 'Rollback preview changed installation'
    $script:DryRun = $false
    Invoke-L0xreInstall
    $restored = Get-Content -Raw -LiteralPath (Join-Path $InstallDir 'INSTALLATION.json') | ConvertFrom-Json
    Assert-Install ($restored.runtime -eq $oldReceipt.runtime -and $restored.port -eq 8190 -and $restored.vision) 'Rollback lost runtime or settings'
    $script:Rollback = $false
    $script:Update = $true
    $script:DryRun = $true
    $script:ModelsDir = ''
    $script:Vision = $false
    $script:Port = 8080
    $updatePreview = Invoke-L0xreInstall | ConvertFrom-Json
    Assert-Install ($updatePreview.models_dir -eq $modelRoot -and $updatePreview.port -eq 8190 -and $updatePreview.vision) 'Update lost installed preferences'
    $script:Update = $false
    $script:DryRun = $false
    $script:ModelsDir = $modelRoot

    $adapterRoot = Join-Path $fixtureRoot 'adapter'
    $sealed = Join-Path $adapterRoot 'sealed'
    [IO.Directory]::CreateDirectory($sealed) | Out-Null
    $adapterStub = @'
param([string]$Model, [string]$Draft, [string]$BindAddress, [int]$Port, [switch]$DryRun)
$packageRoot=$PSScriptRoot
function nvidia-smi.exe { @{ query=@($args); root=$packageRoot; selected=$env:CUDA_VISIBLE_DEVICES } | ConvertTo-Json -Compress }
$gpuRecords=@(& nvidia-smi.exe --query-gpu=uuid,compute_cap --format=csv,noheader)
$gpuRecords
'@
    [IO.File]::WriteAllText((Join-Path $sealed 'l0xre.ps1'), $adapterStub)
    $beforeAdapter = (Get-FileHash (Join-Path $sealed 'l0xre.ps1')).Hash
    $adapterPlan = Get-InstallPlan $catalog $blackwell $false ''
    $adapterPlan.package.gpu_selector_adapter = $true
    $adapterStart = Write-InstallLauncher $adapterRoot $sealed $modelRoot $adapterPlan $false 8080
    $adapterResult = & $native -NoProfile -File $adapterStart -DryRun | ConvertFrom-Json
    Assert-Install ($LASTEXITCODE -eq 0 -and $adapterResult.query -contains '--id=GPU-fixture') 'SM120 adapter ignores selected GPU'
    Assert-Install ($adapterResult.root -eq $sealed) 'SM120 adapter changed package root'
    Assert-Install ((Get-FileHash (Join-Path $sealed 'l0xre.ps1')).Hash -eq $beforeAdapter) 'Sealed launcher was modified'
    $adapterReceipt = Get-Content -Raw (Join-Path $adapterRoot 'INSTALLATION.json') | ConvertFrom-Json
    $adapterRelative = @($adapterReceipt.selection.launcher_files.PSObject.Properties.Name)[0]
    [IO.File]::AppendAllText((Join-Path $adapterRoot $adapterRelative), '# tampered')
    try { $ErrorActionPreference = 'Continue'; & $native -NoProfile -File $adapterStart -DryRun 2>$null | Out-Null }
    finally { $ErrorActionPreference = 'Stop' }
    Assert-Install ($LASTEXITCODE -ne 0) 'Tampered SM120 adapter accepted'
    $adapterPlan.package.gpu_selector_adapter = $false

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

    $profileCatalog = $catalog | ConvertTo-Json -Depth 20 | ConvertFrom-Json
    $profileCatalog.packages.windows.sm89 | Add-Member NoteProperty launch_profiles @(
        [pscustomobject]@{ gpu_name='NVIDIA GeForce RTX 4070 Ti'; min_memory_mib=12000; profile='r6-sm89-4070ti' })
    $selectedProfile = Get-InstallPlan $profileCatalog $smallAda $true ''
    Assert-Install ($selectedProfile.launch_profile -eq 'r6-sm89-4070ti') 'Measured Windows profile not selected for its named GPU'
    Assert-Install ((Get-InstallPlan $profileCatalog $ada $true '').launch_profile -eq 'r6') 'Another SM89 card inherited the measured 4070 Ti profile'
    $selectedProfile.package.launcher = 'fixture.ps1'
    $profileStart = Write-InstallLauncher $candidateDir $candidateDir $modelRoot $selectedProfile $false 8080
    $profileResponse = & $native -NoProfile -File $profileStart -DryRun | ConvertFrom-Json
    Assert-Install (@($profileResponse.argv) -contains 'r6-sm89-4070ti') 'Selected profile missing from the actual generated launcher'

    $dualDir = Join-Path $fixtureRoot 'dual-install'
    $dualRuntime = Join-Path $dualDir 'runtime'
    $addon = Join-Path $dualDir 'addon'
    [IO.Directory]::CreateDirectory($dualRuntime) | Out-Null
    [IO.Directory]::CreateDirectory((Join-Path $addon 'sm86')) | Out-Null
    $fastProfile = @{ argv=@('@PACKAGE_ROOT@/bin/llama-server.exe','--port','8080','-c','81920','--spec-draft-n-max','7')
        env=@{ ESCHA_OFFICIAL_RAW_BRIDGE='1'; L0XRE_INT8_PREFILL='1' }
        model_sha256=@{ target=$catalog.models.target.sha256; draft=$catalog.models.draft.sha256 } }
    $profileFile = Join-Path $dualRuntime 'profile.json'
    [IO.File]::WriteAllText($profileFile, ($fastProfile | ConvertTo-Json -Depth 8))
    [IO.File]::WriteAllText((Join-Path $dualRuntime 'SHA256SUMS'), ((Get-FileHash $profileFile).Hash.ToLowerInvariant() + "  profile.json`n"))
    $addonFile = Join-Path $addon 'DUAL-EXPERIMENT.json'
    [IO.File]::WriteAllText($addonFile, '{"hardware_qualified":false}')
    [IO.File]::WriteAllText((Join-Path $addon 'SHA256SUMS'), ((Get-FileHash $addonFile).Hash.ToLowerInvariant() + "  DUAL-EXPERIMENT.json`n"))
    $pair | Add-Member NoteProperty dual_addon_root $addon
    Write-InstallLauncher $dualDir $dualRuntime $modelRoot $pair $false 8192 | Out-Null
    $dualReceipt = Get-Content -Raw -LiteralPath (Join-Path $dualDir 'INSTALLATION.json') | ConvertFrom-Json
    $helperRelative = @($dualReceipt.selection.launcher_files.PSObject.Properties.Name)[0]
    function nvidia-smi {
        $global:LASTEXITCODE=0
        $selector=(@($args | Where-Object { $_ -like '--id=*' })[0]).Substring(5)
        return "$selector, NVIDIA GeForce RTX 3060, 8.6, 12288"
    }
    $env:ESCHA_UNKNOWN='1'
    $dualResult = & (Join-Path $dualDir $helperRelative) -DryRun | ConvertFrom-Json
    Assert-Install ($dualResult.env.CUDA_VISIBLE_DEVICES -eq 'GPU-0,GPU-1') 'Dual visibility collapsed'
    Assert-Install ($dualResult.argv -contains '81920' -and $dualResult.argv -contains '7' -and $dualResult.argv -contains 'layer' -and $dualResult.argv -contains '1,1') 'Dual profile lost R6 settings'
    Assert-Install (-not $dualResult.hardware_qualified -and $dualResult.external_fast_bridges) 'Dual route loses fast bridge or claims certification'
    Assert-Install ($dualResult.env.ESCHA_OFFICIAL_RAW_BRIDGE -eq '1' -and $dualResult.env.L0XRE_INT8_PREFILL -eq '1') 'Fast paths disabled'
    Assert-Install ($dualResult.env.ESCHA_OFFICIAL_BRIDGE_LIBRARY -eq (Join-Path $addon 'sm86\bridge-dual.dll')) 'Wrong dual bridge selected'
    Assert-Install (-not (Test-Path Env:ESCHA_UNKNOWN)) 'Stale experiment environment leaked'

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
