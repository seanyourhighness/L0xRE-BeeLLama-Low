param(
    [string]$InstallDir = (Join-Path ([Environment]::GetFolderPath('LocalApplicationData')) 'L0xRE'),
    [string]$ModelsDir,
    [string]$Gpu,
    [string]$Gpus,
    [string]$CatalogPath,
    [string]$RuntimeArchive,
    [switch]$AllowCandidate,
    [switch]$Update,
    [switch]$Rollback,
    [switch]$Vision,
    [switch]$RuntimeOnly,
    [switch]$DryRun,
    [switch]$Yes,
    [ValidateRange(1,65535)][int]$Port = 8080
)
$ErrorActionPreference = 'Stop'
$installBoundParameters = @{} + $PSBoundParameters
Set-StrictMode -Version 2.0

function Get-InstallCatalog([string]$Path) {
    $local = if ($Path) { $Path } else { Join-Path $PSScriptRoot 'catalog.json' }
    if (Test-Path -LiteralPath $local -PathType Leaf) {
        $catalog = Get-Content -Raw -LiteralPath $local | ConvertFrom-Json
    } elseif ($Path) {
        throw "Catalog not found: $local"
    } else {
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        $url = 'https://raw.githubusercontent.com/seanyourhighness/L0xRE-BeeLLama-Low/main/install/catalog.json'
        $catalog = (Invoke-WebRequest -UseBasicParsing -Uri $url).Content | ConvertFrom-Json
    }
    if ($catalog.schema_version -ne 1) { throw 'Unsupported installer catalog; download the installer again.' }
    return $catalog
}

function Get-InstallGpu([string]$Selector) {
    if (-not $Selector) {
        $Selector = if (Test-Path Env:CUDA_VISIBLE_DEVICES) { ($env:CUDA_VISIBLE_DEVICES -split ',')[0] } else { '0' }
    }
    if (-not $Selector -or $Selector -eq '-1') { throw 'No GPU selected; supply -Gpu INDEX or UUID.' }
    $rows = @(& nvidia-smi "--id=$Selector" '--query-gpu=index,uuid,name,compute_cap,memory.total' '--format=csv,noheader,nounits')
    if ($LASTEXITCODE -ne 0 -or $rows.Count -ne 1) { throw 'GPU detection failed; check nvidia-smi and the NVIDIA driver.' }
    $row = $rows[0] | ConvertFrom-Csv -Header index,uuid,name,cc,memory
    return [pscustomobject]@{
        index = [int]$row.index.Trim(); uuid = $row.uuid.Trim(); name = $row.name.Trim()
        arch = 'sm' + $row.cc.Trim().Replace('.', ''); memory_mib = [int]$row.memory.Trim()
    }
}

function Get-InstallPlan($Catalog, $SelectedGpu, [bool]$Candidates, [string]$Archive) {
    $entry = $Catalog.packages.windows.PSObject.Properties[$SelectedGpu.arch]
    if ($SelectedGpu.arch -eq 'sm120' -and ($SelectedGpu.memory_mib -lt 30000 -or -not $SelectedGpu.name.Contains('RTX 5090'))) {
        $variant = $Catalog.packages.windows.PSObject.Properties['sm120-12gb']
        if ($variant) { $entry = $variant }
    }
    if (-not $entry) { throw "Unsupported GPU architecture: $($SelectedGpu.arch)" }
    $package = $entry.Value
    if ($SelectedGpu.memory_mib -lt $package.min_memory_mib) { throw 'Insufficient VRAM for this packaged profile.' }
    if ($package.gpu_name_contains -and -not $SelectedGpu.name.Contains($package.gpu_name_contains)) {
        throw "This setup is scoped to $($package.gpu_name_contains)."
    }
    if ($package.status -eq 'candidate' -and -not $Candidates) {
        throw "$($SelectedGpu.arch) Windows is a candidate. Use -AllowCandidate only to opt into testing it."
    }
    if (-not $package.url -and -not $Archive) { throw 'This archive is not published; supply -RuntimeArchive PATH.' }
    $measured = $package.gpu_name_contains
    if ($package.PSObject.Properties['certified_gpu_name_contains']) { $measured = $package.certified_gpu_name_contains }
    return [pscustomobject]@{ gpu = $SelectedGpu; package = $package; platform = 'windows'
        hardware_qualified_for_gpu = ($package.status.StartsWith('certified') -and [bool]$measured -and $SelectedGpu.name.Contains($measured)) }
}

function Get-DualInstallPlan($Catalog, [string]$Selectors, [bool]$Candidates, [string]$Archive) {
    if (-not $Candidates) { throw 'Dual-GPU native layer split is experimental; use -AllowCandidate.' }
    $ids = @($Selectors -split ',')
    if ($ids.Count -ne 2 -or -not $ids[0].Trim() -or -not $ids[1].Trim()) { throw 'Supply exactly two GPU indices or UUIDs: -Gpus 0,1.' }
    $cards = @($ids | ForEach-Object { Get-InstallGpu $_.Trim() })
    if ($cards[0].uuid -eq $cards[1].uuid -or $cards[0].arch -ne $cards[1].arch -or $cards[0].name -ne $cards[1].name -or [Math]::Abs($cards[0].memory_mib - $cards[1].memory_mib) -gt 256) {
        throw 'Dual mode requires two distinct matched cards of the same model and VRAM size.'
    }
    if ($cards[0].memory_mib -lt 12000 -or $cards[1].memory_mib -lt 12000) { throw 'Dual mode requires at least 12GB VRAM per card.' }
    if (-not $Catalog.PSObject.Properties['dual_addons']) { throw 'This catalog does not contain the experimental dual fast-path add-on.' }
    $plan = Get-InstallPlan $Catalog $cards[0] $true $Archive
    $plan | Add-Member NoteProperty gpus $cards
    $plan | Add-Member NoteProperty mode 'dual-fast-experimental'
    $plan | Add-Member NoteProperty dual_addon $Catalog.dual_addons.windows
    $plan.hardware_qualified_for_gpu = $false
    return $plan
}

function Test-InstallArtifact([string]$Path, $Record) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf) -or
        (Get-Item -LiteralPath $Path).Length -ne $Record.bytes -or
        (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $Record.sha256) {
        throw "Size or SHA-256 mismatch: $Path. Existing files are never overwritten."
    }
}

function Get-InstallArtifact($Record, [string]$Directory) {
    if ([IO.Path]::GetFileName($Record.filename) -ne $Record.filename) { throw 'Invalid artifact filename.' }
    $destination = Join-Path $Directory $Record.filename
    if (Test-Path -LiteralPath $destination) {
        Test-InstallArtifact $destination $Record
        Write-Host "Reusing verified $($Record.filename)"
        return $destination
    }
    [IO.Directory]::CreateDirectory($Directory) | Out-Null
    $partial = $destination + '.partial'
    if ((Test-Path -LiteralPath $partial) -and (Get-Item -LiteralPath $partial).Length -eq $Record.bytes) {
        try { Test-InstallArtifact $partial $Record; [IO.File]::Move($partial, $destination); return $destination }
        catch { Remove-Item -LiteralPath $partial }
    }
    Write-Host ("Downloading {0} ({1:N2} GB)" -f $Record.filename, ($Record.bytes / 1e9))
    & curl.exe --fail --location --retry 3 --continue-at - --output $partial $Record.url
    if ($LASTEXITCODE -ne 0) { throw 'Download failed; rerun setup to resume the partial download.' }
    try { Test-InstallArtifact $partial $Record }
    catch { if (Test-Path -LiteralPath $partial) { Remove-Item -LiteralPath $partial }; throw }
    [IO.File]::Move($partial, $destination)
    return $destination
}

function Get-PackagePath([string]$Root, [string]$Relative) {
    if (-not $Relative -or $Relative.Contains(':') -or [IO.Path]::IsPathRooted($Relative) -or ($Relative -split '[/\\]') -contains '..') {
        throw "Unsafe package path: $Relative"
    }
    $path = [IO.Path]::GetFullPath((Join-Path $Root $Relative))
    $prefix = [IO.Path]::GetFullPath($Root).TrimEnd('\') + '\'
    if (-not $path.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) { throw "Package path escapes root: $Relative" }
    return $path
}

function Test-InstallPackage([string]$Root) {
    $manifest = Join-Path $Root 'SHA256SUMS'
    if (-not (Test-Path -LiteralPath $manifest -PathType Leaf)) { throw 'Package is missing SHA256SUMS.' }
    $count = 0
    foreach ($line in [IO.File]::ReadAllLines($manifest)) {
        if (-not $line) { continue }
        if ($line -notmatch '^([a-fA-F0-9]{64}) [ *](.+)$') { throw 'Invalid package checksum entry.' }
        $expected = $Matches[1].ToLowerInvariant(); $relative = $Matches[2]
        $path = Get-PackagePath $Root $relative
        if (-not (Test-Path -LiteralPath $path -PathType Leaf) -or
            (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected) {
            throw "Package checksum mismatch: $relative"
        }
        $count++
    }
    if (-not $count) { throw 'Empty package checksum manifest.' }
    Write-Host "Verified $count package files"
}

function Expand-InstallPackage([string]$Archive, [string]$Destination, $Package) {
    if (Test-Path -LiteralPath $Destination) { Test-InstallPackage $Destination; return $Destination }
    $stage = Join-Path ([IO.Path]::GetDirectoryName($Destination)) ('.extract-' + [guid]::NewGuid())
    [IO.Directory]::CreateDirectory($stage) | Out-Null
    try {
        Add-Type -AssemblyName System.IO.Compression.FileSystem
        $zip = [IO.Compression.ZipFile]::OpenRead($Archive)
        try {
            foreach ($entry in $zip.Entries) {
                Get-PackagePath $stage $entry.FullName | Out-Null
                if ((($entry.ExternalAttributes -shr 16) -band 0xF000) -eq 0xA000) { throw 'ZIP symlinks are not supported.' }
            }
        } finally { $zip.Dispose() }
        [IO.Compression.ZipFile]::ExtractToDirectory($Archive, $stage)
        $roots = @(@($stage) + @(Get-ChildItem -LiteralPath $stage -Directory | ForEach-Object { $_.FullName }) |
            Where-Object { Test-Path -LiteralPath (Join-Path $_ $Package.launcher) -PathType Leaf })
        if ($roots.Count -ne 1) { throw 'Cannot identify the package launcher in this archive.' }
        Test-InstallPackage $roots[0]
        [IO.Directory]::Move($roots[0], $Destination)
    } finally { if (Test-Path -LiteralPath $stage) { Remove-Item -LiteralPath $stage -Recurse -Force } }
    return $Destination
}

function Test-InstallSpace([string]$Root, [string]$ModelRoot, $Records, [long]$Reserve) {
    $required = @{}
    $drive = [IO.Path]::GetPathRoot($Root)
    $required[$drive] = $Reserve
    $modelDrive = [IO.Path]::GetPathRoot($ModelRoot)
    if (-not $required.ContainsKey($modelDrive)) { $required[$modelDrive] = [long]0 }
    foreach ($record in $Records) {
        if (-not (Test-Path -LiteralPath (Join-Path $ModelRoot $record.filename))) { $required[$modelDrive] += $record.bytes }
    }
    foreach ($key in $required.Keys) {
        if ((New-Object IO.DriveInfo $key).AvailableFreeSpace -lt $required[$key]) {
            throw ("Insufficient disk space on {0}: need about {1:N1} GB" -f $key, ($required[$key] / 1e9))
        }
    }
}

function Quote-InstallString([string]$Value) { return "'" + $Value.Replace("'", "''") + "'" }

function Get-InstallFullPath([string]$Path) {
    return $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Path)
}

function Test-InstallCpu {
    $affinity = [Diagnostics.Process]::GetCurrentProcess().ProcessorAffinity.ToInt64()
    if ([Environment]::ProcessorCount -lt 8 -or ($affinity -band 255) -ne 255) {
        throw 'The packaged profile requires CPU cores 0-7 to be accessible.'
    }
}

function Write-InstallLauncher([string]$Root, [string]$Runtime, [string]$ModelRoot, $Plan, [bool]$WithVision, [int]$ListenPort) {
    $receiptPath = Join-Path $Root 'INSTALLATION.json'
    $target = Join-Path $ModelRoot 'L0xRE-27b-Low.gguf'
    $draft = Join-Path $ModelRoot 'Qwen3.8-27B-DFlash2-Q4_K_M.gguf'
    if ($Plan.package.launcher_style -eq 'sm120-windows') {
        $parameters = @{ Model = $target; Draft = $draft; BindAddress = '127.0.0.1'; Port = $ListenPort }
        if ($Plan.package.status -eq 'candidate') { $parameters.QualificationProbe = $true }
        if ($WithVision) { $parameters.Mmproj = Join-Path $ModelRoot 'mmproj-Qwen3.8-27B-Q8_0.gguf' }
        $entries = @($parameters.Keys | ForEach-Object { $_ + ' = ' + $(if ($parameters[$_] -is [bool]) { '$true' } else { Quote-InstallString $parameters[$_] }) })
        $callLines = @(('$serverParameters = @{ ' + ($entries -join '; ') + ' }'),
            'if ($DryRun) { $serverParameters.DryRun = $true }', '& $command @serverParameters')
    } else {
        $arguments = @('serve', '--profile', 'r6', '-m', $target, '-md', $draft, '--host', '127.0.0.1', '--port', "$ListenPort")
        $dryFlag = '--dry-run'
        if ($Plan.package.status -eq 'candidate') { $arguments += '--qualification-probe' }
        if ($WithVision) { $arguments += @('--mmproj', (Join-Path $ModelRoot 'mmproj-Qwen3.8-27B-Q8_0.gguf')) }
        $callLines = @(('$argsForServer = @(' + (($arguments | ForEach-Object { Quote-InstallString $_ }) -join ', ') + ')'),
            ('if ($DryRun) { $argsForServer += ' + (Quote-InstallString $dryFlag) + ' }'), '& $command @argsForServer')
    }
    $command = Join-Path $Runtime $Plan.package.launcher
    $integrityLines = @()
    if ($Plan.package.PSObject.Properties['gpu_selector_adapter'] -and $Plan.package.gpu_selector_adapter) {
        # The sealed SM120 launcher queries all GPUs and picks physical GPU 0.
        # Keep it intact and adapt only selection and package-root resolution in a versioned copy.
        $source = [IO.File]::ReadAllText($command)
        $query = '$gpuRecords=@(& nvidia-smi.exe --query-gpu=uuid,compute_cap --format=csv,noheader)'
        $rootLine = '$packageRoot=$PSScriptRoot'
        if (-not $source.Contains($query) -or -not $source.Contains($rootLine)) { throw 'Unexpected SM120 launcher; update the installer.' }
        $source = $source.Replace($rootLine, ('$packageRoot=' + (Quote-InstallString $Runtime))).Replace($query,
            '$gpuRecords=@(& nvidia-smi.exe --id=$env:CUDA_VISIBLE_DEVICES --query-gpu=uuid,compute_cap --format=csv,noheader)')
        $bytes = (New-Object Text.UTF8Encoding $false).GetBytes($source)
        $hasher = [Security.Cryptography.SHA256]::Create()
        try { $hash = ([BitConverter]::ToString($hasher.ComputeHash($bytes))).Replace('-','').ToLowerInvariant() } finally { $hasher.Dispose() }
        $relative = 'launchers\' + $hash.Substring(0,16) + '\single-sm120.ps1'
        $command = Join-Path $Root $relative
        [IO.Directory]::CreateDirectory((Split-Path $command -Parent)) | Out-Null
        [IO.File]::WriteAllBytes($command, $bytes)
        $hashes = @{}; $hashes[$relative] = $hash
        $Plan | Add-Member NoteProperty launcher_files $hashes -Force
        $integrityLines = @('if ((Get-FileHash -LiteralPath $command -Algorithm SHA256).Hash.ToLowerInvariant() -ne ' + (Quote-InstallString $hash) + ') { throw "Launcher integrity mismatch." }')
    }
    $lines = @('param([switch]$DryRun)', '$ErrorActionPreference = "Stop"',
        ('$env:CUDA_VISIBLE_DEVICES = ' + (Quote-InstallString $Plan.gpu.uuid)),
        ('$command = ' + (Quote-InstallString $command)))
    $lines += $integrityLines
    $lines += $callLines
    if ($Plan.PSObject.Properties['mode'] -and $Plan.mode -eq 'dual-fast-experimental') {
        $stage = Join-Path $Root ('.dual-launch-' + [guid]::NewGuid() + '.ps1')
        try {
            $local = Join-Path $PSScriptRoot 'dual-launch.ps1'
            if (Test-Path -LiteralPath $local) { Copy-Item -LiteralPath $local -Destination $stage }
            else { Invoke-WebRequest -UseBasicParsing 'https://raw.githubusercontent.com/seanyourhighness/L0xRE-BeeLLama-Low/main/install/dual-launch.ps1' -OutFile $stage }
            $hash = (Get-FileHash -LiteralPath $stage -Algorithm SHA256).Hash.ToLowerInvariant()
            $helperRoot = Join-Path $Root ('launchers\' + $hash.Substring(0,16))
            [IO.Directory]::CreateDirectory($helperRoot) | Out-Null
            $helper = Join-Path $helperRoot 'dual-launch.ps1'
            Copy-Item -LiteralPath $stage -Destination $helper -Force
            $hashes = @{}; $hashes['launchers\' + $hash.Substring(0,16) + '\dual-launch.ps1'] = $hash
            $Plan | Add-Member NoteProperty launcher_files $hashes -Force
            $lines = @('param([switch]$DryRun)', '$ErrorActionPreference = "Stop"', ('& ' + (Quote-InstallString $helper) + ' -DryRun:$DryRun'))
        } finally { if (Test-Path -LiteralPath $stage) { Remove-Item -LiteralPath $stage } }
    }
    $lines += 'if ($null -ne $LASTEXITCODE) { exit $LASTEXITCODE }'
    $script = Join-Path $Root 'start.ps1'
    $scriptText = ($lines -join "`r`n") + "`r`n"
    $changedRuntime = $false
    if (Test-Path -LiteralPath $receiptPath) {
        $old = Get-Content -Raw -LiteralPath $receiptPath | ConvertFrom-Json
        $oldAddon = if ($old.selection.PSObject.Properties['dual_addon_root']) { $old.selection.dual_addon_root } else { '' }
        $newAddon = if ($Plan.PSObject.Properties['dual_addon_root']) { $Plan.dual_addon_root } else { '' }
        $changedRuntime = $old.runtime -ne $Runtime -or $oldAddon -ne $newAddon
    }
    if ((Test-Path -LiteralPath $receiptPath) -and (Test-Path -LiteralPath $script) -and ($changedRuntime -or [IO.File]::ReadAllText($script) -ne $scriptText)) {
        $previous = Join-Path $Root 'previous'
        [IO.Directory]::CreateDirectory($previous) | Out-Null
        foreach ($name in @('INSTALLATION.json','start.ps1')) {
            Copy-Item -LiteralPath (Join-Path $Root $name) -Destination (Join-Path $previous $name) -Force
        }
    }
    [IO.File]::WriteAllText($script, $scriptText, (New-Object Text.UTF8Encoding $true))
    $receipt = @{ selection = $Plan; runtime = $Runtime; models_dir = $ModelRoot; vision = $WithVision; port = $ListenPort }
    [IO.File]::WriteAllText((Join-Path $Root 'INSTALLATION.json'), ($receipt | ConvertTo-Json -Depth 12), (New-Object Text.UTF8Encoding $false))
    $updateLines = @('param([switch]$DryRun, [switch]$Yes, [switch]$Rollback)', '$ErrorActionPreference = "Stop"',
        '$stage = Join-Path ([IO.Path]::GetTempPath()) ([guid]::NewGuid().ToString())',
        '[IO.Directory]::CreateDirectory($stage) | Out-Null', 'try {',
        '  $installer = Join-Path $stage "install.ps1"',
        '  Invoke-WebRequest -UseBasicParsing "https://raw.githubusercontent.com/seanyourhighness/L0xRE-BeeLLama-Low/main/install/install.ps1" -OutFile $installer',
        ('  & $installer -Update -InstallDir ' + (Quote-InstallString $Root) + ' -DryRun:$DryRun -Yes:$Yes -Rollback:$Rollback'),
        '} finally { Remove-Item -LiteralPath $stage -Recurse -Force }')
    [IO.File]::WriteAllText((Join-Path $Root 'update.ps1'), ($updateLines -join "`r`n"), (New-Object Text.UTF8Encoding $true))
    return $script
}

function Invoke-L0xreInstall {
    $root = Get-InstallFullPath $InstallDir
    if ($Rollback) {
        $previous = Join-Path $root 'previous'
        $receipt = Get-Content -Raw -LiteralPath (Join-Path $previous 'INSTALLATION.json') | ConvertFrom-Json
        Test-InstallPackage $receipt.runtime
        if ($receipt.selection.PSObject.Properties['dual_addon_root']) { Test-InstallPackage $receipt.selection.dual_addon_root }
        if ($receipt.selection.PSObject.Properties['launcher_files']) {
            foreach ($entry in $receipt.selection.launcher_files.PSObject.Properties) {
                if ((Get-FileHash -LiteralPath (Get-PackagePath $root $entry.Name)).Hash.ToLowerInvariant() -ne $entry.Value) { throw 'Previous launcher integrity mismatch.' }
            }
        }
        if (-not $DryRun) {
            foreach ($name in @('INSTALLATION.json','start.ps1')) {
                Copy-Item -LiteralPath (Join-Path $previous $name) -Destination (Join-Path $root $name) -Force
            }
        }
        Write-Host "Previous runtime: $($receipt.runtime)"
        return
    }
    if ($Update) {
        $old = Get-Content -Raw -LiteralPath (Join-Path $root 'INSTALLATION.json') | ConvertFrom-Json
        if (-not $Gpu -and -not $Gpus) {
            if ($old.selection.PSObject.Properties['mode'] -and $old.selection.mode -eq 'dual-fast-experimental') {
                $Gpus = ($old.selection.gpus | ForEach-Object { $_.uuid }) -join ','
                $AllowCandidate = $true
            } else { $Gpu = $old.selection.gpu.uuid }
        }
        if (-not $ModelsDir) { $ModelsDir = $old.models_dir }
        if (-not $installBoundParameters.ContainsKey('Vision')) { $Vision = [bool]$old.vision }
        if (-not $installBoundParameters.ContainsKey('Port')) { $Port = [int]$old.port }
        if ($old.selection.package.status -eq 'candidate') { $AllowCandidate = $true }
    }
    if ($Update -and -not $CatalogPath) {
        $catalog = (Invoke-WebRequest -UseBasicParsing 'https://raw.githubusercontent.com/seanyourhighness/L0xRE-BeeLLama-Low/main/install/catalog.json').Content | ConvertFrom-Json
        if ($catalog.schema_version -ne 1) { throw 'Download the current installer to read this catalog.' }
    } else { $catalog = Get-InstallCatalog $CatalogPath }
    if ($Gpu -and $Gpus) { throw 'Choose -Gpu or -Gpus, not both.' }
    $plan = if ($Gpus) { Get-DualInstallPlan $catalog $Gpus ([bool]$AllowCandidate) $RuntimeArchive }
            else { Get-InstallPlan $catalog (Get-InstallGpu $Gpu) ([bool]$AllowCandidate) $RuntimeArchive }
    $modelRoot = if ($ModelsDir) { Get-InstallFullPath $ModelsDir } else { Join-Path $root 'models' }
    if ($DryRun) {
        @{ selection = $plan; install_dir = $root; models_dir = $modelRoot; vision = [bool]$Vision; port = $Port } | ConvertTo-Json -Depth 12
        return
    }
    if (-not (Get-Command curl.exe -ErrorAction SilentlyContinue)) { throw 'curl.exe is required (included with current Windows installations).' }
    Test-InstallCpu
    $keys = if ($Vision) { @('target', 'draft', 'vision') } else { @('target', 'draft') }
    $records = @($keys | ForEach-Object { $catalog.models.PSObject.Properties[$_].Value })
    if (-not $RuntimeOnly) {
        foreach ($record in $records) {
            $path = Join-Path $modelRoot $record.filename
            if (Test-Path -LiteralPath $path) { Test-InstallArtifact $path $record }
        }
    }
    Write-Host "GPU: $($plan.gpu.name) / $($plan.gpu.arch)"
    Write-Host "Package: $($plan.package.status)"
    Write-Host "Certified on selected GPU: $($plan.hardware_qualified_for_gpu)"
    if ($plan.PSObject.Properties['mode']) { Write-Host 'Dual GPU: EXPERIMENTAL / UNTESTED / UNCERTIFIED; performance parity is unmeasured.' }
    Write-Host "Profile: $($plan.package.profile)"
    Write-Host "Install: $root`nModels: $modelRoot"
    if (-not $Yes) {
        if ($RuntimeOnly) { Write-Host 'Runtime-only setup; model downloads are skipped.' }
        else { Write-Host 'Setup may download about 10 GB of models plus the runtime. Matching files will be reused.' }
        if ((Read-Host 'Continue? [y/N]') -notmatch '^(y|yes)$') { return }
    }
    $needed = if ($RuntimeOnly) { @() } else { $records }
    Test-InstallSpace $root $modelRoot $needed $plan.package.install_reserve_bytes
    [IO.Directory]::CreateDirectory($root) | Out-Null
    $archive = if ($RuntimeArchive) { Get-InstallFullPath $RuntimeArchive } else { Get-InstallArtifact $plan.package (Join-Path $root 'downloads') }
    Test-InstallArtifact $archive $plan.package
    $runtime = Join-Path $root ($plan.gpu.arch + '-' + $plan.package.sha256.Substring(0,12))
    $runtime = Expand-InstallPackage $archive $runtime $plan.package
    if ($plan.PSObject.Properties['mode'] -and $plan.mode -eq 'dual-fast-experimental') {
        $addonArchive = Get-InstallArtifact $plan.dual_addon (Join-Path $root 'downloads')
        $addonRoot = Join-Path $root ('dual-' + $plan.dual_addon.sha256.Substring(0,12))
        $addonRoot = Expand-InstallPackage $addonArchive $addonRoot $plan.dual_addon
        $plan | Add-Member NoteProperty dual_addon_root $addonRoot
    }
    if (-not $RuntimeOnly) { foreach ($record in $records) { Get-InstallArtifact $record $modelRoot | Out-Null } }
    $script = Write-InstallLauncher $root $runtime $modelRoot $plan ([bool]$Vision) $Port
    Write-Host "Setup complete. Start: & $(Quote-InstallString $script)"
    Write-Host "Check the launcher: & $(Quote-InstallString $script) -DryRun"
    Write-Host "API: http://127.0.0.1:$Port/v1"
    if ($RuntimeOnly) { Write-Host 'Runtime-only setup: supply the pinned models or rerun without -RuntimeOnly before starting.' }
}

if ($MyInvocation.InvocationName -ne '.') { Invoke-L0xreInstall }
