$ErrorActionPreference = "Stop"
$RootPath = $PSScriptRoot
$ManifestPath = Join-Path $RootPath "SHA256SUMS"
$count = 0
foreach ($line in [IO.File]::ReadAllLines($ManifestPath)) {
    if (-not $line) { continue }
    if ($line -notmatch '^([a-f0-9]{64})  (.+)$') { throw "Invalid checksum line: $line" }
    $expected = $Matches[1]; $relative = $Matches[2]
    $path = [IO.Path]::GetFullPath((Join-Path $RootPath $relative))
    if (-not $path.StartsWith($RootPath + '\', [StringComparison]::OrdinalIgnoreCase)) { throw "Unsafe manifest path: $relative" }
    if (-not [IO.File]::Exists($path)) { throw "Missing file: $relative" }
    if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected) { throw "Checksum mismatch: $relative" }
    $count++
}
Write-Host "Verified $count package files."
