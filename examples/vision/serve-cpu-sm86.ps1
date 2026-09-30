# Matching Windows recipe; argument example only, not GPU-qualified on Windows.
param(
    [string]$RuntimeDirectory = (Get-Location).Path,
    [string]$ModelsDirectory = "",
    [ValidateRange(1, 1024)][int]$Workers = 32,
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$ServerOptions
)
$ErrorActionPreference = "Stop"
if (-not $ModelsDirectory) { $ModelsDirectory = Join-Path $RuntimeDirectory "models" }
$VisionLauncher = Join-Path $RuntimeDirectory "l0xre.ps1"
if (-not (Test-Path -LiteralPath $VisionLauncher -PathType Leaf)) {
    throw "Set -RuntimeDirectory to the extracted universal runtime directory."
}
foreach ($VisionFile in @("L0xRE-27b-Low.gguf", "Qwen3.8-27B-DFlash2-Q4_K_M.gguf", "mmproj-Qwen3.8-27B-Q8_0.gguf")) {
    if (-not (Test-Path -LiteralPath (Join-Path $ModelsDirectory $VisionFile) -PathType Leaf)) {
        throw "Missing model: $VisionFile in $ModelsDirectory"
    }
}
$VisionArguments = @(
    "serve", "--profile", "12gb-b84",
    "-m", (Join-Path $ModelsDirectory "L0xRE-27b-Low.gguf"),
    "-md", (Join-Path $ModelsDirectory "Qwen3.8-27B-DFlash2-Q4_K_M.gguf"),
    "--mmproj", (Join-Path $ModelsDirectory "mmproj-Qwen3.8-27B-Q8_0.gguf"),
    "--no-mmproj-offload", "--image-min-tokens", "1024", "--image-max-tokens", "1024",
    "-t", "$Workers", "-tb", "$Workers", "--reasoning-effort", "medium",
    "--host", "127.0.0.1", "--port", "8080"
)
& $VisionLauncher @VisionArguments @ServerOptions
exit $LASTEXITCODE
