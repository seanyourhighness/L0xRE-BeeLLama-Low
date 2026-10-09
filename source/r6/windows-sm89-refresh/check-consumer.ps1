$ErrorActionPreference='Stop'
$base='C:\work\l0xre-sm89-r6-20261009'
. '\\wsl.localhost\Ubuntu\home\sean\work\l0xre-universal-r6-20261009\install\install.ps1'
$record=Get-Content -Raw (Join-Path $base 'SM89-ARTIFACT.json') | ConvertFrom-Json
$record | Add-Member NoteProperty launcher 'tools/universal/r6-launch.ps1'
$archive=Join-Path $base $record.filename
Test-InstallArtifact $archive $record
$consumer=Expand-InstallPackage $archive (Join-Path $base 'consumer') $record
Test-InstallPackage $consumer
$global:LASTEXITCODE=0
function nvidia-smi { $global:LASTEXITCODE=0; 'GPU-offline-fixture, 8.9' }
$env:CUDA_VISIBLE_DEVICES='GPU-offline-fixture'
$target='C:\work\l0xre-sm120-certified-windows-20261008\models\L0xRE-27b-Low.gguf'
$draft='C:\work\l0xre-sm120-certified-windows-20261008\models\Qwen3.8-27B-DFlash2-Q4_K_M.gguf'
$launcher=Join-Path $consumer 'tools/universal/r6-launch.ps1'
$rejected=$false
try { & $launcher serve -m $target -md $draft --dry-run | Out-Null }
catch { if ($_.Exception.Message -notmatch 'qualification is pending') { throw }; $rejected=$true }
if (-not $rejected) { throw 'Candidate gate missing' }
$result=& $launcher serve --qualification-probe -m $target -md $draft --dry-run | ConvertFrom-Json
if ($result.hardware_qualified -or $result.uuid -ne 'GPU-offline-fixture' -or $result.argv -notcontains '81920' -or $result.env.L0XRE_INT8_PREFILL_CACHE_DOWN -ne '0') { throw 'Candidate profile incorrect' }
@{status='pass';hardware_qualified=$false;gpu_inference_run=$false;gpu_detection='mock SM89 identity';candidate_gate_rejected_without_opt_in=$rejected;resolved_profile=$result} | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 (Join-Path $base 'CONSUMER-DRYRUN.json')
Write-Host 'SM89_CLEAN_CONSUMER_PASS'
