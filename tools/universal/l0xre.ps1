# Universal Windows runtime: NVIDIA SM86 / SM89 / SM120.
param([Parameter(ValueFromRemainingArguments=$true)][string[]]$Rest)
$ErrorActionPreference = "Stop"
$ROOT = $PSScriptRoot
if (-not $Rest -or $Rest[0] -in @("--help", "-h", "help")) {
    Write-Host "L0xRE BeeLLama universal Windows runtime"
    Write-Host '.\l0xre.cmd serve --profile 12gb -m models\L0xRE-27b-Low.gguf -md models\Qwen3.8-27B-DFlash2-Q4_K_M.gguf'
    Write-Host "SM86 12gb / 12gb-quality: 80K, KVarN4/4, INT8 + gated FP16 prefill; other GPUs retain KVarN3/3, draft Q4_0/Q4_0, 32 workers, medium reasoning, CPU projector."
    Write-Host "12gb-b84 is a compatibility alias on SM86. Other profiles: 16gb, full32k."
    Write-Host "See README.md for model downloads, verification, and hardware test scope."
    exit 0
}
$cc = $env:L0XRE_ARCH
if (-not $cc) {
    $query = @("--query-gpu=compute_cap", "--format=csv,noheader")
    if (Test-Path Env:CUDA_VISIBLE_DEVICES) {
        $selected = ($env:CUDA_VISIBLE_DEVICES -split ',')[0]
        if (-not $selected -or $selected -eq "-1") { throw "CUDA_VISIBLE_DEVICES exposes no GPU." }
        $query += @("-i", $selected)
    }
    $cc = (& nvidia-smi @query 2>$null | Select-Object -First 1)
    if ($LASTEXITCODE -ne 0 -or -not $cc) { throw "GPU detection failed. Check the NVIDIA driver and nvidia-smi." }
}
$ccI = [int]($cc -replace '[^0-9]', '')
if ($ccI -notin @(86,89,120)) { throw "Unsupported compute capability: $cc (supported: SM86, SM89, SM120)." }
$ARCH = "sm$ccI"
$BRIDGE = Join-Path $ROOT "bridge"
$env:PATH = "$ROOT\bin;$env:PATH"
$env:ESCHA_BRIDGE_ROOT = $BRIDGE
$env:ESCHA_OFFICIAL_RAW_BRIDGE = "1"
$env:ESCHA_OFFICIAL_RAW_SWIGLU_FUSION = "1"
$env:ESCHA_OFFICIAL_RAW_DECODE_BRIDGE = "1"
$env:ESCHA_OFFICIAL_RAW_DECODE_F32_INPUT = "1"
$env:ESCHA_OFFICIAL_RAW_DECODE_SWIGLU_FUSION = "1"
$env:ESCHA_OFFICIAL_RAW_DECODE_K3_SPLIT_DIV2 = "0"
$env:ESCHA_OFFICIAL_BRIDGE_ACCUMULATION = "mixed"
$env:ESCHA_OFFICIAL_CODE_GEMM_CUBIN = "$BRIDGE\$ARCH\code-gemm.cubin"
$env:ESCHA_OFFICIAL_F32_DECODE_CUBIN = "$BRIDGE\$ARCH\code-gemm.cubin"
$env:ESCHA_OFFICIAL_BRIDGE_LIBRARY = "$BRIDGE\escha_official_bridge_cuda.dll"
$env:ESCHA_GDN_CHUNK_BRIDGE = "1"
$env:ESCHA_GDN_CHUNK_BRIDGE_LIBRARY = "$BRIDGE\escha_gdn_chunk_bridge.dll"
$env:ESCHA_GDN_CHUNK_CUBIN_ROOT = "$BRIDGE\$ARCH\gdn-cubins"
$env:ESCHA_E3_EMBED_GPU = "1"
$env:ESCHA_OFFICIAL_RAW_DECODE_DOWN_ADD_RMS_FUSION = "1"
$env:ESCHA_OFFICIAL_RAW_DECODE_K3_DOWN_SPLITS = "12"
$env:ESCHA_OFFICIAL_RAW_DECODE_QKV_Z_BETA_ALPHA_OVERLAP = "1"
$env:ESCHA_DIRECT_RECURRENT_STATE_R258 = "1"
$env:ESCHA_DIRECT_RECURRENT_CONV_R260 = "1"
$env:ESCHA_FUSE_GDN_DECODE_PREP_R262 = "1"
$env:ESCHA_FUSE_GDN_NORM_GATE_R264 = "1"
$env:ESCHA_W2_I8_HEAD = "0"
$env:ESCHA_E3_HEAD_MROW = "8"
$env:ESCHA_E3_HEAD_RT = "2"
$env:ESCHA_F16_SMALL_N_NATIVE = "1"
# Prevent SM86-only choices from leaking into other architectures.
Remove-Item Env:ESCHA_E3_HEAD_RT_BLOCK128,Env:L0XRE_K3_VECTOR_CUBIN,Env:GGML_KVARN_WINDOW_CHUNK,Env:L0XRE_INT8_PREFILL,Env:L0XRE_INT8_PROJ_MASK,Env:L0XRE_KVARN_QK_FP16_ACC,Env:L0XRE_KVARN_PREFILL_HEAD_TILE,Env:L0XRE_GDN_MASKED_PREFILL -ErrorAction SilentlyContinue
if ($ARCH -eq "sm86") {
    $env:ESCHA_E3_HEAD_RT_BLOCK128 = "1"
    $env:ESCHA_OFFICIAL_BRIDGE_LIBRARY = "$BRIDGE\sm86\bridge-int8-allproj.dll"
    $env:L0XRE_K3_VECTOR_CUBIN = "$BRIDGE\sm86\k3-vector-all.cubin"
}
$Profile = "12gb"
$HaveDraft = $false
$ServerArgs = @()
$i = 0
while ($i -lt $Rest.Count) {
    $a = $Rest[$i]
    if ($a -eq "serve") { $i++; continue }
    if ($a -eq "--profile") {
        if ($i+1 -ge $Rest.Count) { throw "Missing value for --profile." }
        $Profile = $Rest[$i+1]; $i += 2; continue
    }
    if ($a -in @("-md","--draft-model","--spec-draft-model")) { $HaveDraft = $true }
    $ServerArgs += $a; $i++
}
if ($Profile -eq "12gb-b84" -and $ARCH -ne "sm86") { throw "12gb-b84 is supported only on SM86." }
if ($Profile -in @("12gb", "12gb-quality", "12gb-b84")) { $Profile = "12gb-quality" }
$Placement = @(); $DraftUB = "128"; $DraftNGL = "99"
$Ctx = @(); $NMax = 3; $DraftKV = "q4_0"; $Fit = @(); $Vision = @()
$Cpu = @("-t", "8"); $Effort = "low"
switch ($Profile) {
    "12gb-quality" {
        $env:GGML_KVARN_WINDOW_CHUNK = "16384"
        $Ctx = @("-c","81920","-b","1024","-ub","256","-ctk","kvarn3","-ctv","kvarn3","--kv-tail-tokens","128")
        $Cpu = @("-t", "32", "-tb", "32"); $Effort = "medium"
        if ($ARCH -eq "sm86") {
            $Ctx = @("-c","81920","-b","1024","-ub","512","-ctk","kvarn4","-ctv","kvarn4","--kv-tail-tokens","128")
            $Placement = @("-ot","token_embd.lowgpu_.*=CPU","--no-op-offload","--fit-target","768")
            $DraftUB = "32"; $DraftNGL = "99"
            $env:L0XRE_INT8_PREFILL = "1"; $env:L0XRE_INT8_PROJ_MASK = "255"
            $env:L0XRE_KVARN_QK_FP16_ACC = "2"
            $env:L0XRE_KVARN_PREFILL_HEAD_TILE = "0"; $env:L0XRE_GDN_MASKED_PREFILL = "0"
        }
        $Fit = @("--cache-ram","0","--fit","off")
        $Vision = @("--no-mmproj-offload", "--image-min-tokens", "1024", "--image-max-tokens", "1024")
    }
    "16gb" {
        $env:GGML_KVARN_WINDOW_CHUNK = "16384"
        $Ctx = @("-c","262144","-b","1024","-ub","512","-ctk","kvarn3","-ctv","kvarn3")
    }
    "full32k" { $Ctx = @("-c","32768","-b","2048","-ub","1024","-ctk","kvarn3","-ctv","kvarn2"); $NMax = 5 }
    default { throw "Unknown profile: $Profile." }
}
$Spec = @()
if ($HaveDraft) {
    $Spec = @("--spec-type","draft-dflash","--spec-draft-n-max","$NMax","--spec-draft-ngl",$DraftNGL,"--spec-draft-ubatch-size",$DraftUB,"--spec-draft-type-k",$DraftKV,"--spec-draft-type-v",$DraftKV)
}
$required = @("$ROOT\bin\llama-server.exe",$env:ESCHA_OFFICIAL_BRIDGE_LIBRARY,$env:ESCHA_OFFICIAL_CODE_GEMM_CUBIN,$env:ESCHA_GDN_CHUNK_BRIDGE_LIBRARY)
if ($ARCH -eq "sm86") { $required += $env:L0XRE_K3_VECTOR_CUBIN }
foreach ($file in $required) { if (-not (Test-Path -LiteralPath $file -PathType Leaf)) { throw "Package component missing: $file. Re-extract the complete ZIP." } }
Write-Host "L0xRE: $ARCH / $Profile" -ForegroundColor DarkGray
& "$ROOT\bin\llama-server.exe" --alias L0xRE-27b-Low -np 1 @Cpu -ngl 99 -fa on `
    --jinja --reasoning on --reasoning-effort $Effort --reasoning-budget 8192 `
    --temp 0.7 --top-p 0.95 --top-k 20 --no-webui @Ctx @Spec @Fit @Vision @Placement @ServerArgs
exit $LASTEXITCODE
