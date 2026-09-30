# Integration tests with a stub executable; no GPU execution or performance claim.
$ErrorActionPreference = 'Stop'
$LauncherSource = Join-Path $PSScriptRoot 'l0xre.ps1'
$TestRoot = Join-Path ([IO.Path]::GetTempPath()) ('l0xre-launcher-' + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path "$TestRoot\bin" -Force | Out-Null
Copy-Item $LauncherSource "$TestRoot\l0xre.ps1"
$code = @'
using System;
public class LauncherProbe {
 public static void Main(string[] args) {
  foreach (var a in args) Console.WriteLine("ARG:" + a);
  Console.WriteLine("HEAD:" + Environment.GetEnvironmentVariable("ESCHA_E3_HEAD_RT_BLOCK128"));
  Console.WriteLine("VECTOR:" + Environment.GetEnvironmentVariable("L0XRE_K3_VECTOR_CUBIN"));
  Console.WriteLine("BRIDGE:" + Environment.GetEnvironmentVariable("ESCHA_OFFICIAL_BRIDGE_LIBRARY"));
 }
}
'@
Add-Type -TypeDefinition $code -OutputAssembly "$TestRoot\bin\llama-server.exe" -OutputType ConsoleApplication
foreach ($arch in @('sm86','sm89','sm120')) {
 New-Item -ItemType Directory -Path "$TestRoot\bridge\$arch" -Force | Out-Null
 [IO.File]::WriteAllText("$TestRoot\bridge\$arch\code-gemm.cubin",'stub')
}
foreach ($file in @('escha_official_bridge_cuda.dll','escha_gdn_chunk_bridge.dll','sm86\bridge-b74-k3-vector.dll','sm86\k3-vector-all.cubin')) { [IO.File]::WriteAllText("$TestRoot\bridge\$file",'stub') }
function Run-Case([string]$Arch,[string[]]$Params,[int]$Expected=0) {
 $env:L0XRE_ARCH=$Arch
 $env:ESCHA_E3_HEAD_RT_BLOCK128='1';$env:L0XRE_K3_VECTOR_CUBIN='wrong-sm86.cubin'
 $saved=$ErrorActionPreference;$ErrorActionPreference='Continue'
 $lines = & powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$TestRoot\l0xre.ps1" @Params 2>&1
 $rc=$LASTEXITCODE;$ErrorActionPreference=$saved
 if ($rc -ne $Expected) { throw "Expected $Expected, got $rc for $Arch / $Params : $lines" }
 return @($lines | ForEach-Object { $_.ToString() })
}
function Arg-Value($Lines,[string]$Name) {
 $arguments=@($Lines | Where-Object { $_.StartsWith('ARG:') } | ForEach-Object { $_.Substring(4) })
 $index=[Array]::IndexOf($arguments,$Name)
 if ($index -lt 0) { throw "Missing argument $Name" }
 return $arguments[$index+1]
}
$count=0
try {
 $params=@('serve','--profile','12gb','-m','model with spaces.gguf','-md','draft with spaces.gguf','--port','9099')
 $lines=Run-Case 'sm86' $params
 if ((Arg-Value $lines '--alias') -ne 'L0xRE-27b-Low' -or (Arg-Value $lines '-c') -ne '98304' -or (Arg-Value $lines '-ub') -ne '256' -or (Arg-Value $lines '--spec-draft-type-k') -ne 'q2_0' -or 'HEAD:1' -notin $lines -or (Arg-Value $lines '-m') -ne 'model with spaces.gguf') { throw 'SM86 B84 routing failed' };$count++
 foreach ($arch in @('sm89','sm120')) {
  $lines=Run-Case $arch $params
  if ((Arg-Value $lines '-c') -ne '81920' -or 'HEAD:' -notin $lines -or 'VECTOR:' -notin $lines) { throw "$arch profile/isolation failed" };$count++
 }
 $lines=Run-Case 'sm89' @('serve','--profile','16gb','-m','model.gguf')
 if ((Arg-Value $lines '-c') -ne '262144') { throw '16gb routing failed' };$count++
 Run-Case 'sm86' @('serve','--profile') 1 | Out-Null;$count++
 Run-Case 'sm86' @('serve','--profile','wrong') 1 | Out-Null;$count++
 Run-Case 'sm89' @('serve','--profile','12gb-b84') 1 | Out-Null;$count++
 Run-Case 'sm80' $params 1 | Out-Null;$count++
 Run-Case 'sm86' @('--help') | Out-Null;$count++
 Write-Host "PASS: $count Windows launcher integration cases (stub executable; no GPU execution)."
} finally { Remove-Item $TestRoot -Recurse -Force }
