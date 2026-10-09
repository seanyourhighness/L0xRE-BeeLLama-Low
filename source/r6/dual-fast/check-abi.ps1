$ErrorActionPreference='Stop'
$env:L0XRE_INT8_PREFILL='0'
$env:PATH='C:\work\l0xre-sm120-cuda130-windows-20261008\package\bin;'+$env:PATH
Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class DualAbi {
 [DllImport("kernel32",SetLastError=true,CharSet=CharSet.Unicode)] public static extern IntPtr LoadLibrary(string path);
 [DllImport("kernel32",SetLastError=true)] public static extern IntPtr GetProcAddress(IntPtr module,string name);
 [DllImport("kernel32")] public static extern bool FreeLibrary(IntPtr module);
 [UnmanagedFunctionPointer(CallingConvention.Cdecl)] public delegate int Abi();
 [UnmanagedFunctionPointer(CallingConvention.Cdecl)] public delegate int Mask(IntPtr stream,int device,IntPtr q,IntPtr k,IntPtr v,IntPtr g,IntPtr b,IntPtr state,IntPtr dst,IntPtr outstate,long S,long H,long T,long N,long sq1,long sq2,long sv1,long sv2,long sb1,long sb2,float scale,long valid);
 public static void Check(string bridge,string gdn) {
  IntPtr a=LoadLibrary(bridge),b=LoadLibrary(gdn);
  if(a==IntPtr.Zero || b==IntPtr.Zero)throw new Exception("DLL load error "+Marshal.GetLastWin32Error());
  var abi=(Abi)Marshal.GetDelegateForFunctionPointer(GetProcAddress(a,"escha_official_bridge_abi_version"),typeof(Abi));
  if(abi()!=1)throw new Exception("ABI mismatch");
  if(GetProcAddress(a,"l0xre_int8_prefill_init_v1")==IntPtr.Zero)throw new Exception("Missing explicit initializer");
  var mask=(Mask)Marshal.GetDelegateForFunctionPointer(GetProcAddress(b,"escha_gdn_chunk_prefill_masked_v1"),typeof(Mask));
  foreach(int device in new int[]{-1,0,1,64}) {
   if(mask(IntPtr.Zero,device,IntPtr.Zero,IntPtr.Zero,IntPtr.Zero,IntPtr.Zero,IntPtr.Zero,IntPtr.Zero,IntPtr.Zero,IntPtr.Zero,128,48,512,1,128,6144,128,6144,1,48,1.0f,512)!=1)throw new Exception("Guard mismatch");
  }
  FreeLibrary(b);FreeLibrary(a);
 }
}
'@
$checks=@()
foreach ($arch in @('sm86','sm89','sm120')) {
 [DualAbi]::Check((Join-Path $PSScriptRoot 'bridge\bridge-dual.dll'),(Join-Path $PSScriptRoot ($arch+'\gdn-dual.dll')))
 $checks+=@{arch=$arch;abi=1;null_and_invalid_device_guards='pass';gpu_work_launched=$false}
}
$checks | ConvertTo-Json -Depth 5
