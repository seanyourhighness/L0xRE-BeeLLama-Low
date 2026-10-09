from pathlib import Path
import hashlib,json,shutil,zipfile,subprocess,time,re
BASE=Path(__file__).resolve().parent
BUILD=Path('/mnt/c/work/l0xre-sm89-r6-20261009')
OLD=Path('/home/sean/work/l0xre-r6-windows-prefill-port2-20261007/package-sm89')
CERT=BASE/'package-sm120-12gb-windows/package'
OUT=BASE/'package-sm89-windows'
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
assert not OUT.exists();OUT.mkdir()
shutil.copytree(CERT/'bin',OUT/'bin')
shutil.copytree(OLD/'bridge',OUT/'bridge')
(OUT/'tools/universal').mkdir(parents=True)
for name in ['r6-launch.ps1','r6-windows-profile.json']:shutil.copy2(OLD/'tools/universal'/name,OUT/'tools/universal'/name)
shutil.copy2(OLD/'l0xre-r6.cmd',OUT/'l0xre-r6.cmd')
p=OUT/'tools/universal/r6-windows-profile.json';v=json.loads(p.read_text());v['env'].update(L0XRE_INT8_PREFILL_CACHE_DOWN='0',L0XRE_INT8_PREFILL_SYNC='1',L0XRE_INT8_PREFILL_SYNC_INTERVAL='8',CUDA_LAUNCH_BLOCKING='0');v['windows_scope']='SM89 R6 refresh: published SM120 startup-env fixes and INT8 code; retained SM89 eligibility, down12, QK16, 16K prefill chunks. Cache-down disabled for12GB. SM120-specific sync fence stays architecture-gated. RTX4070Ti hardware qualification pending.';p.write_text(json.dumps(v,indent=2)+'\n')
p=OUT/'tools/universal/r6-launch.ps1';s=p.read_text().replace('$uuid = $fields[0]','$uuid = $fields[0]\n$env:CUDA_VISIBLE_DEVICES = $uuid').replace('$env:PATH =',"$env:GGML_BACKEND_PATH = Join-Path $ROOT 'bin'\n$env:PATH =");s=s.replace("'^(L0XRE_|ESCHA_|GGML_)'", "'^(L0XRE_|ESCHA_|GGML_|LLAMA_ARG_)'").replace("& $server @argv", "[System.Diagnostics.Process]::GetCurrentProcess().ProcessorAffinity = [IntPtr]255\n& $server @argv");p.write_text(s)
(OUT/'README.md').write_text('''# R6 SM89 Windows refresh candidate

Built from the exact published certified SM120 source inputs, including the startup environment cache and INT8 fixes. SM89 masked-GDN eligibility and its existing12GB profile are retained. No5.7GB down-weight cache. CUDA13.0, MSVC portable CPU build, one80K slot, KVarN4/4, DFlash2Q4 N7, medium reasoning.

**UNTESTED / UNCERTIFIED on RTX4070Ti.** This archive is ready for hardware testing after the included offline build/import/ABI/hash checks. The5090/3060 certificates do not certify this binary. No new speed, quality, VRAM capacity or stability claims.

Reuse the exact models from an earlier release (filenames may differ; SHA256 must match). From this directory:

```powershell
.\\l0xre-r6.cmd serve --qualification-probe -m D:\\models\\L0xRE-27b-Low.gguf -md D:\\models\\Qwen3.8-27B-DFlash2-Q4_K_M.gguf --dry-run
.\\l0xre-r6.cmd serve --qualification-probe -m D:\\models\\L0xRE-27b-Low.gguf -md D:\\models\\Qwen3.8-27B-DFlash2-Q4_K_M.gguf
```

Use the same canonical bench.sh prompts/sampler/warmups/repeats for speed comparisons. Record actual loaded DLL hashes, GPU/driver/OS, per-card VRAM, route markers and errors. Then run matched quality, CPU vision, staged context/capacity and a30-minute soak plus fresh restart. Preserve failures. The SM89 Q4 MMQ route remains a separate opt-in (`--sm89-q4-mmq`); do not mix measurements across profiles.

The exact published base source archive SHA is in BUILD-PLAN.json. The separate SM89 source-input archive overlays the recorded certified two-TU/header/bridge source and the two SM89 GDN gate changes. Certified SM86/SM120 packages remain unchanged.
''')
print('WAITING_FOR_NATIVE_BUILD',flush=True)
while not (BUILD/'BUILD-RECEIPT.json').exists():time.sleep(5)
r=json.loads((BUILD/'BUILD-RECEIPT.json').read_text());assert r['exit_code']==0,r
for p in (BUILD/'build/bin').iterdir():
 if p.is_file() and p.suffix in ['.exe','.dll']:shutil.copy2(p,OUT/'bin'/p.name)
shutil.copy2(BUILD/'bridge/bridge-packed.dll',OUT/'bridge/sm89/escha_r6_packed_bridge.dll')
shutil.copy2(BUILD/'companions/escha_r6_gdn_masked.dll',OUT/'bridge/sm89/escha_r6_gdn_masked.dll')
(OUT/'evidence').mkdir()
for n in ['BUILD-PLAN.json','BUILD-RECEIPT.json','build.cmd']:shutil.copy2(BUILD/n,OUT/'evidence'/n)
files=sorted(p for d in ['bin','bridge'] for p in (OUT/d).rglob('*') if p.is_file() and p.suffix in ['.dll','.exe'])
system={p.name.lower() for p in Path('/mnt/c/Windows/System32').glob('*.dll')};bundled={p.name.lower() for p in files}
imports=0
for p in files:
 raw=subprocess.check_output(['objdump','-p',str(p)],text=True)
 for dep in re.findall(r'DLL Name:\s*(\S+)',raw):
  assert dep.lower() in bundled|system or dep.lower().startswith(('api-ms-','ext-ms-')),(p.name,dep)
  imports+=1
runtime={str(p.relative_to(OUT)):sha(p) for d in ['bin','bridge','tools'] for p in (OUT/d).rglob('*') if p.is_file()}
m=OUT/'architectures/sm89';m.mkdir(parents=True)
(m/'R6-MANIFEST.json').write_text(json.dumps({'arch':'sm89','built':True,'hardware_qualified':False,'status':'windows-r6-sm120-fixes-4070ti-test-candidate','runtime_sha256':runtime},indent=2)+'\n')
(OUT/'evidence/OFFLINE-CHECKS.json').write_text(json.dumps({'status':'pass','hardware_qualified':False,'pe_files':len(files),'imports_checked':imports,'runtime_hashes':len(runtime)},indent=2)+'\n')
(OUT/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(OUT)}\n' for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='SHA256SUMS'))
a=BASE/'L0xRE-BeeLLama-Low-R6-sm89-windows-refresh-candidate.zip'
with zipfile.ZipFile(a,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in sorted(OUT.rglob('*')):
  if p.is_file():z.write(p,str(p.relative_to(OUT)))
with zipfile.ZipFile(a) as z:
 assert z.testzip() is None
 for line in z.read('SHA256SUMS').decode().splitlines():
  digest,n=line.split('  ',1)
  with z.open(n) as f:assert hashlib.file_digest(f,'sha256').hexdigest()==digest,n
h=sha(a);a.with_name(a.name+'.sha256').write_text(h+'  '+a.name+'\n')
receipt={'filename':a.name,'bytes':a.stat().st_size,'sha256':h,'hardware_qualified':False,'archive_hashes_pass':True}
(BASE/'SM89-ARTIFACT.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
