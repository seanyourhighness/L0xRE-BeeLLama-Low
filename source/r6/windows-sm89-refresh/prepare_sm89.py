from pathlib import Path
import hashlib,json,shutil,zipfile
out=Path('/mnt/c/work/l0xre-sm89-r6-20261009')
archive=Path('/mnt/d/CODEX WORKSPACE/L0xRE/release-artifacts/l0xre-beellama-low-v0.4.7-r6-windows-sm120-source-inputs.zip')
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
assert sha(archive)=='91bcffbcc55a396cbbfed41ffec257885324d2814e5cd5c54ba669b53443478a'
assert not out.exists()
out.mkdir()
with zipfile.ZipFile(archive) as z:
 for i in z.infolist():
  if i.filename.startswith(('control-source/','experiment-source/','companions/')):z.extract(i,out/'published-inputs')
inputs=out/'published-inputs'
shutil.copytree(inputs/'control-source',out/'source')
shutil.copytree(inputs/'experiment-source/bridge',out/'bridge')
for name in ['ggml-cuda.cu','escha-moe.cu','l0xre-startup-env.h']:
 shutil.copy2(inputs/'experiment-source/source'/name,out/'source/ggml/src/ggml-cuda'/name)
# Preserve the published bridge's relative include layout.
shutil.copy2(inputs/'experiment-source/source/l0xre-startup-env.h',out/'source/l0xre-startup-env.h')
changes=[]
for name,old,new in [('gdn-compact.cuh','ggml_cuda_info().devices[ctx.device].cc==860 ||','ggml_cuda_info().devices[ctx.device].cc==860 || ggml_cuda_info().devices[ctx.device].cc==890 ||'),('gated_delta_net.cu','ggml_cuda_info().devices[ctx.device].cc == 860 ||','ggml_cuda_info().devices[ctx.device].cc == 860 || ggml_cuda_info().devices[ctx.device].cc == 890 ||')]:
 p=out/'source/ggml/src/ggml-cuda'/name;s=p.read_text();assert s.count(old)==1;before=sha(p);p.write_text(s.replace(old,new));changes.append({'file':str(p.relative_to(out)),'before':before,'after':sha(p),'change':'Retain Windows port2 SM89 masked-GDN eligibility with exact native rollback tail.'})
# CUDA13.0 toolchain, same portable CPU/shared/backend/graph/default-quant flags.
s=Path('/mnt/d/CODEX WORKSPACE/L0xRE/build-cuda130-experiment.cmd').read_text()
s=s[:s.index('cd /d "C:\\work\\l0xre-sm120-cuda130-windows-20261008\\companions"')]
s=s.replace('C:\\work\\l0xre-sm120-cuda130-windows-20261008\\source','C:\\work\\l0xre-sm89-r6-20261009\\source').replace('C:\\work\\l0xre-sm120-cuda130-windows-20261008\\build','C:\\work\\l0xre-sm89-r6-20261009\\build').replace('C:\\work\\l0xre-sm120-cuda130-windows-20261008\\bridge','C:\\work\\l0xre-sm89-r6-20261009\\bridge')
s=s.replace('CMAKE_CUDA_ARCHITECTURES=120','CMAKE_CUDA_ARCHITECTURES=89').replace('-gencode=arch=compute_120,code=sm_120 -gencode=arch=compute_120a,code=sm_120a','-gencode=arch=compute_89,code=sm_89')
s+='exit /b %errorlevel%\n'
(out/'build.cmd').write_text(s)
receipt={'status':'prepared','hardware_qualified':False,'source_archive_sha256':sha(archive),'source_release':'beellama-v0.4.7-r6-sm120-windows-cuda130','arch':'sm89','changes':changes,'profile_note':'12GB: down-weight cache disabled (5.7GB allocation); preserve SM89 down12/QK16/chunk16384 pending hardware testing. SM120-only fence remains architecture gated.','source_sha256':{str(p.relative_to(out)):sha(p) for d in ['source/ggml/src/ggml-cuda','bridge'] for p in (out/d).rglob('*') if p.is_file() and p.suffix in ['.cu','.cuh','.h','.def']}}
(out/'BUILD-PLAN.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(out, 'prepared',len(receipt['source_sha256']))
