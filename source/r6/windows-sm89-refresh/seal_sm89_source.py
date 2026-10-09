from pathlib import Path
import hashlib,json,zipfile
B=Path(__file__).resolve().parent;W=Path('/mnt/c/work/l0xre-sm89-r6-20261009')
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
files={}
for d in ['source','bridge','companions']:
 for p in (W/d).rglob('*'):
  if not p.is_file():continue
  if d!='source' and p.suffix not in ['.cu','.cuh','.h','.def','.cpp','.cmd']:continue
  if p.suffix in ['.dll','.obj','.lib','.exp','.exe','.pdb','.log']:continue
  files[str(p.relative_to(W))]=p
for n in ['BUILD-PLAN.json','build.cmd']:files[n]=W/n
hashes=''.join(f'{sha(p)}  {n}\n' for n,p in sorted(files.items()))
a=B/'L0xRE-BeeLLama-Low-R6-sm89-windows-refresh-source-inputs.zip'
with zipfile.ZipFile(a,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for n,p in sorted(files.items()):z.write(p,n)
 z.writestr('SOURCE-SHA256SUMS',hashes)
 z.writestr('README.txt','Exact prepared source inputs for the SM89 Windows R6 refresh candidate. Source starts from the published certified SM120 source archive identified in BUILD-PLAN.json, overlays its final startup-env source, and enables the retained SM89 masked-GDN gates. CUDA13.0 native Windows build; adapt paths in build.cmd. This source archive does not establish hardware qualification.\n')
with zipfile.ZipFile(a) as z:
 assert z.testzip() is None
 for line in z.read('SOURCE-SHA256SUMS').decode().splitlines():
  h,n=line.split('  ',1)
  with z.open(n) as f:assert hashlib.file_digest(f,'sha256').hexdigest()==h,n
v={'filename':a.name,'bytes':a.stat().st_size,'sha256':sha(a),'source_files':len(files),'source_checksums_pass':True,'hardware_qualified':False}
a.with_name(a.name+'.sha256').write_text(v['sha256']+'  '+a.name+'\n');(B/'SM89-SOURCE-ARTIFACT.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))
