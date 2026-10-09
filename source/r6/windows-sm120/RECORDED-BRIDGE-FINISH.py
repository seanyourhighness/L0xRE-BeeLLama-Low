import hashlib,json,subprocess,time
from pathlib import Path
root=Path(r'C:\work\l0xre-sm120-envcache-windows-20261008')
r=json.loads((root/'BUILD-PROVENANCE.json').read_text())
env_output=subprocess.check_output('cmd /d /s /c "call D:\\VS\\BuildTools2022\\VC\\Auxiliary\\Build\\vcvars64.bat >nul && set"',text=True)
env=dict(line.split('=',1) for line in env_output.splitlines() if '=' in line and not line.startswith('='))
with (root/'bridge-retry.compile.log').open('w') as log:p=subprocess.run(r['bridge_command'],cwd=root/'bridge',env=env,stdout=log,stderr=subprocess.STDOUT)
assert p.returncode==0
r['initial_attempt']={'status':r['status'],'error':r.get('error')};r.pop('error',None)
r['status']='built';r['finished']=time.time()
r['binary_hashes']={s:hashlib.sha256((root/'package'/s).read_bytes()).hexdigest() for s in ['bin/ggml-cuda.dll','bridge/bridge-packed.dll']}
r['source_hashes']={s:hashlib.sha256((root/s).read_bytes()).hexdigest() for s in ['source/ggml-cuda.cu','source/escha-moe.cu','source/l0xre-startup-env.h','bridge/bridge-packed.cu']}
(root/'BUILD-PROVENANCE.json').write_text(json.dumps(r,indent=2));print(json.dumps({'status':r['status'],'binary_hashes':r['binary_hashes']}))
