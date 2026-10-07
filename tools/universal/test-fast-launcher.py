#!/usr/bin/env python3
"""Exercise fast profile parity, integrity and fail-closed routing with fixtures.

These launcher tests do not execute CUDA or certify cross-built runtimes.
"""
import hashlib,json,os,shutil,subprocess,tempfile
from pathlib import Path
S=Path(__file__).resolve().parent
count=0
with tempfile.TemporaryDirectory(prefix='l0xre-fast-launch-') as tmp:
    root=Path(tmp);(root/'tools/universal').mkdir(parents=True)
    for n in ['fast-launch.py','fast-profile.json']:shutil.copy2(S/n,root/'tools/universal'/n)
    shutil.copy2(S/'l0xre',root/'l0xre');(root/'l0xre').chmod(0o755)
    model=root/'target with spaces.gguf';model.write_text('test model')
    draft=root/'draft with spaces.gguf';draft.write_text('test draft')
    def manifest(arch,built=True,qualified=True):
        d=root/'architectures'/arch;(d/'bin').mkdir(parents=True,exist_ok=True)
        binary=d/'bin/llama-server'
        if not binary.exists():binary.write_text('stub')
        v={'arch':arch,'built':built,'hardware_qualified':qualified,'runtime_sha256':{'bin/llama-server':hashlib.sha256(binary.read_bytes()).hexdigest()}}
        (d/'FAST-MANIFEST.json').write_text(json.dumps(v))
    def run(arch,extra=None,status=0):
        global count
        cmd=[str(root/'l0xre'),'serve','--profile','fast','-m',str(model),'-md',str(draft),'--dry-run']+(extra or [])
        env=dict(os.environ,L0XRE_ARCH=arch,ESCHA_OFFICIAL_BRIDGE_LIBRARY='/bad.so',L0XRE_GDN_MASKED_LIBRARY='/bad.so',L0XRE_QUANTILE_SPEC='1',GGML_KVARN_WINDOW_CHUNK='9',LD_LIBRARY_PATH='/foreign')
        r=subprocess.run(cmd,env=env,text=True,capture_output=True);count+=1
        assert r.returncode==status,(arch,r.returncode,r.stderr)
        return json.loads(r.stdout) if status==0 else r.stderr
    for arch in ['sm86','sm89','sm120']:
        manifest(arch);d=run(arch)
        a=d['argv'];v=lambda k:a[a.index(k)+1]
        assert v('-c')=='81920' and v('-ub')=='512' and v('-ctk')==v('-ctv')=='kvarn4'
        assert v('--spec-draft-n-max')=='7' and v('-t')==v('-tb')=='8'
        assert v('--reasoning-budget')=='8192' and v('--min-p')=='0.05'
        assert '--reasoning-loop-min-tokens' not in a and '--reasoning-budget-message' not in a
        assert d['env']['L0XRE_REJECTION_SPEC']=='1' and d['env']['L0XRE_HEAD_NEXT']=='1'
        assert d['env']['L0XRE_GDN_MASKED_PREFILL']=='1'
        assert '/foreign' not in d['env']['LD_LIBRARY_PATH']
        assert '/bad.so' not in str(d['env']) and '/home/sean' not in str(d['env'])
        assert v('-m')==str(model) and v('-md')==str(draft)
        manifest(arch,built=False,qualified=False);assert 'not yet installed' in run(arch,status=2)
        manifest(arch,built=True,qualified=False);assert 'qualification is pending' in run(arch,status=2)
        run(arch,['--qualification-probe'])
        manifest(arch);(root/'architectures'/arch/'bin/llama-server').write_text('tampered')
        assert 'integrity check failed' in run(arch,status=2)
        manifest(arch);assert 'differs from the certified file' in run(arch,['--verify-models'],status=2)
    print('PASS:',count,'fast launcher checks; all three profile configurations equal, pending/tampered payloads rejected.')
