#!/usr/bin/env python3
"""Stage frozen qualified-candidate bytes; hardware qualification stays pending."""
import argparse,hashlib,json,shutil
from pathlib import Path
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);p.add_argument('--qualification',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();repo=Path(__file__).resolve().parents[2]
    assert not (a.output/'architectures/sm86/R6-MANIFEST.json').exists(),'Never overwrite a sealed staging directory'
    out=a.output.resolve();arch=out/'architectures/sm86';arch.mkdir(parents=True,exist_ok=True)
    frozen=json.loads((a.candidate/'FROZEN.json').read_text())
    for folder in ['bin','deps']:
        if not (arch/folder).exists():shutil.copytree(a.candidate/folder,arch/folder,symlinks=True,
            ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    excluded=[n for n in frozen['runtime_files'] if '__pycache__' in Path(n).parts or n.endswith('.pyc')]
    runtime={n:record['sha256'] for n,record in frozen['runtime_files'].items() if n not in excluded}
    for name,value in runtime.items():assert sha(arch/name)==value,name
    for name,target in frozen['loader_links'].items():assert (arch/name).readlink()==Path(target),name
    base=json.loads((a.qualification/'base-launch.json').read_text());selected=json.loads((a.qualification/'selected-arm.json').read_text())
    env={}
    for k,v in selected['env'].items():
        if k=='CUDA_VISIBLE_DEVICES':continue
        env[k]=v.replace(str(a.qualification/'candidate'),'@ARCH_ROOT@').replace('sm86','@ARCH@')
        assert '/home/sean' not in env[k],(k,env[k])
    argv=base['argv']+selected['args'][2:]
    # Keep the final effective value for repeated flags, removing only overrides.
    values={};switches=[];i=1
    while i<len(argv):
        flag=argv[i]
        if i+1<len(argv) and not argv[i+1].startswith('-'):values[flag]=argv[i+1];i+=2
        else:switches.append(flag);i+=1
    for key in ['-m','-md','--mmproj','--seed']:values.pop(key,None)
    values.update({'--host':'127.0.0.1','--port':'8080'})
    command=selected['prefix']+['@ARCH_ROOT@/bin/llama-server']
    for flag,value in values.items():command+=[flag,value]
    command+=list(dict.fromkeys(switches))
    config={'argv':command,'env':env,'model_sha256':{'target':frozen['models']['-m']['sha256'],
        'draft':frozen['models']['-md']['sha256'],'vision':frozen['models']['--mmproj']['sha256']},
        'candidate':frozen['candidate'],'scope':'R6 defaults,oneGPU,one81920slot,KVarN4/4,Q4_K_M drafter,N7,CPUvision; per-architecture certification required.'}
    (repo/'tools/universal/r6-profile.json').write_text(json.dumps(config,indent=2)+'\n')
    tools=out/'tools/universal';tools.mkdir(parents=True)
    for name in ['r6-launch.py','r6-profile.json']:shutil.copy2(repo/'tools/universal'/name,tools/name)
    (out/'l0xre').write_text('#!/usr/bin/env bash\nset -euo pipefail\nroot="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"\nexec python3 "$root/tools/universal/r6-launch.py" "$@"\n')
    (out/'l0xre').chmod(0o755)
    manifest={'arch':'sm86','built':True,'hardware_qualified':False,'status':'qualification-package',
        'candidate':frozen['candidate'],'runtime_sha256':runtime,'loader_links':frozen['loader_links'],'guard_min_tokens':512,
        'excluded_nonruntime_python_caches':excluded}
    (arch/'R6-MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for name in ['r6','fast']:shutil.copytree(repo/'source'/name,out/'source'/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    evidence=out/'evidence/r6';evidence.mkdir(parents=True)
    shutil.copy2(a.candidate/'FROZEN.json',evidence/'FROZEN.json')
    print('R6_PACKAGE_STAGED_UNQUALIFIED',out,flush=True)
if __name__=='__main__':main()
