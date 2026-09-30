#!/usr/bin/env python3
"""Integration checks for profile routing and argument/environment isolation.
Uses stub servers; does not execute CUDA or claim GPU performance.
"""
import json, os, shutil, subprocess, tempfile
from pathlib import Path
source = Path(__file__).resolve().parent
profile_source = source.parent.parent/'profiles'
with tempfile.TemporaryDirectory(prefix='l0xre-launcher-') as tmp:
    root = Path(tmp)
    shutil.copy2(source/'l0xre',root/'l0xre')
    for arch in ('sm86','sm89','sm120'):
        d=root/'architectures'/arch
        (d/'bin').mkdir(parents=True)
        shutil.copy2(source/('l0xre-'+arch),d/'l0xre')
        if arch=='sm120':
            shutil.copytree(profile_source,d/'profiles')
        server=d/'bin/llama-server'
        server.write_text('#!/usr/bin/python3\nimport sys,os,json\nprint(json.dumps({"args":sys.argv[1:],"env":{k:v for k,v in os.environ.items() if k.startswith(("ESCHA_","L0XRE_","GGML_"))}}))\n')
        server.chmod(0o755)
    model=root/'model with spaces.gguf';model.touch()
    draft=root/'draft with spaces.gguf';draft.touch()
    env=dict(os.environ,PATH='/usr/bin:/bin',ESCHA_E3_HEAD_RT_BLOCK128='1',L0XRE_K3_VECTOR_CUBIN='wrong-sm86.cubin')
    def run(arch,args,code=0,extra=None):
        e=dict(env,L0XRE_ARCH=arch)
        if extra:e.update(extra)
        r=subprocess.run([str(root/'l0xre'),*args],env=e,text=True,capture_output=True)
        assert r.returncode==code,(arch,args,r.returncode,r.stderr)
        if code==0:
            data=json.loads(r.stdout)
            assert data['args'][data['args'].index('--alias')+1]=='L0xRE-27b-Low'
            return data
        return None
    def value(data,key):return data['args'][data['args'].index(key)+1]
    args=['serve','--profile','12gb','-m',str(model),'-md',str(draft),'--port','9099']
    d=run('sm86',args);assert value(d,'-c')=='98304' and value(d,'-ub')=='256' and value(d,'--spec-draft-type-k')=='q2_0' and d['env']['ESCHA_E3_HEAD_RT_BLOCK128']=='1' and value(d,'--port')=='9099'
    d=run('sm89',args);assert value(d,'-c')=='81920' and value(d,'-ub')=='64' and 'ESCHA_E3_HEAD_RT_BLOCK128' not in d['env'] and 'L0XRE_K3_VECTOR_CUBIN' not in d['env']
    d=run('sm120',args);assert value(d,'-c')=='8192' and value(d,'-ub')=='512' and value(d,'--spec-draft-n-max')=='4' and value(d,'-m')==str(model) and 'ESCHA_E3_HEAD_RT_BLOCK128' not in d['env']
    d=run('sm120',['serve','--profile','ordinary-32k','-m',str(model)]);assert value(d,'-c')=='32768' and value(d,'-ub')=='1024'
    run('sm80',args,2)
    for arch in ('sm86','sm89','sm120'):run(arch,['serve','--profile','wrong','-m',str(model)],2)
    run('sm86',['serve','--profile'],2)
    fake=root/'nvidia-smi';fake.write_text('#!/bin/sh\ncase "$*" in *"-i 1"*) echo 8.9;; *) echo 12.0;; esac\n');fake.chmod(0o755)
    d=run('',args,extra={'PATH':str(root)+':/usr/bin:/bin','CUDA_VISIBLE_DEVICES':'1'});assert value(d,'-c')=='81920'
    run('',args,2,{'CUDA_VISIBLE_DEVICES':''})
print('PASS: 11 Linux launcher integration cases (stub servers; no GPU execution).')
