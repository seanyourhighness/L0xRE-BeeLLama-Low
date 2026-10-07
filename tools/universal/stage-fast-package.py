#!/usr/bin/env python3
"""Overlay certified bytes, portable paths and qualification receipts on r5."""
import argparse,hashlib,json,shutil
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);p.add_argument('--certification',type=Path,required=True);p.add_argument('--certified-bin',type=Path,required=True);a=p.parse_args()
    repo=Path(__file__).resolve().parents[2];pkg=a.package.resolve();q=a.certification.resolve();arch=pkg/'architectures/sm86';bins=a.certified_bin.resolve()
    frozen=json.loads((q/'candidate/runtime-sha256.json').read_text())
    copies={
     'libggml-cuda.so.0.23.0':'bin/libggml-cuda.so.0.23.0',
     'libllama-common.so.0.4.7':'bin/libllama-common.so.0.4.7',
     'libllama-server-impl.so':'bin/libllama-server-impl.so',
     'libbridge-transform.so':'bridge/libbridge-transform.so',
     'libbridge-gdn512.so':'bridge/libbridge-gdn512.so',
     'pretransformed-direct.cubin':'bridge/pretransformed-direct.cubin',
    }
    for name,dest in copies.items():
        src=bins/name;h=hashlib.sha256(src.read_bytes()).hexdigest();assert h==frozen['bin/'+name],name
        shutil.copy2(src,arch/dest)
    c=json.loads((q/'candidate/launch-config.json').read_text());env={}
    oldroot='/home/sean/work/l0xre-r5-release-20261006/l0xre-beellama-low-v0.4.7-wsl-linux-x86_64-universal-r5/architectures/sm86'
    changes={
      'ESCHA_OFFICIAL_BRIDGE_LIBRARY':'@ARCH_ROOT@/bridge/libbridge-transform.so',
      'LD_PRELOAD':'@ARCH_ROOT@/bridge/libqk16-context-gate.so:@ARCH_ROOT@/bridge/libbridge-transform.so',
      'LD_LIBRARY_PATH':'@ARCH_ROOT@/bin:@PACKAGE_ROOT@/shared/cuda-12.8',
      'L0XRE_GDN_MASKED_LIBRARY':'@ARCH_ROOT@/bridge/libbridge-gdn512.so',
      'L0XRE_DECODE_TRANSFORM_CUBIN':'@ARCH_ROOT@/bridge/pretransformed-direct.cubin',
      'L0XRE_ARCH':'@ARCH@',
    }
    for k,v in c['env'].items():
        if k in ['CUDA_VISIBLE_DEVICES','PATH']:continue
        env[k]=changes.get(k,v.replace(oldroot,'@ARCH_ROOT@').replace('sm86','@ARCH@'))
        assert '/home/sean' not in env[k],(k,env[k])
    argv=[];i=0
    while i<len(c['argv']):
        x=c['argv'][i]
        if x in ['-m','-md','--mmproj','--host','--port']:i+=2;continue
        argv.append('@ARCH_ROOT@/bin/llama-server' if x.endswith('/bin/llama-server') else x);i+=1
    argv+=['--host','127.0.0.1','--port','8080']
    profile={'argv':argv,'env':env,'model_sha256':{'target':'b0849250c633aa93853bf119a877dbdafdd7b1a4ebb672bd6b6a1439906f3543','draft':'e3eb7705404817cdbcdabe56049a1952b3b37bcc8df6e4d4efaec5d41563fb7e'},'model_revision':'a18987908d220da9c40cf2887913e727cdbf65a4','scope':'Common source/profile for SM86/89/120. Only payloads whose FAST-MANIFEST marks built may run; hardware qualification is separate.'}
    (repo/'tools/universal/fast-profile.json').write_text(json.dumps(profile,indent=2)+'\n')
    (pkg/'tools/universal').mkdir(parents=True,exist_ok=True)
    for name in ['fast-launch.py','fast-profile.json']:shutil.copy2(repo/'tools/universal'/name,pkg/'tools/universal'/name)
    shutil.copy2(repo/'tools/universal/l0xre',pkg/'l0xre')
    # Every inherited runtime/bridge/cubin dependency has the exact certified bytes.
    external=json.loads((q/'candidate/external-dependencies.json').read_text())['sha256']
    for name,h in external.items():
        if name.startswith(oldroot+'/'):
            dest=arch/name[len(oldroot)+1:]
            if str(dest.relative_to(arch)) in copies.values():continue
            assert dest.is_file() and hashlib.sha256(dest.read_bytes()).hexdigest()==h,name
    runtime={str(f.relative_to(arch)):hashlib.sha256(f.read_bytes()).hexdigest() for folder in ['bin','bridge'] for f in (arch/folder).rglob('*') if f.is_file() and '__pycache__' not in f.parts and f.suffix!='.pyc'}
    for gpu in ['sm86','sm89','sm120']:
        data={'arch':gpu,'built':gpu=='sm86','hardware_qualified':gpu=='sm86','features':['modified-rejection-verification','fused-head','decode-transform-reuse','gdn512-masked-prefill','int8-prefill','fp16-attention-gate'],'runtime_sha256':runtime if gpu=='sm86' else {},'status':'qualified-fast' if gpu=='sm86' else 'common-source-and-kernels-prepared; inherited runtime retained','guard_min_tokens':512}
        (pkg/'architectures'/gpu/'FAST-MANIFEST.json').write_text(json.dumps(data,indent=2)+'\n')
    target=pkg/'evidence/fast';target.mkdir(parents=True,exist_ok=True)
    for n in ['CERTIFICATION.md','certification-final.json','quality-comparison.json','quality-review.json','quality-attestation-audit.json','performance-summary.json','performance-attestation-audit.json','package-probe.json','activation-receipt.json','package-config-audit.json']:shutil.copy2(q/n,target/n)
    shutil.copytree(repo/'source/fast',pkg/'source/fast',dirs_exist_ok=True)
    print('Certified SM86 bytes and portable fast profile staged; SM89/SM120 remain explicitly gated pending runtime builds.')

if __name__=='__main__':main()
