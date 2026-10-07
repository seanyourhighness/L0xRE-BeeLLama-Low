#!/usr/bin/env python3
"""Exercise release gates using small payload/model fixtures and a GPU selector."""
import hashlib,json,os,shutil,subprocess,tempfile
from pathlib import Path
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    source=Path(__file__).with_name('r6-launch.py')
    with tempfile.TemporaryDirectory(prefix='r6-launch-test-') as tmp:
        root=Path(tmp);tools=root/'tools/universal';tools.mkdir(parents=True);shutil.copy2(source,tools/'r6-launch.py')
        arch=root/'architectures/sm86';(arch/'bin').mkdir(parents=True)
        payload=arch/'bin/llama-server';payload.write_bytes(b'frozen fixture')
        model=root/'target.gguf';model.write_bytes(b'target');draft=root/'draft.gguf';draft.write_bytes(b'q4')
        manifest={'built':True,'hardware_qualified':False,'runtime_sha256':{'bin/llama-server':sha(payload)},'loader_links':{}}
        path=arch/'R6-MANIFEST.json';path.write_text(json.dumps(manifest))
        config={'argv':['@ARCH_ROOT@/bin/llama-server','-c','81920'],'env':{'L0XRE_ARCH':'@ARCH@','L0XRE_PACKED_DECODE_INPUT':'1'},'model_sha256':{'target':sha(model),'draft':sha(draft)}}
        (tools/'r6-profile.json').write_text(json.dumps(config))
        fake=root/'fake';fake.mkdir();smi=fake/'nvidia-smi';smi.write_text('#!/bin/sh\nprintf "GPU-fixture, 8.6\\n"\n');smi.chmod(0o755)
        env=os.environ.copy();env['PATH']=str(fake)+':'+env['PATH'];env['L0XRE_PACKED_DECODE_INPUT']='poison'
        args=['python3',str(tools/'r6-launch.py'),'serve','--profile','r6','-m',str(model),'-md',str(draft),'--dry-run']
        def run(extra=()):return subprocess.run(args+list(extra),env=env,capture_output=True,text=True)
        assert run().returncode!=0,'unqualified payload served'
        result=run(['--qualification-probe']);assert result.returncode==0,result.stderr
        data=json.loads(result.stdout);assert data['argv'][0]==str(payload) and data['env']['L0XRE_PACKED_DECODE_INPUT']=='1'
        assert data['env']['L0XRE_ARCH']=='sm86' and data['env']['CUDA_VISIBLE_DEVICES']=='GPU-fixture'
        payload.write_bytes(b'changed');assert run(['--qualification-probe']).returncode!=0,'changed runtime accepted';payload.write_bytes(b'frozen fixture')
        draft.write_bytes(b'q2');assert run(['--qualification-probe']).returncode!=0,'wrong drafter accepted';draft.write_bytes(b'q4')
        smi.write_text('#!/bin/sh\nprintf "GPU-fixture, 12.0\\n"\n');assert run(['--qualification-probe','--arch','sm86']).returncode!=0,'wrong architecture accepted'
        smi.write_text('#!/bin/sh\nprintf "GPU-fixture, 8.6\\n"\n');manifest['hardware_qualified']=True;path.write_text(json.dumps(manifest));assert run().returncode==0
    print('R6_LAUNCH_GATES_PASS:qualification,runtimehash,modelhash,architecture,ambientenv,qualifiedroute')
if __name__=='__main__':main()
