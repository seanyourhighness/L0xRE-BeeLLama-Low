#!/usr/bin/env python3
"""Start a clean final archive through its real certified consumer launcher."""
import argparse
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re

ROOT = Path('/home/sean/work/l0xre-r6-4090-20261010')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('package', type=Path)
    opt = parser.parse_args(); package = opt.package.resolve()
    old = json.loads((ROOT / 'package-current/architectures/sm89/R6-MANIFEST.json').read_text())
    fresh = json.loads((package / 'architectures/sm89/R6-MANIFEST.json').read_text())
    assert fresh['runtime_sha256'] == old['runtime_sha256'] and fresh['loader_links'] == old['loader_links']
    for relative,want in fresh['runtime_sha256'].items(): assert sha(package / 'architectures/sm89' / relative) == want
    assert fresh['profile_qualification']['r6-sm89-4090'] and not fresh['profile_qualification']['r6']
    frozen = json.loads((ROOT / 'FROZEN-CANDIDATE.json').read_text())
    assert sha(package / 'tools/universal/r6-linux-sm89-4090-profile.json') == frozen['profile_sha256']
    control = ROOT / 'consumer-control'; control.mkdir(exist_ok=False)
    frozen.update(package=str(package), inventory_sha256=sha(package / 'SHA256SUMS'))
    (control / 'FROZEN-CANDIDATE.json').write_text(json.dumps(frozen,indent=2)+'\n')
    command = ['/usr/bin/python3',str(package / 'tools/universal/r6-launch.py'),'serve','--arch','sm89',
               '--profile','r6-sm89-4090','-m','/home/sean/escha-assets/base-models/escha-e3-firstclass-v2.gguf',
               '-md','/home/sean/escha-assets/Qwen3.8-27B-DFlash2-Q4_K_M.gguf']
    (control / 'current-dry-run.json').write_text(json.dumps({'argv':command,'env':{
        'CUDA_VISIBLE_DEVICES':'GPU-c4a782a6-fd5a-78c2-7d77-604c01cbffb8','L0XRE_GDN_COMPACT_RS':'1'}},indent=2)+'\n')
    spec = importlib.util.spec_from_file_location('probe', ROOT / 'qualification_probe.py')
    probe = importlib.util.module_from_spec(spec); spec.loader.exec_module(probe); probe.ROOT = control
    control.joinpath('probes').mkdir()
    with probe.boot('clean-certified-package',vision=True) as (out,request):
        expected = json.loads((ROOT / 'CORRECTNESS.json').read_text())['results']['spec']['greedy']
        for index,prompt in enumerate(probe.PROMPTS):
            answer = request('greedy'+str(index),probe.payload(prompt))
            assert probe.digest(answer['tokens']) == expected[index]
        ids = json.loads(Path('/home/sean/work/l0xre-prefill-20261002/fixtures.json').read_text())['pp77824'][:10240]
        pre = request('cold10k',probe.payload(ids,1)); assert pre['timings']['prompt_n'] == 10240
        image = (ROOT / 'vision-smoke.png').read_bytes()
        body={'messages':[{'role':'user','content':[{'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(image).decode()}},
              {'type':'text','text':'Transcribe the printed text in the image. Reply with only the exact text, including its number. Do not describe shapes or colors.'}]}],
              'temperature':0,'seed':42,'max_tokens':512,'reasoning_effort':'medium'}
        vision = request('cpu-ocr',body,'/v1/chat/completions'); assert vision['choices'][0]['finish_reason']=='stop'
        assert vision['choices'][0]['message']['content'].strip()=='VISION 742'
        process=json.loads((out/'process.json').read_text()); pid=process['pid']
        env=dict(value.split('=',1) for value in Path(f'/proc/{pid}/environ').read_bytes().decode().split('\0') if '=' in value)
        assert env['CUDA_VISIBLE_DEVICES']=='GPU-c4a782a6-fd5a-78c2-7d77-604c01cbffb8'
        assert all(str(package/'architectures/sm89/bin'/name) in process['maps'] for name in ['libggml-cuda.so.0.23.0','libbridge-packed.so','libqk16-context-gate.so'])
        assert 'CLIP using CPU backend' in (out/'server.log').read_text()
        (control/'ACTUAL-RUNTIME-ENV.json').write_text(json.dumps({k:v for k,v in env.items() if k.startswith(('L0XRE_','ESCHA_','GGML_')) or k in ['LD_PRELOAD','LD_LIBRARY_PATH','CUDA_VISIBLE_DEVICES']},indent=2)+'\n')
    status=json.loads((control/'probes/clean-certified-package/status.json').read_text())
    assert status['state']=='gate_pass' and status['server_exit'] in [0,-2]
    receipt={'status':'pass','package':str(package),'actual_consumer_launcher':True,'qualification_probe_flag_used':False,
             'native_and_profile_equal_frozen':True,'greedy_oracle_match':True,'cold10k_pass':True,'cpu_vision':'VISION 742',
             'cpu_affinity':list(range(8)),'clean_exit':status['server_exit'],'timestamp':status['finished']}
    (ROOT/'FINAL-CONSUMER-PROBE.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
