#!/usr/bin/env python3
"""Bounded clean-package GPU smoke; restores the existing service in finally."""
import argparse,base64,hashlib,json,os,subprocess,time,urllib.request
from datetime import datetime,timezone
from pathlib import Path

def sh(*args):return subprocess.run(args,check=True,text=True,capture_output=True).stdout.strip()
def http(url,body=None):
 r=urllib.request.Request(url,data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(r,timeout=300) as f:return json.load(f)
def wait(url):
 for _ in range(150):
  try:
   if http(url+'/health')['status']=='ok':return
  except Exception:pass
  time.sleep(1)
 raise RuntimeError('Package server did not become healthy')

def main():
 p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);p.add_argument('--target',type=Path,required=True);p.add_argument('--draft',type=Path,required=True);p.add_argument('--projector',type=Path,required=True);p.add_argument('--vision-fixture',type=Path,required=True);p.add_argument('--gpu',required=True);p.add_argument('--restore-unit',required=True);p.add_argument('--receipt',type=Path,required=True);a=p.parse_args()
 root=a.package.resolve();url='http://127.0.0.1:30173';unit='l0xre-public-fast-probe'
 assert not any(s['is_processing'] for s in http('http://127.0.0.1:30172/slots')),'Production server busy; try later'
 assert json.loads((root/'architectures/sm86/FAST-MANIFEST.json').read_text())['hardware_qualified']
 for f in [a.target,a.draft,a.projector,a.vision_fixture]:assert f.is_file(),str(f)
 assert hashlib.sha256(a.vision_fixture.read_bytes()).hexdigest()=='145534c3b1df9de079da2c59eefeea210a6013578f0355ce31a9af0ca7331c5f'
 command=[str(root/'l0xre'),'serve','--profile','fast','--verify-models','-m',str(a.target),'-md',str(a.draft),'--mmproj',str(a.projector),'--host','127.0.0.1','--port','30173']
 result={'passed':False,'package':root.name,'started_at_utc':datetime.now(timezone.utc).isoformat(),'checks':{}}
 stopped=False
 try:
  sh('systemctl','--user','stop',a.restore_unit);stopped=True
  subprocess.run(['systemctl','--user','stop',unit],capture_output=True)
  sh('systemd-run','--user','--collect','--unit='+unit,'--property=Restart=no','--property=TimeoutStopSec=120','--setenv=CUDA_VISIBLE_DEVICES='+a.gpu,'--setenv=L0XRE_ARCH=sm86',*command)
  wait(url);pid=int(sh('systemctl','--user','show',unit,'-p','MainPID','--value'))
  maps=Path(f'/proc/{pid}/maps').read_text()
  assert '/candidate/bin/' not in maps and '/l0xre-r5-decode-update-' not in maps
  required=['libggml-cuda.so.0.23.0','libllama-server-impl.so','libllama-common.so.0.4.7','libbridge-transform.so']
  for name in required:assert any(str(root) in line and name in line for line in maps.splitlines()),name
  assert sorted(os.sched_getaffinity(pid))==list(range(8))
  for prompt,wanted in [('Write a Python implementation of quicksort with comments explaining each step.','0db3539adee4b63dceb8a541a1eae4a9d41b4d841a70bb00924e22240e358594'),('Write a detailed 800-word essay explaining transformer attention.','af0d5d75aaa9b0dbf0fdc7bd199f682b1f57f2fb7b9b59e8b21539d79cefeee7')]:
   r=http(url+'/completion',{'prompt':prompt,'n_predict':128,'temperature':0,'top_k':1,'seed':42,'ignore_eos':True,'cache_prompt':False,'return_tokens':True,'reasoning_loop_guard':'off'})
   assert len(r['tokens'])==128 and not r.get('truncated')
   assert hashlib.sha256(json.dumps(r['tokens']).encode()).hexdigest()==wanted
  result['checks']['greedy_code_prose_hashes']=True
  r=http(url+'/v1/chat/completions',{'model':'L0xRE-27b-Low','messages':[{'role':'user','content':'Return only this JSON object: {"status":"ready"}'}],'max_tokens':256,'temperature':.7,'top_p':.95,'top_k':20,'min_p':.05,'reasoning_effort':'medium'})
  assert r['choices'][0]['finish_reason']=='stop' and json.loads(r['choices'][0]['message']['content'])=={'status':'ready'}
  result['checks']['production_json']=True
  image=base64.b64encode(a.vision_fixture.read_bytes()).decode()
  r=http(url+'/v1/chat/completions',{'messages':[{'role':'user','content':[{'type':'text','text':'Transcribe the printed text in the image. Reply with only the exact text, including its number. Do not describe shapes or colors.'},{'type':'image_url','image_url':{'url':'data:image/png;base64,'+image}}]}],'max_tokens':512,'temperature':0,'seed':42,'reasoning_effort':'medium'})
  assert r['choices'][0]['finish_reason']=='stop' and r['choices'][0]['message']['content'].strip()=='VISION 742'
  result['checks']['cpu_vision']=True
  assert pid==int(sh('systemctl','--user','show',unit,'-p','MainPID','--value'))
  log=sh('journalctl','--user','_PID='+str(pid),'--no-pager','-o','cat')
  assert 'CUDA error:' not in log and 'L0XRE_REJECTION_VERIFY_ACTIVE' in log
  result.update(passed=True,pid=pid,cpu_affinity=list(range(8)),maps_sha256=hashlib.sha256(maps.encode()).hexdigest(),runtime_manifest=json.loads((root/'architectures/sm86/FAST-MANIFEST.json').read_text()),props=http(url+'/props'))
  print('CLEAN_PACKAGE_GPU_PROBE_PASS',flush=True)
 finally:
  subprocess.run(['systemctl','--user','stop',unit],capture_output=True)
  if stopped:
   sh('systemctl','--user','start',a.restore_unit);wait('http://127.0.0.1:30172')
   result['production_restored']=True
  result['finished_at_utc']=datetime.now(timezone.utc).isoformat();a.receipt.write_text(json.dumps(result,indent=2)+'\n')
  print('PRODUCTION_RESTORED',flush=True)

if __name__=='__main__':main()
