#!/usr/bin/env python3
"""Launch an R6 architecture payload with its sealed profile and model identities."""
import argparse,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    root=Path(__file__).resolve().parents[2];args=sys.argv[1:]
    if 'serve' in args:args.remove('serve')
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--arch',choices=['sm86','sm89','sm120'])
    p.add_argument('--profile',choices=['r6'],default='r6')
    p.add_argument('-m','--model',required=True);p.add_argument('-md','--draft-model','--spec-draft-model',dest='draft',required=True)
    p.add_argument('--mmproj');p.add_argument('--dry-run',action='store_true');p.add_argument('--verify-models',action='store_true')
    p.add_argument('--qualification-probe',action='store_true');opt,extra=p.parse_known_args(args)
    selector=os.environ.get('CUDA_VISIBLE_DEVICES','0').split(',')[0]
    try:
        raw=subprocess.check_output(['nvidia-smi','--id='+selector,'--query-gpu=uuid,compute_cap','--format=csv,noheader'],text=True)
        uuid,cc=[x.strip() for x in raw.strip().splitlines()[0].split(',')]
    except (OSError,subprocess.CalledProcessError,ValueError,IndexError):p.error('Cannot identify the selected CUDA GPU')
    detected='sm'+cc.replace('.','');arch=opt.arch or detected
    if arch!=detected:p.error(f'Selected GPU is {detected}, requested payload is {arch}')
    archroot=root/'architectures'/arch;path=archroot/'R6-MANIFEST.json'
    if not path.is_file():p.error(f'{arch} R6 payload is not installed')
    manifest=json.loads(path.read_text())
    if not manifest.get('built'):p.error(f'{arch} R6 payload has not been built')
    if not manifest.get('hardware_qualified') and not opt.qualification_probe:p.error(f'{arch} R6 hardware qualification is pending')
    config=json.loads((root/'tools/universal/r6-profile.json').read_text())
    for name,value in manifest['runtime_sha256'].items():
        path=archroot/name
        if not path.is_file() or sha(path)!=value:p.error(f'R6 payload integrity check failed: {name}')
    for name,target in manifest.get('loader_links',{}).items():
        if os.readlink(archroot/name)!=target:p.error(f'R6 loader link changed: {name}')
    model=Path(opt.model).expanduser().resolve();draft=Path(opt.draft).expanduser().resolve()
    for key,path in [('target',model),('draft',draft)]:
        if not path.is_file() or sha(path)!=config['model_sha256'][key]:p.error(f'{key} differs from the R6 qualified model')
    if opt.mmproj:
        vision=Path(opt.mmproj).expanduser().resolve()
        if not vision.is_file() or sha(vision)!=config['model_sha256']['vision']:p.error('Vision projector differs from the R6 qualified file')
    def expand(value):return value.replace('@PACKAGE_ROOT@',str(root)).replace('@ARCH_ROOT@',str(archroot)).replace('@ARCH@',arch)
    env={k:v for k,v in os.environ.items() if not k.startswith(('L0XRE_','ESCHA_','GGML_')) and k not in ['LD_PRELOAD','LD_LIBRARY_PATH']}
    env.update({k:expand(v) for k,v in config['env'].items()});env['CUDA_VISIBLE_DEVICES']=uuid
    command=[expand(x) for x in config['argv']]+['-m',str(model),'-md',str(draft)]
    if opt.mmproj:command+=['--mmproj',str(vision)]
    command+=extra
    if opt.dry_run:
        print(json.dumps({'arch':arch,'hardware_qualified':manifest.get('hardware_qualified',False),'argv':command,
            'env':{k:env[k] for k in [*config['env'],'CUDA_VISIBLE_DEVICES']}},indent=2));return
    if not shutil.which('numactl'):p.error('R6 requires numactl for its qualified CPU/NUMA policy')
    os.execvpe(command[0],command,env)
if __name__=='__main__':main()
