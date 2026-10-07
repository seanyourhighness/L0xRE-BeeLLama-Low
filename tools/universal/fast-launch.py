#!/usr/bin/env python3
"""Portable launch of an architecture's sealed S71 fast components."""
import argparse,hashlib,json,os,shutil,sys
from pathlib import Path

def main():
    root=Path(__file__).resolve().parents[2]
    args=sys.argv[1:]
    if 'serve' in args:args.remove('serve')
    parser=argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--arch',choices=['sm86','sm89','sm120'],required=True)
    parser.add_argument('--profile',choices=['fast'],default='fast')
    parser.add_argument('-m','--model',required=True)
    parser.add_argument('-md','--draft-model','--spec-draft-model',required=True,dest='draft')
    parser.add_argument('--mmproj')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--verify-models',action='store_true')
    parser.add_argument('--qualification-probe',action='store_true')
    opt,extra=parser.parse_known_args(args)
    archroot=root/'architectures'/opt.arch
    manifest_path=archroot/'FAST-MANIFEST.json'
    if not manifest_path.is_file():parser.error(f'{opt.arch} fast payload is not installed; see PARITY.md and source/fast/README.md')
    manifest=json.loads(manifest_path.read_text())
    if not manifest.get('built'):parser.error(f'{opt.arch} fast build is prepared but not yet installed; retain its existing profile until hardware qualification')
    if not manifest.get('hardware_qualified') and not opt.qualification_probe:
        parser.error(f'{opt.arch} fast hardware qualification is pending; use --qualification-probe only for the qualification workflow')
    config=json.loads((root/'tools/universal/fast-profile.json').read_text())
    def expand(s):return s.replace('@PACKAGE_ROOT@',str(root)).replace('@ARCH_ROOT@',str(archroot)).replace('@ARCH@',opt.arch)
    env={k:v for k,v in os.environ.items() if not k.startswith(('L0XRE_','ESCHA_','GGML_')) and k not in ['LD_PRELOAD','LD_LIBRARY_PATH']}
    env.update({k:expand(v) for k,v in config['env'].items()})
    env['L0XRE_ARCH']=opt.arch
    for name,digest in manifest['runtime_sha256'].items():
        p=archroot/name
        if not p.is_file() or hashlib.file_digest(p.open('rb'),'sha256').hexdigest()!=digest:
            parser.error(f'Fast runtime integrity check failed: {p}')
    model=Path(opt.model).expanduser().resolve();draft=Path(opt.draft).expanduser().resolve()
    for p in [model,draft]+([Path(opt.mmproj).expanduser().resolve()] if opt.mmproj else []):
        if not p.is_file():parser.error(f'Model/projector unreadable: {p}')
    if opt.verify_models:
        for key,path in [('target',model),('draft',draft)]:
            if hashlib.file_digest(path.open('rb'),'sha256').hexdigest()!=config['model_sha256'][key]:parser.error(f'{key} model differs from the certified file: {path}')
    command=[expand(x) for x in config['argv']]+['-m',str(model),'-md',str(draft)]
    if opt.mmproj:command+=['--mmproj',str(Path(opt.mmproj).expanduser().resolve())]
    command+=extra
    if opt.dry_run:
        print(json.dumps({'arch':opt.arch,'hardware_qualified':manifest.get('hardware_qualified',False),'argv':command,'env':{k:env[k] for k in config['env']}},indent=2));return
    if not shutil.which('numactl'):parser.error('The certified fast NUMA policy requires numactl; install it or use a legacy profile')
    os.execvpe(command[0],command,env)

if __name__=='__main__':main()
