#!/usr/bin/env python3
"""Apply the same qualified source overlay for all three CUDA targets."""
import argparse,hashlib,json,shutil,tarfile
from pathlib import Path
BASE_SHA='a9dd23bfa3ed38c040ffed6447e7085e083cb6abbb640a5b0cb3cf48f7de7263'

def prepare(archive,destination):
    source=Path(__file__).resolve().parent
    archive=Path(archive).resolve();destination=Path(destination).resolve()
    assert hashlib.file_digest(archive.open('rb'),'sha256').hexdigest()==BASE_SHA,'Wrong base source archive'
    if destination.exists() and any(destination.iterdir()):raise SystemExit('Destination must be empty; use a new source directory')
    destination.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive,'r:xz') as t:t.extractall(destination,filter='data')
    replaced={}
    for p in sorted((source/'rejection/src').rglob('*')):
        if p.is_file():
            rel=p.relative_to(source/'rejection/src');out=destination/rel;out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,out)
            replaced[str(rel)]=hashlib.sha256(p.read_bytes()).hexdigest()
    for name in ['lowgpu-candidate.cu','head-fused.cuh']:
        rel=Path('ggml/src/ggml-cuda')/('lowgpu.cu' if name=='lowgpu-candidate.cu' else name)
        p=source/'head'/name;shutil.copy2(p,destination/rel);replaced[str(rel)]=hashlib.sha256(p.read_bytes()).hexdigest()
    (destination/'FAST-SOURCE.json').write_text(json.dumps({'base_sha256':BASE_SHA,'overlay_sha256':replaced,'supported_architectures':['sm86','sm89','sm120'],'hardware_qualification':'Source preparation is not hardware qualification'},indent=2)+'\n')
    print('Prepared common fast source:',destination)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--destination',type=Path,required=True);a=p.parse_args();prepare(a.archive,a.destination)
