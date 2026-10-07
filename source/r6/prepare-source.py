#!/usr/bin/env python3
"""Prepare the frozen C07 R6 source over the published R5 source snapshot."""
import argparse,hashlib,importlib.util,json,shutil
from pathlib import Path
def prepare(archive,destination):
    root=Path(__file__).resolve().parent;destination=Path(destination).resolve()
    spec=importlib.util.spec_from_file_location('fast_prepare',root.parent/'fast/prepare-source.py')
    fast=importlib.util.module_from_spec(spec);spec.loader.exec_module(fast)
    fast.prepare(archive,destination)
    identity=json.loads((root/'SOURCE-IDENTITY.json').read_text())
    merged=json.loads((destination/'FAST-SOURCE.json').read_text())['overlay_sha256']
    for name,digest in identity['native_overlay_sha256'].items():
        src=root/'overlay'/name;assert hashlib.sha256(src.read_bytes()).hexdigest()==digest,name
        out=destination/name;out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,out);merged[name]=digest
    (destination/'R6-SOURCE.json').write_text(json.dumps({'base_sha256':identity['base_archive_sha256'],
        'overlay_sha256':merged,'candidate':identity['candidate'],'projection_sha256':identity['projection_sha256'],
        'hardware_qualification':'Prepared source is not a hardware certificate'},indent=2)+'\n')
    print('Prepared R6 C07 source:',destination)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--destination',type=Path,required=True)
    a=p.parse_args();prepare(a.archive,a.destination)
