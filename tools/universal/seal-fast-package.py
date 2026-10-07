#!/usr/bin/env python3
import argparse,hashlib,json,shutil
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);a=p.parse_args()
    repo=Path(__file__).resolve().parents[2];pkg=a.package.resolve()
    for n in ['README.md','FAST-RELEASE.md','PARITY.md','CHANGELOG-UNIVERSAL.md','MANIFEST-LINUX-UNIVERSAL.json','MANIFEST.json','MODEL-SHA256SUMS']:
        shutil.copy2(repo/n,pkg/n)
    shutil.copy2(repo/'tools/universal/l0xre',pkg/'l0xre')
    shutil.copytree(repo/'tools/universal',pkg/'tools/universal',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copytree(repo/'source/fast',pkg/'source/fast',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copytree(repo/'evidence/parity',pkg/'evidence/parity',dirs_exist_ok=True)
    src=json.loads((pkg/'SOURCE.json').read_text());src['fast_source']={'base_snapshot':'source/sm86-r5/runtime-source.tar.xz','base_sha256':'a9dd23bfa3ed38c040ffed6447e7085e083cb6abbb640a5b0cb3cf48f7de7263','overlay':'source/fast','overlay_sha256':{str(f.relative_to(repo/'source/fast')):hashlib.sha256(f.read_bytes()).hexdigest() for f in (repo/'source/fast').rglob('*') if f.is_file() and '__pycache__' not in f.parts and f.name!='SOURCE-IDENTITY.json'},'common_architectures':['sm86','sm89','sm120'],'certified_binary_architecture':'sm86','other_fast_runtime_qualification':'pending'}
    (pkg/'SOURCE.json').write_text(json.dumps(src,indent=2)+'\n')
    identity=json.dumps(src['fast_source'],indent=2)+'\n'
    (repo/'source/fast/SOURCE-IDENTITY.json').write_text(identity)
    (pkg/'source/fast/SOURCE-IDENTITY.json').write_text(identity)
    for arch in ['sm86','sm89','sm120']:
        base=pkg/'architectures'/arch
        fm=base/'FAST-MANIFEST.json'
        data=json.loads(fm.read_text())
        data['runtime_sha256']={n:h for n,h in data['runtime_sha256'].items() if '__pycache__' not in Path(n).parts and not n.endswith('.pyc')}
        fm.write_text(json.dumps(data,indent=2)+'\n')
        lines=[]
        for f in sorted(base.rglob('*')):
            if f.is_file() and f.name!='SHA256SUMS' and '__pycache__' not in f.parts and f.suffix!='.pyc':
                with f.open('rb') as handle:d=hashlib.file_digest(handle,'sha256').hexdigest()
                lines.append(d+'  '+str(f.relative_to(base))+'\n')
        (base/'SHA256SUMS').write_text(''.join(lines))
    files=[]
    for f in pkg.rglob('*'):
        if f.is_symlink():
            assert f.resolve().is_relative_to(pkg),('External symlink',str(f),str(f.readlink()))
            assert f.exists(),('Dangling symlink',str(f))
        if f.is_file() and f.relative_to(pkg)!=Path('SHA256SUMS') and '__pycache__' not in f.parts and f.suffix!='.pyc':files.append(f)
    rows=[]
    for f in sorted(files):
        with f.open('rb') as h:d=hashlib.file_digest(h,'sha256').hexdigest()
        rows.append(d+'  '+str(f.relative_to(pkg))+'\n')
    (pkg/'SHA256SUMS').write_text(''.join(rows))
    # Source identity is mirrored in the repository; raw host-specific SOURCE
    # metadata is preserved only in the archive's inherited provenance record.
    print('PACKAGE_SEALED',len(files),'files',flush=True)

if __name__=='__main__':main()
