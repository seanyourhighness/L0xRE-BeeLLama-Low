#!/usr/bin/env python3
import argparse,hashlib,json,os,subprocess,tarfile,tempfile
from pathlib import Path,PurePosixPath

def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True);p.add_argument('--extract-to',type=Path,required=True);a=p.parse_args()
    proc=subprocess.Popen(['zstd','-dc',str(a.archive)],stdout=subprocess.PIPE)
    roots=set();count=0
    with tarfile.open(fileobj=proc.stdout,mode='r|') as t:
        for m in t:
            q=PurePosixPath(m.name);assert not q.is_absolute() and '..' not in q.parts,m.name
            roots.add(q.parts[0]);assert m.isdir() or m.isfile() or m.issym() or m.islnk(),m.name
            if m.issym() or m.islnk():
                assert not PurePosixPath(m.linkname).is_absolute(),m.name
            count+=1
    assert proc.wait()==0 and len(roots)==1
    a.extract_to.mkdir(parents=True,exist_ok=True)
    assert not any(a.extract_to.iterdir()),'Extraction directory must be empty'
    subprocess.run(['tar','--zstd','-xf',str(a.archive),'-C',str(a.extract_to),'--no-same-owner'],check=True)
    root=a.extract_to/next(iter(roots));checked=0
    for line in (root/'SHA256SUMS').read_text().splitlines():
        digest,name=line.split('  ',1);f=root/name
        assert f.resolve().is_relative_to(root.resolve()),name
        with f.open('rb') as h:observed=hashlib.file_digest(h,'sha256').hexdigest()
        assert observed==digest,name;checked+=1
    m=json.loads((root/'architectures/sm86/FAST-MANIFEST.json').read_text())
    assert m['built'] and m['hardware_qualified']
    for arch in ['sm89','sm120']:
        x=json.loads((root/'architectures'/arch/'FAST-MANIFEST.json').read_text());assert not x['built'] and not x['hardware_qualified']
    nested_checked=0
    for arch in ['sm86','sm89','sm120']:
        base=root/'architectures'/arch
        for line in (base/'SHA256SUMS').read_text().splitlines():
            h,n=line.split('  ',1)
            with (base/n).open('rb') as handle:assert hashlib.file_digest(handle,'sha256').hexdigest()==h,(arch,n)
            nested_checked+=1
    receipt={'passed':True,'archive':a.archive.name,'bytes':a.archive.stat().st_size,'sha256':hashlib.file_digest(a.archive.open('rb'),'sha256').hexdigest(),'archive_entries':count,'file_hashes_verified':checked,'architecture_hashes_verified':nested_checked,'safe_paths':True,'clean_extract_root':str(root),'fast_sm86_payload_hardware_qualified':True,'sm89_sm120_fast_payloads_gated':True,'scope':'Archive integrity and clean extraction; portable GPU launch has a separate receipt.'}
    a.receipt.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2),flush=True)

if __name__=='__main__':main()
