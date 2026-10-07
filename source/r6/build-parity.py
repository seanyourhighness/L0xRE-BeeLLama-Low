#!/usr/bin/env python3
"""Compile the common fast overlay and CUDA kernels for SM86/89/120.

Kernel-only preparation does not create an installable or hardware-qualified runtime.
Full builds require the matching base source archive and inherited architecture bridge
payload for unchanged code-GEMM/GDN fallback components. See README.md.
"""
import argparse,hashlib,json,os,re,shutil,subprocess,tempfile
from pathlib import Path

def run(argv):
    print(' '.join(map(str,argv)),flush=True)
    subprocess.run(list(map(str,argv)),check=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--arch',choices=['sm86','sm89','sm120'],required=True)
    p.add_argument('--cuda',type=Path,default=Path('/usr/local/cuda'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--build',type=Path)
    p.add_argument('--runtime',action='store_true')
    p.add_argument('--jobs',type=int,default=4)
    a=p.parse_args();root=Path(__file__).resolve().parent;legacy=root.parent/'fast';out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    nvcc=a.cuda/'bin/nvcc';ptxas=a.cuda/'bin/ptxas'
    assert nvcc.is_file() and ptxas.is_file(),'CUDA compiler and assembler required'
    cuda_arch=a.arch[2:];sass=a.arch.replace('sm','sm_')
    toolkit=subprocess.check_output([str(nvcc),'--version'],text=True)
    flags=['-O3','-std=c++17','-arch='+sass,'--cudart=shared','-Xcompiler=-fPIC']
    lib=a.cuda/'targets/x86_64-linux/lib';stub=lib/'stubs'
    if not stub.is_dir():stub=a.cuda/'lib64/stubs'
    bridge=out/'bridge';bridge.mkdir(exist_ok=True)
    source=a.source.resolve();arch_overlay_info=None
    includes=['-I'+str(source/'ggml/include'),'-I'+str(source/'ggml/src'),'-I'+str(source/'ggml/src/ggml-cuda'),'-I'+str(root/'bridge')]
    # The same host bridge and registered entry-point ABI on every architecture.
    run([nvcc,*flags,'-shared',*includes,root/'bridge/bridge-packed.cu','-L'+str(stub),'-lcuda','-lcublas','-o',bridge/'libbridge-packed.so'])
    run([nvcc,*flags,'-shared',legacy/'gdn/gdn-matched-prefix-v2/bridge.cu','-L'+str(stub),'-lcuda','-o',bridge/'libbridge-gdn512.so'])
    # Compile the qualified fused-head templates for every target, even in the
    # bounded kernel-only preparation. Full runtime compilation happens below.
    with tempfile.TemporaryDirectory(prefix='head-arch-') as tmp:
        unit=Path(tmp)/'head.cu'
        head_variants=sorted(set(re.findall(r'head_fused<([0-9, ]+)>',(legacy/'head/lowgpu-candidate.cu').read_text())))
        assert len(head_variants)==4,head_variants
        unit.write_text('#include "head-fused.cuh"\n'+''.join('template __global__ void head_fused<'+v+'>(const uint8_t*,const half*,const uint8_t*,const half*,float*,int,int);\n' for v in head_variants))
        run([nvcc,*flags,'-I'+str(legacy/'head'),'-c',unit,'-o',bridge/'head-arch-check.o'])
    inputs={'packed-input':root/'transform/packed-input.ptx','k3-vector-all':legacy/'transform/k3-vector.ptx'}
    for name,src in inputs.items():
        text=src.read_text().replace('.target sm_86','.target '+sass)
        port=bridge/(name+'.ptx');port.write_text(text)
        run([ptxas,'-arch='+sass,port,'-o',bridge/(name+'.cubin')])
    gdn=bridge/('gdn-cubins-'+a.arch);gdn.mkdir(exist_ok=True)
    for src in sorted((legacy/'gdn/gdn-matched-sm86').glob('*.ptx')):
        port=gdn/src.name;port.write_text(src.read_text().replace('.target sm_86','.target '+sass))
        run([ptxas,'-arch='+sass,port,'-o',port.with_suffix('.cubin')])
    if a.runtime:
        assert a.source and a.build,'--runtime requires --source and --build'
        source=a.source.resolve();build=a.build.resolve()
        arch_manifest_path=root/'ARCHITECTURE-OVERLAYS.json'
        arch_manifest=json.loads(arch_manifest_path.read_text()) if arch_manifest_path.is_file() else {}
        # A previous SM89 incremental build leaves its architecture overlay in
        # the prepared tree. Restore the common R6 input before verifying its
        # identity, then apply only the selected architecture overlay below.
        for entry in arch_manifest.values():
            common=root/'overlay'/entry['target']
            shutil.copy2(common,source/entry['target'])
        identity=json.loads((source/'R6-SOURCE.json').read_text())
        assert identity['base_sha256']=='a9dd23bfa3ed38c040ffed6447e7085e083cb6abbb640a5b0cb3cf48f7de7263'
        for name,digest in identity['overlay_sha256'].items():assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest,name
        if a.arch in arch_manifest:
            entry=arch_manifest[a.arch];overlay=root/entry['source'];target=source/entry['target']
            digest=hashlib.sha256(overlay.read_bytes()).hexdigest();assert digest==entry['sha256']
            shutil.copy2(overlay,target)
            arch_overlay_info={'arch':a.arch,'source':entry['source'],'target':entry['target'],'sha256':digest,'opt_in':entry['opt_in'],'default':entry['default'],'qualification':entry['qualification']}
            identity['architecture_overlay']=arch_overlay_info
        else:
            identity.pop('architecture_overlay',None)
        (source/'R6-SOURCE.json').write_text(json.dumps(identity,indent=2)+'\n')
        run(['cmake','-S',source,'-B',build,'-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DGGML_CUDA=ON','-DGGML_CUDA_FA=ON','-DGGML_CUDA_KVARN=ON','-DGGML_NATIVE=OFF','-DBUILD_SHARED_LIBS=ON','-DGGML_BACKEND_DL=OFF','-DLLAMA_BUILD_TESTS=OFF','-DLLAMA_BUILD_EXAMPLES=ON','-DLLAMA_BUILD_SERVER=ON','-DCMAKE_CUDA_COMPILER='+str(nvcc),'-DCMAKE_CUDA_ARCHITECTURES='+cuda_arch,'-DCMAKE_EXPORT_COMPILE_COMMANDS=ON'])
        run(['cmake','--build',build,'--parallel',a.jobs,'--target','llama-server','llama-cli','llama-bench'])
        bin_out=out/'bin'
        if bin_out.is_symlink() or bin_out.is_file():bin_out.unlink()
        elif bin_out.exists():shutil.rmtree(bin_out)
        shutil.copytree(build/'bin',bin_out,symlinks=True)
        includes=['-I'+str(source/'ggml/include'),'-I'+str(source/'ggml/src'),'-I'+str(source/'ggml/src/ggml-cuda'),'-I'+str(root/'bridge')]
        defs=['-DGGML_BACKEND_BUILD','-DGGML_BACKEND_SHARED','-DGGML_CUDA_KVARN','-DGGML_CUDA_USE_GRAPHS','-DGGML_SCHED_MAX_COPIES=4','-DGGML_SHARED','-Dggml_cuda_EXPORTS','-DNDEBUG']
        for name in ['qk16-kernels','qk16-host-wrapper']:
            run([nvcc,*flags,'--use_fast_math','--extended-lambda',*defs,*includes,'-c',root/'bridge'/(name+'.cu'),'-o',bridge/(name+'.o')])
        run([nvcc,*flags,'-shared','-Xlinker=--no-undefined',bridge/'qk16-kernels.o',bridge/'qk16-host-wrapper.o','-L'+str(out/'bin'),'-L'+str(stub),'-lggml-cuda','-lggml-base','-lcublas','-lcuda','-ldl','-o',bridge/'libqk16-context-gate.so'])
    sha={str(f.relative_to(out)):hashlib.sha256(f.read_bytes()).hexdigest() for f in out.rglob('*') if f.is_file() and f.name!='BUILD-R6.json'}
    receipt={'arch':a.arch,'compiler':toolkit,'common_source':'R5 S71 plus frozen C07 R6 overlay; architecture qualification required','architecture_overlay':arch_overlay_info,'head_instantiations':head_variants,'kernel_compile_passed':True,'runtime_build_passed':a.runtime,'installable_r6_payload':False,'hardware_qualified':False,'sha256':sha,'next':'Complete architecture packaging with inherited fallback bridge/cubin dependencies, then run the same greedy, prefill, capacity, Bench.sh, full150 and CPU vision gates before marking hardware qualified.'}
    (out/'BUILD-R6.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('ARCHITECTURE_COMPILE_PASS',a.arch,flush=True)

if __name__=='__main__':main()
