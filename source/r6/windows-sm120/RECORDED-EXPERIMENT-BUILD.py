"""Rebuild isolated host configuration snapshots with the original CUDA13.0 flags."""
import hashlib,json,re,shutil,subprocess,time
from pathlib import Path
base=Path(r'C:\work\l0xre-sm120-cuda130-windows-20261008')
root=Path(r'C:\work\l0xre-sm120-envcache-windows-20261008')
build=base/'build'
ninja=r'D:\VS\BuildTools2022\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe'
commands=subprocess.check_output([ninja,'-C',str(build),'-t','commands','ggml-cuda'],text=True).splitlines()
env_output=subprocess.check_output('cmd /d /s /c "call D:\\VS\\BuildTools2022\\VC\\Auxiliary\\Build\\vcvars64.bat >nul && set"',text=True)
env=dict(line.split('=',1) for line in env_output.splitlines() if '=' in line and not line.startswith('='))
receipt={'started':time.time(),'purpose':'Cache startup feature flags at selected host dispatch call sites; attestation stays dynamic.','compiler_commands':[],'control_root':str(base),'serving_profile_locked':True}
try:
    assert not (root/'package').exists(),'Refusing to replace existing experiment package'
    shutil.copytree(base/'package',root/'package')
    replacements={}
    for name in ['ggml-cuda.cu','escha-moe.cu']:
        oldsrc=str(base/'source/ggml/src/ggml-cuda'/name)
        oldobj='ggml\\src\\ggml-cuda\\CMakeFiles\\ggml-cuda.dir\\'+name+'.obj'
        newobj=str(root/(name+'.obj'))
        command=next(c for c in commands if 'nvcc.exe ' in c and oldsrc in c)
        command=command.replace(oldsrc,str(root/'source'/name)).replace(oldobj,newobj)
        command=re.sub(r'-Xcompiler=-Fd[^ ]+', '-Xcompiler=-Fd'+root.as_posix()+'/,-FS',command)
        command+=' -I'+str(base/'source/ggml/src/ggml-cuda')
        receipt['compiler_commands'].append(command)
        with (root/(name+'.compile.log')).open('w') as log:
            result=subprocess.run(command,cwd=build,env=env,stdout=log,stderr=subprocess.STDOUT)
        assert result.returncode==0, name+' compile failed'
        replacements[oldobj]=newobj
        print('COMPILED '+name,flush=True)
    text=(build/'build.ninja').read_text()
    line=next(s for s in text.splitlines() if s.startswith('build bin\\ggml-cuda.dll:'))
    objects=[replacements.get(p,str(build/p)) for p in line.split(' ',3)[3].split(' | ')[0].split()]
    block=text[text.index(line):].split('\n\n',1)[0]
    libs=next(s.strip().split(' = ',1)[1] for s in block.splitlines() if s.strip().startswith('LINK_LIBRARIES =')).split()
    libs=[str(build/p) if p.startswith('ggml\\') else p for p in libs]
    rsp=root/'link.rsp';rsp.write_text('\n'.join('"'+p+'"' for p in objects+libs))
    args=[shutil.which('link.exe',path=env['Path']),'/nologo','@'+str(rsp),'/out:'+str(root/'package/bin/ggml-cuda.dll'),'/implib:'+str(root/'ggml-cuda.lib'),'/pdb:'+str(root/'ggml-cuda.pdb'),'/dll','/version:0.23','-shared','/machine:x64','/INCREMENTAL:NO','-LIBPATH:'+str(base/'cuda/lib/x64')]
    receipt['link_command']=args
    with (root/'link.log').open('w') as log:result=subprocess.run(args,cwd=build,env=env,stdout=log,stderr=subprocess.STDOUT)
    assert result.returncode==0,'CUDA DLL link failed'
    args=[str(base/'cuda/bin/nvcc.exe'),'-O3','-std=c++17','-gencode=arch=compute_120,code=sm_120','-gencode=arch=compute_120a,code=sm_120a','-shared','-Xcompiler','/MD','-Xlinker','/DEF:bridge-packed.def','-Xlinker','/NODEFAULTLIB:LIBCMT','-I.','-I'+str(base/'source/ggml/src/ggml-cuda'),'-I'+str(base/'source/ggml/src'),'-I'+str(base/'source/ggml/include'),'bridge-packed.cu','-lcublas','-lcuda','-o',str(root/'package/bridge/bridge-packed.dll')]
    receipt['bridge_command']=args
    with (root/'bridge.compile.log').open('w') as log:result=subprocess.run(args,cwd=root/'bridge',env=env,stdout=log,stderr=subprocess.STDOUT)
    assert result.returncode==0,'bridge compile failed'
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    receipt['binary_hashes']={p:sha(root/'package'/p) for p in ['bin/ggml-cuda.dll','bridge/bridge-packed.dll']}
    receipt['source_hashes']={str(p.relative_to(root)):sha(p) for p in [root/'source/ggml-cuda.cu',root/'source/escha-moe.cu',root/'source/l0xre-startup-env.h',root/'bridge/bridge-packed.cu']}
    receipt['status']='built'
except Exception as error:receipt.update(status='error',error=repr(error))
finally:
    receipt['finished']=time.time();(root/'BUILD-PROVENANCE.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)
