#!/usr/bin/env python3
"""Build Linux SM89 from the pinned current R6 native source-input archive."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request
import zipfile

SOURCE_SHA256 = '398fa0d234f16bbfc56ad2fe5e9217837deeaa535a694a0a68fe2f8f19e9dfeb'
SOURCE_URL = ('https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/releases/download/'
              'beellama-v0.4.7-universal-r6/L0xRE-BeeLLama-Low-R6-sm89-windows-refresh-source-inputs.zip')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def run(argv):
    print(' '.join(map(str, argv)), flush=True)
    subprocess.run(list(map(str, argv)), check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-inputs', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cuda', type=Path, default=Path('/usr/local/cuda'))
    parser.add_argument('--jobs', type=int, default=6)
    options = parser.parse_args()
    if options.jobs < 1:
        parser.error('--jobs must be positive')
    output = options.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    archive = options.source_inputs
    if archive is None:
        archive = output / 'source-inputs.zip'
        if not archive.exists():
            temporary = archive.with_suffix('.download')
            urllib.request.urlretrieve(SOURCE_URL, temporary)
            if sha(temporary) != SOURCE_SHA256:
                raise RuntimeError('Downloaded source archive identity differs')
            temporary.replace(archive)
    if sha(archive) != SOURCE_SHA256:
        raise RuntimeError('Source archive identity differs')
    inputs = output / 'inputs'
    inputs.mkdir(exist_ok=True)
    inventory = {}
    with zipfile.ZipFile(archive) as bundle:
        for name in bundle.namelist():
            if not name.startswith(('source/', 'bridge/', 'companions/')) or name.endswith('/'):
                continue
            destination = inputs / name
            if not destination.resolve().is_relative_to(inputs.resolve()):
                raise RuntimeError('Source member escapes output directory')
            data = bundle.read(name)
            want = hashlib.sha256(data).hexdigest()
            if destination.exists():
                if sha(destination) != want:
                    raise RuntimeError('Existing source input differs: ' + name)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
            inventory[name] = want
    source = inputs / 'source'
    bridge = inputs / 'bridge'
    build = output / 'build'
    nvcc = options.cuda / 'bin/nvcc'
    stub = options.cuda / 'targets/x86_64-linux/lib/stubs'
    if not nvcc.is_file() or not stub.is_dir():
        raise RuntimeError('CUDA toolkit with Linux driver stubs is required')
    run(['cmake', '-S', source, '-B', build, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
         '-DGGML_CUDA=ON', '-DGGML_CUDA_FA=ON', '-DGGML_CUDA_KVARN=ON', '-DGGML_NATIVE=OFF',
         '-DBUILD_SHARED_LIBS=ON', '-DGGML_BACKEND_DL=OFF', '-DLLAMA_BUILD_TESTS=OFF',
         '-DLLAMA_BUILD_EXAMPLES=ON', '-DLLAMA_BUILD_SERVER=ON',
         '-DCMAKE_CUDA_COMPILER=' + str(nvcc), '-DCMAKE_CUDA_ARCHITECTURES=89',
         '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON'])
    run(['cmake', '--build', build, '--parallel', str(options.jobs), '--target',
         'llama-server', 'llama-cli', 'llama-bench'])
    flags = ['-O3', '-std=c++17', '-arch=sm_89', '--cudart=shared', '-Xcompiler=-fPIC']
    includes = ['-I' + str(source / p) for p in ['ggml/include', 'ggml/src', 'ggml/src/ggml-cuda']]
    includes.append('-I' + str(bridge))
    run([nvcc, *flags, '-shared', *includes, bridge / 'bridge-packed.cu',
         '-L' + str(stub), '-lcuda', '-lcublas', '-o', output / 'libbridge-packed.so'])
    run([nvcc, *flags, '-shared', inputs / 'companions/bridge.cu',
         '-L' + str(stub), '-lcuda', '-o', output / 'libbridge-gdn512.so'])
    definitions = ['-DGGML_BACKEND_BUILD', '-DGGML_BACKEND_SHARED', '-DGGML_CUDA_KVARN',
                   '-DGGML_CUDA_USE_GRAPHS', '-DGGML_SCHED_MAX_COPIES=4', '-DGGML_SHARED',
                   '-Dggml_cuda_EXPORTS', '-DNDEBUG']
    for name in ['qk16-kernels', 'qk16-host-wrapper']:
        run([nvcc, *flags, '--use_fast_math', '--extended-lambda', *definitions, *includes,
             '-c', bridge / (name + '.cu'), '-o', output / (name + '.o')])
    run([nvcc, *flags, '-shared', '-Xlinker=--no-undefined', output / 'qk16-kernels.o',
         output / 'qk16-host-wrapper.o', '-L' + str(build / 'bin'), '-L' + str(stub),
         '-lggml-cuda', '-lggml-base', '-lcublas', '-lcuda', '-ldl',
         '-o', output / 'libqk16-context-gate.so'])
    native = {str(p.relative_to(output)): sha(p) for p in (build / 'bin').iterdir()
              if p.is_file() and not p.is_symlink()}
    for name in ['libbridge-packed.so', 'libbridge-gdn512.so', 'libqk16-context-gate.so']:
        native[name] = sha(output / name)
    receipt = {'status': 'built', 'hardware_qualified': False, 'arch': 'sm89',
               'source_archive_sha256': SOURCE_SHA256, 'source_url': SOURCE_URL,
               'source_sha256': inventory, 'native_sha256': native,
               'compiler': subprocess.check_output([str(nvcc), '--version'], text=True),
               'next': 'Relocate and package with the inherited SM89 CUDA libraries/cubins, then optimize and independently certify the final package.'}
    (output / 'BUILD-RECEIPT.json').write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    main()
