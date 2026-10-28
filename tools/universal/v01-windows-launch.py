#!/usr/bin/env python3
"""Start a sealed Windows runtime with its certified memory preset."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

from control_vectors import add_arguments, startup_settings, namespace_disk_cache
from windows_placement import placement, apply_affinity
from vram_preflight import add_preflight_arguments, resolve_preflight, query_free, emit_preflight


def main():
    if sys.argv[1:2] == ['selftest']:
        from selftest import main as selftest_main
        raise SystemExit(selftest_main(sys.argv[2:], launcher=Path(__file__)))
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location('l0xre_profiles', Path(__file__).with_name('v01-launch.py'))
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--arch', choices=['sm86', 'sm89', 'sm120'])
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--memory-profile', choices=['12gb', '16gb'])
    parser.add_argument('-m', '--model', required=True, type=Path)
    parser.add_argument('-md', '--draft-model', required=True, type=Path)
    parser.add_argument('--mmproj', required=True, type=Path)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--numa-node', type=int, help='GPU-local NUMA node; required on systems with multiple nodes')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--qualification-probe', action='store_true')
    add_arguments(parser)
    add_preflight_arguments(parser)
    argv = sys.argv[1:]
    options = parser.parse_args(argv[1:] if argv[:1] == ['serve'] else argv)
    if os.name != 'nt':
        parser.error('Use the Linux launcher on this operating system')
    if not 1 <= options.port <= 65535:
        parser.error('Port must be between 1 and 65535')
    try:
        smi = shutil.which('nvidia-smi')
        if smi is None:
            raise ValueError('nvidia-smi is unavailable; install a supported NVIDIA driver')
        result = subprocess.check_output([smi, '--id=' + options.gpu,
            '--query-gpu=uuid,compute_cap,memory.total,pci.bus_id',
            '--format=csv,noheader,nounits'], text=True)
        uuid, capability, memory, bus = [value.strip() for value in result.strip().split(',')]
        detected = 'sm' + capability.replace('.', '')
        architecture = options.arch or detected
        if architecture != detected:
            raise ValueError(f'Selected GPU is {detected}; requested payload is {architecture}')
        common.require_supported_memory(int(memory))
        archroot = root / 'architectures' / architecture
        manifest = json.loads((archroot / 'V0.1-MANIFEST.json').read_text())
        if manifest.get('platform') != 'windows':
            raise ValueError('This is not a Windows runtime payload')
        config_path, profile_sha, profile = common.select_profile(
            root, archroot, manifest, options.memory_profile, int(memory))
        qualified = manifest['hardware_qualified'] and (profile is None or profile.get('hardware_qualified', False))
        build_only = (architecture in {'sm86', 'sm89'} and manifest.get('build_verified') is True
                      and manifest.get('qualification_status') == 'hardware_validation_in_progress')
        if not qualified and not options.qualification_probe and not build_only:
            raise ValueError(f'{architecture} Windows hardware qualification is pending')
        if common.sha(config_path) != profile_sha:
            raise ValueError('V0.1 profile integrity check failed')
        config = json.loads(config_path.read_text())
        config, preflight = resolve_preflight(config, int(memory), lambda: query_free(smi, options.gpu),
            smi, options.gpu, options.desktop_reserve_mib, options.arena_mib,
            accept_slower=options.accept_slower)
        emit_preflight(preflight, smi, options.gpu, options.dry_run)
        if int(memory) < config['minimum_gpu_memory_mib']:
            raise ValueError(f'This preset requires at least {config["minimum_gpu_memory_mib"]} MiB VRAM; selected GPU has {memory} MiB')
        if manifest.get('loader_links'):
            raise ValueError('Windows payloads must contain real files instead of Linux loader links')
        for name, digest in manifest['runtime_sha256'].items():
            if common.sha(root / name) != digest:
                raise ValueError('V0.1 payload integrity check failed: ' + name)
        models = dict(target=options.model.expanduser().resolve(),
            draft=options.draft_model.expanduser().resolve(), vision=options.mmproj.expanduser().resolve())
        for name, path in models.items():
            if not path.is_file() or common.sha(path) != config['model_sha256'][name]:
                raise ValueError(f'{name} differs from the qualified V0.1 model')
        runtime_identity = hashlib.sha256(json.dumps(dict(architecture=architecture,
            runtime_sha256=manifest['runtime_sha256'], loader_links={}),
            sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        vector_args, vector_env, steering = startup_settings(options, profile_sha, runtime_identity)
        if steering.get('mode') == 'project' and not manifest.get('features', {}).get('control_vector_projection', False):
            raise ValueError('Control-vector projection is not available in this payload')
        def expand(value):
            return value.replace('@PACKAGE_ROOT@', str(root)).replace('@ARCH_ROOT@', str(archroot)).replace('@ARCH@', architecture)
        command = [expand(value) for value in config['argv']] + vector_args
        if any(key in config['env'] for key in ['LD_PRELOAD', 'LD_LIBRARY_PATH', 'L0XRE_VISION_PRELOAD']):
            raise ValueError('Windows preset contains Linux-only loader settings')
        env = {key: value for key, value in os.environ.items() if not key.upper().startswith(
            ('L0XRE_', 'ESCHA_', 'GGML_', 'CUDA_', 'LD_', 'LLAMA_', 'NSYS', 'NVTX_INJECTION'))}
        env.update({key: expand(value) for key, value in config['env'].items()})
        env.update(vector_env)
        env['CUDA_VISIBLE_DEVICES'] = uuid
        env['PATH'] = str(archroot / 'bin') + os.pathsep + env.get('PATH', '')
        command, state_directory = namespace_disk_cache(command, options.state_cache_dir, steering, options.dry_run)
        command += ['-m', str(models['target']), '-md', str(models['draft']), '--mmproj',
            str(models['vision']), '--host', options.host, '--port', str(options.port)]
        node, cpus = placement(config['cpu_threads'], options.numa_node)
    except (OSError, subprocess.SubprocessError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    report = dict(architecture=architecture, gpu=uuid, memory_profile=profile['name'] if profile else 'legacy',
        context_tokens=config.get('context_tokens', config.get('context')), arena_mib=config.get('arena_mib'),
        vram_preflight=preflight,
        hardware_qualified=qualified, build_verified=manifest.get('build_verified', False),
        hardware_validation_in_progress=build_only and not qualified,
        steering=steering, state_cache_directory=state_directory,
        numa_node=node, cpu_affinity=cpus)
    if options.dry_run:
        report.update(argv=command, env={key: env[key] for key in [*config['env'], *vector_env, 'CUDA_VISIBLE_DEVICES', 'PATH']})
        print(json.dumps(report, indent=2))
        return
    if build_only and not qualified:
        print('Windows: build-verified, hardware validation in progress. '
              'Please run l0xre selftest and submit its hardware validation receipt.', file=sys.stderr, flush=True)
    apply_affinity(cpus)
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGBREAK, interrupted)
    child = None
    try:
        child = subprocess.Popen(command, env=env, cwd=root, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        report['owned_server_pid'] = child.pid
        print(json.dumps(report), flush=True)
        code = child.wait()
    except KeyboardInterrupt:
        if child is None:
            raise SystemExit(130)
        child.send_signal(signal.CTRL_BREAK_EVENT)
        try:
            code = child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            child.terminate()
            code = child.wait(timeout=10)
    raise SystemExit(code)


if __name__ == '__main__':
    main()
