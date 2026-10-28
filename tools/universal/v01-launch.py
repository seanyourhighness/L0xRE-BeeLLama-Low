#!/usr/bin/env python3
"""Launch a sealed V0.1 payload using the existing architecture/profile layout."""
import argparse, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path
from control_vectors import add_arguments, startup_settings, namespace_disk_cache
from vram_preflight import add_preflight_arguments, resolve_preflight, query_free, emit_preflight, wsl_settings

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def cpu_list(value):
    result = set()
    for item in value.strip().split(','):
        ends = item.split('-')
        result.update(range(int(ends[0]), int(ends[-1]) + 1))
    return result

def require_supported_memory(memory_mib):
    if memory_mib < 12000:
        raise ValueError(f'Unsupported GPU: {memory_mib} MiB VRAM; this release requires a card with at least 12GB')

def select_profile(root, archroot, manifest, requested, memory_mib):
    """Choose a sealed preset; never manufacture unqualified flag overrides."""
    require_supported_memory(memory_mib)
    catalog = manifest.get('memory_profiles')
    if catalog is None:
        if requested is not None:
            raise ValueError('This payload has no selectable memory profiles')
        return archroot / 'profile.json', manifest['profile_sha256'], None
    sm86_12_only = archroot.name == 'sm86' and set(catalog) == {'12gb'}
    if set(catalog) != {'12gb', '16gb'} and not sm86_12_only:
        raise ValueError('Payload must declare exactly the 12gb and 16gb profiles')
    name = requested or ('16gb' if memory_mib >= 16000 and '16gb' in catalog else '12gb')
    if name not in catalog:
        raise ValueError(f'{name} is not installed for {archroot.name}; use the validated 12gb preset')
    entry = catalog[name]
    path = (root / entry['path']).resolve()
    if not path.is_relative_to(archroot.resolve()):
        raise ValueError('Memory profile must be inside the selected architecture')
    return path, entry['sha256'], dict(entry, name=name)

def placement(bus, count):
    domain, device, function = bus.lower().split(':')
    pci = f'{int(domain, 16):04x}:{device}:{function}'
    node_file = Path('/sys/bus/pci/devices') / pci / 'numa_node'
    node = int(node_file.read_text()) if node_file.is_file() else -1
    allowed = os.sched_getaffinity(0)
    if node >= 0:
        allowed &= cpu_list((Path('/sys/devices/system/node') / f'node{node}' / 'cpulist').read_text())
    selected, cores = [], set()
    for cpu in sorted(allowed):
        topology = Path('/sys/devices/system/cpu') / f'cpu{cpu}' / 'topology'
        if (topology / 'core_id').is_file():
            core = ((topology / 'physical_package_id').read_text().strip(),
                    (topology / 'core_id').read_text().strip())
        else:
            core = ('unknown', str(cpu))
        if core not in cores:
            cores.add(core)
            selected.append(cpu)
        if len(selected) == count:
            break
    if len(selected) < count:
        raise ValueError(f'The selected GPU needs {count} available CPU cores')
    return node, selected

def main():
    root = Path(__file__).resolve().parents[2]
    arguments = sys.argv[1:]
    if arguments[:1] == ['selftest']:
        from selftest import main as selftest_main
        raise SystemExit(selftest_main(arguments[1:], launcher=Path(__file__)))
    if arguments[:1] == ['serve']:
        arguments = arguments[1:]
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--arch', choices=['sm86', 'sm89', 'sm120'])
    parser.add_argument('--gpu', default=os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0])
    parser.add_argument('--memory-profile', choices=['12gb', '16gb'],
                        help='Select a certified preset; defaults to 16gb on GPUs with at least 16000 MiB, otherwise 12gb. Batch, arena and context settings come from the preset.')
    parser.add_argument('-m', '--model', required=True, type=Path)
    parser.add_argument('-md', '--draft-model', required=True, type=Path)
    parser.add_argument('--mmproj', required=True, type=Path)
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--qualification-probe', action='store_true')
    add_arguments(parser)
    add_preflight_arguments(parser)
    options = parser.parse_args(arguments)
    if not 1 <= options.port <= 65535:
        parser.error('Port must be between 1 and 65535')
    try:
        output = subprocess.check_output([
            shutil.which('nvidia-smi') or '/usr/lib/wsl/lib/nvidia-smi',
            '--id=' + options.gpu,
            '--query-gpu=uuid,compute_cap,memory.total,pci.bus_id',
            '--format=csv,noheader,nounits'], text=True)
        uuid, capability, memory, bus = [value.strip() for value in output.strip().split(',')]
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        parser.error('Cannot identify the selected CUDA GPU: ' + str(error))
    detected = 'sm' + capability.replace('.', '')
    architecture = options.arch or detected
    if architecture != detected:
        parser.error(f'Selected GPU is {detected}; requested payload is {architecture}')
    try:
        require_supported_memory(int(memory))
    except ValueError as error:
        parser.error(str(error))
    archroot = root / 'architectures' / architecture
    manifest_file = archroot / 'V0.1-MANIFEST.json'
    if not manifest_file.is_file():
        parser.error(f'{architecture} V0.1 payload is not installed')
    manifest = json.loads(manifest_file.read_text())
    try:
        config_file, expected_profile_sha, profile = select_profile(
            root, archroot, manifest, options.memory_profile, int(memory))
    except (KeyError, TypeError, ValueError) as error:
        parser.error('Invalid memory profile: ' + str(error))
    qualified = manifest['hardware_qualified'] and (profile is None or profile.get('hardware_qualified', False))
    if not qualified and not options.qualification_probe:
        parser.error(f'{architecture} V0.1 hardware qualification is pending')
    if not config_file.is_file() or sha(config_file) != expected_profile_sha:
        parser.error('V0.1 profile integrity check failed')
    config = json.loads(config_file.read_text())
    try:
        smi = shutil.which('nvidia-smi') or '/usr/lib/wsl/lib/nvidia-smi'
        config, preflight = resolve_preflight(config, int(memory), lambda: query_free(smi, options.gpu),
            smi, options.gpu, options.desktop_reserve_mib, options.arena_mib,
            accept_slower=options.accept_slower)
        emit_preflight(preflight, smi, options.gpu, options.dry_run)
    except (OSError, KeyError, TypeError, ValueError, subprocess.SubprocessError) as error:
        parser.error('VRAM preflight: ' + str(error))
    try:
        runtime_identity = hashlib.sha256(json.dumps(dict(architecture=architecture,
            runtime_sha256=manifest['runtime_sha256'], loader_links=manifest['loader_links']),
            sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        vector_arguments, vector_environment, steering = startup_settings(options, expected_profile_sha, runtime_identity)
        if steering.get('mode') == 'project' and not manifest.get('features', {}).get('control_vector_projection', False):
            raise ValueError('Control-vector projection is not available in this payload')
    except (ValueError, OSError) as error:
        parser.error(str(error))
    if int(memory) < config['minimum_gpu_memory_mib']:
        parser.error(f'This preset requires at least {config["minimum_gpu_memory_mib"]} MiB VRAM; selected GPU has {memory} MiB')
    for name, expected in manifest['runtime_sha256'].items():
        if sha(root / name) != expected:
            parser.error('V0.1 payload integrity check failed: ' + name)
    for name, target in manifest['loader_links'].items():
        if os.readlink(root / name) != target:
            parser.error('V0.1 loader link changed: ' + name)
    models = {'target': options.model.expanduser().resolve(),
              'draft': options.draft_model.expanduser().resolve(),
              'vision': options.mmproj.expanduser().resolve()}
    for name, path in models.items():
        if not path.is_file() or sha(path) != config['model_sha256'][name]:
            parser.error(f'{name} differs from the qualified V0.1 model')
    if config.get('projector_ram_cache') and not options.dry_run:
        cache = Path('/dev/shm/l0xre-v01')
        cache.mkdir(mode=0o700, exist_ok=True)
        if cache.is_symlink() or cache.stat().st_uid != os.getuid():
            parser.error('Projector RAM cache must belong to the current user')
        cached = cache / ('mmproj-' + config['model_sha256']['vision'][:16] + '.gguf')
        if not cached.exists():
            if shutil.disk_usage(cache).free < models['vision'].stat().st_size + 64 * 1024 * 1024:
                parser.error('Not enough shared memory for the qualified projector RAM cache')
            temporary = cache / ('projector-' + str(os.getpid()) + '.tmp')
            shutil.copy2(models['vision'], temporary)
            temporary.chmod(0o600)
            os.replace(temporary, cached)
        if cached.is_symlink() or cached.stat().st_uid != os.getuid() or sha(cached) != config['model_sha256']['vision']:
            parser.error('Projector RAM cache integrity check failed')
        models['vision'] = cached
    def expand(value):
        return value.replace('@PACKAGE_ROOT@', str(root)).replace('@ARCH_ROOT@', str(archroot)).replace('@ARCH@', architecture)
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(('L0XRE_', 'ESCHA_', 'GGML_', 'CUDA_', 'LD_', 'LLAMA_', 'NSYS', 'NVTX_INJECTION'))}
    env.update({key: expand(value) for key, value in config['env'].items()})
    env.update(vector_environment)
    env['CUDA_VISIBLE_DEVICES'] = uuid
    try:
        wsl = wsl_settings(env, config, [path.stat().st_size for path in models.values()])
    except (OSError, ValueError, StopIteration, IndexError) as error:
        parser.error('WSL preflight: ' + str(error))
    if wsl:
        for warning in wsl['warnings']:
            print('WARNING: ' + warning, file=sys.stderr, flush=True)
    command = [expand(value) for value in config['argv']]
    command += vector_arguments
    try:
        command, state_directory = namespace_disk_cache(command, options.state_cache_dir, steering, options.dry_run)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    command += ['-m', str(models['target']), '-md', str(models['draft']),
                '--mmproj', str(models['vision']), '--host', options.host, '--port', str(options.port)]
    node, cpus = placement(bus, config['cpu_threads'])
    # Apply CUDA preloads only in the server, after CPU/NUMA placement. This
    # preserves the qualified launcher's allocator and interposition order.
    env['L0XRE_VISION_PRELOAD'] = env.pop('LD_PRELOAD')
    command = [sys.executable, str(root / 'tools/universal/v01-exec-server.py'), *command]
    if shutil.which('numactl'):
        prefix = ['numactl', '--physcpubind=' + ','.join(map(str, cpus))]
        if node >= 0:
            prefix += ['--preferred=' + str(node)]
        command = prefix + command
    elif not options.dry_run:
        os.sched_setaffinity(0, set(cpus))
    if options.dry_run:
        print(json.dumps(dict(architecture=architecture, gpu=uuid, numa_node=node,
                              memory_profile=profile['name'] if profile else 'legacy',
                              context_tokens=config.get('context_tokens', config.get('context')),
                              arena_mib=config.get('arena_mib'),
                              vram_preflight=preflight,
                              wsl=wsl,
                              steering=steering, state_cache_directory=state_directory,
                              cpu_affinity=cpus, hardware_qualified=qualified,
                              argv=command, env={key: env[key] for key in [*config['env'], *vector_environment, 'CUDA_VISIBLE_DEVICES', 'L0XRE_VISION_PRELOAD'] if key in env}), indent=2))
        return
    print(json.dumps(dict(release='V0.1', architecture=architecture, gpu=uuid,
                          memory_profile=profile['name'] if profile else 'legacy',
                          vram_preflight=preflight,
                          wsl=wsl,
                          steering=steering, state_cache_directory=state_directory,
                          numa_node=node, cpu_affinity=cpus, port=options.port)), flush=True)
    os.execvpe(command[0], command, env)

if __name__ == '__main__':
    main()
