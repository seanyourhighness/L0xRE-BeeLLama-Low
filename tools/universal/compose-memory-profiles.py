#!/usr/bin/env python3
"""Stage two sealed presets without modifying or promoting the input payload."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def generated_cache(relative):
    path = Path(relative)
    return '__pycache__' in path.parts or path.suffix in {'.pyc', '.pyo'}


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--arch', choices=['sm86', 'sm89', 'sm120'], required=True)
    parser.add_argument('--profile-12gb', type=Path, required=True)
    parser.add_argument('--profile-16gb', type=Path,
                        help='Required for SM89/SM120; omit for the supported SM86 12GB-only release')
    parser.add_argument('--context-12gb', type=int, choices=[65536, 131072], default=131072,
                        help='Use 65536 only for a separately qualified desktop fallback; never changes input flags')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--require-env', action='append', default=[], metavar='NAME=VALUE',
                        help='require a correctness setting in both input presets; never insert it silently')
    args = parser.parse_args()
    if args.arch != 'sm86' and args.profile_16gb is None:
        parser.error('SM89 and SM120 require both 12GB and 16GB profiles')
    if args.output.exists():
        parser.error('Output must be a new directory')
    source_manifest = args.base / 'architectures' / args.arch / 'V0.1-MANIFEST.json'
    manifest = json.loads(source_manifest.read_text())
    required = dict(manifest.get('required_profile_env',
                                 manifest.get('required_profile_env246', {})))
    for setting in args.require_env:
        if '=' not in setting or not setting.split('=', 1)[0]:
            parser.error('--require-env must be NAME=VALUE')
        key, value = setting.split('=', 1)
        if key in required and required[key] != value:
            parser.error('Required setting conflicts with the base manifest: ' + key)
        required[key] = value
    profiles = {}
    inputs = [('12gb', args.profile_12gb, args.context_12gb, 12000)]
    if args.profile_16gb is not None:
        inputs.append(('16gb', args.profile_16gb, 262144, 16000))
    for name, source, context, minimum in inputs:
        profile = json.loads(source.read_text())
        for key, value in required.items():
            if profile.get('env', {}).get(key) != value:
                parser.error(name + ' must set ' + key + '=' + str(value))
        argv = profile['argv']
        if int(argv[argv.index('-c') + 1]) != context:
            parser.error(name + ' has the wrong context capacity')
        if profile['model_sha256'] != json.loads(args.profile_12gb.read_text())['model_sha256']:
            parser.error('Presets must use the same matched models')
        profile.update(minimum_gpu_memory_mib=minimum, memory_profile=name,
                       context_tokens=context)
        profiles[name] = (source, profile)
    # Refuse stale or cross-architecture inputs before writing any payload.
    for relative, expected in manifest['runtime_sha256'].items():
        if generated_cache(relative):
            continue
        if sha(args.base / relative) != expected:
            parser.error('Base integrity failed: ' + relative)
    for relative, expected in manifest['loader_links'].items():
        if (args.base / relative).readlink().as_posix() != expected:
            parser.error('Base loader link changed: ' + relative)
    shutil.copytree(args.base, args.output, symlinks=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.pyo'))
    manifest['runtime_sha256'] = {path: digest for path, digest in manifest['runtime_sha256'].items()
                                  if not generated_cache(path)}
    manifest['loader_links'] = {path: target for path, target in manifest['loader_links'].items()
                                if not generated_cache(path)}
    archroot = args.output / 'architectures' / args.arch
    (archroot / 'profiles').mkdir(exist_ok=True)
    catalog = {}
    provenance = {}
    for name, (source, config) in profiles.items():
        path = archroot / 'profiles' / (name + '.json')
        path.write_text(json.dumps(config, indent=2) + '\n')
        relative = str(path.relative_to(args.output))
        catalog[name] = dict(path=relative, sha256=sha(path), hardware_qualified=False,
                             status='development; final certification pending')
        provenance[name] = dict(source=str(source.resolve()), source_sha256=sha(source))
        manifest['runtime_sha256'][relative] = sha(path)
    launcher_files = ['v01-launch.py', 'v01-exec-server.py', 'control_vectors.py', 'vram_preflight.py',
                      'selftest.py', 'matched-depth-probe.py']
    if manifest.get('platform') == 'windows':
        launcher_files += ['v01-windows-launch.py', 'windows_placement.py']
    for name in launcher_files:
        target = args.output / 'tools/universal' / name
        shutil.copy2(Path(__file__).with_name(name), target)
        manifest['runtime_sha256'][str(target.relative_to(args.output))] = sha(target)
    legacy = archroot / 'profile.json'
    manifest['runtime_sha256'].pop(str(legacy.relative_to(args.output)), None)
    legacy.unlink(missing_ok=True)
    manifest.pop('profile_sha256', None)
    manifest.update(memory_profiles=catalog, hardware_qualified=False, promoted=False,
                    status='development profile selection; no release certification',
                    profile_composition=dict(base_manifest_sha256=sha(source_manifest),
                                             profiles=provenance))
    if required:
        manifest['required_profile_env'] = required
    (archroot / 'V0.1-MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(output=str(args.output), architecture=args.arch,
                          profiles=list(catalog), promoted=False)))


if __name__ == '__main__':
    main()
