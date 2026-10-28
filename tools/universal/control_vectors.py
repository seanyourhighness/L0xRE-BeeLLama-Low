"""Startup-only control-vector settings shared by Linux and Windows launchers."""
import csv
import hashlib
import io
import json
import math
from pathlib import Path


def add_arguments(parser):
    parser.add_argument('--control-vector', type=Path,
                        help='Optional control-vector GGUF; omitted by default')
    parser.add_argument('--control-vector-mode', choices=['add', 'project'],
                        help='Vector operation; defaults to project when a vector is supplied')
    parser.add_argument('--control-vector-scale', type=float,
                        help='Nonnegative vector strength; defaults to 1')
    parser.add_argument('--control-vector-layer-range', nargs=2, type=int,
                        metavar=('START', 'END'), help='Inclusive layers; defaults to 1 63')
    parser.add_argument('--state-cache-dir', type=Path,
                        help='Optional disk slot-cache root; isolated by runtime and vector identity')


def startup_settings(options, profile_sha256, runtime_sha256=None):
    vector = options.control_vector
    if vector is None:
        if any(value is not None for value in [options.control_vector_mode,
                options.control_vector_scale, options.control_vector_layer_range]):
            raise ValueError('Control-vector options require --control-vector')
        identity = {'enabled': False, 'profile_sha256': profile_sha256,
                    'runtime_sha256': runtime_sha256}
        arguments = []
        environment = {'L0XRE_CVEC_MODE': 'add'}
    else:
        vector = vector.expanduser().resolve()
        mode = options.control_vector_mode or 'project'
        scale = 1.0 if options.control_vector_scale is None else options.control_vector_scale
        first, last = options.control_vector_layer_range or (1, 63)
        if not math.isfinite(scale) or scale < 0:
            raise ValueError('Control-vector scale must be finite and nonnegative')
        if not 1 <= first <= last <= 63:
            raise ValueError('Control-vector layer range must be within 1..63')
        if not vector.is_file():
            raise ValueError('Control-vector file does not exist: ' + str(vector))
        with vector.open('rb') as stream:
            if stream.read(4) != b'GGUF':
                raise ValueError('Control-vector file must be GGUF')
            stream.seek(0)
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        identity = dict(enabled=True, profile_sha256=profile_sha256, runtime_sha256=runtime_sha256,
                        vector_sha256=digest, mode=mode, scale=scale,
                        layer_start=first, layer_end=last)
        # Native scaled-vector parsing is CSV. Quote commas and embedded quotes;
        # the native parser uses the final colon, allowing Windows drive paths.
        writer_buffer = io.StringIO()
        csv.writer(writer_buffer, lineterminator='').writerow([str(vector) + ':' + repr(scale)])
        arguments = ['--control-vector-scaled', writer_buffer.getvalue(),
                     '--control-vector-layer-range', str(first), str(last)]
        environment = {'L0XRE_CVEC_MODE': mode}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    identity['cache_identity'] = fingerprint
    environment['L0XRE_STEERING_ID'] = fingerprint
    return arguments, environment, identity


def namespace_disk_cache(arguments, root, identity, dry_run=False):
    """Keep disk slot files from different steering/model presets separate."""
    args = list(arguments)
    flag = '--slot-save-path'
    if flag in args:
        index = args.index(flag)
        if root is not None:
            raise ValueError('--state-cache-dir conflicts with the profile slot-save path')
        root = Path(args[index + 1])
        del args[index:index + 2]
    if root is None:
        return args, None
    directory = root.expanduser().resolve() / identity['cache_identity']
    if not dry_run:
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    return args + [flag, str(directory)], str(directory)
