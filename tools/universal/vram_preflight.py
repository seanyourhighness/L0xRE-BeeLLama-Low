"""Choose only receipt-backed arena sizes before loading CUDA weights."""
import copy
import math
import os
from pathlib import Path
import platform
import subprocess
import sys


ADVICE = ('Close browsers, overlays, game launchers and GPU wallpapers. If your board has '
          'integrated graphics, plug the monitor into the motherboard so the GPU is free. '
          'On a 16GB card, the validated 12GB preset may need less memory.')


def wsl_settings(env, config, model_sizes, *, kernel=None, meminfo=None):
    """Apply the WSL allocator guard and warn without changing validated cache budgets."""
    kernel = platform.release() if kernel is None else kernel
    if 'microsoft' not in kernel.lower():
        return None
    warnings = []
    for key in ['PYTORCH_ALLOC_CONF', 'PYTORCH_CUDA_ALLOC_CONF']:
        parts = env.get(key, '').split(',')
        kept = [p for p in parts if p.strip().replace(' ', '').lower() != 'expandable_segments:true']
        if len(kept) != len(parts):
            if any(p.strip() for p in kept):
                env[key] = ','.join(kept)
            else:
                env.pop(key, None)
            warnings.append(f'WSL: removed unsupported expandable_segments:True from {key}.')
    meminfo = Path('/proc/meminfo').read_text() if meminfo is None else meminfo
    total_kib = int(next(row.split()[1] for row in meminfo.splitlines() if row.startswith('MemTotal:')))
    if total_kib <= 0:
        raise ValueError('WSL VM memory limit must be positive')
    argv = config['argv']
    cache_mib = int(argv[argv.index('--cache-ram') + 1]) if '--cache-ram' in argv else int(config.get('cache_ram_mib', 0))
    estimated_mib = cache_mib + sum(model_sizes) / 2**20
    total_mib = total_kib / 1024
    if estimated_mib > total_mib * .7:
        warnings.append(f'WSL: host cache plus model files need about {estimated_mib:.0f} MiB, '
                        f'over 70% of the VM memory limit ({total_mib:.0f} MiB). '
                        'Increase the WSL memory limit or select a smaller validated preset. '
                        'The launcher has kept the validated cache setting unchanged.')
    return dict(kernel=kernel, vm_memory_mib=total_mib, cache_mib=cache_mib,
                estimated_cache_and_model_mib=estimated_mib, warnings=warnings)


def add_preflight_arguments(parser):
    parser.add_argument('--desktop-reserve-mib', type=int,
                        help='Set the minimized-desktop warning level; safety margin still applies')
    parser.add_argument('--arena-mib', type=int,
                        help='Explicitly select an already validated arena; never bypasses the free-VRAM check')


    parser.add_argument('--accept-slower', action='store_true',
                        help='Explicitly accept the fastest fitting measured slower rung')


def query_free(smi, gpu):
    value = subprocess.check_output([smi, '--id=' + gpu,
        '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True,
        timeout=15).strip()
    free = int(value)
    if free < 0:
        raise ValueError('Driver returned negative free VRAM')
    return free


class PreflightRefusal(ValueError):
    """A valid policy needs more VRAM or consent before loading anything."""
    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


def speed_cost(entry, fastest):
    """Use the largest measured headline loss; never average away a slow metric."""
    speeds, baseline = entry['headline_speeds'], fastest['headline_speeds']
    common = set(speeds) & set(baseline)
    if not common:
        raise ValueError('Validated rungs need matching measured headline speeds')
    return max(0., max((1 - speeds[k] / baseline[k]) * 100 for k in common))


def choose_arena(config, total, free, reserve_override=None, arena_override=None, *, consent='none'):
    """Choose in measured speed order, using actual free memory and explicit consent.

    Desktop reserve is a warning only. required_free_mib includes the measured
    vision/drafter peak; the positive safety margin is always enforced.
    """
    policy = config.get('vram_preflight')
    if not isinstance(policy, dict):
        raise ValueError('Preset lacks receipt-backed VRAM preflight data; rebuild the bundle')
    default_reserve = policy['desktop_reserve_mib']
    if 'microsoft' in platform.release().lower():
        default_reserve = policy.get('wsl_desktop_reserve_mib', default_reserve)
    reserve = default_reserve if reserve_override is None else reserve_override
    margin = policy['safety_margin_mib']
    if any(type(v) is not int or v < 0 for v in [total, free, reserve, margin]) or free > total:
        raise ValueError('Invalid VRAM capacity, reserve or safety margin')
    if not margin:
        raise ValueError('VRAM safety margin must be positive')
    if consent not in ('none', 'interactive', 'flag', 'env'):
        raise ValueError('Invalid slower-speed consent source')
    entries = policy['validated_arenas']
    if not isinstance(entries, list) or not entries:
        raise ValueError('Preset has no validated arena measurements')
    seen = set()
    for entry in entries:
        arena, required, ubatch = entry['arena_mib'], entry['required_free_mib'], entry['ubatch']
        speeds = entry['headline_speeds']
        if (any(type(v) is not int or v <= 0 for v in [arena, required, ubatch])
                or not entry.get('evidence') or arena in seen
                or not isinstance(speeds, dict) or not speeds
                or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
                       for v in speeds.values())):
            raise ValueError('Invalid or duplicate validated arena measurement')
        speed_cost(entry, entries[0])
        seen.add(arena)
    if arena_override is not None and (type(arena_override) is not int or arena_override not in seen):
        raise ValueError(f'Arena {arena_override} MiB has not been validated for this preset')
    argv = config['argv']
    for flag in ['--shared-device-memory-mib', '-ub']:
        if argv.count(flag) != 1 or argv.index(flag) + 1 >= len(argv):
            raise ValueError(f'Preset must declare exactly one {flag}')
    candidates = [(i, e) for i, e in enumerate(entries)
                  if arena_override is None or e['arena_mib'] == arena_override]
    fits = [(i, e) for i, e in candidates if e['required_free_mib'] + margin <= free]
    report = dict(free_mib=free, total_mib=total, desktop_reserve_mib=reserve,
        safety_margin_mib=margin, available_mib=free,
        required_mib=entries[0]['required_free_mib'] + margin,
        explicit_arena=arena_override is not None, reserve_override=reserve_override is not None,
        fitting_rungs=[dict(e, rung=i, speed_cost_pct=speed_cost(e, entries[0])) for i, e in fits])
    if not fits:
        required = min(e['required_free_mib'] for _, e in candidates) + margin
        raise PreflightRefusal(f'VRAM preflight refused before loading: free {free} MiB; '
            f'required {required} MiB including safety margin. {ADVICE}', report)
    rung, selected = fits[0]
    if rung and arena_override is None and consent == 'none':
        raise PreflightRefusal(f'Full-speed preset needs {report["required_mib"]} MiB; '
            f'free {free} MiB. Slower speeds require explicit consent. {ADVICE}', report)
    result = copy.deepcopy(config)
    for flag, value in [('--shared-device-memory-mib', selected['arena_mib']), ('-ub', selected['ubatch'])]:
        result['argv'][argv.index(flag) + 1] = str(value)
    result.update(arena_mib=selected['arena_mib'], qualified_ubatch=selected['ubatch'])
    source = 'explicit-arena' if rung and arena_override is not None else consent if rung else 'none'
    report.update(rung=rung, arena_mib=selected['arena_mib'], chosen_arena_mib=selected['arena_mib'],
        ubatch=selected['ubatch'], required_mib=selected['required_free_mib'] + margin,
        evidence=selected['evidence'], consent=source, speed_cost_pct=speed_cost(selected, entries[0]),
        reason='full-speed default' if not rung else 'reduced speed explicitly accepted',
        warning=free - selected['required_free_mib'] - margin < reserve)
    return result, report


def show_memory(report, smi, gpu):
    print(f'GPU memory: free {report["free_mib"]} MiB; full-speed required '
          f'{report["required_mib"]} MiB; in use {report["total_mib"] - report["free_mib"]} MiB.',
          file=sys.stderr, flush=True)
    try:
        processes = subprocess.check_output([smi, '--id=' + gpu,
            '--query-compute-apps=pid,process_name,used_gpu_memory',
            '--format=csv,noheader,nounits'], text=True, timeout=15).strip()
        if processes:
            print('GPU processes (WDDM may omit usage):\n' + processes, file=sys.stderr, flush=True)
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    print(ADVICE, file=sys.stderr, flush=True)
    if report['free_mib'] - report['required_mib'] < report['desktop_reserve_mib']:
        print(f'WARNING: full-speed loading would leave less than the '
              f'{report["desktop_reserve_mib"]} MiB minimized-desktop warning level.', file=sys.stderr)
    for row in report['fitting_rungs']:
        if row['rung']:
            speeds = ', '.join(f'{k}: {v:g} tok/s' for k, v in row['headline_speeds'].items())
            print(f'Slower rung {row["rung"]}: arena {row["arena_mib"]} MiB / '
                  f'ub{row["ubatch"]}; {speeds}; up to {row["speed_cost_pct"]:.1f}% slower.',
                  file=sys.stderr, flush=True)


def resolve_preflight(config, total, free_query, smi, gpu, reserve_override=None,
                      arena_override=None, *, accept_slower=False, interactive=None, environ=None):
    """Retry queries the driver again; no server can start without this returning."""
    environ = os.environ if environ is None else environ
    consent = 'flag' if accept_slower else 'env' if environ.get('L0XRE_ACCEPT_SLOWER') == '1' else 'none'
    if interactive is None:
        interactive = sys.stdin.isatty() and environ.get('L0XRE_NONINTERACTIVE') != '1'
    while True:
        try:
            return choose_arena(config, total, free_query(), reserve_override, arena_override, consent=consent)
        except PreflightRefusal as error:
            show_memory(error.report, smi, gpu)
            if not interactive:
                raise ValueError(str(error) + ' Non-interactive launch refused; use --accept-slower '
                                 'or L0XRE_ACCEPT_SLOWER=1 to consent to a fitting slower rung.') from error
            has_slower = bool(error.report['fitting_rungs']) and arena_override is None
            prompt = '[R] Retry / [S] Start slower / [Q] Quit' if has_slower else '[R] Retry / [Q] Quit'
            while True:
                print(prompt, file=sys.stderr, flush=True)
                try:
                    choice = input().strip().lower()
                except (EOFError, OSError):
                    raise ValueError('VRAM preflight quit before loading') from error
                if choice == 'q':
                    raise ValueError('VRAM preflight quit before loading') from error
                if choice == 'r':
                    break
                if choice == 's' and has_slower:
                    consent = 'interactive'
                    break


def emit_preflight(report, smi, gpu, dry_run=False):
    if report['warning']:
        print(f'WARNING: loading leaves less than the {report["desktop_reserve_mib"]} MiB '
              'minimized-desktop warning level. ' + ADVICE, file=sys.stderr, flush=True)
    if not dry_run:
        label = 'full-speed arena' if report['rung'] == 0 else 'reduced-speed arena'
        accepted = '' if report['rung'] == 0 else ' (user accepted)'
        print(f'VRAM preflight: {label} {report["arena_mib"]} MiB{accepted}; '
              f'free {report["free_mib"]}, required {report["required_mib"]} MiB; '
              f'ubatch {report["ubatch"]}; consent {report["consent"]}; '
              f'speed cost {report["speed_cost_pct"]:.1f}%.', flush=True)
