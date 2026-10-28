"""Run the shipped hardware-validation protocol against an owned local server."""
import argparse
import copy
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request


ISSUE_URL = ('https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/issues/new'
             '?template=hardware-validation-report.yml')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def configuration_signature(config, manifest, boot):
    """Bind references to numerical settings and binaries, excluding host paths.

    Reference files and launcher metadata are not numerical inputs. Excluding
    them also avoids making the reference hash depend on its own contents.
    """
    argv = list(config['argv'])
    flag = '--shared-device-memory-mib'
    argv[argv.index(flag) + 1] = str(boot['arena_mib'])
    decision = boot.get('vram_preflight', {})
    if 'ubatch' in decision:
        argv[argv.index('-ub') + 1] = str(decision['ubatch'])
    keys = ['enabled', 'vector_sha256', 'mode', 'scale', 'layer_start', 'layer_end']
    steering = {key: boot['steering'][key] for key in keys if key in boot['steering']}
    prefix = 'architectures/' + boot['architecture'] + '/'
    runtime = {key.replace('\\', '/'): value for key, value in manifest['runtime_sha256'].items()
               if key.replace('\\', '/').startswith(tuple(prefix + p for p in ['bin/', 'bridge/', 'deps/']) + ('shared/',))}
    identity = dict(architecture=boot['architecture'], argv=argv, env=config['env'],
                    model_sha256=config['model_sha256'], steering=steering, runtime=runtime)
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def populated(response):
    choices = response.get('choices', [])
    if len(choices) != 1:
        return False
    rows = (choices[0].get('logprobs') or {}).get('content', [])
    try:
        return bool(rows) and all(len(row.get('top_logprobs', [])) == 10
            and math.isfinite(float(row['logprob']))
            and all(math.isfinite(float(item['logprob'])) for item in row['top_logprobs'])
            for row in rows)
    except (KeyError, TypeError, ValueError):
        return False


def vision_checks(response, reference):
    content = response['choices'][0]['message'].get('content', '')
    lower = content.lower()
    rgb = [tuple(int(value[i:i + 2], 16) for i in [0, 2, 4])
           for value in re.findall(r'#([0-9a-fA-F]{6})', content)]
    colors = reference.get('colors', {})
    return dict(printed_text=reference['printed_text'] in content,
        shapes=all(shape.lower() in lower for shape in reference['shapes']),
        colors=all(name.lower() in lower or any(sum((a - b) ** 2 for a, b in zip(expected, actual)) <= 70 ** 2
                                               for actual in rgb)
                   for name, expected in colors.items()))


def exercise(api, reference, output):
    """Keep strict token/top10 equality; HTTP success alone never passes a check."""
    checks, records = {}, []

    def call(label, endpoint, body):
        started = time.monotonic()
        response = api(endpoint, body)
        records.append(dict(case=label, request=body, response=response,
                            seconds=time.monotonic() - started))
        return response

    try:
        body = copy.deepcopy(reference['greedy']['request'])
        if (body.get('temperature') != 0 or body.get('cache_prompt') is not False
                or body.get('logprobs') is not True or body.get('top_logprobs') != 10):
            raise ValueError('Shipped greedy reference must request cold greedy output and populated top10')
        cold = call('greedy-cold', '/v1/chat/completions', body)
        checks['greedy_top10_populated'] = populated(cold)
        checks['greedy_reference_exact'] = cold.get('choices') == reference['greedy']['choices']
        if not all(checks.values()):
            raise ValueError('Greedy tokens or populated top10 differ from the shipped reference')
        follow = copy.deepcopy(body)
        follow['messages'] += [cold['choices'][0]['message'],
                               dict(role='user', content='Summarize your answer in one sentence.')]
        follow.update(max_tokens=32, cache_prompt=True)
        live = call('cache-live', '/v1/chat/completions', follow)
        follow = dict(follow, cache_prompt=False)
        repeated = call('cache-cold', '/v1/chat/completions', follow)
        again = call('cache-cold-repeat', '/v1/chat/completions', follow)
        checks.update(cache_live_reused=live.get('timings', {}).get('cache_source') == 'live'
                      and live.get('timings', {}).get('cache_n', 0) > 0,
            cache_cold_not_reused=all(b.get('timings', {}).get('cache_n') == 0 for b in [repeated, again]),
            cache_top10_populated=all(populated(b) for b in [live, repeated, again]),
            cache_tokens_top10_exact=live.get('choices') == repeated.get('choices') == again.get('choices'))
        if not all(checks.values()):
            raise ValueError('Live/cold cache tokens, populated top10 or reuse checks failed')
        vision = call('vision', '/v1/chat/completions', reference['vision']['request'])
        checks.update({'vision_' + key: value for key, value in vision_checks(vision, reference['vision']).items()})
        if not all(checks.values()):
            raise ValueError('Vision ground-truth check failed')
        performance = {}
        for name, prompt in [('narrative', 'Describe a quiet walk through a forest after rain in about 180 words.'),
                             ('code', 'Write a Python function that merges two sorted integer lists without using sorted(). Include an example.')]:
            response = call('bench-' + name, '/completion', dict(prompt=prompt, n_predict=192,
                temperature=0, seed=17, cache_prompt=False, return_tokens=True))
            timing = response.get('timings', {})
            tokens = response.get('tokens', [])
            rate = timing.get('predicted_per_second')
            checks['bench_' + name] = bool(tokens) and type(rate) in (float, int) and math.isfinite(rate) and rate > 0
            performance[name] = dict(tokens=len(tokens), decode_tps=rate, timings=timing)
        if not all(checks.values()):
            raise ValueError('Short benchmark failed to produce tokens and valid timing')
        return dict(passed=True, checks=checks, performance=performance,
                    performance_scope='Selftest 192-token narrative/code samples; not canonical bench.sh')
    finally:
        save(output / 'requests.json', records)
        save(output / 'checks.json', checks)


def inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Bundle reference path escapes its architecture')
    return path


def load_reference(root, boot):
    arch = root / 'architectures' / boot['architecture']
    manifest_path = arch / 'V0.1-MANIFEST.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    entry = manifest.get('selftest', {}).get('reference')
    if not entry:
        raise ValueError('This bundle has no shipped selftest reference; install a complete release bundle')
    path = inside(root, entry['path'])
    if not path.is_relative_to(arch.resolve()) or sha(path) != entry['sha256']:
        raise ValueError('Selftest reference integrity check failed')
    profile_entry = manifest['memory_profiles'][boot['memory_profile']]
    profile = inside(root, profile_entry['path'])
    if not profile.is_relative_to(arch.resolve()) or sha(profile) != profile_entry['sha256']:
        raise ValueError('Selftest profile integrity check failed')
    config = json.loads(profile.read_text(encoding='utf-8'))
    signature = configuration_signature(config, manifest, boot)
    reference = json.loads(path.read_text(encoding='utf-8'))
    if reference.get('architecture') != boot['architecture']:
        raise ValueError('Shipped selftest reference is for a different GPU architecture')
    variants = [r for r in reference['variants'] if r['configuration_sha256'] == signature]
    if len(variants) != 1:
        raise ValueError('No unique shipped reference for this preset/arena/vector combination. Try the default vector-OFF preset.')
    return variants[0], dict(manifest_sha256=sha(manifest_path), profile_sha256=sha(profile),
        configuration_sha256=signature, reference_sha256=sha(path),
        binary_and_library_sha256={key: value for key, value in manifest['runtime_sha256'].items()
                                  if '/bin/' in key or '/bridge/' in key or '/deps/' in key},
        model_sha256=config['model_sha256'])


def api_for(port):
    def call(endpoint, body=None):
        request = urllib.request.Request(f'http://127.0.0.1:{port}' + endpoint,
            data=None if body is None else json.dumps(body).encode(),
            headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=3 if body is None else 600) as response:
            return json.load(response)
    return call


def stop_owned(worker):
    if worker is None:
        return dict(started=False, launcher_exited=True)
    if worker.poll() is None:
        if os.name == 'nt':
            worker.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            os.killpg(worker.pid, signal.SIGINT)
        try:
            worker.wait(timeout=35)
        except subprocess.TimeoutExpired:
            if os.name == 'nt':
                subprocess.run(['taskkill', '/PID', str(worker.pid), '/T', '/F'], capture_output=True, check=True)
            else:
                os.killpg(worker.pid, signal.SIGKILL)
            worker.wait(timeout=10)
    return dict(started=True, launcher_pid=worker.pid, launcher_returncode=worker.returncode,
                launcher_exited=worker.poll() is not None)


def main(arguments=None, *, launcher=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--output', type=Path, help='New receipt directory (never overwrites a previous run)')
    parser.add_argument('--deep', action='store_true', help='Also run a 128K depth smoke on a preset supporting at least 128K')
    parser.add_argument('--startup-timeout', type=float, default=360)
    options, forwarded = parser.parse_known_args(arguments)
    if not math.isfinite(options.startup_timeout) or options.startup_timeout <= 0:
        parser.error('--startup-timeout must be positive and finite')
    launcher = Path(launcher or Path(__file__).with_name('v01-windows-launch.py' if os.name == 'nt' else 'v01-launch.py')).resolve()
    root = launcher.parents[2]
    output = (options.output or Path.cwd() / ('l0xre-selftest-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))).resolve()
    if output.exists():
        parser.error('Output already exists; choose a new receipt directory')
    output.mkdir(parents=True, exist_ok=False)
    receipt = dict(schema_version=1, started_utc=datetime.now(timezone.utc).isoformat(),
        passed=False, os=platform.platform(), kernel=platform.release(),
        wsl2='microsoft' in platform.release().lower(), output=str(output), issue_url=ISSUE_URL)
    worker = None
    stop = threading.Event()
    monitor = None
    readings, errors = [], []
    native_handle = None
    try:
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        base = [sys.executable, '-E', '-s', '-X', 'utf8', str(launcher), 'serve', *forwarded,
                '--host', '127.0.0.1', '--port', str(port)]
        child_env = dict(os.environ, L0XRE_NONINTERACTIVE='1')
        plan = subprocess.run([*base, '--dry-run'], env=child_env, text=True, capture_output=True, stdin=subprocess.DEVNULL, timeout=300)
        receipt['launcher_preflight_returncode'] = plan.returncode
        receipt['launcher_preflight_stderr'] = plan.stderr
        if plan.returncode:
            raise ValueError('Launcher preflight refused: ' + plan.stderr.strip())
        boot = json.loads(plan.stdout)
        receipt['startup'] = boot
        reference, identity = load_reference(root, boot)
        receipt.update(identity)
        if options.deep and boot['context_tokens'] < 131072:
            raise ValueError('--deep needs a validated preset with at least 128K context')
        smi = shutil.which('nvidia-smi') or '/usr/lib/wsl/lib/nvidia-smi'
        def gpu_query(fields):
            return subprocess.check_output([smi, '--id=' + boot['gpu'], '--query-gpu=' + fields,
                '--format=csv,noheader,nounits'], text=True, timeout=15).strip()
        fields = [part.strip() for part in gpu_query('name,uuid,compute_cap,driver_version,memory.total,memory.free').split(',')]
        if len(fields) != 6:
            raise ValueError('Cannot read GPU metadata')
        receipt['gpu'] = dict(zip(['name', 'uuid', 'compute_capability', 'driver', 'total_mib', 'free_before_load_mib'], fields))
        for key in ['total_mib', 'free_before_load_mib']:
            receipt['gpu'][key] = int(receipt['gpu'][key])
        def memory_monitor():
            while not stop.is_set():
                try:
                    used, free = map(int, gpu_query('memory.used,memory.free').split(','))
                    readings.append(dict(used_mib=used, free_mib=free))
                except (OSError, ValueError, subprocess.SubprocessError) as error:
                    errors.append(str(error))
                stop.wait(.5)
        monitor = threading.Thread(target=memory_monitor, daemon=True)
        monitor.start()
        # Lock the already validated dry-run choice; startup still checks current free VRAM.
        command = [*base, '--arena-mib', str(boot['arena_mib'])]
        kwargs = dict(creationflags=subprocess.CREATE_NEW_PROCESS_GROUP) if os.name == 'nt' else dict(start_new_session=True)
        with (output / 'server.log').open('x', encoding='utf-8') as log:
            worker = subprocess.Popen(command, env=child_env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, **kwargs)
        api = api_for(port)
        begun = time.monotonic()
        while True:
            if worker.poll() is not None:
                raise ValueError('Owned server exited during startup; see server.log')
            try:
                if api('/health').get('status') == 'ok':
                    break
            except (OSError, ValueError):
                pass
            if time.monotonic() - begun > options.startup_timeout:
                raise TimeoutError('Owned server startup timeout')
            time.sleep(.5)
        if os.name == 'nt':
            report = None
            with (output / 'server.log').open(encoding='utf-8', errors='replace') as log:
                for _ in range(512):
                    line = log.readline()
                    if not line:
                        break
                    try:
                        value = json.loads(line)
                        if isinstance(value, dict) and value.get('owned_server_pid'):
                            report = value
                            break
                    except ValueError:
                        pass
            if report is None:
                raise ValueError('Native launcher did not report its owned server PID')
            kernel = ctypes.WinDLL('kernel32', use_last_error=True)
            kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
            kernel.OpenProcess.restype = ctypes.c_void_p
            kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
            kernel.CloseHandle.argtypes = [ctypes.c_void_p]
            native_handle = kernel.OpenProcess(0x00100000, False, report['owned_server_pid'])
            if not native_handle:
                raise ValueError('Cannot retain the owned native process handle')
            receipt['owned_server_pid'] = report['owned_server_pid']
        receipt['protocol'] = exercise(api, reference, output)
        if options.deep:
            command = [sys.executable, str(launcher.with_name('matched-depth-probe.py')),
                '--url', f'http://127.0.0.1:{port}', '--context', str(boot['context_tokens']),
                '--depth', '1024', '130048', '1024', '--repeats', '1', '--output', str(output / 'deep')]
            with (output / 'deep.log').open('x') as log:
                done = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=7200)
            deep = json.loads((output / 'deep/results.json').read_text())
            if done.returncode or not deep['complete'] or not all(all(row['checks'].values()) for row in deep['results']):
                raise ValueError('128K depth smoke failed')
            receipt['deep'] = deep
        receipt['passed'] = True
    except (Exception, KeyboardInterrupt) as error:
        receipt['error'] = type(error).__name__ + ': ' + str(error)
    finally:
        try:
            receipt['cleanup'] = stop_owned(worker)
            if native_handle is not None:
                exited = kernel.WaitForSingleObject(native_handle, 10000) == 0
                receipt['cleanup']['native_process_exited'] = exited
                if not exited:
                    receipt['passed'] = False
                    receipt['cleanup']['error'] = 'Owned native server has not exited'
        except Exception as error:
            receipt['passed'] = False
            receipt['cleanup'] = dict(error=str(error))
        finally:
            if native_handle is not None:
                kernel.CloseHandle(native_handle)
        stop.set()
        if monitor:
            monitor.join(timeout=20)
        receipt['memory'] = dict(peak_adapter_used_mib=max((r['used_mib'] for r in readings), default=None),
            minimum_adapter_free_mib=min((r['free_mib'] for r in readings), default=None),
            samples=len(readings), errors=errors, scope='Total adapter usage, including other desktop processes')
        if worker is not None and not readings:
            receipt['passed'] = False
            receipt.setdefault('error', 'No VRAM samples could be collected')
        receipt['finished_utc'] = datetime.now(timezone.utc).isoformat()
        if (output / 'checks.json').is_file():
            receipt['checks'] = json.loads((output / 'checks.json').read_text())
        save(output / 'receipt.json', receipt)
        summary = ('PASS' if receipt['passed'] else 'FAIL') + ' — L0xRE hardware selftest\n'
        summary += 'Receipt: ' + str(output / 'receipt.json') + '\n'
        if receipt.get('error'):
            summary += receipt['error'] + '\n'
        if receipt.get('protocol'):
            for name, timing in receipt['protocol']['performance'].items():
                summary += f'{name}: {timing["decode_tps"]:.2f} tok/s ({timing["tokens"]} tokens)\n'
        summary += 'Peak adapter VRAM: ' + str(receipt['memory']['peak_adapter_used_mib']) + ' MiB\n'
        summary += 'Submit receipt.json using Hardware validation report:\n' + ISSUE_URL + '\n'
        (output / 'SUMMARY.txt').write_text(summary, encoding='utf-8')
        print(summary, end='', flush=True)
    return 0 if receipt['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
