#!/usr/bin/env python3
"""Frozen RTX4090 correctness, CPU vision, restart and mixed-depth soak gates."""
import argparse
import base64
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time
import urllib.request

ROOT = Path('/home/sean/work/l0xre-r6-4090-20261010')
URL = 'http://127.0.0.1:30174'
UUID = 'GPU-c4a782a6-fd5a-78c2-7d77-604c01cbffb8'
PROMPTS = ['Write a Python implementation of quicksort with comments explaining each step.',
           'Write a detailed 800-word essay explaining transformer attention.']
ERRORS = re.compile(r'CUDA error|illegal memory|device-side assert|GPU is lost|CUBLAS_STATUS_EXECUTION_FAILED', re.I)


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def post(path, payload, timeout=900):
    request = urllib.request.Request(URL + path, json.dumps(payload).encode(), {'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def digest(tokens):
    return hashlib.sha256(json.dumps(tokens).encode()).hexdigest()


def payload(prompt, count=128, temperature=0, seed=42):
    return {'prompt': prompt, 'n_predict': count, 'temperature': temperature, 'seed': seed,
            'top_k': 1 if temperature == 0 else 20, 'top_p': .95, 'min_p': 0,
            'ignore_eos': True, 'cache_prompt': False, 'return_tokens': True, 'reasoning_loop_guard': 'off'}


@contextmanager
def boot(name, serial=False, vision=False):
    out = ROOT / 'probes' / name
    out.mkdir(parents=True, exist_ok=False)
    frozen = json.loads((ROOT / 'FROZEN-CANDIDATE.json').read_text())
    package = Path(frozen['package'])
    if hashlib.sha256((package / 'SHA256SUMS').read_bytes()).hexdigest() != frozen['inventory_sha256']:
        raise RuntimeError('Frozen candidate inventory changed')
    subprocess.run(['sha256sum', '-c', '--quiet', 'SHA256SUMS'], cwd=package, check=True, stdout=subprocess.DEVNULL)
    installed = json.loads((ROOT / 'current-dry-run.json').read_text())
    command = installed['argv'] + ['--port', '30174']
    env = {k: v for k, v in os.environ.items() if not k.startswith(('L0XRE_', 'ESCHA_', 'GGML_'))
           and k not in ['LD_PRELOAD', 'LD_LIBRARY_PATH']}
    env.update(installed['env'])
    env['CUDA_VISIBLE_DEVICES'] = UUID
    if serial:
        filtered = []
        index = 0
        while index < len(command):
            if command[index].startswith('--spec-') or command[index] == '-md':
                index += 2
            else:
                filtered.append(command[index]); index += 1
        command = filtered + ['--spec-type', 'none']
        env['L0XRE_GDN_COMPACT_RS'] = '0'
    if vision:
        model = Path('/home/sean/models/qwen38-mmproj-reference/mmproj-Qwen3.8-27B-Q8_0.gguf')
        if hashlib.file_digest(model.open('rb'), 'sha256').hexdigest() != '2e968a6af97ce35d8971890b257b9b7edabf20ad91450501fa53162a19ee33eb':
            raise RuntimeError('CPU projector identity differs')
        command += ['--mmproj', str(model), '--verbosity', '4']
    save(out / 'launch.json', {'argv': command, 'env': installed['env'] | {'CUDA_VISIBLE_DEVICES': UUID,
         'L0XRE_GDN_COMPACT_RS': env['L0XRE_GDN_COMPACT_RS']}, 'serial': serial, 'vision': vision,
         'scope': 'Serial reference disables compact recurrent state; release candidate retains it.'})
    done = threading.Event()
    faults = []
    server = None
    watcher = None
    result = {'state': 'starting', 'started': time.time(), 'hardware_qualified': False}
    save(out / 'status.json', result)
    try:
        with (out / 'server.log').open('w') as log:
            server = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)

        def watch():
            try:
                with (out / 'telemetry.jsonl').open('w') as data:
                    while not done.wait(2):
                        raw = subprocess.check_output(['sudo', '-n', 'cat', '/sys/kernel/tracing/instances/l0xre_r6_4090/trace'], text=True, timeout=10)
                        if 'mce_record:' in raw:
                            raise RuntimeError('New raw machine-check event')
                        row = subprocess.check_output(['nvidia-smi', '--id=' + UUID,
                              '--query-gpu=memory.used,temperature.gpu,power.draw,power.limit,utilization.gpu,clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.hw_thermal_slowdown',
                              '--format=csv,noheader,nounits'], text=True, timeout=10).strip()
                        data.write(json.dumps({'time': time.time(), 'gpu': row}) + '\n'); data.flush()
                        if float(row.split(',')[1]) >= 82:
                            raise RuntimeError('82C temperature guard')
                        if ERRORS.search((out / 'server.log').read_text(errors='replace')):
                            raise RuntimeError('CUDA/runtime error in server log')
                        if server.poll() is not None:
                            raise RuntimeError('Server exited during gate')
            except Exception as error:
                faults.append(str(error))
                if server.poll() is None: os.killpg(server.pid, signal.SIGTERM)

        watcher = threading.Thread(target=watch, daemon=True); watcher.start()
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if faults: raise RuntimeError(faults[-1])
            try:
                with urllib.request.urlopen(URL + '/health', timeout=2) as response:
                    if json.load(response).get('status') == 'ok': break
            except Exception: pass
            time.sleep(1)
        else: raise RuntimeError('Server readiness timeout')
        if sorted(os.sched_getaffinity(server.pid)) != list(range(8)):
            raise RuntimeError('CPU affinity differs from the frozen profile')
        save(out / 'process.json', {'pid': server.pid, 'maps': Path(f'/proc/{server.pid}/maps').read_text(),
                                   'numa_maps': Path(f'/proc/{server.pid}/numa_maps').read_text(),
                                   'status': Path(f'/proc/{server.pid}/status').read_text()})
        result.update(state='running', pid=server.pid); save(out / 'status.json', result)

        def request(label, body, endpoint='/completion'):
            if faults: raise RuntimeError(faults[-1])
            reply = post(endpoint, body)
            if faults: raise RuntimeError(faults[-1])
            save(out / (label + '.json'), {'request': body, 'response': reply})
            if endpoint == '/completion' and (len(reply.get('tokens', [])) != body['n_predict'] or reply.get('truncated')):
                raise RuntimeError(label + ': count/truncation mismatch')
            return reply

        yield out, request
        if faults: raise RuntimeError(faults[-1])
        result.update(state='gate_pass')
    except Exception as error:
        result.update(state='failed', error=str(error))
        raise
    finally:
        done.set()
        if server is not None and server.poll() is None:
            os.killpg(server.pid, signal.SIGINT)
            try: server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL); server.wait(timeout=10)
        if watcher: watcher.join(timeout=15)
        if (out / 'server.log').exists() and ERRORS.search((out / 'server.log').read_text(errors='replace')):
            result.update(state='failed', error='CUDA/runtime error in final server log')
        if result['state'] == 'gate_pass' and server is not None and server.returncode not in [0, -signal.SIGINT]:
            result.update(state='failed', error='Unexpected server shutdown status: ' + str(server.returncode))
        result.update(finished=time.time(), server_exit=server.returncode if server else None)
        save(out / 'status.json', result)
        if result['state'] == 'failed':
            raise RuntimeError(result['error'])


def correctness():
    hashes = {}
    fixture = json.loads(Path('/home/sean/work/l0xre-prefill-20261002/fixtures.json').read_text())['pp77824']
    for name, serial in [('spec', False), ('serial', True), ('restart', False)]:
        with boot('correctness-' + name, serial=serial) as (out, request):
            values = []
            for index, prompt in enumerate(PROMPTS):
                first = request(f'greedy{index}', payload(prompt))
                repeat = request(f'greedy{index}-repeat', payload(prompt))
                if first['tokens'] != repeat['tokens']: raise RuntimeError('Greedy immediate repeat differs')
                values.append(digest(first['tokens']))
            pre = request('cold10k', payload(fixture[:10240], 1))
            first = request('seeded64', payload('Explain why the sky appears blue in clear, detailed prose.', 64, .6))
            repeat = request('seeded64-repeat', payload('Explain why the sky appears blue in clear, detailed prose.', 64, .6))
            if first['tokens'] != repeat['tokens']: raise RuntimeError('Seeded immediate repeat differs')
            hashes[name] = {'greedy': values, 'prefill_token': pre['tokens'], 'seeded': digest(first['tokens'])}
            if name != 'serial' and 'L0XRE_REJECTION_VERIFY_ACTIVE' not in (out / 'server.log').read_text():
                raise RuntimeError('Rejection verification route not observed')
    if any(hashes[key]['greedy'] != hashes['spec']['greedy'] or hashes[key]['prefill_token'] != hashes['spec']['prefill_token'] for key in hashes):
        raise RuntimeError('Serial/spec/restart greedy or cold-prefill tokens differ')
    if hashes['restart']['seeded'] != hashes['spec']['seeded']:
        raise RuntimeError('Seeded speculative tokens differ after restart')
    save(ROOT / 'CORRECTNESS.json', {'status': 'pass', 'results': hashes,
         'scope': 'Greedy128 code/prose and10K prefill serial/spec/restart equivalence; seeded64 repeat per mode and speculative cross-restart. Sampled serial/spec equality is reported separately.',
         'sampled_serial_spec_equal': hashes['serial']['seeded'] == hashes['spec']['seeded']})


def vision():
    image = ROOT / 'vision-smoke.png'
    if hashlib.sha256(image.read_bytes()).hexdigest() != '145534c3b1df9de079da2c59eefeea210a6013578f0355ce31a9af0ca7331c5f':
        raise RuntimeError('Vision fixture identity differs')
    with boot('cpu-vision', vision=True) as (out, request):
        text = 'Read the printed text exactly and identify the three colored shapes. Return only JSON with keys "text" and "shapes". Each shape must have keys "color" and "shape". Do not include markdown.'
        body = {'model': 'L0xRE-27b-Low', 'messages': [{'role': 'user', 'content': [
                {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(image.read_bytes()).decode()}},
                {'type': 'text', 'text': text}]}], 'temperature': 0, 'seed': 42, 'max_tokens': 1024, 'reasoning_effort': 'medium'}
        reply = request('ocr-shapes', body, '/v1/chat/completions')
        choice = reply['choices'][0]
        if choice['finish_reason'] == 'length': raise RuntimeError('Vision answer truncated')
        raw = re.sub(r'^```(?:json)?\s*|\s*```$', '', choice['message']['content'].strip())
        answer = json.loads(raw)
        shapes = {(v['color'].lower().strip(), v['shape'].lower().strip()) for v in answer['shapes']}
        if answer['text'].strip().upper() != 'VISION 742' or len(shapes) != 3 or ('blue', 'circle') not in shapes or ('green', 'triangle') not in shapes or not shapes.intersection({('red','square'),('red','rectangle'),('red','box')}):
            raise RuntimeError('OCR/shape answer differs')
        if 'CLIP using CPU backend' not in (out / 'server.log').read_text():
            raise RuntimeError('CPU vision backend not observed')
        save(ROOT / 'CPU-VISION.json', {'status': 'pass', 'answer': answer, 'scope': 'One fixed native CPU OCR/color/shape image, not broad vision accuracy.'})


def soak(seconds):
    if seconds < 1800: raise RuntimeError('A soak must run at least30minutes')
    fixture = json.loads(Path('/home/sean/work/l0xre-prefill-20261002/fixtures.json').read_text())['pp77824']
    baselines = {}
    requests = comparisons = cycles = 0
    started = time.monotonic()
    with boot('stability-soak') as (out, request):
        started = time.monotonic()
        while time.monotonic() - started < seconds:
            for depth in [2048,16384,32768,65536,77824]:
                answer = request(f'cycle{cycles}-prefill{depth}', payload(fixture[:depth], 1))
                if answer['timings']['prompt_n'] != depth or answer['timings'].get('cache_n',0):
                    raise RuntimeError('Cold prefill depth/cache differs')
                key = 'prefill' + str(depth)
                if key in baselines:
                    comparisons += 1
                    if answer['tokens'] != baselines[key]: raise RuntimeError(key + ': repeated token differs')
                baselines[key] = answer['tokens']; requests += 1
                for index,prompt in enumerate(PROMPTS):
                    for temp,count in [(0,128),(.7,64)]:
                        key = f'short{index}-{temp}'
                        answer = request(f'cycle{cycles}-{depth}-{key}', payload(prompt,count,temp))
                        if key in baselines:
                            comparisons += 1
                            if answer['tokens'] != baselines[key]: raise RuntimeError(key + ': long-to-short repeat differs')
                        baselines[key] = answer['tokens']; requests += 1
            cycles += 1
        elapsed = time.monotonic() - started
        save(ROOT / 'SOAK.json', {'status':'pass','elapsed_seconds':elapsed,'cycles':cycles,'requests':requests,
                               'repeat_comparisons':comparisons,'depths':[2048,16384,32768,65536,77824]})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['correctness','vision','soak'])
    parser.add_argument('--seconds', type=int, default=1800)
    options = parser.parse_args()
    occupied = subprocess.check_output(['nvidia-smi','--id='+UUID,'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if occupied: raise RuntimeError('RTX4090 is occupied: ' + occupied)
    ROOT.joinpath('probes').mkdir(exist_ok=True)
    {'correctness':correctness,'vision':vision,'soak':lambda:soak(options.seconds)}[options.mode]()


if __name__ == '__main__': main()
