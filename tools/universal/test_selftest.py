"""Check the receipt protocol with deterministic HTTP fixtures, never a fake certificate."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import selftest


def choices(text='READY'):
    return [dict(index=0, message=dict(role='assistant', content=text), finish_reason='stop',
        logprobs=dict(content=[dict(token=text, logprob=-.125,
            top_logprobs=[dict(token=str(i), logprob=-float(i + 1)) for i in range(10)])]))]


def reference():
    return dict(greedy=dict(request=dict(model='fixture', messages=[dict(role='user', content='Say READY.')],
        max_tokens=32, temperature=0, top_p=1, seed=17, cache_prompt=False, logprobs=True,
        top_logprobs=10, chat_template_kwargs=dict(enable_thinking=False)), choices=choices()),
        vision=dict(request=dict(fixture='vision'), printed_text='VISION 742',
                    shapes=['square', 'circle', 'triangle'], colors={'red': [220, 20, 35]}))


def response(endpoint, body):
    if endpoint == '/completion':
        return dict(tokens=[1, 2], timings=dict(predicted_per_second=123.4))
    if body.get('fixture') == 'vision':
        return dict(choices=choices('VISION 742: red square, blue circle and green triangle.'))
    live = body.get('cache_prompt') is True
    return dict(choices=choices(), timings=dict(cache_source='live' if live else 'none', cache_n=20 if live else 0))


class Protocol(unittest.TestCase):
    def test_complete_protocol_and_raw_receipts(self):
        with tempfile.TemporaryDirectory() as name:
            out = Path(name)
            result = selftest.exercise(response, reference(), out)
            self.assertTrue(result['passed'])
            self.assertTrue(all(result['checks'].values()))
            self.assertEqual(len(json.loads((out / 'requests.json').read_text())), 7)
            self.assertEqual(set(result['performance']), {'narrative', 'code'})

    def test_frozen_greedy_request_settings_are_not_rewritten(self):
        golden = reference()
        golden['greedy']['request']['seed'] = 42
        del golden['greedy']['request']['top_p']
        calls = []
        def observed(endpoint, body):
            calls.append(copy.deepcopy(body))
            return response(endpoint, body)
        with tempfile.TemporaryDirectory() as name:
            selftest.exercise(observed, golden, Path(name))
        self.assertEqual(calls[0], golden['greedy']['request'])

    def test_any_logprob_difference_is_a_failure(self):
        def changed(endpoint, body):
            result = response(endpoint, body)
            result['choices'][0]['logprobs']['content'][0]['top_logprobs'][4]['logprob'] += 1e-12
            return result
        with tempfile.TemporaryDirectory() as name:
            with self.assertRaisesRegex(ValueError, 'shipped reference'):
                selftest.exercise(changed, reference(), Path(name))
            self.assertFalse(json.loads((Path(name) / 'checks.json').read_text())['greedy_reference_exact'])

    def test_empty_top10_never_passes_even_when_outputs_are_equal(self):
        def empty(endpoint, body):
            result = response(endpoint, body)
            if len(body.get('messages', [])) > 1:
                result['choices'][0]['logprobs']['content'] = []
            return result
        with tempfile.TemporaryDirectory() as name:
            with self.assertRaisesRegex(ValueError, 'cache tokens'):
                selftest.exercise(empty, reference(), Path(name))

    def test_cache_response_requires_positive_live_reuse(self):
        def not_live(endpoint, body):
            result = response(endpoint, body)
            result['timings'] = dict(cache_n=0, cache_source='none')
            return result
        with tempfile.TemporaryDirectory() as name:
            with self.assertRaisesRegex(ValueError, 'reuse checks failed'):
                selftest.exercise(not_live, reference(), Path(name))

    def test_incorrect_vision_ground_truth_fails(self):
        def wrong(endpoint, body):
            result = response(endpoint, body)
            if body.get('fixture') == 'vision':
                result['choices'][0]['message']['content'] = 'VISION 000 red square blue circle green triangle'
            return result
        with tempfile.TemporaryDirectory() as name:
            with self.assertRaisesRegex(ValueError, 'Vision ground-truth'):
                selftest.exercise(wrong, reference(), Path(name))

    def test_zero_or_boolean_throughput_is_not_a_valid_benchmark(self):
        for rate in [0, True]:
            def wrong(endpoint, body):
                result = response(endpoint, body)
                if endpoint == '/completion':
                    result['timings']['predicted_per_second'] = rate
                return result
            with self.subTest(rate=rate), tempfile.TemporaryDirectory() as name:
                with self.assertRaisesRegex(ValueError, 'Short benchmark'):
                    selftest.exercise(wrong, reference(), Path(name))

    def test_signature_excludes_reference_metadata_but_binds_numerical_settings(self):
        config = dict(argv=['server', '--shared-device-memory-mib', '2048'], env={'I8': '1024'},
                      model_sha256={'target': 'model'})
        boot = dict(architecture='sm86', arena_mib=2048, steering={'enabled': False})
        manifest = dict(runtime_sha256={'architectures/sm86/bin/engine': 'binary'})
        before = selftest.configuration_signature(config, manifest, boot)
        manifest['runtime_sha256']['architectures/sm86/selftest/reference.json'] = 'reference'
        self.assertEqual(selftest.configuration_signature(config, manifest, boot), before)
        manifest['runtime_sha256']['architectures/sm86/bin/engine'] = 'different-binary'
        self.assertNotEqual(selftest.configuration_signature(config, manifest, boot), before)
        manifest['runtime_sha256']['architectures/sm86/bin/engine'] = 'binary'
        manifest['runtime_sha256']['shared/cuda/libcublas.so.13'] = 'different-cublas'
        self.assertNotEqual(selftest.configuration_signature(config, manifest, boot), before)

    def test_signature_binds_selected_ubatch_and_arena(self):
        config = dict(argv=['server', '--shared-device-memory-mib', '2240', '-ub', '1024'],
                      env={}, model_sha256={'target':'model'})
        manifest = dict(runtime_sha256={'architectures/sm120/bin/server':'binary'})
        boot = dict(architecture='sm120', arena_mib=1280, steering={'enabled':False},
                    vram_preflight={'ubatch':512})
        actual = selftest.configuration_signature(config, manifest, boot)
        selected = copy.deepcopy(config)
        selected['argv'] = ['server', '--shared-device-memory-mib', '1280', '-ub', '512']
        self.assertEqual(actual, selftest.configuration_signature(selected, manifest, boot))
        boot['vram_preflight']['ubatch'] = 1024
        self.assertNotEqual(actual, selftest.configuration_signature(config, manifest, boot))


FAKE_LAUNCHER = r'''
import argparse,json,os,signal,sys
from pathlib import Path
from http.server import BaseHTTPRequestHandler,HTTPServer
assert os.environ.get('L0XRE_NONINTERACTIVE') == '1', 'selftest child must be noninteractive'
p=argparse.ArgumentParser();p.add_argument('command');p.add_argument('--dry-run',action='store_true');p.add_argument('--port',type=int);a,rest=p.parse_known_args()
root=Path(__file__).resolve().parents[2]
if a.dry_run:
 print((root/'boot.json').read_text());raise SystemExit(0)
if os.name=='nt':
 def interrupted(sig,frame):raise KeyboardInterrupt
 signal.signal(signal.SIGBREAK,interrupted)
print(json.dumps(dict(owned_server_pid=os.getpid())),flush=True)
def choices(text='READY'):
 return [dict(index=0,message=dict(role='assistant',content=text),finish_reason='stop',logprobs=dict(content=[dict(token=text,logprob=-.125,top_logprobs=[dict(token=str(i),logprob=-float(i+1)) for i in range(10)])]))]
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*args):pass
 def reply(self,data):
  self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(json.dumps(data).encode())
 def do_GET(self):self.reply(dict(status='ok'))
 def do_POST(self):
  b=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
  if self.path=='/completion':self.reply(dict(tokens=[1,2],timings=dict(predicted_per_second=123.4)));return
  text='VISION 742 red square blue circle green triangle' if b.get('fixture')=='vision' else 'READY'
  live=b.get('cache_prompt') is True
  self.reply(dict(choices=choices(text),timings=dict(cache_source='live' if live else 'none',cache_n=20 if live else 0)))
server=HTTPServer(('127.0.0.1',a.port),Handler)
try:server.serve_forever()
except KeyboardInterrupt:pass
finally:server.server_close()
'''


class OwnedIntegration(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        tools = self.root / 'tools/universal'
        tools.mkdir(parents=True)
        self.launcher = tools / 'launcher.py'
        self.launcher.write_text(FAKE_LAUNCHER)
        arch = self.root / 'architectures/sm120'
        (arch / 'profiles').mkdir(parents=True)
        (arch / 'selftest').mkdir()
        self.boot = dict(architecture='sm120', memory_profile='12gb', arena_mib=2048,
            context_tokens=131072, gpu='GPU-fixture', steering={'enabled': False},
            vram_preflight=dict(chosen_arena_mib=2048, arena_mib=2048, free_mib=16000,
                rung=0, ubatch=1024, required_mib=11064, consent='none', speed_cost_pct=0.), cpu_affinity=[0])
        (self.root / 'boot.json').write_text(json.dumps(self.boot))
        config = dict(argv=['server', '--shared-device-memory-mib', '2048', '-ub', '1024'], env={}, model_sha256={'target': 'model'})
        profile = arch / 'profiles/12gb.json'
        profile.write_text(json.dumps(config))
        self.manifest = dict(runtime_sha256={'architectures/sm120/bin/engine': 'fixture-binary-hash'},
            memory_profiles={'12gb': dict(path='architectures/sm120/profiles/12gb.json', sha256=selftest.sha(profile))})
        variant = reference()
        variant['configuration_sha256'] = selftest.configuration_signature(config, self.manifest, self.boot)
        self.reference = arch / 'selftest/reference.json'
        self.reference.write_text(json.dumps(dict(architecture='sm120', variants=[variant])))
        self.manifest['selftest'] = dict(reference=dict(path='architectures/sm120/selftest/reference.json', sha256=selftest.sha(self.reference)))
        self.manifest_path = arch / 'V0.1-MANIFEST.json'
        self.manifest_path.write_text(json.dumps(self.manifest))
        self.smi = self.root / ('nvidia-smi.cmd' if os.name == 'nt' else 'nvidia-smi')
        if os.name == 'nt':
            self.smi.write_text('@echo off\necho %* | findstr /c:"name,uuid" >nul\nif not errorlevel 1 (echo Fixture GPU, GPU-fixture, 12.0, fixture-driver, 16384, 16000) else (echo 384, 16000)\n')
        else:
            self.smi.write_text('#!' + sys.executable + '\nimport sys\nprint("Fixture GPU, GPU-fixture, 12.0, fixture-driver, 16384, 16000" if any("name,uuid" in v for v in sys.argv) else "384, 16000")\n')
            self.smi.chmod(0o700)

    def tearDown(self):
        self.temp.cleanup()

    def test_owned_startup_protocol_receipt_and_cleanup(self):
        out = self.root / 'receipt'
        with patch('selftest.shutil.which', return_value=str(self.smi)):
            code = selftest.main(['--output', str(out)], launcher=self.launcher)
        receipt = json.loads((out / 'receipt.json').read_text())
        self.assertEqual(code, 0, receipt)
        self.assertTrue(receipt['passed'])
        self.assertTrue(receipt['cleanup']['launcher_exited'])
        if os.name == 'nt':
            self.assertTrue(receipt['cleanup']['native_process_exited'])
        self.assertEqual(receipt['startup']['vram_preflight'], self.boot['vram_preflight'])
        self.assertEqual(receipt['gpu']['free_before_load_mib'], 16000)
        self.assertEqual(receipt['memory']['peak_adapter_used_mib'], 384)
        self.assertEqual(receipt['reference_sha256'], selftest.sha(self.reference))
        self.assertTrue(all(receipt['checks'].values()))
        self.assertIn('Hardware validation report', (out / 'SUMMARY.txt').read_text())

    def test_tampered_reference_refuses_before_server_start(self):
        self.reference.write_text('{}')
        out = self.root / 'receipt'
        code = selftest.main(['--output', str(out)], launcher=self.launcher)
        receipt = json.loads((out / 'receipt.json').read_text())
        self.assertEqual(code, 1)
        self.assertFalse(receipt['passed'])
        self.assertFalse(receipt['cleanup']['started'])
        self.assertIn('reference integrity', receipt['error'])

    def test_ambiguous_reference_cannot_silently_choose_a_golden(self):
        data = json.loads(self.reference.read_text())
        data['variants'].append(copy.deepcopy(data['variants'][0]))
        self.reference.write_text(json.dumps(data))
        self.manifest['selftest']['reference']['sha256'] = selftest.sha(self.reference)
        self.manifest_path.write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, 'No unique shipped reference'):
            selftest.load_reference(self.root, self.boot)


if __name__ == '__main__':
    unittest.main()
