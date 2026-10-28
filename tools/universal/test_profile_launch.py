"""Exercise the public selector and integrity checks without a GPU or real weights."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@unittest.skipIf(os.name == 'nt', 'Linux launcher integration')
class ProfileLaunch(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.arch = self.root / 'architectures/sm120'
        (self.arch / 'profiles').mkdir(parents=True)
        tools = self.root / 'tools/universal'
        tools.mkdir(parents=True)
        self.launch = tools / 'v01-launch.py'
        shutil.copy2(Path(__file__).with_name('v01-launch.py'), self.launch)
        shutil.copy2(Path(__file__).with_name('control_vectors.py'), tools / 'control_vectors.py')
        shutil.copy2(Path(__file__).with_name('vram_preflight.py'), tools / 'vram_preflight.py')
        self.model = self.root / 'model.gguf'
        self.model.write_bytes(b'fixture only')
        self.catalog = {}
        for name, minimum, context in [('12gb', 12000, 131072), ('16gb', 16000, 262144)]:
            path = self.arch / 'profiles' / (name + '.json')
            path.write_text(json.dumps(dict(minimum_gpu_memory_mib=minimum,
                cpu_threads=1, context_tokens=context, arena_mib=2048,
                argv=['@ARCH_ROOT@/bin/llama-server', '--shared-device-memory-mib', '2048', '-ub', '1024'],
                vram_preflight=dict(desktop_reserve_mib=640, safety_margin_mib=64,
                    validated_arenas=[dict(arena_mib=2048, required_free_mib=11000, ubatch=1024,
                                          headline_speeds={"depth_decode":100},
                                          evidence='synthetic fixture; no hardware certification')]),
                env={'LD_PRELOAD': '@ARCH_ROOT@/bin/libllama.so'},
                model_sha256={name: digest(self.model) for name in ['target', 'draft', 'vision']})))
            self.catalog[name] = dict(path=str(path.relative_to(self.root)),
                                      sha256=digest(path), hardware_qualified=True)
        self.manifest = dict(hardware_qualified=True, memory_profiles=self.catalog,
                             runtime_sha256={}, loader_links={})
        self.write_manifest()
        self.bin = self.root / 'fake-bin'
        self.bin.mkdir()
        smi = self.bin / 'nvidia-smi'
        smi.write_text('#!' + sys.executable + '\nimport os,sys\nprint(os.environ.get("FIXTURE_FREE", os.environ.get("FIXTURE_MEMORY", "32607"))) if "--query-gpu=memory.free" in sys.argv else print("GPU-fixture, 12.0, " + os.environ.get("FIXTURE_MEMORY", "32607") + ", 00000000:ff:00.0")\n')
        smi.chmod(0o700)

    def tearDown(self):
        self.tmp.cleanup()

    def write_manifest(self):
        (self.arch / 'V0.1-MANIFEST.json').write_text(json.dumps(self.manifest))

    def run_launch(self, *extra, memory=32607, free=None):
        env = dict(os.environ, PATH=str(self.bin) + ':' + os.environ['PATH'],
                   FIXTURE_MEMORY=str(memory), FIXTURE_FREE=str(memory if free is None else free))
        return subprocess.run([sys.executable, str(self.launch), 'serve', '--dry-run',
            '-m', str(self.model), '-md', str(self.model), '--mmproj', str(self.model),
            *extra], env=env, text=True, capture_output=True)

    def test_selects_12gb_on_actual_12gb_capacity(self):
        p = self.run_launch(memory=12288)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(json.loads(p.stdout)['context_tokens'], 131072)

    def test_sub_12gb_gpu_reports_unsupported(self):
        p = self.run_launch(memory=8192)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn('Unsupported GPU: 8192 MiB VRAM', p.stderr)

    def test_insufficient_free_memory_refuses_before_startup(self):
        p = self.run_launch(memory=12288, free=10000)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn('free 10000 MiB; required 11064 MiB', p.stderr)
        self.assertIn('Close browsers, overlays, game launchers and GPU wallpapers', p.stderr)
        self.assertEqual(p.stdout, '')

    def test_override_cannot_invent_arena(self):
        p = self.run_launch('--arena-mib', '2000')
        self.assertNotEqual(p.returncode, 0)
        self.assertIn('has not been validated', p.stderr)

    def test_16gb_default_and_explicit_12gb_override(self):
        for args, expected in [((), '16gb'), (('--memory-profile', '12gb'), '12gb')]:
            p = self.run_launch(*args)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertEqual(json.loads(p.stdout)['memory_profile'], expected)

    def test_16gb_rejected_on_12gb_card(self):
        p = self.run_launch('--memory-profile', '16gb', memory=12288)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn('requires at least 16000 MiB', p.stderr)

    def test_modified_profile_cannot_launch(self):
        (self.arch / 'profiles/16gb.json').write_text('{}')
        p = self.run_launch()
        self.assertNotEqual(p.returncode, 0)
        self.assertIn('profile integrity check failed', p.stderr)

    def test_development_profile_requires_probe(self):
        self.catalog['16gb']['hardware_qualified'] = False
        self.write_manifest()
        self.assertNotEqual(self.run_launch().returncode, 0)
        p = self.run_launch('--qualification-probe')
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertFalse(json.loads(p.stdout)['hardware_qualified'])

    def test_profile_cannot_escape_architecture(self):
        self.catalog['16gb']['path'] = '../other.json'
        self.write_manifest()
        p = self.run_launch()
        self.assertNotEqual(p.returncode, 0)
        self.assertIn('inside the selected architecture', p.stderr)

    def test_missing_preset_is_not_silently_substituted(self):
        del self.catalog['16gb']
        self.write_manifest()
        p = self.run_launch()
        self.assertNotEqual(p.returncode, 0)
        self.assertIn('exactly the 12gb and 16gb profiles', p.stderr)

    def test_projection_requires_feature_in_payload(self):
        vector = self.root / 'vector.gguf'
        vector.write_bytes(b'GGUFfixture')
        p = self.run_launch('--control-vector', str(vector))
        self.assertNotEqual(p.returncode, 0)
        self.assertIn('not available in this payload', p.stderr)
        self.manifest['features'] = dict(control_vector_projection=True)
        self.write_manifest()
        p = self.run_launch('--control-vector', str(vector), '--control-vector-scale', '0.5')
        self.assertEqual(p.returncode, 0, p.stderr)
        data = json.loads(p.stdout)
        self.assertEqual(data['env']['L0XRE_CVEC_MODE'], 'project')
        self.assertEqual(data['steering']['scale'], 0.5)
        self.assertIn('--control-vector-scaled', data['argv'])

    def test_composition_preserves_source_and_cannot_claim_certification(self):
        source12 = self.arch / 'profiles/12gb.json'
        source16 = self.arch / 'profiles/16gb.json'
        for path, context in [(source12, 131072), (source16, 262144)]:
            config = json.loads(path.read_text())
            config['argv'] += ['-c', str(context)]
            path.write_text(json.dumps(config))
        before = digest(self.arch / 'V0.1-MANIFEST.json')
        output = self.root / 'composed'
        p = subprocess.run([sys.executable,
            str(Path(__file__).with_name('compose-memory-profiles.py')),
            '--base', str(self.root), '--arch', 'sm120',
            '--profile-12gb', str(source12), '--profile-16gb', str(source16),
            '--output', str(output)], text=True, capture_output=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(digest(self.arch / 'V0.1-MANIFEST.json'), before)
        manifest = json.loads((output / 'architectures/sm120/V0.1-MANIFEST.json').read_text())
        self.assertEqual(set(manifest['memory_profiles']), {'12gb', '16gb'})
        self.assertFalse(manifest['hardware_qualified'])
        self.assertFalse(manifest['promoted'])
        for entry in manifest['memory_profiles'].values():
            self.assertFalse(entry['hardware_qualified'])
            self.assertEqual(digest(output / entry['path']), entry['sha256'])

    def test_slower_rung_requires_consent_and_swaps_arena_and_ubatch(self):
        path = self.arch / 'profiles/12gb.json'
        config = json.loads(path.read_text())
        config['vram_preflight']['validated_arenas'].append(dict(arena_mib=1280,
            ubatch=512, required_free_mib=10000, headline_speeds={'depth_decode':67},
            evidence='synthetic fixture; no hardware certification'))
        path.write_text(json.dumps(config))
        self.catalog['12gb']['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.write_manifest()
        denied = self.run_launch('--memory-profile', '12gb', memory=12288, free=10500)
        self.assertNotEqual(denied.returncode, 0)
        self.assertEqual(denied.stdout, '')
        self.assertIn('Non-interactive launch refused', denied.stderr)
        for args, source in [(('--accept-slower',), 'flag'), (('--arena-mib','1280'), 'explicit-arena')]:
            accepted = self.run_launch('--memory-profile', '12gb', *args, memory=12288, free=10500)
            self.assertEqual(accepted.returncode, 0, accepted.stderr)
            report = json.loads(accepted.stdout)
            self.assertEqual(report['arena_mib'], 1280)
            self.assertEqual(report['argv'][report['argv'].index('-ub')+1], '512')
            decision = report['vram_preflight']
            self.assertEqual(decision['consent'], source)
            self.assertEqual(decision['rung'], 1)
            self.assertAlmostEqual(decision['speed_cost_pct'], 33)


if __name__ == '__main__':
    unittest.main()
