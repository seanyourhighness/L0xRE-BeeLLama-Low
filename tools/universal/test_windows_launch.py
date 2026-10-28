"""Windows wrapper integration with real CPU affinity and an owned stub worker."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

from windows_placement import physical_cores


class TopologyDecoder(unittest.TestCase):
    def test_retains_physical_core_and_group_boundaries(self):
        data = bytearray()
        for group, mask in [(0, 3), (0, 12), (1, 3)]:
            data.extend(struct.pack('<II', 0, 48) + bytes(22) + struct.pack('<H', 1)
                        + struct.pack('<QH', mask, group) + bytes(6))
        self.assertEqual(physical_cores(data), [[(3, 0)], [(12, 0)], [(3, 1)]])

    def test_rejects_truncated_topology(self):
        with self.assertRaises(ValueError):
            physical_cores(struct.pack('<II', 0, 48))


@unittest.skipUnless(os.name == 'nt', 'native Windows integration')
class WindowsLauncher(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='l0xre windows ')
        self.root = Path(self.tmp.name)
        self.arch = self.root / 'architectures/sm120'
        tools = self.root / 'tools/universal'
        tools.mkdir(parents=True)
        (self.arch / 'profiles').mkdir(parents=True)
        for name in ['v01-launch.py', 'v01-windows-launch.py', 'windows_placement.py', 'control_vectors.py', 'vram_preflight.py']:
            shutil.copy2(Path(__file__).with_name(name), tools / name)
        self.launch = tools / 'v01-windows-launch.py'
        self.model = self.root / 'model with spaces.gguf'
        self.model.write_bytes(b'fixture only')
        self.vector = self.root / 'vector, with spaces.gguf'
        self.vector.write_bytes(b'GGUFfixture only')
        self.probe = self.root / 'worker.py'
        self.probe.write_text('''import ctypes,json,os,sys
k=ctypes.WinDLL('kernel32',use_last_error=True);k.GetCurrentProcess.restype=ctypes.c_void_p
k.GetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t)]
a,b=ctypes.c_size_t(),ctypes.c_size_t();assert k.GetProcessAffinityMask(k.GetCurrentProcess(),ctypes.byref(a),ctypes.byref(b))
print(json.dumps(dict(worker=True,mask=a.value,argv=sys.argv[1:],mode=os.environ['L0XRE_CVEC_MODE'],poison=os.environ.get('ESCHA_UNOWNED_POISON'))))
''')
        digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        self.catalog = {}
        for name, minimum, context in [('12gb', 12000, 131072), ('16gb', 16000, 262144)]:
            path = self.arch / 'profiles' / (name + '.json')
            path.write_text(json.dumps(dict(minimum_gpu_memory_mib=minimum, cpu_threads=2,
                context_tokens=context, arena_mib=2048,
                argv=[sys.executable, str(self.probe), '--shared-device-memory-mib', '2048', '-ub', '1024'],
                vram_preflight=dict(desktop_reserve_mib=640, safety_margin_mib=64,
                    validated_arenas=[dict(arena_mib=2048, required_free_mib=11000, ubatch=1024,
                                          headline_speeds={"depth_decode":100},
                                          evidence='synthetic fixture; no hardware certification')]),
                env={}, model_sha256={name: digest(self.model) for name in ['target', 'draft', 'vision']})))
            self.catalog[name] = dict(path=path.relative_to(self.root).as_posix(),
                                      sha256=digest(path), hardware_qualified=True)
        self.manifest = dict(platform='windows', hardware_qualified=True, memory_profiles=self.catalog,
            runtime_sha256={'worker.py': digest(self.probe)}, loader_links={},
            features=dict(control_vector_projection=True))
        self.write_manifest()
        self.bin = self.root / 'fake-bin'
        self.bin.mkdir()
        (self.bin / 'nvidia-smi.cmd').write_text('@echo off\necho %* | findstr /c:"--query-gpu=memory.free" >nul\nif not errorlevel 1 (echo %FIXTURE_FREE%) else (echo GPU-fixture, %FIXTURE_CAP%, %FIXTURE_MEMORY%, 00000000:ff:00.0)\n')

    def tearDown(self):
        self.tmp.cleanup()

    def write_manifest(self):
        (self.arch / 'V0.1-MANIFEST.json').write_text(json.dumps(self.manifest))

    def run_launch(self, *extra, memory=32607, free=None, dry=True, capability='12.0'):
        env = dict(os.environ, PATH=str(self.bin) + os.pathsep + os.environ['PATH'],
                   FIXTURE_MEMORY=str(memory), FIXTURE_FREE=str(memory if free is None else free),
                   FIXTURE_CAP=capability,
                   ESCHA_UNOWNED_POISON='must not leak')
        return subprocess.run([sys.executable, str(self.launch), 'serve',
            *(['--dry-run'] if dry else []), '-m', str(self.model), '-md', str(self.model),
            '--mmproj', str(self.model), *extra], env=env, text=True, capture_output=True)

    def test_memory_defaults_and_explicit_selection(self):
        for memory, args, expected in [(12288, (), '12gb'), (32607, (), '16gb'),
            (32607, ('--memory-profile', '12gb'), '12gb')]:
            p = self.run_launch(*args, memory=memory)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertEqual(json.loads(p.stdout)['memory_profile'], expected)

    def test_rejects_16gb_on_12gb_card(self):
        p = self.run_launch('--memory-profile', '16gb', memory=12288)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn('requires at least 16000 MiB', p.stderr)

    def test_insufficient_free_memory_never_starts_child(self):
        p = self.run_launch(memory=12288, free=10000, dry=False)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn('free 10000 MiB; required 11064 MiB', p.stderr)
        self.assertEqual(p.stdout, '')

    def test_override_cannot_invent_arena(self):
        p = self.run_launch('--arena-mib', '2000')
        self.assertNotEqual(p.returncode, 0)
        self.assertIn('has not been validated', p.stderr)

    def test_hardware_matrix_detection_and_build_only_status(self):
        original16 = dict(self.catalog['16gb'])
        self.manifest.update(hardware_qualified=False, build_verified=True,
                             qualification_status='hardware_validation_in_progress')
        for architecture, rows in [
            ('sm86', [('3060', '8.6', 12288, '12gb'),
                      ('3080 12GB', '8.6', 12288, '12gb'),
                      ('3090', '8.6', 24576, '12gb')]),
            ('sm89', [('4060 Ti 16GB', '8.9', 16384, '16gb'),
                      ('4070', '8.9', 12288, '12gb'),
                      ('4080', '8.9', 16384, '16gb'),
                      ('4090', '8.9', 24576, '16gb')])]:
            target = self.root / 'architectures' / architecture
            self.arch.rename(target)
            self.arch = target
            if architecture == 'sm86':
                del self.catalog['16gb']  # No physical SM86 16GB certification.
            else:
                self.catalog['16gb'] = dict(original16)
            for name, entry in self.catalog.items():
                entry['path'] = f'architectures/{architecture}/profiles/{name}.json'
            self.write_manifest()
            for card, cap, memory, expected in rows:
                with self.subTest(card=card):
                    p = self.run_launch(memory=memory, capability=cap)
                    self.assertEqual(p.returncode, 0, p.stderr)
                    report = json.loads(p.stdout)
                    self.assertEqual(report['architecture'], architecture)
                    self.assertEqual(report['memory_profile'], expected)
                    self.assertFalse(report['hardware_qualified'])
                    self.assertTrue(report['hardware_validation_in_progress'])
            if architecture == 'sm86':
                p = self.run_launch('--memory-profile', '16gb', memory=24576, capability='8.6')
                self.assertNotEqual(p.returncode, 0)
                self.assertIn('16gb is not installed for sm86', p.stderr)
        for card, cap, memory in [('3080 10GB', '8.6', 10240), ('4060 Ti 8GB', '8.9', 8192)]:
            with self.subTest(card=card):
                p = self.run_launch(memory=memory, capability=cap)
                self.assertNotEqual(p.returncode, 0)
                self.assertIn('Unsupported GPU:', p.stderr)

    def test_build_only_label_never_bypasses_sm120_hardware_gate(self):
        self.manifest.update(hardware_qualified=False, build_verified=True,
                             qualification_status='hardware_validation_in_progress')
        self.write_manifest()
        self.assertIn('hardware qualification is pending', self.run_launch().stderr)

    def test_rejects_unqualified_runtime_without_probe(self):
        self.manifest['hardware_qualified'] = False
        self.write_manifest()
        self.assertNotEqual(self.run_launch().returncode, 0)
        self.assertEqual(self.run_launch('--qualification-probe').returncode, 0)

    def test_rejects_wrong_platform_and_modified_payload(self):
        self.manifest['platform'] = 'linux'
        self.write_manifest()
        self.assertIn('not a Windows runtime payload', self.run_launch().stderr)
        self.manifest['platform'] = 'windows'
        self.write_manifest()
        self.probe.write_text('modified')
        self.assertIn('payload integrity check failed', self.run_launch().stderr)

    def test_vector_settings_and_disk_cache_identity(self):
        off = self.run_launch('--state-cache-dir', str(self.root / 'state'))
        on = self.run_launch('--state-cache-dir', str(self.root / 'state'),
            '--control-vector', str(self.vector), '--control-vector-scale', '0.5')
        self.assertEqual(off.returncode, 0, off.stderr)
        self.assertEqual(on.returncode, 0, on.stderr)
        a, b = json.loads(off.stdout), json.loads(on.stdout)
        self.assertFalse(a['steering']['enabled'])
        self.assertEqual(b['env']['L0XRE_CVEC_MODE'], 'project')
        self.assertNotEqual(a['state_cache_directory'], b['state_cache_directory'])
        self.assertFalse(Path(b['state_cache_directory']).exists())

    def test_real_child_inherits_affinity_and_clean_environment(self):
        p = self.run_launch('--control-vector', str(self.vector), dry=False)
        self.assertEqual(p.returncode, 0, p.stderr)
        lines = p.stdout.splitlines()
        self.assertTrue(lines[0].startswith('VRAM preflight: full-speed arena 2048 MiB;'))
        parent, child = [json.loads(line) for line in lines[1:]]
        self.assertTrue(child['worker'])
        self.assertEqual(child['mode'], 'project')
        self.assertIsNone(child['poison'])
        self.assertEqual(child['mask'], sum(1 << cpu for cpu in parent['cpu_affinity']))
        self.assertEqual(len(parent['cpu_affinity']), 2)
        self.assertIn('--control-vector-scaled', child['argv'])

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
