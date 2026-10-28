import argparse
import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest

from control_vectors import add_arguments, startup_settings, namespace_disk_cache


class ControlVectors(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.vector = self.root / 'vector, with spaces.gguf'
        self.vector.write_bytes(b'GGUFfixture-vector-bytes')
        self.parser = argparse.ArgumentParser()
        add_arguments(self.parser)

    def tearDown(self):
        self.tmp.cleanup()

    def settings(self, *args, profile='profile-sha'):
        return startup_settings(self.parser.parse_args(list(args)), profile)

    def test_no_vector_preserves_default(self):
        args, env, identity = self.settings()
        self.assertEqual(args, [])
        self.assertEqual(env['L0XRE_CVEC_MODE'], 'add')
        self.assertFalse(identity['enabled'])

    def test_project_flags_with_comma_filename(self):
        args, env, identity = self.settings('--control-vector', str(self.vector))
        self.assertEqual(env['L0XRE_CVEC_MODE'], 'project')
        self.assertEqual(next(csv.reader([args[1]])), [str(self.vector) + ':1.0'])
        self.assertEqual(args[2:], ['--control-vector-layer-range', '1', '63'])
        self.assertEqual(identity['mode'], 'project')

    def test_all_effective_settings_change_cache_identity(self):
        base = ['--control-vector', str(self.vector)]
        key = self.settings(*base)[2]['cache_identity']
        for extra in [['--control-vector-mode', 'add'], ['--control-vector-scale', '0.5'],
                      ['--control-vector-layer-range', '2', '63']]:
            self.assertNotEqual(key, self.settings(*base, *extra)[2]['cache_identity'])
        self.assertNotEqual(key, self.settings(*base, profile='other-model-profile')[2]['cache_identity'])
        options = self.parser.parse_args(base)
        self.assertNotEqual(startup_settings(options, 'profile-sha', 'runtime-a')[2]['cache_identity'],
                            startup_settings(options, 'profile-sha', 'runtime-b')[2]['cache_identity'])
        self.vector.write_bytes(b'GGUFchanged-vector')
        self.assertNotEqual(key, self.settings(*base)[2]['cache_identity'])

    def test_invalid_inputs_rejected(self):
        for options in [['--control-vector-mode', 'project'],
                        ['--control-vector', str(self.vector), '--control-vector-scale', 'nan'],
                        ['--control-vector', str(self.vector), '--control-vector-scale', '-1'],
                        ['--control-vector', str(self.vector), '--control-vector-layer-range', '1', '64'],
                        ['--control-vector', str(self.root / 'missing.gguf')]]:
            with self.assertRaises(ValueError):
                self.settings(*options)
        self.vector.write_bytes(b'not a GGUF')
        with self.assertRaises(ValueError):
            self.settings('--control-vector', str(self.vector))

    def test_disk_restore_namespaces_are_separate_and_dry_run_does_not_write(self):
        off = self.settings()[2]
        on = self.settings('--control-vector', str(self.vector))[2]
        _, first = namespace_disk_cache([], self.root / 'cache', off, True)
        _, second = namespace_disk_cache([], self.root / 'cache', on, True)
        self.assertNotEqual(first, second)
        self.assertFalse((self.root / 'cache').exists())
        command, directory = namespace_disk_cache([], self.root / 'cache', on)
        self.assertTrue(Path(directory).is_dir())
        self.assertEqual(command, ['--slot-save-path', directory])

    def test_profile_slot_directory_also_namespaced(self):
        identity = self.settings()[2]
        command, directory = namespace_disk_cache(['server', '--slot-save-path', str(self.root)], None, identity, True)
        self.assertEqual(command[0], 'server')
        self.assertEqual(directory, str(self.root / identity['cache_identity']))
        with self.assertRaises(ValueError):
            namespace_disk_cache(command, self.root, identity, True)


if __name__ == '__main__':
    unittest.main()
