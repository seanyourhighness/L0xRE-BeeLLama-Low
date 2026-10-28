"""Exercise memory boundaries without treating synthetic figures as measurements."""
import copy
import contextlib
import io
import unittest
from unittest.mock import patch
from vram_preflight import choose_arena, resolve_preflight, emit_preflight, wsl_settings


class Preflight(unittest.TestCase):
    def setUp(self):
        self.config = dict(arena_mib=2240,
            argv=['server', '-ub', '1024', '--shared-device-memory-mib', '2240'],
            vram_preflight=dict(desktop_reserve_mib=640, safety_margin_mib=64,
                validated_arenas=[dict(arena_mib=arena, required_free_mib=required, ubatch=ubatch,
                    headline_speeds={"depth_decode":speed},
                    evidence='synthetic unit-test fixture')
                    for arena, required, ubatch, speed in [(2240, 11600, 1024, 100), (2144, 11500, 1024, 93), (2048, 11400, 512, 67)]]))

    def test_largest_fit_with_minimized_desktop(self):
        original = copy.deepcopy(self.config)
        config, report = choose_arena(self.config, 12288, 12288)
        self.assertEqual(config['arena_mib'], 2240)
        self.assertEqual(config['argv'][-1], '2240')
        self.assertEqual(report['available_mib'], 12288)
        self.assertEqual(self.config, original)

    def test_desktop_is_not_subtracted_twice(self):
        a, _ = choose_arena(self.config, 12288, 12288)
        b, _ = choose_arena(self.config, 12288, 11664)
        self.assertEqual(a, b)

    def test_held_allocations_boundary_fallback_and_refusal(self):
        for held, expected in [(400, 2240), (600, 2240), (800, 2048), (1200, None)]:
            with self.subTest(held=held):
                if expected is None:
                    with self.assertRaisesRegex(ValueError, 'free 11088 MiB; required 11464 MiB'):
                        choose_arena(self.config, 12288, 12288 - held, consent="flag")
                else:
                    config, report = choose_arena(self.config, 12288, 12288 - held, consent="flag")
                    self.assertEqual(config['arena_mib'], expected)
                    self.assertEqual(report['warning'], 12288-held-report['required_mib'] < 640)

    def test_exact_boundary(self):
        self.assertEqual(choose_arena(self.config, 16384, 11464, consent="flag")[0]['arena_mib'], 2048)
        with self.assertRaises(ValueError):
            choose_arena(self.config, 16384, 11463)

    def test_linux_and_wsl_use_their_measured_capacity_policy(self):
        self.config['vram_preflight'].update(desktop_reserve_mib=0,
                                             wsl_desktop_reserve_mib=1024)
        with patch('vram_preflight.platform.release', return_value='6.8-generic'):
            selected, report = choose_arena(self.config, 12288, 12288)
            self.assertEqual(selected['arena_mib'], 2240)
            self.assertEqual(report['desktop_reserve_mib'], 0)
        with patch('vram_preflight.platform.release', return_value='6.18-microsoft-standard-WSL2'):
            selected, report = choose_arena(self.config, 12288, 12288)
            self.assertEqual(selected['arena_mib'], 2240)
            self.assertTrue(report['warning'])
            selected, report = choose_arena(self.config, 12288, 12288,
                                             reserve_override=0)
            self.assertEqual(selected['arena_mib'], 2240)
            self.assertTrue(report['reserve_override'])

    def test_explicit_override_still_requires_validation_and_space(self):
        with self.assertRaisesRegex(ValueError, 'has not been validated'):
            choose_arena(self.config, 32768, 32768, arena_override=2000)
        with self.assertRaisesRegex(ValueError, 'refused before loading'):
            choose_arena(self.config, 12288, 11663, arena_override=2240)
        config, report = choose_arena(self.config, 12288, 12288, reserve_override=400, arena_override=2240)
        self.assertEqual(config['arena_mib'], 2240)
        self.assertTrue(report['reserve_override'])

    def test_missing_and_invalid_measurements_refuse(self):
        for change in ['missing', 'empty', 'duplicate', 'no-evidence', 'negative', 'zero-margin', 'invalid-ubatch', 'invalid-speed']:
            with self.subTest(change=change):
                config = copy.deepcopy(self.config)
                policy = config['vram_preflight']
                if change == 'missing':
                    del config['vram_preflight']
                elif change == 'empty':
                    policy['validated_arenas'] = []
                elif change == 'duplicate':
                    policy['validated_arenas'].append(policy['validated_arenas'][0])
                elif change == 'no-evidence':
                    policy['validated_arenas'][0]['evidence'] = ''
                elif change == 'negative':
                    policy['desktop_reserve_mib'] = -1
                elif change == 'zero-margin':
                    policy['safety_margin_mib'] = 0
                elif change == 'invalid-ubatch':
                    policy['validated_arenas'][0]['ubatch'] = 0
                else:
                    policy['validated_arenas'][0]['headline_speeds']['depth_decode'] = float('nan')
                with self.assertRaises(ValueError):
                    choose_arena(config, 12288, 12288)

    def resolve(self, frees, answers=(), *, tty=True, **kwargs):
        self.query = iter(frees)
        with patch('vram_preflight.sys.stdin.isatty', return_value=tty), \
             patch('builtins.input', side_effect=answers) as prompt, \
             patch('vram_preflight.subprocess.check_output', return_value='42, browser, 800'), \
             contextlib.redirect_stderr(io.StringIO()) as output:
            value = resolve_preflight(self.config, 12288, lambda: next(self.query),
                                      'smi', '0', environ=kwargs.pop('environ', {}), **kwargs)
        self.output, self.prompts = output.getvalue(), prompt.call_count
        return value

    def test_full_speed_never_prompts(self):
        _, report = self.resolve([12288])
        self.assertEqual(self.prompts, 0)
        self.assertEqual(report['consent'], 'none')
        self.assertEqual(report['rung'], 0)

    def test_retry_requeries_until_full_speed_fits(self):
        _, report = self.resolve([11488, 11580, 12288], ['r', 'r'])
        self.assertEqual(self.prompts, 2)
        self.assertEqual(report['rung'], 0)
        self.assertEqual(report['free_mib'], 12288)
        self.assertIn('[R] Retry / [S] Start slower / [Q] Quit', self.output)
        self.assertIn('motherboard', self.output)
        self.assertIn('browser', self.output)

    def test_retry_then_start_slower_swaps_ubatch(self):
        config, report = self.resolve([11488, 11488, 11488], ['r', 's'])
        self.assertEqual(config['arena_mib'], 2048)
        self.assertEqual(config['argv'][config['argv'].index('-ub')+1], '512')
        self.assertEqual(report['consent'], 'interactive')
        self.assertAlmostEqual(report['speed_cost_pct'], 33)
        self.assertEqual(report['ubatch'], 512)

    def test_speed_order_is_not_arena_size_order(self):
        self.config['vram_preflight']['validated_arenas'][1]['arena_mib'] = 2300
        config, report = self.resolve([11580], tty=False, accept_slower=True)
        self.assertEqual(config['arena_mib'], 2300)
        self.assertEqual(report['rung'], 1)

    def test_quit_loads_nothing(self):
        with self.assertRaisesRegex(ValueError, 'quit before loading'):
            self.resolve([11488], ['q'])

    def test_no_tty_without_consent_refuses(self):
        with self.assertRaisesRegex(ValueError, 'Non-interactive launch refused'):
            self.resolve([11488], tty=False)

    def test_flag_and_env_consent_use_fastest_fitting_rung(self):
        for kwargs, source in [(dict(accept_slower=True), 'flag'),
                               (dict(environ={'L0XRE_ACCEPT_SLOWER':'1'}), 'env')]:
            _, report = self.resolve([11580], tty=False, **kwargs)
            self.assertEqual(report['rung'], 1)
            self.assertEqual(report['consent'], source)
            self.assertEqual(self.prompts, 0)

    def test_no_rung_fits_noninteractive_even_with_consent(self):
        with self.assertRaisesRegex(ValueError, 'refused before loading'):
            self.resolve([10000], tty=False, accept_slower=True)

    def test_no_rung_fits_interactive_retry_then_quit(self):
        with patch('vram_preflight.sys.stdin.isatty', return_value=True), \
             patch('builtins.input', side_effect=['s', 'r', 'q']), \
             patch('vram_preflight.subprocess.check_output', side_effect=OSError), \
             contextlib.redirect_stderr(io.StringIO()) as output:
            with self.assertRaisesRegex(ValueError, 'quit before loading'):
                resolve_preflight(self.config, 12288, lambda:10000, 'smi', '0', environ={})
        self.assertIn('[R] Retry / [Q] Quit', output.getvalue())
        self.assertNotIn('[S]', output.getvalue())

    def test_slower_explicit_arena_is_consent_and_is_validated(self):
        _, report = self.resolve([12288], tty=False, arena_override=2048)
        self.assertEqual(report['consent'], 'explicit-arena')
        self.assertEqual(report['rung'], 2)
        with self.assertRaisesRegex(ValueError, 'has not been validated'):
            self.resolve([12288], tty=False, arena_override=2000)

    def test_service_cannot_prompt_even_with_tty(self):
        with self.assertRaisesRegex(ValueError, 'Non-interactive launch refused'):
            self.resolve([11488], environ={'L0XRE_NONINTERACTIVE':'1'})

    def test_emit_banner_and_required_receipt_fields(self):
        _, report = self.resolve([11488], tty=False, accept_slower=True)
        for key in ['rung','arena_mib','ubatch','free_mib','required_mib','consent','speed_cost_pct']:
            self.assertIn(key, report)
        with contextlib.redirect_stdout(io.StringIO()) as output, contextlib.redirect_stderr(io.StringIO()):
            emit_preflight(report, 'smi', '0')
        self.assertIn('reduced-speed arena 2048 MiB (user accepted)', output.getvalue())

    def test_closed_stdin_quits_instead_of_loading(self):
        with self.assertRaisesRegex(ValueError, 'quit before loading'):
            self.resolve([11488], [EOFError()])


class WslSettings(unittest.TestCase):
    def test_disables_only_unsupported_allocator_option_and_preserves_cache(self):
        env = dict(PYTORCH_ALLOC_CONF='max_split_size_mb:64, expandable_segments:True',
                   PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
        config = dict(argv=['server', '--cache-ram', '4096'])
        original = copy.deepcopy(config)
        report = wsl_settings(env, config, [8 * 2**30], kernel='6.18-microsoft-standard-WSL2',
                              meminfo='MemTotal: 16777216 kB\n')
        self.assertEqual(env['PYTORCH_ALLOC_CONF'], 'max_split_size_mb:64')
        self.assertNotIn('PYTORCH_CUDA_ALLOC_CONF', env)
        self.assertEqual(config, original)
        self.assertEqual(report['vm_memory_mib'], 16384)
        self.assertEqual(len(report['warnings']), 3)
        self.assertIn('70%', report['warnings'][-1])

    def test_native_linux_leaves_allocator_unchanged(self):
        env = dict(PYTORCH_ALLOC_CONF='expandable_segments:True')
        self.assertIsNone(wsl_settings(env, {}, [], kernel='6.8-generic'))
        self.assertEqual(env['PYTORCH_ALLOC_CONF'], 'expandable_segments:True')

    def test_large_wsl_vm_has_no_memory_warning(self):
        report = wsl_settings({}, dict(argv=['server', '--cache-ram', '4096']), [8 * 2**30],
                              kernel='microsoft-WSL2', meminfo='MemTotal: 67108864 kB\n')
        self.assertEqual(report['warnings'], [])


if __name__ == '__main__':
    unittest.main()
