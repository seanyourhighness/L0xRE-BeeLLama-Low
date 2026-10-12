#!/usr/bin/env python3
"""Independently audit the complete paired150 outcomes and their runtime receipts."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

ROOT = Path('/home/sean/work/l0xre-r6-4090-lab-20261010')
REMOTE = '/home/sean/work/l0xre-r6-4090-20261010'
EXPECTED = {'temperature': .7, 'top_p': .95, 'top_k': 20, 'min_p': .05}


def audit(name):
    data = json.loads((ROOT / f'quality-{name}.json').read_text())
    command = json.loads((ROOT / f'quality-{name}-request-command.json').read_text())
    process = json.loads((ROOT / f'quality-{name}-process.json').read_text())
    seen = json.loads((ROOT / f'quality-{name}-sampler-attestation.json').read_text())
    log = (ROOT / f'quality-{name}-server.log').read_text(errors='replace')
    assert '--full' in command['argv'] and '--strict-thinking' in command['argv']
    assert data['totals']['total'] == 150 and data['pass_at_k']['total'] == 150 and data['pass_at_k']['k'] == 3
    assert len(data['packs']) == 8 and data['thinking_enabled'] and data['reasoning_effort'] == 'medium'
    assert data.get('thinking_validity') and all(v['status'] == 'ok' for v in data['thinking_validity'].values())
    scenarios, definitions, packs, failures = {}, {}, {}, []
    attempts = 0
    modes = Counter()
    for pack in data['packs']:
        assert pack['status'] == 'ok' and not pack.get('skipped')
        packs[pack['pack_id']] = {'version': pack['version'], 'upstream_commit': pack.get('upstream_commit'),
                                  'count': len(pack['scenarios'])}
        for case in pack['scenarios']:
            key = pack['pack_id'] + '/' + case['id']
            assert key not in scenarios
            definitions[key] = hashlib.sha256(json.dumps(case['raw_scenario'], sort_keys=True).encode()).hexdigest()
            all_attempts = [case] + case.get('retry_attempts', [])
            assert len(all_attempts) == case['attempt_count'] <= 3
            for attempt in all_attempts:
                params = attempt['sampling_params']
                assert all(abs(params[k] - value) < 1e-5 for k, value in EXPECTED.items()), key
                assert params.get('reasoning_effort') == 'medium' and params.get('seed') is None
                assert (attempt.get('request') or {}).get('seed') is None
                attempts += 1; modes[attempt['failure_mode']] += 1
            scenarios[key] = {'first': bool(case['passed']), 'within_three': bool(case['pass_at_k'])}
            if not case['passed']:
                failures.append({'id': key, 'within_three': bool(case['pass_at_k']), 'attempts': case['attempt_count'],
                                 'outcomes': [{k: a.get(k) for k in ['passed','failure_mode','detail','status_code']} for a in all_attempts]})
    assert len(scenarios) == 150
    scored = calibration = auxiliary = 0
    for params in seen.values():
        temp = params['temperature']
        if abs(temp) < 1e-5 and params.get('n_predict') == 200 and abs(params['top_p'] - 1) < 1e-5:
            calibration += 1
        else:
            assert all(abs(params[k] - EXPECTED[k]) < 1e-5 for k in ['top_p','top_k','min_p'])
            if abs(temp - .7) < 1e-5: scored += 1
            elif abs(temp - .1) < 1e-5 and params.get('n_predict') == 10000: auxiliary += 1
            else: raise AssertionError(('Unexpected observed sampler', params))
    assert scored >= 75
    expected_root = REMOTE + ('/package-current' if name == 'candidate' else '/package')
    assert process['env']['CUDA_VISIBLE_DEVICES'] == 'GPU-c4a782a6-fd5a-78c2-7d77-604c01cbffb8'
    assert 'Cpus_allowed_list:\t0-7' in process['status']
    assert '--seed' not in process['argv']
    for relative in ['bin/libggml-cuda.so.0.23.0', 'bin/libbridge-packed.so', 'bin/libllama.so.0.4.7']:
        assert expected_root + '/architectures/sm89/' + relative in process['maps'], relative
    assert 'L0XRE_REJECTION_VERIFY_ACTIVE' in log
    errors = [line for line in log.splitlines() if 'CUDA error' in line or 'illegal memory' in line.lower()]
    assert not errors
    first = sum(v['first'] for v in scenarios.values())
    third = sum(v['within_three'] for v in scenarios.values())
    report = {'pass_at_1': first, 'pass_at_3': third, 'attempts': attempts, 'all150_request_settings_verified': True,
              'observed_scored_requests': scored, 'observed_calibrations': calibration, 'observed_auxiliary_requests': auxiliary,
              'packs': packs, 'attempt_failure_modes': dict(modes), 'failures': failures,
              'guard_closures': log.count('loop guard force-closing'), 'cuda_error_lines': errors,
              'scope': 'Every recorded scored attempt setting checked; at least75 effective live scored request observations. Guard interventions retained separately.'}
    return report, scenarios, definitions


def paired(baseline, candidate, metric):
    gains = [k for k in baseline if not baseline[k][metric] and candidate[k][metric]]
    losses = [k for k in baseline if baseline[k][metric] and not candidate[k][metric]]
    n = len(gains) + len(losses)
    p = min(1., 2 * sum(math.comb(n, k) for k in range(min(len(gains), len(losses)) + 1)) / 2**n) if n else 1.
    return {'gains': gains, 'losses': losses, 'exact_mcnemar_p': p,
            'interpretation': 'A non-significant paired result is not proof of equivalence or improvement.'}


def main():
    before, bc, bd = audit('baseline')
    after, ac, ad = audit('candidate')
    assert before['packs'] == after['packs'] and bd == ad, 'Scenario definitions or pack versions changed'
    result = {'status': 'request-runtime-and-paired-audit-passed', 'accepted_for_release': False,
              'full_certification_complete': False, 'case_definitions_identical': 150,
              'baseline': before, 'candidate': after,
              'paired_first': paired(bc, ac, 'first'), 'paired_third': paired(bc, ac, 'within_three'),
              'manual_review_required': 'Review every failed/retried case, all new losses, guard events, timeouts and adapter traces before acceptance.'}
    (ROOT / 'QUALITY-AUDIT.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k,v in result.items() if k not in ['baseline','candidate']}, indent=2))


if __name__ == '__main__': main()
