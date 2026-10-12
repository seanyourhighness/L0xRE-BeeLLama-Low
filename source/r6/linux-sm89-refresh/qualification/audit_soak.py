#!/usr/bin/env python3
"""Audit the completed soak records independently of the controller's pass flag."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

LAB = Path('/home/sean/work/l0xre-r6-4090-lab-20261010')


def main():
    summary = json.loads((LAB / 'SOAK.json').read_text())
    root = LAB / 'probes/stability-soak'
    state = json.loads((root / 'status.json').read_text())
    assert summary['status'] == 'pass' and summary['elapsed_seconds'] >= 1800
    assert state['state'] == 'gate_pass' and state['server_exit'] in [0,-2]
    assert state['finished'] - state['started'] >= summary['elapsed_seconds']
    baselines = {}; counts = Counter(); repeats = 0
    records = sorted(root.glob('cycle*.json'))
    assert len(records) == summary['requests']
    for path in records:
        item = json.loads(path.read_text()); request,response = item['request'],item['response']
        assert not request['cache_prompt'] and len(response['tokens']) == request['n_predict'] and not response.get('truncated')
        assert response['timings']['predicted_n'] == request['n_predict']
        if '-prefill' in path.stem:
            depth = int(path.stem.split('-prefill')[1]); key = 'prefill' + str(depth)
            assert response['timings']['prompt_n'] == depth and response['timings'].get('cache_n',0) == 0
            assert request['temperature'] == 0 and request['n_predict'] == 1
        else:
            key = re.sub(r'^cycle\d+-\d+-', '', path.stem)
            assert request['n_predict'] == (128 if request['temperature'] == 0 else 64)
        value = hashlib.sha256(json.dumps(response['tokens']).encode()).hexdigest()
        if key in baselines:
            repeats += 1; assert baselines[key] == value, path.name
        baselines[key] = value; counts[key] += 1
    for depth in [2048,16384,32768,65536,77824]:assert counts['prefill'+str(depth)] == summary['cycles']
    for key in ['short0-0','short0-0.7','short1-0','short1-0.7']:assert counts[key] == 5*summary['cycles']
    assert repeats == summary['repeat_comparisons']
    log = (root / 'server.log').read_text(errors='replace')
    assert not re.search(r'CUDA error|illegal memory|device-side assert|GPU is lost|CUBLAS_STATUS_EXECUTION_FAILED',log,re.I)
    samples = [json.loads(line) for line in (root / 'telemetry.jsonl').read_text().splitlines()]
    assert samples and all(float(row['gpu'].split(',')[1]) < 82 for row in samples)
    assert all('Not Active' in row['gpu'] for row in samples)
    peak_temp = max(float(row['gpu'].split(',')[1]) for row in samples)
    peak_memory = max(float(row['gpu'].split(',')[0]) for row in samples)
    result = {'status':'pass','duration_seconds':summary['elapsed_seconds'],'requests':len(records),
              'cycles':summary['cycles'],'repeat_comparisons':repeats,'scenario_counts':dict(counts),
              'telemetry_samples':len(samples),'peak_temperature_c':peak_temp,'peak_memory_mib':peak_memory,
              'server_exit':state['server_exit'],'zero_cuda_error_lines':True,
              'scope':'One frozen RTX4090 profile mixed-depth30-minute workload; does not resolve earlier system RAM findings.'}
    (LAB / 'SOAK-AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
