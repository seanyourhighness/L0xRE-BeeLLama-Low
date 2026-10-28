#!/usr/bin/env python3
"""Measure the same seeded generation workload at explicit prompt depths.

Run against an isolated, already-started qualification worker. This tool does
not launch a GPU worker or claim that one sampled run establishes parity.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request

PREFIX = 'Read this archive carefully. After the archive, write the requested essay.\n'
FILLER = ('Archive record: The blue oak tree stands beside the quiet river. '
          'Inventory includes paper, pencils, folders, and green books. Status is normal.\n')
SUFFIX = ('\nWrite a detailed 1000-word essay explaining transformer attention, with '
          'equations and examples. Continue until every aspect is covered.\nEssay:')


def post(url, path, body, timeout=60):
    request = urllib.request.Request(url + path, json.dumps(body).encode(),
                                     {'Content-Type': 'application/json'})
    return urllib.request.urlopen(request, timeout=timeout)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--url', required=True, help='Isolated worker URL, e.g. http://127.0.0.1:30192')
    parser.add_argument('--context', required=True, type=int)
    parser.add_argument('--depth', required=True, type=int, nargs='+')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--repeats', type=int, default=1)
    args = parser.parse_args()
    if args.repeats < 1 or args.context < 1025 or any(x < 1 or x + 1023 >= args.context for x in args.depth):
        parser.error('Each depth must leave room for 1,023 generated tokens and one terminal position')
    if args.output.exists():
        parser.error('Output already exists; preserve previous evidence')
    base = args.url.rstrip('/')
    def tokenize(text, special):
        with post(base, '/tokenize', {'content': text, 'add_special': special}) as response:
            return json.load(response)['tokens']
    prefix, filler, suffix = tokenize(PREFIX, True), tokenize(FILLER, False), tokenize(SUFFIX, False)
    if not filler or min(args.depth) < len(prefix) + len(suffix):
        parser.error('The requested depth cannot contain the common prompt and suffix')
    args.output.mkdir(parents=True)
    rows = []
    for depth in args.depth:
        count = depth - len(prefix) - len(suffix)
        tokens = prefix + (filler * ((count + len(filler) - 1) // len(filler)))[:count] + suffix
        assert len(tokens) == depth
        prompt_hash = hashlib.sha256(json.dumps(tokens, separators=(',', ':')).encode()).hexdigest()
        for rep in range(args.repeats):
            body = dict(prompt=tokens, n_predict=1023, temperature=.6, top_p=.95, top_k=20,
                        min_p=0, seed=17, cache_prompt=False, stream=True, ignore_eos=True,
                        return_tokens=True)
            begin = time.monotonic()
            first = None
            last = None
            text = []
            with post(base, '/completion', body, timeout=2400) as response:
                for raw in response:
                    if not raw.startswith(b'data: '):
                        continue
                    data = raw[6:].strip()
                    if data == b'[DONE]':
                        break
                    event = json.loads(data)
                    if event.get('content'):
                        if first is None:
                            first = time.monotonic() - begin
                        text.append(event['content'])
                    if 'timings' in event:
                        last = event
            wall = time.monotonic() - begin
            if last is None or first is None:
                raise RuntimeError('Missing streamed content or final timings')
            timing = last['timings']
            checks = dict(prompt_exact=timing['prompt_n'] == depth,
                          full_generation=timing['predicted_n'] == 1023,
                          no_truncation=not last.get('truncated', False),
                          cache_cold=timing.get('cache_n', 0) == 0,
                          within_context=depth + timing['predicted_n'] < args.context)
            row = dict(prompt_tokens=depth, rep=rep, prompt_sha256=prompt_hash,
                       content_sha256=hashlib.sha256(''.join(text).encode()).hexdigest(),
                       ttft_s=first, wall_s=wall, timings=timing, checks=checks)
            rows.append(row)
            temp = args.output / 'results.tmp'
            temp.write_text(json.dumps(dict(context=args.context, results=rows,
                                            complete=False), indent=2) + '\n')
            temp.replace(args.output / 'results.json')
            print(json.dumps(row), flush=True)
            if not all(checks.values()):
                raise RuntimeError('Depth run failed a completeness check')
    temp = args.output / 'results.tmp'
    temp.write_text(json.dumps(dict(context=args.context, results=rows, complete=True,
                                    repetitions=args.repeats), indent=2) + '\n')
    temp.replace(args.output / 'results.json')


if __name__ == '__main__':
    main()
