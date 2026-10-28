#!/usr/bin/env python3
"""Compare completed same-prompt depth receipts without assuming outputs match."""
import argparse
import json
from pathlib import Path


def source(value):
    label, separator, path = value.partition('=')
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError('Use LABEL=PATH')
    return label, Path(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',action='append',type=source,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if len(args.input)<2 or args.output.exists():
        p.error('Provide at least two input receipts and a new output path')
    inputs={}
    for label,path in args.input:
        if label in inputs:
            p.error('Input labels must be unique')
        d=json.loads(path.read_text())
        if not d.get('complete'):
            p.error(f'{label}: receipt is incomplete')
        rows={}
        for x in d['results']:
            depth=x['prompt_tokens']
            if depth in rows or not all(x['checks'].values()):
                p.error(f'{label}: duplicate depth or failed run')
            t=x['timings']
            rows[depth]={'prompt_sha256':x['prompt_sha256'],'content_sha256':x['content_sha256'],
                         'decode_tps':t['predicted_per_second'],'prefill_tps':t['prompt_per_second'],
                         'drafted':t.get('draft_n'),'accepted':t.get('draft_n_accepted')}
        inputs[label]=rows
    shared=set.intersection(*(set(rows) for rows in inputs.values()))
    if not shared:
        p.error('Inputs have no shared depths')
    results=[]
    for depth in sorted(shared):
        entries={label:rows[depth] for label,rows in inputs.items()}
        if len({r['prompt_sha256'] for r in entries.values()})!=1:
            p.error(f'Prompt tokens differ at {depth}')
        results.append({'prompt_tokens':depth,'inputs':entries,
                        'same_output_across_all':len({r['content_sha256'] for r in entries.values()})==1})
    out={'complete':True,'single_run_per_depth':True,'results':results,
         'caveat':'Cross-architecture output and speculative acceptance may differ; rates need repeated runs.'}
    args.output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))


if __name__=='__main__':
    main()
