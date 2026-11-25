"""Freeze the transfer cohort and use the existing unchanged proposal generator."""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pilot_perspective_rank import C3, PROJECT, OUT, ServerGenerator, read, run_case, sha256_file, write_json

ROOT = PROJECT/'results/c_evidence_challenge/v1'
ARC = PROJECT/'results/c_agent90/round1/joint'
TRANSFER = ROOT/'transfer_proposals'


def freeze(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and read(path)!=data:
        raise ValueError(f'Frozen input changed: {path}')
    write_json(path,data)


def main():
    protocol = read(OUT/'protocol.json')
    previous = protocol['samples'] + read(OUT.parent/'confirmation90_v1/protocol.json')['samples']
    seen = {(s['repository'],s['issue_id']) for s in previous}
    groups = defaultdict(list)
    source = {s['position']:s for s in read(C3/'samples.json')}
    for s in read(ARC.parent/'selection.json')['samples']:
        if (s['repository'],s['issue_id']) not in seen:
            c = source[s['position']]
            assert all(s[k]==c[k] for k in ('repository','issue_id','before_fix_sha','dataset_row'))
            groups[s['repository']].append(dict(s,query_sha256=c['query_sha256']))
    rows = [s for name in sorted(groups) for s in sorted(groups[name],key=lambda s:s['position'])[:5]]
    assert len(rows)==45 and len({(s['repository'],s['issue_id']) for s in rows})==45
    freeze(TRANSFER/'selection.json',dict(samples=rows,rule='first five by position per project, excluding dev27 and confirmation90'))
    cfg = dict(protocol['identity']['settings'],inference_endpoint='http://127.0.0.1:8113',
               server_manifest='models/BLAgent-Qwen3-8B-BF16/manifest.json')
    gen = ServerGenerator(PROJECT,cfg)
    assert gen.identity==protocol['identity']
    freeze(TRANSFER/'protocol.json',dict(generator=gen.identity,script_sha256=sha256_file(Path(__file__)),
           proposal_sha256=sha256_file(Path(__file__).with_name('pilot_perspective_rank.py'))))
    with ThreadPoolExecutor(max_workers=2) as pool:
        for i,_ in enumerate(pool.map(lambda s:run_case(s,gen,ARC,TRANSFER),rows),1):
            print(f'proposals {i}/45',flush=True)


if __name__=='__main__':
    main()
