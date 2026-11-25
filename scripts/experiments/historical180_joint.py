"""Resolve the historical 180-report manifest; generate only missing full-evidence heads."""
import copy
from pathlib import Path

from pilot_c3_joint import BL,C3,PROJECT,ServerGenerator,infer,inference_input,read,sha256_file,write_json
from pilot_perspective_rank import DEV,OUT
from prepare_challenge_transfer import ARC,ROOT,freeze

RUN=ROOT.parent/'historical180_v1'
SEL=PROJECT/'configs/experiments/selections/crossproject_180_seed20261110.json'
FIELDS=('position','dataset_row','repository','issue_id','before_fix_sha')


def cases():
    rows=read(SEL)['samples']
    assert len(rows)==len({(r['repository'],r['issue_id']) for r in rows})==180
    return rows


def locate(s):
    key=f'{s["position"]:04d}'
    for archive in (DEV/'confirm_joint10',ARC,DEV/'dev_joint10',RUN/'joint'):
        p=archive/'cases'/key
        if (p/'record.json').exists():
            rec=read(p/'record.json')
            assert all(rec[k]==s[k] for k in FIELDS)
            if rec['status']=='ok':
                assert sha256_file(p/'result.json')==rec['result_sha256']
            return archive
    return None


def main():
    samples=cases()
    cs={s['position']:s for s in read(C3/'samples.json')}
    cfg=dict(read((DEV/'confirm_joint10')/'protocol.json')['generator_identity']['settings'],
             server_manifest='models/BLAgent-Qwen3-8B-BF16/manifest.json',inference_endpoint='http://127.0.0.1:8113',server_slots=1)
    gen=ServerGenerator(PROJECT,cfg)
    freeze(RUN/'joint_protocol.json',dict(identity=gen.identity,selection_sha256=sha256_file(SEL),
           prompt_script_sha256=sha256_file(Path(__file__).with_name('pilot_c3_joint.py')),
           script_sha256=sha256_file(Path(__file__)),note='single inference slot, same BF16/context/temperature/prompt/retry'))
    missing=[s for s in samples if locate(s) is None]
    for i,s in enumerate(missing,1):
        key=f'{s["position"]:04d}'
        src=cs[s['position']]
        assert all(src[k]==s[k] for k in (*FIELDS,'query_sha256','labels_sha256')) and src['status']=='ok'
        path=C3/src['result_file']
        assert sha256_file(path)==src['result_sha256']
        query,head,ev,tail=inference_input(read(path),read(BL/'cases'/f'{s["dataset_row"]+1:04d}'/'inputs.json'),10)
        dest=RUN/'joint/cases'/key
        dest.mkdir(parents=True,exist_ok=True)
        g=copy.copy(gen)
        g.start_case(dest)
        try:
            ranking,reason=infer(g,query,head,ev,dest)
        except ValueError as e:
            if not any(x in str(e) for x in ('exact candidate permutation','context overflow','exhausted the output budget')):
                raise
            rec=dict(s,status='failed',error=str(e))
        else:
            write_json(dest/'result.json',dict(ranking=ranking+tail,reasoning=reason,model_calls=g.calls))
            rec=dict(s,status='ok',result_sha256=sha256_file(dest/'result.json'))
        write_json(dest/'record.json',rec)
        print(f'joint {i}/{len(missing)}: {key} {rec["status"]}',flush=True)


if __name__=='__main__':
    main()
