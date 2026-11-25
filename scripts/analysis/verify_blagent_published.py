"""Recompute published BLAgent predictions; this is NOT a fresh inference run."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT/'src'))
from bug_localization.batch import write_json
from bug_localization.snapshot import sha256_file

REVISION='9f91f63e50aec8ad87a11e11148b086dfcd5e66d'
BLOB='767c22721cc55882cecfdb16e81d1932e6ca07df'


def verify(path):
    actual=subprocess.check_output(['git','hash-object',str(path)],text=True).strip()
    if actual!=BLOB: raise ValueError('Published artifact differs from pinned author Git blob')
    rows=json.loads(path.read_text())
    if len(rows)!=300 or {r['swe_data_index'] for r in rows}!=set(range(300)):
        raise ValueError('Expected each author SWE-bench Lite index exactly once')
    rank=[]; lengths=[]
    for row in rows:
        paths=row['final_reranked_files'];gold=row['patch_file']
        if not isinstance(gold,str) or not gold or len(set(paths))!=len(paths):
            raise ValueError('Invalid labels or duplicated predictions')
        rank.append(paths.index(gold)+1 if gold in paths else None)
        lengths.append(len(paths))
    metrics={'MRR':sum(1/r for r in rank if r)/300,
             **{f'Top-{k}':sum(r is not None and r<=k for r in rank)/300 for k in (1,3,5,10)}}
    expected={'MRR':.8514,'Top-1':.7867,'Top-3':.9233,'Top-5':.9333,'Top-10':.9433}
    if any(round(metrics[k],4)!=v for k,v in expected.items()):
        raise ValueError('Published score mismatch')
    return {'status':'passed','kind':'published_predictions_recalculation_not_fresh_inference',
            'author_revision':REVISION,'author_git_blob':BLOB,'sha256':sha256_file(path),
            'samples':300,'metrics':metrics,'readme_rounded_metrics':expected,
            'minimum_ranking_length':min(lengths),'maximum_ranking_length':max(lengths),
            'gold_protocol':'Single patch_file supplied by author; official evaluator takes first patch file',
            'scope':'Checks artifact completeness and arithmetic, not independence, original model execution or all-file ground truth'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=PROJECT/'results/blagent/author-gpt-oss-300.json')
    p.add_argument('--output',type=Path,default=PROJECT/'results/blagent/published-verification.json')
    args=p.parse_args();result=verify(args.input);args.output.parent.mkdir(parents=True,exist_ok=True)
    write_json(args.output,result);print(json.dumps(result,indent=2))
