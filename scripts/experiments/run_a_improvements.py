"""Staged A-only study: aggregate, ground, then constrained dense feedback."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import mmap
import os
from pathlib import Path
import re
import sqlite3
import sys
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
from bug_localization.lightweight_hypothesis import definitions
from bug_localization.paired_statistics import inference
from bug_localization.semantic_improvements import aggregation_scores, family, feedback_scores, grounded_scores, rank
from run_b_lightweight import IDENTITY, digest, metrics, read, save, summarize

CONFIG = ROOT/'configs/experiments/a_improvements_20261122.json'
WORKER_INPUTS = None


def initialize_worker(source, protocol):
    global WORKER_INPUTS
    WORKER_INPUTS = Inputs(Path(source), protocol)


def worker_evaluate(args):
    meta, checkpoint, requested, config, phase = args
    before = WORKER_INPUTS.cost.copy()
    record = evaluate(meta, checkpoint, requested, config, WORKER_INPUTS, phase)
    record['input_cost'] = {k:v-before.get(k,0) for k,v in WORKER_INPUTS.cost.items()
                            if k != 'max_feedback_control_error'}
    record['feedback_control_error'] = WORKER_INPUTS.cost['max_feedback_control_error']
    return record


class Inputs:
    def __init__(self, source, protocol):
        from datasets import load_from_disk
        self.source = source
        self.dataset = load_from_disk(str(ROOT/'datasets'/protocol['config']['dataset']))[protocol['config']['split']]
        self.symbols = {}
        self.cost = Counter()
        folder = ROOT/'cache/embeddings'/protocol['embedding']['fingerprint']
        assert read(folder/'model.json') == protocol['embedding']['identity']
        self.dimension = protocol['embedding']['identity']['dimension']
        self.db = sqlite3.connect(f'file:{folder / "vectors.sqlite3"}?mode=ro', uri=True)

    def vector(self, text):
        key = digest(text.encode())
        row = self.db.execute('SELECT data,checksum FROM vectors WHERE key=?', (key,)).fetchone()
        if row is None:
            raise ValueError('Missing frozen vector: '+key)
        raw, checksum = row
        assert digest(raw) == checksum
        v = np.frombuffer(raw, dtype='<f4')
        assert v.shape == (self.dimension,) and np.isfinite(v).all() and abs(np.linalg.norm(v)-1)<1e-3
        self.cost['vector_reads'] += 1
        return v

    def source_symbols(self, meta, inventory, paths):
        hashes = {r['file_path']:r['sha256'] for r in inventory['files']}
        symbols = {}
        for path in paths:
            sha = hashes[path]
            if sha not in self.symbols:
                raw = (ROOT/'repos'/meta['repository']/meta['before_fix_sha']/'source'/path).read_bytes()
                assert digest(raw) == sha, path
                self.symbols[sha] = definitions(raw.decode('utf-8', errors='replace'))
                self.cost['source_bytes'] += len(raw)
                self.cost['source_files_hashed'] += 1
            symbols[path] = self.symbols[sha]
        return symbols

    def feedback(self, result, assets, rows):
        needed = {r['best_chunk_id']: r['file_path'] for r in rows}
        texts = {}
        path = assets/'chunks.json'
        with path.open('rb') as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as raw:
            asset_sha = hashlib.sha256(raw).hexdigest()
            for match in re.finditer(rb'\n    "chunk_id": "([0-9a-f]{64})"', raw):
                key = match[1].decode()
                if key not in needed:
                    continue
                start = raw.rfind(b'\n  {', 0, match.start())+1
                end = raw.find(b'\n  }', match.end())+4
                chunk = json.loads(raw[start:end])
                assert chunk['chunk_id'] == key and chunk['file_path'] == needed[key]
                texts[key] = chunk['embedding_text']
            self.cost['chunk_asset_bytes_scanned'] += len(raw)
        assert set(texts) == set(needed)
        q = self.vector(result['query_info']['embedding_text'])
        vectors = np.stack([self.vector(texts[r['best_chunk_id']]) for r in rows])
        expected = np.array([r['best_chunk_similarity'] for r in rows])
        error = float(np.max(np.abs(vectors.astype(np.float64) @ q.astype(np.float64)-expected)))
        assert error <= 2e-5, error
        self.cost['max_feedback_control_error'] = max(self.cost['max_feedback_control_error'], error)
        return q, vectors, asset_sha


def names(config):
    output = ['A']
    output += [f'top{k}_mean' for k in config['aggregation']['top_mean_k']]
    output += [f'max_top3_w{w:g}' for w in config['aggregation']['max_top3_mix']]
    output += [f'lse_t{t:g}' for t in config['aggregation']['lse_temperature']]
    output += [f'length_p{p:g}' for p in config['length_penalty']]
    output += [f'anchor_d{d}_w{w:g}' for d in config['anchor_depths'] for w in config['anchor_weights']]
    output += [f'prf_k{k}_b{b:g}' for k in config['prf_feedback_files'] for b in config['prf_beta']]
    output += [f'combo_{base}_d{d}_w{w:g}' for base in config['combination_bases']
               for d in config['anchor_depths'] for w in config['anchor_weights']]
    assert len(output) == len(set(output))
    return output


def evaluate(meta, checkpoint, requested, config, inputs, phase):
    started = time.monotonic()
    path = inputs.source/checkpoint['result_file']
    raw = path.read_bytes()
    assert digest(raw) == checkpoint['result_sha256']
    result = json.loads(raw)
    assert all(meta[k] == checkpoint[k] == result[k] for k in IDENTITY)
    assert checkpoint['status'] == result['status'] == 'ok'
    assert result['ground_truth'] == checkpoint['ground_truth']
    body = inputs.dataset[meta['dataset_row']]['body']
    assert digest(json.dumps({'body':body}, sort_keys=True).encode()) == meta['query_sha256']
    assert result['query'] == body.strip()
    assert result['file_aggregation'] == 'max'
    rows = result['ranking']
    original = {r['file_path']:r['rank'] for r in rows}
    assert len(original) == len(rows)
    assert [r['rank'] for r in rows] == list(range(1,len(rows)+1))
    scores = aggregation_scores(result['chunk_scores'], config)
    assert set(scores['A']) == set(original)
    assert all(abs(scores['A'][r['file_path']]-r['best_chunk_similarity'])<1e-12 for r in rows)
    # A is reconstructed from all chunk scores, without labels or checkpoint rankings.
    assert rank(scores['A']) == [r['file_path'] for r in rows[:50]]
    for old_name in ['top3_mean', 'normalized_logsumexp']:
        new_name = old_name if old_name == 'top3_mean' else 'lse_t0.1'
        assert rank(scores[new_name]) == [r['file_path'] for r in result['aggregations'][old_name]['ranking'][:50]]
    assets = inputs.source/checkpoint['assets']
    inventory = read(assets/'candidates.json')
    assert set(original) == {r['file_path'] for r in inventory['files']}
    output = {**meta, 'phase':phase, 'status':'ok', 'ground_truth':checkpoint['ground_truth'],
              'source_result_sha256':checkpoint['result_sha256'],
              'query_truncated':result['query_info']['truncated'], 'methods':{}, 'anchor_evidence':{}}
    symbol_paths = set()
    derived_rows = {}
    for name in requested:
        if family(name) not in ('anchor', 'combination'):
            continue
        base = 'A' if name.startswith('anchor_') else name[6:].rsplit('_d',1)[0]
        depth = int(name.rsplit('_d',1)[1].split('_')[0])
        ordered = rank(scores[base], limit=len(rows))
        derived_rows[name] = [{'rank':i+1,'file_path':p,'score':scores[base][p]} for i,p in enumerate(ordered)]
        symbol_paths.update(ordered[:depth])
    symbols = inputs.source_symbols(meta, inventory, sorted(symbol_paths))
    feedback = None
    if any(name.startswith('prf_') for name in requested):
        feedback = inputs.feedback(result, assets, rows[:50])
        output['feedback_asset_sha256'] = feedback[2]
    for name in requested:
        kind = family(name)
        if kind in ('baseline','aggregation','length'):
            ranking = rank(scores[name])
        elif kind in ('anchor','combination'):
            depth = int(name.rsplit('_d',1)[1].split('_')[0])
            weight = float(name.rsplit('_w',1)[1])
            ordered = derived_rows[name]
            combined, evidence, matches = grounded_scores(ordered, body, symbols, weight, depth)
            ranking = rank(combined, {r['file_path']:r['rank'] for r in ordered})
            # Save contributing evidence for validation auditing, not gold-dependent features.
            if phase != 'development500':
                output['anchor_evidence'][name] = {p:{'score':evidence[p],**matches[p]} for p in ranking if evidence[p]>0}
        elif kind == 'prf':
            k = int(name.split('_k')[1].split('_')[0]); beta = float(name.split('_b')[1])
            values = feedback_scores(feedback[0],feedback[1],k,beta)
            ranking = rank({r['file_path']:float(s) for r,s in zip(rows[:50],values)},original)
            assert set(ranking) == {r['file_path'] for r in rows[:50]}
        else:
            raise ValueError(name)
        m = metrics(ranking,set(checkpoint['ground_truth']))
        if name == 'A':
            assert abs(m['MAP@10']-checkpoint['metrics']['average_precision_at_k'])<1e-12
            assert abs(m['MRR@10']-checkpoint['metrics']['reciprocal_rank_at_k'])<1e-12
        output['methods'][name] = {'ranking':ranking,'metrics':m}
    output['elapsed_seconds'] = time.monotonic()-started
    return output


def eligible(summary, name):
    b, m = summary['A'], summary[name]
    return m['MAP@10'] >= b['MAP@10']+.005 and all(m[k]>=b[k]-1e-12
           for k in ('MRR@10','Hit@10','Hit@50','MeanRecall@50'))


def select(summary):
    def best(ns):
        return min(ns,key=lambda n:(-summary[n]['MAP@10'],n))
    winners = {f:best([n for n in summary if family(n)==f])
               for f in sorted({family(n) for n in summary}-{'baseline'})}
    choices = [n for n in summary if n!='A' and eligible(summary,n)]
    chosen = best(choices) if choices else 'A'
    return {'families':winners,'selected':chosen,'eligible':choices,
            'family_eligible':{f:eligible(summary,n) for f,n in winners.items()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    parser.add_argument('--stop-after-development', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    config = read(CONFIG); source = ROOT/config['source_run']
    out = args.output or ROOT/config['output']
    out.mkdir(parents=True, exist_ok=True)
    frozen_files = [CONFIG,Path(__file__),ROOT/'src/bug_localization/semantic_improvements.py',
                    ROOT/'src/bug_localization/lightweight_hypothesis.py',ROOT/'src/bug_localization/paired_statistics.py',
                    ROOT/'scripts/experiments/run_b_lightweight.py',source/'selection.json',source/'S0/samples.json',
                    source/'protocol.json',source/'verification.json']
    hashes = {str(p):digest(p.read_bytes()) for p in frozen_files}
    protocol_path = out/'protocol.json'
    if protocol_path.exists():
        assert read(protocol_path)['hashes'] == hashes, 'Study inputs/implementation changed; use a new output'
    else:
        save(protocol_path,{'config':config,'hashes':hashes,'created_unix':time.time(),
                            'methods':names(config),'new_model_calls':0,'gpu_calls':0})
    assert read(source/'verification.json')['status']=='passed'
    selection = read(source/'selection.json'); checkpoints = read(source/'S0/samples.json')
    assert len(checkpoints)==len(selection['samples'])==config['expected_samples']
    assert all(r['status']=='ok' for r in checkpoints), 'This study requires the audited all-success A cohort'
    save(out/'selection.json',selection)
    dev = set(selection['development_rows'])
    executor = ProcessPoolExecutor(max_workers=args.workers, initializer=initialize_worker,
                                   initargs=(str(source),read(source/'protocol.json')))
    cases = {}; started = time.monotonic()
    for phase, isdev in [('development500',True),('remaining1794',False)]:
        if isdev:
            requested = names(config)
        else:
            frozen = read(out/'frozen_choice.json')
            requested = sorted({'A',frozen['selected'],*frozen['families'].values()})
        subset = [(m,c) for m,c in zip(selection['samples'],checkpoints) if (m['dataset_row'] in dev)==isdev]
        assert len(subset)==(500 if isdev else 1794)
        jobs = {}
        for meta,checkpoint in subset:
            if not (out/'cases'/f'{meta["position"]:04d}.json').exists():
                jobs[meta['position']] = executor.submit(worker_evaluate,(meta,checkpoint,requested,config,phase))
        phase_cases = []
        for step,(meta,checkpoint) in enumerate(subset,1):
            p = out/'cases'/f'{meta["position"]:04d}.json'
            if p.exists():
                record=read(p)
                assert all(record[k]==meta[k] for k in IDENTITY)
                assert set(record['methods'])==set(requested)
            else:
                record=jobs.pop(meta['position']).result()
                save(p,record)
            cases[meta['position']]=record; phase_cases.append(record)
            if step%10==0 or step==len(subset):
                print(f'{phase} {step}/{len(subset)} elapsed={time.monotonic()-started:.1f}s',flush=True)
                save(out/'status.json',{'phase':phase,'completed':step,'total':len(subset),'updated_unix':time.time()})
        summary=summarize(phase_cases); save(out/(phase+'_summary.json'),summary)
        if isdev:
            chosen=select(summary)
            frozen={**chosen,'frozen_unix':time.time(),'selection_data':'development500 only',
                    'development_metrics':{n:summary[n] for n in {'A',chosen['selected'],*chosen['families'].values()}}}
            if (out/'frozen_choice.json').exists():
                assert read(out/'frozen_choice.json')['selected']==chosen['selected']
            else:
                save(out/'frozen_choice.json',frozen)
            print(json.dumps(frozen,ensure_ascii=False),flush=True)
            if args.stop_after_development:
                executor.shutdown()
                return
    executor.shutdown()
    all_cases=[cases[k] for k in sorted(cases)]
    chosen=read(out/'frozen_choice.json')
    common=sorted({'A',chosen['selected'],*chosen['families'].values()})
    # Development contains the entire grid; full summaries contain only frozen methods.
    paired=[{**r,'methods':{n:r['methods'][n] for n in common}} for r in all_cases]
    summaries={'all2294':summarize(paired),
               'development500':read(out/'development500_summary.json'),
               'remaining1794':read(out/'remaining1794_summary.json')}
    save(out/'summary.json',summaries)
    save(out/'project_summary.json',{p:summarize([r for r in paired if r['repository']==p])
                                    for p in sorted({r['repository'] for r in paired})})
    validation=[r for r in paired if r['phase']=='remaining1794']; stats={}
    for name in common:
        if name=='A':continue
        stats[name]={key:inference([r['methods'][name]['metrics'][key]-r['methods']['A']['metrics'][key]
                                     for r in validation],[r['repository'] for r in validation],
                                   seed=20261122,repetitions=5000)
                     for key in config['primary_validation_metrics']}
    tests=[v for s in stats.values() for v in s.values()]; previous=0.
    for i,test in enumerate(sorted(tests,key=lambda x:x['p_two_sided_paired_randomization'])):
        previous=min(1.,max(previous,(len(tests)-i)*test['p_two_sided_paired_randomization']))
        test['p_holm_all_frozen_methods_metrics']=previous
    save(out/'validation_inference.json',{'comparisons':stats,'holm_family_size':len(tests)})
    with (out/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
        rs=[{'cohort':c,'method':n,**m} for c,methods in summaries.items() for n,m in methods.items()]
        writer=csv.DictWriter(f,fieldnames=list(rs[0]));writer.writeheader();writer.writerows(rs)
    for p,sha in hashes.items():assert digest(Path(p).read_bytes())==sha
    cost=Counter()
    for record in all_cases:cost.update(record['input_cost'])
    save(out/'cost.json',{**cost,'session_seconds':time.monotonic()-started,
                         'workers':args.workers,'max_feedback_control_error':max(r['feedback_control_error'] for r in all_cases),
                         'all_case_seconds':sum(r['elapsed_seconds'] for r in all_cases),
                         'new_generation_calls':0,'new_embedding_calls':0,'reranker_calls':0,'gpu_calls':0})
    save(out/'verification.json',{'status':'passed','samples':len(all_cases),'failed':0,
                                  'checks':['all result SHA256','sample identity and body hash',
                                            'candidate identity','before-fix source SHA256',
                                            'Max and two historical aggregation rank reproductions',
                                            'baseline MAP/MRR reproduction','embedding checksum and norm',
                                            'best-chunk cosine reproduction within2e-5','frozen input integrity']})
    save(out/'status.json',{'phase':'completed','samples':len(all_cases)})
    print('Completed: '+str(out),flush=True)


if __name__=='__main__':main()
