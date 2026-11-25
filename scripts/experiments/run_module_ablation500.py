"""Frozen RRF module ablation, two resumable GPU shards and independent audit."""
import argparse
from collections import defaultdict
import csv
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from bug_localization.batch import aggregate_results, write_json
from bug_localization.bm25 import ranking_metrics
from bug_localization.chunk_ablation import select_evidence
from bug_localization.reranking import LocalReranker, read_json, rerank
from bug_localization.snapshot import sha256_file
from run_chunk_ablation500 import FixedSources
from run_mixed_recall500 import same_sample, now


def fuse(case):
    a = case['original']['ranking']
    b = case['hypothesis']['ranking']
    ar = {r['file_path']: r['rank'] for r in a}
    br = {r['file_path']: r['rank'] for r in b}
    if ar.keys() != br.keys():
        raise ValueError('Unpaired inventories')
    scores = {p: 1/(60+ar[p]) + 1/(60+br[p]) for p in ar}
    ordered = sorted(ar, key=lambda p: (-scores[p], p))
    pool = {r['file_path'] for r in case['candidates']}
    ordered = [p for p in ordered if p in pool] + [p for p in ordered if p not in pool]
    return [{'file_path': p, 'rank': i, 'score': scores[p]} for i, p in enumerate(ordered, 1)]


def audit(case, saved):
    """Rebuild pool, RRF order, mixed chunk IDs and file maxima independently."""
    a = case['original']['ranking']; b = case['hypothesis']['ranking']
    primary = [r['file_path'] for r in b[:40]]
    pool = primary + [r['file_path'] for r in a if r['file_path'] not in set(primary)][:50-len(primary)]
    if pool != [r['file_path'] for r in saved['candidates']]:
        raise ValueError('Candidate reconstruction failed')
    ar = {r['file_path']: i for i, r in enumerate(a, 1)}
    br = {r['file_path']: i for i, r in enumerate(b, 1)}
    order = sorted(pool, key=lambda p: (-(1/(60+ar[p])+1/(60+br[p])), p))
    if saved['ab_ranking'] != order:
        raise ValueError('RRF order failed')
    grouped = {}
    for route in ('original', 'hypothesis'):
        grouped[route] = defaultdict(list)
        for r in case[route]['chunk_scores']:
            grouped[route][r['file_path']].append(r)
        for rows in grouped[route].values():
            rows.sort(key=lambda r: (-r['cosine_similarity'], r['chunk_id']))
    wanted = []
    for p in order:
        ids = [grouped['original'][p][0]['chunk_id']]
        for r in grouped['hypothesis'][p]:
            if len(ids) == 3: break
            if r['chunk_id'] not in ids: ids.append(r['chunk_id'])
        wanted.extend(ids)
    if wanted != [r['chunk_id'] for r in saved['scored_chunks']]:
        raise ValueError('Mixed evidence/order failed')
    lookup = {r['chunk_id']: r for r in case['chunks']}
    best = {}
    for r in saved['scored_chunks']:
        if any(r[k] != lookup[r['chunk_id']][k] for k in ('file_path', 'embedding_text')):
            raise ValueError('Scored text failed')
        if not -float('inf') < r['score'] < float('inf'):
            raise ValueError('Nonfinite score')
        if r['file_path'] not in best or r['score'] > best[r['file_path']]['score']:
            best[r['file_path']] = r
    expected = sorted(order, key=lambda p: (-best[p]['score'], order.index(p)))
    if [r['file_path'] for r in saved['ranking']] != expected:
        raise ValueError('Max aggregation/tie failed')
    for i, r in enumerate(saved['ranking'], 1):
        if r['rank'] != i or r['score'] != best[r['file_path']]['score'] or r['best_chunk_id'] != best[r['file_path']]['chunk_id']:
            raise ValueError('Ranking metadata failed')
    if saved['query'] != case['query'] or saved['ground_truth'] != case['gold'] or saved['source_hashes'] != case['source_hashes']:
        raise ValueError('Source/query/label changed')
    for name, paths in (('AB', order), ('ABC', expected)):
        if saved['metrics'][name] != ranking_metrics(paths, case['gold'], 10):
            raise ValueError('Metrics failed')


def verify_protocol(folder):
    protocol = read_json(folder/'protocol.json')
    for p, h in protocol['files_sha256'].items():
        if sha256_file(Path(p)) != h:
            raise ValueError('Frozen input/implementation changed: '+p)
    return protocol


def worker(folder, gpu):
    protocol = verify_protocol(folder)
    source = FixedSources(protocol['source_config'])
    samples = [s for s in source.samples if (s['position']-1) % 2 == gpu]
    target = folder/f'gpu{gpu}'; target.mkdir(exist_ok=True)
    records = read_json(target/'samples.json') if (target/'samples.json').exists() else []
    for record, sample in zip(records, samples):
        same_sample(record, sample)
        if record['position'] != sample['position']: raise ValueError('Shard checkpoint mismatch')
        if record['status'] == 'ok' and sha256_file(folder/record['result_file']) != record['result_sha256']:
            raise ValueError('Checkpoint changed')
    model = LocalReranker(ROOT, protocol['reranker'])
    reference = read_json(source.reference/'C/effective_config.json')['model']
    if model.identity != reference: raise ValueError('Model identity differs from reference')
    write_json(target/'model.json', {'physical_gpu': gpu, 'identity': model.identity})
    for sample in samples[len(records):]:
        if source.failures(sample):
            record = {**sample, 'status': 'failed', 'error_type': 'SourceDependencyFailure',
                      'error': 'Existing hypothesis generation failure retained; no fallback'}
        else:
            started = time.monotonic()
            case = source.load(sample)
            full = fuse(case)
            head, chosen = select_evidence(case['candidates'], full, case['chunks'],
                case['original']['chunk_scores'], case['hypothesis']['chunk_scores'], 'mixed')
            scored = model.score(case['query'], [r['embedding_text'] for r in chosen])
            ranking = rerank(head, chosen, scored)
            ab = [r['file_path'] for r in head]
            saved = {**sample, 'query': case['query'], 'ground_truth': case['gold'],
                'source_hashes': case['source_hashes'], 'candidates': case['candidates'],
                'ab_ranking': ab, 'ranking': ranking,
                'scored_chunks': [{**r, **s} for r, s in zip(chosen, scored)],
                'metrics': {'AB': ranking_metrics(ab, case['gold'], 10),
                            'ABC': ranking_metrics([r['file_path'] for r in ranking], case['gold'], 10)}}
            audit(case, saved)
            path = folder/'cases'/f"{sample['position']:03d}.json"
            write_json(path, saved)
            record = {**sample, 'status': 'ok', 'metrics': saved['metrics'],
                'missing_gold': sorted(set(case['gold'])-case['inventory']),
                'result_file': str(path.relative_to(folder)), 'result_sha256': sha256_file(path),
                'elapsed_seconds': time.monotonic()-started}
        records.append(record)
        write_json(target/'samples.json', records)
        write_json(target/'progress.json', {'processed': len(records), 'selected': len(samples),
            'failures': sum(r['status'] != 'ok' for r in records), 'updated_at': now()})
        print(f"GPU {gpu}: {len(records)}/{len(samples)}, global {sample['position']}", flush=True)


def finalize(folder):
    protocol = verify_protocol(folder)
    source = FixedSources(protocol['source_config'])
    paired = sorted(sum([read_json(folder/f'gpu{g}/samples.json') for g in (0,1)], []), key=lambda r:r['position'])
    if len(paired) != 500 or [r['position'] for r in paired] != list(range(1,501)):
        raise ValueError('Incomplete shards')
    baselines = read_json(ROOT/'results/recall_baselines/20261112_L0_S0_H0_500/samples.json')
    records = {'A': [], 'B': [], 'A+B': [], 'A+B+C': []}
    for sample, record in zip(source.samples, paired):
        same_sample(sample, record)
        roots, pos, source_rows, _ = source.locate(sample)
        for name, key, root, filename, row in zip(('A','B'), ('S0','H0'), roots, ('q0_result.json','direct_result.json'), source_rows):
            base = baselines[key][sample['position']-1]; same_sample(base, sample)
            if base['status'] != row['status']: raise ValueError('Baseline status differs')
            if base['status'] == 'ok':
                result = read_json(root/'cases'/f'{pos:03d}'/filename)
                if ranking_metrics([r['file_path'] for r in result['ranking']], result['ground_truth'], 10) != base['metrics']:
                    raise ValueError('Baseline metric differs')
            records[name].append(base)
        if record['status'] == 'ok':
            path = folder/record['result_file']
            if sha256_file(path) != record['result_sha256']: raise ValueError('Result changed')
            audit(source.load(sample), read_json(path))
            for name, key in (('A+B','AB'), ('A+B+C','ABC')):
                records[name].append({**record, 'metrics': record['metrics'][key]})
        else:
            if sample['position'] != 308 or not source.failures(sample): raise ValueError('Unexpected failure')
            for name in ('A+B','A+B+C'): records[name].append(record)
    pilot = {s['position'] for s in read_json(ROOT/'results/fusion_pilot/20261114_fixed45/selection.json')['samples']}
    groups = {'all500': set(range(1,501)), 'previous180': set(range(1,181)),
        'new320': set(range(181,501)), 'pilot45': pilot, 'remaining455': set(range(1,501))-pilot,
        'new_nonpilot275': set(range(181,501))-pilot}
    summary = {cohort: {m: aggregate_results([r for r in rows if r['position'] in positions], 10)
                        for m,rows in records.items()} for cohort,positions in groups.items()}
    write_json(folder/'samples.json', records)
    write_json(folder/'final_summary.json', summary)
    write_json(folder/'project_summary.json', {repo: {m: aggregate_results([r for r in rows if r['repository']==repo],10)
        for m,rows in records.items()} for repo in sorted({s['repository'] for s in source.samples})})
    with (folder/'comparison.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['cohort','method','successful','failed','MAP@10','MRR@10','Top-1','Top-5','Top-10','MeanRecall@10'])
        writer.writeheader()
        for cohort, methods in summary.items():
            for m,s in methods.items(): writer.writerow({'cohort':cohort,'method':m,'successful':s['successful_samples'],'failed':s['failed_samples'],**s['metrics']})
    write_json(folder/'verification.json', {'status':'passed','selected':500,'verified_reranked':499,
        'expected_failures':[308],'unexpected_failures':[], 'physical_gpus':[0,1],
        'checks':['frozen sources/model','paired identities','pool reconstruction','RRF order','mixed chunks and text','Max aggregation/ties','all metrics']})
    write_json(folder/'status.json', {'status':'completed_with_failures','updated_at':now()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, choices=(0,1))
    args = parser.parse_args(); folder = args.output.resolve()
    if args.gpu is not None:
        if os.environ.get('CUDA_VISIBLE_DEVICES') != str(args.gpu): raise ValueError('Wrong GPU visibility')
        worker(folder, args.gpu); return
    folder.mkdir(parents=True, exist_ok=True); (folder/'cases').mkdir(exist_ok=True)
    with (folder/'controller.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not (folder/'protocol.json').exists():
            source_config = read_json(ROOT/'configs/experiments/chunk_ablation500.json')
            source = FixedSources(source_config)
            files = source.metadata + [Path(__file__), ROOT/'configs/experiments/chunk_ablation500.json',
                ROOT/'configs/retrieval/reranker.json', ROOT/'src/bug_localization/chunk_ablation.py',
                ROOT/'src/bug_localization/reranking.py', ROOT/'src/bug_localization/bm25.py',
                ROOT/'src/bug_localization/batch.py', ROOT/'scripts/experiments/run_chunk_ablation500.py',
                ROOT/'scripts/experiments/run_mixed_recall500.py',
                ROOT/'results/recall_baselines/20261112_L0_S0_H0_500/samples.json',
                ROOT/'results/fusion_pilot/20261114_fixed45/selection.json']
            write_json(folder/'protocol.json', {'created_at':now(),'source_config':source_config,
                'reranker':read_json(ROOT/'configs/retrieval/reranker.json'),
                'fusion':{'method':'RRF','k':60,'weights':[1,1],'pool':'existing hypothesis40 + original padding50',
                    'tie_break':'path','reranker_tie_break':'RRF rank','evidence':'original top1 + hypothesis padding3'},
                'scope':'Development 500; includes pilot45; report nonpilot subsets separately; no parameter search',
                'files_sha256':{str(p.resolve()):sha256_file(p) for p in files}})
            write_json(folder/'selection.json', source.selection)
        verify_protocol(folder)
        write_json(folder/'status.json', {'status':'running','updated_at':now(),'physical_gpus':[0,1]})
        processes=[]; handles=[]
        try:
            for gpu in (0,1):
                handle=(folder/f'gpu{gpu}.log').open('a'); handles.append(handle)
                env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false'}
                processes.append(subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--output',str(folder),'--gpu',str(gpu)],env=env,stdout=handle,stderr=subprocess.STDOUT))
            while any(p.poll() is None for p in processes):
                if any(p.poll() not in (None,0) for p in processes): raise RuntimeError('GPU worker failed; inspect logs; checkpoints retained')
                time.sleep(5)
            if any(p.returncode for p in processes): raise RuntimeError('GPU worker failed')
            write_json(folder/'status.json', {'status':'auditing','updated_at':now()})
            finalize(folder)
        except BaseException as error:
            for p in processes:
                if p.poll() is None: p.terminate()
            for p in processes: p.wait()
            write_json(folder/'status.json', {'status':'failed','error':str(error),'updated_at':now()})
            raise
        finally:
            for h in handles: h.close()


if __name__ == '__main__': main()
