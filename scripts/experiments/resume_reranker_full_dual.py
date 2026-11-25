"""Continue frozen full reranking with disjoint GPU workers and one summary writer."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import run_reranker_full as old
s = old.source
ROOT = old.ROOT
METHODS = old.METHODS


def assignments(metas, completed):
    pending = [m for m in metas if any(m['position'] not in completed[k] for k in METHODS)]
    return {str(g): [m['position'] for m in pending[g::2]] for g in (0, 1)}


def verify_protocol(folder):
    protocol = s.read(folder / 'protocol.json')
    for path, checksum in protocol['files_sha256'].items():
        if old.sha256_file(Path(path)) != checksum:
            raise ValueError('Original frozen input changed: ' + path)
    extra = folder / 'dual_resume/protocol.json'
    if extra.exists():
        continuation = s.read(extra)
        if continuation['runner_sha256'] != old.sha256_file(Path(__file__)):
            raise ValueError('Continuation implementation changed')
        for path, checksum in continuation['original_files_sha256'].items():
            if old.sha256_file(folder / path) != checksum:
                raise ValueError('Original protocol changed: ' + path)
    return protocol


def publish(folder, metas, development, records):
    for method in METHODS:
        rows = sorted(records[method].values(), key=lambda r: r['position'])
        summary = old.aggregate_results(rows, 10, len(metas))
        summary['cohorts'] = {name: old.aggregate_results(
            [r for r in rows if (r['dataset_row'] in development) == is_dev], 10, count)
            for name, is_dev, count in [('development500', True, len(development)),
                ('remaining1794', False, len(metas)-len(development))]}
        s.save(folder / method / 'samples.json', rows)
        s.save(folder / method / 'summary.json', summary)
        s.state(folder, method, summary['status'], selected=len(metas), processed=len(rows),
                failures=summary['failed_samples'], execution='dual_gpu_continuation')


def load_new(folder, metas, records):
    for method in METHODS:
        for meta in metas:
            if meta['position'] not in records[method]:
                row = s.resumed(folder, method, meta)
                if row is not None:
                    records[method][meta['position']] = row


def worker(folder, gpu):
    protocol = verify_protocol(folder)
    plan = s.read(folder / 'dual_resume/protocol.json')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(gpu):
        raise ValueError('Worker GPU visibility mismatch')
    allowed = set(plan['assignments'][str(gpu)])
    metas = [m for m in s.read(folder / 'selection.json')['samples'] if m['position'] in allowed]
    src = ROOT / protocol['config']['source_run']
    instruction = s.read(src / 'protocol.json')['semantic']['query_instruction']
    model = None
    stage = 'dual_gpu' + str(gpu)
    for i, meta in enumerate(metas):
        needed = [m for m in METHODS if s.resumed(folder, m, meta) is None]
        if not needed:
            continue
        s.state(folder, stage, 'running', position=meta['position'], processed=i, selected=len(metas))
        a = s.resumed(src, 'S0', meta); b = s.resumed(src, 'H0', meta)
        if a is None or b is None:
            raise ValueError('Full retrieval checkpoint missing')
        for method in list(needed):
            dep, dep_method = (a, 'S0') if a['status'] != 'ok' else (b, 'H0')
            if a['status'] != 'ok' or (method != 'R0' and b['status'] != 'ok'):
                p = src / dep_method / 'cases' / f"{meta['position']:04d}" / 'record.json'
                row = {**meta, 'status': 'failed', 'error_type': 'DependencyFailure',
                       'error': dep_method + ': ' + dep.get('error', 'Source retrieval failed'),
                       'source_hashes': {str(p.resolve()): old.sha256_file(p)}}
                s.save(folder / method / 'cases' / f"{meta['position']:04d}" / 'record.json', row)
                needed.remove(method)
        if not needed:
            continue
        case = old.load_case(src, a, b, instruction)
        base = {'query': case['query'], 'ground_truth': case['gold'], 'source_hashes': case['source_hashes']}
        for method in needed:
            started = time.monotonic()
            if method == 'F':
                head = old.fuse(case)[:len(case['candidates'])]
                saved = {**base, 'candidates': case['candidates'], 'ranking': head}
            else:
                if model is None:
                    if s.gpu_memory().get(gpu, 0) < protocol['config']['minimum_free_mib']:
                        raise ValueError('Insufficient free GPU memory')
                    model = old.LocalReranker(ROOT, protocol['reranker'])
                    if model.identity != protocol['reference_model']:
                        raise ValueError('Model identity differs from original run')
                    s.pin(folder / 'dual_resume' / f'model_gpu{gpu}.json', {'physical_gpu': gpu, 'identity': model.identity})
                if method == 'R0':
                    head = case['original']['ranking']
                    chosen = old.select_chunks(head, case['chunks'], case['original']['chunk_scores'], 50, 3)
                else:
                    head, chosen = old.select_evidence(case['candidates'], old.fuse(case), case['chunks'],
                        case['original']['chunk_scores'], case['hypothesis']['chunk_scores'], 'mixed')
                scored = model.score(case['query'], [c['embedding_text'] for c in chosen])
                ranking = old.rerank(head, chosen, scored)
                saved = {**base, 'ranking': ranking,
                         'scored_chunks': [{**c, **v} for c, v in zip(chosen, scored)]}
                if method == 'R0':
                    old.audit_original(case, saved)
                else:
                    saved.update(candidates=case['candidates'], ab_ranking=[r['file_path'] for r in head],
                        metrics={'AB': old.ranking_metrics([r['file_path'] for r in head], case['gold'], 10),
                                 'ABC': old.ranking_metrics([r['file_path'] for r in ranking], case['gold'], 10)})
                    old.audit(case, saved)
                    saved['module_metrics'] = saved.pop('metrics')
            row = s.record(folder, method, meta, saved, case['inventory'], case['gold'], case['missing'],
                           source_hashes=case['source_hashes'], physical_gpu=gpu,
                           execution='dual_gpu_continuation', elapsed_seconds=time.monotonic()-started)
            s.save(folder / method / 'cases' / f"{meta['position']:04d}" / 'record.json', row)
            print(f'{s.now()} GPU{gpu} {method} position={meta["position"]} ok', flush=True)
        del case
    s.state(folder, stage, 'completed', processed=len(metas), selected=len(metas))


def supervise(folder):
    with (folder / 'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol = verify_protocol(folder)
        selection = s.read(folder / 'selection.json');metas = selection['samples']
        development = set(selection['development_rows'])
        s.state(folder, 'worker', 'validating_resume', gpu_ids=[0, 1])
        records = {m: {} for m in METHODS}
        load_new(folder, metas, records)
        hashes = {}
        for rows in records.values():
            for row in rows.values():
                for path, checksum in row.get('source_hashes', {}).items():
                    if path in hashes and hashes[path] != checksum:
                        raise ValueError('Conflicting saved source hashes')
                    hashes[path] = checksum
        for path, checksum in hashes.items():
            if old.sha256_file(Path(path)) != checksum:
                raise ValueError('Saved source changed: ' + path)
        src = ROOT / protocol['config']['source_run']
        if s.read(src / 'verification.json')['status'] != 'passed':
            raise ValueError('Require completed verified full retrieval')
        for route in ('S0','H0'):
            if s.read(src / 'status' / (route+'.json'))['status'] != 'completed':
                raise ValueError('Retrieval incomplete')
        extra = folder / 'dual_resume/protocol.json'
        if not extra.exists():
            plan = {'created_at': s.now(), 'runner_sha256': old.sha256_file(Path(__file__)),
                'original_files_sha256': {p: old.sha256_file(folder / p) for p in ['protocol.json','selection.json']},
                'assignments': assignments(metas, records), 'initial_counts': {m: len(r) for m,r in records.items()},
                'policy': 'Each report belongs to one GPU; same logical cuda:0, model, batch16, query/evidence/scoring; completed cases immutable'}
            s.pin(extra, plan)
            s.save(folder / 'dual_resume/prior_records.json', records)
        publish(folder, metas, development, records)
        processes = [];logs = []
        try:
            for gpu in (0,1):
                log = (folder / 'dual_resume' / f'gpu{gpu}.log').open('a');logs.append(log)
                processes.append(subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()),
                    '--output', str(folder), '--worker', str(gpu)], cwd=ROOT, start_new_session=True,
                    env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'OMP_NUM_THREADS': '4',
                         'TOKENIZERS_PARALLELISM': 'false'}, stdout=log, stderr=subprocess.STDOUT))
            while True:
                load_new(folder, metas, records)
                publish(folder, metas, development, records)
                codes = [p.poll() for p in processes]
                if any(c not in (None,0) for c in codes):
                    raise RuntimeError('GPU worker failed: '+str(codes))
                s.state(folder, 'worker', 'running', gpu_ids=[0,1], worker_pids=[p.pid for p in processes],
                        processed={m: len(v) for m,v in records.items()})
                if all(c == 0 for c in codes):break
                time.sleep(10)
            load_new(folder, metas, records)
            if any(len(v) != len(metas) for v in records.values()):
                raise ValueError('Workers exited with missing results')
            publish(folder, metas, development, records)
            prior = s.read(folder / 'dual_resume/prior_records.json')
            for method, rows in prior.items():
                for position, row in rows.items():
                    if records[method][int(position)] != row:
                        raise ValueError('Previously completed record changed')
            s.save(folder / 'dual_resume/verification.json', {'status':'passed','samples':len(metas),
                'checks':['original protocol unchanged','all prior records retained unchanged',
                          '2294 unique records per method','new result checksums and metrics verified',
                          'per-case independent evidence and Max/tie audits','both GPU model identities match']})
            s.state(folder, 'worker', 'completed', gpu_ids=[0,1], processed={m:len(v) for m,v in records.items()})
        finally:
            for p in processes:
                if p.poll() is None:
                    os.killpg(p.pid, signal.SIGTERM)
                    try:p.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(p.pid, signal.SIGKILL);p.wait()
            for log in logs:log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', type=int, choices=(0,1))
    args=parser.parse_args();folder=args.output.resolve()
    def interrupted(signum, frame):raise InterruptedError('Interrupted')
    signal.signal(signal.SIGTERM, interrupted);signal.signal(signal.SIGINT, interrupted)
    try:
        if args.worker is None:supervise(folder)
        else:worker(folder,args.worker)
    except BaseException as error:
        s.state(folder,'worker' if args.worker is None else 'dual_gpu'+str(args.worker),'failed',
                error_type=type(error).__name__,error=str(error))
        traceback.print_exc();raise

if __name__ == '__main__':main()
