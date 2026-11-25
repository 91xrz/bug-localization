"""C2/C3 evidence ablation against frozen C1, with dual-GPU resumable execution."""
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

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bug_localization.chunk_ablation import select_evidence
from bug_localization.reranking import LocalReranker, read_json, rerank
from bug_localization.batch import aggregate_results, write_json
from bug_localization.bm25 import ranking_metrics
from bug_localization.snapshot import sha256_file
from run_mixed_recall500 import Sources, same_sample, now, checkpoint, paired_difference, verify_frozen, pool_metrics

CONFIG = PROJECT / 'configs/experiments/chunk_ablation500.json'
METHODS = ('C2', 'C3')


class FixedSources(Sources):
    def __init__(self, config):
        super().__init__(config)
        self.reference = PROJECT / config['candidate_reference']
        self.baseline = read_json(self.reference / 'C/samples.json')
        audit = read_json(self.reference / 'C/verification.json')
        if audit['verified_successes'] != 499 or audit['expected_source_failures'] != [308] or audit['unexpected_failures']:
            raise ValueError('C1 reference did not pass its completed audit')
        if len(self.baseline) != len(self.samples):
            raise ValueError('C1 reference is incomplete')
        for sample, record in zip(self.samples, self.baseline):
            same_sample(sample, record)
            if record['position'] != sample['position'] or (record['status'] != 'ok') != bool(self.failures(sample)):
                raise ValueError('C1 selection/failure protocol differs from source')
        self.metadata += [self.reference / name for name in ('experiment.json', 'selection.json', 'C/samples.json',
                          'C/verification.json', 'C/effective_config.json')]
        self.metadata += [self.reference / 'C' / r['result_file'] for r in self.baseline if r['status'] == 'ok']

    def load(self, sample):
        case = super().load(sample)
        path = self.reference / 'C' / self.baseline[sample['position'] - 1]['result_file']
        baseline = read_json(path)
        same_sample(sample, baseline)
        if baseline['query'] != case['query'] or baseline['ground_truth'] != case['gold']:
            raise ValueError('C1 query or labels differ from source')
        hashes = {str(p.resolve()): sha256_file(p) for p in case['paths']}
        if baseline['source_files_sha256'] != hashes:
            raise ValueError('Source files changed since C1 was evaluated')
        case.update(baseline=baseline, candidates=baseline['candidates'], source_hashes=hashes,
                    reference_file=str(path.resolve()), reference_sha256=sha256_file(path))
        return case


def initialize(folder):
    config = read_json(CONFIG)
    previous = verify_frozen(PROJECT / config['candidate_reference'])
    source, model_config = FixedSources(config), read_json(PROJECT / config['reranker_config'])
    if model_config != previous['reranker'] or model_config['chunks_per_file'] != 3 or model_config['top_n_files'] != 50:
        raise ValueError('Reranker settings must match C1 exactly')
    files = source.metadata + [CONFIG, PROJECT / config['reranker_config'], Path(__file__),
                               PROJECT / 'src/bug_localization/chunk_ablation.py']
    hashes = {**previous['files_sha256'], **{str(p.resolve()): sha256_file(p) for p in files}}
    protocol = {'config': config, 'reranker': model_config, 'files_sha256': hashes}
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'experiment.json').exists():
        saved = verify_frozen(folder)
        if any(saved[k] != v for k, v in protocol.items()):
            raise ValueError('Protocol changed; use a new output directory')
        return saved
    for sample in (source.samples[0], source.samples[-1]):
        case = source.load(sample)
        for policy in config['methods'].values():
            select_case(case, policy)
    protocol.update(created_at=now(), expected_source_failures=[s['position'] for s in source.samples if source.failures(s)])
    write_json(folder / 'selection.json', source.selection)
    write_json(folder / 'experiment.json', protocol)
    write_json(folder / 'status.json', {'status': 'prepared', 'updated_at': now()})
    return protocol


def select_case(case, policy):
    return select_evidence(case['candidates'], case['original']['ranking'], case['chunks'],
                           case['original']['chunk_scores'], case['hypothesis']['chunk_scores'], policy)


def worker(folder, method):
    protocol = verify_frozen(folder)
    config, settings = protocol['config'], protocol['reranker']
    if os.environ.get('CUDA_VISIBLE_DEVICES') != config['physical_gpus'][method]:
        raise ValueError('Unexpected physical GPU')
    source, target = FixedSources(config), folder / method
    target.mkdir(exist_ok=True)
    records = read_json(target / 'samples.json') if (target / 'samples.json').exists() else []
    if len(records) > len(source.samples):
        raise ValueError('Checkpoint exceeds selection')
    for record, sample in zip(records, source.samples):
        same_sample(record, sample)
        if record['position'] != sample['position']:
            raise ValueError('Checkpoint is not a consecutive prefix')
        if record['status'] == 'ok' and sha256_file(target / record['result_file']) != record['result_sha256']:
            raise ValueError('Checkpoint result changed')
    model = LocalReranker(PROJECT, settings)
    if model.identity != read_json(source.reference / 'C/effective_config.json')['model']:
        raise ValueError('Reranker model/runtime differs from C1')
    write_json(target / 'effective_config.json', {'method': method, 'chunk_policy': config['methods'][method],
        'reranker': settings, 'model': model.identity, 'physical_gpu': os.environ['CUDA_VISIBLE_DEVICES'],
        'query_source': 'original_report', 'tie_breaker': config['tie_breaker'], 'candidate_reference': str(source.reference.resolve())})
    checkpoint(target, records)
    for sample in source.samples[len(records):]:
        started = time.monotonic()
        failures = source.failures(sample)
        if failures:
            record = {**sample, 'status': 'failed', 'error_type': 'SourceDependencyFailure',
                      'error': '; '.join(r.get('error', 'Unknown failure') for r in failures)}
        else:
            try:
                case = source.load(sample)
                head, chosen = select_case(case, config['methods'][method])
                scored = model.score(case['query'], [r['embedding_text'] for r in chosen])
                ranking = rerank(head, chosen, scored)
                metrics = ranking_metrics([r['file_path'] for r in ranking], case['gold'], 10)
                path = target / 'cases' / f"{sample['position']:03d}.json"
                path.parent.mkdir(exist_ok=True)
                pool = pool_metrics(case['candidates'], case['gold'])
                write_json(path, {**sample, 'method': method, 'chunk_policy': config['methods'][method],
                    'query': case['query'], 'reranker_query_source': 'original_report',
                    'query_truncated_by_semantic_encoder': case['original']['query_info']['truncated'],
                    'ground_truth': case['gold'], 'candidates': case['candidates'], 'pool_metrics': pool,
                    'metrics': metrics, 'ranking': ranking,
                    'scored_chunks': [{**{k: r[k] for k in ('file_path', 'chunk_id', 'embedding_text', 'semantic_similarity',
                                         'selected_by', 'original_similarity', 'hypothesis_similarity')}, **s}
                                      for r, s in zip(chosen, scored)],
                    'source_files_sha256': case['source_hashes'], 'reference_file': case['reference_file'],
                    'reference_sha256': case['reference_sha256']})
                record = {**sample, 'status': 'ok', 'metrics': metrics, 'pool_metrics': pool,
                    'missing_gold': sorted(set(case['gold']) - case['inventory']), 'scored_pairs': len(scored),
                    'result_file': str(path.relative_to(target)), 'result_sha256': sha256_file(path)}
            except Exception as error:
                record = {**sample, 'status': 'failed', 'error_type': type(error).__name__,
                          'error': str(error), 'traceback': traceback.format_exc()}
        record['elapsed_seconds'] = time.monotonic() - started
        records.append(record)
        checkpoint(target, records)
        if len(records) % 10 == 0 or record['status'] != 'ok':
            print(f"{method}: {len(records)}/500; failures={sum(r['status'] != 'ok' for r in records)}", flush=True)
    write_json(target / 'summary.json', aggregate_results(records, 10))


def audit_case(case, saved, record, policy):
    """Independent reconstruction of code-block choices and file maxima."""
    if saved['query'] != case['query'] or saved['reranker_query_source'] != 'original_report':
        raise ValueError('Reranker query must be the original report')
    if saved['candidates'] != case['candidates'] or saved['ground_truth'] != case['gold']:
        raise ValueError('Candidate pool or labels differ from C1')
    if saved['source_files_sha256'] != case['source_hashes'] or saved['reference_sha256'] != case['reference_sha256'] or saved['reference_file'] != case['reference_file']:
        raise ValueError('Source/reference hashes differ')
    if saved['chunk_policy'] != policy:
        raise ValueError('Chunk policy changed')
    grouped, similarities = {}, {}
    for route in ('original', 'hypothesis'):
        groups = defaultdict(list)
        similarities[route] = {}
        for r in case[route]['chunk_scores']:
            groups[r['file_path']].append(r)
            similarities[route][r['chunk_id']] = r['cosine_similarity']
        grouped[route] = {p: sorted(rows, key=lambda r: (-r['cosine_similarity'], r['chunk_id'])) for p, rows in groups.items()}
    ranks = {r['file_path']: r['rank'] for r in case['original']['ranking']}
    paths = sorted([r['file_path'] for r in case['candidates']], key=ranks.__getitem__)
    wanted, provenance = [], []
    for path in paths:
        if policy == 'hypothesis':
            ids = [r['chunk_id'] for r in grouped['hypothesis'][path][:3]]
            origins = ['hypothesis'] * len(ids)
        elif policy == 'mixed':
            ids = [grouped['original'][path][0]['chunk_id']]
            origins = ['original']
            for row in grouped['hypothesis'][path]:
                if len(ids) == 3:
                    break
                if row['chunk_id'] not in ids:
                    ids.append(row['chunk_id'])
                    origins.append('hypothesis')
        else:
            raise ValueError('Unknown audit policy')
        wanted.extend(ids)
        provenance.extend(origins)
    rows = saved['scored_chunks']
    if [r['chunk_id'] for r in rows] != wanted:
        raise ValueError('Selected chunks violate ablation policy or duplicate-padding rule')
    lookup, best = {r['chunk_id']: r for r in case['chunks']}, {}
    for row, origin in zip(rows, provenance):
        source = lookup[row['chunk_id']]
        if row['embedding_text'] != source['embedding_text'] or row['file_path'] != source['file_path']:
            raise ValueError('Scored code text differs from source')
        if row['selected_by'] != origin or row['semantic_similarity'] != similarities[origin][row['chunk_id']]:
            raise ValueError('Chunk provenance differs')
        if any(row[route + '_similarity'] != similarities[route][row['chunk_id']] for route in similarities):
            raise ValueError('Chunk similarity metadata differs')
        if not (-float('inf') < row['score'] < float('inf')):
            raise ValueError('Nonfinite reranker score')
        path = row['file_path']
        if path not in best or row['score'] > best[path]['score']:
            best[path] = row
    ordered = sorted(paths, key=lambda p: (-best[p]['score'], ranks[p]))
    if [r['file_path'] for r in saved['ranking']] != ordered:
        raise ValueError('File Max aggregation/tie order differs')
    for i, row in enumerate(saved['ranking'], 1):
        if row['rank'] != i or row['score'] != best[row['file_path']]['score'] or row['best_chunk_id'] != best[row['file_path']]['chunk_id']:
            raise ValueError('File score metadata differs')
    metrics = ranking_metrics(ordered, case['gold'], 10)
    pool = pool_metrics(case['candidates'], case['gold'])
    if saved['metrics'] != metrics or record['metrics'] != metrics:
        raise ValueError('Metrics differ from recomputation')
    if saved['pool_metrics'] != pool or record['pool_metrics'] != pool:
        raise ValueError('Candidate pool metrics differ')
    if record['missing_gold'] != sorted(set(case['gold']) - case['inventory']):
        raise ValueError('Missing gold metadata differs')


def audit(folder, method, source):
    target = folder / method
    records = read_json(target / 'samples.json')
    if len(records) != len(source.samples):
        raise ValueError('Audit requires complete selection')
    expected, unexpected, verified = [], [], 0
    for sample, record in zip(source.samples, records):
        same_sample(sample, record)
        if sample['position'] != record['position']:
            raise ValueError('Result position differs')
        dependencies = source.failures(sample)
        if record['status'] == 'failed':
            if record.get('metrics'):
                raise ValueError('Failure has nonzero metrics')
            (expected if dependencies else unexpected).append(sample['position'])
            continue
        if dependencies or record['status'] != 'ok':
            raise ValueError('Unexpected source fallback or status')
        path = target / record['result_file']
        if sha256_file(path) != record['result_sha256']:
            raise ValueError('Saved result hash changed')
        case, saved = source.load(sample), read_json(path)
        same_sample(saved, sample)
        audit_case(case, saved, record, source.config['methods'][method])
        verified += 1
    result = {'verified_successes': verified, 'expected_source_failures': expected, 'unexpected_failures': unexpected,
              'full_denominator': len(records), 'fixed_C1_candidates_and_original_query_verified': True,
              'chunk_policy_and_max_aggregation_verified': True, 'metrics_verified': True, 'completed_at': now()}
    write_json(target / 'verification.json', result)
    return result


def finalize(folder):
    protocol = verify_frozen(folder)
    source = FixedSources(protocol['config'])
    audits = {m: audit(folder, m, source) for m in METHODS}
    # Absolute baseline result paths make C1 reused artifacts unambiguous.
    baseline = [{**r, **({'result_file': str((source.reference / 'C' / r['result_file']).resolve())} if r['status'] == 'ok' else {})}
                for r in source.baseline]
    records = {'C1': baseline, **{m: read_json(folder / m / 'samples.json') for m in METHODS}}
    cohorts = {name: {m: rows[start:stop] for m, rows in records.items()} for name, start, stop in
               [('all500', 0, 500), ('previous180', 0, 180), ('new320', 180, 500)]}
    summaries, projects = {}, {}
    for name, group in cohorts.items():
        summaries[name] = {'selected_samples': len(group['C1']),
            'methods': {m: aggregate_results(rows, 10) for m, rows in group.items()},
            'paired_AP': {f'{a}_vs_{b}': paired_difference(group[a], group[b])
                          for a, b in [('C2', 'C1'), ('C3', 'C1'), ('C3', 'C2')]}}
        projects[name] = {repo: {m: aggregate_results([r for r in rows if r['repository'] == repo], 10)
                               for m, rows in group.items()} for repo in sorted({r['repository'] for r in group['C1']})}
    final = {'status': 'completed_with_failures' if any(r['status'] != 'ok' for rows in records.values() for r in rows) else 'completed',
             'completed_at': now(), 'cohorts': summaries, 'audits': audits, 'protocol': protocol['config']}
    write_json(folder / 'paired_samples.json', {'cohorts': cohorts})
    write_json(folder / 'project_summary.json', projects)
    write_json(folder / 'final_summary.json', final)
    with (folder / 'comparison.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['cohort', 'method', 'successful', 'failed', 'MAP@10', 'MRR@10', 'Top-1', 'Top-5', 'Top-10', 'MeanRecall@10'])
        writer.writeheader()
        for name, group in summaries.items():
            for method, stats in group['methods'].items():
                writer.writerow({'cohort': name, 'method': method, 'successful': stats['successful_samples'],
                                 'failed': stats['failed_samples'], **stats['metrics']})
    write_json(folder / 'status.json', {'status': final['status'], 'updated_at': now(), 'audits': audits})


def supervise(folder):
    initialize(folder)
    with (folder / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        processes, logs = {}, []
        try:
            config = verify_frozen(folder)['config']
            for method in METHODS:
                target = folder / method
                target.mkdir(exist_ok=True)
                log = (target / 'worker.log').open('a', encoding='utf-8')
                logs.append(log)
                env = {**os.environ, 'CUDA_VISIBLE_DEVICES': config['physical_gpus'][method],
                       'TOKENIZERS_PARALLELISM': 'false', 'OMP_NUM_THREADS': '4'}
                processes[method] = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--output', str(folder),
                    '--stage', 'worker', '--method', method], cwd=PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT)
            while True:
                workers = {m: {'pid': p.pid, 'physical_gpu': config['physical_gpus'][m], 'exit_code': p.poll(),
                    'progress': read_json(folder / m / 'progress.json') if (folder / m / 'progress.json').exists() else None}
                    for m, p in processes.items()}
                write_json(folder / 'status.json', {'status': 'running', 'updated_at': now(), 'workers': workers})
                if any(p.poll() not in (None, 0) for p in processes.values()):
                    raise RuntimeError('Worker failed; see worker.log')
                if all(p.poll() is not None for p in processes.values()):
                    break
                time.sleep(20)
            write_json(folder / 'status.json', {'status': 'auditing', 'updated_at': now(), 'workers': workers})
            finalize(folder)
        except BaseException as error:
            write_json(folder / 'status.json', {'status': 'interrupted', 'updated_at': now(), 'error': str(error),
                                              'traceback': traceback.format_exc()})
            raise
        finally:
            for p in processes.values():
                if p.poll() is None:
                    p.terminate()
                    try:
                        p.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        p.kill()
                        p.wait()
            for log in logs:
                log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=['prepare', 'worker', 'audit', 'run'], default='run')
    parser.add_argument('--method', choices=METHODS)
    args = parser.parse_args()
    folder = (PROJECT / args.output).resolve()
    if args.stage == 'prepare':
        initialize(folder)
        print(folder)
    elif args.stage == 'worker':
        if not args.method:
            parser.error('--method is required for worker')
        worker(folder, args.method)
    elif args.stage == 'audit':
        finalize(folder)
    else:
        supervise(folder)


if __name__ == '__main__':
    main()
