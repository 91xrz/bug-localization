"""Frozen 500-report C/D mixed recall experiment; one tmux worker per GPU."""
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'src'))
from bug_localization.batch import aggregate_results, write_json
from bug_localization.bm25 import ranking_metrics
from bug_localization.embeddings import digest
from bug_localization.mixed_recall import mixed_candidates, original_chunks
from bug_localization.reranking import LocalReranker, read_json, rerank
from bug_localization.snapshot import sha256_file

CONFIG = PROJECT / 'configs/experiments/mixed_recall_reranker500.json'
IDENTITY = ('dataset_row', 'repository', 'issue_id', 'before_fix_sha', 'query_sha256', 'labels_sha256')


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def same_sample(a, b):
    if any(a[k] != b[k] for k in IDENTITY):
        raise ValueError('Sample identity/order changed')


class Sources:
    def __init__(self, config):
        self.config = config
        self.selection = read_json(PROJECT / config['selection'])
        self.samples = self.selection['samples']
        if len(self.samples) != 500 or [r['position'] for r in self.samples] != list(range(1, 501)):
            raise ValueError('Require the frozen complete 500-report selection')
        control = PROJECT / config['control_run']
        self.roots = [(PROJECT / config['old_semantic'], PROJECT / config['old_hypothesis']),
                      (control / 'new320/semantic', control / 'new320/hypothesis')]
        self.groups, self.metadata = [], [PROJECT / config['selection'], control / 'paired_samples.json',
                                         control / 'final_summary.json', PROJECT / config['reference_reranker']]
        for index, (original, hypothesis) in enumerate(self.roots):
            selections = [read_json(p / 'selection.json')['samples'] for p in (original, hypothesis)]
            records = [read_json(original / 'samples.json')['q0'],
                       read_json(hypothesis / 'samples.json')['direct_hypothesis']]
            expected = self.samples[:180] if index == 0 else self.samples[180:]
            for rows in selections + records:
                if len(rows) != len(expected):
                    raise ValueError('Incomplete source cohort')
                for local, (a, b) in enumerate(zip(rows, expected), 1):
                    same_sample(a, b)
                    if a['position'] != local:
                        raise ValueError('Source positions are not consecutive')
            for root in (original, hypothesis):
                self.metadata.extend(root / name for name in ('selection.json', 'samples.json'))
            self.metadata.append(original / 'effective_config.json')
            instruction = read_json(original / 'effective_config.json')['semantic']['query_instruction']
            self.groups.append((records, instruction))
        self.controls = read_json(control / 'paired_samples.json')['cohorts']['all500']
        for rows in self.controls.values():
            if len(rows) != 500:
                raise ValueError('Incomplete historical control')
            for a, b in zip(rows, self.samples):
                same_sample(a, b)

    def locate(self, sample):
        global_pos = sample['position']
        group, pos = (0, global_pos) if global_pos <= 180 else (1, global_pos - 180)
        roots = self.roots[group]
        records, instruction = self.groups[group]
        return roots, pos, [rows[pos - 1] for rows in records], instruction

    def failures(self, sample):
        return [r for r in self.locate(sample)[2] if r['status'] != 'ok']

    def load(self, sample):
        (original, hypothesis), pos, records, instruction = self.locate(sample)
        if self.failures(sample):
            raise ValueError('Cannot load a failed source')
        oc, hc = original / 'cases' / f'{pos:03d}', hypothesis / 'cases' / f'{pos:03d}'
        paths = [oc / 'q0_retrieval.json', oc / 'q0_result.json', hc / 'direct_retrieval.json',
                 hc / 'direct_result.json', oc / 'chunks.json', oc / 'candidates.json']
        q0, qr, hyp, hr, chunks, inventory = [read_json(p) for p in paths]
        inventories = {r['file_path'] for r in inventory['files']}
        for retrieval, result, record in ((q0, qr, records[0]), (hyp, hr, records[1])):
            if retrieval['ranking'] != result['ranking'] or result['metrics'] != record['metrics']:
                raise ValueError('Source ranking/metric disagreement')
            if digest(retrieval['chunk_scores']) != retrieval['chunk_scores_sha256']:
                raise ValueError('Source chunk score checksum mismatch')
            if {r['file_path'] for r in retrieval['ranking']} != inventories:
                raise ValueError('Source file inventory mismatch')
        if qr['ground_truth'] != hr['ground_truth']:
            raise ValueError('Source ground truth mismatch')
        wrapper = 'Instruct: ' + instruction + '\nQuery: '
        if not q0['query_info']['embedding_text'].startswith(wrapper):
            raise ValueError('Unknown original-query wrapper')
        query = q0['query_info']['embedding_text'][len(wrapper):]
        if not query or not q0['query'].startswith(query):
            raise ValueError('Reranker query differs from original report')
        return {'original': q0, 'hypothesis': hyp, 'query': query, 'chunks': chunks,
                'gold': qr['ground_truth'], 'inventory': inventories, 'paths': paths}


def initialize(folder):
    config = read_json(CONFIG)
    source = Sources(config)
    model_config = read_json(PROJECT / config['reranker_config'])
    if model_config['top_n_files'] != 50 or model_config['chunks_per_file'] != 3:
        raise ValueError('C/D require 50 files and three chunks per file')
    reference = read_json(PROJECT / config['reference_reranker'])
    if any(reference['reranker'][k] != v for k, v in model_config.items()):
        raise ValueError('Model settings differ from historical controls')
    files = source.metadata + [CONFIG, PROJECT / config['reranker_config'], Path(__file__),
        PROJECT / 'src/bug_localization/mixed_recall.py', PROJECT / 'src/bug_localization/reranking.py',
        PROJECT / 'src/bug_localization/bm25.py', PROJECT / 'src/bug_localization/batch.py',
        PROJECT / 'src/bug_localization/embeddings.py', PROJECT / 'src/bug_localization/snapshot.py']
    protocol = {'config': config, 'reranker': model_config,
                'files_sha256': {str(p.resolve()): sha256_file(p) for p in files}}
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'experiment.json'
    if path.exists():
        saved = read_json(path)
        if any(saved[k] != v for k, v in protocol.items()):
            raise ValueError('Frozen protocol changed; use a new run folder')
        return saved
    # Preflight both cohorts and both candidate directions without loading any model.
    for sample in (source.samples[0], source.samples[-1]):
        case = source.load(sample)
        for primary in config['methods'].values():
            candidates = mixed_candidates(case['original']['ranking'], case['hypothesis']['ranking'], primary)
            original_chunks(candidates, case['original']['ranking'], case['chunks'], case['original']['chunk_scores'])
    write_json(folder / 'selection.json', source.selection)
    protocol.update(created_at=now(), source_failures=[{'position': s['position'], 'records': source.failures(s)}
                    for s in source.samples if source.failures(s)])
    write_json(path, protocol)
    write_json(folder / 'status.json', {'status': 'prepared', 'updated_at': now(), 'selected_per_method': 500})
    return protocol


def verify_frozen(folder):
    protocol = read_json(folder / 'experiment.json')
    for path, checksum in protocol['files_sha256'].items():
        if sha256_file(Path(path)) != checksum:
            raise ValueError('Frozen input changed: ' + path)
    if read_json(folder / 'selection.json') != read_json(PROJECT / protocol['config']['selection']):
        raise ValueError('Frozen selection changed')
    return protocol


def pool_metrics(candidates, gold):
    gold = set(gold)
    pool = {r['file_path'] for r in candidates}
    primary = {r['file_path'] for r in candidates if r['is_primary']}
    return {'files': len(pool), 'gold_hits': len(gold & pool),
            'recall': len(gold & pool) / len(gold) if gold else 0,
            'hit': int(bool(gold & pool)), 'primary_hit': int(bool(gold & primary)),
            'supplement_rescued': int(not (gold & primary) and bool(gold & pool))}


def checkpoint(target, records):
    write_json(target / 'samples.json', records)
    write_json(target / 'progress.json', {'processed': len(records), 'selected': 500,
               'failed': sum(r['status'] != 'ok' for r in records), 'updated_at': now()})


def worker(folder, method):
    protocol = verify_frozen(folder)
    config, model_config = protocol['config'], protocol['reranker']
    if os.environ.get('CUDA_VISIBLE_DEVICES') != config['physical_gpus'][method]:
        raise ValueError('Worker launched on unexpected physical GPU')
    source, target = Sources(config), folder / method
    target.mkdir(exist_ok=True)
    records = read_json(target / 'samples.json') if (target / 'samples.json').exists() else []
    if len(records) > 500:
        raise ValueError('Too many checkpoint records')
    for record, sample in zip(records, source.samples):
        same_sample(record, sample)
        if record['position'] != sample['position']:
            raise ValueError('Checkpoint is not a selection prefix')
        if record['status'] == 'ok' and not (target / record['result_file']).exists():
            raise ValueError('Checkpoint result file missing')
    model = LocalReranker(PROJECT, model_config)
    reference = read_json(PROJECT / config['reference_reranker'])['model']
    if model.identity != reference:
        raise ValueError('Reranker model/runtime differs from historical A/B controls')
    write_json(target / 'effective_config.json', {'method': method, 'primary': config['methods'][method],
        'reranker': model_config, 'model': model.identity, 'physical_gpu': os.environ['CUDA_VISIBLE_DEVICES'],
        'chunk_policy': config['chunk_policy'], 'query_source': 'original_report', 'tie_breaker': config['tie_breaker']})
    checkpoint(target, records)
    for sample in source.samples[len(records):]:
        started = time.monotonic()
        failure = source.failures(sample)
        if failure:
            record = {**sample, 'status': 'failed', 'error_type': 'SourceDependencyFailure',
                      'error': '; '.join(r.get('error', 'Unknown source failure') for r in failure)}
        else:
            try:
                case = source.load(sample)
                candidates = mixed_candidates(case['original']['ranking'], case['hypothesis']['ranking'],
                    config['methods'][method], config['primary_count'], config['total_candidates'])
                head, chosen = original_chunks(candidates, case['original']['ranking'], case['chunks'],
                                               case['original']['chunk_scores'], model_config['chunks_per_file'])
                scored = model.score(case['query'], [c['embedding_text'] for c in chosen])
                ranking = rerank(head, chosen, scored)
                metrics = ranking_metrics([r['file_path'] for r in ranking], case['gold'], 10)
                path = target / 'cases' / f"{sample['position']:03d}.json"
                path.parent.mkdir(exist_ok=True)
                pool = pool_metrics(candidates, case['gold'])
                write_json(path, {**sample, 'method': method, 'query': case['query'],
                    'reranker_query_source': 'original_report', 'chunk_policy': config['chunk_policy'],
                    'query_truncated_by_semantic_encoder': case['original']['query_info']['truncated'],
                    'ground_truth': case['gold'], 'candidates': candidates, 'pool_metrics': pool,
                    'ranking': ranking, 'metrics': metrics,
                    'scored_chunks': [{**{k: c[k] for k in ('file_path', 'chunk_id', 'embedding_text', 'semantic_similarity')}, **s}
                                      for c, s in zip(chosen, scored)],
                    'source_files_sha256': {str(p.resolve()): sha256_file(p) for p in case['paths']}})
                record = {**sample, 'status': 'ok', 'metrics': metrics, 'pool_metrics': pool,
                          'missing_gold': sorted(set(case['gold']) - case['inventory']),
                          'scored_pairs': len(scored), 'result_file': str(path.relative_to(target))}
            except Exception as error:
                record = {**sample, 'status': 'failed', 'error_type': type(error).__name__,
                          'error': str(error), 'traceback': traceback.format_exc()}
        record['elapsed_seconds'] = time.monotonic() - started
        records.append(record)
        checkpoint(target, records)
        if len(records) % 10 == 0 or record['status'] == 'failed':
            print(f"{method}: {len(records)}/500, failures={sum(r['status'] != 'ok' for r in records)}", flush=True)
    write_json(target / 'summary.json', aggregate_results(records, 10))


def audit(folder, method, source):
    """Reconstruct candidates and per-file maxima without using the selection/rerank helpers."""
    target, config = folder / method, source.config
    records = read_json(target / 'samples.json')
    if len(records) != 500:
        raise ValueError('Cannot audit an incomplete experiment')
    verified, expected_failures, unexpected_failures = 0, [], []
    for record, sample in zip(records, source.samples):
        same_sample(record, sample)
        if record['position'] != sample['position']:
            raise ValueError('Result positions changed')
        dependencies = source.failures(sample)
        if record['status'] != 'ok':
            if record.get('metrics'):
                raise ValueError('Failed record carries nonzero metrics')
            (expected_failures if dependencies else unexpected_failures).append(sample['position'])
            continue
        if dependencies:
            raise ValueError('Unexpected fallback for failed source')
        case, saved = source.load(sample), read_json(target / record['result_file'])
        same_sample(saved, sample)
        expected_hashes = {str(p.resolve()): sha256_file(p) for p in case['paths']}
        if saved['source_files_sha256'] != expected_hashes:
            raise ValueError('Per-case source files changed')
        if saved['query'] != case['query'] or saved['reranker_query_source'] != 'original_report':
            raise ValueError('Reranker must only see the original report')
        if saved['ground_truth'] != case['gold']:
            raise ValueError('Ground truth changed')
        primary = config['methods'][method]
        secondary = 'original' if primary == 'hypothesis' else 'hypothesis'
        expected = [r['file_path'] for r in case[primary]['ranking'][:40]]
        expected += [r['file_path'] for r in case[secondary]['ranking'] if r['file_path'] not in set(expected)][:max(0, 50 - len(expected))]
        actual = [r['file_path'] for r in saved['candidates']]
        if actual != expected or len(actual) != min(50, len(case['inventory'])):
            raise ValueError('Candidate pool does not implement primary40 plus distinct secondary padding')
        ranks = {route: {r['file_path']: r['rank'] for r in case[route]['ranking']} for route in ('original', 'hypothesis')}
        for i, item in enumerate(saved['candidates']):
            is_primary = i < min(40, len(expected))
            if item != {'file_path': expected[i], 'candidate_rank': i + 1, 'introduced_by': primary if is_primary else secondary,
                        'is_primary': is_primary, 'original_rank': ranks['original'][expected[i]],
                        'hypothesis_rank': ranks['hypothesis'][expected[i]]}:
                raise ValueError('Candidate provenance mismatch')
        grouped = defaultdict(list)
        lookup = {c['chunk_id']: c for c in case['chunks']}
        for row in case['original']['chunk_scores']:
            grouped[row['file_path']].append(row)
        ordered_files = sorted(expected, key=ranks['original'].__getitem__)
        wanted = [r for path in ordered_files for r in sorted(grouped[path], key=lambda x: (-x['cosine_similarity'], x['chunk_id']))[:3]]
        scores = saved['scored_chunks']
        if [c['chunk_id'] for c in scores] != [c['chunk_id'] for c in wanted]:
            raise ValueError('Code chunks do not match original-report semantic Top3')
        best = {}
        for row, selected in zip(scores, wanted):
            if row['file_path'] != selected['file_path'] or row['embedding_text'] != lookup[row['chunk_id']]['embedding_text'] or row['semantic_similarity'] != selected['cosine_similarity']:
                raise ValueError('Scored chunk input differs from source')
            if not (-float('inf') < row['score'] < float('inf')):
                raise ValueError('Invalid model score')
            if row['file_path'] not in best or row['score'] > best[row['file_path']]['score']:
                best[row['file_path']] = row
        ranked_paths = sorted(expected, key=lambda p: (-best[p]['score'], ranks['original'][p]))
        if [r['file_path'] for r in saved['ranking']] != ranked_paths:
            raise ValueError('File Max aggregation or original-rank tie order changed')
        for i, row in enumerate(saved['ranking'], 1):
            if row['rank'] != i or row['score'] != best[row['file_path']]['score'] or row['best_chunk_id'] != best[row['file_path']]['chunk_id']:
                raise ValueError('File score metadata mismatch')
        metrics = ranking_metrics(ranked_paths, case['gold'], 10)
        if metrics != saved['metrics'] or metrics != record['metrics']:
            raise ValueError('Saved ranking metrics mismatch')
        if pool_metrics(saved['candidates'], case['gold']) != saved['pool_metrics'] or saved['pool_metrics'] != record['pool_metrics']:
            raise ValueError('Candidate pool recall mismatch')
        if record['missing_gold'] != sorted(set(case['gold']) - case['inventory']):
            raise ValueError('Inventory coverage mismatch')
        verified += 1
    result = {'verified_successes': verified, 'expected_source_failures': expected_failures,
              'unexpected_failures': unexpected_failures, 'full_denominator': 500,
              'original_query_verified': True, 'candidate_and_chunk_selection_verified': True,
              'max_aggregation_and_metrics_verified': True, 'completed_at': now()}
    write_json(target / 'verification.json', result)
    return result


def paired_difference(left, right):
    deltas = [a.get('metrics', {}).get('average_precision_at_k', 0) - b.get('metrics', {}).get('average_precision_at_k', 0)
              for a, b in zip(left, right)]
    return {'wins': sum(d > 1e-12 for d in deltas), 'ties': sum(abs(d) <= 1e-12 for d in deltas),
            'losses': sum(d < -1e-12 for d in deltas), 'delta_MAP@10': sum(deltas) / len(deltas)}


def finalize(folder):
    protocol = verify_frozen(folder)
    source = Sources(protocol['config'])
    audits = {method: audit(folder, method, source) for method in ('C', 'D')}
    records = {'A': source.controls['original_reranked'], 'B': source.controls['hypothesis_reranked'],
               **{m: read_json(folder / m / 'samples.json') for m in ('C', 'D')}}
    cohorts = {name: {m: rows[start:stop] for m, rows in records.items()} for name, start, stop in
               [('all500', 0, 500), ('previous180', 0, 180), ('new320', 180, 500)]}
    summaries, projects = {}, {}
    for name, group in cohorts.items():
        summaries[name] = {'selected_samples': len(group['A']),
            'methods': {m: aggregate_results(rows, 10) for m, rows in group.items()},
            'paired_AP': {f'{m}_vs_{b}': paired_difference(group[m], group[b])
                          for m, b in [('C', 'D'), ('C', 'A'), ('C', 'B'), ('D', 'A'), ('D', 'B')]},
            'candidate_pool': {m: {metric: sum(r.get('pool_metrics', {}).get(metric, 0) for r in group[m]) / len(group[m])
                for metric in ('recall', 'hit', 'primary_hit', 'supplement_rescued')} for m in ('C', 'D')}}
        projects[name] = {repo: {m: aggregate_results([r for r in rows if r['repository'] == repo], 10)
                               for m, rows in group.items()} for repo in sorted({r['repository'] for r in group['A']})}
    final = {'status': 'completed_with_failures' if any(r['status'] != 'ok' for m in ('C', 'D') for r in records[m]) else 'completed',
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
    return final


def supervise(folder):
    initialize(folder)
    processes, logs = {}, []
    try:
        config = verify_frozen(folder)['config']
        for method in ('C', 'D'):
            target = folder / method
            target.mkdir(exist_ok=True)
            log = (target / 'worker.log').open('a', encoding='utf-8')
            logs.append(log)
            env = {**os.environ, 'CUDA_VISIBLE_DEVICES': config['physical_gpus'][method],
                   'TOKENIZERS_PARALLELISM': 'false', 'OMP_NUM_THREADS': '4'}
            processes[method] = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--output', str(folder),
                '--stage', 'worker', '--method', method], cwd=PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT)
        while True:
            status = {m: {'pid': p.pid, 'physical_gpu': config['physical_gpus'][m], 'exit_code': p.poll(),
                         'progress': read_json(folder / m / 'progress.json') if (folder / m / 'progress.json').exists() else None}
                      for m, p in processes.items()}
            write_json(folder / 'status.json', {'status': 'running', 'updated_at': now(), 'workers': status})
            if all(p.poll() is not None for p in processes.values()):
                break
            time.sleep(20)
        if any(p.returncode != 0 for p in processes.values()):
            raise RuntimeError('Worker process failed; inspect worker.log and resume unchanged protocol')
        write_json(folder / 'status.json', {'status': 'auditing', 'updated_at': now(), 'workers': status})
        finalize(folder)
    except BaseException as error:
        write_json(folder / 'status.json', {'status': 'interrupted', 'updated_at': now(), 'error': str(error),
                   'traceback': traceback.format_exc()})
        raise
    finally:
        for p in processes.values():
            if p.poll() is None:
                p.terminate()
        for log in logs:
            log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--stage', choices=['prepare', 'worker', 'audit', 'run'], default='run')
    parser.add_argument('--method', choices=['C', 'D'])
    args = parser.parse_args()
    folder = (PROJECT / args.output).resolve()
    if args.stage == 'prepare':
        initialize(folder)
        print(folder)
    elif args.stage == 'worker':
        if not args.method:
            parser.error('--method required for worker')
        worker(folder, args.method)
    elif args.stage == 'audit':
        finalize(folder)
    else:
        supervise(folder)


if __name__ == '__main__':
    main()
