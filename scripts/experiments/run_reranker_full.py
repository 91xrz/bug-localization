"""Stream frozen full-corpus recall checkpoints into F, R0 and FULL on GPU0."""
import argparse
from collections import defaultdict
import fcntl
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import run_recall_full as source
from run_module_ablation500 import fuse, audit
from bug_localization.batch import aggregate_results
from bug_localization.bm25 import ranking_metrics
from bug_localization.chunk_ablation import select_evidence
from bug_localization.embeddings import digest
from bug_localization.reranking import LocalReranker, select_chunks, rerank
from bug_localization.snapshot import sha256_file

METHODS = ('F', 'R0', 'FULL')


def original_query(result, instruction):
    prefix = 'Instruct: ' + instruction + '\nQuery: '
    text = result['query_info']['embedding_text']
    if not text.startswith(prefix):
        raise ValueError('Unexpected original query instruction')
    query = text[len(prefix):]
    if not query.strip() or not result['query'].startswith(query):
        raise ValueError('Reranker query is not the original report prefix')
    return query


def candidate_pool(original, hypothesis):
    primary = [r['file_path'] for r in hypothesis['ranking'][:40]]
    paths = primary + [r['file_path'] for r in original['ranking']
                       if r['file_path'] not in set(primary)][:50-len(primary)]
    return [{'file_path': p} for p in paths]


def prepare(folder, config_path):
    config = source.read(config_path)
    src = ROOT / config['source_run']
    source.context(src)  # Verify the frozen dataset and recall implementation.
    selection = source.read(src / 'selection.json')
    if len(selection['samples']) != config['expected_samples']:
        raise ValueError('Full selection size changed')
    reranker = source.read(ROOT / config['reranker_config'])
    if (reranker['top_n_files'], reranker['chunks_per_file'], reranker['batch_size'],
        reranker['max_tokens'], reranker['evaluation_k'], reranker['device']) != (50, 3, 16, 1536, 10, 'cuda:0'):
        raise ValueError('Require the frozen 500-sample reranker parameters')
    reference = source.read(ROOT / config['model_reference'])
    if any(reference['reranker'].get(k) != v for k, v in reranker.items()):
        raise ValueError('Reranker config differs from reference')
    # Include local transitive implementation imports without modifying recall files.
    files = [Path(__file__), config_path, ROOT / config['reranker_config'],
             ROOT / config['model_reference'], src / 'protocol.json', src / 'paired_protocol.json',
             src / 'selection.json', *(ROOT / 'src/bug_localization' / (name + '.py')
                 for name in (*source.DEPENDENCIES, 'reranking', 'chunk_ablation', 'mixed_recall')),
             *(ROOT / 'scripts/experiments' / name for name in (
                 'run_recall_full.py', 'run_module_ablation500.py', 'run_chunk_ablation500.py',
                 'run_mixed_recall500.py', 'run_direct_hypothesis.py'))]
    protocol = {'config': config, 'reranker': reranker, 'reference_model': reference['model'],
        'methods': {'F': 'H top40 + S replenish50; equal RRF k60, path tie',
                    'R0': 'S top50; S top3 chunks/file; original query; Max; S rank tie',
                    'FULL': 'F pool/order; S top1 + H distinct supplement to3; original query; Max; F rank tie'},
        'query_policy': 'Exact original semantic input body after frozen instruction removal (512-token embedding budget)',
        'failure_policy': 'Dependency failures contribute zero; integrity/infrastructure errors stop the worker',
        'files_sha256': {str(p.resolve()): sha256_file(p) for p in files}}
    source.pin(folder / 'protocol.json', protocol)
    source.pin(folder / 'selection.json', selection)
    source.state(folder, 'prepare', 'completed', selected=len(selection['samples']))


def audit_original(case, saved):
    """Independently reconstruct original top chunks, file maxima, and ties."""
    grouped = defaultdict(list)
    for row in case['original']['chunk_scores']:
        grouped[row['file_path']].append(row)
    paths = [r['file_path'] for r in case['original']['ranking'][:50]]
    expected_ids = [r['chunk_id'] for p in paths for r in
                    sorted(grouped[p], key=lambda r: (-r['cosine_similarity'], r['chunk_id']))[:3]]
    scored = saved['scored_chunks']
    if [r['chunk_id'] for r in scored] != expected_ids:
        raise ValueError('R0 evidence mismatch')
    lookup = {r['chunk_id']: r for r in case['chunks']}
    best = {}
    for row in scored:
        if any(row[k] != lookup[row['chunk_id']][k] for k in ('file_path', 'embedding_text')):
            raise ValueError('R0 scored source text mismatch')
        if not -float('inf') < row['score'] < float('inf'):
            raise ValueError('Nonfinite score')
        p = row['file_path']
        if p not in best or row['score'] > best[p]['score']:
            best[p] = row
    ordered = sorted(paths, key=lambda p: (-best[p]['score'], paths.index(p)))
    ordered += [r['file_path'] for r in case['original']['ranking'][50:]]
    if [r['file_path'] for r in saved['ranking']] != ordered or saved['query'] != case['query']:
        raise ValueError('R0 aggregation/query mismatch')
    for i, row in enumerate(saved['ranking'][:len(paths)], 1):
        b = best[row['file_path']]
        if row['rank'] != i or row['score'] != b['score'] or row['best_chunk_id'] != b['chunk_id']:
            raise ValueError('R0 rank metadata mismatch')


def load_case(src, original_record, hypothesis_record, instruction):
    paths = [src / original_record['result_file']]
    original = source.read(paths[0])
    assets = src / original_record['assets']
    paths += [assets / 'candidates.json', assets / 'chunks.json']
    inventory = source.read(paths[1])['files']
    chunks = source.read(paths[2])
    if digest(inventory) != original_record['candidate_fingerprint']:
        raise ValueError('Candidate inventory mismatch')
    if {r['file_path'] for r in inventory} != {r['file_path'] for r in original['ranking']}:
        raise ValueError('Ranking inventory mismatch')
    case = {'original': original, 'chunks': chunks, 'inventory': inventory,
            'gold': original_record['ground_truth'], 'missing': original_record['missing_gold'],
            'query': original_query(original, instruction)}
    if hypothesis_record is not None and hypothesis_record['status'] == 'ok':
        if any(original_record[k] != hypothesis_record[k] for k in
               ('candidate_fingerprint', 'ground_truth', 'missing_gold', 'assets')):
            raise ValueError('Unpaired source inputs')
        paths.append(src / hypothesis_record['result_file'])
        case['hypothesis'] = source.read(paths[-1])
        case['candidates'] = candidate_pool(original, case['hypothesis'])
    for name in ('original', 'hypothesis'):
        if name in case and digest(case[name]['chunk_scores']) != case[name]['chunk_scores_sha256']:
            raise ValueError('Chunk score checksum mismatch')
    case['source_hashes'] = {str(p.resolve()): sha256_file(p) for p in paths}
    return case


def run(folder):
    protocol = source.read(folder / 'protocol.json')
    for path, checksum in protocol['files_sha256'].items():
        if sha256_file(Path(path)) != checksum:
            raise ValueError('Frozen input/implementation changed: ' + path)
    config = protocol['config']
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(config['physical_gpu']):
        raise ValueError('Require CUDA_VISIBLE_DEVICES=0; never use GPU1')
    src = ROOT / config['source_run']
    source.context(src)
    instruction = source.read(src / 'protocol.json')['semantic']['query_instruction']
    selection = source.read(folder / 'selection.json')
    metas = selection['samples']
    development = set(selection['development_rows'])
    records = {method: {} for method in METHODS}
    for method in METHODS:
        for meta in metas:
            row = source.resumed(folder, method, meta)
            if row is not None:
                for path, checksum in row.get('source_hashes', {}).items():
                    if sha256_file(Path(path)) != checksum:
                        raise ValueError('Checkpoint source changed: ' + path)
                records[method][meta['position']] = row

    def publish(method):
        rows = sorted(records[method].values(), key=lambda r: r['position'])
        source.save(folder / method / 'samples.json', rows)
        summary = aggregate_results(rows, 10, len(metas))
        summary['cohorts'] = {name: aggregate_results(
            [r for r in rows if (r['dataset_row'] in development) == is_dev], 10, count)
            for name, is_dev, count in [('development500', True, len(development)),
                                        ('remaining1794', False, len(metas)-len(development))]}
        source.save(folder / method / 'summary.json', summary)
        source.state(folder, method, summary['status'], selected=len(metas), processed=len(rows),
                     failures=summary['failed_samples'])

    def commit(method, meta, row):
        source.save(folder / method / 'cases' / f"{meta['position']:04d}" / 'record.json', row)
        records[method][meta['position']] = row
        publish(method)
        print(f"{source.now()} {method} {len(records[method])}/{len(metas)} "
              f"row={meta['dataset_row']} status={row['status']}", flush=True)

    def failed(method, meta, dependency, dependency_method):
        path = src / dependency_method / 'cases' / f"{meta['position']:04d}" / 'record.json'
        commit(method, meta, {**meta, 'status': 'failed', 'error_type': 'DependencyFailure',
            'error': dependency_method + ': ' + dependency.get('error', 'Source retrieval failed'),
            'source_hashes': {str(path.resolve()): sha256_file(path)}})

    model = None
    for method in METHODS:
        publish(method)
    while any(len(rows) < len(metas) for rows in records.values()):
        advanced = False
        for meta in metas:
            position = meta['position']
            needed = [m for m in METHODS if position not in records[m]]
            if not needed:
                continue
            a = source.resumed(src, 'S0', meta)
            if a is None:
                continue
            b = source.resumed(src, 'H0', meta) if any(m != 'R0' for m in needed) else None
            if a['status'] == 'failed':
                for method in needed:
                    failed(method, meta, a, 'S0')
                advanced = True
                continue
            if b is not None and b['status'] == 'failed':
                for method in ('F', 'FULL'):
                    if method in needed:
                        failed(method, meta, b, 'H0')
                        needed.remove(method)
                        advanced = True
            if 'R0' not in needed and (b is None or b['status'] != 'ok'):
                continue
            start = time.monotonic()
            case = load_case(src, a, b, instruction)
            base = {'query': case['query'], 'ground_truth': case['gold'],
                    'source_hashes': case['source_hashes']}
            for method in needed:
                if method != 'R0' and 'hypothesis' not in case:
                    continue
                source.state(folder, 'worker', 'running', method=method, position=position,
                             physical_gpu=config['physical_gpu'])
                if method == 'F':
                    full_order = fuse(case)
                    head = full_order[:len(case['candidates'])]
                    saved = {**base, 'candidates': case['candidates'], 'ranking': head}
                else:
                    if model is None:
                        while source.gpu_memory().get(config['physical_gpu'], 0) < config['minimum_free_mib']:
                            source.state(folder, 'worker', 'waiting_for_gpu', physical_gpu=config['physical_gpu'])
                            time.sleep(config['poll_seconds'])
                        model = LocalReranker(ROOT, protocol['reranker'])
                        if model.identity != protocol['reference_model']:
                            raise ValueError('Model identity differs from 500-sample reference')
                        source.pin(folder / 'model.json', {'physical_gpu': config['physical_gpu'], 'identity': model.identity})
                    if method == 'R0':
                        head = case['original']['ranking']
                        chosen = select_chunks(head, case['chunks'], case['original']['chunk_scores'], 50, 3)
                    else:
                        head, chosen = select_evidence(case['candidates'], fuse(case), case['chunks'],
                            case['original']['chunk_scores'], case['hypothesis']['chunk_scores'], 'mixed')
                    scored = model.score(case['query'], [r['embedding_text'] for r in chosen])
                    ranking = rerank(head, chosen, scored)
                    saved = {**base, 'ranking': ranking,
                             'scored_chunks': [{**chunk, **score} for chunk, score in zip(chosen, scored)]}
                    if method == 'R0':
                        audit_original(case, saved)
                    else:
                        saved.update(candidates=case['candidates'], ab_ranking=[r['file_path'] for r in head],
                            metrics={'AB': ranking_metrics([r['file_path'] for r in head], case['gold'], 10),
                                     'ABC': ranking_metrics([r['file_path'] for r in ranking], case['gold'], 10)})
                        audit(case, saved)
                        saved['module_metrics'] = saved.pop('metrics')
                row = source.record(folder, method, meta, saved, case['inventory'], case['gold'], case['missing'],
                    source_hashes=case['source_hashes'], elapsed_seconds=time.monotonic()-start)
                commit(method, meta, row)
                advanced = True
            del case
        if not advanced:
            source.state(folder, 'worker', 'waiting_for_retrieval', physical_gpu=config['physical_gpu'],
                         processed={m: len(v) for m, v in records.items()})
            statuses = [source.read(src / 'status' / (m+'.json')) for m in ('S0', 'H0')]
            if all(s['status'] == 'completed' for s in statuses):
                raise ValueError('Retrieval completed but required checkpoints are missing')
            time.sleep(config['poll_seconds'])
    source.state(folder, 'worker', 'completed', processed={m: len(v) for m, v in records.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/experiments/reranker_full_swe.json')
    args = parser.parse_args()
    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'worker.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if args.action == 'prepare':
                prepare(folder, args.config.resolve())
            else:
                run(folder)
        except Exception as error:
            source.state(folder, args.action, 'failed', error_type=type(error).__name__, error=str(error))
            if args.action == 'run':
                source.state(folder, 'worker', 'failed', error_type=type(error).__name__, error=str(error))
            traceback.print_exc()
            raise


if __name__ == '__main__':
    main()
