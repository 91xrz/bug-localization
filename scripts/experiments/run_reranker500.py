"""Resumable 500-report experiment: reuse 180 controls, evaluate 320 unseen reports."""
import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import traceback

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bug_localization.batch import aggregate_results, write_json
from bug_localization.bm25 import collect_files, prepare_report, ranking_metrics
from bug_localization.embeddings import LocalEmbedder, digest
from bug_localization.reranking import read_json, run_reranking
from bug_localization.snapshot import get_snapshot, sha256_file
from bug_localization.vector import select_reports, retrieve
from run_direct_hypothesis import Generator

METHODS = ('original', 'original_reranked', 'hypothesis', 'hypothesis_reranked')


def context(config):
    reference = read_json(PROJECT / config['semantic_reference'] / 'effective_config.json')
    semantic = {**reference['semantic'], 'selection_manifest': config['new_selection']}
    base = reference['base']
    selected, selection = select_reports(PROJECT, semantic, base, max_samples=500)
    return semantic, base, selected, selection


def generate(folder, config):
    _, _, selected, selection = context(config)
    target = folder / 'new320/hypothesis'
    target.mkdir(parents=True, exist_ok=True)
    write_json(target / 'selection.json', selection)
    generator = Generator(read_json(PROJECT / config['generation_config']))
    for meta, sample in zip(selection['samples'], selected):
        path = target / f"{meta['position']:03d}_queries.json"
        if path.exists():
            saved = read_json(path)
            if saved.get('identity') == generator.identity and saved.get('payload') == {'original_report': sample['report']['body']}:
                continue
        try:
            value = generator.generate(sample['report']['body'])
        except Exception as error:
            value = {'query': '', 'generation_error': str(error), 'error_type': type(error).__name__,
                     'payload': {'original_report': sample['report']['body']}}
        write_json(path, value)
        write_json(folder / 'generation_progress.json', {'processed': meta['position'], 'selected': len(selected)})
        print(f"Generation {meta['position']}/{len(selected)}", flush=True)


def semantic_retrieval(folder, config):
    from elasticsearch import Elasticsearch
    from bug_localization.ast_chunking import RoutingChunkCache
    from bug_localization.parallel_chunking import create_pool, chunk_documents
    semantic, base, selected, selection = context(config)
    assets, direct = folder / 'new320/semantic', folder / 'new320/hypothesis'
    assets.mkdir(parents=True, exist_ok=True)
    write_json(assets / 'selection.json', selection)
    write_json(assets / 'effective_config.json', {'semantic': semantic, 'base': base})
    encoder = LocalEmbedder(PROJECT, semantic)
    expected = read_json(PROJECT / config['semantic_reference'] / 'embedding_identity.json')
    if encoder.identity != expected:
        raise ValueError('Embedding model/environment differs from the frozen 180-report control')
    write_json(assets / 'embedding_identity.json', encoder.identity)
    cache = RoutingChunkCache(PROJECT, encoder.tokenizer, semantic, encoder.fingerprint)
    pool = create_pool(PROJECT, semantic, encoder.fingerprint)
    records = {'q0': [], 'direct_hypothesis': []}
    try:
        with Elasticsearch(base['endpoint'], request_timeout=120) as client:
            if client.info()['cluster_name'] != base['cluster_name']:
                raise ValueError('Unexpected Elasticsearch cluster')
            for meta, sample in zip(selection['samples'], selected):
                pos = f"{meta['position']:03d}"
                case, hypothesis_case = assets / 'cases' / pos, direct / 'cases' / pos
                case.mkdir(parents=True, exist_ok=True)
                hypothesis_case.mkdir(parents=True, exist_ok=True)
                generated = read_json(direct / (pos + '_queries.json'))
                checkpoint = case / 'checkpoint.json'
                checkpoint_key = digest({'sample': meta, 'generated': generated, 'encoder': encoder.identity})
                if checkpoint.exists():
                    saved = read_json(checkpoint)
                    if saved['key'] != checkpoint_key:
                        raise ValueError('Semantic checkpoint identity changed')
                    for name in records:
                        records[name].append(saved['records'][name])
                else:
                    created_index = None
                    values = {}
                    try:
                        row = sample['report']
                        _, gold, body = prepare_report(row, {**base, 'repository': row['repo_name'], 'issue_id': row['issue_id']})
                        root, manifest = get_snapshot(PROJECT, row['repo_name'], row['before_fix_sha'])
                        documents, skipped = collect_files(root, manifest, base)
                        inventory = [{'file_path': d['file_path'], 'sha256': d['sha256']} for d in documents]
                        write_json(case / 'candidates.json', {'files': inventory, 'skipped': skipped})
                        chunks = [chunk for item in chunk_documents(pool, documents, 'blagent_ast') for chunk in item[0]]
                        write_json(case / 'chunks.json', chunks)
                        vectors = encoder.encode([c['embedding_text'] for c in chunks])
                        identity = {'repository': row['repo_name'], 'commit': row['before_fix_sha'], 'route': 'blagent_ast',
                                    'model_fingerprint': encoder.fingerprint, **cache.index_identity(documents, 'blagent_ast')}
                        fingerprint = digest({'identity': identity, 'chunks': [c['chunk_id'] for c in chunks],
                                              'vectors_sha256': hashlib.sha256(vectors.astype('<f4').tobytes()).hexdigest()})
                        index = 'bl-vector-blagent_ast-' + row['before_fix_sha'][:12] + '-' + fingerprint[:16]
                        if not client.indices.exists(index=index):
                            created_index = index
                        missing = sorted(set(gold) - {d['file_path'] for d in documents})
                        for name, query in [('q0', body), ('direct_hypothesis', generated['query'])]:
                            if name == 'direct_hypothesis' and (generated['generation_error'] or not query.strip()):
                                values[name] = {**meta, 'status': 'failed', 'error_type': 'GenerationError',
                                                'error': generated['generation_error'] or 'Empty hypothesis'}
                                continue
                            vector, query_info = encoder.query(query)
                            result = retrieve(client, documents, chunks, vectors, vector, identity, 'max', semantic['aggregation_temperature'])
                            if result['index'] != index:
                                raise ValueError('Unexpected semantic index identity')
                            metrics = ranking_metrics([r['file_path'] for r in result['ranking']], gold, 10)
                            values[name] = {**meta, 'status': 'ok', 'metrics': metrics, 'missing_gold': missing}
                            target = case if name == 'q0' else hypothesis_case
                            stem = 'q0' if name == 'q0' else 'direct'
                            write_json(target / (stem + '_retrieval.json'), {**result, 'query': query, 'query_info': query_info})
                            write_json(target / (stem + '_result.json'), {'ranking': result['ranking'], 'metrics': metrics, 'ground_truth': gold})
                    except Exception as error:
                        failure = {**meta, 'status': 'failed', 'error_type': type(error).__name__, 'error': str(error)}
                        write_json(case / 'error.json', {**failure, 'traceback': traceback.format_exc()})
                        for name in records:
                            values.setdefault(name, failure)
                    finally:
                        # Keep all pre-existing indexes. Only this case's newly created, verified index is disposable.
                        if created_index and config['delete_only_new_temporary_indexes'] and client.indices.exists(index=created_index):
                            ownership = client.indices.get_mapping(index=created_index)[created_index]['mappings'].get('_meta', {})
                            if ownership.get('project') != 'bug-localization' or ownership.get('fingerprint') != fingerprint:
                                raise ValueError('Refusing cleanup of an unexpected index')
                            client.indices.delete(index=created_index)
                    write_json(checkpoint, {'key': checkpoint_key, 'records': values})
                    for name in records:
                        records[name].append(values[name])
                write_json(assets / 'samples.json', {'q0': records['q0']})
                write_json(direct / 'samples.json', {'direct_hypothesis': records['direct_hypothesis'], 'original': records['q0']})
                write_json(folder / 'retrieval_progress.json', {'processed': meta['position'], 'selected': len(selected),
                    'failures': {name: sum(r['status'] != 'ok' for r in rows) for name, rows in records.items()}})
                print(f"Retrieval {meta['position']}/{len(selected)}", flush=True)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        encoder.close()
    write_json(direct / 'audit.json', {'selection_verified': True, 'embedding_identity_matches': True,
        'candidate_and_chunk_inputs_match': True, 'same_snapshots_and_vectors_used_for_both_queries': True})


def reranking(folder, config, hypothesis):
    work = folder / 'new320'
    model_config = read_json(PROJECT / config['reranker_config'])
    source = work / ('hypothesis' if hypothesis else 'semantic')
    output = work / ('reranking' if hypothesis else 'original_reranking')
    if hypothesis:
        model_config.update(source_query_route='direct_hypothesis', semantic_assets_run=str(work / 'semantic'),
                            reranker_query_source='original_report')
    kwargs = {'resume_folder': output} if (output / 'effective_config.json').exists() else {'output_folder': output}
    _, summary = run_reranking(PROJECT, source, model_config, ['blagent_ast'], **kwargs)
    if summary['status'] == 'completed':
        subprocess.run([sys.executable, str(PROJECT / 'scripts/analysis/analyze_reranking.py'), '--run', str(output),
                        '--selection', str(PROJECT / config['new_selection'])], check=True, cwd=PROJECT)
    return summary


def paired_stats(records):
    rows = records['hypothesis_reranked']
    result = {}
    for name, baseline in records.items():
        if name == 'hypothesis_reranked':
            continue
        deltas = [(b.get('metrics', {}).get('average_precision_at_k', 0) - a.get('metrics', {}).get('average_precision_at_k', 0))
                  for a, b in zip(baseline, rows)]
        result[name] = {'wins': sum(x > 1e-12 for x in deltas), 'ties': sum(abs(x) <= 1e-12 for x in deltas),
                        'losses': sum(x < -1e-12 for x in deltas), 'delta_MAP@10': sum(deltas) / len(deltas)}
    return result


def finalize(folder, config):
    old_path = PROJECT / config['prior_run']
    expected_model = read_json(old_path / 'reranking/effective_config.json')['model']
    for name in ('original_reranking', 'reranking'):
        if read_json(folder / 'new320' / name / 'effective_config.json')['model'] != expected_model:
            raise ValueError('Reranker model/environment differs from the corrected 180-report control')
    old = read_json(old_path / 'paired_samples.json')
    original = read_json(folder / 'new320/original_reranking/samples.json')['blagent_ast']
    hypothesis = read_json(folder / 'new320/reranking/samples.json')['blagent_ast']
    new = {'original': original['baseline'], 'original_reranked': original['reranked'],
           'hypothesis': hypothesis['baseline'], 'hypothesis_reranked': hypothesis['reranked']}
    new_selection = read_json(PROJECT / config['new_selection'])['samples']
    old_selection = read_json(old_path / 'selection.json')['samples']
    for records, selection in ((old, old_selection), (new, new_selection)):
        for name in METHODS:
            if len(records[name]) != len(selection):
                raise ValueError('Incomplete cohort: ' + name)
            for record, meta in zip(records[name], selection):
                if any(record[k] != meta[k] for k in ('dataset_row', 'repository', 'issue_id', 'before_fix_sha')):
                    raise ValueError('Cohort sample order/identity mismatch')
    for a, b in zip(new['original_reranked'], new['hypothesis_reranked']):
        if a['status'] != 'ok' or b['status'] != 'ok':
            continue
        x = read_json(folder / 'new320/original_reranking' / a['result_file'])
        y = read_json(folder / 'new320/reranking' / b['result_file'])
        if x['query'] != y['query'] or x['candidate_fingerprint'] != y['candidate_fingerprint'] or x['ground_truth'] != y['ground_truth']:
            raise ValueError('Paired reranker original query/candidate/label mismatch')
    combined = {name: old[name] + [{**r, 'position': len(old_selection) + i} for i, r in enumerate(new[name], 1)] for name in METHODS}
    cohorts = {'all500': combined, 'new320': new, 'previous180': old}
    summary = {cohort: {'selected_samples': len(records['original']),
        'methods': {name: aggregate_results(rows, 10) for name, rows in records.items()},
        'hypothesis_reranked_vs': paired_stats(records)} for cohort, records in cohorts.items()}
    projects = {cohort: {repo: {name: aggregate_results([r for r in rows if r['repository'] == repo], 10)
        for name, rows in records.items()} for repo in sorted({r['repository'] for r in records['original']})}
        for cohort, records in cohorts.items()}
    failures = any(r['status'] != 'ok' for records in cohorts.values() for rows in records.values() for r in rows)
    audits = {}
    for name in ('original_reranking', 'reranking'):
        path = folder / 'new320' / name / 'verification.json'
        audits[name] = read_json(path) if path.exists() else None
    final = {'status': 'completed_with_failures' if failures else 'completed', 'cohorts': summary,
        'query_policy': config['query_policy'], 'new_samples_separate_from_previous180': True,
        'same_original_report_reranker_query_verified_for_successful_pairs': True, 'audits': audits}
    if not failures and not all(audits.values()):
        raise ValueError('Successful result requires both independent reranking audits')
    write_json(folder / 'final_summary.json', final)
    write_json(folder / 'project_summary.json', projects)
    write_json(folder / 'paired_samples.json', {'cohorts': cohorts})
    with (folder / 'comparison.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['cohort', 'method', 'successful', 'failed',
            'MAP@10', 'MRR@10', 'Top-1', 'Top-5', 'Top-10', 'MeanRecall@10'])
        writer.writeheader()
        for cohort, group in summary.items():
            for name, stats in group['methods'].items():
                writer.writerow({'cohort': cohort, 'method': name, 'successful': stats['successful_samples'],
                    'failed': stats['failed_samples'], **stats['metrics']})
    return final['status']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PROJECT / 'configs/experiments/hypothesis_reranker500.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=['generate', 'retrieve', 'rerank_original', 'rerank_hypothesis'])
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    config, folder = read_json(PROJECT / args.config), (PROJECT / args.output).resolve()
    if args.stage:
        protocol_path = folder / 'experiment.json'
        if protocol_path.exists():
            protocol = read_json(protocol_path)
            if protocol['config'] != config:
                raise ValueError('Experiment configuration changed between stages')
            for section in ('input_sha256', 'implementation_sha256'):
                for name, checksum in protocol[section].items():
                    if sha256_file(Path(name)) != checksum:
                        raise ValueError('Frozen experiment input/implementation changed: ' + name)
        {'generate': generate, 'retrieve': semantic_retrieval,
         'rerank_original': lambda f, c: reranking(f, c, False),
         'rerank_hypothesis': lambda f, c: reranking(f, c, True)}[args.stage](folder, config)
        return
    _, _, _, selection = context(config)
    whole = read_json(PROJECT / config['selection'])
    previous = read_json(PROJECT / config['prior_run'] / 'selection.json')
    if len(whole['samples']) != 500 or len(selection['samples']) != 320 or whole['samples'][:180] != previous['samples']:
        raise ValueError('Invalid 180+320 selection')
    if [{**r, 'position': i} for i, r in enumerate(whole['samples'][180:], 1)] != selection['samples']:
        raise ValueError('New selection does not match the 500-case manifest')
    prior = read_json(PROJECT / config['prior_run'] / 'final_summary.json')
    if prior['status'] != 'completed' or not prior['same_query_as_original_reranker_verified']:
        raise ValueError('Require the completed, corrected 180-case control')
    if read_json(PROJECT / config['reranker_config']) != read_json(PROJECT / config['prior_run'] / 'experiment.json')['reranker']:
        raise ValueError('Reranker settings differ from the 180-case control')
    if args.check_only:
        print('Verified 500 = previous 180 + new 320; corrected original-report reranker protocol and fixed parameters.')
        return
    folder.mkdir(parents=True, exist_ok=True)
    inputs = [PROJECT / config[key] for key in ('selection', 'new_selection', 'generation_config', 'reranker_config')]
    inputs += [PROJECT / config['prior_run'] / name for name in ('paired_samples.json', 'final_summary.json')]
    implementation = [Path(__file__), PROJECT / 'scripts/experiments/run_direct_hypothesis.py',
                      PROJECT / 'src/bug_localization/reranking.py', PROJECT / 'scripts/analysis/analyze_reranking.py']
    protocol = {'config': config, 'input_sha256': {str(p): sha256_file(p) for p in inputs},
                'implementation_sha256': {str(p): sha256_file(p) for p in implementation}}
    if (folder / 'experiment.json').exists() and read_json(folder / 'experiment.json') != protocol:
        raise ValueError('Cannot resume a run after its protocol/inputs/implementation changed')
    write_json(folder / 'experiment.json', protocol)
    write_json(folder / 'selection.json', whole)
    status = {'pid': os.getpid(), 'run': str(folder), 'tmux_session': config['tmux_session']}
    def update(stage, **extra):
        status.update(stage=stage, updated_at=datetime.now(timezone(timedelta(hours=8))).isoformat(), **extra)
        write_json(folder / 'automation_status.json', status)
    try:
        for stage in ('generate', 'retrieve', 'rerank_original', 'rerank_hypothesis'):
            marker = folder / (stage + '.completed.json')
            if marker.exists():
                continue
            update(stage)
            with (folder / (stage + '.log')).open('a') as log:
                subprocess.run([sys.executable, '-u', str(Path(__file__)), '--config', str(PROJECT / args.config),
                    '--output', str(folder), '--stage', stage], cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT, check=True)
            write_json(marker, {'completed': True})
        update('summarizing')
        for section in ('input_sha256', 'implementation_sha256'):
            for name, checksum in protocol[section].items():
                if sha256_file(Path(name)) != checksum:
                    raise ValueError('Frozen experiment input/implementation changed: ' + name)
        result = finalize(folder, config)
        update(result, exit_code=0 if result == 'completed' else 1)
    except BaseException as error:
        update('failed', exit_code=1, error=str(error), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
