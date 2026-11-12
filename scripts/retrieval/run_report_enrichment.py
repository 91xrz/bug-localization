"""Generate OB/EB/S2R, then compare AST64 queries and a fixed RRF grid."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'src'))
from bug_localization.batch import aggregate_results, write_json
from bug_localization.bm25 import prepare_report, collect_files, ranking_metrics
from bug_localization.ast_chunking import RoutingChunkCache
from bug_localization.embeddings import LocalEmbedder
from bug_localization.hybrid import ranked_paths
from bug_localization.report_enrichment import Enricher
from bug_localization.report_enrichment_audit import cached_with_evidence_filter
from bug_localization.report_views import weighted_rrf
from bug_localization.snapshot import get_snapshot, sha256_file
from bug_localization.vector import select_reports, validate_vector_config, retrieve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PROJECT / 'configs/experiments/report_enrichment_ast64.json')
    phase = parser.add_mutually_exclusive_group(required=True)
    phase.add_argument('--prepare', action='store_true')
    phase.add_argument('--run', action='store_true')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    semantic = json.loads((PROJECT / config['semantic_config']).read_text())
    semantic.update(repositories=config['repositories'], limit=args.limit or config['limit'])
    if config.get('selection_manifest') and args.limit is None:
        semantic['selection_manifest'] = config['selection_manifest']
    base = json.loads((PROJECT / semantic['base_config']).read_text())
    validate_vector_config(semantic, base)
    if (semantic['prefix_max_tokens'], semantic['ast_max_chars'], semantic['max_tokens']) != (64, 1500, 512):
        raise ValueError('This experiment requires AST64 / 1500 chars / 512 tokens')
    selected, selection = select_reports(PROJECT, semantic, base, max_samples=config.get('selection_max_samples', 50))
    provider = config.get('query_provider', 'enrichment')
    if provider not in ('enrichment', 'hypothesis'):
        raise ValueError('Unknown query provider')
    folder = PROJECT / ('results/report_hypothesis' if provider == 'hypothesis' else 'results/report_enrichment') / (datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f') + ('_prepare' if args.prepare else '_retrieval'))
    folder.mkdir(parents=True)
    print('Results:', folder, flush=True)
    write_json(folder / 'selection.json', selection)
    write_json(folder / 'effective_config.json', {'experiment': config, 'semantic': semantic, 'base': base,
        'runner_sha256': sha256_file(Path(__file__)), 'implementation_sha256': {
            name: sha256_file(PROJECT / 'src/bug_localization' / name) for name in
            ('report_enrichment.py', 'report_enrichment_audit.py', 'report_hypothesis.py', 'vector.py', 'embeddings.py', 'ast_chunking.py', 'aggregation.py')}})
    write_json(folder / 'status.json', {'status': 'running'})
    try:
        if provider == 'hypothesis':
            from bug_localization.report_hypothesis import HypothesisGenerator
            enricher = HypothesisGenerator(PROJECT, config)
        else:
            enricher = Enricher(PROJECT, config)
    except BaseException as error:
        write_json(folder / 'status.json', {'status': 'failed', 'error': str(error)})
        raise
    generated, failures = [], []
    for i, sample in enumerate(selected, 1):
        try:
            # No labels, code, rankings or patches enter generation.
            try:
                value = enricher.generate(sample['report']['body'], cache_only=args.run)
            except ValueError:
                if provider == 'hypothesis':
                    raise
                value = cached_with_evidence_filter(enricher, sample['report']['body'])
            if config.get('query_routes'):
                wanted = config['query_routes']
                required = {'q0', config.get('fusion_query', 'OB_EB_S2R')}
                if len(wanted) != len(set(wanted)) or not required.issubset(wanted) or not set(wanted).issubset(value['queries']):
                    raise ValueError('Invalid query route subset')
                value['queries'] = {name: value['queries'][name] for name in wanted}
            generated.append(value)
            write_json(folder / f'{i:03d}_queries.json', value)
            print(f'Query {i}/{len(selected)}: ok', flush=True)
        except Exception as error:
            generated.append(None)
            failures.append({'position': i, 'error': str(error)})
            print(f'Query {i}/{len(selected)}: {error}', flush=True)
        write_json(folder / 'status.json', {'status': 'running', 'processed_queries': i, 'failures': failures})
    if args.prepare or failures:
        write_json(folder / 'status.json', {'status': 'failed' if failures else 'completed',
            'selected': len(selected), 'successful': len(selected) - len(failures), 'failures': failures})
        return 1 if failures else 0
    from elasticsearch import Elasticsearch
    encoder = LocalEmbedder(PROJECT, semantic)
    records = {}
    try:
        write_json(folder / 'embedding_identity.json', encoder.identity)
        cache = RoutingChunkCache(PROJECT, encoder.tokenizer, semantic, encoder.fingerprint)
        with Elasticsearch(base['endpoint'], request_timeout=120) as client:
            info = client.info()
            if info['cluster_name'] != base['cluster_name'] or not info['version']['number'].startswith('9.'):
                raise ValueError('Unexpected Elasticsearch cluster')
            for i, (sample, value) in enumerate(zip(selected, generated), 1):
                case = folder / 'cases' / f'{i:03d}'
                case.mkdir(parents=True)
                row = sample['report']
                meta = selection['samples'][i - 1]
                names = list(value['queries']) + [f'rrf_w{w}_c{k}' for w in config['fusion_weights'] for k in config['rrf_constants']]
                try:
                    _, gold, original = prepare_report(row, {**base, 'repository': row['repo_name'], 'issue_id': row['issue_id']})
                    if original != value['queries']['q0']:
                        raise ValueError('Original query mismatch')
                    source, manifest = get_snapshot(PROJECT, row['repo_name'], row['before_fix_sha'])
                    docs, skipped = collect_files(source, manifest, base)
                    write_json(case / 'candidates.json', {'files': [{'file_path': d['file_path'], 'sha256': d['sha256']} for d in docs], 'skipped': skipped})
                    chunks = [c for d in docs for c in cache.get(d, 'blagent_ast')[0]]
                    write_json(case / 'chunks.json', chunks)
                    vectors = encoder.encode([c['embedding_text'] for c in chunks])
                    identity = {'repository': row['repo_name'], 'commit': row['before_fix_sha'], 'route': 'blagent_ast',
                        'model_fingerprint': encoder.fingerprint, **cache.index_identity(docs, 'blagent_ast')}
                    rankings = {}
                    for name, text in value['queries'].items():
                        if not text:
                            rankings[name] = []
                            continue
                        vector, query_info = encoder.query(text)
                        output = retrieve(client, docs, chunks, vectors, vector, identity, 'max', semantic['aggregation_temperature'])
                        rankings[name] = output['ranking']
                        write_json(case / (name + '_retrieval.json'), {**output, 'query': text, 'query_info': query_info})
                    for w in config['fusion_weights']:
                        for k in config['rrf_constants']:
                            rankings[f'rrf_w{w}_c{k}'] = weighted_rrf({'q0': rankings['q0'], 'enriched': rankings[config.get('fusion_query', 'OB_EB_S2R')]}, {'q0': 1, 'enriched': w}, k)
                    missing = sorted(set(gold) - {d['file_path'] for d in docs})
                    case_records = {}
                    for name, ranking in rankings.items():
                        metrics = ranking_metrics(ranked_paths(ranking), gold, 10)
                        write_json(case / (name + '_result.json'), {'ranking': ranking, 'metrics': metrics, 'ground_truth': gold})
                        case_records[name] = {**meta, 'status': 'ok', 'metrics': metrics, 'missing_gold': missing}
                    for name, record in case_records.items():
                        records.setdefault(name, []).append(record)
                except Exception as error:
                    for name in names:
                        records.setdefault(name, []).append({**meta, 'status': 'failed', 'error': str(error), 'error_type': type(error).__name__})
                    write_json(case / 'error.json', {'error': str(error)})
                write_json(folder / 'samples.json', records)
                summaries = {name: aggregate_results(rows, 10, len(selected)) for name, rows in records.items()}
                write_json(folder / 'summary.json', summaries)
                write_json(folder / 'status.json', {'status': 'running', 'processed_samples': i})
                print(f'Retrieval {i}/{len(selected)}: {records["q0"][-1]["status"]}', flush=True)
        failed = any(r['status'] == 'failed' for r in records['q0'])
        write_json(folder / 'status.json', {'status': 'failed' if failed else 'completed', 'processed_samples': len(selected)})
        return int(failed)
    except BaseException as error:
        write_json(folder / 'status.json', {'status': 'failed', 'error': str(error)})
        raise
    finally:
        encoder.close()


if __name__ == '__main__':
    raise SystemExit(main())
