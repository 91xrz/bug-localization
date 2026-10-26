"""Separate generation/retrieval phases reusing existing IRBL implementations."""

from datetime import datetime, timezone
import json
from pathlib import Path
import time

from .batch import aggregate_results, write_checkpoints, write_json
from .bm25 import prepare_report, collect_files, ranking_metrics, run as run_bm25
from .dependency_context import DependencyRoutingCache
from .embeddings import LocalEmbedder, digest
from .hybrid import full_lexical_ranking, ranked_paths
from .report_views import QueryGenerator, report_input, weighted_rrf
from .snapshot import get_snapshot, sha256_file
from .vector import select_reports, validate_vector_config, retrieve, BASELINE_ROUTES


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def load_settings(project, config):
    semantic = read(project / config['semantic_config'])
    base = read(project / semantic['base_config'])
    validate_vector_config(semantic, base)
    if config['semantic_route'] not in BASELINE_ROUTES:
        raise ValueError('Module 2 requires a baseline chunker to isolate query effects')
    if config['evaluation_k'] != 10 or semantic['file_aggregation'] != 'max':
        raise ValueError('Initial module 2 protocol requires Max and @10')
    if set(config['weights']) != {'q0', 'q1', 'q2'}:
        raise ValueError('Exactly q0, q1, q2 weights are required')
    weighted_rrf({q: [] for q in config['weights']}, config['weights'], config['rrf_constant'])
    if config['weights'] != {'q0': 1, 'q1': 1, 'q2': 1} and not config.get('weight_development_reference'):
        raise ValueError('Nondefault weights require a documented development-set reference')
    semantic = {**semantic, **{k: config[k] for k in ('limit', 'repositories', 'selection_manifest') if k in config}}
    selected, selection = select_reports(project, semantic, base)
    return semantic, base, selected, selection


def create_run(project, config, semantic, base, selection, phase):
    folder = project / 'results/multiview' / (datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f') + '_' + phase)
    folder.mkdir(parents=True)
    write_json(folder / 'selection.json', selection)
    write_json(folder / 'effective_config.json', {'multiview': config, 'semantic': semantic, 'base': base,
        'implementation_sha256': {name: sha256_file(Path(__file__).with_name(name)) for name in
            ('multiview.py', 'report_views.py', 'vector.py', 'bm25.py', 'aggregation.py', 'snapshot.py', 'bench4bl.py')}})
    write_json(folder / 'status.json', {'status': 'running', 'phase': phase})
    return folder


def prepare_queries(project, config):
    semantic, base, selected, selection = load_settings(project, config)
    generator = QueryGenerator(project, config['llm'])
    folder = create_run(project, config, semantic, base, selection, 'prepare')
    records = []
    for position, sample in enumerate(selected, 1):
        row = sample['report']
        record = {**selection['samples'][position - 1]}
        try:
            generated = generator.generate(report_input(row, base['query_fields']))
            write_json(folder / f'{position:03d}_queries.json', generated)
            record.update(status='ok', cache_key=generated['key'], cache_hit=generated['cache_hit'],
                          empty_views=[q for q, text in generated['views'].items() if not text])
        except Exception as error:
            record.update(status='failed', error_type=type(error).__name__, error=str(error))
        records.append(record)
        write_json(folder / 'samples.json', records)
        print(f"[{position}/{len(selected)}] #{row['issue_id']}: {record['status']}", flush=True)
        write_json(folder / 'status.json', {'status': 'running', 'processed_samples': position,
                                           'selected_samples': len(selected)})
    status = {'status': 'failed' if any(r['status'] != 'ok' for r in records) else 'completed',
              'successful_samples': sum(r['status'] == 'ok' for r in records), 'selected_samples': len(selected)}
    write_json(folder / 'status.json', status)
    return folder, status


def run_multiview(project, config):
    from elasticsearch import Elasticsearch

    semantic, base, selected, selection = load_settings(project, config)
    generator = QueryGenerator(project, config['llm'])
    # Fail before loading an embedding model or creating indices if extraction is incomplete.
    queries = [generator.generate(report_input(s['report'], base['query_fields']), cache_only=True) for s in selected]
    folder = create_run(project, config, semantic, base, selection, 'retrieval')
    names = ('q0', 'q1', 'q2', 'q0_bm25', 'original_rrf', 'multiview_rrf')
    records = {name: [] for name in names}
    started = time.monotonic()
    embedder = None
    try:
        with Elasticsearch(base['endpoint'], request_timeout=120) as client:
            info = client.info()
            if info['cluster_name'] != base['cluster_name'] or not info['version']['number'].startswith('9.'):
                raise ValueError('Unexpected Elasticsearch cluster')
            embedder = LocalEmbedder(project, semantic)
            write_json(folder / 'embedding_identity.json', embedder.identity)
            cache = DependencyRoutingCache(project, embedder.tokenizer, semantic, embedder.fingerprint)
            for position, (sample, generated) in enumerate(zip(selected, queries), 1):
                row = sample['report']
                record = {k: selection['samples'][position - 1][k] for k in
                          ('position', 'dataset_row', 'repository', 'issue_id', 'before_fix_sha')}
                case = folder / 'cases' / f'{position:03d}'
                case.mkdir(parents=True)
                write_json(case / 'queries.json', generated)
                try:
                    report_config = {**base, 'repository': row['repo_name'], 'issue_id': row['issue_id']}
                    _, gold, original = prepare_report(row, report_config)
                    if original != generated['views']['q0']:
                        raise ValueError('Original query changed')
                    source, manifest = get_snapshot(project, row['repo_name'], row['before_fix_sha'],
                        **({'source_provider': base['source_provider'], 'source_dataset': base['dataset']}
                           if base.get('source_provider') else {}))
                    documents, skipped = collect_files(source, manifest, base)
                    inventory = [{'file_path': d['file_path'], 'sha256': d['sha256']} for d in documents]
                    missing = sorted(set(gold) - {d['file_path'] for d in documents})
                    write_json(case / 'candidates.json', {'files': inventory, 'skipped': skipped})
                    chunks, parents = [], []
                    route = config['semantic_route']
                    for document in documents:
                        c, p = cache.get(document, route)
                        chunks.extend(c)
                        parents.extend({'file_path': document['file_path'], **v} for v in p)
                    write_json(case / 'chunks.json', {'chunks': chunks, 'original_blocks': parents})
                    vectors = embedder.encode([c['embedding_text'] for c in chunks])
                    identity = {'repository': row['repo_name'], 'commit': row['before_fix_sha'], 'route': route,
                                'model_fingerprint': embedder.fingerprint, **cache.index_identity(documents, route)}
                    rankings = {}
                    for name in ('q0', 'q2'):
                        text = generated['views'][name]
                        if not text:
                            rankings[name] = []
                            continue
                        query_vector, query_info = embedder.query(text)
                        output = retrieve(client, documents, chunks, vectors, query_vector, identity,
                                          semantic['file_aggregation'], semantic['aggregation_temperature'])
                        if 'bench4bl_source' in manifest:
                            output['source_provenance'] = manifest['bench4bl_source']
                        rankings[name] = output['ranking']
                        write_json(case / (name + '_retrieval.json'), {**output, 'query': text, 'query_info': query_info})
                    for name, text in (('q0_bm25', original), ('q1', generated['views']['q1'])):
                        if not text:
                            rankings[name] = []
                            continue
                        if name == 'q1' and not client.indices.analyze(index=lexical['index'], analyzer='code', text=text)['tokens']:
                            rankings[name] = []
                            write_json(case / 'q1_audit.json', {'empty_reason': 'No terms from the existing code analyzer'})
                            continue
                        # Override only the retrieval text; original report and extraction stay in queries.json.
                        query_row = {**row, 'body': text}
                        lexical = run_bm25(project, {**report_config, 'query_fields': ['body']}, report=query_row,
                                           output_directory=case / name, verbose=False)
                        if lexical['candidate_files'] != inventory:
                            raise ValueError('Lexical and semantic candidate inventories differ')
                        rankings[name], audit = full_lexical_ranking(client, lexical)
                        write_json(case / (name + '_audit.json'), audit)
                    rankings['original_rrf'] = weighted_rrf({'q0': rankings['q0'], 'bm25': rankings['q0_bm25']},
                                                            {'q0': 1, 'bm25': 1}, config['rrf_constant'])
                    rankings['multiview_rrf'] = weighted_rrf({q: rankings[q] for q in ('q0', 'q1', 'q2')},
                                                             config['weights'], config['rrf_constant'])
                    for name, ranking in rankings.items():
                        metrics = ranking_metrics(ranked_paths(ranking), gold, config['evaluation_k'])
                        path = case / (name + '_result.json')
                        write_json(path, {**record, 'route': name, 'ranking': ranking, 'metrics': metrics,
                                         'ground_truth': gold, 'missing_gold': missing,
                                         **({'source_provenance': manifest['bench4bl_source']} if 'bench4bl_source' in manifest else {}),
                                         'candidate_fingerprint': digest(inventory), 'query_key': generated['key']})
                        records[name].append({**record, 'status': 'ok', 'metrics': metrics,
                                              'indexed_files': len(documents), 'missing_gold': missing,
                                              'result_file': path.relative_to(folder).as_posix()})
                except Exception as error:
                    # A paired case is all-or-nothing: never compare routes on different denominators.
                    for name in names:
                        records[name] = [r for r in records[name] if r['position'] != position]
                        records[name].append({**record, 'status': 'failed', 'error_type': type(error).__name__, 'error': str(error)})
                    write_json(case / 'error.json', records['q0'][-1])
                summaries = {name: aggregate_results(values, config['evaluation_k'], len(selected)) for name, values in records.items()}
                for name in names:
                    sub = folder / name
                    sub.mkdir(exist_ok=True)
                    write_checkpoints(sub, records[name], summaries[name])
                print(f"[{position}/{len(selected)}] #{row['issue_id']}: {records['q0'][-1]['status']}", flush=True)
                write_json(folder / 'status.json', {'status': 'running', 'processed_samples': position, 'selected_samples': len(selected)})
        summary = {'status': 'failed' if any(r['status'] == 'failed' for r in records['q0']) else 'completed',
                   'selection_sha256': selection['selection_sha256'], 'routes': summaries,
                   'weights': config['weights'], 'rrf_constant': config['rrf_constant'],
                   'elapsed_seconds': time.monotonic() - started,
                   'input_depth': 'complete semantic rankings and all positive BM25 matches',
                   'empty_view_policy': 'empty ranking, zero contribution, no weight renormalization',
                   'scope': 'Exploratory query experiment; weights not fitted on this test set'}
        write_json(folder / 'summary.json', summary)
        write_json(folder / 'status.json', {'status': summary['status'], 'processed_samples': len(selected)})
        return folder, summary
    except BaseException as error:
        write_json(folder / 'status.json', {'status': 'failed', 'error_type': type(error).__name__, 'error': str(error)})
        raise
    finally:
        if embedder is not None:
            embedder.close()
