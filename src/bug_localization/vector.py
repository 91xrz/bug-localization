"""Dynamic/method/fixed Elasticsearch vector retrieval and file RRF evaluation."""

import csv
import hashlib
from datetime import datetime, timedelta, timezone
from importlib.metadata import version
import json
from pathlib import Path
import time
import traceback

from .batch import aggregate_results, select_samples, write_checkpoints, write_json
from .aggregation import AGGREGATIONS, aggregate_chunk_scores, validate_temperature
from .bm25 import collect_files, prepare_report, ranking_metrics, validate_config
from .token_ast import EXPERIMENTAL_ROUTES
from .dependency_context import DEPENDENCY_ROUTE, DependencyRoutingCache as RoutingChunkCache, validate_dependency_config
from .languages import grammar_versions, language_for_path, parser_for_language, validate_language_config
from .embeddings import LocalEmbedder, digest
from .snapshot import get_snapshot, sha256_file
from .parallel_chunking import create_pool, chunk_documents

BASELINE_ROUTES = ('dynamic', 'method', 'fixed', 'blagent_ast')
RETRIEVAL_ROUTES = BASELINE_ROUTES + EXPERIMENTAL_ROUTES + (DEPENDENCY_ROUTE,)
FUSIONS = {'rrf': ('dynamic', 'method'), 'rrf_three': ('dynamic', 'method', 'fixed')}


def active_routes(config):
    routes = config.get('chunking_routes', list(BASELINE_ROUTES))
    if not isinstance(routes, (list, tuple)) or not routes or any(route not in RETRIEVAL_ROUTES for route in routes):
        raise ValueError('chunking_routes must contain registered semantic routes')
    if len(routes) != len(set(routes)):
        raise ValueError('chunking_routes must not contain duplicate routes')
    return tuple(routes)


def active_fusions(config):
    enabled = config.get('enable_fusion', True)
    if type(enabled) is not bool:
        raise ValueError('enable_fusion must be a boolean')
    return FUSIONS if enabled else {}


def validate_vector_config(config, base):
    validate_config(base)
    active_fusions(config)
    routes = active_routes(config)
    if DEPENDENCY_ROUTE in routes:
        validate_dependency_config(config)
    if any(not set(inputs).issubset(routes) for inputs in active_fusions(config).values()):
        raise ValueError('RRF requires all of its input chunking routes')
    if config.get('file_aggregation', 'max') not in AGGREGATIONS:
        raise ValueError('Unknown file_aggregation')
    validate_temperature(config.get('aggregation_temperature', 0.1))
    validate_language_config(base['extensions'], config.get('language_overrides', {}))
    if type(config.get('ast_max_chars', 1500)) is not int or not 1 <= config.get('ast_max_chars', 1500) <= 100000:
        raise ValueError('ast_max_chars must be an integer in [1, 100000]')
    for field, minimum, maximum in [('max_tokens', 128, 32768), ('prefix_max_tokens', 16, 256),
                                    ('embedding_batch_size', 1, 128), ('dynamic_general_cost', 6, 10000),
                                    ('dynamic_max_lines', 1, 1000), ('evaluation_k', 10, 10000),
                                    ('fixed_chunk_tokens', 1, 32768),
                                    ('rrf_constant', 1, 10000)]:
        if type(config.get(field)) is not int or not minimum <= config[field] <= maximum:
            raise ValueError(f'{field} must be an integer in [{minimum}, {maximum}]')
    if config['prefix_max_tokens'] + 32 >= config['max_tokens']:
        raise ValueError('Prefix budget leaves insufficient code space')
    if config['fixed_chunk_tokens'] + config['prefix_max_tokens'] + 8 > config['max_tokens']:
        raise ValueError('Fixed body plus prefix and special-token allowance exceeds max_tokens')
    if config.get('device') not in ('cuda', 'cpu'):
        raise ValueError('device must be cuda or cpu')
    gpu_count = config.get('gpu_count', 1)
    if type(gpu_count) is not int or not 1 <= gpu_count <= 8:
        raise ValueError('gpu_count must be an integer in [1, 8]')
    if config['device'] == 'cpu' and gpu_count != 1:
        raise ValueError('gpu_count greater than 1 requires cuda')
    if type(config.get('chunk_workers', 1)) is not int or not 1 <= config.get('chunk_workers', 1) <= 64:
        raise ValueError('chunk_workers must be an integer in [1, 64]')
    if type(config.get('chunk_prefetch', False)) is not bool:
        raise ValueError('chunk_prefetch must be a boolean')
    if not isinstance(config.get('query_instruction'), str) or not config['query_instruction'].strip():
        raise ValueError('A fixed query instruction is required')
    if not isinstance(config.get('model_path'), str) or not config['model_path']:
        raise ValueError('A local model_path is required')


def rrf(rankings, constant=60):
    if constant <= 0 or not rankings:
        raise ValueError('RRF requires rankings and a positive constant')
    candidates, ranks = set(), []
    for ranking in rankings:
        paths = [item['file_path'] for item in ranking]
        if len(paths) != len(set(paths)):
            raise ValueError('RRF inputs must be unique file rankings')
        candidates.update(paths)
        ranks.append({path: position for position, path in enumerate(paths, 1)})
    scored = [{'file_path': path, 'score': sum(1 / (constant + route[path]) for route in ranks if path in route),
               'route_ranks': [route.get(path) for route in ranks]} for path in candidates]
    scored.sort(key=lambda item: (-item['score'], item['file_path']))
    return [{'rank': i, **item} for i, item in enumerate(scored, 1)]


def select_reports(project, config, base):
    from datasets import load_from_disk

    data = load_from_disk(str(project / 'datasets' / base['dataset']))[base['split']]
    metadata_path = project / 'datasets' / base['dataset'] / base['split'] / 'dataset_info.json'
    metadata_hash = sha256_file(metadata_path)
    reference = None
    if config.get('selection_manifest'):
        path = Path(config['selection_manifest'])
        reference = json.loads((path if path.is_absolute() else project / path).read_text(encoding='utf-8'))
        if reference['dataset_metadata_sha256'] != metadata_hash:
            raise ValueError('Dataset metadata differs from the selected baseline manifest')
        selected = []
        for position, entry in enumerate(reference['samples'], 1):
            if entry['position'] != position:
                raise ValueError('Selection positions must be sequential')
            selected.append({'dataset_row': entry['dataset_row'], 'report': data[entry['dataset_row']]})
        if not 1 <= len(selected) <= 50:
            raise ValueError('The semantic smoke batch must contain 1-50 samples')
    else:
        selected = select_samples(data, config['repositories'], config['limit'])
    manifest = []
    seen = set()
    for position, sample in enumerate(selected, 1):
        row = sample['report']
        key = row['repo_name'], row['issue_id']
        if key in seen:
            raise ValueError('Duplicate reports in semantic selection')
        seen.add(key)
        manifest.append({'position': position, 'dataset_row': sample['dataset_row'],
                         'repository': row['repo_name'], 'issue_id': row['issue_id'],
                         'before_fix_sha': row.get('before_fix_sha'),
                         'query_sha256': hashlib.sha256(json.dumps({field: row.get(field) for field in base['query_fields']}, sort_keys=True).encode()).hexdigest(),
                         # Match the existing BM25 selection-manifest convention exactly.
                         'labels_sha256': hashlib.sha256(str(row.get('updated_files')).encode()).hexdigest()})
    # Existing BM25 uses json.dumps defaults (ASCII escapes); report fields here
    # are ASCII in the selection manifest, so the fingerprints are identical.
    selection_hash = digest(manifest)
    if reference and (manifest != reference['samples'] or selection_hash != reference['selection_sha256']):
        raise ValueError('Reports, queries or ground truth changed since the baseline run')
    return selected, {'dataset_metadata_sha256': metadata_hash, 'selection_sha256': selection_hash,
                      'selection_rule': 'Pinned baseline manifest' if reference else 'First matching records in original dataset order; no outcome filtering',
                      'samples': manifest}


def retrieve(client, documents, chunks, vectors, query_vector, identity,
             file_aggregation='max', temperature=0.1):
    """Score every chunk once in ES, then compare three file aggregations."""
    import numpy as np
    from elasticsearch import helpers

    vector_checksum = hashlib.sha256(vectors.astype('<f4').tobytes()).hexdigest()
    fingerprint = digest({'identity': identity, 'chunks': [c['chunk_id'] for c in chunks],
                          'vectors_sha256': vector_checksum})
    index = 'bl-vector-' + identity['route'] + '-' + identity['commit'][:12] + '-' + fingerprint[:16]
    ownership = {'project': 'bug-localization', 'fingerprint': fingerprint, **identity}
    definition = {'settings': {'number_of_shards': 1, 'number_of_replicas': 0},
                  'mappings': {'dynamic': 'strict', '_meta': ownership, 'properties': {
                      'file_path': {'type': 'keyword'}, 'chunk_id': {'type': 'keyword'},
                      'record': {'type': 'object', 'enabled': False},
                      'embedding': {'type': 'dense_vector', 'dims': vectors.shape[1], 'index': False}}}}
    if client.indices.exists(index=index):
        if client.indices.get_mapping(index=index)[index]['mappings'].get('_meta') != ownership:
            raise ValueError('Vector index ownership mismatch')
    else:
        client.indices.create(index=index, **definition)
    count = client.count(index=index)['count']
    reused = count == len(chunks)
    if count < len(chunks):
        helpers.bulk(client, ({'_index': index, '_id': chunk['chunk_id'], '_source': {
            'file_path': chunk['file_path'], 'chunk_id': chunk['chunk_id'], 'record': chunk,
            'embedding': vector.tolist()}} for chunk, vector in zip(chunks, vectors)),
                     chunk_size=100, max_chunk_bytes=5 * 1024 * 1024)
        client.indices.refresh(index=index)
    if client.count(index=index)['count'] != len(chunks):
        raise ValueError('Vector index document count mismatch')
    chunk_scores = read_chunk_scores(client, index, chunks, query_vector)
    rankings = aggregate_chunk_scores(chunk_scores, temperature)
    if file_aggregation not in rankings:
        raise ValueError('Unknown file aggregation: ' + str(file_aggregation))
    chunk_lookup = {chunk['chunk_id']: chunk for chunk in chunks}
    # Retain the existing +1 public score offset, including for Max. The
    # nested aggregation rankings always expose S(f) itself with no offset.
    ranking = [{**item, 'score': item['score'] + 1.0, 'aggregated_similarity': item['score'],
                'cosine_similarity': item['best_chunk_similarity'],
                'best_chunk_start_line': chunk_lookup[item['best_chunk_id']]['start_line'],
                'best_chunk_end_line': chunk_lookup[item['best_chunk_id']]['end_line']}
               for item in rankings[file_aggregation]]
    paths = {doc['file_path'] for doc in documents}
    if len(ranking) != len(paths) or {r['file_path'] for r in ranking} != paths:
        raise ValueError('ES file collapse lost candidate files')
    # Verify EVERY ES chunk score locally. The three rankings above use only
    # the common ES scores; these NumPy calculations never change rankings.
    scores = (vectors @ query_vector) / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(query_vector))
    error = max(abs(row['cosine_similarity'] - float(score)) for row, score in zip(chunk_scores, scores))
    if error > 2e-5:
        raise ValueError(f'ES cosine scores differ from independent calculations: {error}')
    local_rows = [{**row, 'cosine_similarity': float(score)} for row, score in zip(chunk_scores, scores)]
    local_rankings = aggregate_chunk_scores(local_rows, temperature)
    checks = {}
    for method in AGGREGATIONS:
        expected = {item['file_path']: item['score'] for item in local_rankings[method]}
        file_error = max(abs(item['score'] - expected[item['file_path']]) for item in rankings[method])
        if file_error > 2e-5:
            raise ValueError('File aggregation differs from local cosine calculations: ' + method)
        checks[method] = {'max_file_score_error': file_error,
                         'top_10_matches_local_order': [r['file_path'] for r in rankings[method][:10]] ==
                                                      [r['file_path'] for r in local_rankings[method][:10]]}
    return {'index': index, 'index_reused': reused, 'index_fingerprint': fingerprint,
            'indexed_chunks': len(chunks), 'ranking': ranking,
            'file_aggregation': file_aggregation, 'aggregation_temperature': temperature,
            'aggregations': {method: {'ranking': rankings[method]} for method in AGGREGATIONS},
            'chunk_scores': chunk_scores, 'chunk_scores_sha256': digest(chunk_scores),
            'verification': {'max_cosine_error': error, 'all_chunk_scores_verified': True,
                             'all_file_scores_verified': True, 'aggregations': checks,
                             'same_chunk_scores_for_all_aggregations': True,
                             'top_10_matches_local_order': checks[file_aggregation]['top_10_matches_local_order']},
            'score_definition': 'score = 1 + S(f); aggregations.*.ranking exposes unshifted S(f)',
            'retrieval': 'Elasticsearch exact script_score for every chunk; complete file rankings'}


def read_chunk_scores(client, index, chunks, query_vector):
    """Scroll the full scored inventory, rejecting missing/extra/duplicate IDs."""
    from elasticsearch import helpers

    expected = {chunk['chunk_id']: chunk['file_path'] for chunk in chunks}
    if len(expected) != len(chunks):
        raise ValueError('Duplicate chunk IDs before scoring')
    scored = {}
    query = {'query': {'script_score': {'query': {'match_all': {}}, 'script': {
        'source': "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
        'params': {'query_vector': query_vector.tolist()}}}},
        '_source': ['file_path', 'chunk_id'], 'track_scores': True}
    for hit in helpers.scan(client, index=index, query=query, size=1000, scroll='5m',
                            request_timeout=120):
        row = hit['_source']
        chunk_id = row['chunk_id']
        if chunk_id not in expected or chunk_id in scored or row['file_path'] != expected[chunk_id]:
            raise ValueError('ES score inventory differs from the supplied chunks')
        scored[chunk_id] = {'file_path': row['file_path'], 'chunk_id': chunk_id,
                            'cosine_similarity': float(hit['_score']) - 1.0}
    if set(scored) != set(expected):
        raise ValueError('Elasticsearch did not score the complete chunk inventory')
    return [scored[chunk['chunk_id']] for chunk in chunks]


def run_semantic_batch(project, config):
    from elasticsearch import Elasticsearch

    base_path = Path(config['base_config'])
    base = json.loads((base_path if base_path.is_absolute() else project / base_path).read_text(encoding='utf-8'))
    validate_vector_config(config, base)
    # Check grammar availability/ABI before creating a run or fetching snapshots.
    for extension in base['extensions']:
        parser_for_language(language_for_path('source' + extension, config.get('language_overrides', {})))
    selected, selection = select_reports(project, config, base)
    with Elasticsearch(base['endpoint'], request_timeout=120) as client:
        info = dict(client.info())
        if info['cluster_name'] != base['cluster_name'] or not info['version']['number'].startswith('9.'):
            raise ValueError('The configured project Elasticsearch 9.x must be running')
        embedder = LocalEmbedder(project, config)
        pool = None
        try:
            if config.get('chunk_workers', 1) > 1:
                pool = create_pool(project, config, embedder.fingerprint)
            return evaluate(project, config, base, selected, selection, client, info, embedder, pool)
        finally:
            if pool is not None:
                pool.shutdown(wait=True, cancel_futures=True)
            embedder.close()


def evaluate(project, config, base, selected, selection, client, info, embedder, pool=None):
    run_id = base['dataset'] + '_' + base['split'] + '_' + datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
    folder = project / 'results/vector/batches' / run_id
    folder.mkdir(parents=True)
    chunk_cache = RoutingChunkCache(project, embedder.tokenizer, config, embedder.fingerprint)
    semantic_routes = active_routes(config)
    fusions = active_fusions(config)
    routes = semantic_routes + tuple(fusions)
    write_json(folder / 'selection.json', selection)
    write_json(folder / 'effective_config.json', {'semantic': config, 'candidate_config': base,
                                               'model': embedder.identity, 'server_version': info['version']['number'],
                                               'tree-sitter': version('tree-sitter'), 'tree-sitter-python': version('tree-sitter-python'),
                                               'grammar_versions': grammar_versions(),
                                               'implementation_sha256': {name: sha256_file(Path(__file__).with_name(name))
                                                                         for name in ('chunking.py', 'chunk_cache.py', 'parallel_chunking.py', 'ast_chunking.py', 'token_ast.py', 'dependency_context.py', 'languages.py', 'snapshot.py', 'bench4bl.py',
                                                                                      'embeddings.py', 'aggregation.py', 'vector.py')}})
    records = {route: [] for route in routes}
    context = {'run_id': run_id, 'dataset': base['dataset'], 'split': base['split'],
               'selection_sha256': selection['selection_sha256'], 'scope': 'Small semantic subset; not full benchmark performance',
               'fusion': {route: list(inputs) for route, inputs in fusions.items()},
               'chunking_routes': list(semantic_routes),
               'file_aggregation': config.get('file_aggregation', 'max'),
               'aggregation_temperature': config.get('aggregation_temperature', 0.1),
               'rrf_constant': config['rrf_constant']}
    checkpoint(folder, context, records, config['evaluation_k'], len(selected), embedder)
    for position, sample in enumerate(selected, 1):
        row = sample['report']
        print(f"[{position}/{len(selected)}] {row['repo_name']} #{row['issue_id']}", flush=True)
        started = time.monotonic()
        case = folder / 'cases' / (f'{position:03d}_' + row['repo_name'].replace('/', '__') + '-' + str(row['issue_id']))
        case.mkdir(parents=True)
        record = {'position': position, 'dataset_row': sample['dataset_row'], 'repository': row['repo_name'],
                  'issue_id': row['issue_id'], 'before_fix_sha': row.get('before_fix_sha')}
        outputs, route_records = {}, {}
        try:
            _, gold, body = prepare_report(row, {**base, 'repository': row['repo_name'], 'issue_id': row['issue_id']})
            source, manifest = get_snapshot(project, row['repo_name'], row['before_fix_sha'],
                                            **({'source_provider': base['source_provider'], 'source_dataset': base['dataset']}
                                               if base.get('source_provider') else {}))
            documents, skipped = collect_files(source, manifest, base)
            inventory = [{'file_path': doc['file_path'], 'sha256': doc['sha256']} for doc in documents]
            write_json(case / 'candidates.json', {'files': inventory, 'skipped': skipped})
            query_vector, query_info = embedder.query(body)
            write_json(case / 'query.json', {'body': body, **query_info, 'model_fingerprint': embedder.fingerprint})
            missing = sorted(set(gold) - {d['file_path'] for d in documents})
            prefetched = {}
            for route_index, route in enumerate(semantic_routes):
                route_started = time.monotonic()
                try:
                    if route == DEPENDENCY_ROUTE:
                        chunk_cache.prepare(documents)
                    cache_hits_before, cache_misses_before = chunk_cache.hits, chunk_cache.misses
                    chunks, parents = [], []
                    parallel_hits = parallel_misses = 0
                    if pool is not None and route != DEPENDENCY_ROUTE:
                        results = prefetched.pop(route, None)
                        if results is None:
                            results = chunk_documents(pool, documents, route)
                    else:
                        results = ((*(chunk_cache.get(document, route)), 0, 0) for document in documents)
                    for document, (file_chunks, file_parents, hits, misses) in zip(documents, results):
                        parallel_hits += hits
                        parallel_misses += misses
                        chunks.extend(file_chunks)
                        parents.extend({'file_path': document['file_path'], **parent} for parent in file_parents)
                    if pool is not None and config.get('chunk_prefetch', False) and route_index + 1 < len(semantic_routes):
                        next_route = semantic_routes[route_index + 1]
                        if next_route != DEPENDENCY_ROUTE:
                            prefetched[next_route] = chunk_documents(pool, documents, next_route)
                    print(f'  {route}: {len(chunks)} chunks / {len(documents)} files', flush=True)
                    write_json(case / (route + '_chunks.json'), {'chunks': chunks, 'original_blocks': parents,
                                                               'source_reconstruction_verified': True})
                    before_hits, before_misses = embedder.hits, embedder.misses
                    vectors = embedder.encode([chunk['embedding_text'] for chunk in chunks])
                    identity = {'repository': row['repo_name'], 'commit': row['before_fix_sha'], 'route': route,
                                'model_fingerprint': embedder.fingerprint, **chunk_cache.index_identity(documents, route)}
                    output = retrieve(client, documents, chunks, vectors, query_vector, identity,
                                      config.get('file_aggregation', 'max'), config.get('aggregation_temperature', 0.1))
                    if 'bench4bl_source' in manifest:
                        output['source_provenance'] = manifest['bench4bl_source']
                    scores_path = case / (route + '_chunk_scores.json')
                    write_json(scores_path, {'scores': output.pop('chunk_scores'),
                                            'sha256': output['chunk_scores_sha256'],
                                            'score_definition': 'raw cosine similarity (no +1 offset)',
                                            'index': output['index']})
                    output['chunk_scores_file'] = scores_path.relative_to(folder).as_posix()
                    output.update({'route': route, 'repository': row['repo_name'], 'issue_id': row['issue_id'],
                                   'before_fix_sha': row['before_fix_sha'], 'candidate_fingerprint': digest(inventory),
                                   'indexed_files': len(documents), 'ground_truth': gold, 'missing_gold': missing,
                                   'query_truncated': query_info['truncated'],
                                   'parse_error_files': sorted({c['file_path'] for c in chunks if c['parse_has_error']}),
                                   'prefix_truncated_chunks': sum(c['prefix_truncated'] for c in chunks),
                                   'oversized_statement_splits': sum(c['split_reason'] == 'oversized_statement_token_boundary' for c in chunks),
                                   'maximum_input_tokens': max(c['token_count'] for c in chunks),
                                   'maximum_prefix_tokens': max(c['prefix_tokens'] for c in chunks),
                                   'chunk_cache_hits': chunk_cache.hits - cache_hits_before + parallel_hits,
                                   'chunk_cache_misses': chunk_cache.misses - cache_misses_before + parallel_misses,
                                   'cache_hits': embedder.hits - before_hits, 'new_embeddings': embedder.misses - before_misses})
                    if route == 'blagent_ast':
                        output.update({'ast_max_chars': config.get('ast_max_chars', 1500), 'overlap_tokens': 0,
                                       'languages': sorted({c['language'] for c in chunks}),
                                       'ast_character_fallback_parents': sum('boundary' in p['kind'] or 'fallback' in p['kind'] or
                                                                            'oversized_leaf' in p['kind'] for p in parents)})
                    if route == DEPENDENCY_ROUTE:
                        output['dependency_statistics'] = {
                            'enriched_chunks': sum(bool(c['dependency_context']) for c in chunks),
                            'chunks_with_functions': sum(bool(c['dependency_functions']) for c in chunks),
                            'function_previews': sum(len(c['dependency_functions']) for c in chunks),
                            'field_previews': sum(len(c['dependency_fields']) for c in chunks),
                            'maximum_functions_per_chunk': max(len(c['dependency_functions']) for c in chunks),
                            'maximum_context_tokens': max(c['dependency_context_tokens'] for c in chunks),
                            'resolved_candidates': sum(c['dependency_resolved_candidates'] for c in chunks),
                            'unresolved_calls': sum(c['dependency_unresolved_calls'] for c in chunks)}
                    if route == 'fixed':
                        output.update({'fixed_chunk_tokens': config['fixed_chunk_tokens'], 'overlap_tokens': 0,
                                       'maximum_body_tokens': max(c['body_tokens'] for c in chunks),
                                       'boundary_adjusted_chunks': sum(c['fixed_window']['boundary_adjusted'] for c in chunks)})
                    output['metrics'] = ranking_metrics([item['file_path'] for item in output['ranking']], gold, config['evaluation_k'])
                    for aggregation in output['aggregations'].values():
                        aggregation['metrics'] = ranking_metrics([item['file_path'] for item in aggregation['ranking']],
                                                                 gold, config['evaluation_k'])
                    write_json(case / (route + '_result.json'), output)
                    outputs[route] = output
                    route_records[route] = {**record, 'status': 'ok', 'metrics': output['metrics'], 'missing_gold': missing,
                                            'indexed_files': len(documents), 'query_truncated': query_info['truncated'],
                                            'aggregation_metrics': {method: values['metrics'] for method, values in output['aggregations'].items()},
                                            'chunk_scores_sha256': output['chunk_scores_sha256'],
                                            'elapsed_seconds': round(time.monotonic() - route_started, 3),
                                            'result_file': (case / (route + '_result.json')).relative_to(folder).as_posix()}
                    print(f"  {route}: AP@10={output['metrics']['average_precision_at_k']:.4f}, RR@10={output['metrics']['reciprocal_rank_at_k']:.4f}", flush=True)
                except Exception as error:
                    route_records[route] = {**record, 'status': 'failed', 'error_type': type(error).__name__, 'error': str(error)}
                    write_json(case / (route + '_error.json'), {**route_records[route], 'traceback': traceback.format_exc()})
                    print(f'  {route} FAILED: {error}', flush=True)
            for fusion, inputs in fusions.items():
                if all(route in outputs for route in inputs):
                    ranking = rrf([outputs[route]['ranking'] for route in inputs], config['rrf_constant'])
                    metrics = ranking_metrics([item['file_path'] for item in ranking], gold, config['evaluation_k'])
                    output = {'route': fusion, **record, 'ranking': ranking, 'metrics': metrics, 'ground_truth': gold,
                              'missing_gold': missing, 'indexed_files': len(documents), 'candidate_fingerprint': digest(inventory),
                              'rrf_constant': config['rrf_constant'], 'weights': [1] * len(inputs), 'input_routes': list(inputs),
                              'query_truncated': query_info['truncated']}
                    write_json(case / (fusion + '_result.json'), output)
                    route_records[fusion] = {**record, 'status': 'ok', 'metrics': metrics, 'missing_gold': missing,
                                            'indexed_files': len(documents), 'query_truncated': query_info['truncated'],
                                            'result_file': (case / (fusion + '_result.json')).relative_to(folder).as_posix()}
                else:
                    route_records[fusion] = {**record, 'status': 'failed', 'error_type': 'DependencyError',
                                              'error': 'Required semantic routes failed: ' + ', '.join(route for route in inputs if route not in outputs)}
        except Exception as error:
            write_json(case / 'error.json', {**record, 'error': str(error), 'traceback': traceback.format_exc()})
            for route in routes:
                route_records.setdefault(route, {**record, 'status': 'failed', 'error_type': type(error).__name__, 'error': str(error)})
            print('  CASE FAILED: ' + str(error), flush=True)
        for route in routes:
            route_records[route].setdefault('elapsed_seconds', round(time.monotonic() - started, 3))
            records[route].append(route_records[route])
        summary = checkpoint(folder, context, records, config['evaluation_k'], len(selected), embedder)
    print(json.dumps({route: {'successful': values['successful_samples'], 'failed': values['failed_samples'],
                              'metrics': values['metrics']} for route, values in summary['routes'].items()}, indent=2), flush=True)
    print('Semantic results: ' + str(folder), flush=True)
    return folder, summary


def checkpoint(folder, context, records, k, selected_count, embedder):
    summary = {**context, 'routes': {}, 'embedding_cache': {'hits': embedder.hits, 'new_vectors': embedder.misses}}
    rows = []
    for route in records:
        values = aggregate_results(records[route], k, selected_count)
        summary['routes'][route] = values
        route_folder = folder / route
        route_folder.mkdir(exist_ok=True)
        write_checkpoints(route_folder, records[route], {**context, **values})
        rows.append({'route': route, 'selected_samples': selected_count, 'successful_samples': values['successful_samples'],
                     'failed_samples': values['failed_samples'], **(values['metrics'] or {})})
    if 'file_aggregation' in context:
        aggregation_rows = []
        sample_rows = []
        summary['aggregation_routes'] = {}
        for route in (r for r in records if r in RETRIEVAL_ROUTES):
            for method in AGGREGATIONS:
                paired = [{**record, 'metrics': record['aggregation_metrics'][method]} if record['status'] == 'ok'
                          else record for record in records[route]]
                values = aggregate_results(paired, k, selected_count)
                key = route + '/' + method
                summary['aggregation_routes'][key] = values
                for record in paired:
                    metrics = record.get('metrics') or {}
                    sample_rows.append({'chunking': route, 'aggregation': method,
                                        'position': record['position'], 'repository': record['repository'],
                                        'issue_id': record['issue_id'], 'status': record['status'],
                                        'AP@10': metrics.get('average_precision_at_k'),
                                        'RR@10': metrics.get('reciprocal_rank_at_k'),
                                        'Top-1': metrics.get('top_1'), 'Top-5': metrics.get('top_5'),
                                        'Top-10': metrics.get('top_10'), 'Recall@10': metrics.get('recall_at_k'),
                                        'chunk_scores_sha256': record.get('chunk_scores_sha256'),
                                        'error': record.get('error')})
                aggregation_rows.append({'chunking': route, 'aggregation': method,
                                         'temperature': context['aggregation_temperature'] if method == 'normalized_logsumexp' else '',
                                         'selected_samples': selected_count, 'successful_samples': values['successful_samples'],
                                         'failed_samples': values['failed_samples'], **(values['metrics'] or {})})
        write_json(folder / 'aggregation_summary.json', {'file_score_definition': 'unshifted cosine aggregation S(f)',
                   'same_chunk_scores_for_all_aggregations': True, 'selection_sha256': context['selection_sha256'],
                   'temperature': context['aggregation_temperature'], 'routes': summary['aggregation_routes']})
        write_json(folder / 'aggregation_samples.json', sample_rows)
        for name, data in [('aggregation_summary.csv', aggregation_rows), ('aggregation_samples.csv', sample_rows)]:
            with (folder / name).open('w', encoding='utf-8-sig', newline='') as handle:
                fields = list(dict.fromkeys(field for row in data for field in row))
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(data)
    write_json(folder / 'summary.json', summary)
    with (folder / 'summary.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        fields = list(dict.fromkeys(field for row in rows for field in row))
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return summary
