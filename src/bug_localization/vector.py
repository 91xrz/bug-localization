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
from .bm25 import collect_files, prepare_report, ranking_metrics, validate_config
from .chunking import PythonChunker
from .chunk_cache import ChunkCache
from .embeddings import LocalEmbedder, digest
from .snapshot import get_snapshot, sha256_file

RETRIEVAL_ROUTES = ('dynamic', 'method', 'fixed')
FUSIONS = {'rrf': ('dynamic', 'method'), 'rrf_three': RETRIEVAL_ROUTES}


def active_fusions(config):
    enabled = config.get('enable_fusion', True)
    if type(enabled) is not bool:
        raise ValueError('enable_fusion must be a boolean')
    return FUSIONS if enabled else {}


def validate_vector_config(config, base):
    validate_config(base)
    active_fusions(config)
    if base['extensions'] != ['.py']:
        raise ValueError('The initial semantic pipeline supports Python candidates only')
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


def retrieve(client, documents, chunks, vectors, query_vector, identity):
    """Score all chunks in ES; collapse by file BEFORE choosing the file top K."""
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
    if len(documents) > 10000:
        raise ValueError('Exact full-file ranking currently supports at most 10000 candidate files')
    response = client.search(index=index, size=len(documents), track_total_hits=True,
                             query={'script_score': {'query': {'match_all': {}}, 'script': {
                                 'source': "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                                 'params': {'query_vector': query_vector.tolist()}}}},
                             collapse={'field': 'file_path'}, source_includes=['file_path', 'chunk_id', 'record'],
                             sort=[{'_score': 'desc'}, {'file_path': 'asc'}, {'chunk_id': 'asc'}])
    if response.get('timed_out') or response['_shards']['failed'] or response['hits']['total']['value'] != len(chunks):
        raise ValueError('Elasticsearch did not score the complete chunk inventory')
    ranking = [{'rank': i, 'file_path': hit['_source']['file_path'], 'score': hit['_score'],
                'cosine_similarity': hit['_score'] - 1, 'best_chunk_id': hit['_source']['chunk_id'],
                'best_chunk_start_line': hit['_source']['record']['start_line'],
                'best_chunk_end_line': hit['_source']['record']['end_line']}
               for i, hit in enumerate(response['hits']['hits'], 1)]
    paths = {doc['file_path'] for doc in documents}
    if len(ranking) != len(paths) or {r['file_path'] for r in ranking} != paths:
        raise ValueError('ES file collapse lost candidate files')
    # Independent local cosine calculations verify all ES file scores, not just
    # relevant files. This verification has no effect on the returned ranking.
    scores = (vectors @ query_vector) / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(query_vector))
    expected = {path: -float('inf') for path in paths}
    for chunk, score in zip(chunks, scores):
        expected[chunk['file_path']] = max(expected[chunk['file_path']], float(score))
    error = max(abs(item['cosine_similarity'] - expected[item['file_path']]) for item in ranking)
    if error > 2e-5:
        raise ValueError(f'ES cosine scores differ from independent calculations: {error}')
    local_order = sorted(paths, key=lambda path: (-expected[path], path))
    return {'index': index, 'index_reused': reused, 'index_fingerprint': fingerprint,
            'indexed_chunks': len(chunks), 'ranking': ranking,
            'verification': {'max_cosine_error': error, 'all_file_scores_verified': True,
                             'top_10_matches_local_order': [r['file_path'] for r in ranking[:10]] == local_order[:10]},
            'score_definition': '1 + cosine_similarity; file score is maximum over all its chunks',
            'retrieval': 'Elasticsearch exact script_score with file_path collapse; complete file ranking'}


def run_semantic_batch(project, config):
    from elasticsearch import Elasticsearch

    base_path = Path(config['base_config'])
    base = json.loads((base_path if base_path.is_absolute() else project / base_path).read_text(encoding='utf-8'))
    validate_vector_config(config, base)
    selected, selection = select_reports(project, config, base)
    with Elasticsearch(base['endpoint'], request_timeout=120) as client:
        info = dict(client.info())
        if info['cluster_name'] != base['cluster_name'] or not info['version']['number'].startswith('9.'):
            raise ValueError('The configured project Elasticsearch 9.x must be running')
        embedder = LocalEmbedder(project, config)
        try:
            return evaluate(project, config, base, selected, selection, client, info, embedder)
        finally:
            embedder.close()


def evaluate(project, config, base, selected, selection, client, info, embedder):
    run_id = base['dataset'] + '_' + base['split'] + '_' + datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
    folder = project / 'results/vector/batches' / run_id
    folder.mkdir(parents=True)
    chunker = PythonChunker(embedder.tokenizer, config)
    chunk_cache = ChunkCache(project, chunker, embedder.fingerprint)
    fusions = active_fusions(config)
    routes = RETRIEVAL_ROUTES + tuple(fusions)
    write_json(folder / 'selection.json', selection)
    write_json(folder / 'effective_config.json', {'semantic': config, 'candidate_config': base,
                                               'model': embedder.identity, 'server_version': info['version']['number'],
                                               'tree-sitter': version('tree-sitter'), 'tree-sitter-python': version('tree-sitter-python'),
                                               'implementation_sha256': {name: sha256_file(Path(__file__).with_name(name))
                                                                         for name in ('chunking.py', 'chunk_cache.py', 'embeddings.py', 'vector.py')}})
    records = {route: [] for route in routes}
    context = {'run_id': run_id, 'dataset': base['dataset'], 'split': base['split'],
               'selection_sha256': selection['selection_sha256'], 'scope': 'Small semantic subset; not full benchmark performance',
               'fusion': {route: list(inputs) for route, inputs in fusions.items()},
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
            source, manifest = get_snapshot(project, row['repo_name'], row['before_fix_sha'])
            documents, skipped = collect_files(source, manifest, base)
            inventory = [{'file_path': doc['file_path'], 'sha256': doc['sha256']} for doc in documents]
            write_json(case / 'candidates.json', {'files': inventory, 'skipped': skipped})
            query_vector, query_info = embedder.query(body)
            write_json(case / 'query.json', {'body': body, **query_info, 'model_fingerprint': embedder.fingerprint})
            missing = sorted(set(gold) - {d['file_path'] for d in documents})
            for route in RETRIEVAL_ROUTES:
                route_started = time.monotonic()
                try:
                    cache_hits_before, cache_misses_before = chunk_cache.hits, chunk_cache.misses
                    chunks, parents = [], []
                    for document in documents:
                        file_chunks, file_parents = chunk_cache.get(document, route)
                        chunks.extend(file_chunks)
                        parents.extend({'file_path': document['file_path'], **parent} for parent in file_parents)
                    print(f'  {route}: {len(chunks)} chunks / {len(documents)} files', flush=True)
                    write_json(case / (route + '_chunks.json'), {'chunks': chunks, 'original_blocks': parents,
                                                               'source_reconstruction_verified': True})
                    before_hits, before_misses = embedder.hits, embedder.misses
                    vectors = embedder.encode([chunk['embedding_text'] for chunk in chunks])
                    identity = {'repository': row['repo_name'], 'commit': row['before_fix_sha'], 'route': route,
                                'model_fingerprint': embedder.fingerprint, 'chunker_sha256': sha256_file(Path(__file__).with_name('chunking.py'))}
                    output = retrieve(client, documents, chunks, vectors, query_vector, identity)
                    output.update({'route': route, 'repository': row['repo_name'], 'issue_id': row['issue_id'],
                                   'before_fix_sha': row['before_fix_sha'], 'candidate_fingerprint': digest(inventory),
                                   'indexed_files': len(documents), 'ground_truth': gold, 'missing_gold': missing,
                                   'query_truncated': query_info['truncated'],
                                   'parse_error_files': sorted({c['file_path'] for c in chunks if c['parse_has_error']}),
                                   'prefix_truncated_chunks': sum(c['prefix_truncated'] for c in chunks),
                                   'oversized_statement_splits': sum(c['split_reason'] == 'oversized_statement_token_boundary' for c in chunks),
                                   'maximum_input_tokens': max(c['token_count'] for c in chunks),
                                   'maximum_prefix_tokens': max(c['prefix_tokens'] for c in chunks),
                                   'chunk_cache_hits': chunk_cache.hits - cache_hits_before,
                                   'chunk_cache_misses': chunk_cache.misses - cache_misses_before,
                                   'cache_hits': embedder.hits - before_hits, 'new_embeddings': embedder.misses - before_misses})
                    if route == 'fixed':
                        output.update({'fixed_chunk_tokens': config['fixed_chunk_tokens'], 'overlap_tokens': 0,
                                       'maximum_body_tokens': max(c['body_tokens'] for c in chunks),
                                       'boundary_adjusted_chunks': sum(c['fixed_window']['boundary_adjusted'] for c in chunks)})
                    output['metrics'] = ranking_metrics([item['file_path'] for item in output['ranking']], gold, config['evaluation_k'])
                    write_json(case / (route + '_result.json'), output)
                    outputs[route] = output
                    route_records[route] = {**record, 'status': 'ok', 'metrics': output['metrics'], 'missing_gold': missing,
                                            'indexed_files': len(documents), 'query_truncated': query_info['truncated'],
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
    write_json(folder / 'summary.json', summary)
    with (folder / 'summary.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        fields = list(dict.fromkeys(field for row in rows for field in row))
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return summary
