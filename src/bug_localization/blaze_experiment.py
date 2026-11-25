"""Run the BLAZE released encoder against the project's frozen evaluation inputs."""

from datetime import datetime, timedelta, timezone
from importlib.metadata import version
import json
from pathlib import Path
import time
import traceback

from .batch import aggregate_results, write_checkpoints, write_json
from .blaze import (LANGUAGES, cached_vectors, chunk_rrf, dense_rank, digest,
                    lexical_rank, unique_dense_files)
from .blaze_model import BlazeEncoder
from .bm25 import collect_files, prepare_report, ranking_metrics, validate_config as validate_base
from .snapshot import get_snapshot, sha256_file
from .vector import select_reports

ROUTES = ('blaze_bm25', 'blaze_dense', 'blaze_rrf')


def validate_config(config, base):
    validate_base(base)
    if base['top_k'] < 10:
        raise ValueError('BLAZE evaluation requires top_k >= 10')
    if any(extension not in LANGUAGES for extension in base['extensions']):
        raise ValueError('BLAZE release supports only its Python/Java/C++/JavaScript extension map')
    for field, expected in [('dense_depth', 1000), ('lexical_depth', 1000), ('rrf_constant', 60),
                            ('source_max_tokens', 512), ('report_max_tokens', 1024)]:
        if type(config.get(field)) is not int or config[field] != expected:
            raise ValueError(f'{field} must be {expected} for this BLAZE release protocol')
    if base['top_k'] > min(config['dense_depth'], config['lexical_depth']):
        raise ValueError('Evaluation cutoff exceeds retrieval depth')
    for field, upper in [('embedding_batch_size', 128), ('cpu_threads', 64)]:
        if type(config.get(field)) is not int or not 1 <= config[field] <= upper:
            raise ValueError(field + ' is outside its supported range')
    if config.get('device') != 'cpu' and not re_device(config.get('device')):
        raise ValueError('device must be cpu, cuda, or cuda:N')
    if not isinstance(config.get('model_path'), str) or not config['model_path']:
        raise ValueError('model_path is required')


def re_device(device):
    import re
    return isinstance(device, str) and re.fullmatch(r'cuda(?::\d+)?', device)


def evaluate_case(project, base, config, row, encoder, client):
    effective = {**base, 'repository': row['repo_name'], 'issue_id': row['issue_id']}
    _, gold, query = prepare_report(row, effective)
    started = time.monotonic()
    source, manifest = get_snapshot(project, row['repo_name'], row['before_fix_sha'],
                                    **({'source_provider': base['source_provider'], 'source_dataset': base['dataset']}
                                       if base.get('source_provider') else {}))
    documents, skipped = collect_files(source, manifest, base)
    candidates = [{'file_path': doc['file_path'], 'sha256': doc['sha256']} for doc in documents]
    candidate_paths = {doc['file_path'] for doc in documents}
    source_seconds = time.monotonic() - started
    started = time.monotonic()
    lexical, lexical_audit = lexical_rank(client, documents, query, config['lexical_depth'])
    lexical_seconds = time.monotonic() - started
    result = {
        'repository': row['repo_name'], 'issue_id': row['issue_id'], 'before_fix_sha': row['before_fix_sha'],
        'query': query, 'query_fields': base['query_fields'], 'query_sha256': digest(query),
        'gold_files': gold, 'missing_gold': sorted(set(gold) - candidate_paths),
        'source_manifest': {key: value for key, value in manifest.items() if key != 'files'},
        'candidate_fingerprint': digest(candidates), 'candidates': candidates, 'skipped_files': skipped,
        'indexed_files': len(documents), 'chunk_count': None, 'lexical': lexical_audit,
        'embedding_cache': None, 'query_token_audit': None,
        'chunk_metadata': [], 'dense_chunk_ranking': [],
        'routes': {'blaze_bm25': {'ranking': lexical, 'metrics': ranking_metrics(
            [item['file_path'] for item in lexical], gold, base['top_k'])}},
        'route_errors': {},
        'timing_seconds': {'source': source_seconds, 'lexical_index_and_search': lexical_seconds,
                           'chunking_and_source_embeddings': None,
                           'query_embedding': None, 'dense_search_and_fusion': None},
    }
    stage = 'chunking_and_source_embeddings'
    started = time.monotonic()
    try:
        chunks, vectors, cache_audit = cached_vectors(project, documents, encoder, config['source_max_tokens'])
        result.update(chunk_count=len(chunks), embedding_cache=cache_audit,
                      chunk_metadata=[{k: v for k, v in chunk.items() if k != 'text'} for chunk in chunks])
        result['timing_seconds'][stage] = time.monotonic() - started
        stage, started = 'query_embedding', time.monotonic()
        query_vectors, query_audit = encoder.encode([query], config['report_max_tokens'])
        result['query_token_audit'] = query_audit[0]
        result['timing_seconds'][stage] = time.monotonic() - started
        stage, started = 'dense_search_and_fusion', time.monotonic()
        dense = dense_rank(chunks, vectors, query_vectors[0], config['dense_depth'])
        result['dense_chunk_ranking'] = dense
        dense_files = unique_dense_files(dense)
        result['routes']['blaze_dense'] = {'ranking': dense_files, 'metrics': ranking_metrics(
            [item['file_path'] for item in dense_files], gold, base['top_k'])}
    except Exception as error:
        result['timing_seconds'][stage] = time.monotonic() - started
        failure = {'error_type': type(error).__name__, 'error': str(error),
                   'stage': stage, 'traceback': traceback.format_exc()}
        result['route_errors']['blaze_dense'] = failure
        result['route_errors']['blaze_rrf'] = {**failure, 'blocked_by': 'blaze_dense'}
        return result
    try:
        fused = chunk_rrf(dense, lexical, config['rrf_constant'])
        result['routes']['blaze_rrf'] = {'ranking': fused, 'metrics': ranking_metrics(
            [item['file_path'] for item in fused], gold, base['top_k'])}
    except Exception as error:
        result['route_errors']['blaze_rrf'] = {
            'error_type': type(error).__name__, 'error': str(error),
            'stage': 'fusion', 'traceback': traceback.format_exc()}
    result['timing_seconds']['dense_search_and_fusion'] = time.monotonic() - started
    return result


def run_batch(project, config, output_directory=None):
    from elasticsearch import Elasticsearch

    base = json.loads((project / config['base_config']).read_text(encoding='utf-8'))
    validate_config(config, base)
    selected, selection = select_reports(project, config, base,
                                         max_samples=5000 if config.get('selection_manifest') else 50)
    with Elasticsearch(base['endpoint'], request_timeout=120) as client:
        info = dict(client.info())
        if info['cluster_name'] != base['cluster_name'] or not info['version']['number'].startswith('9.'):
            raise ValueError('Expected the configured local Elasticsearch 9.x cluster')
        load_started = time.monotonic()
        encoder = BlazeEncoder(project / config['model_path'], device=config['device'],
                               batch_size=config['embedding_batch_size'], cpu_threads=config['cpu_threads'])
        model_load_seconds = time.monotonic() - load_started
        now = datetime.now(timezone(timedelta(hours=8)))
        folder = Path(output_directory) if output_directory else project / 'results/blaze' / now.strftime('%Y%m%d_%H%M%S_%f')
        if not folder.is_absolute():
            folder = project / folder
        folder.mkdir(parents=True, exist_ok=False)
        context = {'created_at': now.isoformat(), 'dataset': base['dataset'], 'split': base['split'],
                   'selection_sha256': selection['selection_sha256'],
                   'dataset_metadata_sha256': selection['dataset_metadata_sha256'],
                   'variant': encoder.identity['variant'], 'model': encoder.identity,
                   'model_load_seconds': model_load_seconds, 'server_version': info['version']['number'],
                   'dependencies': {name: version(name) for name in
                                    ('torch', 'transformers', 'tokenizers', 'numpy', 'datasets', 'elasticsearch', 'safetensors')},
                   'implementation_sha256': {name: sha256_file(Path(__file__).with_name(name)) for name in
                        ('blaze.py', 'blaze_model.py', 'blaze_experiment.py', 'bm25.py', 'batch.py',
                         'vector.py', 'snapshot.py', 'chunking.py', 'languages.py')},
                   'protocol': {'dense_search': 'exact float32 cosine, stable chunk order for ties',
                                'fusion': 'all top-1000 chunk contributions plus top-1000 lexical files; k=60',
                                'dense_file_control': 'first chunk occurrence per file after top-1000 truncation',
                                'lexical_query': 'full allowed report text, no max_query_terms truncation',
                                'bm25_parameters': 'Elasticsearch default k1=1.2, b=0.75; base.bm25 is not used',
                                'failure_policy': 'Preserve completed routes; dense failure blocks dense/RRF, fusion failure blocks only RRF; shared input or lexical failure blocks all routes',
                                'scope': 'shared-project protocol adaptation; not original paper score reproduction'}}
        write_json(folder / 'selection.json', selection)
        write_json(folder / 'effective_config.json', {'base': base, 'blaze': config})
        write_json(folder / 'provenance.json', context)
        records = {route: [] for route in ROUTES}
        for route in ROUTES:
            (folder / route).mkdir()

        def checkpoint():
            summaries = {}
            for route in ROUTES:
                summaries[route] = aggregate_results(records[route], base['top_k'], len(selected))
                write_checkpoints(folder / route, records[route], summaries[route])
            summary = {**context, 'status': summaries['blaze_rrf']['status'], 'routes': summaries}
            write_json(folder / 'summary.json', summary)
            return summary

        checkpoint()
        for position, sample in enumerate(selected, 1):
            row = sample['report']
            common = {'position': position, 'dataset_row': sample['dataset_row'], 'repository': row['repo_name'],
                      'issue_id': row['issue_id'], 'before_fix_sha': row.get('before_fix_sha')}
            print(f"[{position}/{len(selected)}] BLAZE {row['repo_name']} #{row['issue_id']}", flush=True)
            case = folder / 'cases' / f'{position:04d}'
            case.mkdir(parents=True)
            started = time.monotonic()
            try:
                result = evaluate_case(project, base, config, row, encoder, client)
                write_json(case / 'result.json', result)
                errors = result.get('route_errors', {})
                if errors:
                    write_json(case / 'route_errors.json', errors)
                route_records, messages = {}, []
                for route in ROUTES:
                    record = {**common, 'elapsed_seconds': time.monotonic() - started,
                              'result_file': '../' + (case / 'result.json').relative_to(folder).as_posix()}
                    if route in errors:
                        error = errors[route]
                        record.update(status='failed', error_type=error['error_type'], error=error['error'],
                                      error_stage=error['stage'],
                                      error_file='../' + (case / 'route_errors.json').relative_to(folder).as_posix())
                        messages.append(f"{route} failed: {error['error_type']}: {error['error']}")
                    else:
                        record.update(status='ok', metrics=result['routes'][route]['metrics'],
                                      missing_gold=result['missing_gold'], indexed_files=result['indexed_files'],
                                      query_truncated=False if route == 'blaze_bm25' else result['query_token_audit']['truncated'])
                        messages.append(f"{route} AP@{base['top_k']}={record['metrics']['average_precision_at_k']:.4f}")
                    route_records[route] = record
                print('  ' + ', '.join(messages), flush=True)
            except Exception as error:
                failure = {**common, 'status': 'failed', 'error_type': type(error).__name__, 'error': str(error),
                           'elapsed_seconds': time.monotonic() - started}
                write_json(case / 'error.json', {**failure, 'traceback': traceback.format_exc()})
                route_records = {route: dict(failure) for route in ROUTES}
                print(f'  Failed: {type(error).__name__}: {error}', flush=True)
            for route in ROUTES:
                records[route].append(route_records[route])
            summary = checkpoint()
        print('BLAZE results:', folder, flush=True)
        return folder, summary
