"""File-level BM25/semantic RRF over verified saved runs; no model inference."""

import csv
from datetime import datetime, timedelta, timezone
import math
from pathlib import Path
import re

from .batch import aggregate_results, write_checkpoints, write_json
from .bm25 import ranking_metrics
from .comparison import paired_records, read_json
from .embeddings import digest
from .snapshot import sha256_file
from .vector import rrf


def ranked_paths(rows):
    paths = [row['file_path'] for row in rows]
    if len(set(paths)) != len(paths) or [row['rank'] for row in rows] != list(range(1, len(rows) + 1)):
        raise ValueError('Input ranking must have unique files and consecutive one-based ranks')
    return paths


def full_lexical_ranking(client, saved):
    """Read all matching files from the existing owned index, preserving its query.

    Nonmatching files have no lexical rank and receive zero RRF contribution.
    Never invent a ranking for zero-match files or silently fuse a top-10 prefix.
    """
    expected = {v['file_path']: v['sha256'] for v in saved['candidate_files']}
    if len(expected) != saved['indexed_files']:
        raise ValueError('BM25 candidate inventory is incomplete or duplicated')
    old_paths = ranked_paths(saved['results'])
    total = saved['total_matching_files']
    if len(old_paths) == total:
        if not set(old_paths).issubset(expected):
            raise ValueError('Saved BM25 ranking contains an unknown candidate')
        return saved['results'], {'source': 'complete_saved_ranking', 'matching_files': total}
    if len(old_paths) > total or client is None:
        raise ValueError('BM25 ranking is truncated; Elasticsearch is required to recover all matching files')
    from elasticsearch import helpers

    index = saved['index']
    ownership = {'project': 'bug-localization', 'repository': saved['repository'],
                 'commit': saved['before_fix_sha'], 'fingerprint': saved['index_fingerprint']}
    if client.indices.get_mapping(index=index)[index]['mappings'].get('_meta') != ownership:
        raise ValueError('BM25 index ownership differs from saved result')
    if client.count(index=index)['count'] != len(expected):
        raise ValueError('BM25 index document count differs from saved inventory')
    inventory = list(helpers.scan(client, index=index, query={'query': {'match_all': {}},
                         '_source': ['file_path', 'sha256']}, size=1000, request_timeout=120))
    actual = {v['_source']['file_path']: v['_source']['sha256'] for v in inventory}
    if len(inventory) != len(expected) or actual != expected:
        raise ValueError('BM25 index source inventory differs from saved run')
    hits = helpers.scan(client, index=index, query={'query': saved['query'], '_source': ['file_path'],
                        'sort': [{'_score': 'desc'}, {'file_path': 'asc'}]}, preserve_order=True,
                        size=1000, request_timeout=120)
    ranking = [{'rank': i, 'file_path': h['_source']['file_path'], 'score': float(h['_score'])}
               for i, h in enumerate(hits, 1)]
    paths = ranked_paths(ranking)
    if len(paths) != total or not set(paths).issubset(expected) or paths[:len(old_paths)] != old_paths:
        raise ValueError('Expanded BM25 result differs from saved total, candidates or leading ranks')
    if any(not math.isfinite(v['score']) for v in ranking):
        raise ValueError('Nonfinite BM25 score')
    if any(not math.isclose(a['score'], b['score'], rel_tol=2e-5, abs_tol=2e-5)
           for a, b in zip(saved['results'], ranking)):
        raise ValueError('Expanded BM25 leading scores differ from saved run')
    return ranking, {'source': 'existing_index_read_only', 'index': index,
                     'saved_depth': len(old_paths), 'matching_files': total,
                     'ownership_inventory_and_leading_ranks_verified': True}


def verify_case(lexical, semantic, candidates, query):
    if lexical['candidate_files'] != candidates or query['body'] != lexical['query_text']:
        raise ValueError('BM25 and semantic candidates or raw reports differ')
    if any(lexical[k] != semantic[k] for k in ('repository', 'issue_id', 'before_fix_sha', 'indexed_files')):
        raise ValueError('BM25 and semantic sample or snapshot differs')
    if (set(lexical['ground_truth']) != set(semantic['ground_truth']) or
            lexical['ground_truth_missing_from_candidates'] != semantic['missing_gold']):
        raise ValueError('BM25 and semantic labels or candidate coverage differ')
    paths = ranked_paths(semantic['ranking'])
    if set(paths) != {v['file_path'] for v in candidates} or semantic['candidate_fingerprint'] != digest(candidates):
        raise ValueError('Semantic ranking is incomplete or has a different inventory')


def fuse_files(lexical, semantic, constant):
    """Reuse the existing equal-weight RRF and expose each input contribution."""
    if type(constant) is not int or not 1 <= constant <= 10000:
        raise ValueError('RRF constant must be an integer in [1, 10000]')
    ranked_paths(lexical)
    ranked_paths(semantic)
    bm, sem = ({v['file_path']: v for v in ranking} for ranking in (lexical, semantic))
    return [{**row, 'bm25_rank': bm.get(row['file_path'], {}).get('rank'),
             'semantic_rank': sem.get(row['file_path'], {}).get('rank'),
             'bm25_score': bm.get(row['file_path'], {}).get('score'),
             'semantic_similarity': sem.get(row['file_path'], {}).get('aggregated_similarity'),
             'bm25_contribution': 1 / (constant + bm[row['file_path']]['rank']) if row['file_path'] in bm else 0.0,
             'semantic_contribution': 1 / (constant + sem[row['file_path']]['rank']) if row['file_path'] in sem else 0.0}
            for row in rrf([lexical, semantic], constant)]


def run_hybrid(project, lexical_folder, semantic_folders, route='blagent_ast', constant=60):
    """Fuse one or more paired semantic runs against the same full BM25 ranks."""
    from elasticsearch import Elasticsearch

    if type(constant) is not int or not 1 <= constant <= 10000:
        raise ValueError('RRF constant must be an integer in [1, 10000]')
    labels = list(semantic_folders)
    names = ['bm25'] + labels + ['rrf_' + label for label in labels]
    if (not labels or len(names) != len(set(names)) or
            any(not re.fullmatch(r'[a-z][a-z0-9_]{0,31}', label) for label in labels)):
        raise ValueError('Use distinct semantic labels without reserved or colliding route names')
    selection = read_json(lexical_folder / 'selection.json')
    lexical_summary = read_json(lexical_folder / 'summary.json')
    lexical_config = read_json(lexical_folder / 'effective_config.json')['base']
    source_records = {'bm25': read_json(lexical_folder / 'samples.json')}
    configs = {}
    for label, source in semantic_folders.items():
        semantic = read_json(source / 'summary.json')
        if (semantic['selection_sha256'] != lexical_summary['selection_sha256'] or
                read_json(source / 'selection.json')['samples'] != selection['samples']):
            raise ValueError('Fusion requires identical pinned selections: ' + label)
        configs[label] = read_json(source / 'effective_config.json')
        if configs[label]['candidate_config'] != lexical_config:
            raise ValueError('Fusion input candidate configurations differ')
        source_records[label] = read_json(source / route / 'samples.json')
        if semantic['routes'][route]['cutoff'] != 10:
            raise ValueError('This fusion experiment requires evaluation_k=10')
    paired_records(source_records['bm25'], source_records, labels)
    n = len(selection['samples'])
    if (lexical_summary['cutoff'] != 10 or
            any(len(data) != n or any(r['status'] != 'ok' for r in data) for data in source_records.values())):
        raise ValueError('Finish all input samples at @10 before running this paired fusion')
    now = datetime.now(timezone(timedelta(hours=8)))
    folder = project / 'results/hybrid' / now.strftime('%Y%m%d_%H%M%S_%f')
    folder.mkdir(parents=True)
    write_json(folder / 'selection.json', selection)
    write_json(folder / 'effective_config.json', {
        'rrf_constant': constant, 'weights': [1, 1], 'semantic_source_route': route,
        'evaluation_k': 10, 'lexical_run': str(lexical_folder),
        'semantic_runs': {label: str(p) for label, p in semantic_folders.items()},
        'semantic_configs': configs, 'candidate_config': lexical_config,
        'source_summaries_sha256': {'bm25': sha256_file(lexical_folder / 'summary.json'),
                                   **{label: sha256_file(p / 'summary.json') for label, p in semantic_folders.items()}},
        'implementation_sha256': {name: sha256_file(Path(__file__).with_name(name))
                                  for name in ('hybrid.py', 'vector.py', 'bm25.py', 'batch.py')}})
    write_json(folder / 'status.json', {'status': 'running'})
    records, all_ranks, audits = {name: [] for name in names}, [], []
    try:
        with Elasticsearch(lexical_config['endpoint'], request_timeout=120) as client:
            info = client.info()
            if info['cluster_name'] != lexical_config['cluster_name'] or not info['version']['number'].startswith('9.'):
                raise ValueError('Unexpected Elasticsearch cluster or major version')
            for i, original in enumerate(source_records['bm25']):
                case = folder / 'cases' / (f'{i + 1:03d}_' + original['repository'].replace('/', '__') + '-' + str(original['issue_id']))
                case.mkdir(parents=True)
                bm_path = lexical_folder / original['result_file']
                bm = read_json(bm_path)
                rankings = {}
                rankings['bm25'], audit = full_lexical_ranking(client, bm)
                if ranking_metrics(ranked_paths(rankings['bm25']), bm['ground_truth'], 10) != original['metrics']:
                    raise ValueError('Expanded BM25 @10 metrics changed')
                audit.update({'position': i + 1, 'issue_id': original['issue_id'],
                              'source_result_sha256': sha256_file(bm_path), 'semantic_inputs': {}})
                for label, source in semantic_folders.items():
                    semantic_path = source / source_records[label][i]['result_file']
                    semantic = read_json(semantic_path)
                    verify_case(bm, semantic, read_json(semantic_path.parent / 'candidates.json')['files'],
                                read_json(semantic_path.parent / 'query.json'))
                    if semantic['metrics'] != source_records[label][i]['metrics']:
                        raise ValueError('Saved semantic metrics disagree')
                    rankings[label] = semantic['ranking']
                    rankings['rrf_' + label] = fuse_files(rankings['bm25'], rankings[label], constant)
                    audit['semantic_inputs'][label] = {'source_result_sha256': sha256_file(semantic_path),
                                                       'candidate_and_query_verified': True}
                write_json(case / 'inputs.json', {'query_text': bm['query_text'], 'bm25_query': bm['query'],
                           'candidate_files': bm['candidate_files'], 'ground_truth': bm['ground_truth'],
                           'lexical_audit': audit})
                for name, ranking in rankings.items():
                    ranking = [{**v, 'is_ground_truth': v['file_path'] in bm['ground_truth']} for v in ranking]
                    metrics = ranking_metrics(ranked_paths(ranking), bm['ground_truth'], 10)
                    if name in labels and metrics != source_records[name][i]['metrics']:
                        raise ValueError('Semantic baseline metrics changed during export')
                    result_file = case / (name + '_result.json')
                    write_json(result_file, {'route': name, 'ranking': ranking, 'metrics': metrics,
                               'ground_truth': bm['ground_truth'], 'missing_gold': bm['ground_truth_missing_from_candidates'],
                               'repository': bm['repository'], 'issue_id': bm['issue_id'],
                               'before_fix_sha': bm['before_fix_sha'], 'rrf_constant': constant if name.startswith('rrf_') else None})
                    record = {k: original[k] for k in ('position', 'dataset_row', 'repository', 'issue_id', 'before_fix_sha')}
                    records[name].append({**record, 'status': 'ok', 'metrics': metrics, 'indexed_files': bm['indexed_files'],
                                          'missing_gold': bm['ground_truth_missing_from_candidates'],
                                          'result_file': result_file.relative_to(folder).as_posix()})
                    for row in ranking:
                        all_ranks.append({'position': i + 1, 'issue_id': bm['issue_id'], 'route': name,
                                          **{k: row.get(k) for k in ('rank', 'file_path', 'score', 'bm25_rank', 'semantic_rank',
                                             'bm25_score', 'semantic_similarity', 'bm25_contribution', 'semantic_contribution', 'is_ground_truth')}})
                audits.append(audit)
                print(f"[{i + 1}/{n}] {bm['repository']} #{bm['issue_id']}: {len(rankings['bm25'])} BM25 matching files; paired RRF saved", flush=True)
                write_json(folder / 'status.json', {'status': 'running', 'processed_samples': i + 1, 'selected_samples': n})
        summaries = {name: aggregate_results(data, 10) for name, data in records.items()}
        summary = {'status': 'completed', 'created_at': now.isoformat(), 'selected_samples': n, 'evaluation_k': 10,
                   'selection_sha256': selection['selection_sha256'], 'rrf_constant': constant, 'weights': [1, 1],
                   'input_depth': 'All BM25 matching files and complete semantic file rankings',
                   'missing_input_rank': 'Zero contribution; no fabricated tail ranks',
                   'scope': 'Exploratory single-project reports; not a full or held-out benchmark', 'routes': summaries}
        for name in records:
            sub = folder / name
            sub.mkdir()
            write_checkpoints(sub, records[name], summaries[name])
        write_json(folder / 'verification.json', {'all_samples_paired_and_verified': True,
                   'models_encoded': False, 'indexes_rebuilt': False, 'cases': audits})
        write_json(folder / 'summary.json', summary)
        for filename, rows in [('summary.csv', [{'route': name, **value['metrics']} for name, value in summaries.items()]),
                               ('rankings.csv', all_ranks)]:
            with (folder / filename).open('w', encoding='utf-8-sig', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        write_json(folder / 'status.json', {'status': 'completed', 'processed_samples': n, 'selected_samples': n})
        print('Hybrid results: ' + str(folder), flush=True)
        return folder, summary
    except Exception as error:
        write_json(folder / 'status.json', {'status': 'failed', 'error_type': type(error).__name__, 'error': str(error)})
        raise
