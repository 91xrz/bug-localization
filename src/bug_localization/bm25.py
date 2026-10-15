"""File-level Elasticsearch BM25 indexing and retrieval for a pinned snapshot."""

import ast
import hashlib
import json
from pathlib import Path

from .snapshot import get_snapshot


def validate_config(config):
    required = {'endpoint', 'cluster_name', 'dataset', 'split', 'repository', 'issue_id',
                'query_fields', 'extensions', 'exclude_directories', 'max_file_bytes',
                'max_query_terms', 'top_k', 'bm25'}
    missing = required - config.keys()
    if missing:
        raise ValueError('Missing configuration fields: ' + ', '.join(sorted(missing)))
    for name in ('dataset', 'split'):
        if not isinstance(config[name], str) or not config[name] or config[name] in ('.', '..') or any(c in config[name] for c in '/\\:'):
            raise ValueError(name + ' must be a single local directory name')
    for name, limit in (('top_k', 10000), ('max_query_terms', 512), ('max_file_bytes', 100_000_000)):
        if type(config[name]) is not int or not 1 <= config[name] <= limit:
            raise ValueError(f'{name} must be an integer between 1 and {limit}')
    fields = config['query_fields']
    if not isinstance(fields, list) or not fields or any(field not in ('title', 'body') for field in fields):
        raise ValueError('query_fields must be a non-empty list containing title/body only')
    if config['dataset'] == 'swe_bench_formatted' and 'title' in fields:
        raise ValueError('Use body only for this SWE-bench formatted version; title may contain the solution PR title')
    extensions = config['extensions']
    if not isinstance(extensions, list) or not extensions or any(not isinstance(e, str) or not e.startswith('.') or e != e.lower() for e in extensions):
        raise ValueError('extensions must contain lowercase file suffixes such as .py')
    excluded = config['exclude_directories']
    if not isinstance(excluded, list) or any(not isinstance(d, str) or not d for d in excluded):
        raise ValueError('exclude_directories must be a list of directory names')
    parameters = config['bm25']
    if not isinstance(parameters, dict) or set(parameters) != {'k1', 'b'}:
        raise ValueError('bm25 must define k1 and b')
    if type(parameters['k1']) not in (int, float) or not 0 < parameters['k1'] < float('inf'):
        raise ValueError('BM25 k1 must be positive and finite')
    if type(parameters['b']) not in (int, float) or not 0 <= parameters['b'] <= 1:
        raise ValueError('BM25 b must be between 0 and 1')


def ranking_metrics(ranking, gold, k):
    if k <= 0:
        raise ValueError('k must be positive')
    gold = set(gold)
    if not gold:
        raise ValueError('Ground truth cannot be empty')
    # A file contributes only once, even if a caller accidentally supplies duplicates.
    ranking = list(dict.fromkeys(ranking))[:k]
    ranks = [i for i, path in enumerate(ranking, 1) if path in gold]
    return {'top_1': int(any(r <= 1 for r in ranks)),
            'top_5': int(any(r <= min(5, k) for r in ranks)),
            'top_10': int(any(r <= min(10, k) for r in ranks)),
            'recall_at_k': len(ranks) / len(gold),
            'reciprocal_rank_at_k': 1 / ranks[0] if ranks else 0.0,
            'average_precision_at_k': sum(i / rank for i, rank in enumerate(ranks, 1)) / len(gold),
            'k': k, 'gold_files': len(gold), 'retrieved_gold_files': len(ranks)}


def index_definition(parameters):
    return {
        'settings': {
            'number_of_shards': 1, 'number_of_replicas': 0,
            'similarity': {'file_bm25': {'type': 'BM25', **parameters}},
            'analysis': {
                'char_filter': {'identifier_boundaries': {'type': 'pattern_replace',
                    'pattern': r'(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])', 'replacement': ' '}},
                'tokenizer': {'code_tokens': {'type': 'pattern', 'pattern': r'[\W_]+'}},
                'filter': {'code_parts': {'type': 'word_delimiter_graph', 'preserve_original': True}},
                'analyzer': {'code': {'type': 'custom', 'char_filter': ['identifier_boundaries'], 'tokenizer': 'code_tokens',
                                      'filter': ['code_parts', 'lowercase', 'flatten_graph']}},
            },
        },
        'mappings': {'dynamic': 'strict', 'properties': {
            'file_path': {'type': 'keyword'},
            'content': {'type': 'text', 'analyzer': 'code', 'similarity': 'file_bm25'},
            'repo': {'type': 'keyword'}, 'commit': {'type': 'keyword'},
            'sha256': {'type': 'keyword'},
        }},
    }


def select_report(project, config):
    from datasets import load_from_disk

    data = load_from_disk(str(project / 'datasets' / config['dataset']))[config['split']]
    matches = [row for row in data if row['repo_name'] == config['repository']
               and row['issue_id'] == config['issue_id']]
    if len(matches) != 1:
        raise ValueError(f'Expected one bug report, found {len(matches)}')
    return prepare_report(matches[0], config)


def prepare_report(row, config):
    if row['repo_name'] != config['repository'] or row['issue_id'] != config['issue_id']:
        raise ValueError('Provided report does not match the configured repository/issue')
    gold = ast.literal_eval(row['updated_files'])
    if not isinstance(gold, list) or not gold or not all(isinstance(path, str) and path for path in gold):
        raise ValueError('updated_files must encode a non-empty list of file paths')
    # Only report fields configured explicitly can become the query.
    text = '\n'.join(row.get(field) or '' for field in config['query_fields']).strip()
    if not text:
        raise ValueError('Bug report query is empty')
    return row, gold, text


def collect_files(source, manifest, config):
    documents, skipped = [], []
    for relative, sha256 in sorted(manifest['files'].items()):
        path = Path(relative)
        if path.suffix.lower() not in config['extensions']:
            skipped.append({'file_path': relative, 'reason': 'extension_not_selected'})
            continue
        if any(part in config['exclude_directories'] for part in path.parts):
            skipped.append({'file_path': relative, 'reason': 'excluded_directory'})
            continue
        if (source / path).stat().st_size > config['max_file_bytes']:
            skipped.append({'file_path': relative, 'reason': 'too_large'})
            continue
        raw = (source / path).read_bytes()
        reason = None
        if len(raw) > config['max_file_bytes']:
            reason = 'too_large'
        elif b'\x00' in raw:
            reason = 'binary'
        try:
            content = raw.decode('utf-8-sig') if reason is None else None
        except UnicodeDecodeError:
            reason = 'not_utf8'
        if reason:
            skipped.append({'file_path': relative, 'reason': reason})
            continue
        documents.append({'file_path': relative, 'content': content,
                          'repo': manifest['repository'], 'commit': manifest['commit'], 'sha256': sha256})
    if not documents:
        raise ValueError('No eligible source files in the snapshot')
    return documents, skipped


def run(project, config, *, report=None, output_directory=None, verbose=True):
    validate_config(config)
    from elasticsearch import Elasticsearch, helpers

    row, gold, query_text = select_report(project, config) if report is None else prepare_report(report, config)
    client = Elasticsearch(config['endpoint'], request_timeout=60)
    try:
        info = dict(client.info())
        if info['cluster_name'] != config['cluster_name']:
            raise ValueError('Unexpected Elasticsearch cluster; refusing to create an index')
        if not info['version']['number'].startswith('9.'):
            raise ValueError('The current client/configuration expects Elasticsearch 9.x')
        source, manifest = get_snapshot(project, row['repo_name'], row['before_fix_sha'])
        documents, skipped = collect_files(source, manifest, config)
        definition = index_definition(config['bm25'])
        fingerprint = hashlib.sha256(json.dumps({'definition': definition,
                                                'files': [(doc['file_path'], doc['sha256']) for doc in documents]},
                                               sort_keys=True).encode()).hexdigest()
        index = 'bl-bm25-' + row['repo_name'].replace('/', '-').replace('_', '-').lower() + '-' + row['before_fix_sha'][:12] + '-' + fingerprint[:10]
        ownership = {'project': 'bug-localization', 'repository': row['repo_name'],
                     'commit': row['before_fix_sha'], 'fingerprint': fingerprint}
        definition['mappings']['_meta'] = ownership
        if client.indices.exists(index=index):
            existing = client.indices.get_mapping(index=index)[index]['mappings'].get('_meta')
            if existing != ownership:
                raise ValueError('Existing index has different ownership or content settings')
        else:
            client.indices.create(index=index, **definition)
        count = client.count(index=index)['count']
        if count < len(documents):
            # Stable IDs also make an interrupted bulk upload safely resumable.
            actions = ({'_index': index, '_id': hashlib.sha256(doc['file_path'].encode()).hexdigest(),
                        '_source': doc} for doc in documents)
            helpers.bulk(client, actions, chunk_size=100, max_chunk_bytes=5 * 1024 * 1024)
            client.indices.refresh(index=index)
        count = client.count(index=index)['count']
        if count != len(documents):
            raise ValueError(f'Index is incomplete: {count} documents vs {len(documents)} eligible files')
        analyzed = client.indices.analyze(index=index, analyzer='code', text=query_text)
        terms = list(dict.fromkeys(token['token'] for token in analyzed['tokens']))
        selected_terms = terms[:config['max_query_terms']]
        if not selected_terms:
            raise ValueError('The analyzer produced no query terms')
        # Terms are already analyzed. Avoid tokenizing them a second time or
        # creating more clauses than the configured unique-term limit.
        query = {'bool': {'should': [{'term': {'content': term}} for term in selected_terms],
                          'minimum_should_match': 1}}
        response = client.search(index=index, query=query, size=config['top_k'],
                                 source_includes=['file_path'], track_total_hits=True,
                                 sort=[{'_score': 'desc'}, {'file_path': 'asc'}])
        results = [{'rank': i, 'file_path': hit['_source']['file_path'], 'score': hit['_score']}
                   for i, hit in enumerate(response['hits']['hits'], 1)]
        candidate_paths = {doc['file_path'] for doc in documents}
        missing_gold = sorted(set(gold) - candidate_paths)
        output = {
            'dataset': config['dataset'], 'split': config['split'], 'repository': row['repo_name'],
            'issue_id': row['issue_id'], 'before_fix_sha': row['before_fix_sha'],
            'server_version': info['version']['number'], 'cluster_name': info['cluster_name'], 'index': index,
            'indexed_files': count, 'skipped_files': skipped, 'query_fields': config['query_fields'],
            'candidate_files': [{'file_path': doc['file_path'], 'sha256': doc['sha256']} for doc in documents],
            'query_text': query_text, 'query_unique_terms': len(terms), 'query_used_terms': len(selected_terms),
            'query_truncated': len(terms) > len(selected_terms), 'query': query,
            'bm25': config['bm25'], 'extensions': config['extensions'],
            'ground_truth': gold, 'ground_truth_missing_from_candidates': missing_gold,
            'results': results, 'total_matching_files': response['hits']['total']['value'],
            'metrics': ranking_metrics([r['file_path'] for r in results], gold, config['top_k']),
            'snapshot_url': manifest['source_url'], 'snapshot_archive_sha256': manifest['archive_sha256'],
            'index_fingerprint': fingerprint,
            'scope': 'Single-report end-to-end validation; not a full benchmark evaluation',
            'ground_truth_coverage_complete': not missing_gold,
        }
        folder = Path(output_directory) if output_directory is not None else project / 'results/bm25' / (row['repo_name'].replace('/', '__') + '-' + str(row['issue_id']))
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'result.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        (folder / 'effective_config.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
        if verbose:
            print(json.dumps({key: output[key] for key in ('repository', 'issue_id', 'before_fix_sha', 'server_version',
                                                          'indexed_files', 'ground_truth', 'ground_truth_missing_from_candidates',
                                                          'query_used_terms', 'metrics')}, ensure_ascii=False, indent=2))
            for result in results[:10]:
                print(f"{result['rank']:2d}  {result['score']:.4f}  {result['file_path']}")
            print('Results:', folder)
        return output
    finally:
        client.close()
