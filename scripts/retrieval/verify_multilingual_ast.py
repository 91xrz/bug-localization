"""Exercise real grammars, local Qwen and ES on small multi-language fixtures."""

from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'src'))

from bug_localization.ast_chunking import RoutingChunkCache
from bug_localization.batch import write_json
from bug_localization.embeddings import LocalEmbedder, digest
from bug_localization.languages import grammar_versions, language_for_path
from bug_localization.vector import RETRIEVAL_ROUTES, retrieve, validate_vector_config


def main():
    from elasticsearch import Elasticsearch

    config = json.loads((PROJECT / 'configs/retrieval/vector.json').read_text(encoding='utf-8'))
    base = json.loads((PROJECT / config['base_config']).read_text(encoding='utf-8'))
    fixtures = PROJECT / 'tests/fixtures/multilingual'
    documents = []
    for path in sorted(fixtures.iterdir()):
        if path.is_file():
            content = path.read_text(encoding='utf-8')
            documents.append({'file_path': path.name, 'content': content,
                              'sha256': hashlib.sha256(content.encode('utf-8')).hexdigest()})
    if not documents:
        raise ValueError('No multi-language fixtures present')
    validate_vector_config(config, {**base, 'extensions': sorted({Path(d['file_path']).suffix for d in documents})})
    fixture_hash = digest([(d['file_path'], d['sha256']) for d in documents])
    run_id = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
    folder = PROJECT / 'results/vector/multilingual' / run_id
    folder.mkdir(parents=True)
    with Elasticsearch(base['endpoint'], request_timeout=120) as client:
        if client.info()['cluster_name'] != base['cluster_name']:
            raise ValueError('Unexpected Elasticsearch cluster')
        embedder = LocalEmbedder(PROJECT, config)
        try:
            cache = RoutingChunkCache(PROJECT, embedder.tokenizer, config, embedder.fingerprint)
            query, query_info = embedder.query('A cache lookup returns an incorrect value for a supplied key.')
            report = {'status': 'passed', 'scope': 'Small grammar/embedding/ES smoke test; not a benchmark',
                      'fixture_sha256': fixture_hash, 'fixtures': len(documents), 'grammar_versions': grammar_versions(),
                      'languages': sorted({language_for_path(d['file_path']) for d in documents}), 'routes': {}}
            write_json(folder / 'query.json', query_info)
            for route in RETRIEVAL_ROUTES:
                chunks = []
                for document in documents:
                    file_chunks, _ = cache.get(document, route)
                    if ''.join(c['code'] for c in file_chunks) != document['content']:
                        raise ValueError('Fixture source reconstruction failed')
                    if any(c['parse_has_error'] for c in file_chunks):
                        raise ValueError('Valid fixture failed syntax parsing')
                    chunks.extend(file_chunks)
                vectors = embedder.encode([c['embedding_text'] for c in chunks])
                identity = {'repository': 'local/multilingual-smoke', 'commit': fixture_hash[:40], 'route': route,
                            'model_fingerprint': embedder.fingerprint, **cache.index_identity(documents, route)}
                result = retrieve(client, documents, chunks, vectors, query, identity, temperature=0.1)
                if not all(v['top_10_matches_local_order'] for v in result['verification']['aggregations'].values()):
                    raise ValueError('Fixture ES file order differs from local verification')
                write_json(folder / (route + '_chunks.json'), chunks)
                write_json(folder / (route + '_result.json'), result)
                report['routes'][route] = {'files': len(documents), 'chunks': len(chunks),
                                           'maximum_tokens': max(c['token_count'] for c in chunks),
                                           'maximum_prefix_tokens': max(c['prefix_tokens'] for c in chunks),
                                           'verification': result['verification']}
                print(f'{route}: {len(documents)} files / {len(chunks)} chunks; ES verification passed', flush=True)
            write_json(folder / 'verification.json', report)
            print('Multi-language smoke results: ' + str(folder), flush=True)
        finally:
            embedder.close()


if __name__ == '__main__':
    main()
