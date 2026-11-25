"""BLAZE release adaptation: dynamic chunks, cosine and chunk-contribution RRF.

Preprocessing / DP conventions and RRF are adapted from Partha Chakraborty
et al., BLAZE (TSE 2025), https://zenodo.org/records/15122980 (CC BY 4.0).
The original reference files remain unchanged in references/blaze/.
"""

import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re

from .batch import write_json
from .chunking import dynamic_ranges
from .languages import parser_for_language
from .snapshot import sha256_file

LANGUAGES = {'.py': 'python', '.java': 'java', '.cpp': 'cpp', '.cxx': 'cpp',
             '.cc': 'cpp', '.h': 'cpp', '.js': 'javascript', '.ts': 'javascript'}
CLASS_TYPES = {'class_definition', 'class_specifier', 'class_declaration',
               'enum_specifier', 'interface_declaration', 'struct_type'}
FUNCTION_TYPES = {'function_definition', 'function_declaration',
                  'generator_function_declaration', 'method_definition'}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def clean_comments(content, language, parser):
    """Preserve author preprocessing, including global replacement semantics.

The author uses Tree-sitter byte offsets to slice a Python string. Retaining
that behavior avoids silently substituting a different baseline; Unicode
comment handling is therefore a known inherited limitation.
"""
    comments, stack = [], [parser.parse(content.encode('utf-8')).root_node]
    while stack:
        node = stack.pop()
        if 'comment' in node.type:
            comments.append(content[node.start_byte:node.end_byte])
        stack.extend(reversed(node.children))
    for comment in sorted(comments, key=len, reverse=True):
        comment = comment.replace('\n', '').strip()
        if (language == 'python' and len(comment) > 2) or language != 'python':
            content = content.replace(comment, '')
    return re.sub(r'\n+', '\n', content)


def split_file(path, content):
    language = LANGUAGES[Path(path).suffix.lower()]
    parser = parser_for_language(language)
    cleaned = clean_comments(content, language, parser)
    root = parser.parse(cleaned.encode('utf-8')).root_node
    # Match the author's first-child ERROR fallback, not root.has_error.
    fallback = bool(root.children and root.children[0].type == 'ERROR')
    if fallback:
        pieces = [cleaned[i:i + 2000] for i in range(0, len(cleaned), 2000)]
    else:
        points, stack = [], [(root, 0)]
        while stack:
            node, depth = stack.pop()
            if depth > 2000:
                continue
            if node.type in CLASS_TYPES | FUNCTION_TYPES and b'lambda' not in node.text:
                points.append((node.start_point[0] + 1, 3 if node.type in CLASS_TYPES else 5))
            stack.extend((child, depth + 1) for child in reversed(node.children))
        lines = cleaned.split('\n')
        pieces = ['\n'.join(lines[start:end]).strip()
                  for start, end in dynamic_ranges(points, len(lines), 30, 50)]
    return [{'file_path': path, 'chunk_position': i, 'text': text,
             'chunk_id': digest([path, i, text]), 'parse_error': root.has_error,
             'character_fallback': fallback} for i, text in enumerate(pieces)]


def chunk_rrf(dense_chunks, lexical_files, constant=60):
    """Every retrieved chunk contributes, even for an already encountered file."""
    if constant <= 0:
        raise ValueError('RRF constant must be positive')
    scores = {}
    for branch, ranking in (('dense', dense_chunks), ('lexical', lexical_files)):
        for rank, item in enumerate(ranking, 1):
            entry = scores.setdefault(item['file_path'], {'file_path': item['file_path'],
                                      'score': 0.0, 'dense_chunk_ranks': [], 'lexical_rank': None})
            entry['score'] += 1 / (constant + rank)
            if branch == 'dense':
                entry['dense_chunk_ranks'].append(rank)
            else:
                entry['lexical_rank'] = rank
    # Author ties retain insertion order: dense first, then lexical.
    return [{'rank': i, **item} for i, item in enumerate(
        sorted(scores.values(), key=lambda item: -item['score']), 1)]


def dense_rank(chunks, vectors, query, depth):
    import numpy as np

    if vectors.ndim != 2 or len(chunks) != len(vectors) or query.shape != (vectors.shape[1],):
        raise ValueError('Chunk/vector/query dimensions do not match')
    if not np.isfinite(vectors).all() or not np.isfinite(query).all():
        raise ValueError('Nonfinite vectors')
    scores = vectors @ query  # Encoder returns L2-normalized float32 vectors.
    order = np.argsort(-scores, kind='stable')[:depth]
    return [{'rank': rank, 'file_path': chunks[i]['file_path'], 'chunk_id': chunks[i]['chunk_id'],
             'chunk_position': chunks[i]['chunk_position'], 'score': float(scores[i])}
            for rank, i in enumerate(order, 1)]


def unique_dense_files(ranking):
    seen, result = set(), []
    for item in ranking:
        if item['file_path'] not in seen:
            result.append({**item, 'rank': len(result) + 1, 'first_chunk_rank': item['rank']})
            seen.add(item['file_path'])
    return result


def cached_vectors(project, documents, encoder, max_tokens):
    import numpy as np

    raw_chunks = [chunk for doc in documents for chunk in split_file(doc['file_path'], doc['content'])]
    chunks, excluded_empty_chunks = [], []
    for chunk in raw_chunks:
        if encoder.tokenizer.encode(chunk['text'], add_special_tokens=True):
            chunks.append(chunk)
        else:
            excluded_empty_chunks.append({key: chunk[key] for key in
                                          ('file_path', 'chunk_id', 'chunk_position')})
    if not chunks:
        raise ValueError('No BLAZE dense chunks remain after excluding '
                         + str(len(excluded_empty_chunks)) + ' zero-token chunks')
    languages = {LANGUAGES[Path(doc['file_path']).suffix.lower()] for doc in documents}
    identity = {'files': [(doc['file_path'], doc['sha256']) for doc in documents],
                'chunks': [c['chunk_id'] for c in chunks], 'model': encoder.identity,
                'source_max_tokens': max_tokens, 'device': encoder.device,
                'batch_size': encoder.batch_size,
                'empty_chunk_policy': 'Exclude zero-token chunks from dense encoding only; retain file candidates and gold',
                'excluded_empty_chunks': excluded_empty_chunks,
                'implementation_sha256': {name: sha256_file(Path(__file__).with_name(name))
                                          for name in ('blaze.py', 'blaze_model.py', 'chunking.py', 'languages.py')},
                'dependencies': {name: version(name) for name in
                                 ['torch', 'transformers', 'tokenizers', 'numpy', 'tree-sitter'] +
                                 sorted('tree-sitter-' + lang for lang in languages)}}
    key = digest(identity)
    folder = project / 'cache/blaze' / key
    metadata_path, vectors_path = folder / 'metadata.json', folder / 'vectors.npy'
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata['identity'] != json.loads(json.dumps(identity)) or sha256_file(vectors_path) != metadata['vectors_sha256']:
            raise ValueError('BLAZE cache identity or vector checksum mismatch')
        vectors = np.load(vectors_path, allow_pickle=False)
        cache_hit = True
    else:
        vectors, audits = encoder.encode([c['text'] for c in chunks], max_tokens)
        folder.mkdir(parents=True, exist_ok=True)
        temporary = folder / 'vectors.tmp'
        with temporary.open('wb') as handle:
            np.save(handle, vectors, allow_pickle=False)
        temporary.replace(vectors_path)
        metadata = {'identity': identity, 'vectors_sha256': sha256_file(vectors_path), 'token_audit': audits}
        write_json(metadata_path, metadata)
        cache_hit = False
    if (vectors.shape != (len(chunks), encoder.model.config.hidden_size) or vectors.dtype != np.float32
            or not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
            or len(metadata['token_audit']) != len(chunks)):
        raise ValueError('Invalid BLAZE vector cache')
    return chunks, vectors, {'fingerprint': key, 'cache_hit': cache_hit,
                             'vectors_sha256': metadata['vectors_sha256'],
                             'token_audit': metadata['token_audit'],
                             'empty_chunk_policy': identity['empty_chunk_policy'],
                             'excluded_empty_chunks': excluded_empty_chunks}


def lexical_rank(client, documents, query, depth):
    """Original standard-analyzer, whole-file BM25; no custom code analyzer."""
    from elasticsearch import helpers

    definition = {
        'settings': {'number_of_shards': 1, 'number_of_replicas': 0},
        'mappings': {'dynamic': 'strict', 'properties': {
            'code': {'type': 'text', 'analyzer': 'standard'},
            'file_path': {'type': 'keyword'}}},
    }
    fingerprint = digest({'definition': definition, 'files': [(d['file_path'], d['sha256']) for d in documents],
                          'repository': documents[0]['repo'], 'commit': documents[0]['commit']})
    index = 'bl-blaze-' + fingerprint[:32]
    owner = {'project': 'bug-localization', 'method': 'BLAZE-release', 'fingerprint': fingerprint}
    definition['mappings']['_meta'] = owner
    if client.indices.exists(index=index):
        if client.indices.get_mapping(index=index)[index]['mappings'].get('_meta') != owner:
            raise ValueError('BLAZE index belongs to another configuration')
    else:
        client.indices.create(index=index, **definition)
    if client.count(index=index)['count'] != len(documents):
        helpers.bulk(client, ({'_index': index, '_id': digest(d['file_path']),
                               '_source': {'file_path': d['file_path'], 'code': d['content']}}
                              for d in documents), refresh='wait_for')
    if client.count(index=index)['count'] != len(documents):
        raise ValueError('BLAZE index candidate count mismatch')
    response = client.search(index=index, query={'match': {'code': query}}, size=depth,
                             sort=[{'_score': 'desc'}, {'file_path': 'asc'}])
    if response.get('timed_out') or response['_shards']['failed']:
        raise RuntimeError('Incomplete BLAZE lexical search')
    ranking = [{'rank': i, 'file_path': hit['_source']['file_path'], 'score': hit['_score']}
               for i, hit in enumerate(response['hits']['hits'], 1)]
    return ranking, {'index': index, 'index_fingerprint': fingerprint, 'analyzer': 'standard',
                     'query_truncated': False, 'bm25': {'k1': 1.2, 'b': 0.75},
                     'tie_break': 'file_path ascending'}
