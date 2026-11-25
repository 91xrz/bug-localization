"""RRL's raw-file Lucene BM25 index, using the author's Anserini backend.

RRL source: gencau/boatse-query-reformulation, Apache-2.0. The author leaves
Pyserini unversioned. We pin 0.44.0 and call its bundled Java SimpleSearcher
directly: the old Python import names no longer exist, while the modern Python
search package imports unrelated neural encoders. This preserves the indexer,
English analyzer, query parser and BM25 implementation without loading them.
"""

import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import time

PYSERINI_VERSION = '0.44.0'
PYJNIUS_VERSION = '1.6.1'


def _sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def set_runtime(project):
    """Configure the isolated packages and JDK before the first JVM import."""
    project = Path(project).resolve()
    dependencies = project / '.runtime/rrl-deps'
    jdk = project / '.runtime/elasticsearch/elasticsearch-9.5.5/jdk'
    if not dependencies.is_dir() or not (jdk / 'bin/java').is_file():
        raise FileNotFoundError('Prepare .runtime/rrl-deps and the local Elasticsearch 9.5.5 bundled JDK first')
    dependency_string = str(dependencies)
    if dependency_string not in sys.path:
        sys.path.insert(0, dependency_string)
    for name, expected in [('pyserini', PYSERINI_VERSION), ('pyjnius', PYJNIUS_VERSION)]:
        if version(name) != expected:
            raise ValueError(f'RRL requires the isolated {name}=={expected}')
    if 'jnius' in sys.modules and os.environ.get('JAVA_HOME') != str(jdk):
        raise RuntimeError('JVM already imported with a different JAVA_HOME; start a fresh RRL process')
    os.environ['JAVA_HOME'] = str(jdk)
    return {'dependency_path': dependency_string, 'java_home': str(jdk),
            'java_release_sha256': _sha256(jdk / 'release'),
            'pyserini': PYSERINI_VERSION, 'pyjnius': PYJNIUS_VERSION,
            'anserini_jar_sha256': _sha256(dependencies / 'pyserini/resources/jars/anserini-0.39.0-fatjar.jar')}


def _searcher(index_dir):
    from pyserini.pyclass import autoclass

    return autoclass('io.anserini.search.SimpleSearcher')(str(index_dir))


def search_index(repo_dir, query, top_k=10):
    """Same Java calls as Pyserini LuceneSearcher, with RRL's k1=.9, b=.4."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError('RRL BM25 query must be nonempty text')
    if type(top_k) is not int or not 1 <= top_k <= 30:
        raise ValueError('RRL BM25 top_k must be in [1, 30]')
    searcher = _searcher(Path(repo_dir) / 'indexes')
    try:
        searcher.set_bm25(0.9, 0.4)
        hits = searcher.search(query, top_k)
        files, scores = [], []
        for hit in hits:
            raw = json.loads(searcher.doc(hit.docid).get('raw'))
            files.append(raw['path'])
            scores.append(float(hit.score))
        return files, scores
    finally:
        searcher.close()


def prepare_index(project, repo_dir, documents):
    """Index the shared candidate list in a private per-report workspace.

Unlike the author's convenience file collector, this function never removes
test files beyond the caller's already-frozen candidate policy.
"""
    runtime = set_runtime(project)
    repo_dir = Path(repo_dir).resolve()
    repo_dir.mkdir(parents=True, exist_ok=True)
    if not documents:
        raise ValueError('RRL needs at least one candidate file')
    records, seen = [], set()
    for i, document in enumerate(documents):
        path = document['file_path']
        relative = PurePosixPath(path)
        if (not isinstance(path, str) or relative.is_absolute() or '..' in relative.parts
                or '\\' in path or path in seen or not isinstance(document['content'], str)):
            raise ValueError('RRL candidates must have unique, relative paths and text contents')
        seen.add(path)
        records.append({'id': str(i), 'contents': document['content'], 'path': path})
    identity = {'records_sha256': _digest(records), 'documents': len(records), 'runtime': runtime,
                'collection': 'JsonCollection', 'generator': 'DefaultLuceneDocumentGenerator',
                'analyzer': 'Anserini default English analyzer', 'threads': 1,
                'store_positions': True, 'store_docvectors': True, 'store_raw': True,
                'search_bm25': {'k1': 0.9, 'b': 0.4},
                'candidates': 'caller-provided shared candidates; no extra skip_tests filter'}
    fingerprint = _digest(identity)
    manifest_path = repo_dir / 'rrl-index.json'
    index_dir, collection_dir = repo_dir / 'indexes', repo_dir / '.rrl-json'
    if index_dir.exists():
        if not manifest_path.exists():
            raise ValueError('Existing RRL index has no ownership manifest')
        manifest = json.loads(manifest_path.read_text())
        files = {p.relative_to(index_dir).as_posix(): _sha256(p) for p in sorted(index_dir.rglob('*')) if p.is_file()}
        if manifest['fingerprint'] != fingerprint or files != manifest['index_files_sha256']:
            raise ValueError('Existing RRL index differs from the shared candidates or stored checksums')
        return {**manifest, 'cache_hit': True}
    build_dir = repo_dir / '.rrl-index-build'
    if build_dir.exists() or collection_dir.exists():
        raise ValueError('Incomplete RRL index preparation exists; retain it for diagnosis and use a new run directory')
    collection_dir.mkdir()
    for record in records:
        (collection_dir / (record['id'] + '.json')).write_text(json.dumps(record) + '\n', encoding='utf-8')
    command = [sys.executable, '-m', 'pyserini.index.lucene', '--collection', 'JsonCollection',
               '--input', str(collection_dir), '--index', str(build_dir),
               '--generator', 'DefaultLuceneDocumentGenerator', '--threads', '1',
               '--storePositions', '--storeDocvectors', '--storeRaw']
    environment = dict(os.environ)
    environment['PYTHONPATH'] = runtime['dependency_path'] + (os.pathsep + environment['PYTHONPATH']
                                                            if environment.get('PYTHONPATH') else '')
    started = time.monotonic()
    with (repo_dir / 'rrl-index.log').open('w', encoding='utf-8') as log:
        subprocess.run(command, env=environment, check=True, stdout=log, stderr=subprocess.STDOUT)
    searcher = _searcher(build_dir)
    try:
        count = searcher.get_total_num_docs()
        empty_records = []
        for record in records:
            document = searcher.doc(record['id'])
            if document is None and not record['contents'].strip():
                empty_records.append(record['path'])
                continue
            if document is None:
                raise ValueError('RRL index omitted a nonempty candidate: ' + record['path'])
            stored = json.loads(document.get('raw'))
            if stored != record:
                raise ValueError('RRL index stored content does not match its candidate file')
        if count != len(records) - len(empty_records):
            raise ValueError('RRL index document count mismatch')
    finally:
        searcher.close()
    build_dir.rename(index_dir)
    manifest = {'fingerprint': fingerprint, 'identity': identity,
                'candidate_files': len(records), 'indexed_files': count,
                'empty_files_omitted_by_lucene': empty_records,
                'index_seconds': time.monotonic() - started,
                'command': command,
                'index_files_sha256': {p.relative_to(index_dir).as_posix(): _sha256(p)
                                        for p in sorted(index_dir.rglob('*')) if p.is_file()}}
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return {**manifest, 'cache_hit': False}
