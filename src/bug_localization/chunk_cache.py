"""Checksum-verified per-file chunk caching across unchanged repository snapshots."""

from pathlib import Path
import json

from .batch import write_json
from .embeddings import digest
from .snapshot import sha256_file


class ChunkCache:
    def __init__(self, project, chunker, model_fingerprint):
        self.chunker = chunker
        settings = {key: chunker.config[key] for key in ('max_tokens', 'prefix_max_tokens',
                    'dynamic_max_lines', 'dynamic_general_cost', 'fixed_chunk_tokens')}
        self.identity = {'format': 1, 'model_fingerprint': model_fingerprint, 'settings': settings,
                         'chunker_sha256': sha256_file(Path(__file__).with_name('chunking.py'))}
        self.folder = project / 'cache/chunks' / digest(self.identity)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.hits = self.misses = 0

    def get(self, document, route):
        key = {'file_path': document['file_path'], 'source_sha256': document['sha256'],
               'route': route, 'identity': self.identity}
        path = self.folder / (digest(key) + '.json')
        if path.exists():
            stored = json.loads(path.read_text(encoding='utf-8'))
            payload = stored['payload']
            if stored['key'] != key or stored['checksum'] != digest(payload):
                raise ValueError('Chunk cache checksum or identity mismatch: ' + document['file_path'])
            chunks, parents = payload['chunks'], payload['parents']
            if not chunks or ''.join(c['code'] for c in chunks) != document['content']:
                raise ValueError('Cached chunks do not reconstruct the source')
            position = 0
            for chunk in chunks:
                if (chunk['file_path'] != document['file_path'] or chunk['route'] != route or
                        chunk['start_byte'] != position or
                        chunk['end_byte'] - position != len(chunk['code'].encode('utf-8')) or
                        chunk['token_count'] > self.chunker.config['max_tokens'] or
                        chunk['prefix_tokens'] > self.chunker.config['prefix_max_tokens']):
                    raise ValueError('Invalid cached chunk metadata')
                if route == 'fixed' and chunk['body_tokens'] > self.chunker.config['fixed_chunk_tokens']:
                    raise ValueError('Cached fixed chunk exceeds the body budget')
                position = chunk['end_byte']
            self.hits += 1
            return chunks, parents
        chunks, parents = self.chunker.chunks(document['file_path'], document['content'], route)
        payload = {'chunks': chunks, 'parents': parents}
        write_json(path, {'key': key, 'payload': payload, 'checksum': digest(payload)})
        self.misses += 1
        return chunks, parents
