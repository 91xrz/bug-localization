"""Frozen local CodeXEmbed CLS embeddings with audited, verified caching."""

import hashlib
import importlib
from importlib.metadata import version
import json
from pathlib import Path
import sqlite3
import sys
import time
import types

from .embeddings import digest
from .snapshot import sha256_file


class CodeXEmbed:
    """Use the official Salesforce encoder and its pinned local custom classes.

    The caller implements GenLoc's cl100k query preprocessing. This class never
    silently truncates an input using a second tokenizer.
    """

    def __init__(self, project, config):
        import torch
        from transformers import AutoTokenizer

        self.project = Path(project)
        self.config = dict(config)
        self.path = (self.project / config.get('model_path', 'models/SFR-Embedding-Code-400M_R')).resolve()
        self.device = config.get('device', 'cuda:1')
        dtype_name = config.get('dtype', 'float32')
        if dtype_name not in ('float32', 'float16', 'bfloat16'):
            raise ValueError('CodeXEmbed dtype must be float32, float16 or bfloat16')
        self.dtype = getattr(torch, dtype_name)
        self.max_tokens = config.get('max_tokens', 8192)
        self.batch_size = config.get('embedding_batch_size', 8)
        if not 1 <= self.max_tokens <= 8192 or self.batch_size < 1:
            raise ValueError('Invalid CodeXEmbed token or batch budget')
        if str(self.device).startswith('cuda') and not torch.cuda.is_available():
            raise ValueError('CodeXEmbed CUDA is unavailable; select CPU explicitly')
        manifest_path = self.path / 'manifest.json'
        self.manifest = json.loads(manifest_path.read_text())
        for name, expected in self.manifest['files_sha256'].items():
            if sha256_file(self.path / name) != expected:
                raise ValueError('CodeXEmbed asset checksum mismatch: ' + name)
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.path), local_files_only=True,
                                                      trust_remote_code=False, padding_side='right')
        self.tokenizer.model_max_length = 10**9
        self.dimension = json.loads((self.path / 'config.json').read_text())['hidden_size']
        self.identity = {
            'model_repository': self.manifest['model_repository'],
            'model_revision': self.manifest['model_revision'],
            'code_repository': self.manifest['code_repository'],
            'code_revision': self.manifest['code_revision'],
            'files_sha256': self.manifest['files_sha256'],
            'dimension': self.dimension, 'pooling': 'CLS: last_hidden_state[:, 0]',
            'normalization': 'L2 in float32', 'dtype': dtype_name, 'device': self.device,
            'max_tokens': self.max_tokens, 'embedding_batch_size': self.batch_size,
            'padding_side': 'right', 'overflow_policy': 'explicit_error_no_truncation',
            'torch': version('torch'), 'transformers': version('transformers'),
            'tokenizers': version('tokenizers'),
            'implementation_sha256': sha256_file(Path(__file__)),
            'gpu': torch.cuda.get_device_name(self.device) if str(self.device).startswith('cuda') else None,
        }
        self.fingerprint = digest(self.identity)
        self.cache_dir = self.project / config.get('cache_dir', 'cache/genloc_embeddings') / self.fingerprint
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / 'identity.json').write_text(json.dumps(self.identity, indent=2) + '\n')
        self.database = sqlite3.connect(self.cache_dir / 'vectors.sqlite3', timeout=60)
        self.database.execute('CREATE TABLE IF NOT EXISTS vectors '
                              '(key TEXT PRIMARY KEY, data BLOB NOT NULL, checksum TEXT NOT NULL, tokens INTEGER NOT NULL)')
        self.length_audit_path = self.cache_dir / 'length_audit.jsonl'
        self.last_audit = []
        self.hits = self.misses = 0
        self.model = None

    def load_model(self):
        if self.model is not None:
            return
        # Import only the frozen, checksum-verified files. Loading these classes
        # directly avoids auto_map fetching mutable code from a second HF repo.
        package_name = '_genloc_codexembed_' + self.manifest['code_revision']
        if package_name not in sys.modules:
            package = types.ModuleType(package_name)
            package.__path__ = [str(self.path / 'custom_code')]
            sys.modules[package_name] = package
        configuration = importlib.import_module(package_name + '.configuration')
        modeling = importlib.import_module(package_name + '.modeling')
        model_config = configuration.NewConfig.from_pretrained(str(self.path), local_files_only=True)
        self.model = modeling.NewModel.from_pretrained(str(self.path), config=model_config,
                                                      local_files_only=True, torch_dtype=self.dtype)
        self.model.to(self.device).eval()

    def _record_audit(self, records):
        with self.length_audit_path.open('a') as handle:
            handle.write(json.dumps({'time_unix': time.time(), 'identity': self.fingerprint,
                                     'inputs': records}, ensure_ascii=False) + '\n')

    def encode(self, texts):
        import numpy as np
        import torch
        import torch.nn.functional as functional

        if not texts:
            self.last_audit = []
            return np.empty((0, self.dimension), dtype=np.float32)
        if any(not isinstance(text, str) for text in texts):
            raise TypeError('CodeXEmbed inputs must be strings')
        encoded = self.tokenizer(list(texts), add_special_tokens=True, padding=False, truncation=False)
        lengths = [len(ids) for ids in encoded['input_ids']]
        keys = [hashlib.sha256(text.encode('utf-8')).hexdigest() for text in texts]
        self.last_audit = [{'text_sha256': key, 'original_tokens': length, 'used_tokens': length,
                            'max_tokens': self.max_tokens, 'truncated': False,
                            'over_budget': length > self.max_tokens, 'cache_hit': False}
                           for key, length in zip(keys, lengths)]
        if any(length > self.max_tokens for length in lengths):
            self._record_audit(self.last_audit)
            raise ValueError(f'CodeXEmbed input exceeds {self.max_tokens} model tokens; '
                             'lengths saved in length audit, no silent truncation')
        vectors, missing = {}, {}
        for i, key in enumerate(keys):
            if key in vectors or key in missing:
                continue
            stored = self.database.execute('SELECT data, checksum, tokens FROM vectors WHERE key=?', (key,)).fetchone()
            if stored is not None:
                raw, checksum, tokens = stored
                if hashlib.sha256(raw).hexdigest() != checksum or tokens != lengths[i]:
                    raise ValueError('CodeXEmbed cache integrity or token-length mismatch')
                vector = np.frombuffer(raw, dtype='<f4').copy()
                self._validate_vector(vector)
                vectors[key] = vector
                self.hits += 1
            else:
                missing[key] = i
        for record, key in zip(self.last_audit, keys):
            record['cache_hit'] = key in vectors
        self._record_audit(self.last_audit)
        items = list(missing.items())
        if items:
            self.load_model()
        for start in range(0, len(items), self.batch_size):
            batch = items[start:start + self.batch_size]
            inputs = [{name: values[i] for name, values in encoded.items()} for _, i in batch]
            tensors = self.tokenizer.pad(inputs, padding=True, return_tensors='pt')
            tensors = {name: value.to(self.device) for name, value in tensors.items()}
            if self.tokenizer.cls_token_id is not None and not bool(
                    (tensors['input_ids'][:, 0] == self.tokenizer.cls_token_id).all()):
                raise ValueError('CodeXEmbed CLS pooling requires a leading CLS token')
            with torch.inference_mode():
                hidden = self.model(**tensors).last_hidden_state
                output = functional.normalize(hidden[:, 0].float(), p=2, dim=1).cpu().numpy()
            for (key, i), vector in zip(batch, output):
                self._validate_vector(vector)
                raw = vector.astype('<f4').tobytes()
                self.database.execute('INSERT OR REPLACE INTO vectors VALUES (?, ?, ?, ?)',
                                      (key, raw, hashlib.sha256(raw).hexdigest(), lengths[i]))
                vectors[key] = vector
                self.misses += 1
            self.database.commit()
            completed = min(start + self.batch_size, len(items))
            if start == 0 or completed % 256 < self.batch_size or completed == len(items):
                print(f'CodeXEmbed: encoded {completed}/{len(items)} new texts; cached={self.hits}', flush=True)
        return np.stack([vectors[key] for key in keys])

    def _validate_vector(self, vector):
        import numpy as np
        if (vector.shape != (self.dimension,) or not np.isfinite(vector).all()
                or abs(float(np.linalg.norm(vector)) - 1.0) > 1e-3):
            raise ValueError('CodeXEmbed produced or loaded an invalid normalized vector')

    def query(self, text):
        vector = self.encode([text])[0]
        audit = dict(self.last_audit[0], embedding_text=text, embedding_identity=self.fingerprint,
                     preprocessing='caller-provided text; no additional instruction or truncation')
        return vector, audit

    def close(self):
        self.database.close()
        self.model = None
