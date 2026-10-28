"""Offline Qwen last-token embeddings with a verified, content-addressed cache."""

import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sqlite3

from .snapshot import sha256_file


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class LocalEmbedder:
    def __init__(self, project, config):
        import torch
        from transformers import AutoTokenizer

        self.config = config
        self.path = (project / config['model_path']).resolve()
        required = ['model.safetensors', 'config.json', 'tokenizer.json', 'tokenizer_config.json']
        hashes = {name: sha256_file(self.path / name) for name in required}
        self.device = config['device']
        self.gpu_count = config.get('gpu_count', 1) if self.device == 'cuda' else 0
        if self.device == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA is not available; install the CUDA PyTorch wheel or explicitly select cpu')
        if self.device == 'cuda' and torch.cuda.device_count() < self.gpu_count:
            raise ValueError(f'Requested {self.gpu_count} GPUs, but only {torch.cuda.device_count()} are visible')
        self.dtype = torch.bfloat16 if self.device == 'cuda' else torch.float32
        for index in range(self.gpu_count):
            with torch.cuda.device(index):
                if not torch.cuda.is_bf16_supported():
                    raise ValueError(f'CUDA device {index} does not support bfloat16')
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.path), local_files_only=True,
                                                     trust_remote_code=False, padding_side='left')
        self.tokenizer.model_max_length = 10**9  # Chunking measures length explicitly, without silent truncation.
        model_config = json.loads((self.path / 'config.json').read_text(encoding='utf-8'))
        self.dimension = model_config['hidden_size']
        self.identity = {'model_path': config['model_path'], 'files_sha256': hashes,
                         'dimension': self.dimension, 'pooling': 'last_nonpadding_token',
                         'normalization': 'L2', 'dtype': str(self.dtype), 'device': self.device,
                         'gpu_count': self.gpu_count, 'embedding_batch_size': config['embedding_batch_size'],
                         'max_tokens': config['max_tokens'],
                         'torch': version('torch'), 'transformers': version('transformers'),
                         'encoder_sha256': sha256_file(Path(__file__)),
                         'gpu': [torch.cuda.get_device_name(i) for i in range(self.gpu_count)]}
        self.fingerprint = digest(self.identity)
        folder = project / 'cache/embeddings' / self.fingerprint
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'model.json').write_text(json.dumps(self.identity, indent=2) + '\n', encoding='utf-8')
        self.database = sqlite3.connect(folder / 'vectors.sqlite3')
        self.database.execute('CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, data BLOB NOT NULL, checksum TEXT NOT NULL)')
        self.model = None
        self.hits = self.misses = 0

    def load_model(self):
        if self.model is None:
            import torch
            from transformers import AutoModel
            print('Loading local Qwen model on ' + self.device + ' (no model download)', flush=True)
            self.model = AutoModel.from_pretrained(str(self.path), local_files_only=True, trust_remote_code=False,
                                                  dtype=self.dtype, attn_implementation='sdpa')
            self.model.to(self.device).eval()
            if self.gpu_count > 1:
                self.model = torch.nn.DataParallel(self.model, device_ids=list(range(self.gpu_count)))

    def encode(self, texts):
        import numpy as np
        import torch
        import torch.nn.functional as functional

        if not texts:
            raise ValueError('Cannot encode an empty list')
        keys = [hashlib.sha256(text.encode()).hexdigest() for text in texts]
        vectors, missing = {}, {}
        for key, text in zip(keys, texts):
            if key in vectors or key in missing:
                continue
            stored = self.database.execute('SELECT data, checksum FROM vectors WHERE key=?', (key,)).fetchone()
            if stored:
                raw, checksum = stored
                if hashlib.sha256(raw).hexdigest() != checksum:
                    raise ValueError('Embedding cache checksum mismatch')
                vector = np.frombuffer(raw, dtype='<f4').copy()
                if vector.shape != (self.dimension,) or not np.isfinite(vector).all() or abs(np.linalg.norm(vector) - 1) > 1e-3:
                    raise ValueError('Invalid cached embedding')
                vectors[key] = vector
                self.hits += 1
            else:
                missing[key] = text
        items = list(missing.items())
        if items:
            self.load_model()
        batch_size = self.config['embedding_batch_size']
        for start in range(0, len(items), batch_size):
            batch = items[start:start + batch_size]
            encoded = self.tokenizer([text for _, text in batch], padding=True, truncation=False,
                                     add_special_tokens=True, return_tensors='pt')
            if encoded['input_ids'].shape[1] > self.config['max_tokens']:
                raise ValueError('Embedding input exceeds configured token budget')
            encoded = {name: value.to(self.device) for name, value in encoded.items()}
            with torch.inference_mode():
                hidden = self.model(**encoded, use_cache=False).last_hidden_state
                # Left padding puts every last non-padding token in the last column.
                if not bool(encoded['attention_mask'][:, -1].all()):
                    raise ValueError('Last-token pooling requires left padding')
                output = functional.normalize(hidden[:, -1].float(), p=2, dim=1).cpu().numpy()
            for (key, _), vector in zip(batch, output):
                if vector.shape != (self.dimension,) or not np.isfinite(vector).all() or abs(np.linalg.norm(vector) - 1) > 1e-3:
                    raise ValueError('Model produced an invalid embedding')
                raw = vector.astype('<f4').tobytes()
                self.database.execute('INSERT OR REPLACE INTO vectors VALUES (?, ?, ?)',
                                      (key, raw, hashlib.sha256(raw).hexdigest()))
                vectors[key] = vector
                self.misses += 1
            self.database.commit()
            if start == 0 or (start // batch_size) % 8 == 0 or start + batch_size >= len(items):
                print(f'  Encoded {min(start + batch_size, len(items))}/{len(items)} new texts; cached={self.hits}', flush=True)
        return np.stack([vectors[key] for key in keys])

    def query(self, body):
        instruction = 'Instruct: ' + self.config['query_instruction'] + '\nQuery: '
        ids = self.tokenizer(body, add_special_tokens=False, return_offsets_mapping=True)
        original_count = len(self.tokenizer.encode(instruction + body, add_special_tokens=True))
        if original_count <= self.config['max_tokens']:
            text = instruction + body
        else:
            stop = ids['offset_mapping'][min(len(ids['input_ids']), self.config['max_tokens']) - 1][1]
            used = body[:stop]
            while len(self.tokenizer.encode(instruction + used, add_special_tokens=True)) > self.config['max_tokens']:
                used = used[:-1]
            if not used:
                raise ValueError('Query instruction leaves no room for report text')
            text = instruction + used
        return self.encode([text])[0], {'embedding_text': text, 'original_tokens': original_count,
                                       'used_tokens': len(self.tokenizer.encode(text, add_special_tokens=True)),
                                       'truncated': original_count > self.config['max_tokens']}

    def close(self):
        self.database.close()
