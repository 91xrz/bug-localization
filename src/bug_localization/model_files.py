"""Local safetensors discovery and narrowly scoped encoder cache compatibility."""

import json
from pathlib import Path

from .snapshot import sha256_file


# Only this exact, reviewed loader-only change may reuse the previous encoder
# identity. Any later edit to embeddings.py automatically creates a new cache.
LEGACY_ENCODER_SHA256 = 'a9daae2c14b9776e37647999d9a883cdc116b6ec1539c76f600b00ddd1b0068c'
SHARDED_LOADER_ENCODER_SHA256 = '4ea4bedf315868ee9966b0ac4d82518855e843832df52df45e9b1d0ac76d521f'


def encoder_implementation_sha256(path):
    current = sha256_file(path)
    return LEGACY_ENCODER_SHA256 if current == SHARDED_LOADER_ENCODER_SHA256 else current


def required_model_files(root):
    root = Path(root).resolve()
    single = root / 'model.safetensors'
    if single.is_file():
        weights = ['model.safetensors']
    else:
        index_name = 'model.safetensors.index.json'
        index = json.loads((root / index_name).read_text(encoding='utf-8'))
        mapping = index.get('weight_map')
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError('Safetensors index needs a non-empty weight_map')
        for tensor, name in mapping.items():
            if (not isinstance(tensor, str) or not tensor or not isinstance(name, str)
                    or not name.endswith('.safetensors') or '/' in name or '\\' in name
                    or ':' in name or name in ('.', '..')):
                raise ValueError('Unsafe or invalid safetensors shard name')
        weights = [index_name, *sorted(set(mapping.values()))]
    required = [*weights, 'config.json', 'tokenizer.json', 'tokenizer_config.json']
    for name in required:
        path = root / name
        if not path.resolve().is_relative_to(root):
            raise ValueError('Model file escapes the local model directory')
        if not path.is_file():
            raise FileNotFoundError('Missing local model file: ' + str(path))
    return required
