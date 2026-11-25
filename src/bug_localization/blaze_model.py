"""Pinned BLAZE released encoder, following the author's masked-mean demo.

The public safetensors contain only CodeSage, not the training code's two
residual adapters. This loader deliberately does not invent their weights.
See docs/blaze-baseline.md for provenance and the experiment's identity.
"""

import hashlib
import importlib
import json
from pathlib import Path
import sys
import types
import urllib.request

from .snapshot import sha256_file

MODEL_REPO = 'bug-localization/BLAZE'
MODEL_REVISION = '1b14d31e5b3094e357ab00e971689f402ed62ca1'
CODE_REPO = 'codesage/codesage-small'
CODE_REVISION = '4d2a4f3fa73a3c8a4dd59dba2712202f1f54e109'
WEIGHT_SHA256 = '15363f2a8398bceeecb427041e20560abfae5dda0ebef83a52162a417575b920'
# Git blob IDs of the author's pinned small files; not hashes of local exports.
MODEL_BLOBS = {
    'config.json': 'c30ff26cc0abc761119435ca6d2b2352300fa35b',
    'added_tokens.json': 'c0f1788ed14fd450d89b760c6d9c7538933bac08',
    'merges.txt': 'f8479fb696fe07332c55300a6accf8cc191acc6a',
    'special_tokens_map.json': '827c5e8ba78de7e95cd0adf8a81c4ddbe1a95d0b',
    'tokenizer.json': 'b71c74e04095ea890bd27f1936ccdea57529bde7',
    'tokenizer_config.json': '18c3ca6c5570e00af0f16cf0727dcecc6a9c2dfc',
    'vocab.json': '795a5a47f65fcf89e7957e8462d42c45cf9ca240',
}
CODE_HASHES = {
    'config_codesage.py': '2de01c5a3d81be4e4b62f0d4089f75e830250a52935b070a6970fef61abe4bf7',
    'modeling_codesage.py': '0219cbc801fcd92f2c96d03de7785fb54af58dbe1a6fe8ca24fe1aecf94c8e8e',
}


def git_blob(raw):
    return hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()


def compatible_code(name, raw):
    """API-only changes for Transformers 5; no change to inference math."""
    if hashlib.sha256(raw).hexdigest() != CODE_HASHES[name]:
        raise ValueError('Unrecognized CodeSage source: ' + name)
    if name == 'config_codesage.py':
        return raw
    text = raw.decode('utf-8')
    replacements = {
        'self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)\n\n        self.init_weights()':
        'self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)\n\n        self.post_init()',
        'from transformers.modeling_utils import Conv1D, PreTrainedModel':
        'from transformers.modeling_utils import PreTrainedModel\nfrom transformers.pytorch_utils import Conv1D',
        'head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)':
        "if head_mask is not None:\n            raise ValueError('BLAZE inference does not support head pruning')\n        head_mask = [None] * self.config.num_hidden_layers",
    }
    for old, new in replacements.items():
        if text.count(old) != 1:
            raise ValueError('CodeSage compatibility patch does not match pinned source')
        text = text.replace(old, new)
    return text.encode('utf-8')


def _fetch(url):
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


def prepare_model(destination, reuse_weights=None):
    """Download only pinned, verified public assets; never alter existing exports."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for name, expected in MODEL_BLOBS.items():
        target = destination / name
        raw = target.read_bytes() if target.exists() else _fetch(
            f'https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/{name}')
        if git_blob(raw) != expected:
            raise ValueError('Model file differs from pinned release: ' + str(target))
        if not target.exists():
            target.write_bytes(raw)
    target = destination / 'model.safetensors'
    if not target.exists():
        if reuse_weights and Path(reuse_weights).exists() and sha256_file(Path(reuse_weights)) == WEIGHT_SHA256:
            target.symlink_to(Path(reuse_weights).resolve())
        else:
            temporary = target.with_suffix('.download')
            with urllib.request.urlopen(
                    f'https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/model.safetensors', timeout=120
            ) as response, temporary.open('wb') as output:
                while block := response.read(8 * 1024 * 1024):
                    output.write(block)
            if sha256_file(temporary) != WEIGHT_SHA256:
                raise ValueError('Downloaded BLAZE checkpoint failed checksum verification')
            temporary.replace(target)
    if sha256_file(target) != WEIGHT_SHA256:
        raise ValueError('Local BLAZE checkpoint differs from pinned release')
    source = destination / 'codesage_source'
    source.mkdir(exist_ok=True)
    for name in CODE_HASHES:
        original = source / (name + '.original')
        raw = original.read_bytes() if original.exists() else _fetch(
            f'https://huggingface.co/{CODE_REPO}/resolve/{CODE_REVISION}/{name}')
        patched = compatible_code(name, raw)
        original.write_bytes(raw)
        (source / name).write_bytes(patched)
    identity = verify_model(destination)
    (destination / 'provenance.json').write_text(json.dumps(identity, indent=2) + '\n', encoding='utf-8')
    return identity


def verify_model(path):
    path = Path(path)
    files = {}
    for name, blob in MODEL_BLOBS.items():
        raw = (path / name).read_bytes()
        if git_blob(raw) != blob:
            raise ValueError('Pinned BLAZE asset changed: ' + name)
        files[name] = hashlib.sha256(raw).hexdigest()
    if sha256_file(path / 'model.safetensors') != WEIGHT_SHA256:
        raise ValueError('Pinned BLAZE weights changed')
    files['model.safetensors'] = WEIGHT_SHA256
    for name in CODE_HASHES:
        original = (path / 'codesage_source' / (name + '.original')).read_bytes()
        expected = compatible_code(name, original)
        actual = (path / 'codesage_source' / name).read_bytes()
        if actual != expected:
            raise ValueError('Pinned CodeSage implementation changed: ' + name)
        files['codesage_source/' + name] = hashlib.sha256(actual).hexdigest()
    return {'model_repo': MODEL_REPO, 'model_revision': MODEL_REVISION,
            'code_repo': CODE_REPO, 'code_revision': CODE_REVISION, 'files_sha256': files,
            'variant': 'BLAZE released encoder / author-demo adaptation',
            'pooling': 'attention-mask mean + L2 normalization', 'residual_adapters': False,
            'compatibility': 'Conv1D import relocation; post_init; unpruned head mask is a list of None'}


class BlazeEncoder:
    def __init__(self, path, *, device='cpu', batch_size=4, cpu_threads=4):
        import torch
        from safetensors.torch import load_file
        from transformers import AutoTokenizer

        self.identity = verify_model(path)
        torch.set_num_threads(cpu_threads)
        path = Path(path).resolve()
        # Import only source bytes checked above. No remote-code discovery at runtime.
        package_name = '_blaze_codesage_' + CODE_REVISION
        package = types.ModuleType(package_name)
        package.__path__ = [str(path / 'codesage_source')]
        sys.modules[package_name] = package
        module = importlib.import_module(package_name + '.modeling_codesage')
        config_module = importlib.import_module(package_name + '.config_codesage')
        config = config_module.CodeSageConfig(**json.loads((path / 'config.json').read_text()))
        model = module.CodeSageModel(config)
        # Strict loading rejects missing or unexpected tensors, including adapters.
        model.load_state_dict(load_file(str(path / 'model.safetensors')), strict=True)
        self.model = model.to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
        self.device, self.batch_size = device, batch_size

    def encode(self, texts, max_tokens):
        import numpy as np
        import torch

        if not texts or not 1 <= max_tokens <= self.model.config.max_position_embeddings:
            raise ValueError('Expected nonempty texts and a valid BLAZE token limit')
        vectors, audits = [], []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            counts = [len(self.tokenizer.encode(text, add_special_tokens=True)) for text in batch]
            if any(n == 0 for n in counts):
                raise ValueError('BLAZE cannot encode zero-token text; audit and omit empty source chunks first')
            encoded = self.tokenizer(batch, padding=True, truncation=True,
                                     max_length=max_tokens, return_tensors='pt').to(self.device)
            with torch.inference_mode():
                hidden = self.model(**encoded).last_hidden_state
                mask = encoded['attention_mask'].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1e-9)
                if not torch.isfinite(pooled).all() or (pooled.norm(dim=1) == 0).any():
                    raise ValueError('BLAZE produced invalid or zero vectors')
                normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.append(normalized.cpu().numpy().astype('float32'))
            audits.extend({'tokens_before_truncation': n, 'truncated': n > max_tokens} for n in counts)
        return np.concatenate(vectors), audits
