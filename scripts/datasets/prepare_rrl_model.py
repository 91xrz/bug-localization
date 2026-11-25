"""Convert the existing local Qwen2.5-Coder-7B to a CPU RRL smoke-test GGUF.

This is an explicitly smaller-model adaptation, not the paper's 30B/32B model.
Conversion dependencies are installed in .runtime/rrl-model-deps only.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[2]
LLAMA_COMMIT = 'b2e5e9b28b2484fbf94b543432ece638996a8b97'
LLAMA_TAG = 'b10549'
MODEL_REVISION = 'c03e6d358207e414f1eca0bb1891e29f1db0e242'


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def command(args, **kwargs):
    subprocess.run([str(arg) for arg in args], check=True, **kwargs)


def prepare(source, destination, *, keep_intermediate=False):
    checkout = PROJECT / '.runtime/rrl-model-source/llama.cpp'
    deps = PROJECT / '.runtime/rrl-model-deps'
    binary = PROJECT / '.runtime/llama/b10549/llama-b10549/llama-quantize'
    if not source.is_dir() or not binary.is_file():
        raise FileNotFoundError('Local HF model and llama.cpp b10549 binaries are required')
    if not checkout.exists():
        checkout.parent.mkdir(parents=True, exist_ok=True)
        command(['git', 'clone', '--filter=blob:none', '--no-checkout', '--depth', '1',
                 '--branch', LLAMA_TAG, 'https://github.com/ggml-org/llama.cpp.git', checkout])
        command(['git', 'sparse-checkout', 'set', 'conversion', 'gguf-py', 'requirements'], cwd=checkout)
        command(['git', 'checkout', '--detach', LLAMA_TAG], cwd=checkout)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=checkout, text=True).strip()
    if commit != LLAMA_COMMIT:
        raise ValueError('Converter checkout is not the pinned llama.cpp version')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=checkout, text=True).strip():
        raise ValueError('Converter source has local modifications')
    if not (deps / 'transformers').is_dir():
        command([sys.executable, '-m', 'pip', 'install', '--target', deps,
                 'transformers==4.57.6', 'numpy==1.26.4', 'sentencepiece==0.2.1', 'protobuf==4.25.8'])
    environment = dict(os.environ, PYTHONPATH=str(deps), OMP_NUM_THREADS='8', CUDA_VISIBLE_DEVICES='',
                       HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    dependency_versions = json.loads(subprocess.check_output(
        [sys.executable, '-c', "import json; from importlib.metadata import version; "
         "print(json.dumps({name: version(name) for name in "
         "['transformers', 'tokenizers', 'numpy', 'sentencepiece', 'protobuf', 'torch']}))"],
        env=environment, text=True))
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / 'manifest.json'
    quantized = destination / 'Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf'
    model_files = [p for p in sorted(source.iterdir()) if p.is_file() and
                   (p.suffix in ('.json', '.safetensors', '.txt') or p.name == 'LICENSE')]
    source_hashes = {p.name: sha256(p) for p in model_files}
    for path in model_files:
        metadata = source / '.cache/huggingface/download' / (path.name + '.metadata')
        if metadata.exists():
            records = metadata.read_text().splitlines()
            revision = records[0]
            if revision != MODEL_REVISION:
                raise ValueError('Local model assets do not share the expected HF revision')
            if path.suffix == '.safetensors' and records[1] != source_hashes[path.name]:
                raise ValueError('Source safetensors do not match their downloaded SHA-256 metadata')
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing['source_files_sha256'] != source_hashes or sha256(quantized) != existing['gguf_sha256']:
            raise ValueError('Existing RRL GGUF or source model differs from its manifest')
        print(manifest_path)
        return
    intermediate = destination / 'Qwen2.5-Coder-7B-Instruct-F16.gguf'
    if intermediate.exists() or quantized.exists():
        raise ValueError('Unmanifested conversion output exists; review before retrying')
    convert = [sys.executable, checkout / 'convert_hf_to_gguf.py', source,
               '--outfile', intermediate, '--outtype', 'f16']
    quantize = [binary, intermediate, quantized, 'Q4_K_M', '8']
    with (destination / 'conversion.log').open('w') as log:
        command(convert, env=environment, stdout=log, stderr=subprocess.STDOUT)
    intermediate_hash = sha256(intermediate)
    with (destination / 'quantization.log').open('w') as log:
        command(quantize, env=environment, stdout=log, stderr=subprocess.STDOUT)
    manifest = {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'variant': 'RRL smoke adaptation with Qwen2.5-Coder-7B-Instruct Q4_K_M on CPU',
        'source_repo': 'Qwen/Qwen2.5-Coder-7B-Instruct', 'source_revision': MODEL_REVISION,
        'source_path': str(source.resolve()), 'source_files_sha256': source_hashes,
        'llama_cpp_commit': LLAMA_COMMIT, 'llama_cpp_tag': LLAMA_TAG,
        'converter_sha256': sha256(checkout / 'convert_hf_to_gguf.py'),
        'conversion_dependencies': dependency_versions,
        'quantizer_sha256': sha256(binary), 'intermediate_f16_sha256': intermediate_hash,
        'gguf_path': str(quantized.resolve()), 'gguf_sha256': sha256(quantized),
        'gguf_bytes': quantized.stat().st_size, 'quantization': 'Q4_K_M',
        'convert_command': list(map(str, convert)), 'quantize_command': list(map(str, quantize)),
        'tokenizer_source': 'The same local HF checkpoint tokenizer.json and tokenizer_config.json; embedded in GGUF',
        'code_license': 'llama.cpp MIT', 'model_license': 'Qwen2.5-Coder-7B-Instruct Apache-2.0',
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    if not keep_intermediate:
        intermediate.unlink()
    print(manifest_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=PROJECT / 'models/Qwen2.5-Coder-7B-Instruct')
    parser.add_argument('--destination', type=Path, default=PROJECT / 'models/RRL-Qwen2.5-Coder-7B-GGUF')
    parser.add_argument('--keep-intermediate', action='store_true')
    args = parser.parse_args()
    prepare(args.source.resolve(), args.destination.resolve(), keep_intermediate=args.keep_intermediate)


if __name__ == '__main__':
    main()
