"""Prepare frozen GenLoc source, CodeXEmbed weights and isolated dependencies."""

import argparse
import hashlib
from importlib.metadata import distributions
import json
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[2]
AUTHOR_REPOSITORY = 'https://github.com/seal-hub/Towards-Explorative-IRBL.git'
AUTHOR_REVISION = '92e4dd4aea0b35f31b80f160e1f59e643a81a71d'
MODEL_REPOSITORY = 'Salesforce/SFR-Embedding-Code-400M_R'
MODEL_REVISION = 'cb950dc80d677c6fdc00f56c8ddd20ca2642c59e'
MODEL_WEIGHTS_SHA256 = 'f383222e9b1b8fbb915295d01c6da2df8769bb319854e5986d934f0aa08d6703'
CODE_REPOSITORY = 'Alibaba-NLP/new-impl'
CODE_REVISION = '40ced75c3017eb27626c9d4ea981bde21a2662f4'
MODEL_FILES = ['README.md', 'config.json', 'configuration.py', 'model.safetensors',
               'modules.json', '1_Pooling/config.json', 'sentence_bert_config.json',
               'special_tokens_map.json', 'tokenizer.json', 'tokenizer_config.json', 'vocab.txt']
CODE_FILES = ['configuration.py', 'modeling.py', 'README.md']
PINS = {'tree-sitter': '0.25.0', 'tree-sitter-language-pack': '0.9.0', 'rapidfuzz': '3.10.1',
        'tree-sitter-c-sharp': '0.23.1', 'tree-sitter-embedded-template': '0.23.2',
        'tree-sitter-yaml': '0.7.0', 'transformers': '4.57.3',
        'tokenizers': '0.22.1', 'huggingface-hub': '0.36.0',
        'jsonschema': '4.25.1', 'jsonschema-specifications': '2025.9.1',
        'referencing': '0.37.0', 'rpds-py': '2026.9.1', 'datasets': '4.4.1',
        'dill': '0.4.0', 'multiprocess': '0.70.18', 'fsspec': '2025.10.0'}
SHARED_PINS = {'tiktoken': '0.11.0', 'langchain-text-splitters': '0.3.11'}


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def git(checkout, *args):
    return subprocess.check_output(['git', '-C', str(checkout), *args], text=True).strip()


def prepare_source(project):
    checkout = project / '.runtime/external-baselines/genloc-code'
    if not checkout.exists():
        checkout.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', '--filter=blob:none', '--no-checkout',
                        AUTHOR_REPOSITORY, str(checkout)], check=True)
    # A just-cloned --no-checkout repository reports staged deletions. Never
    # replace an existing populated checkout with edits or a different revision.
    populated = (checkout / 'source-code').exists()
    if populated:
        if git(checkout, 'rev-parse', 'HEAD') != AUTHOR_REVISION:
            raise ValueError('Existing GenLoc checkout is not the frozen revision')
        if git(checkout, 'status', '--porcelain', '--untracked-files=no'):
            raise ValueError('Existing GenLoc author source has local modifications')
    else:
        subprocess.run(['git', '-C', str(checkout), 'sparse-checkout', 'set',
                        '--cone', 'source-code', 'docs'], check=True)
        subprocess.run(['git', '-C', str(checkout), 'checkout', '--detach', AUTHOR_REVISION], check=True)
    files = [p for folder in ('source-code', 'docs') for p in (checkout / folder).rglob('*')
             if p.is_file() and '__pycache__' not in p.parts]
    record = {'repository': AUTHOR_REPOSITORY, 'revision': AUTHOR_REVISION,
              'files_sha256': {p.relative_to(checkout).as_posix(): sha256(p) for p in sorted(files)}}
    write_json(project / '.runtime/genloc-source-manifest.json', record)
    return record


def package_versions(paths):
    return {d.metadata['Name'].lower().replace('_', '-'): d.version
            for d in distributions(path=[str(p) for p in paths])}


def prepare_dependencies(project):
    target = project / '.runtime/genloc-deps'
    shared = [project / '.runtime/rrl-deps', project / '.runtime/igsil-deps']
    present = package_versions([target])
    needed = [f'{name}=={pin}' for name, pin in PINS.items() if present.get(name) != pin]
    existing = package_versions([target, *shared])
    needed += [f'{name}=={pin}' for name, pin in SHARED_PINS.items() if existing.get(name) != pin]
    if needed:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--target', str(target),
                        '--no-deps', *needed], check=True)
    paths = [target, *shared]
    # Validate in a clean interpreter so an already imported main-environment
    # tree-sitter cannot mask the isolated version.
    probe = """import sys, json
sys.path[:0] = sys.argv[1:]
from importlib.metadata import version
from tree_sitter_language_pack import get_parser
import tiktoken, langchain_text_splitters, rapidfuzz
tree = get_parser('python').parse(b'def example():\\n    return 1\\n')
assert tree.root_node.type == 'module'
assert tiktoken.get_encoding('cl100k_base').encode('example')
import transformers, datasets
from pathlib import Path
local_dataset = Path(sys.argv[1]).parents[1] / 'datasets/swe_bench_formatted'
if local_dataset.exists():
    assert len(datasets.load_from_disk(str(local_dataset))['test']) > 0
print(json.dumps({n:version(n) for n in ['tree-sitter','tree-sitter-language-pack','rapidfuzz','tiktoken','langchain-text-splitters','transformers','tokenizers','huggingface-hub','datasets','dill','multiprocess','fsspec']}))
"""
    result = json.loads(subprocess.check_output([sys.executable, '-c', probe, *map(str, paths)], text=True))
    environment_path = target / 'environment.json'
    record = json.loads(environment_path.read_text()) if environment_path.exists() else {}
    record.update(pins={**record.get('pins', {}), **PINS},
                  shared_pins={**record.get('shared_pins', {}), **SHARED_PINS},
                  sys_path=list(map(str, paths)),
                  validated_versions={**record.get('validated_versions', {}), **result},
                  main_environment_modified=False)
    write_json(target / 'environment.json', record)
    return record


def prepare_model(project):
    import os
    # The XET route stalled on this host; use the verified HTTP asset route.
    os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
    from huggingface_hub import snapshot_download
    destination = project / 'models/SFR-Embedding-Code-400M_R'
    code_destination = destination / 'custom_code'
    existing_manifest = destination / 'manifest.json'
    if existing_manifest.exists():
        existing = json.loads(existing_manifest.read_text())
        if existing.get('model_revision') != MODEL_REVISION or existing.get('code_revision') != CODE_REVISION:
            raise ValueError('Existing CodeXEmbed manifest has different frozen revisions')
        for name, expected in existing['files_sha256'].items():
            if sha256(destination / name) != expected:
                raise ValueError('Existing CodeXEmbed asset changed: ' + name)
    # Explicit allowlists omit all ONNX variants and keep the original config.
    snapshot_download(MODEL_REPOSITORY, revision=MODEL_REVISION, local_dir=destination,
                      allow_patterns=MODEL_FILES, token=False, max_workers=4)
    snapshot_download(CODE_REPOSITORY, revision=CODE_REVISION, local_dir=code_destination,
                      allow_patterns=CODE_FILES, token=False, max_workers=3)
    files = MODEL_FILES + ['custom_code/' + p for p in CODE_FILES]
    hashes = {p: sha256(destination / p) for p in files}
    if hashes['model.safetensors'] != MODEL_WEIGHTS_SHA256:
        raise ValueError('CodeXEmbed weights differ from the frozen HF LFS SHA-256')
    record = {'model_repository': MODEL_REPOSITORY, 'model_revision': MODEL_REVISION,
              'code_repository': CODE_REPOSITORY, 'code_revision': CODE_REVISION,
              'license': 'cc-by-nc-4.0', 'files_sha256': hashes,
              'original_config_unmodified': True,
              'custom_code_loading': 'Explicit frozen local NewConfig/NewModel classes; no runtime network imports',
              'pooling': 'last_hidden_state[:, 0] (CLS)', 'normalization': 'L2',
              'max_tokens': 8192, 'omitted_assets': ['onnx/*', '.DS_Store']}
    write_json(destination / 'manifest.json', record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-model', action='store_true')
    parser.add_argument('--skip-dependencies', action='store_true')
    args = parser.parse_args()
    summary = {'source': prepare_source(PROJECT)}
    if not args.skip_dependencies:
        summary['dependencies'] = prepare_dependencies(PROJECT)
    if not args.skip_model:
        summary['model'] = prepare_model(PROJECT)
    write_json(PROJECT / '.runtime/genloc-preparation.json', summary)
    print(json.dumps({'source_revision': AUTHOR_REVISION, 'model_revision': MODEL_REVISION,
                      'code_revision': CODE_REVISION, 'prepared': list(summary)}, indent=2))


if __name__ == '__main__':
    main()
