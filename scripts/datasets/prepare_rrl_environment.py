"""Prepare isolated RRL agent imports and the frozen author source checkout.

No packages are installed into the project's existing virtual environment.
The author's requirements are unversioned; pins here describe our adaptation.
"""

import argparse
from importlib.metadata import distributions
import json
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[2]
AUTHOR_REVISION = '50903acf3d6124514ff4f4c179cfd67a3e8ea414'
PACKAGES = {
    'pyserini': '0.44.0', 'pyjnius': '1.6.1',
    'python-dotenv': '1.1.1', 'tenacity': '9.1.2', 'gitpython': '3.1.45', 'unidiff': '0.7.5',
    'langchain-ollama': '0.3.8', 'langchain-text-splitters': '0.3.11', 'langchain-core': '0.3.86',
    'anthropic': '0.64.0', 'tiktoken': '0.11.0', 'sentencepiece': '0.2.1',
    'omegaconf': '2.3.0', 'antlr4-python3-runtime': '4.9.3',
}


def normalize(name):
    return name.lower().replace('_', '-')


def prepare():
    target = PROJECT / '.runtime/rrl-deps'
    target.mkdir(parents=True, exist_ok=True)
    installed = {normalize(d.metadata['Name']): d.version for d in distributions(path=[str(target)])}
    missing = [f'{name}=={version}' for name, version in PACKAGES.items() if installed.get(name) != version]
    if missing:
        # Pyserini's optional dense stack would install another Torch and NumPy.
        # Install its two retrieval packages separately without those extras.
        bare = [name for name in missing if name.split('==')[0] in ('pyserini', 'pyjnius')]
        normal = [name for name in missing if name not in bare]
        if bare:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--target', str(target), '--no-deps', *bare], check=True)
        if normal:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--target', str(target), *normal], check=True)
    checkout = PROJECT / '.runtime/external-baselines/rrl'
    if not checkout.exists():
        checkout.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', '--filter=blob:none', '--no-checkout',
                        'https://github.com/gencau/boatse-query-reformulation.git', str(checkout)], check=True)
        subprocess.run(['git', '-C', str(checkout), 'sparse-checkout', 'set',
                        'bug_localization/src', 'bug_localization/configs', 'bug_localization/datasets'], check=True)
        subprocess.run(['git', '-C', str(checkout), 'checkout', '--detach', AUTHOR_REVISION], check=True)
    revision = subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != AUTHOR_REVISION:
        raise ValueError('Existing author checkout is not the pinned RRL revision')
    sys.path[:0] = [str(target), str(PROJECT / 'src')]
    from bug_localization.rrl_index import set_runtime
    runtime = set_runtime(PROJECT)
    installed = {d.metadata['Name']: d.version for d in distributions(path=[str(target)])}
    record = {'author_revision': revision, 'direct_pins': PACKAGES,
              'isolated_distributions': installed, 'runtime': runtime,
              'shared_dependencies': 'Existing project numpy, torch, transformers, pandas and datasets remain unchanged'}
    (target / 'rrl-environment.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(runtime, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    prepare()
