"""Prepare pinned BLAgent source and isolated compatibility dependencies."""
import argparse
from importlib.metadata import distributions
import json
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[2]
REVISION = '9f91f63e50aec8ad87a11e11148b086dfcd5e66d'
PACKAGES = {'accelerate':'1.10.1','langchain':'0.3.27','langchain-anthropic':'0.3.19',
            'langchain-openai':'0.3.30','openai':'1.99.9','psutil':'7.0.0'}
FILES = ['README.md','requirements.txt','blagent/agent/ranker_agent.py','blagent/agent/prompts/__init__.py',
         'blagent/util/query_augmentation.py','blagent/util/signature_extractor.py',
         'blagent/util/code_util.py','blagent/util/code_pruner.py',
         'pipelines/query_transformation.py','pipelines/agentic_pipeline.py',
         'blagent/chroma/chroma_store.py','blagent/retriever/code_retriever.py',
         'evaluation/ranking_evaluation.py']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    checkout=PROJECT/'.runtime/external-baselines/blagent-code'
    if not checkout.exists():
        subprocess.run(['git','clone','--filter=blob:none','--no-checkout',
                        'https://github.com/afifaniks/BLAgent.git',str(checkout)],check=True)
        subprocess.run(['git','-C',str(checkout),'sparse-checkout','set','--no-cone','--stdin'],
                        input='\n'.join('/'+p for p in FILES)+'\n',text=True,check=True)
        subprocess.run(['git','-C',str(checkout),'checkout','--detach',REVISION],check=True)
    actual=subprocess.check_output(['git','-C',str(checkout),'rev-parse','HEAD'],text=True).strip()
    if actual!=REVISION: raise ValueError('Existing BLAgent checkout is not the pinned revision')
    target=PROJECT/'.runtime/blagent-deps'
    installed={d.metadata['Name'].lower():d.version for d in distributions(path=[str(target)])}
    missing=[f'{k}=={v}' for k,v in PACKAGES.items() if installed.get(k)!=v]
    if missing:
        subprocess.run([sys.executable,'-m','pip','install','--target',str(target),'--no-deps',*missing],check=True)
    sys.path[:0]=[str(PROJECT/p) for p in ('.runtime/blagent-deps','.runtime/rrl-deps','.runtime/igsil-deps','src')]
    from bug_localization.blagent import author_modules
    _, hashes=author_modules(PROJECT,{'author_checkout':str(checkout)})
    from accelerate import infer_auto_device_map
    import psutil
    record={'revision':actual,'author_files_sha256':hashes,'pins':PACKAGES,
            'shared_dependencies':'Existing project .venv, .runtime/rrl-deps and .runtime/igsil-deps; main environment unchanged',
            'installed':{d.metadata['Name']:d.version for d in distributions(path=[str(target)])}}
    (target/'environment.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__': main()
