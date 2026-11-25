"""Gate the frozen 500-report BLAgent experiment on an audited real smoke run."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT/'src'))
from bug_localization.batch import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    smoke=(PROJECT/args.smoke).resolve();folder=(PROJECT/args.output).resolve()
    folder.mkdir(parents=True,exist_ok=True)
    launcher_lock=(folder/'.automation.lock').open('w')
    fcntl.flock(launcher_lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
    # The smoke must finish before this launcher is invoked.
    subprocess.run([sys.executable,str(PROJECT/'scripts/analysis/audit_blagent.py'),'--run',str(smoke)],check=True,cwd=PROJECT)
    summary=json.loads((smoke/'summary.json').read_text())
    if summary['selected_samples']<3 or summary['failed_samples'] or summary['pending_samples']:
        raise ValueError('Require >=3 real successful cases before scaling')
    small=json.loads((smoke/'effective_config.json').read_text())['blagent']
    large=json.loads((PROJECT/'configs/experiments/blagent500.json').read_text())
    for key in small.keys()|large.keys():
        if key not in {'selection_manifest','repositories','limit'} and small.get(key)!=large.get(key):
            raise ValueError('Scientific configuration differs from the smoke run: '+key)
    provenance=json.loads((smoke/'provenance.json').read_text())
    from bug_localization.snapshot import sha256_file
    for path, checksum in provenance['identity']['implementation'].items():
        if sha256_file(PROJECT/'src/bug_localization'/path)!=checksum:
            raise ValueError('Implementation changed since smoke: '+path)
    folder.mkdir(parents=True,exist_ok=True)
    status={'status':'running','smoke':str(smoke),'started_at':datetime.now(timezone.utc).isoformat()}
    write_json(folder/'automation_status.json',status)
    command=[sys.executable,'-u',str(PROJECT/'scripts/retrieval/run_blagent.py'),'--config',
             'configs/experiments/blagent500.json','--output',str(folder)]
    with (folder/'run.log').open('a') as log:
        proc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,cwd=PROJECT)
    report=folder/'summary.json'
    if report.exists() and json.loads(report.read_text())['pending_samples']==0:
        audit=subprocess.run([sys.executable,str(PROJECT/'scripts/analysis/audit_blagent.py'),'--run',str(folder)],cwd=PROJECT)
        status['status']=json.loads(report.read_text())['status'] if audit.returncode==0 else 'audit_failed'
    else: status['status']='stopped_incomplete'
    status.update(exit_code=proc.returncode,finished_at=datetime.now(timezone.utc).isoformat())
    write_json(folder/'automation_status.json',status)
    print(json.dumps(status,indent=2))
    if status['status'] not in ('completed','completed_with_failures'): raise SystemExit(1)


if __name__=='__main__': main()
