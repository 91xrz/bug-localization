"""Two BLAgent report workers sharing the unquantized BF16 inference service.

A fresh uniform protocol run; original failed/partial experiment is immutable.
"""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(PROJECT/'src'),str(PROJECT/'scripts/analysis')]
from bug_localization.batch import aggregate_results,write_checkpoints,write_json
from bug_localization.embeddings import digest
from bug_localization.snapshot import sha256_file
from audit_blagent import audit


def read(path):return json.loads(path.read_text())


def scientific(config):
    return {k:v for k,v in config.items() if k not in ('selection_manifest','repositories','limit')}


def partitions(selection, count=2):
    groups=[]
    for worker in range(count):
        original=list(range(worker,len(selection['samples']),count))
        samples=[dict(selection['samples'][index],position=i) for i,index in enumerate(original,1)]
        groups.append((original,{**selection,'samples':samples,'selection_sha256':digest(samples),
                                'selection_rule':'Deterministic strided worker partition of '+selection['selection_sha256']}))
    return groups


def merge(folder,selection,groups,config):
    records=[]
    identities=[]
    for number,(indices,_) in enumerate(groups):
        worker=folder/'workers'/str(number)
        if (worker/'provenance.json').exists():identities.append(read(worker/'provenance.json'))
        if not (worker/'samples.json').exists():continue
        for row in read(worker/'samples.json'):
            index=indices[row['position']-1];entry=selection['samples'][index]
            if any(row[k]!=entry[k] for k in ('dataset_row','repository','issue_id','before_fix_sha')):
                raise ValueError('Worker sample identity mismatch')
            case=folder/'cases'/f'{index+1:04d}';case.mkdir(parents=True,exist_ok=True)
            source=worker/'cases'/f'{row["position"]:04d}'
            rec={**row,'position':index+1,'source_worker':number,'source_position':row['position']}
            for artifact in source.iterdir():
                if artifact.name in ('result.json','record.json') or artifact.name.endswith('.tmp'):continue
                dest=case/artifact.name
                if not dest.exists():dest.symlink_to(artifact.resolve())
            if row['status']=='ok':
                if sha256_file(source/'result.json')!=row['result_sha256']:
                    raise ValueError('Worker result hash mismatch')
                result={**read(source/'result.json'),'position':index+1}
                write_json(case/'result.json',result)
                rec.update(result_file=(case/'result.json').relative_to(folder).as_posix(),
                           result_sha256=sha256_file(case/'result.json'))
            write_json(case/'record.json',rec)
            records.append(rec)
    records.sort(key=lambda r:r['position'])
    if len({r['position'] for r in records})!=len(records):raise ValueError('Duplicate merged positions')
    summary=aggregate_results(records,10,len(selection['samples']))
    if summary['status']=='completed' and summary['failed_samples']:summary['status']='completed_with_failures'
    summary.update(variant='BLAgent bounded-v2 + Qwen3-8B BF16 llama.cpp two-slot shared inference',workers=2)
    write_checkpoints(folder,records,summary)
    if len(identities)==2:
        first=identities[0]['identity'];second=identities[1]['identity']
        for key in ('generation','embedding','implementation','author_files_sha256','base'):
            if first[key]!=second[key]:raise ValueError('Worker scientific identities differ: '+key)
        # Preserve child manifests; merged identity uses canonical full selection/config.
        identity={**first,'config':config,'selection_sha256':selection['selection_sha256']}
        write_json(folder/'provenance.json',{'identity':identity,'identity_sha256':digest(identity),
                   'worker_provenance_sha256':[sha256_file(folder/'workers'/str(i)/'provenance.json') for i in range(2)]})
    return summary


def run(config_path,folder,gate=None):
    config=read(config_path);selection=read(PROJECT/config['selection_manifest'])
    if len(selection['samples'])<2:raise ValueError('Parallel run needs at least two reports')
    if gate:
        verified=audit(gate)
        if verified['failed'] or verified['samples']<3:raise ValueError('Recovery gate did not fully pass')
        if scientific(read(gate/'effective_config.json')['blagent'])!=scientific(config):
            raise ValueError('Recovery gate scientific configuration differs')
        for name,checksum in read(gate/'provenance.json')['identity']['implementation'].items():
            if sha256_file(PROJECT/'src/bug_localization'/name)!=checksum:raise ValueError('Implementation changed after gate')
    folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/'.parallel.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    protocol={'config':config,'selection_sha256':selection['selection_sha256'],
              'launcher_sha256':sha256_file(Path(__file__)),'audit_sha256':sha256_file(PROJECT/'scripts/analysis/audit_blagent.py')}
    if (folder/'protocol.json').exists() and read(folder/'protocol.json')!=protocol:
        raise ValueError('Parallel resume protocol changed')
    write_json(folder/'protocol.json',protocol);write_json(folder/'selection.json',selection)
    write_json(folder/'effective_config.json',{'blagent':config})
    groups=partitions(selection)
    processes=[];handles=[]
    status={'status':'running','started_at':datetime.now(timezone.utc).isoformat(),'workers':2,'gate':str(gate) if gate else None}
    write_json(folder/'automation_status.json',status)
    try:
        for number,(_,subset) in enumerate(groups):
            worker=folder/'workers'/str(number);worker.mkdir(parents=True,exist_ok=True)
            manifest=folder/f'selection_worker{number}.json';write_json(manifest,subset)
            child={**config,'selection_manifest':str(manifest)}
            child_path=folder/f'config_worker{number}.json';write_json(child_path,child)
            log=(worker/'run.log').open('a');handles.append(log)
            # Bound native BLAS thread fan-out while keeping GPU/model settings.
            env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
            processes.append(subprocess.Popen([sys.executable,'-u',str(PROJECT/'scripts/retrieval/run_blagent.py'),
                '--config',str(child_path),'--output',str(worker)],cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT))
        write_json(folder/'workers.json',[{'worker':i,'pid':p.pid} for i,p in enumerate(processes)])
        while any(p.poll() is None for p in processes):
            summary=merge(folder,selection,groups,config)
            status.update(attempted_samples=summary['attempted_samples'],successful_samples=summary['successful_samples'],
                          failed_samples=summary['failed_samples'],pending_samples=summary['pending_samples'])
            write_json(folder/'automation_status.json',status)
            time.sleep(10)
        summary=merge(folder,selection,groups,config)
        if summary['pending_samples']:
            status['status']='stopped_incomplete'
        else:
            audit(folder)
            status['status']=summary['status']
        status.update(exit_codes=[p.returncode for p in processes],finished_at=datetime.now(timezone.utc).isoformat(),
                      attempted_samples=summary['attempted_samples'],successful_samples=summary['successful_samples'],
                      failed_samples=summary['failed_samples'],pending_samples=summary['pending_samples'])
        write_json(folder/'automation_status.json',status)
        print(json.dumps(status,indent=2),flush=True)
        return status
    except BaseException as error:
        status.update(status='orchestrator_failed',error=repr(error))
        write_json(folder/'automation_status.json',status)
        for p in processes:
            if p.poll() is None:p.terminate()
        for p in processes:
            try:p.wait(timeout=15)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        raise
    finally:
        for h in handles:h.close()
        lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gate',type=Path)
    a=p.parse_args();status=run((PROJECT/a.config).resolve(),(PROJECT/a.output).resolve(),(PROJECT/a.gate).resolve() if a.gate else None)
    if status['status']!='completed':raise SystemExit(1)
