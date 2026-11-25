"""Overlap completed-prefix reranking on GPU 1 with live retrieval on GPU 0."""
import argparse
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bug_localization.batch import write_json
from bug_localization.embeddings import digest
from bug_localization.reranking import read_json, run_reranking
from bug_localization.snapshot import sha256_file


def verify_protocol(folder):
    protocol = read_json(folder / 'experiment.json')
    for section in ('input_sha256', 'implementation_sha256'):
        for name, checksum in protocol[section].items():
            if sha256_file(Path(name)) != checksum:
                raise ValueError('Frozen experiment changed: ' + name)
    return protocol['config']


def export_partition(folder, name, start, stop):
    """Freeze metadata for committed cases; refer to immutable per-case artifacts."""
    base, target = folder / 'new320', folder / 'parallel' / name
    selection = read_json(base / 'semantic/selection.json')
    selected = selection['samples'][start:stop]
    originals = read_json(base / 'semantic/samples.json')['q0'][start:stop]
    hypotheses = read_json(base / 'hypothesis/samples.json')['direct_hypothesis'][start:stop]
    if not selected or len(originals) != len(selected) or len(hypotheses) != len(selected):
        raise ValueError('Partition is not fully committed')
    for meta, a, b in zip(selected, originals, hypotheses):
        if any(meta[k] != r[k] for r in (a, b) for k in ('position', 'dataset_row', 'repository', 'issue_id', 'before_fix_sha')):
            raise ValueError('Partition sample identity mismatch')
        checkpoint = read_json(base / 'semantic/cases' / f"{meta['position']:03d}" / 'checkpoint.json')
        if checkpoint['records']['q0'] != a or checkpoint['records']['direct_hypothesis'] != b:
            raise ValueError('Partition includes uncommitted records')
    frozen = {**selection, 'samples': selected, 'selection_sha256': digest(selected)}
    for kind in ('semantic', 'hypothesis'):
        path = target / kind
        path.mkdir(parents=True)
        write_json(path / 'selection.json', frozen)
        (path / 'cases').symlink_to(base / kind / 'cases', target_is_directory=True)
    write_json(target / 'semantic/effective_config.json', read_json(base / 'semantic/effective_config.json'))
    write_json(target / 'semantic/samples.json', {'q0': originals})
    write_json(target / 'hypothesis/samples.json', {'direct_hypothesis': hypotheses})
    write_json(target / 'hypothesis/audit.json', {
        'selection_verified': True, 'embedding_identity_matches': True,
        'candidate_and_chunk_inputs_match': True,
        'provenance': 'Committed original and hypothesis retrieval share the same source chunks and vectors',
    })
    for meta in selected:
        filename = f"{meta['position']:03d}_queries.json"
        (target / 'hypothesis' / filename).symlink_to(base / 'hypothesis' / filename)
    write_json(target / 'partition.json', {'start': start, 'stop': stop, 'selection_sha256': digest(selected)})


def worker(folder, partition, route):
    config = verify_protocol(folder)
    shard = folder / 'parallel' / partition
    model_config = read_json(PROJECT / config['reranker_config'])
    source = shard / ('semantic' if route == 'original' else 'hypothesis')
    if route == 'hypothesis':
        model_config.update(source_query_route='direct_hypothesis', semantic_assets_run=str(shard / 'semantic'),
                            reranker_query_source='original_report')
    output = shard / (route + '_results')
    kwargs = {'resume_folder': output} if (output / 'effective_config.json').exists() else {'output_folder': output}
    _, summary = run_reranking(PROJECT, source, model_config, ['blagent_ast'], **kwargs)
    if summary['status'] not in ('completed', 'completed_with_failures'):
        raise ValueError('Partition did not finish')


def ordered_records(pairs, expected):
    result = {}
    for variant in ('baseline', 'reranked'):
        rows = sorted([row for pair in pairs for row in pair[variant]], key=lambda row: row['position'])
        if len(rows) != len(expected) or [r['position'] for r in rows] != [s['position'] for s in expected]:
            raise ValueError('Missing/duplicate partition records')
        for row, meta in zip(rows, expected):
            if any(row[k] != meta[k] for k in ('dataset_row', 'repository', 'issue_id', 'before_fix_sha')):
                raise ValueError('Merged sample identity mismatch')
        result[variant] = rows
    return result


def merge(folder, config, partitions, route):
    base = folder / 'new320'
    source = base / ('semantic' if route == 'original' else 'hypothesis')
    selection = read_json(source / 'selection.json')
    outputs = [folder / 'parallel' / p / (route + '_results') for p in partitions]
    effective = read_json(outputs[0] / 'effective_config.json')
    for output in outputs:
        if read_json(output / 'effective_config.json')['model'] != effective['model']:
            raise ValueError('GPU workers have different model identities')
    records = ordered_records([read_json(p / 'samples.json')['blagent_ast'] for p in outputs], selection['samples'])
    config_model = read_json(PROJECT / config['reranker_config'])
    metadata_paths = [source / 'selection.json', base / 'semantic/effective_config.json']
    if route == 'hypothesis':
        config_model.update(source_query_route='direct_hypothesis', semantic_assets_run=str(base / 'semantic'),
                            reranker_query_source='original_report')
        metadata_paths.extend([base / 'semantic/selection.json', source / 'samples.json', source / 'audit.json'])
    effective.update(reranker=config_model, source_run=str(source), source_config=read_json(base / 'semantic/effective_config.json'),
        source_metadata_sha256={str(p): sha256_file(p) for p in metadata_paths},
        partition_runs=[str(p) for p in outputs], physical_gpus=[0, 1])
    target = base / ('original_reranking' if route == 'original' else 'reranking')
    staging = base / ('.merge_' + route)
    staging.mkdir()
    (staging / 'cases').mkdir()
    for output in outputs:
        for case in (output / 'cases').glob('*'):
            (staging / 'cases' / case.name).symlink_to(case, target_is_directory=True)
    write_json(staging / 'selection.json', selection)
    write_json(staging / 'effective_config.json', effective)
    write_json(staging / 'samples.json', {'blagent_ast': records})
    if target.exists():
        raise ValueError('Refusing to overwrite an existing full reranking run')
    staging.rename(target)


def supervise(folder):
    config = verify_protocol(folder)
    active, logs, finished = {}, [], []
    retrieval = None
    status = {'pid': os.getpid(), 'tmux_session': config['tmux_session'],
              'run': str(folder), 'physical_gpus': [0, 1]}
    def update(stage, **extra):
        status.update(stage=stage, updated_at=datetime.now(timezone(timedelta(hours=8))).isoformat(), **extra)
        write_json(folder / 'parallel_status.json', status)
        write_json(folder / 'automation_status.json', status)
    def stage_process(stage, gpu):
        log = (folder / (stage + '.log')).open('a'); logs.append(log)
        return subprocess.Popen([sys.executable, '-u', str(PROJECT / 'scripts/experiments/run_reranker500.py'),
            '--output', str(folder), '--stage', stage], cwd=PROJECT,
            env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu)}, stdout=log, stderr=subprocess.STDOUT)
    try:
        for name in ('original_reranking', 'reranking'):
            if (folder / 'new320' / name).exists():
                raise ValueError('Full reranking already started; refuse conflicting writers')
        prefix = folder / 'parallel/prefix/partition.json'
        if prefix.exists():
            ready = read_json(prefix)['stop']
        else:
            ready = min(len(read_json(folder / 'new320/semantic/samples.json')['q0']),
                        len(read_json(folder / 'new320/hypothesis/samples.json')['direct_hypothesis']))
            export_partition(folder, 'prefix', 0, ready)
        total = len(read_json(folder / 'new320/semantic/selection.json')['samples'])
        partitions = ['prefix']
        queue = [('prefix', 'original'), ('prefix', 'hypothesis')]
        retrieval_done = (folder / 'retrieve.completed.json').exists()
        if not retrieval_done:
            retrieval = stage_process('retrieve', 0)
        tail_exported = False
        while True:
            if retrieval is not None:
                code = retrieval.poll()
                if code is not None:
                    if code:
                        raise RuntimeError('Resumed retrieval failed; see retrieve.log')
                    retrieval_done = True
                    retrieval = None
                    write_json(folder / 'retrieve.completed.json', {'completed': True, 'physical_gpu': 0})
            if retrieval_done and not tail_exported:
                if ready < total:
                    export_partition(folder, 'tail', ready, total)
                    partitions.append('tail')
                    queue.extend([('tail', 'original'), ('tail', 'hypothesis')])
                tail_exported = True
            for gpu, item in list(active.items()):
                code = item['process'].poll()
                if code is not None:
                    if code:
                        raise RuntimeError(f"GPU {gpu} partition failed: {item['job']}")
                    finished.append({'gpu': gpu, 'partition': item['job'][0], 'route': item['job'][1]})
                    del active[gpu]
            available = [gpu for gpu in (1, 0) if gpu not in active and (gpu == 1 or retrieval_done)]
            for gpu in available:
                if not queue:
                    break
                partition, route = queue.pop(0)
                path = folder / 'parallel' / f'{partition}_{route}_gpu{gpu}.log'
                log = path.open('a'); logs.append(log)
                proc = subprocess.Popen([sys.executable, '-u', str(Path(__file__)), '--run', str(folder),
                    '--worker', partition, '--route', route], cwd=PROJECT,
                    env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu)}, stdout=log, stderr=subprocess.STDOUT)
                active[gpu] = {'process': proc, 'job': (partition, route)}
            update('retrieve_and_rerank_parallel' if not retrieval_done else 'rerank_parallel',
                committed_prefix=ready, retrieval_completed=retrieval_done,
                retrieval_pid=retrieval.pid if retrieval else None,
                workers={str(gpu): {'pid': item['process'].pid, 'partition': item['job'][0], 'route': item['job'][1]}
                         for gpu, item in active.items()}, finished=finished)
            if retrieval_done and not active and not queue:
                break
            time.sleep(5)
        update('merging_parallel_reranks')
        for route in ('original', 'hypothesis'):
            merge(folder, config, partitions, route)
        update('auditing_parallel_reranks')
        # Fully populated checkpoints: the frozen stages only regenerate summaries and audit.
        checks = [(stage, stage_process(stage, gpu)) for stage, gpu in
                  (('rerank_original', 0), ('rerank_hypothesis', 1))]
        for stage, proc in checks:
            if proc.wait():
                raise RuntimeError('Merged reranking audit failed: ' + stage)
            write_json(folder / (stage + '.completed.json'), {'completed': True, 'parallel_inference': True})
        update('summarizing')
        verify_protocol(folder)
        from run_reranker500 import finalize
        result = finalize(folder, config)
        update(result, exit_code=0 if result == 'completed' else 1)
    except BaseException as error:
        processes = [item['process'] for item in active.values()] + ([retrieval] if retrieval else [])
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()
        for proc in processes:
            proc.wait()
        update('failed', error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        for log in logs:
            log.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--worker', choices=['prefix', 'tail'])
    parser.add_argument('--route', choices=['original', 'hypothesis'])
    args = parser.parse_args()
    folder = (PROJECT / args.run).resolve()
    if args.worker:
        worker(folder, args.worker, args.route)
    else:
        supervise(folder)
