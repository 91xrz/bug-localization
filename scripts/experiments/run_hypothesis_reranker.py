"""Fixed 180-case direct-hypothesis + reranker experiment with four paired controls."""

import argparse
import csv
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import subprocess
import sys
import traceback

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'src'))
from bug_localization.batch import aggregate_results, write_json
from bug_localization.reranking import read_json, run_reranking, validate_config
from bug_localization.snapshot import sha256_file


def preflight(config):
    if config.get('reranker_query_source') != 'original_report':
        raise ValueError('The reranker must use the original report, not the generated hypothesis')
    selection = read_json(PROJECT / config['selection'])
    if len(selection['samples']) != 180:
        raise ValueError('This experiment requires the frozen 180-case selection')
    for key in ('hypothesis_run', 'semantic_assets_run', 'original_reranker_run'):
        if read_json(PROJECT / config[key] / 'selection.json')['samples'] != selection['samples']:
            raise ValueError('Control selection differs: ' + key)
    control = PROJECT / config['original_reranker_run']
    if read_json(control / 'automation_status.json')['stage'] != 'completed':
        raise ValueError('Original-report reranker control has not completed its audit')
    reranker = read_json(PROJECT / config['reranker_config'])
    if reranker != read_json(control / 'effective_config.json')['reranker']:
        raise ValueError('Reranker parameters differ from the original-report control')
    validate_config(reranker)
    direct = PROJECT / config['hypothesis_run']
    for meta in selection['samples']:
        pos = f"{meta['position']:03d}"
        query = read_json(direct / (pos + '_queries.json'))
        if query['generation_error'] is not None or not query['query'].strip():
            raise ValueError('Missing/failed hypothesis: ' + pos)
        for path in (direct / 'cases' / pos / 'direct_result.json', direct / 'cases' / pos / 'direct_retrieval.json'):
            if not path.is_file():
                raise ValueError('Missing hypothesis retrieval: ' + str(path))
    return selection, reranker


def compare(folder, config):
    for path, checksum in read_json(folder / 'experiment.json')['input_metadata_sha256'].items():
        if sha256_file(Path(path)) != checksum:
            raise ValueError('Control metadata changed during experiment: ' + path)
    selection = read_json(folder / 'selection.json')['samples']
    combined = read_json(folder / 'reranking/samples.json')['blagent_ast']
    control_path = PROJECT / config['original_reranker_run']
    if read_json(control_path / 'effective_config.json')['model'] != read_json(folder / 'reranking/effective_config.json')['model']:
        raise ValueError('Reranker model/environment differs between original and hypothesis controls')
    control = read_json(control_path / 'samples.json')['blagent_ast']
    records = {'original': control['baseline'], 'original_reranked': control['reranked'],
               'hypothesis': combined['baseline'], 'hypothesis_reranked': combined['reranked']}
    for name, rows in records.items():
        if len(rows) != len(selection) or any(row['status'] != 'ok' for row in rows):
            raise ValueError('Incomplete control: ' + name)
        for meta, row in zip(selection, rows):
            if any(meta[key] != row[key] for key in ('position', 'dataset_row', 'repository', 'issue_id', 'before_fix_sha')):
                raise ValueError('Control sample identity mismatch: ' + name)
    for left, right in zip(control['reranked'], combined['reranked']):
        a, b = read_json(control_path / left['result_file']), read_json(folder / 'reranking' / right['result_file'])
        if a['candidate_fingerprint'] != b['candidate_fingerprint'] or a['ground_truth'] != b['ground_truth']:
            raise ValueError('Original and hypothesis candidate/label mismatch')
        if b.get('reranker_query_source') != 'original_report' or a['query'] != b['query']:
            raise ValueError('Combined reranker query must exactly match the original-report control')
    metrics = {name: aggregate_results(rows, 10) for name, rows in records.items()}
    pairs = {}
    for name, rows in records.items():
        if name == 'hypothesis_reranked':
            continue
        deltas = [b['metrics']['average_precision_at_k'] - a['metrics']['average_precision_at_k']
                  for a, b in zip(rows, records['hypothesis_reranked'])]
        pairs[name] = {'wins': sum(x > 1e-12 for x in deltas), 'ties': sum(abs(x) <= 1e-12 for x in deltas),
                       'losses': sum(x < -1e-12 for x in deltas), 'delta_MAP@10': sum(deltas) / len(deltas)}
    projects = {repo: {name: aggregate_results([r for r in rows if r['repository'] == repo], 10)
                       for name, rows in records.items()} for repo in sorted({r['repository'] for r in selection})}
    final = {'status': 'completed', 'selected_samples': len(selection), 'methods': metrics,
             'hypothesis_reranked_vs': pairs, 'candidate_and_labels_verified': True,
             'reranker_query_source': 'original_report', 'same_query_as_original_reranker_verified': True,
             'query_policy': config['query_policy'],
             'verification': read_json(folder / 'reranking/verification.json')}
    write_json(folder / 'final_summary.json', final)
    write_json(folder / 'project_summary.json', projects)
    write_json(folder / 'paired_samples.json', records)
    fields = ['method', 'MAP@10', 'MRR@10', 'Top-1', 'Top-5', 'Top-10', 'MeanRecall@10']
    with (folder / 'comparison.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, stats in metrics.items():
            writer.writerow({'method': name, **stats['metrics']})
    with (folder / 'project_comparison.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['repository'] + fields)
        writer.writeheader()
        for repo, methods in projects.items():
            for name, stats in methods.items():
                writer.writerow({'repository': repo, 'method': name, **stats['metrics']})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PROJECT / 'configs/experiments/hypothesis_reranker180.json')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    config = read_json(PROJECT / args.config)
    selection, reranker = preflight(config)
    if args.check_only:
        print('Verified 180 frozen samples, four-control inputs, generated hypotheses and unchanged reranker settings.')
        return
    folder = PROJECT / args.output if args.output else PROJECT / 'results/hypothesis_reranking' / datetime.now(
        timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
    folder.mkdir(parents=True)
    write_json(folder / 'selection.json', selection)
    write_json(folder / 'experiment.json', {'config': config, 'reranker': reranker,
        'implementation_sha256': sha256_file(Path(__file__)),
        'input_metadata_sha256': {str(PROJECT / config[key] / name): sha256_file(PROJECT / config[key] / name)
            for key in ('hypothesis_run', 'semantic_assets_run', 'original_reranker_run')
            for name in ('selection.json', 'samples.json')}})
    status = {'pid': os.getpid(), 'run': str(folder), 'tmux_session': config['tmux_session']}

    def update(stage, **extra):
        status.update(stage=stage, updated_at=datetime.now(timezone(timedelta(hours=8))).isoformat(), **extra)
        write_json(folder / 'automation_status.json', status)

    try:
        update('reranking')
        _, summary = run_reranking(PROJECT, Path(config['hypothesis_run']), {
            **reranker, 'source_query_route': 'direct_hypothesis', 'semantic_assets_run': config['semantic_assets_run'],
            'reranker_query_source': config['reranker_query_source']},
            ['blagent_ast'], output_folder=folder / 'reranking')
        if summary['status'] != 'completed':
            raise RuntimeError('Reranking contains failures; inspect reranking/samples.json')
        update('auditing')
        subprocess.run([sys.executable, str(PROJECT / 'scripts/analysis/analyze_reranking.py'),
            '--run', str(folder / 'reranking'), '--selection', str(PROJECT / config['selection'])], check=True, cwd=PROJECT)
        update('comparing')
        compare(folder, config)
        update('completed', exit_code=0)
    except BaseException as error:
        update('failed', exit_code=1, error=str(error), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
