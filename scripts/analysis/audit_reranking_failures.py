"""Audit successful cases without removing failed cases from primary denominators."""
import argparse
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'src'))
from bug_localization.batch import aggregate_results, write_json
from bug_localization.embeddings import digest
from bug_localization.reranking import read_json
from analyze_reranking import analyze


def audit(folder, expected_selection):
    selection = read_json(folder / 'selection.json')
    if selection['samples'] != read_json(expected_selection)['samples']:
        raise ValueError('Full selection differs from frozen manifest')
    summary, records = read_json(folder / 'summary.json'), read_json(folder / 'samples.json')
    if summary['status'] not in ('completed', 'completed_with_failures'):
        raise ValueError('Run is incomplete')
    verified = {}
    for route, pair in records.items():
        count = len(selection['samples'])
        for name in ('baseline', 'reranked'):
            if len(pair[name]) != count:
                raise ValueError('Incomplete route')
            recalculated = aggregate_results(pair[name], summary['evaluation_k'], count)
            for key in ('metrics', 'metrics_successful_only', 'failed_samples', 'successful_samples'):
                if recalculated[key] != summary['routes'][route][name][key]:
                    raise ValueError('Primary summary/denominator mismatch: ' + key)
            for meta, row in zip(selection['samples'], pair[name]):
                if any(meta[key] != row[key] for key in ('position', 'dataset_row', 'repository', 'issue_id', 'before_fix_sha')):
                    raise ValueError('Full sample identity mismatch')
        good = [i for i in range(count) if all(pair[name][i]['status'] == 'ok' for name in ('baseline', 'reranked'))]
        if not good:
            raise ValueError('No successful pairs to audit')
        view = folder / ('successful_subset_audit_' + route)
        view.mkdir(exist_ok=True)
        if not (view / 'cases').exists():
            (view / 'cases').symlink_to(folder / 'cases', target_is_directory=True)
        subset = [selection['samples'][i] for i in good]
        subset_records = {name: [pair[name][i] for i in good] for name in ('baseline', 'reranked')}
        write_json(view / 'selection.json', {**selection, 'samples': subset, 'selection_sha256': digest(subset),
            'scope': 'Diagnostic successful pairs only; primary metrics remain on the full selection'})
        write_json(view / 'effective_config.json', read_json(folder / 'effective_config.json'))
        write_json(view / 'samples.json', {route: subset_records})
        write_json(view / 'summary.json', {'status': 'completed', 'selected_samples': len(good),
            'evaluation_k': summary['evaluation_k'], 'routes': {route: {
                name: aggregate_results(rows, summary['evaluation_k']) for name, rows in subset_records.items()}}})
        analyze(view)
        verified[route] = {**read_json(view / 'verification.json')['routes'][route],
            'full_selected_samples': count, 'primary_denominator_verified': True,
            'primary_metrics_recomputed_with_failures_as_zero': True,
            'unscored_positions': [selection['samples'][i]['position'] for i in range(count) if i not in good],
            'scope': 'All successful paired rankings verified; failed samples retained as zero in primary metrics'}
    result = {'routes': verified, 'expected_selection_verified': True}
    write_json(folder / 'verification_with_failures.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    args = parser.parse_args()
    audit((PROJECT / args.run).resolve(), (PROJECT / args.selection).resolve())
