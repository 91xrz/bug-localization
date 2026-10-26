"""Rank-free route voting over saved, verified lexical/semantic file lists."""

from collections import Counter, defaultdict
import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .batch import aggregate_results, write_checkpoints, write_json
from .bm25 import ranking_metrics
from .comparison import paired_records, read_json
from .hybrid import ranked_paths
from .snapshot import sha256_file
from .vector import rrf


def vote_files(rankings, depth=None):
    """Each route casts at most one vote per file; ranks/scores are not read.

    Optional depth chooses the first N unique candidates before voting. Once
    that membership is fixed, ordering inside either list has no effect.
    """
    if depth is not None and (type(depth) is not int or depth < 1):
        raise ValueError('Voting depth must be a positive integer or None for all files')
    if not rankings:
        raise ValueError('At least one voting route is required')
    membership = defaultdict(list)
    for route, rows in rankings.items():
        paths = list(dict.fromkeys(row['file_path'] for row in rows))
        for path in paths if depth is None else paths[:depth]:
            membership[path].append(route)
    groups = Counter(len(routes) for routes in membership.values())
    starts, offset = {}, 0
    for votes, size in sorted(groups.items(), reverse=True):
        starts[votes] = offset + 1
        offset += size
    ordered = sorted(membership, key=lambda path: (-len(membership[path]), path))
    return [{'rank': i, 'file_path': path, 'score': len(membership[path]), 'votes': len(membership[path]),
             'voting_routes': sorted(membership[path]), 'tie_group_size': groups[len(membership[path])],
             'tie_rank_start': starts[len(membership[path])],
             'tie_rank_end': starts[len(membership[path])] + groups[len(membership[path])] - 1}
            for i, path in enumerate(ordered, 1)]


def expected_tie_metrics(ranking, ground_truth, k=10):
    """Exact expected original metrics under uniform permutations inside ties.

    Gold labels are used only for evaluation, never for scoring or display
    order. This is a diagnostic, separate from the path-tiebroken ranking.
    """
    if not ground_truth or k < 10:
        raise ValueError('Nonempty ground truth and cutoff >= 10 are required')
    grouped = defaultdict(list)
    for row in ranking:
        grouped[row['votes']].append(row['file_path'])
    gold, offset, earlier_gold, expected_ap, expected_retrieved = set(ground_truth), 0, 0, 0.0, 0.0
    first_gold_group = None
    for _, paths in sorted(grouped.items(), reverse=True):
        m, g = len(paths), len(set(paths) & gold)
        if g and first_gold_group is None:
            first_gold_group = offset, m, g
        take = min(m, max(0, k - offset))
        expected_retrieved += take * g / m
        for j in range(1, take + 1):
            # E[I_j * relevant_so_far] includes correlations within this tie.
            numerator = g / m * (earlier_gold + 1)
            if m > 1:
                numerator += (j - 1) * g * (g - 1) / (m * (m - 1))
            expected_ap += numerator / (offset + j)
        earlier_gold += g
        offset += m
    hits = {cut: 0.0 for cut in (1, 5, 10)}
    reciprocal = 0.0
    if first_gold_group is not None:
        before, m, g = first_gold_group
        for cut in hits:
            survival = 1.0
            for j in range(1, min(m, max(0, min(cut, k) - before)) + 1):
                survival *= (m - g - j + 1) / (m - j + 1)
                if survival == 0:
                    break
            hits[cut] = 1 - survival
        survival = 1.0
        for j in range(1, min(m, max(0, k - before)) + 1):
            reciprocal += survival * g / (m - j + 1) / (before + j)
            survival *= (m - g - j + 1) / (m - j + 1)
            if survival == 0:
                break
    return {'top_1': hits[1], 'top_5': hits[5], 'top_10': hits[10],
            'recall_at_k': expected_retrieved / len(gold), 'reciprocal_rank_at_k': reciprocal,
            'average_precision_at_k': expected_ap / len(gold), 'k': k,
            'gold_files': len(gold), 'retrieved_gold_files': expected_retrieved}


def run_voting(project, source, labels=None, depths=(None, 10)):
    summary = read_json(source / 'summary.json')
    if summary['status'] != 'completed' or summary['evaluation_k'] != 10:
        raise ValueError('Use a completed verified hybrid run evaluated at @10')
    if not read_json(source / 'verification.json')['all_samples_paired_and_verified']:
        raise ValueError('Verify input runs before voting')
    labels = list(labels) if labels else [name for name in summary['routes']
                                         if name != 'bm25' and not name.startswith('rrf_')]
    if not labels or len(set(labels)) != len(labels) or any(label not in summary['routes'] or label == 'bm25' for label in labels):
        raise ValueError('Select distinct semantic baseline labels from the input run')
    if not depths or len(set(depths)) != len(depths):
        raise ValueError('Use distinct candidate depths')
    for depth in depths:
        vote_files({'empty': []}, depth)  # Validate before writing outputs.
    data = {name: read_json(source / name / 'samples.json') for name in ['bm25'] + labels}
    paired_records(data['bm25'], data, labels)
    selection = read_json(source / 'selection.json')
    n = len(selection['samples'])
    if any(len(rows) != n or any(r['status'] != 'ok' for r in rows) for rows in data.values()):
        raise ValueError('Voting requires complete successful input runs')
    folder = project / 'results/voting' / datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
    folder.mkdir(parents=True)
    write_json(folder / 'selection.json', selection)
    write_json(folder / 'effective_config.json', {
        'source_run': str(source), 'source_summary_sha256': sha256_file(source / 'summary.json'),
        'semantic_labels': labels, 'depths': ['all' if d is None else d for d in depths],
        'votes_per_route': 1, 'tie_break': 'Relative file path ascending; no input rank/score fallback',
        'expected_metrics': 'Exact expectation over uniform permutations inside each equal-vote group',
        'evaluation_k': 10, 'implementation_sha256': sha256_file(Path(__file__))})
    records, expected, diagnostics, flat, source_hashes, controls = {}, {}, defaultdict(list), [], [], {}
    for i in range(n):
        inputs = {name: read_json(source / rows[i]['result_file']) for name, rows in data.items()}
        for name, value in inputs.items():
            if (value['before_fix_sha'] != inputs['bm25']['before_fix_sha'] or
                    value['ground_truth'] != inputs['bm25']['ground_truth'] or
                    ranking_metrics(ranked_paths(value['ranking']), value['ground_truth'], 10) != data[name][i]['metrics']):
                raise ValueError('Input snapshot, labels or saved metrics changed')
            source_hashes.append({'position': i + 1, 'route': name,
                                  'sha256': sha256_file(source / data[name][i]['result_file'])})
        gold = inputs['bm25']['ground_truth']
        for label in labels:
            for depth in depths:
                name = 'vote_' + ('all' if depth is None else 'top' + str(depth)) + '_' + label
                ranking = vote_files({'bm25': inputs['bm25']['ranking'], label: inputs[label]['ranking']}, depth)
                actual_metrics = ranking_metrics([r['file_path'] for r in ranking], gold, 10)
                expectation = expected_tie_metrics(ranking, gold)
                case = folder / 'cases' / f'{i + 1:03d}_{name}.json'
                case.parent.mkdir(exist_ok=True)
                write_json(case, {'route': name, 'repository': data[label][i]['repository'],
                    'issue_id': data[label][i]['issue_id'], 'before_fix_sha': data[label][i]['before_fix_sha'],
                    'ground_truth': gold, 'missing_gold': inputs[label]['missing_gold'],
                    'ranking': [{**r, 'is_ground_truth': r['file_path'] in gold} for r in ranking],
                    'metrics_path_tie_break': actual_metrics, 'metrics_uniform_tie_expectation': expectation})
                record = {**{k: data[label][i][k] for k in ('position', 'dataset_row', 'repository', 'issue_id', 'before_fix_sha', 'indexed_files')},
                          'status': 'ok', 'metrics': actual_metrics, 'missing_gold': inputs[label]['missing_gold'],
                          'result_file': case.relative_to(folder).as_posix()}
                records.setdefault(name, []).append(record)
                expected.setdefault(name, []).append({**record, 'metrics': expectation})
                if depth is not None:
                    # A rank-based control over exactly the same candidate
                    # lists separates the scoring rule from truncation effects.
                    control_name = 'rrf_top' + str(depth) + '_' + label
                    control = rrf([inputs['bm25']['ranking'][:depth], inputs[label]['ranking'][:depth]], 60)
                    control_file = folder / 'cases' / f'{i + 1:03d}_{control_name}.json'
                    control_metrics = ranking_metrics([r['file_path'] for r in control], gold, 10)
                    write_json(control_file, {'route': control_name, 'rrf_constant': 60,
                        'candidate_depth_per_route': depth, 'issue_id': record['issue_id'],
                        'ground_truth': gold, 'missing_gold': inputs[label]['missing_gold'],
                        'ranking': [{**r, 'is_ground_truth': r['file_path'] in gold} for r in control],
                        'metrics': control_metrics})
                    controls.setdefault(control_name, []).append({**record, 'metrics': control_metrics,
                        'result_file': control_file.relative_to(folder).as_posix()})
                size = ranking[0]['tie_group_size'] if ranking else 0
                diagnostics[name].append({'issue_id': record['issue_id'], 'candidate_union': len(ranking),
                    'two_vote_files': sum(r['votes'] == 2 for r in ranking), 'one_vote_files': sum(r['votes'] == 1 for r in ranking),
                    'top_vote_group_size': size, 'top10_has_tie': any(r['tie_group_size'] > 1 for r in ranking[:10])})
                flat.extend({'route': name, 'issue_id': record['issue_id'], 'position': i + 1,
                             **{k: r[k] for k in ('rank', 'file_path', 'votes', 'tie_group_size', 'tie_rank_start', 'tie_rank_end')},
                             'voting_routes': ';'.join(r['voting_routes']), 'is_ground_truth': r['file_path'] in gold} for r in ranking)
    output = {'status': 'completed', 'selected_samples': n, 'selection_sha256': summary['selection_sha256'],
        'evaluation_k': 10, 'baseline_routes': summary['routes'],
        'routes': {name: aggregate_results(rows, 10) for name, rows in records.items()},
        'uniform_tie_expectation_routes': {name: aggregate_results(rows, 10)['metrics'] for name, rows in expected.items()},
        'matched_depth_rrf_routes': {name: aggregate_results(rows, 10) for name, rows in controls.items()},
        'tie_diagnostics': dict(diagnostics),
        'scope': 'Exploratory route voting; path ordering inside ties is arbitrary, not retrieval confidence'}
    for name, rows in records.items():
        sub = folder / name
        sub.mkdir()
        write_checkpoints(sub, rows, output['routes'][name])
        write_json(sub / 'expected_samples.json', expected[name])
    for name, rows in controls.items():
        sub = folder / name
        sub.mkdir()
        write_checkpoints(sub, rows, output['matched_depth_rrf_routes'][name])
    write_json(folder / 'summary.json', output)
    write_json(folder / 'verification.json', {'source_result_hashes': source_hashes,
                 'no_elasticsearch_queries': True, 'no_model_inference': True,
                 'source_rankings_and_metrics_verified': True})
    table = [{'route': name, 'ordering': 'original', **v['metrics']} for name, v in summary['routes'].items()]
    table += [{'route': name, 'ordering': 'path_tie_break', **v['metrics']} for name, v in output['routes'].items()]
    table += [{'route': name, 'ordering': 'uniform_tie_expectation', **v} for name, v in output['uniform_tie_expectation_routes'].items()]
    table += [{'route': name, 'ordering': 'matched_depth_rank_based_rrf', **v['metrics']}
              for name, v in output['matched_depth_rrf_routes'].items()]
    for filename, rows in [('summary.csv', table), ('rankings.csv', flat)]:
        with (folder / filename).open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print('Voting results: ' + str(folder), flush=True)
    return folder, output
