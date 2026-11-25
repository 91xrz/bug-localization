"""Report-paired inference, project sensitivity and factorial contrasts."""
import numpy as np

IDENTITY = ('dataset_row', 'repository', 'issue_id', 'before_fix_sha', 'query_sha256', 'labels_sha256')
METRICS = ('average_precision_at_k', 'reciprocal_rank_at_k', 'top_1', 'top_5', 'top_10', 'recall_at_k')


def align(methods):
    rows = next(iter(methods.values()))
    keys = [tuple(r[k] for k in IDENTITY) for r in rows]
    if not keys or len(keys) != len(set(keys)):
        raise ValueError('Empty or duplicate paired reports')
    for name, other in methods.items():
        if [tuple(r[k] for k in IDENTITY) for r in other] != keys:
            raise ValueError('Unpaired selection: ' + name)
        for r in other:
            if r['status'] not in ('ok', 'failed'):
                raise ValueError('Unknown status')
            if r['status'] == 'ok' and any(k not in r['metrics'] for k in METRICS):
                raise ValueError('Missing success metrics')
    return rows


def values(rows, metric):
    return np.array([r['metrics'][metric] if r['status'] == 'ok' else 0. for r in rows])


def inference(delta, repositories, *, seed=20261112, repetitions=20000):
    d = np.asarray(delta, dtype=float)
    if not len(d) or not np.isfinite(d).all():
        raise ValueError('Empty or nonfinite differences')
    rng = np.random.default_rng(seed)
    reps = np.array(repositories)
    groups = [np.flatnonzero(reps == p) for p in sorted(set(repositories))]
    boots, nulls = [], []
    # Within-project resampling preserves the frozen project allocation and pairs.
    for start in range(0, repetitions, 200):
        count = min(200, repetitions - start)
        total = np.zeros(count)
        for indices in groups:
            total += d[rng.choice(indices, (count, len(indices)))].sum(axis=1)
        boots.extend(total / len(d))
        nulls.extend((d * rng.choice([-1, 1], (count, len(d)))).mean(axis=1))
    projects = {p: {'n': int((reps == p).sum()), 'delta': float(d[reps == p].mean()),
                    'contribution_to_overall_delta': float(d[reps == p].sum() / len(d)),
                    'wins': int((d[reps == p] > 1e-12).sum()), 'losses': int((d[reps == p] < -1e-12).sum())}
                for p in sorted(set(repositories))}
    positive = np.sort(d[d > 1e-12])[::-1]
    top = np.argsort(-d, kind='stable')[:min(10, len(d) - 1)]
    return {'n': len(d), 'delta': float(d.mean()), 'ci95_project_stratified_bootstrap': np.quantile(boots, [.025, .975]).tolist(),
            'p_two_sided_paired_randomization': float((1 + (np.abs(nulls) >= abs(d.mean()) - 1e-15).sum()) / (repetitions + 1)),
            'wins': int((d > 1e-12).sum()), 'ties': int((np.abs(d) <= 1e-12).sum()), 'losses': int((d < -1e-12).sum()),
            'projects': projects, 'macro_project_delta': float(np.mean([x['delta'] for x in projects.values()])),
            'leave_one_project_out': {p: float(d[reps != p].mean()) for p in projects if (reps != p).any()},
            'top10_share_of_gross_positive_gain': float(positive[:10].sum() / positive.sum()) if len(positive) else None,
            'delta_without_10_largest_gains': float(np.delete(d, top).mean()) if len(d) > 1 else None,
            'seed': seed, 'repetitions': repetitions}


def analyze(methods, contrasts, repetitions=20000):
    rows = align(methods)
    repos = [r['repository'] for r in rows]
    result = {'denominator': len(rows), 'failures_zero': True,
              'failures': {m: sum(r['status'] != 'ok' for r in rs) for m, rs in methods.items()},
              'primary_metric': 'average_precision_at_k', 'contrasts': {}}
    for label, coefficients in contrasts.items():
        result['contrasts'][label] = {}
        for metric in METRICS:
            d = sum(weight * values(methods[m], metric) for m, weight in coefficients.items())
            stats = inference(d, repos, repetitions=repetitions)
            order = np.argsort(-np.abs(d), kind='stable')[:20]
            stats['largest_absolute_changes'] = [{**{k: rows[i][k] for k in IDENTITY}, 'delta': float(d[i])} for i in order]
            result['contrasts'][label][metric] = stats
    # Holm family includes every requested contrast and metric (secondary included).
    tests = [v for c in result['contrasts'].values() for v in c.values()]
    ordered = sorted(tests, key=lambda x: x['p_two_sided_paired_randomization'])
    previous = 0.
    for i, stats in enumerate(ordered):
        previous = min(1., max(previous, (len(tests) - i) * stats['p_two_sided_paired_randomization']))
        stats['p_holm_all_contrasts_and_metrics'] = previous
    result['multiplicity_family_size'] = len(tests)
    return result
