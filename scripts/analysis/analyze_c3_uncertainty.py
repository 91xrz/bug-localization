"""Report-paired and project-cluster intervals, including singleton-project holdouts."""
import argparse
from pathlib import Path
import sys
import numpy as np
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'src'))
from bug_localization.paired_statistics import align, values, METRICS
from bug_localization.reranking import read_json
from bug_localization.batch import write_json


def intervals(delta, repositories, seed=20261112, repetitions=20000):
    d = np.asarray(delta, dtype=float); repos = np.asarray(repositories)
    if not len(d) or not np.isfinite(d).all() or len(d) != len(repos):
        raise ValueError('Invalid paired input')
    names = sorted(set(repositories))
    counts = np.array([(repos == name).sum() for name in names])
    totals = np.array([d[repos == name].sum() for name in names])
    rng = np.random.default_rng(seed); report_boot, cluster_boot = [], []
    for i in range(0, repetitions, 200):
        n = min(200, repetitions - i)
        report_boot.extend(d[rng.integers(0, len(d), (n, len(d)))].mean(axis=1))
        indices = rng.integers(0, len(names), (n, len(names)))
        cluster_boot.extend(totals[indices].sum(axis=1) / counts[indices].sum(axis=1))
    return {'delta': float(d.mean()), 'ci95_report_paired_bootstrap': np.quantile(report_boot, [.025, .975]).tolist(),
        'ci95_project_cluster_bootstrap': np.quantile(cluster_boot, [.025, .975]).tolist(),
        'reports': len(d), 'projects': len(names), 'singleton_projects': int((counts == 1).sum()),
        'seed': seed, 'repetitions': repetitions}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--factorial', type=Path, required=True)
    p.add_argument('--holdout', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    mixed = read_json(PROJECT / 'results/mixed_recall_reranking/20261111_CD_500/paired_samples.json')['cohorts']['all500']
    chunk = read_json(PROJECT / 'results/chunk_ablation_reranking/20261112_C2_C3_500/paired_samples.json')['cohorts']['all500']
    development = {'A': mixed['A'], 'A3': read_json(a.factorial / 'A3/samples.json'), **chunk}
    contrasts = {'C3_vs_A': {'C3': 1, 'A': -1}, 'C3_vs_C1': {'C3': 1, 'C1': -1},
        'A3_vs_A': {'A3': 1, 'A': -1}, 'C3_vs_A3': {'C3': 1, 'A3': -1},
        'recall_main_effect': {'C1': .5, 'C3': .5, 'A': -.5, 'A3': -.5},
        'chunk_main_effect': {'A3': .5, 'C3': .5, 'A': -.5, 'C1': -.5},
        'interaction': {'C3': 1, 'C1': -1, 'A3': -1, 'A': 1}}
    holdout = {m: read_json(a.holdout / m / 'samples.json') for m in ('A', 'C3')}
    output = {'note': 'For external generalization use project-cluster intervals. Within-project stratified intervals condition on the observed projects and become degenerate with singleton projects; they must not support LCA generalization claims. Intervals are unadjusted; paired randomization and Holm p-values remain in the original statistics files.', 'cohorts': {}}
    common = [i for i in range(len(development['A'])) if all(rs[i]['status'] == 'ok' for rs in development.values())]
    groups = [('development500', development, contrasts),
        ('development_common_success_sensitivity', {m: [rs[i] for i in common] for m, rs in development.items()}, contrasts),
        ('external_LCA47', holdout, {'C3_vs_A': {'C3': 1, 'A': -1}})]
    for label, methods, comparisons in groups:
        rows = align(methods); repos = [r['repository'] for r in rows]
        output['cohorts'][label] = {}
        for name, coefficients in comparisons.items():
            output['cohorts'][label][name] = {metric: intervals(sum(w * values(methods[m], metric) for m, w in coefficients.items()), repos) for metric in METRICS}
    a.output.parent.mkdir(parents=True, exist_ok=True); write_json(a.output, output)
    print(a.output)

if __name__ == '__main__':
    main()
