"""Paired C3 analyses; historical 500 are development data, not holdout."""
import argparse
from pathlib import Path
import sys
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'src'))
from bug_localization.reranking import read_json
from bug_localization.batch import write_json
from bug_localization.paired_statistics import analyze


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--factorial', type=Path)
    p.add_argument('--repetitions', type=int, default=20000)
    a = p.parse_args()
    mixed = read_json(PROJECT / 'results/mixed_recall_reranking/20261111_CD_500/paired_samples.json')['cohorts']['all500']
    chunk = read_json(PROJECT / 'results/chunk_ablation_reranking/20261112_C2_C3_500/paired_samples.json')['cohorts']['all500']
    methods = {'A': mixed['A'], **chunk}
    contrasts = {'C3_vs_A': {'C3': 1, 'A': -1}, 'C3_vs_C1': {'C3': 1, 'C1': -1}, 'C3_vs_C2': {'C3': 1, 'C2': -1}}
    if a.factorial:
        methods['A3'] = read_json(a.factorial / 'A3/samples.json')
        contrasts.update(A3_vs_A={'A3': 1, 'A': -1}, C3_vs_A3={'C3': 1, 'A3': -1},
            recall_main_effect={'C1': .5, 'C3': .5, 'A': -.5, 'A3': -.5},
            chunk_main_effect={'A3': .5, 'C3': .5, 'A': -.5, 'C1': -.5},
            interaction={'C3': 1, 'C1': -1, 'A3': -1, 'A': 1})
    a.output.mkdir(parents=True, exist_ok=True)
    result = analyze(methods, contrasts, a.repetitions)
    write_json(a.output / 'paired_statistics.json', result)
    lines = ['# C3 配对分析（历史 500 条，探索性）', '', '失败计零；项目内配对 bootstrap 95% CI；双侧配对符号随机化检验；Holm 校正覆盖全部对比和指标。', '',
             '| 对比 | ΔMAP@10 | 95% CI | p（Holm） | 胜/平/负 |', '|---|---:|---|---:|---|']
    for name, metrics in result['contrasts'].items():
        s = metrics['average_precision_at_k']; lo, hi = s['ci95_project_stratified_bootstrap']
        lines.append(f"| {name} | {s['delta']:.6f} | [{lo:.6f}, {hi:.6f}] | {s['p_holm_all_contrasts_and_metrics']:.4f} | {s['wins']}/{s['ties']}/{s['losses']} |")
    lines += ['', '项目贡献、去掉单个项目的结果、最大变化报告和收益集中度见 paired_statistics.json。', '已有 500 条参与过方案选择；显著性结果不替代新报告验证。']
    (a.output / 'report.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))

if __name__ == '__main__':
    main()
