"""Write the completed direct-8B report and a paired historical cascade check."""
import argparse
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts/experiments')]
from run_direct8b500 import read
from run_mixed_recall500 import same_sample
from bug_localization.batch import aggregate_results, write_json
from bug_localization.bm25 import ranking_metrics
from bug_localization.paired_statistics import analyze
from bug_localization.snapshot import sha256_file


def table(rows):
    lines = ['| 方法 | 成功/总数 | MAP@10 | MRR@10 | Top-1 | Top-5 | Top-10 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name, row in rows.items():
        m = row['metrics']
        lines.append(f'| {name} | {row["successful_samples"]}/{row["selected_samples"]} | '
                     f'{m["MAP@10"]:.6f} | {m["MRR@10"]:.6f} | {m["Top-1"]:.2%} | '
                     f'{m["Top-5"]:.2%} | {m["Top-10"]:.2%} |')
    return '\n'.join(lines)


def report(output):
    if read(output / 'status.json')['status'] != 'completed' or read(output / 'verification.json')['status'] != 'passed':
        raise ValueError('Require completed, audited results')
    records = read(output / 'samples.json'); by_position = {r['position']: r for r in records}
    old_root = ROOT / 'results/c3_improvement_pilot/20261119/confirm_joint10'
    old = read(old_root / 'samples.json')
    direct = []
    for row in old:
        new = by_position[row['position']]; same_sample(row, new); direct.append(new)
        if row['status'] == 'ok':
            result_path = old_root / row['result_file']
            if sha256_file(result_path) != row['result_sha256']:
                raise ValueError('Historical cascade result changed')
            result = read(result_path)
            if ranking_metrics(result['ranking'], result['ground_truth'], 10) != row['metrics']:
                raise ValueError('Historical cascade metric changed')
            for path, checksum in row['source_sha256'].items():
                if sha256_file(Path(path)) != checksum:
                    raise ValueError('Historical cascade source changed')
    methods = {'direct_8B': direct, 'historical_C3_plus_8B': old}
    paired = {'summary': {name: aggregate_results(rows, 10) for name, rows in methods.items()},
              'statistics': analyze(methods, {'direct_minus_cascade': {'direct_8B': 1, 'historical_C3_plus_8B': -1}}, repetitions=5000),
              'note': 'Exploratory same90 historical comparison, not 500; historical 0.6B uses original-semantic tie order, direct uses RRF Top10'}
    write_json(output / 'historical_cascade90.json', paired)
    comparison = read(output / 'comparison.json'); stats = read(output / 'paired_statistics.json'); cost = read(output / 'cost.json')
    wall = read(output / 'summary.json')['wall_seconds_this_invocation']
    names = {'no_rerank_RRF': '双路融合，无重排', 'reranker_06B': '双路融合＋0.6B重排', 'direct_8B': '双路融合＋直接8B Top10'}
    parts = ['# 直接8B重排500条：完成结果', '',
             '2026-11-21；同一批历史开发500条，全部完成并通过输入、模型请求和排名指标核验。失败按零计入全部分母。', '',
             '## 全部500条', '', table({names[k]: v for k, v in comparison['all500'].items()}), '']
    for label, values in stats['contrasts'].items():
        s = values['average_precision_at_k']; lo, hi = s['ci95_project_stratified_bootstrap']
        parts.append(f'{label}：ΔMAP={s["delta"]:+.6f}，95%配对区间[{lo:+.6f}, {hi:+.6f}]；'
                     f'AP胜/平/负={s["wins"]}/{s["ties"]}/{s["losses"]}；12项Holm校正p={s["p_holm_all_contrasts_and_metrics"]:.6f}。\n')
    for key, title in [('previous180', '原有180条'), ('new320', '新增320条')]:
        parts.extend(['## ' + title, '', table({names[k]: v for k, v in comparison[key].items()}), ''])
    parts.extend(['## 同90条的历史两阶段8B对照（探索性）', '',
                  table({'直接8B': paired['summary']['direct_8B'], '历史0.6B→8B': paired['summary']['historical_C3_plus_8B']}), '',
                  '此表仅比较双方都有结果的原确认90条，不能称作500条两阶段8B比较。历史C3的0.6B同分排序规则为原文语义名次，本次直接8B输入来自RRF；这是历史流程比较，不是完全控制所有变量的重跑。', ''])
    s = paired['statistics']['contrasts']['direct_minus_cascade']['average_precision_at_k']
    parts.extend([f'直接8B相对历史两阶段方案的ΔMAP={s["delta"]:+.6f}，AP胜/平/负='
                  f'{s["wins"]}/{s["ties"]}/{s["losses"]}；配对随机化p={s["p_two_sided_paired_randomization"]:.6f}，'
                  f'本组6项指标Holm校正p={s["p_holm_all_contrasts_and_metrics"]:.6f}。'
                  '点估计更支持保留两阶段，但未通过统计显著性检验，不据此断言两阶段必然更优。', ''])
    failure_counts = dict(Counter(r.get('error_type') for r in records if r['status'] != 'ok'))
    failure_details = dict(Counter(r.get('error') for r in records if r['status'] != 'ok'))
    parts.extend(['## 失败和成本', '', f'失败类型：{failure_counts}。', '', f'失败原因：{failure_details}。', '',
                  f'8B调用{cost["calls"]}次，输入{cost["input_tokens"]:,}、输出{cost["output_tokens"]:,} tokens。'
                  f'逐样本累计新增阶段耗时{cost["incremental_case_seconds"]:.1f}秒；本次双并发墙钟{wall / 60:.2f}分钟。'
                  '复用检索和推测缓存，不能把新增阶段耗时当作端到端耗时。', '',
                  '直接8B只对融合前十重新排序，成功样本Top10集合不变；0.6B能从50文件中提取新的前十。'
                  '本次8B使用完整报告，历史0.6B使用编码阶段截断的查询。因此比较的是实际流程，不能将差异全部归因于参数量。', '',
                  '详细输入协议、命令与限制见docs/direct8b-reranking-500.md。全部500条已经参与过历史开发分析，不构成新盲测。'])
    (output / 'report.md').write_text('\n'.join(parts) + '\n')
    print(table({names[k]: v for k, v in comparison['all500'].items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); report(args.output.resolve())
