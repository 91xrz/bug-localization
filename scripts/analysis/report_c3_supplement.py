"""Render the four completed experiments with explicit uncertainty and cost scope."""
from pathlib import Path
import sys
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'src'))
from bug_localization.reranking import read_json


def main():
    root = PROJECT / 'results/c3_supplement'
    factorial = root / '20261112_factorial500'
    holdout = root / '20261112_lca_py_holdout'
    latency = root / '20261112_end_to_end'
    for folder in (factorial, holdout, latency):
        if read_json(folder / 'status.json')['status'] != 'completed':
            raise ValueError('Report requires completion: ' + str(folder))
    robust = read_json(root / '20261112_robust_intervals.json')['cohorts']
    statistics = read_json(factorial / 'statistics/paired_statistics.json')['contrasts']
    hs = read_json(holdout / 'paired_statistics.json')['contrasts']['C3_vs_A']
    old = read_json(PROJECT / 'results/chunk_ablation_reranking/20261112_C2_C3_500/final_summary.json')['cohorts']['all500']['methods']
    mixed = read_json(PROJECT / 'results/mixed_recall_reranking/20261111_CD_500/final_summary.json')['cohorts']['all500']['methods']
    cells = {'A': mixed['A'], 'A3': read_json(factorial / 'A3/summary.json'), 'C1': old['C1'], 'C3': old['C3']}
    lines = ['# C3 四组补充实验结果', '', '全部评分、独立核验、配对统计和端到端计时已完成。方案与选样规则见 [实验说明](c3-supplement-experiments.md)。', '',
        '## 1. 2×2 消融：收益更偏向选块，但不确定性仍大', '',
        '| 候选召回 | 原文选块 | 混合选块 |', '|---|---:|---:|',
        f"| 原文 Top-50 | A：{cells['A']['metrics']['MAP@10']:.6f} | A3：{cells['A3']['metrics']['MAP@10']:.6f} |",
        f"| 混合 Top-50 | C1：{cells['C1']['metrics']['MAP@10']:.6f} | C3：{cells['C3']['metrics']['MAP@10']:.6f} |", '',
        '均为 MAP@10，500 条为完整分母；A3/C1/C3 各有 1 条已有推测依赖失败，按零计入。', '',
        '| 效应 | MAP 差值 | 按项目重采样 95% CI | Holm p（48 项校正） |', '|---|---:|---|---:|']
    for name, label in [('recall_main_effect', '召回主效应'), ('chunk_main_effect', '选块主效应'), ('interaction', '交互效应'), ('C3_vs_A', 'C3 − A')]:
        r = robust['development500'][name]['average_precision_at_k']; lo, hi = r['ci95_project_cluster_bootstrap']
        pv = statistics[name]['average_precision_at_k']['p_holm_all_contrasts_and_metrics']
        lines.append(f"| {label} | {r['delta']:.6f} | [{lo:.6f}, {hi:.6f}] | {pv:.4f} |")
    share = robust['development500']['chunk_main_effect']['average_precision_at_k']['delta'] / robust['development500']['C3_vs_A']['average_precision_at_k']['delta']
    lines += ['', f'点估计分解中，选块约占总增益的 {share:.1%}；这不是已证实的贡献比例。交互效应接近零；选块主效应区间跨零。另附 499 条共同成功样本敏感性分析，不替代完整分母主表。', '',
        '## 2. 配对统计：不是单个项目独占，但集中在少量报告', '',
        'C3 对 A 的 AP 胜/平/负为 77/382/41。去掉任一项目后均值仍为正；pytest、pylint、sphinx、matplotlib 的正收益较多，xarray 为负。最大的 10 条正收益贡献约 39.3% 的全部正收益；去掉它们后平均差异仅 0.000337。该剔除只用于事后敏感性分析。多重比较校正后，不能宣称总体改善显著。', '',
        '## 3. LCA Python：方向一致，尚不足以确认推广', '',
        '本地 SWE-bench 全部 2,294 条已有基础检索评测记录。LCA Python 测试集预先排除 3 条历史使用报告，剩余 47 条固定评测，A/C3 均成功。保持 body-only 查询及原模型参数。', '',
        '| 方法 | MAP@10 | MRR@10 | Top-1 | Top-5 | Top-10 | MeanRecall@10 |', '|---|---:|---:|---:|---:|---:|---:|']
    for m in ('A', 'C3'):
        metric = read_json(holdout / m / 'summary.json')['metrics']
        lines.append('| ' + m + ' | ' + ' | '.join(f'{metric[k]:.4f}' for k in ('MAP@10', 'MRR@10', 'Top-1', 'Top-5', 'Top-10', 'MeanRecall@10')) + ' |')
    r = robust['external_LCA47']['C3_vs_A']['average_precision_at_k']; lo, hi = r['ci95_project_cluster_bootstrap']
    lines += ['', f"ΔMAP={r['delta']:.6f}，按项目重采样 95% CI=[{lo:.6f}, {hi:.6f}]；双侧配对随机化 p={hs['average_precision_at_k']['p_two_sided_paired_randomization']:.4f}，6 指标 Holm p={hs['average_precision_at_k']['p_holm_all_contrasts_and_metrics']:.4f}。AP 胜/平/负为 9/32/6。", '',
        'LCA 多数项目只有一条报告；因此原始统计文件的“项目内重采样”区间只表示条件化波动，不用于推广判断。本表使用补算的项目整体重采样区间，按报告重采样的区间也保留在 robust_intervals 文件。', '',
        '## 4. 实测端到端耗时与成本', '',
        '固定随机选取 9 条报告，每条 A/C3 各重复 3 次，共 54 次请求；模型常驻同一 RTX 4090，预备代码向量和索引，关闭生成与查询向量缓存。计时从内存中的报告到最终排名，使用连续墙钟与 CUDA 同步。模型加载/预热和共享代码准备另存。', '',
        '| 方法 | 成功/尝试 | 均值（秒） | P50（秒） | P95（秒） | 测量期间设备小时（含失败） |', '|---|---:|---:|---:|---:|---:|']
    costs = read_json(latency / 'summary.json')['methods']
    for m, c in costs.items():
        lines.append(f"| {m} | {c['successes']}/{c['attempts']} | {c['mean_seconds']:.3f} | {c['p50_seconds']:.3f} | {c['p95_seconds']:.3f} | {c['total_measured_device_wall_hours_including_failures']:.5f} |")
    extra = costs['C3']['mean_seconds'] - costs['A']['mean_seconds']; ratio = costs['C3']['mean_seconds'] / costs['A']['mean_seconds']
    lines += ['', f'C3 平均额外耗时 {extra:.3f} 秒，约为 A 的 {ratio:.2f} 倍。准确率收益尚未达到显著，当前证据不足以断言额外开销值得普遍采用；可把 C3 保留为允许额外延迟时的候选方案。', '',
        '这测量的是带检索一致性核验的研究流水线，不是优化后的线上服务。9 条 × 3 次的 P95 估计较粗；重复运行不增加准确率评测样本数。峰值显存记录包含三个常驻模型，不代表单独部署 A 所需显存。', '',
        '所有推理均在本地执行，无 API 账单。未指定设备单价，金额成本保留为空；设备小时可按给定单价换算，不能直接当作实际电费。全量 47 条另有阶段耗时之和，不能与连续请求墙钟时间混为一谈。', '',
        '## 核验与可追溯性', '',
        '新增评分在最初汇总时因缺少 missing_gold 字段中断；该字段已从保存的标签和完整候选清单重建。原始 samples.json、case 分数和排名未改写，补齐字段的副本为 samples_with_coverage.json。独立审计检查候选、选块、文本、源文件哈希、Max 聚合、平局顺序及指标，恢复依据保存在 postprocessing_recovery.json。', '',
        '- 消融与校正统计：`results/c3_supplement/20261112_factorial500/`。',
        '- LCA 外部验证：`results/c3_supplement/20261112_lca_py_holdout/`。',
        '- 重采样区间与共同成功子集敏感性：`results/c3_supplement/20261112_robust_intervals.json`。',
        '- 端到端 54 次请求原始记录：`results/c3_supplement/20261112_end_to_end/samples.json`。']
    path = PROJECT / 'docs/c3-supplement-results.md'; path.write_text('\n'.join(lines) + '\n')
    print(path)

if __name__ == '__main__':
    main()
