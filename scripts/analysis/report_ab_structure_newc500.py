"""Report new AB candidates followed by the frozen evidence/prior C."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'results/ab_structure_newc/20261123_fixed500'


def read(name):return json.loads((RUN/name).read_bytes())


def main():
    s=read('summary.json');stats=read('statistics.json');audit=read('verification.json');diag=read('diagnostics.json')
    if audit['status']!='passed':raise ValueError('Run not verified')
    result=s['new_pool_new_C'];old=s['old_pool_new_C'];maxc=s['new_pool_Max']
    delta=stats['new_pool_new_C_vs_old_pool_new_C']['average_precision_at_k']
    lo,hi=delta['ci95_report_bootstrap'];elapsed=read('status.json')['finished_unix']-read('protocol.json')['started_unix']
    methods=[('new_AB','新AB初排'),('old_pool_Max','旧池＋原Max C'),('new_pool_Max','新池＋原Max C'),
             ('old_pool_new_C','旧池＋新版C（本轮配对）'),('new_pool_prior','新池＋Max及AB分数修正'),
             ('new_pool_new_C','新AB候选池＋新版C（主方案）'),
             ('new_pool_new_C_full50','新池＋新版C调整全部50（消融）'),
             ('old_pool_new_C_historical','旧池＋新版C（已发布历史参考）')]
    lines=['# 新AB候选池＋新版C：固定500条完整组合（2026-11-23）','',
        f"已完成全部500条，499条成功、1条已有上游失败保留计零。主方案MAP@10={result['MAP@10']:.6f}，Top1={result['Top1']:.2%}，Top10={result['Top10']:.2%}。相对同条件旧候选池＋新版C，ΔMAP={delta['delta']:+.6f}，95%配对区间[{lo:+.6f},{hi:+.6f}]。",'',
        '**结论：完整组合可以复现新版C的主要收益，但更换为新AB候选池只带来很小的额外变化，未显示稳定增益。** 新C相对同一新候选池上的原Max C提高MAP约0.042552，首位纠正50条、破坏16条，净增34条；相对旧池新版C只增加MAP约0.000879、Top10净增1条，Top1相同。','',
        '## 固定流程','',
        '沿用原始500条清单和样本顺序，使用已冻结的新AB参数与新版C参数，没有重新选参。500条属于已观察且参与过相关开发的数据，本轮不是独立盲测。','',
        '1. `ab_structure_v1` 从A50∪B50中按 `p=0.5N(A)+0.5N(B)+0.5E` 选择50文件。',
        '2. 每文件取A最优块，再按B补足最多3个不同块；原0.6B逐块Max评分得到新候选池的C初排。',
        '3. 同文件的已选块拼为联合文档，使用同一Qwen3-Reranker-0.6B评分得到u；batch8、4096输入上限与冻结新版C一致。',
        '4. 仅对第2步C初排的前十，计算 `0.75z+0.25u+4p` 并排序，保留这十个文件的集合以及第十一名之后的次序。这里的前十不是AB前十。','',
        '查询仍为原C使用的原报告文本（既有512-token语义输入预算内的内容），未扩大报告上下文。联合文档不含标签、补丁、生成推测或检索分数；p在模型之外计算并参与最后排序。','',
        '## 同条件比较与复用','',
        '逐块Max分数复用上一轮500条对旧、新候选联合重新评分的产物，两个候选池共享所有共同文件的本轮Max分数。旧候选池的全部联合输入与已完成新版C实验逐条核对：查询、文件路径、块ID、块顺序和完整文档文本一致才允许复用。',
        f"复用 {audit['reused_joint_scores']:,} 个旧联合评分，仅为新增文件补算 {audit['fresh_joint_scores']:,} 个联合评分；两种候选池共享共同文件的联合分数。所有500条均重新构建最终排名并评价，不能把本轮描述为对全部文件重新推理。",'',
        '已发布旧池＋新版C使用历史逐块分数；本轮主要旧池对照使用上一轮配对复算的逐块分数，以隔离候选集合影响。两者分别列出，不能把数值批次差异归为新池贡献。','',
        '## 完整500条结果','',
        '| 方案 | MAP@10 | MRR@10 | Top1 | Top5 | Top10 | 候选Hit50 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,label in methods:
        m=s[name]
        lines.append(f"| {label} | {m['MAP@10']:.6f} | {m['MRR@10']:.6f} | {m['Top1']:.2%} | {m['Top5']:.2%} | {m['Top10']:.2%} | {m['Hit50']:.2%} |")
    lines+=['','## 预设配对比较','',
        '| 对比 | ΔMAP | 报告bootstrap 95%区间 | 项目整体bootstrap 95%区间 | AP胜/平/负 | Holm p |',
        '|---|---:|---|---|---|---:|']
    for key,label in [('new_pool_new_C_vs_old_pool_new_C','新池新版C − 旧池新版C'),
                      ('new_pool_new_C_vs_new_pool_Max','新池新版C − 新池原Max C')]:
        x=stats[key]['average_precision_at_k'];a,b=x['ci95_report_bootstrap'];c,d=x['ci95_project_cluster_bootstrap']
        lines.append(f"| {label} | {x['delta']:+.6f} | [{a:+.6f},{b:+.6f}] | [{c:+.6f},{d:+.6f}] | {x['wins']}/{x['ties']}/{x['losses']} | {x['holm_p_10_tests']:.6g} |")
    lines+=['','两个对比各检验MAP、MRR、Top1、Top10、Recall10，共10项Holm校正；重采样及配对随机化各5,000次。','',
        '| 主方案相对 | 首位纠正/破坏 | Top10补回/丢失 |','|---|---:|---:|']
    for key,label in [('old_pool_new_C','旧池新版C'),('new_pool_Max','新池原Max C'),('new_AB','新AB初排')]:
        x=diag[key];lines.append(f"| {label} | {x['top_1']['corrected']}/{x['top_1']['broken']} | {x['top_10']['corrected']}/{x['top_10']['broken']} |")
    lines+=['','## 解释边界','',
        '主方案仍只重排C前十，所以它与新池原Max C的Top10及Recall10必然相同；其额外效果应该看Top1、MRR和MAP。与旧池新版C的Top10差异来自新池和Max阶段带来的前十集合变化。',
        '新版C已经包含新AB分数p，旧池新版C也使用同一个p。因此，新池与旧池的差值衡量的是进一步更换候选及初始C排序的效果，不能把此前结构匹配分修正的收益再次记为新池收益。',
        '全部50排序为预先声明的消融，会改变前十集合；其点估计无论高低，都不替换本次固定的前十主方案。','',
        '## 核验与运行成本','',
        f"新增推测生成、Embedding和8B调用均为0。补算使用同一0.6B模型，两张GPU分担500条；本轮从准备到统计的增量墙钟约{elapsed/60:.2f}分钟。工作进程累计补算时间{audit['sum_incremental_seconds']:.2f}秒（含首次模型加载），不包括历史AB、逐块Max和已缓存联合评分的成本。",'',
        '核验覆盖冻结AB输出逐条复现、查询与联合文档逐项一致、缓存和结果哈希、共同候选共享分数、独立公式重算、前十集合和尾部保持、原新版C已发布成绩复现、全部失败计零。缓存复用与结构输入改变拒绝等单元检查通过。','',
        '## 产物','',
        '- [完整CSV](../results/ab_structure_newc/20261123_fixed500/comparison.csv)。',
        '- [配对统计](../results/ab_structure_newc/20261123_fixed500/statistics.json)。',
        '- [项目结果](../results/ab_structure_newc/20261123_fixed500/project_summary.json)。',
        '- [变化案例](../results/ab_structure_newc/20261123_fixed500/diagnostics.json)。',
        '- [核验记录](../results/ab_structure_newc/20261123_fixed500/verification.json)。',
        '- [独立复算与原始来源校验](../results/ab_structure_newc/20261123_fixed500/independent_audit.json)。',
        '- 配置：`configs/experiments/ab_structure_newc500.json`；入口：`scripts/experiments/run_ab_structure_newc500.py`。',
        '- 新目录执行`prepare`，然后`run`补算并自动评价；同目录`run`可校验并跳过已完成检查点。','']
    (ROOT/'docs/ab-structure-newc500.md').write_text('\n'.join(lines))
    print(json.dumps({'primary':result,'vs_old_pool':delta,'incremental_wall_seconds':elapsed},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
