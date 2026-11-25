"""Write a compact Chinese research report from the frozen CPU experiment."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'results/b_lightweight/20261122_v1'


def read(name):
    return json.loads((RUN/name).read_bytes())


def main():
    s=read('summary.json'); choice=read('frozen_choice.json')['method']
    audit=read('independent_audit.json'); stats=read('validation_inference.json')
    cost=read('cost.json'); validation=s['remaining1794']; full=s['all2294']
    methods=[('A','原报告语义 A'),('B','原 B'),('current_RRF60','现有 B40+A 补足 / RRF60'),
             ('balanced_pool_Border','交错取50、按B排序'),('score_a0.5_e0','只改分数融合'),
             ('B50_anchor_e0.5','B Top50 内加入结构匹配'),(choice,'开发集选定：分数融合+结构匹配')]
    lines=['# B 的轻量改进：论文依据与 CPU 实验（2026-11-22）','',
           '**采用状态（2026-11-22）：用户已决定将“分数融合＋结构匹配”固定为当前召回/初排基线 `ab_structure_v1`，alpha=0.5、lambda=0.5、最终50个文件。后续默认使用固定参数，见[当前基线与入口](current-baseline.md)。**','',
           '后续验证（2026-11-23）：[新AB＋C的500条实验](ab-structure-c500.md)已完成，新旧AB+C基本持平；本页的初排增益不能直接解释为完整ABC增益。','',
           '已完成全部 2,294 条实验。原 500 条用于筛选设置，冻结后应用于其余 1,794 条。两组都属于已观察数据，后者不是独立盲测集。全程没有新增生成、Embedding 或 Reranker 调用。','',
           '## 研究依据与取舍','',
           '| 论文/依据 | 可用思路 | 本次实现与边界 |',
           '|---|---|---|',
           '| [BLUiR，ASE 2013](https://users.ece.utexas.edu/~perry/work/papers/1311-RS-ase.pdf) | 单独利用类名、方法名等程序结构，避免被普通文本淹没 | 从修复前源码扫描函数/类声明，与原报告做区分大小写的精确词匹配；没有复现 Indri、分字段 BM25 或完整 BLUiR |',
           '| [BLIZZARD，FSE 2018](https://arxiv.org/html/1808.00594v1) | 依据报告信息类型处理查询噪声，不能认为栈信息越多越好 | 去除 HTML 注释和 URL，并用候选内 IDF 降低公共标识符的影响；未实现完整栈图 PageRank、反馈检索或报告分类 |',
           '| [An Analysis of Fusion Functions for Hybrid Retrieval，TOIS](https://arxiv.org/html/2210.11934v2) | 归一化分数融合可以保留名次丢掉的分差；RRF 也有参数敏感性 | 采用前50分数范围归一化后融合 A/B，属于对本项目的改造；原论文主要研究词法/语义混合，不能把论文结论直接当本实验结果 |',
           '| [CosFL](https://arxiv.org/html/2411.17230v1) | 功能描述生成与后续检索可以分开优化，并考虑不同层级信息 | 保留现有推测，不重做生成、知识库或代码摘要；本轮优化推测检索结果的利用，不是完整 CosFL 复现 |',
           '| 本项目 Top50 诊断 | B 在 41–50 名仍有正确候选，硬切 Top40 会损失它们 | 对 A50∪B50 评分后取50，另做无评分交错取50对照；所有正式变体保持同样候选上限 |','',
           '优先做结构匹配，是因为源码和原始报告已经存在，只需确定性扫描和加权。进一步拆分 context/functionality/logic 后重新编码也不需要重新生成推测，但仍增加查询编码和检索；本轮未执行。盲目伪相关反馈存在查询漂移风险，本轮也未扩展。','',
           '## 方法与冻结协议','',
           '对每路语义分数使用 `N(s)=clip((s-s50)/(s1-s50),0,1)`，全同分时为零。计算：','',
           '`score(file) = (1-alpha) * N(B) + alpha * N(A) + lambda * E(file)`','',
           '`E` 是原报告对源码函数/类名的 IDF 加权覆盖率，归一化至 [0,1]；出现完整相对路径或点分模块名时设为1。词频按出现与否计算，重复提到同一符号不会重复累加。词表只由修复前源码与原报告产生，忽略 Python 关键字、内建名称和常见双下划线名称。源码使用轻量声明扫描器，未做完整 AST 语义解析。','',
           '特征使用完整原始 body；原语义编码仍受既有512-token限制，因此增益可能同时来自更完整的报告信息与结构匹配，不能全部归因于单一机制。禁止读取独立 PR title、修复补丁、未来历史或把标签传给评分函数。标签只用于开发集选参和事后评价。','',
           '预先固定 alpha={0,0.25,0.5}、lambda={0,0.1,0.25,0.5}，另设 B50 内结构匹配消融。500条选择门槛：MAP 比 B 至少高0.005，MRR、Hit10、Hit50均不低于B；在满足者中选开发集 MAP 最高者，无满足者则保留B。没有根据1,794条结果再修改规则。','',
           'B50内的消融固定原B的候选集合，只改变顺序；为隔离候选范围这一因素，其结构特征和IDF仍与其他变体共用A50∪B50。因此它不等于完全移除A信息的独立单路系统。','',
           f'开发集最终选择 `{choice}`（alpha=0.5，lambda=0.5），先写入 `frozen_choice.json` 再进入验证循环。失败的 Xarray #5662 保留，所有依赖B的路线计零；A独立评价。','',
           '## 固定设置后的1,794条结果','',
           '| 方法 | MAP@10 | MRR@10 | Top1 | Hit10 | Hit50 | 平均标签召回@50 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for name,label in methods:
        m=validation[name]
        lines.append(f"| {label} | {m['MAP@10']:.6f} | {m['MRR@10']:.6f} | {m['Hit@1']:.2%} | {m['Hit@10']:.2%} | {m['Hit@50']:.2%} | {m['MeanRecall@50']:.2%} |")
    chosen=validation[choice]; base=validation['B']
    diag=audit['diagnostics']['remaining1794']
    lines+=['',f"选定方案相对B：ΔMAP={chosen['MAP@10']-base['MAP@10']:+.6f}；Hit10净增 {round((chosen['Hit@10']-base['Hit@10'])*1794)} 条；Hit50净增 {round((chosen['Hit@50']-base['Hit@50'])*1794)} 条。",
            f"逐报告AP胜/平/负：{diag['B']['ap_wins']}/{diag['B']['ap_ties']}/{diag['B']['ap_losses']}。Hit10补回/丢失：{diag['B']['Hit@10']['rescued']}/{diag['B']['Hit@10']['lost']}；Hit50补回/丢失：{diag['B']['Hit@50']['rescued']}/{diag['B']['Hit@50']['lost']}。",'',
            '三个预定验证检验均使用5,000次项目内配对bootstrap和配对随机化；Holm校正覆盖选定方案相对B的MAP、Hit10、Hit50。数据已经被观察过，区间和检验是本数据集上的配对统计证据，不等同于全新盲测保证。','',
            '| 指标 | 差值 | 95%区间 | Holm校正p |', '|---|---:|---|---:|']
    for key,x in stats['tests'].items():
        lo,hi=x['ci95_project_stratified_bootstrap']
        lines.append(f"| {key} | {x['delta']:+.6f} | [{lo:+.6f}, {hi:+.6f}] | {x['p_holm_three_primary_tests']:.6g} |")
    project_stats=stats['tests']['MAP@10']
    positive=sum(v['delta']>0 for v in project_stats['projects'].values())
    lines+=['',f"验证集12个项目中 {positive} 个MAP提高；Flask的11条报告下降0.014478。去掉任意一个项目后，总体MAP差值仍为正；去掉收益最大的10条报告后，差值仍为 {project_stats['delta_without_10_largest_gains']:+.6f}。收益并非仅由少量最大改善案例解释，但小项目统计仍不稳定。"]
    lines+=['','## 开发集与全量描述结果','',
            '| 范围 | B MAP | 选定方案 MAP | B Hit10 | 选定 Hit10 | B Hit50 | 选定 Hit50 |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for cohort,label in [('development500','开发500：用于选参'),('all2294','全量2294：含开发集')]:
        a=s[cohort]['B'];b=s[cohort][choice]
        lines.append(f"| {label} | {a['MAP@10']:.6f} | {b['MAP@10']:.6f} | {a['Hit@10']:.2%} | {b['Hit@10']:.2%} | {a['Hit@50']:.2%} | {b['Hit@50']:.2%} |")
    d=audit['diagnostics']['all2294']['B_exclusive_gold']
    lines+=['',f"全量B补回的 {d['available']} 个 A50 外正确文件，本方案保留 {d['retained']} 个、舍弃 {d['dropped']} 个。旧40+10保留100/139，具体净收益仍需同时考虑丢失的A/B已有正确候选。",'',
            '## 实际建议','',
            '如果目标是最小改动，先使用“B原Top50内加入结构匹配”：它不更换已有候选，在验证集上MAP由0.535908升到0.615861，已经获得主要排序收益。如果同时希望改善固定50候选的覆盖，采用开发集预先选定的alpha=0.5、lambda=0.5，并从两路并集中按新分数取50。',
            '不能将本次提升表述为B生成了更多新的正确功能推测：推测文本完全复用。主要提升发生在原有候选的排序，B独有正确文件的保留只由100升到101；相对现有融合，验证集Top50净增6条（6条补回、0条丢失），明显小于MAP的提升幅度。',
            '按用户决定，组合方案现已固定为当前初排基线；上述B50内方案保留为消融对照。历史实验结果保留，不根据验证集上的事后最优结果改选参数。','',
            '## 实现成本、核验与限制','',
            f"实验总墙钟 {cost['total_seconds']:.1f} 秒（包括读已有排名、源码校验、所有变体评分、写结果及配对统计）；其中特征评分/融合计时 {cost['postprocess_seconds']:.1f} 秒。首次提取 {cost['unique_source_contents']} 份不同源码内容，读取 {cost['source_bytes_read']/1024**3:.3f} GiB。全部在CPU执行，模型调用0次、GPU调用0次；这不是包含原B生成和检索的端到端延迟。",'',
            f"独立审计重算 {audit['metric_checks']} 组逐报告指标。A、B及现有融合的基线AP均逐条复现，原Top50的交集和标签数逐条匹配；现有融合的候选集合和全部排序也逐条匹配。另有5项单元检查覆盖噪声过滤、声明/调用区分、不存在的符号、大小写/模块边界和全同分排序。",'',
            '结构匹配仍可能强化错误线索：例如开发集 SymPy #16450 的大量 assumption 属性词将 `ask.py` 推到前面，压低真正的 `simplify.py`。公共API所在文件也未必是根因文件。不要将得分提高解释为推测内容或根因已验证。',
            '所有结果是召回/初排阶段。没有为新候选重新执行C重排，所以不能声称 A+B+C 的最终MAP已经提高。并集上限诊断是变长候选，其 PoolHit/PoolRecall只能用于诊断，不能与固定50同预算方案混报。','',
            '## 复现与产物','',
            '- 配置：`configs/experiments/b_lightweight_20261122.json`。',
            '- 运行：`.venv/bin/python scripts/experiments/run_b_lightweight.py --output results/b_lightweight/<new_directory>`。',
            '- 审计：`.venv/bin/python scripts/analysis/audit_b_lightweight.py --run results/b_lightweight/<new_directory>`。',
            '- [完整比较CSV](../results/b_lightweight/20261122_v1/comparison.csv)。',
            '- [冻结参数选择](../results/b_lightweight/20261122_v1/frozen_choice.json)。',
            '- [验证配对统计](../results/b_lightweight/20261122_v1/validation_inference.json)。',
            '- [独立审计与改善/退化案例](../results/b_lightweight/20261122_v1/independent_audit.json)。',
            '- [按项目汇总](../results/b_lightweight/20261122_v1/project_summary.json)。',
            '- `cases/`保留逐报告的全部变体排名、指标和具体结构匹配项；`inputs/`保留可复算的两路完整文件分数及无标签特征。','']
    (ROOT/'docs/b-lightweight-improvements.md').write_text('\n'.join(lines))
    print(ROOT/'docs/b-lightweight-improvements.md')


if __name__=='__main__':main()
