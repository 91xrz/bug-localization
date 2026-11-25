"""Write the module experiment report only after all GPU work and audits finish."""
import argparse
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[2]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--wait',action='store_true')
    args=parser.parse_args();run=args.run.resolve()
    while True:
        state=json.loads((run/'status.json').read_text())
        if state['status']=='failed': raise RuntimeError(state)
        if state['status']=='completed_with_failures': break
        if not args.wait: raise RuntimeError('Run is not complete')
        time.sleep(30)
    audit=json.loads((run/'verification.json').read_text())
    if audit['status']!='passed': raise ValueError('Audit not passed')
    summary=json.loads((run/'final_summary.json').read_text())
    samples=json.loads((run/'samples.json').read_text())
    pilot=json.loads((ROOT/'results/fusion_pilot/20261114_fixed45/paired_samples.json').read_text())
    for row in pilot:
        current=samples['A+B'][row['position']-1]
        expected=row['results']['rrf_k60_w1/c3_pool50']
        if current['status']!=expected['status'] or current.get('metrics')!=expected.get('metrics'):
            raise ValueError('RRF differs from frozen pilot')
    lines=['# 500 条模块消融：固定等权 RRF','',
        '本轮在同批固定 500 条上评估 A、B、A+B、A+B+C。A/B 复用原报告/功能推测检索并复算；A+B 使用等权 RRF（k=60），仅在既有推测 Top-40＋原文补足至 50 的候选池内输出；A+B+C 在两张 GPU 上重新执行混合选块及原报告重排，文件分数为所选块最大值，同分按 RRF 名次。没有调参，也没有重新生成推测或编码源码。','',
        'GPU 0/1 各分配 250 条，不停止缓存或其他服务。第 308 条的已有功能推测失败保留，B/A+B/A+B+C 按零计入分母；A 全部成功。新实验与原 C3 的输入顺序及平局规则不同，结果独立保存。','',
        '完整评分及审计已通过：候选重建、两路来源哈希、混合选块及代码文本、Max 聚合、RRF 平局规则、指标复算。45 条小批的 A+B 指标逐条与本轮一致。']
    labels={'all500':'全部 500 条','previous180':'原有 180 条','new320':'新增 320 条','pilot45':'已用于小批探索的 45 条','remaining455':'排除小批后的 455 条','new_nonpilot275':'新增且排除小批后的 275 条'}
    for cohort,methods in summary.items():
        lines+=['', '## '+labels[cohort], '', '| 组合 | 成功/总数 | MAP@10 | MRR@10 | Top-1 | Top-5 | Top-10 | MeanRecall@10 |', '|---|---:|---:|---:|---:|---:|---:|---:|']
        for name,stats in methods.items():
            m=stats['metrics'];lines.append(f"| {name} | {stats['successful_samples']}/{stats['selected_samples']} | {m['MAP@10']:.6f} | {m['MRR@10']:.6f} | {m['Top-1']:.2%} | {m['Top-5']:.2%} | {m['Top-10']:.2%} | {m['MeanRecall@10']:.6f} |")
        for first,second in [('A+B','A'),('A+B','B'),('A+B+C','A+B')]:
            delta=methods[first]['metrics']['MAP@10']-methods[second]['metrics']['MAP@10']
            lines.append(f'\n{first} − {second}：ΔMAP@10={delta:+.6f}。')
    lines+=['','## 解读范围','',
        '500 条为已观察的开发数据，其中包含方案探索用的 45 条。排除小批后的分组用于检查样本依赖，也不等于全新盲测集。本表未新增显著性检验，不据此宣称显著优于或稳定泛化。性能计时与其他 GPU 任务共享资源，不能作为公平延迟基准。', '',
        f'- [完整 CSV]({run}/comparison.csv)。',f'- [逐报告结果]({run}/samples.json)。',f'- [分项目结果]({run}/project_summary.json)。',f'- [审计]({run}/verification.json)。',f'- [冻结协议]({run}/protocol.json)。','']
    text='\n'.join(lines)
    (run/'report.md').write_text(text)
    (ROOT/'docs/module-ablation-rrf500.md').write_text(text)
    (run/'report_status.json').write_text(json.dumps({'status':'completed','pilot45_metrics_match':True},indent=2)+'\n')
    print('Report complete:',run/'report.md')

if __name__=='__main__':main()
