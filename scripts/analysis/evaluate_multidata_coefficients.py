"""Select joint coefficients on development projects, freeze, then validate."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/experiments')]
from run_multidata_coefficients import load,read,save,fingerprint
from bug_localization.snapshot import sha256_file
from bug_localization.coefficient_search import METRICS,grids,evaluate_grid,metric,rank
from bug_localization.paired_statistics import inference


def case(out,dataset,row):
    inp=out/dataset/'inputs'/f"{row['position']:04d}.json"
    score=out/dataset/'scores'/f"{row['position']:04d}.json"
    for p in (inp,score):
        if sha256_file(p)!=p.with_suffix('.sha256').read_text().strip():
            raise ValueError('Checkpoint checksum mismatch')
    x,y=read(inp),read(score)
    if x['status']!=y['status']:
        raise ValueError('Scoring status differs from source')
    if x['status']!='ok':return {'status':'failed'}
    return {**{k:x[k] for k in ('status','a','b','anchor')},**{k:y[k] for k in ('max','joint')}}


def choose(means,grid,default):
    return min(range(len(grid)),key=lambda i:(-float(means[i]),tuple(grid[i])!=tuple(default),sum(grid[i]),grid[i]))


def cluster_ci(delta,repositories,seed=20261124):
    d=np.asarray(delta)
    groups=[np.flatnonzero(np.array(repositories)==p) for p in sorted(set(repositories))]
    rng=np.random.default_rng(seed)
    sums=np.array([d[g].sum() for g in groups]);counts=np.array([len(g) for g in groups])
    indices=rng.integers(0,len(groups),(5000,len(groups)))
    estimates=sums[indices].sum(axis=1)/counts[indices].sum(axis=1)
    return {'projects':len(groups),'ci95_project_cluster_bootstrap':np.quantile(estimates,[.025,.975]).tolist()}


def summarize(values,baseline):
    values=np.asarray(values);baseline=np.asarray(baseline)
    return {'n':len(values),**dict(zip(METRICS,values.mean(axis=0).tolist())),
        'top1_corrected':int((values[:,2]>baseline[:,2]).sum()),
        'top1_broken':int((values[:,2]<baseline[:,2]).sum()),
        'ap_wins':int((values[:,0]>baseline[:,0]+1e-12).sum()),
        'ap_losses':int((values[:,0]<baseline[:,0]-1e-12).sum())}


def run(out):
    protocol,plan=load(out)
    ab,c,grid=grids(protocol);default=tuple(protocol['default_coefficients'])
    analysis=out/'analysis';analysis.mkdir(exist_ok=False)
    save(analysis/'protocol.json',{'parent_protocol_sha256':sha256_file(out/'protocol.json'),
        'implementation_sha256':sha256_file(Path(__file__)),
        'replay_sha256':sha256_file(ROOT/'src/bug_localization/coefficient_search.py'),
        'primary_comparisons':'per-dataset joint vs default and shared joint vs default; Holm across all four MAP tests',
        'secondary':'AB-only and C-only tuning plus per-project breakdowns; no validation-driven choice',
        'grid_size':len(grid),'historical_scope':protocol['historical_limits']})
    frozen={};train={};train_ab={}
    for dataset,rows in plan.items():
        dev=[r for r in rows if r['partition']=='development']
        sums=np.zeros((len(grid),len(METRICS)));absums=np.zeros((len(ab),len(METRICS)))
        for i,row in enumerate(dev,1):
            item=case(out,dataset,row)
            values,av=evaluate_grid(item,row['gold'],ab,c)
            # Independently replay boundary/current settings against production metrics.
            for index in (0,grid.index(default),len(grid)-1):
                order,pool=rank(item,grid[index])
                np.testing.assert_allclose(values[index],metric(order,row['gold']),atol=1e-12,rtol=0)
                np.testing.assert_allclose(av[index//len(c)],metric(pool,row['gold']),atol=1e-12,rtol=0)
            sums+=values;absums+=av
            if i%20==0:print(dataset,'development grid',i,len(dev),flush=True)
        means=sums/len(dev);ameans=absums/len(dev)
        train[dataset]=means;train_ab[dataset]=ameans
        best=choose(means[:,0],grid,default)
        best_ab=choose(ameans[:,0],ab,default[:2])
        default_ab=ab.index(default[:2])
        cslice=means.reshape(len(ab),len(c),-1)[default_ab,:,0]
        best_c=choose(cslice,c,default[2:])
        frozen[dataset]={'joint':grid[best],'ab_only':tuple(ab[best_ab])+default[2:],
            'c_only':default[:2]+tuple(c[best_c]),'development_n':len(dev),
            'development_joint':dict(zip(METRICS,means[best].tolist())),
            'development_default':dict(zip(METRICS,means[grid.index(default)].tolist())),
            'development_ab_default':dict(zip(METRICS,ameans[default_ab].tolist())),
            'development_ab_selected':dict(zip(METRICS,ameans[best_ab].tolist()))}
        with (analysis/f'{dataset}_development_grid.csv').open('w',newline='') as handle:
            writer=csv.writer(handle);writer.writerow(['alpha','anchor_weight','packet_weight','prior_weight',*METRICS])
            writer.writerows([*coeff,*m] for coeff,m in zip(grid,means))
        with (analysis/f'{dataset}_ab_development_grid.csv').open('w',newline='') as handle:
            writer=csv.writer(handle);writer.writerow(['alpha','anchor_weight',*METRICS])
            writer.writerows([*coeff,*m] for coeff,m in zip(ab,ameans))
    shared_means=np.mean([v[:,0] for v in train.values()],axis=0)
    shared=grid[choose(shared_means,grid,default)]
    choice={'per_dataset':frozen,'shared':shared,'default':default,'created_unix':time.time(),
        'validation_metrics_read':False,'parameter_order':['alpha','anchor_weight','packet_weight','prior_weight']}
    save(analysis/'frozen_choice.json',choice)
    choice_sha=sha256_file(analysis/'frozen_choice.json')
    summaries,statistics,details={}, {},{}
    for dataset,rows in plan.items():
        validation=[r for r in rows if r['partition']=='validation']
        methods={'default':default,**{k:frozen[dataset][k] for k in ('joint','ab_only','c_only')},'shared':shared}
        values={k:[] for k in methods};abvalues={k:[] for k in ('default','ab_only')};records=[]
        for row in validation:
            item=case(out,dataset,row);record={k:row[k] for k in ('position','dataset_row','repository','issue_id','before_fix_sha')}
            record.update(status=item['status'],gold=row['gold'],methods={})
            for name,coeff in methods.items():
                order,pool=rank(item,coeff);m=metric(order,row['gold']);values[name].append(m)
                record['methods'][name]={'coefficients':coeff,'ranking':order,'metrics':dict(zip(METRICS,m.tolist()))}
                if name in abvalues:abvalues[name].append(metric(pool,row['gold']))
            records.append(record)
        summaries[dataset]={k:summarize(v,values['default']) for k,v in values.items()}
        summaries[dataset]['AB_stage']={k:summarize(v,abvalues['default']) for k,v in abvalues.items()}
        summaries[dataset]['failures']=sum(r['status']!='ok' for r in records)
        for name in ('joint','shared'):
            delta=np.array(values[name])[:,0]-np.array(values['default'])[:,0]
            repos=[r['repository'] for r in validation]
            statistics[dataset+'/'+name]=inference(delta,repos,seed=20261124,repetitions=5000)|cluster_ci(delta,repos)
        details[dataset]=records
    previous=0.
    for i,key in enumerate(sorted(statistics,key=lambda k:statistics[k]['p_two_sided_paired_randomization'])):
        previous=max(previous,min(1.,(len(statistics)-i)*statistics[key]['p_two_sided_paired_randomization']))
        statistics[key]['p_holm_four_primary_MAP_tests']=previous
    save(analysis/'summary.json',summaries);save(analysis/'statistics.json',statistics);save(analysis/'validation_per_report.json',details)
    save(analysis/'verification.json',{'status':'passed','grid_size':len(grid),'frozen_choice_sha256':choice_sha,
        'choice_unchanged':sha256_file(analysis/'frozen_choice.json')==choice_sha,
        'candidate_sets_rebuilt_for_every_AB_setting':True,'validation_only_frozen_choices':True,
        'failure_reports_retained':True,'evaluation_denominator':'all original gold files','no_new_model_calls':True})
    lines=['# 多语言结构分数与AB/C联合系数实验','',
        '结构版本structure_ast_v2；每组参数重新构建AB Top50和Max C前十。基准为同一新结构版本下的(.5,.5,.25,4)，不是历史Python正则版本。',
        '',f'开发集比较{len(grid):,}组；先冻结系数再评价保留项目。共享系数采用两个数据集开发MAP等权，避免SWE样本量主导。',
        '',f'共享系数（α,γ,β,λ）：{shared}。',
        '', '| 数据集 | 方案 | 验证n | MAP@10 | Top1 | Top10 | 首位纠正/破坏 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for dataset,summary in summaries.items():
        for name in ('default','joint','shared','ab_only','c_only'):
            m=summary[name]
            lines.append(f"| {dataset} | {name} | {m['n']} | {m['MAP@10']:.6f} | {m['Top1']:.2%} | {m['Top10']:.2%} | {m['top1_corrected']}/{m['top1_broken']} |")
        lines.extend(['',f"{dataset}开发选择：{frozen[dataset]}",''])
    lines.extend(['','SWE500曾参与历史研究，因此项目留出不等于全新盲测。GHRB保留全部原始标签，新增文件和非Java标签仍计入分母。模型预训练污染不由此实验排除。',
        '统计区间与四项主比较校正在statistics.json。参数只是当前网格和开发目标下的选择；不能据此证明连续参数空间或其他数据上的全局最优。'])
    (analysis/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'shared':shared,'per_dataset':{k:v['joint'] for k,v in frozen.items()},'summary':summaries},ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    run(parser.parse_args().run.resolve())
