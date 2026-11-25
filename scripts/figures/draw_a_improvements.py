"""Standalone scientific figure from frozen validation estimates, without refitting."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
    root=args.run
    data=json.loads((root/'extended_validation_inference.json').read_bytes())['comparisons']
    frozen=json.loads((root/'frozen_choice.json').read_bytes())
    recall=json.loads((root/'recall_control/protocol.json').read_bytes())['method']
    entries=[(frozen['families']['aggregation'],'Max / Top-3 mixture'),
             (frozen['families']['length'],'File-length correction'),
             (frozen['families']['anchor'],'Structure within top 50'),
             (frozen['families']['prf'],'Constrained dense feedback'),
             (frozen['families']['combination'],'Mixture + structure'),
             (recall,'Structure within top 200')]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'svg.fonttype':'none','pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(10,4.5),sharey=True)
    for ax,metric,title in zip(axes,['MAP@10','MeanRecall@50'],['Ranking: MAP@10','Coverage: Mean Recall@50']):
        for i,(name,_) in enumerate(entries):
            stats=data[name][metric];d=100*stats['delta'];lo,hi=[100*x for x in stats['ci95_project_stratified_bootstrap']]
            significant=stats['p_holm_with_recall_control']<.05
            color='#167b63' if lo>0 and significant else '#b84c48' if hi<0 and significant else '#657080'
            ax.errorbar(d,i,xerr=[[d-lo],[hi-d]],fmt='o',color=color,capsize=4,markersize=6,lw=1.7)
        ax.axvline(0,color='#9aa1aa',lw=1,linestyle='--');ax.grid(axis='x',alpha=.15)
        ax.set_title(title,loc='left',fontweight='bold');ax.set_xlabel('Change vs original A (percentage points)')
        ax.set_yticks(range(len(entries)),[label for _,label in entries]);ax.tick_params(axis='y',length=0)
    axes[0].invert_yaxis()
    fig.suptitle('Original-report retrieval: frozen validation (1,794 reports)',x=.03,ha='left',fontweight='bold')
    fig.text(.03,.015,'95% paired, project-stratified intervals. Colors: Holm-adjusted p < .05. Previously observed corpus, not a blind test.',fontsize=8,color='#555555')
    fig.tight_layout(rect=(0,.04,1,.93))
    out=root/'figures';out.mkdir(exist_ok=True)
    for suffix in ('png','svg','pdf'):fig.savefig(out/f'a_validation.{suffix}',dpi=240,bbox_inches='tight')
    plt.close(fig)
    print(out/'a_validation.png')


if __name__=='__main__':main()
