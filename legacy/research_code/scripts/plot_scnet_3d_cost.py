#!/usr/bin/env python3
"""Plot measured costs with separate offline, online, workflow and RSS scopes."""
import argparse
import csv
import json
from pathlib import Path

from plot_scnet_3d import COLORS, LABELS, read_report, save, solver_caption
import matplotlib.pyplot as plt
import numpy as np


def rss_gib(value):
    units={'K':2**-20,'M':2**-10,'G':1.0,'T':2**10}
    return float(value[:-1])*units[value[-1]] if value and value[-1] in units else float('nan')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]/'outputs/scnet_3d_20260909')
    parser.add_argument('--profile',type=Path)
    parser.add_argument('--node')
    parser.add_argument('--problem',choices=('waveguide','halfspace'),default='waveguide')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args(); root=args.root
    out=args.output or root/'figures/costs'; out.mkdir(parents=True,exist_ok=True)
    profile=json.loads(args.profile.read_text()) if args.profile else {}
    with (root/'accounting.psv').open() as stream:
        accounting={row['JobIDRaw']:row for row in csv.DictReader(stream,delimiter='|')}
    name='gaussian_waveguide' if args.problem=='waveguide' else 'affine_halfspace'
    data=[]
    for path in root.glob('remote_*/full_*/report.json'):
        d=read_report(path)
        if d['problem_name']!=name: continue
        if any(d['solver_config'].get(k)!=v for k,v in profile.items()): continue
        if args.node and d['metadata']['node']!=args.node: continue
        with next(path.parent.glob('helmholtz_3d_candidates_*.csv')).open() as stream:
            rows=list(csv.DictReader(stream))
        d['offline']={m:float(next(r for r in rows if r['method']==m)['offline_s']) for m in ('global','dd')}
        batch=accounting.get(str(d['metadata']['job_id'])+'.batch',{})
        d['rss_GiB']=rss_gib(batch.get('MaxRSS',''))
        d['planned_GiB']=json.loads((path.parent/'configuration.json').read_text())['estimated_peak_GiB']
        data.append(d)
    data.sort(key=lambda d:(d['omega'],d['metadata']['job_id']))
    if not data:
        print('No matching complete reports; no cost data fabricated.'); return
    x=np.arange(len(data)); ticks=[f"{d['omega']:g}{'' if d['all_converged'] else '†'}\n{d['N']/1e6:.2f}M\njob {d['metadata']['job_id']}" for d in data]
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for method,offset in [('global',-.18),('dd',.18)]:
        times=[d['offline'][method]/60 if not d['training_resumed'] else np.nan for d in data]
        axes[0,0].bar(x+offset,times,.34,color=COLORS[method],label=method.upper())
    axes[0,0].set(ylabel='Standalone offline time (min)',title='(a) Each model’s training cost')
    axes[0,0].legend()
    for method in COLORS:
        values=[np.array([q[method]['seconds'] for q in d['tests']]) for d in data]
        means=np.array([np.mean(v) for v in values])
        axes[0,1].errorbar(x,means,yerr=[means-[min(v) for v in values],[max(v) for v in values]-means],
            fmt='o-',capsize=4,color=COLORS[method],label=LABELS[method])
    axes[0,1].set(ylabel='Online time (s)',title='(b) Three queries: mean and range')
    axes[0,1].set_ylim(bottom=0)
    axes[0,1].legend(fontsize=8)
    bars=axes[1,0].bar(x,[d['elapsed_seconds']/60 for d in data],.55,color='#49586D')
    axes[1,0].bar_label(bars,fmt='%.1f',padding=3)
    axes[1,0].set(ylabel='Actual numerical workflow (min)',title='(c) Both models + nine online solves')
    axes[1,0].set_ylim(0,1.2*max(d['elapsed_seconds']/60 for d in data))
    axes[1,1].bar(x-.18,[d['planned_GiB'] for d in data],.34,color='#AAB2BF',label='Planning estimate')
    bars=axes[1,1].bar(x+.18,[d['rss_GiB'] for d in data],.34,color='#316C65',label='Slurm batch MaxRSS')
    axes[1,1].bar_label(bars,fmt='%.1f',padding=3)
    axes[1,1].axhline(192,color='black',ls='--',lw=1,label='Allocation: 192 GiB')
    axes[1,1].set(ylabel='Memory (GiB)',ylim=(0,215),title='(d) Memory estimate and recorded peak')
    axes[1,1].legend(fontsize=8,loc='upper left')
    for ax in axes.flat:
        ax.set_xticks(x,ticks); ax.set_xlabel('Angular frequency ω / unknowns / job'); ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True)
    fig.suptitle('3D '+('waveguide' if args.problem=='waveguide' else 'half-space')+': cost of the measured workflow')
    fig.text(.5,-.035,'Training: 8 workers, thread limit 4 each. Online: serial queries, parent thread limit 8.\n'
        'Standalone training charges the shared first layer to each model; actual workflow shares it once.\n'
        'Online times exclude common matrix assembly; measured MG setup is charged to each method that needs it.\n'
        'Ranges span three parameters, not repeat timings. MaxRSS is distinct from persistent ROM storage.\n'+solver_caption(data)+
        ('\n† At least one online target was not met; its measured cost is retained.' if not all(d['all_converged'] for d in data) else ''),
        ha='center',va='top',fontsize=8)
    save(fig,out,'workflow_cost')
    with (out/'workflow_cost_metrics.csv').open('w',newline='') as stream:
        fields=['job_id','omega','N','global_offline_s','dd_offline_s','workflow_s','planned_GiB','batch_MaxRSS_GiB','training_resumed','all_converged']
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
        for d in data:
            writer.writerow(dict(job_id=d['metadata']['job_id'],omega=d['omega'],N=d['N'],global_offline_s=d['offline']['global'],
                dd_offline_s=d['offline']['dd'],workflow_s=d['elapsed_seconds'],planned_GiB=d['planned_GiB'],batch_MaxRSS_GiB=d['rss_GiB'],
                training_resumed=d['training_resumed'],all_converged=d['all_converged']))
    print(out)


if __name__=='__main__': main()
