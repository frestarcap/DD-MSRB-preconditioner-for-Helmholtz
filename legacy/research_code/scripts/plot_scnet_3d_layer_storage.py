#!/usr/bin/env python3
"""Plot all three trained layers from recorded logs, including adverse layers.

Layer values were printed to 0.01 MiB. Their sum is checked against the full
report's unrounded basis total before plotting; no checkpoint arrays are read.
"""
import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from plot_scnet_3d import COLORS, read_report, save


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]/'outputs/scnet_3d_20260909')
    parser.add_argument('--profile',type=Path)
    parser.add_argument('--node')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args(); root=args.root; out=args.output or root/'figures/layer_storage'
    out.mkdir(parents=True,exist_ok=True)
    plan=json.loads((root/'expected_cases.json').read_text())
    extras=json.loads((root/'extra_runs.json').read_text())
    sources=[]
    for case in plan['cases']:
        local=root/('remote_'+plan['version'])
        sources.append((local/case['tag']/'report.json',local/'logs'/f"helm3d_sweep_{plan['array_id']}_{case['array_index']}.out"))
    for case in extras:
        if case['kind']=='full':
            local=root/('remote_'+case['version'])
            sources.append((local/case['tag']/'report.json',local/'logs'/f"helm3d_{case['job_id']}.out"))
    profile=json.loads(args.profile.read_text()) if args.profile else {}
    data=[]; provenance=[]
    pattern=re.compile(r'layer (\d+)/3: snapshots (\d+) x (\d+), rank=(\d+), storage ([\d.]+) MiB')
    for report_path,log_path in sources:
        if not report_path.exists(): continue
        report=read_report(report_path)
        if any(report['solver_config'].get(k)!=v for k,v in profile.items()): continue
        if args.node and report['metadata']['node']!=args.node: continue
        layers={'global':{},'dd':{}}; style=None
        for line in log_path.read_text().splitlines():
            if line.startswith('--- Training GLOBAL '): style='global'
            elif line.startswith('--- Training DD '): style='dd'
            match=pattern.search(line)
            if match and style:
                index,n,ns,rank,size=match.groups()
                assert int(n)==report['N']
                layers[style][int(index)]=dict(layer=int(index),N=int(n),training_count=int(ns),
                    rank=int(rank),basis_MiB=float(size))
        for style,rows in layers.items():
            if set(rows)!={1,2,3}:
                raise ValueError(f'{log_path}: incomplete {style} layer log; use checkpoint metadata for a resumed run')
            total=sum(row['basis_MiB'] for row in rows.values())
            assert abs(total-report[f'{style}_basis_MiB'])<=.0150001,(log_path,style,total)
        report['layers']=layers; data.append(report)
        provenance.append(dict(job_id=report['metadata']['job_id'],
            online_accepted=report['all_converged'],
            log=str(log_path.resolve()),log_sha256=hashlib.sha256(log_path.read_bytes()).hexdigest(),
            report=str(report_path.resolve()),report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
            layer_storage_precision_MiB=.01))
    if not data:
        print('No matching complete reports; no layer-storage data fabricated.')
        return
    data.sort(key=lambda d:(d['problem_name'],d['omega'],d['metadata']['job_id']))
    ncols=min(2,len(data)); nrows=math.ceil(len(data)/ncols)
    fig,axes=plt.subplots(nrows,ncols,figsize=(6.0*ncols,3.9*nrows),squeeze=False,layout='constrained')
    metrics=[]
    for ax,d in zip(axes.flat,data):
        g=np.array([d['layers']['global'][i]['basis_MiB'] for i in (1,2,3)])
        dd=np.array([d['layers']['dd'][i]['basis_MiB'] for i in (1,2,3)])
        x=np.arange(3)
        ax.bar(x-.18,g,.34,color=COLORS['global'],label='Global')
        bars=ax.bar(x+.18,dd,.34,color=COLORS['dd'],label='DD')
        for i,(a,b,bar) in enumerate(zip(g,dd,bars)):
            if b>a: bar.set_hatch('///'); bar.set_edgecolor('#8F1D1D')
            ax.text(i,max(a,b)+.045*max(g.max(),dd.max()),f'DD saving {100*(1-b/a):.1f}%',ha='center',fontsize=9)
            for style in ('global','dd'):
                row=d['layers'][style][i+1]
                metrics.append(dict(job_id=d['metadata']['job_id'],node=d['metadata']['node'],
                    problem=d['problem_name'],omega=d['omega'],online_accepted=d['all_converged'],method=style,**row))
        problem='Half-space' if d['problem_name']=='affine_halfspace' else 'Waveguide'
        marker='' if d['all_converged'] else '†'
        ax.set(title=f"{problem}, ω={d['omega']:g}{marker}, N={d['N']:,}\nTotal basis saving {d['basis_saving_percent']:.1f}%; job {d['metadata']['job_id']}",
            ylabel='Basis storage (MiB)',ylim=(0,1.25*max(g.max(),dd.max())))
        ax.set_xticks(x,['Layer 1','Layer 2','Layer 3']); ax.legend(fontsize=9)
        ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True)
    for ax in list(axes.flat)[len(data):]: ax.set_visible(False)
    fig.suptitle('Where the DD storage saving occurs')
    fig.text(.5,-.025,'All three layers retained; hatching marks a layer where DD uses more storage.\n'
        'Layer values: saved logs rounded to 0.01 MiB; sums checked against exact report totals. Basis storage only.'+
        ('\n† At least one online target was not met; the training storage is retained.' if not all(d['all_converged'] for d in data) else ''),
        ha='center',va='top',fontsize=9)
    save(fig,out,'layer_storage')
    with (out/'layer_storage_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(metrics[0])); writer.writeheader(); writer.writerows(metrics)
    (out/'layer_storage_sources.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(f'LAYER_STORAGE_TOTALS_VERIFIED: {len(data)} complete cases; '+str(out))


if __name__=='__main__': main()
