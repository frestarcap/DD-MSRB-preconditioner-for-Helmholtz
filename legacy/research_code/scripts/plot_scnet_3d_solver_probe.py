#!/usr/bin/env python3
"""Draw every completed configuration in a retained-RHS solver diagnostic."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); data=json.loads(args.report.read_text())
    rows=data['rows'] if isinstance(data['rows'],list) else [data['rows']]
    args.output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,(ax,bar)=plt.subplots(1,2,figsize=(13,5.6),layout='constrained')
    colors=plt.get_cmap('tab10')(np.arange(len(rows))%10)
    scale=60 if max(r['seconds'] for r in rows)>=120 else 1
    unit='minutes' if scale==60 else 'seconds'; short='min' if scale==60 else 's'
    labels=[]; times=[]
    for i,row in enumerate(rows):
        history=np.atleast_1d(row['history']).astype(float)
        if history.size:
            ax.semilogy(np.arange(history.size),history,lw=1.4,color=colors[i],label=str(i+1))
            if row['true_residual'] is not None:
                ax.semilogy(history.size-1,row['true_residual'],
                    'o' if row['converged'] else 'x',color=colors[i],ms=6)
        coarse=f"{int(np.prod(row['coarse_grid_size'])):,}" if row['coarse_grid_size'] else '?'
        labels.append(f"{i+1}: β={row['shift_beta']:g}, m={row['restart']}, Nc={coarse}")
        times.append(row['seconds']/scale)
    ax.axhline(1e-7,ls='--',color='black',lw=1)
    ax.set(xlabel='GMRES iterations',ylabel='Relative residual',title='(a) All residual histories',ylim=(1e-8,2))
    ax.legend(title='Configuration',ncol=4,fontsize=9); ax.grid(alpha=.2)
    bars=bar.barh(np.arange(len(rows)),times,color=colors)
    for item,row in zip(bars,rows):
        if not row['converged']: item.set_hatch('///'); item.set_edgecolor('#981B1E')
    text=[f"{t:.2f} {short} / {int(r['iterations'])} it" if r['iterations'] is not None else f'{t:.2f} {short} / error' for t,r in zip(times,rows)]
    bar.bar_label(bars,labels=text,padding=4,fontsize=9)
    bar.set_yticks(np.arange(len(rows)),labels,fontsize=9); bar.invert_yaxis()
    bar.set(xlabel=f'Measured wall time ({unit})',title='(b) Cost and acceptance',xlim=(0,max(.01,max(times)*1.5)))
    bar.grid(axis='x',alpha=.2); bar.set_axisbelow(True)
    for axis in (ax,bar): axis.spines[['top','right']].set_visible(False)
    cfg=data['problem_cfg']; baseline=data['baseline']
    fig.suptitle(f"3D solver sensitivity: ω = {cfg['omega']:g}, N = {int(data['N']):,}, μ = {baseline['mu']:.6f}")
    fig.text(.5,-.035,'m = restart; Nc = actual direct coarse-grid unknowns. Hatched bars did not reach 10⁻⁷.\n'
             'Same retained RHS; configurations run concurrently. Interior curve values are Krylov estimates; final residuals are recomputed.',
             ha='center',fontsize=9)
    for suffix in ('png','pdf','svg'):
        fig.savefig(args.output/f'solver_probe.{suffix}',dpi=220,bbox_inches='tight')
    plt.close(fig)
    fields=['mu','shift_beta','restart','mg_coarse_max_dofs','level_count','iterations',
            'true_residual','seconds','setup_seconds','solve_seconds','converged','error']
    with (args.output/'solver_probe_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
        for row in rows: writer.writerow({key:row.get(key) for key in fields})
    print(args.output)


if __name__=='__main__': main()
