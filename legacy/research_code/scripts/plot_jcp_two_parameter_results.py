#!/usr/bin/env python3
"""Combine original Fig. 11(a) cost with original Fig. 12(b) time ratios."""
from pathlib import Path
import csv
import itertools
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'outputs/reviewer_20260909'
PUB=ROOT/'JCP/figures/halfspace_extensions'
selection=json.loads((DATA/'selection_before_holdout.json').read_text())
keys=list(itertools.product(selection['holdout_left'],selection['holdout_right']))
means={}
for method in ['shifted','direct','global','dd']:
    with (DATA/selection['selected']/f'{method}_holdout.csv').open() as f:
        rows=list(csv.DictReader(f))
    groups=[[r for r in rows if (float(r['c_left']),float(r['c_right']))==key] for key in keys]
    assert all(len(g)==5 for g in groups)
    assert max(float(r['true_residual']) for r in rows)<=1e-10
    means[method]=np.array([np.mean([float(r['total_s']) for r in g]) for g in groups])
ratio=means['global']/means['dd']
# The former Fig. 12 used this saved query ordering and these ratios.
with (ROOT/'outputs/paired_times_20260921/halfspace_two_parameter_time_ratios.csv').open() as f:
    old=list(csv.reader(f))[1:]
assert np.allclose(ratio,[float(r[-1]) for r in old],rtol=1e-12,atol=0)
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':12,'pdf.fonttype':42,
                     'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,2,figsize=(8,3.8),layout='constrained');q=np.arange(1,10)
colors={'global':'#c9252d','dd':'#13784d','direct':'#355d9b','shifted':'#777777'}
labels={'global':'Global TA-ROM','dd':'DD-TA-ROM','direct':'Direct','shifted':'Shifted ILU'}
for method,marker in [('shifted','^'),('direct','d'),('global','o'),('dd','s')]:
    axs[0].plot(q,means[method],'-'+marker,color=colors[method],mfc='white',lw=1.3,label=labels[method])
axs[0].set(ylabel='Total online time (s)',title='(a) Online cost by query',ylim=(0,.67))
axs[0].legend(fontsize=9,loc='center',bbox_to_anchor=(.5,.27),ncol=2)
axs[1].plot(q,ratio,'o-',color='#75529A',ms=3,lw=1)
axs[1].axhline(1,color='.35',ls='--',lw=.8)
axs[1].set(ylabel=r'$T_{\mathrm{Global}}/T_{\mathrm{DD}}$',title='(b) Paired time ratios')
axs[1].margins(y=.18)
axs[1].text(.98,.96,'Above 1: DD faster',ha='right',va='top',transform=axs[1].transAxes,fontsize=9,
            bbox={'facecolor':'white','edgecolor':'none','alpha':.85})
for ax in axs:
    ax.set(xlabel='Validation query',xticks=q);ax.grid(alpha=.17);ax.set_axisbelow(True)
for ext in ['pdf','png']:
    fig.savefig(PUB/f'two_parameter_results.{ext}',dpi=250,bbox_inches='tight',pad_inches=.04)
plt.close(fig)
print('PASS: 9 queries x 5 repeats; ratios match original Fig. 12(b).')
