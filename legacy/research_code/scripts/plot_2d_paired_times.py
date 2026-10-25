"""Four paired timing figures, from frozen evidence only."""
from pathlib import Path
import json,hashlib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];F=R/'JCP/figures/paired_times';F.mkdir(exist_ok=True);O=R/'outputs/paired_times_20260921';O.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
C={'global':'#BF3030','dd':'#23794C','shifted':'#667085','direct':'#3768A0'};L={'global':'Global TA-ROM','dd':'DD-TA-ROM','shifted':'Shifted ILU-GMRES','direct':'Sparse direct'};inputs=[]
def read(p):inputs.append(p);return pd.read_csv(p)
def finish(fig,axs,name,xlabel):
 for ax in axs:ax.set_xlabel(xlabel);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
 axs[1].axhline(1,color='.35',ls='--',lw=.8);axs[1].set_ylabel(r'$T_{\mathrm{Global}}/T_{\mathrm{DD}}$');axs[1].text(.98,.96,'Above 1: DD faster',ha='right',va='top',transform=axs[1].transAxes,fontsize=7,bbox={'facecolor':'white','edgecolor':'none','alpha':.85})
 handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=4,fontsize=7,frameon=False)
 for ext in ['pdf','png']:fig.savefig(F/f'{name}.{ext}',dpi=200,bbox_inches='tight')
 plt.close(fig)
def queryplot(d,name,figsize=(7.4,3.0)):
 d=d[d.method.isin(C)].copy();g=d.groupby(['method','query']).total_s.agg(['mean','std','count']);assert (g['count']==5).all()
 fig,axs=plt.subplots(1,2,figsize=figsize,layout='constrained');n=d['query'].nunique()
 for m in C:
  t=g.loc[m];x=t.index.to_numpy();axs[0].errorbar(x,t['mean'],yerr=t['std'],color=C[m],marker='s' if m=='dd' else 'o',ms=3,lw=1,capsize=2,label=L[m])
 ratio=g.loc['global','mean']/g.loc['dd','mean'];axs[1].plot(ratio.index,ratio,'o-',color='#75529A',ms=3,lw=1)
 axs[0].set(yscale='log',ylabel='Total online time (s)',title='(a) Total online time');
 axs[1].set(title='(b) Paired time ratios');axs[1].margins(y=.18)
 ticks=np.arange(1,n+1,max(1,n//6))
 for ax in axs:ax.set_xticks(ticks)
 g.to_csv(O/f'{name}_query_summary.csv');ratio.to_csv(O/f'{name}_ratios.csv');finish(fig,axs,name,'Test query')
d=read(R/'outputs/jcp_shifted_20260914/paper_timings.csv');queryplot(d[d.problem=='halfspace'],'halfspace_representative_time',figsize=(6.8,3.2))
d=read(R/'outputs/ecc_dd_budget_20260919/uniform5_timings.csv');d.method=d.method.str.replace('uniform5_','',regex=False);queryplot(d,'ecc_time')
p=R/'outputs/reviewer_20260909/selection_before_holdout.json';inputs.append(p);sel=json.loads(p.read_text());frames=[]
keys={(l,r):i+1 for i,(l,r) in enumerate(( (l,r) for l in sel['holdout_left'] for r in sel['holdout_right']))}
for m in C:
 d=read(p.parent/sel['selected']/f'{m}_holdout.csv');d['method']=m;d['query']=[keys[(l,r)] for l,r in zip(d.c_left,d.c_right)];frames.append(d)
queryplot(pd.concat(frames),'halfspace_two_parameter_time')
p=R/'outputs/dd_online_audit_20260910';d=read(p/'paired_timings.csv');d=d[d.optimized==1];d=pd.concat([d,read(p/'baseline_timings.csv')]);g=d.groupby(['variant','scale','c_right']).total_s.agg(['mean','count']);assert (g['count']==5).all();fig,axs=plt.subplots(1,2,figsize=(7.4,3.0),layout='constrained')
for m in C:
 t=g.loc[m]['mean'].unstack('c_right');avg=t.mean(axis=1);w=t.index.to_numpy()*53.5;axs[0].errorbar(w,avg,yerr=[avg-t.min(axis=1),t.max(axis=1)-avg],marker='o',color=C[m],lw=1,capsize=3,label=L[m])
ratio=(g.loc['global','mean']/g.loc['dd','mean']).unstack('c_right')
for q in ratio.columns:axs[1].plot(ratio.index.to_numpy()*53.5,ratio[q],'o-',ms=3,lw=1,label=f'{q:.4f}')
axs[0].set(yscale='log',ylabel='Total online time (s)',title='(a) Five-query mean and range');axs[1].set(title='(b) Paired time ratios');axs[1].margins(y=.2);axs[1].legend(title=r'$c_{\rm right}$',fontsize=6,title_fontsize=7,loc='lower right',ncol=2,frameon=False)
for ax in axs:ax.set_xticks([53.5,80.25,107,214],['53.5','80.25','107','214'])
g.to_csv(O/'halfspace_frequency_query_summary.csv');ratio.to_csv(O/'halfspace_frequency_ratios.csv');finish(fig,axs,'halfspace_frequency_time',r'Angular frequency $\omega$')
(O/'sources.json').write_text(json.dumps({str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},indent=2))
print('PASS: four figures; five repetitions per method/query checked.')
