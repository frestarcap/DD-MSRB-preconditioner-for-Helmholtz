#!/usr/bin/env python3
"""Publication figures from frozen half-space fields and verified timing CSVs."""
from pathlib import Path
import csv,json,hashlib,itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.io import loadmat
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/halfspace_figures_20260910';PUB=ROOT/'LaTeX_DL_468198_240419/figures/halfspace_extensions';PUB.mkdir(exist_ok=True)
DATA=ROOT/'outputs/reviewer_20260909';AUDIT=ROOT/'outputs/dd_online_audit_20260910'
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':12,'axes.titlesize':12,'axes.labelsize':12,'legend.fontsize':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
COL={'global':'#c9252d','dd':'#13784d','direct':'#355d9b','shifted':'#777777'}
read=lambda p:list(csv.DictReader(p.open()))
inputs=[]
def rows(p):inputs.append(p);return read(p)
def save(fig,name):
 fig.savefig(PUB/(name+'.pdf'),bbox_inches='tight',pad_inches=.04)
 fig.savefig(OUT/(name+'.png'),dpi=180,bbox_inches='tight',pad_inches=.04);plt.close(fig)
def style(ax):ax.grid(alpha=.17);ax.set_axisbelow(True)
# (b): a field evaluated by exactly the deployed algebraic DD model.
p=OUT/'omega214_field.mat';inputs.append(p);d=loadmat(p);U=d['U'];x=d['x'].ravel();y=d['y'].ravel();source=d['source'].ravel()
assert float(d['rho'].item())<=1e-10 and np.max(np.abs(source-[.2,.5]))<1/504
V=U/np.max(np.abs(U));fig,axs=plt.subplots(1,2,figsize=(8,3.7),layout='constrained')
for ax,values,cmap,lo,hi,title in zip(axs,[V.real,np.log10(np.maximum(abs(V),1e-6))],['RdBu_r','magma'],[-float(np.max(abs(V.real))),-6],[float(np.max(abs(V.real))),0],[r'(a) $\mathrm{Re}(u)/\|u\|_\infty$',r'(b) $\log_{10}(|u|/\|u\|_\infty)$']):
 im=ax.imshow(values,origin='lower',extent=[x[0],x[-1],y[0],y[-1]],cmap=cmap,vmin=lo,vmax=hi,interpolation='nearest',rasterized=True)
 ax.add_patch(Rectangle((0,0),1,1,fill=False,ec='limegreen' if cmap=='magma' else 'black',lw=.8))
 ax.plot([.5,.5],[0,1],'--',color='white' if cmap=='magma' else '.3',lw=.75)
 ax.scatter(*source,s=12,c='white',edgecolors='black',linewidths=.6)
 ax.set(xlabel='$x$',ylabel='$y$',title=title,xticks=[0,.5,1],yticks=[0,.5,1]);fig.colorbar(im,ax=ax,shrink=.87,pad=.02)
save(fig,'omega214_solution')
# (b): frequency storage/time trends, all five queries, with the exceptional
# completion incidence explicitly indicated instead of selecting a subset.
train=rows(DATA/'summary_train.csv')+rows(DATA/'summary_train_frequency_4.csv');rom=[r for r in rows(AUDIT/'paired_timings.csv') if r['optimized']=='1'];base=rows(AUDIT/'baseline_timings.csv');tim=rom+base
omega=np.array([53.5,80.25,107,214]);fig,axs=plt.subplots(1,2,figsize=(8,3.8),layout='constrained')
for method,marker in [('global','o'),('dd','s')]:
 storage=[next(float(r['payload_MiB']) for r in train if float(r['omega'])==w and r['variant']==method and r['study']==f'frequency_{w/53.5:g}') for w in omega]
 axs[0].plot(omega,storage,'-'+marker,color=COL[method],mfc='white',lw=1.6,label='Global TA-ROM' if method=='global' else 'DD-TA-ROM')
 for w,val in zip(omega,storage):
  if method=='dd':axs[0].annotate(f'{val:.1f}',(w,val),xytext=(10,9) if w==53.5 else (0,-15),textcoords='offset points',ha='left' if w==53.5 else 'center',fontsize=9,color=COL[method])
axs[0].set(yscale='log',xlabel=r'$\omega$',ylabel='Persistent ROM storage (MiB)',title='(a) Storage');axs[0].legend(loc='upper left')
# Keep the published minor tick labels across Matplotlib versions.
axs[0].yaxis.set_minor_formatter(matplotlib.ticker.LogFormatterSciNotation(minor_thresholds=(2, .5)))
for method,marker in [('shifted','^'),('direct','d'),('global','o'),('dd','s')]:
 mean=[np.mean([float(r['total_s']) for r in tim if r['variant']==method and float(r['scale'])==w/53.5]) for w in omega]
 axs[1].plot(omega,mean,'-'+marker,color=COL[method],mfc='white',lw=1.5,label={'global':'Global TA-ROM','dd':'DD-TA-ROM','direct':'Direct','shifted':'Shifted ILU'}[method])
axs[1].set(yscale='log',xlabel=r'$\omega$',ylabel='Mean total online time (s)',title='(b) Online cost',ylim=(.005,60));axs[1].legend(loc='upper left',fontsize=9)
axs[1].annotate('1 of 5 queries needs\nSL steps in both models',xy=(214,2.74),xytext=(121,.017),arrowprops={'arrowstyle':'->','color':'.3'},fontsize=9)
for ax in axs:ax.set_xticks(omega,['53.5','80.25','107','214']);style(ax)
save(fig,'frequency_scaling')
# (c): geometry and the actual selected training/independent-validation sets.
selection=json.loads((DATA/'selection_before_holdout.json').read_text());inputs.append(DATA/'selection_before_holdout.json')
fig,axs=plt.subplots(1,2,figsize=(8,3.8),layout='constrained');ax=axs[0]
ax.add_patch(Rectangle((-.25,-.25),1.5,1.5,fc='#e4e4e4',ec='black',lw=1))
ax.add_patch(Rectangle((0,0),.5,1,fc='#dae7f5',ec='black',lw=.8));ax.add_patch(Rectangle((.5,0),.5,1,fc='#fae5d4',ec='black',lw=.8))
ax.plot([.5,.5],[0,1],'k--',lw=1);ax.scatter(.2,.5,s=22,c='black');ax.text(.2,.4,'source',ha='center',fontsize=10)
ax.text(.25,.8,r'$c_{\rm left}$',ha='center');ax.text(.75,.8,r'$c_{\rm right}$',ha='center');ax.text(.5,1.11,'PML',ha='center',fontsize=11)
ax.text(.5,-.14,r'$\Omega=(0,1)^2$',ha='center',fontsize=11)
ax.set(xlim=(-.3,1.3),ylim=(-.3,1.3),aspect='equal',xlabel='$x$',ylabel='$y$',title='(a) Two-material geometry',xticks=[0,.5,1],yticks=[0,.5,1])
ax=axs[1];a,b=np.meshgrid(np.linspace(.95,1.05,8),np.linspace(.8,.95,8));ax.scatter(a,b,s=15,facecolors='white',edgecolors='.5',label=r'$8\times8$ training')
a,b=np.meshgrid(selection['holdout_left'],selection['holdout_right']);ax.scatter(a,b,s=32,c='#6b3a8a',marker='x',label='Independent validation')
ax.set(xlabel=r'$c_{\rm left}$',ylabel=r'$c_{\rm right}$',xlim=(.943,1.057),ylim=(.791,.959),title='(b) Parameter samples');style(ax);ax.legend(loc='upper center',bbox_to_anchor=(.5,-.20),ncol=2,fontsize=9)
save(fig,'two_parameter_setting')
# (c): all nine validation queries; means from five repetitions, residuals max.
selected=DATA/selection['selected'];raw={m:rows(selected/f'{m}_holdout.csv') for m in ['global','dd','direct','shifted']};keys=list(itertools.product(selection['holdout_left'],selection['holdout_right']))
fig,axs=plt.subplots(1,2,figsize=(8,3.8),layout='constrained');q=np.arange(1,10)
for m,marker in [('shifted','^'),('direct','d'),('global','o'),('dd','s')]:
 groups=[[r for r in raw[m] if (float(r['c_left']),float(r['c_right']))==key] for key in keys];assert all(len(g)==5 for g in groups)
 mean=[np.mean([float(r['total_s']) for r in g]) for g in groups]
 axs[0].plot(q,mean,'-'+marker,color=COL[m],mfc='white',lw=1.3,label={'global':'Global TA-ROM','dd':'DD-TA-ROM','direct':'Direct','shifted':'Shifted ILU'}[m])
 if m in ['global','dd']:
  pre=[max(float(r['post_ROM_residual']) for r in g) for g in groups];rho=[max(float(r['true_residual']) for r in g) for g in groups];assert max(rho)<=1e-10
  axs[1].plot(q,pre,'--'+marker,color=COL[m],mfc='white',lw=1,label=f"{'DD' if m=='dd' else 'Global'} after ROM")
  axs[1].plot(q,rho,'-'+marker,color=COL[m],lw=1.2,label=f"{'DD' if m=='dd' else 'Global'} converged")
axs[0].set(ylabel='Total online time (s)',title='(a) Online cost by query',ylim=(0,.67));axs[0].legend(fontsize=9,loc='center',bbox_to_anchor=(.5,.27),ncol=2)
axs[1].axhline(1e-10,color='.45',lw=1,ls=':',label=r'Target $10^{-10}$');axs[1].set(yscale='log',ylabel='True Helmholtz residual',title='(b) After ROM steps and at convergence');axs[1].legend(fontsize=8,loc='center',bbox_to_anchor=(.53,.4))
for ax in axs:ax.set(xlabel='Validation query',xticks=q);style(ax)
save(fig,'two_parameter_results')
with (OUT/'validation_query_key.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['query','c_left','c_right']);w.writerows((i,*key) for i,key in enumerate(keys,1))
(OUT/'figure_provenance.json').write_text(json.dumps({'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},'field_residual':float(d['rho'].item()),'field_reference_error':float(d['relative_error'].item()),'query_order':'left speed outer loop, right speed inner loop','figures':4},indent=2)+'\n')
print('PASS: four figure PDFs; frozen DD field and all nine validation queries verified.')
