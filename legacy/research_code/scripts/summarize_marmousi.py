"""Summarize paired timings; plot actual Marmousi data and solver evidence."""
import csv
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

root = Path(__file__).resolve().parents[1]
out = root / 'outputs/marmousi_20260920' / (sys.argv[1] if len(sys.argv) > 1 else 'paired')
rows = list(csv.DictReader((out / 'timings.csv').open()))
training = list(csv.DictReader((out / 'training.csv').open()))
storage = list(csv.DictReader((out / 'storage.csv').open()))
methods = list(dict.fromkeys(r['method'] for r in rows))
groups = list(dict.fromkeys(r['group'] for r in rows))
metrics = ['total_s', 'ROM_s', 'setup_s', 'solve_s', 'iterations', 'post_ROM']
summary = []
for group in groups:
    for method in methods:
        rr = [r for r in rows if r['group'] == group and r['method'] == method]
        # Median over repeats first, equal weight for each held-out query.
        ids = sorted({r['query'] for r in rr})
        item = dict(group=group, method=method, queries=len(ids))
        for metric in metrics:
            item[metric] = statistics.mean(statistics.median(float(r[metric]) for r in rr if r['query'] == i) for i in ids)
        item['max_residual'] = max(float(r['residual']) for r in rr)
        item['max_error'] = max(float(r['error']) for r in rr)
        summary.append(item)
for r in summary:
    base = next(s['total_s'] for s in summary if s['group'] == r['group'] and s['method'] == 'shifted')
    r['speedup'] = base / r['total_s']
    tr = next((s for s in training if s['method'] == r['method']), None)
    r['break_even_solves'] = (float(tr['offline_s']) / (base-r['total_s']) if tr and r['total_s'] < base else float('nan'))
with (out/'summary.csv').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)

d = loadmat(out/'plot_data.mat', squeeze_me=True, struct_as_record=False)
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(2, 2, figsize=(13, 8), layout='constrained')
fig.suptitle(f"Marmousi feasibility: {d['labels'].size:,} unknowns, snapshot tail {d['cfg'].tail:.0e}",fontsize=13)
x, z = d['x'], d['z']
im = ax[0,0].imshow(d['velocity'].T, extent=[x[0],x[-1],z[-1],z[0]], aspect='auto', cmap='turbo')
ax[0,0].plot(6,.1,'w*',ms=10,mec='k',mew=.5)
ax[0,0].set(title='Marmousi velocity (Devito benchmark)', xlabel='x (km)', ylabel='Depth (km)')
fig.colorbar(im, ax=ax[0,0],label='km/s', fraction=.025)
ax[0,1].imshow(d['labels'].T, origin='upper', aspect='auto', cmap='tab10',interpolation='nearest')
ax[0,1].set(title='DD: 3 physical columns + 4 PML sides', xlabel='Grid index x', ylabel='Grid index z')
chosen=['frequency','iid5','iid10','smooth10']
colors=['#d45135','#247bab','#469968']
labels=['Global TA (frequency trained)','DD TA (frequency trained)','DD TA (pointwise trained)']
drawmethods=['frequency_global','frequency_dd','iid5_dd']
for j,(method,color,label) in enumerate(zip(drawmethods,colors,labels)):
    if method not in methods: continue
    vals=[next(r['speedup'] for r in summary if r['group']==g and r['method']==method) for g in chosen]
    ax[1,0].bar(np.arange(4)+(j-1)*.23,vals,width=.22,color=color,label=label)
ax[1,0].axhline(1,color='k',lw=.8,ls='--')
ax[1,0].set(xticks=np.arange(4),xticklabels=['New frequency','Pointwise ±5%','Pointwise ±10%','Smooth ±10%'],
    ylabel='Speedup vs shifted GMRES',title='Online cost includes ROM and ILU setup')
ax[1,0].set_ylim(0,1.5)
ax[1,0].legend(fontsize=8,loc='upper left')
direct_time=next(r['total_s'] for r in summary if r['group']=='frequency' and r['method']=='direct')
ax[1,0].text(.02,.04,f'Sparse direct at new frequencies: {direct_time:.3f} s (fastest)',
    transform=ax[1,0].transAxes,fontsize=8,bbox={'facecolor':'white','edgecolor':'none','alpha':.9})
trnames=['frequency_global','frequency_dd','iid5_global','iid5_dd']
tr=[next((r for r in training if r['method']==n),None) for n in trnames]
valid=[(n,r) for n,r in zip(trnames,tr) if r]
bottom=np.zeros(len(valid))
for key,color,label in [('basis_bytes','#247bab','Bases'),('reduced_bytes','#e5ae38','Reduced matrices'),('shared_index_bytes','#999999','Shared indices')]:
    heights=np.array([float(next(s[key] for s in storage if s['method']==n))/2**20 for n,r in valid])
    ax[1,1].bar(range(len(valid)),heights,bottom=bottom,color=color,label=label)
    bottom+=heights
display_names={'frequency_global':'Frequency\nGlobal','frequency_dd':'Frequency\nDD',
    'iid5_global':'Pointwise\nGlobal','iid5_dd':'Pointwise\nDD'}
ax[1,1].set(xticks=range(len(valid)),xticklabels=[display_names[n] for n,r in valid],
    ylabel='Deployment numerical arrays (MiB)',title='Three layers, identical snapshot-tail tolerance')
ax[1,1].set_ylim(0,max(bottom)*1.28)
ax[1,1].legend(fontsize=8,loc='upper right')
fig.savefig(out/'overview.png',dpi=180)
fig.savefig(out/'overview.pdf')
plt.close(fig)

fig,ax=plt.subplots(1,2,figsize=(12,4),layout='constrained')
ns=[str(s) for s in np.atleast_1d(d['names'])]
qgroups=[str(s) for s in np.atleast_1d(d['query_groups'])]
for a,qi in zip(ax,[max(i for i,g in enumerate(qgroups) if g=='frequency'),len(qgroups)-1]):
    for method,color in [('shifted','#333333'),('frequency_global','#d45135'),('frequency_dd','#247bab')]:
        hist=np.asarray(d['histories'][qi,ns.index(method)]).ravel()
        a.semilogy(range(len(hist)),hist,label=method,color=color)
    a.axhline(1e-8,color='gray',ls='--',lw=.8)
    a.set(xlabel='ROM + completion steps',ylabel='True relative residual',title=f"{qgroups[qi]}, {d['frequencies'][qi]:.3f} Hz")
    a.legend(fontsize=8)
fig.savefig(out/'residuals.png',dpi=180)
plt.close(fig)
for r in summary:
    if r['group'] in chosen:
        print(f"{r['group']:10s} {r['method']:20s} {r['total_s']:.4f}s speedup={r['speedup']:.3f} it={r['iterations']:.1f} post={r['post_ROM']:.2e}")
print('MARMOUSI_REPORT_PASS')
