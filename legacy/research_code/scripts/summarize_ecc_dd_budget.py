"""Summarize retained candidate evidence and plot measured basis storage/time."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1]
out=root/'outputs/ecc_dd_budget_20260919'
train=pd.read_csv(out/'training_summary.csv').set_index('tag')
audit=pd.read_csv(out/'tail_audit.csv')
val=pd.read_csv(out/'validate_summary.csv').set_index('method')
tags=['uniform6_global','uniform6_dd','last3_global','last3_dd']
labels=['Global\nuniform','DD\nuniform','Global\nlayerwise','DD\nlayerwise']
fig,axes=plt.subplots(1,2,figsize=(10,4.3),layout='constrained')
bottom=np.zeros(4)
for j,color in zip(range(1,4),['#85b6cf','#3b8db2','#164f70']):
    values=np.array([audit[(audit.tag==t)&(audit.layer==j)].basis_entries.iloc[0]*16/2**20 for t in tags])
    axes[0].bar(labels,values,bottom=bottom,label=f'Layer {j}',color=color);bottom+=values
axes[0].set_ylabel('Basis storage (MiB)');axes[0].legend(frameon=False)
for i,v in enumerate(bottom):axes[0].text(i,v+.4,f'{v:.2f}',ha='center')
axes[0].set_ylim(0,49)
values=val.loc[tags,'total_s'].to_numpy()*1000
axes[1].bar(labels,values,color=['#85b6cf','#164f70']*2)
for i,v in enumerate(values):axes[1].text(i,v+1,f'{v:.1f}',ha='center')
axes[1].axhline(val.loc['direct','total_s']*1000,color='#888',linestyle='--',label='Sparse direct')
axes[1].set_ylabel('Mean online time (ms)');axes[1].legend(frameon=False)
axes[1].set_title('22 new queries; 5 repeats; residual ≤ 1e−12')
fig.savefig(out/'storage_timing.png',dpi=180);fig.savefig(out/'storage_timing.pdf')
lines=['# Strict DD budget study','', 'Basis storage excludes index maps and is not process peak memory.','', '| Configuration | Basis MiB | Payload MiB | Time s | Speedup vs shifted | Max field error |','|---|---:|---:|---:|---:|---:|']
for t in tags:
    r=train.loc[t];v=val.loc[t]
    lines.append(f'| {t} | {r.basis_MiB:.3f} | {r.payload_MiB:.3f} | {v.total_s:.5f} | {v.speedup:.2f} | {v.max_error:.3e} |')
lines+=['','All candidate development results: screen_summary.csv; actual tail audit: tail_audit.csv.']
(out/'results.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
