"""Publish frozen uniform5 evidence into JCP figures and tables."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.patches import Rectangle,Ellipse
root=Path(__file__).resolve().parents[1];out=root/'outputs/ecc_dd_budget_20260919';paper=root/'JCP_CAS';figdir=paper/'figures/jcp_shifted'  # active manuscript; table goes to out
plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
colors=['#2879ad','#c64c32'];labels=['Global MSRB','DD-MSRB']
def save(fig,name):
 for ext in ['pdf','png']:fig.savefig(figdir/f'{name}.{ext}',dpi=220,bbox_inches='tight')
 plt.close(fig)
def write_timing_table(target=None):
 v=pd.read_csv(out/'uniform5_summary.csv').set_index('method')
 lines=[r'% Generated from ecc_dd_budget_20260919/uniform5_summary.csv.',r'\begin{table}[!htbp]\centering\normalsize\setlength{\tabcolsep}{3pt}',r'\setlength{\belowcaptionskip}{6pt}',r'\caption{Elliptic-obstacle online cost over 22 fixed test geometries, with five repetitions per query and true residual target $10^{-12}$. Means include conditional shifted-matrix construction and ILU factorization (Setup), the SL steps or the direct solution (Solve), and the final true-residual check (Total). Speedup is relative to the shifted baseline.}',r'\label{tab:ecc-speedup}',r'\begin{tabular}{lrrrrrr}\toprule',r'Method & ROM (s) & Setup (s) & Solve (s) & Total (s) & It. & Speedup \\',r'\midrule']
 for name,label in zip(['uniform5_global','uniform5_dd','shifted','direct'],labels+['Shifted ILU-GMRES','Sparse direct']):
  r=v.loc[name];it='--' if name=='direct' else f'{r.iterations:.2f}';lines.append(f'{label} & {r.ROM_s:.4f} & {r.setup_s:.4f} & {r.solve_s:.4f} & {r.total_s:.4f} & {it} & $ {r.speedup:.2f} \\times $ '+r'\\')
 lines += [r'\bottomrule\end{tabular}',r'\end{table}'];(Path(target) if target is not None else out/'jcp_ecc_timing.tex').write_text('\n'.join(lines)+'\n')

def write_field_figure():
 x=np.loadtxt(out/'paper_x.csv',delimiter=',');y=np.loadtxt(out/'paper_y.csv',delimiter=',');u=np.loadtxt(out/'paper_dd_field.csv',delimiter=',');X,Y=np.meshgrid(x,y);a=.12*np.sqrt(.909176375080444);b=.12/np.sqrt(.909176375080444);u=np.ma.masked_where(((X-.5)/a)**2+((Y-.5)/b)**2<1,u);vmax=np.max(np.abs(u))
 fig=plt.figure(figsize=(3.8,3.8));ax=fig.add_axes([.12,.12,.69,.69]);cax=fig.add_axes([.84,.12,.035,.69]);im=ax.pcolormesh(x,y,u,cmap='RdBu_r',shading='nearest',norm=SymLogNorm(linthresh=vmax*.01,vmin=-vmax,vmax=vmax),rasterized=True);ax.add_patch(Rectangle((0,0),1,1,fill=False,ls='--',lw=.8,ec='.3'));ax.add_patch(Ellipse((.5,.5),2*a,2*b,facecolor='white',edgecolor='.3',lw=.7));ax.set(xlabel='$x$',ylabel='$y$',aspect='equal');cb=fig.colorbar(im,cax=cax,label=r'$\mathrm{Re}\,u$');ticks=np.array([-1,-.1,0,.1,1])*vmax;cb.set_ticks(ticks,labels=[f'{t:.1e}' if t else '0' for t in ticks]);cb.ax.tick_params(labelsize=8);save(fig,'ecc_dd_field')

def write_storage_figure():
 audit=pd.read_csv(out/'uniform5_tail_audit.csv');storage=pd.read_csv(out/'uniform5_storage.csv').set_index('method');fig,axs=plt.subplots(1,2,figsize=(7.2,2.8),layout='constrained')
 for name,offset,col,label in zip(['uniform5_global','uniform5_dd'],[-.18,.18],['#2879ad','#c64c32'],labels):
  t=audit[audit.tag==name];vals=t.basis_entries.to_numpy()*16/2**20;bars=axs[0].bar(np.arange(1,4)+offset,vals,.34,color=col,label=label);axs[0].bar_label(bars,fmt='%.2f',fontsize=7,padding=2)
 axs[0].set(xticks=[1,2,3],xlabel='Layer',ylabel='Basis storage (MiB)',ylim=(0,17));axs[0].legend(frameon=False,fontsize=8,loc='upper left')
 for j in [1,2,3]:
  g=audit[(audit.tag=='uniform5_global')&(audit.layer==j)].iloc[0];d=audit[(audit.tag=='uniform5_dd')&(audit.layer==j)].iloc[0]
  axs[0].text(j+.18,d.basis_entries*16/2**20+1.7,f'{100*(1-d.basis_entries/g.basis_entries):.1f}%\nsaved',ha='center',va='bottom',fontsize=8,color='black')
 for name,offset,col in zip(['uniform5_global','uniform5_dd'],[-.18,.18],colors):
  vals=storage.loc[name,['basis_MiB','payload_MiB','MATLAB_object_MiB']].to_numpy(dtype=float);bars=axs[1].bar(np.arange(3)+offset,vals,.34,color=col);axs[1].bar_label(bars,fmt='%.2f',fontsize=7,padding=2)
 axs[1].set(xticks=np.arange(3),xticklabels=['Basis','Payload','MATLAB object'],ylabel='Persistent storage (MiB)',ylim=(0,41))
 for i,key in enumerate(['basis_MiB','payload_MiB','MATLAB_object_MiB']):axs[1].text(i+.18,float(storage.loc['uniform5_dd',key])+4,f'{100*(1-storage.loc["uniform5_dd",key]/storage.loc["uniform5_global",key]):.1f}%\nsaved',ha='center',va='bottom',fontsize=8,color='black')
 for ax in axs:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
 save(fig,'ecc_storage')

def write_spectra_figure():
 s=pd.read_csv(out/'paper_spectra.csv');fig,ax=plt.subplots(figsize=(3.8,3.5),layout='constrained')
 for j,k,eta,col in zip([1,2,3],[25,42,42],[1e-5,1e-5,1e-5],['#2879ad','#c64c32','#35834b']):
  t=s[s.layer==j];ax.semilogy(t['rank'],t['tail'].clip(lower=1e-16),color=col,label=f'Layer {j}');ax.plot(k,t[t['rank']==k]['tail'].iloc[0],'o',color=col);ax.axhline(eta,color=col,ls='--',lw=.7)
 ax.set(xlabel='Retained rank',ylabel='Relative Frobenius tail',ylim=(1e-10,2));ax.set_box_aspect(1);ax.legend(frameon=False,fontsize=8);ax.grid(alpha=.2)
 save(fig,'ecc_three_layer_spectra')

# Table-only mode leaves all figure and storage-table hand edits untouched.
import sys
if '--spectra-only' in sys.argv:
 write_spectra_figure()
 raise SystemExit(0)
if '--storage-only' in sys.argv:
 write_storage_figure()
 raise SystemExit(0)
if '--ecc-layout-only' in sys.argv:
 write_field_figure();write_storage_figure()
 raise SystemExit(0)
if '--timing-only' in sys.argv or '--check-timing' in sys.argv:
 if '--check-timing' in sys.argv:
  import tempfile
  with tempfile.TemporaryDirectory() as tmp:
   target=Path(tmp)/'timing.tex';write_timing_table(target)
   assert target.read_bytes()==(out/'jcp_ecc_timing.tex').read_bytes(), 'ECC timing generator differs from manuscript'
  print('PASS: ECC timing table matches byte for byte')
 else: write_timing_table()
 raise SystemExit(0)
h=pd.read_csv(out/'paper_histories.csv');fig,ax=plt.subplots(figsize=(3.8,3.15),layout='constrained')
for name,label,col in zip(['uniform5_global','uniform5_dd','shifted'],labels+['Shifted ILU-GMRES'],colors+['#444444']):
 t=h[h.method==name];ax.semilogy(t.step,t.residual,'o-',color=col,ms=3,mfc='white',lw=1,label=label);ax.plot(t.step.iloc[-1],t.residual.iloc[-1],'s',color=col,ms=4)
ax.axhline(1e-12,color='.5',ls='--',lw=.7);ax.set(xlabel='ROM steps + SL steps',ylabel='Relative residual',ylim=(1e-15,2));ax.legend(fontsize=7,loc='upper right');ax.grid(alpha=.2);save(fig,'ecc_residuals')
write_field_figure()
write_spectra_figure()
write_storage_figure()
audit=pd.read_csv(out/'uniform5_tail_audit.csv');storage=pd.read_csv(out/'uniform5_storage.csv').set_index('method')
write_timing_table()
lines=[r'\begin{table}[!htbp]\centering\normalsize',r'\caption{Three-layer elliptic-obstacle ROM storage in MiB. DD ranks are ordered as near field, remaining physical region, and PML. Payload adds one shared index map; the MATLAB object includes retained per-layer indices and metadata. These measures exclude full-order and completion data and do not measure process peak memory.}',r'\label{tab:ecc-storage}',r'\begin{tabular}{cccrrr}\toprule',r'Layer & Global rank & DD ranks & Global & DD & Saving \\',r'\midrule']
for j in [1,2,3]:
 g=audit[(audit.tag=='uniform5_global')&(audit.layer==j)].iloc[0];d=audit[(audit.tag=='uniform5_dd')&(audit.layer==j)].iloc[0];gm=g.basis_entries*16/2**20;dm=d.basis_entries*16/2**20;rank=d['ranks'].replace('[','(').replace(']',')').replace(' ',',');lines.append(f'{j} & {g.dimension} & ${rank}$ & {gm:.3f} & {dm:.3f} & {100*(1-dm/gm):.1f}\\% '+r'\\')
lines += [r'\midrule']
for key,label in [('basis_MiB','Basis total'),('shared_map_MiB','Shared indices'),('payload_MiB','Payload'),('MATLAB_object_MiB',r'MATLAB object')]:
 g=storage.loc['uniform5_global',key];d=storage.loc['uniform5_dd',key];saving='--' if g==0 else f'{100*(1-d/g):.1f}\\%';lines.append(f'\\multicolumn{{3}}{{l}}{{{label}}} & {g:.3f} & {d:.3f} & {saving} '+r'\\')
lines += [r'\bottomrule\end{tabular}',r'\end{table}'];(paper/'jcp_ecc_storage.tex').write_text('\n'.join(lines)+'\n')
