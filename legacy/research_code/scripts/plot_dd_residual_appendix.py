from pathlib import Path
import csv, json, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1]
# Figure-only generator: the manuscript fragment is maintained by hand.
a=root/'outputs/dd_highfreq_mechanism_20260913/analysis/20260913_220451_151478/final_layer_components.csv'
b=root/'outputs/dd_highfreq_mechanism_20260913/interface_analysis_all_44244813/native_physical_interfaces.csv'
rows=list(csv.DictReader(a.open())); interfaces=list(csv.DictReader(b.open()))
print('interface fields',sorted(set(r['field'] for r in interfaces)))
plt.rcParams.update({'font.family':'serif','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axs=plt.subplots(1,2,figsize=(7.0,2.8),layout='constrained')
ws=[24,36,48,60]; colors={'global':'#2369a0','dd':'#d05c24'}
for ax,key,title,ylabel,scale in [(axs[0],'residual_roughness_transverse_over_omega2','(a) Transverse short scales',r'$\|\nabla_\perp r\|^2/(\omega^2\|r\|^2)$',1)]:
 for m in colors:
  vals=[[float(r[key])*scale for r in rows if r['method']==m and float(r['omega'])==w] for w in ws]
  assert all(len(v)==3 for v in vals)
  ax.plot(ws,[np.mean(v) for v in vals],'-o',color=colors[m],label='global' if m=='global' else 'DD',ms=4)
  for w,v in zip(ws,vals): ax.scatter([w]*len(v),v,s=10,alpha=.4,color=colors[m])
 ax.set(title=title,ylabel=ylabel);ax.legend(frameon=False)
ax=axs[1]
for m in colors:
 for component,style in [('pre','-o'),('coarse',':^'),('combined','--s')]:
  vals=[np.mean([float(r[f'mg_{component}_optimal_contraction']) for r in rows if r['method']==m and float(r['omega'])==w]) for w in ws]
  ax.plot(ws,vals,style,color=colors[m],ms=4,label=f'{"global" if m=="global" else "DD"}: {component if component!="combined" else "full cycle"}')
ax.set(title='(b) MG correction-direction response',ylabel=r'$\kappa_i$ (smaller is better)',ylim=(0,1.05));ax.legend(frameon=False,fontsize=7.2,ncol=2,columnspacing=.7)
for ax in axs.flat: ax.set_xticks(ws);ax.set_xlabel(r'$\omega$');ax.grid(alpha=.16)
fig.savefig(root/'JCP/figures/dd_mechanism/residual_mg_diagnostics.pdf')
fig.savefig(root/'outputs/dd_appendix_20260920/residual_mg_diagnostics.png',dpi=180)
(root/'outputs/dd_appendix_20260920/provenance.json').write_text(json.dumps({str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [a,b]},indent=2))
