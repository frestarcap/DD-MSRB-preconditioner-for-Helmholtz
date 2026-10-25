#!/usr/bin/env python3
"""Generate reviewer tables from saved CSVs, preserving adverse configurations."""
from pathlib import Path
import csv, math, statistics as st, sys
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/reviewer_20260909'
TEX=ROOT/'LaTeX_DL_468198_240419'
def read(p):
    with p.open() as f:return list(csv.DictReader(f))
def num(r,k):return float(r[k])
def sci(x):
    if x==0:return '0'
    a,e=f'{x:.2e}'.split('e');return rf'\({a}\times10^{{{int(e)}}}\)'
def ranks(r):return r['ranks'].replace('[',r'\((').replace(']',r')\)').replace(' ',',')
def table(caption,label,cols,header,rows):
    return '\n'.join([r'\begin{table}[t]\centering\scriptsize',rf'\caption{{{caption}}}\label{{{label}}}',r'\setlength{\tabcolsep}{3pt}',rf'\begin{{tabular}}{{{cols}}}\toprule',header+r' \\','\\midrule',*[r' & '.join(map(str,r))+r' \\' for r in rows],r'\bottomrule\end{tabular}\end{table}',''])
def timing(case,method,suffix='timing'):
    rows=read(OUT/case/f'{method}_{suffix}.csv')
    assert all(num(r,'true_residual')<=1e-10 for r in rows)
    groups={}
    for r in rows:groups.setdefault((r['c_left'],r['c_right']),[]).append(num(r,'total_s'))
    reps={len(v) for v in groups.values()};assert len(reps)==1
    return dict(time=st.mean(num(r,'total_s') for r in rows),iterations=st.mean(num(r,'completion_iterations') for r in rows),maxiter=max(num(r,'completion_iterations') for r in rows),pre=max(num(r,'post_ROM_residual') for r in rows),rho=max(num(r,'true_residual') for r in rows),sd=max(st.stdev(v) for v in groups.values()),reps=reps.pop())
# Consolidated frequency table: frozen construction data plus the common
# September 10 implementation and refreshed baseline timing protocol.
AUDIT=ROOT/'outputs/dd_online_audit_20260910'
def frequency_rows(case,method):
    scale=float(case.removeprefix('frequency_'))
    if method in ['global','dd']:
        raw=read(AUDIT/'paired_timings.csv')
        assert len(raw)==400
        rows=[dict(r) for r in raw if num(r,'scale')==scale and r['variant']==method and r['optimized']=='1']
    else:
        raw=read(AUDIT/'baseline_timings.csv')
        assert len(raw)==200, 'Finish all refreshed baseline repetitions before generating the paper table.'
        rows=[dict(r) for r in raw if num(r,'scale')==scale and r['variant']==method]
    assert len(rows)==25
    expected={(mu,rep) for mu in [0.8053,0.8297,0.8761,0.9229,0.9431] for rep in range(1,6)}
    assert {(num(r,'c_right'),int(r['repetition'])) for r in rows}==expected
    assert all(num(r,'true_residual')<=1e-10 for r in rows)
    for r in rows:
        r['c_left']='1'
        r.setdefault('post_ROM_residual','nan')
    return rows

def frequency_timing(case,method):
    rows=frequency_rows(case,method)
    groups={}
    for r in rows:groups.setdefault(r['c_right'],[]).append(num(r,'total_s'))
    return dict(time=st.mean(num(r,'total_s') for r in rows),
        iterations=st.mean(num(r,'completion_iterations') for r in rows),
        maxiter=max(num(r,'completion_iterations') for r in rows),
        pre=max(num(r,'post_ROM_residual') for r in rows),
        rho=max(num(r,'true_residual') for r in rows),
        sd=max(st.stdev(v) for v in groups.values()),reps=5)

base=read(OUT/'summary_train.csv')
extension=OUT/'summary_train_frequency_4.csv'
if extension.exists():
    base=[r for r in base if r['study']!='frequency_4']+read(extension)
lookup={(r['study'],r['variant']):r for r in base}
for r in base:
    assert num(r,'max_snapshot_residual')<1e-9
    assert num(r,'max_tail')<1.001e-5
freq=[]; costs=[]; amort=[]; frequency_results=[]
frequency_cases=['frequency_1','frequency_1.5','frequency_2','frequency_4']
assert all((case,method) in lookup for case in frequency_cases for method in ['global','dd'])
for c in frequency_cases:
    g,d=(lookup[c,m] for m in ['global','dd'])
    save=100*(1-num(d,'payload_MiB')/num(g,'payload_MiB'))
    freq.append([g['omega'],f"{int(num(g,'N')):,}",ranks(g)+' / '+ranks(d),f"{num(g,'payload_MiB'):.2f}/{num(d,'payload_MiB'):.2f}",f"{save:.1f}\\%",f"{num(g,'offline_s'):.2f}/{num(d,'offline_s'):.2f}"])
    tg,td,dr,sl=(frequency_timing(c,m) for m in ['global','dd','direct','shifted'])
    assert all(t['reps']==5 for t in [tg,td,dr,sl])
    iters='/'.join('0' if t['iterations']==0 else f"{t['iterations']:.1f}" for t in [sl,tg,td])
    costs.append([g['omega'],f"{dr['time']:.4f}",f"{sl['time']:.4f}",f"{tg['time']:.4f}",f"{td['time']:.4f}",iters,sci(max(tg['rho'],td['rho']))])
    amort.append(num(d,'offline_s')/(dr['time']-td['time']) if dr['time']>td['time'] else math.inf)
    frequency_results.append(dict(case=c,g=g,d=d,tg=tg,td=td,dr=dr,sl=sl,saving=save))
details=table('Fixed-points-per-wavelength frequency study. Paired entries are Global/DD; payload is persistent ROM storage in MiB. Offline costs are single measurements.','tab:frequency-storage-details','rrlrrr',r'$\omega$ & $N_h$ & Layer ranks (G / DD) & Payload G/DD & Saving & Offline G/DD (s)',freq)
details+=table('Mean complete online times in seconds over five queries and five repetitions. SL is shifted-Laplacian ILU-GMRES. The iteration column lists SL baseline iterations and Global/DD completion iterations; the last column is the worst final verified residual of the ROM workflows.','tab:frequency-cost-details','rrrrrrr',r'$\omega$ & Direct & SL & Global & DD & It. SL/G/DD & Max. $\rho$',costs)
details += r'''
Tables~\ref{tab:frequency-storage-details}--\ref{tab:frequency-cost-details}
supplement Section~\ref{sec:reviewer-supplement} with layer ranks, baseline
costs, and completion counts. The independently trained physical grids are
\(125^2,188^2,251^2,503^2\), with PML thickness approximately 0.25.
Serial timings rotate Global/DD order; baselines use the same thread settings,
and direct costs include fresh factorization. Offline costs include all
reference solves, replay, compression, and reduced assembly.
\FloatBarrier
'''
(TEX/'reviewer_frequency_details.tex').write_text(r"\section{Detailed frequency-study measurements}"+'\n'+r"\label{app:frequency-details}"+'\n'+details)
compact=[]; timing=[]
for r in frequency_results:
    g,d,tg,td=(r[k] for k in ['g','d','tg','td'])
    compact.append([g['omega'],f"{int(num(g,'N')):,}",f"{num(g,'payload_MiB'):.2f}",f"{num(d,'payload_MiB'):.2f}",f"{r['saving']:.1f}\\%"])
    timing.append([g['omega'],f"{num(g,'offline_s'):.2f}/{num(d,'offline_s'):.2f}",f"{tg['time']:.4f}/{td['time']:.4f}"])
text=table(r'Four-frequency comparison. Upper panel: persistent ROM payload in MiB. Lower panel: Global/DD costs; online means include completion over five queries and five repetitions at true tolerance $10^{-10}$, and offline costs are single measurements. Detailed ranks and baseline costs are in Appendix~\ref{app:frequency-details}.','tab:reviewer-frequency','rrrrr',r'$\omega$ & $N_h$ & \shortstack{Global\\storage} & \shortstack{DD\\storage} & \shortstack{Storage\\saving}',compact)
text=text.replace(r'\centering\scriptsize',r'\centering\small')
text=text.replace(r'\end{table}',r'\par\medskip'+'\n'+r'\begin{tabular}{rrr}\toprule'+'\n'+r'$\omega$ & Offline (s) & Online (s) \\'+'\n'+r'\midrule'+'\n'+'\n'.join(' & '.join(map(str,row))+r' \\' for row in timing)+'\n'+r'\bottomrule\end{tabular}\end{table}')
text=text.replace(r'\label{tab:reviewer-frequency}',r'\label{tab:reviewer-frequency}\label{tab:reviewer-frequency-cost}')
r=frequency_results[-1]; g,d,tg,td=(r[k] for k in ['g','d','tg','td'])
text+=rf'''
DD saves \({min(r['saving'] for r in frequency_results):.1f}\%\)--\({max(r['saving'] for r in frequency_results):.1f}\%\)
of persistent storage across the four frequencies. At \(\omega=214\),
the payload falls from \({num(g,'payload_MiB'):.2f}\) to
\({num(d,'payload_MiB'):.2f}\) MiB, while Global/DD complete online means
are \({tg['time']:.4f}/{td['time']:.4f}\) s. The timing gap narrows as the
systems grow: it is about 2.1\% at \(\omega=107\) and below 1\% at
\(\omega=214\), where the difference is within the observed variability.
Thus DD's smaller storage footprint is accompanied by comparable online
cost at the larger sizes, although Global is faster at the two smaller sizes.

Both methods require completion on one of five queries at \(\omega=214\)
(14 Global versus 13 DD iterations); all other frequency/query combinations
are accepted after the reduced steps. The resulting cost increase is
included in the figure and table. All final residuals meet \(10^{{-10}}\),
and both ROM workflows have lower mean online costs than the direct and
shifted-Laplacian baselines on these queries.
'''
(TEX/'reviewer_frequency_tables.tex').write_text('% Generated by scripts/write_reviewer_20260909_evidence.py\n'+text)
if '--frequency-only' in sys.argv:
    print('Generated only the algebraic frequency tables and comparison text.')
    raise SystemExit(0)
rows=[]
for m in ['global','dd']:
    r=lookup['two_parameter',m];t=timing('two_parameter',m)
    rows.append([('DD' if m=='dd' else 'Global'),ranks(r),f"{num(r,'basis_MiB'):.3f}",f"{num(r,'operators_MiB'):.3f}",f"{num(r,'payload_MiB'):.3f}",f"{t['time']:.4f}",f"{t['iterations']:.1f}"])
text=table('Two-material reference with 36 training parameters and nine development queries. Both methods use three common $10^{-5}$ budgets. Persistent payload includes three affine operators per layer and one DD index map.','tab:reviewer-two-parameter','llrrrrr',r'Method & Layer ranks & Basis (MiB) & Operators & Payload & Time (s) & It.',rows)
text+=r'''The \(6\times6\), \(10^{-5}\) reference in
Table~\ref{tab:reviewer-two-parameter} illustrates the reversal: DD saves
\(14.9\%\) in basis entries but uses \(4.9\%\) more persistent storage
because its three reduced affine matrices are larger. Both methods meet
\(10^{-10}\), yet DD provides little acceleration over shifted ILU
(\(0.3689\) versus \(0.3818\) s).
'''
(TEX/'reviewer_two_parameter_reference.tex').write_text('% Generated adverse reference, retained in full-scan appendix\n'+text)
text=''
rows=[]
for m,name in [('global','Global'),('dd','Material/depth'),('merged','Merged PML'),('signed','Signed/depth'),('cartesian','Uniform tiles')]:
    r=lookup['frequency_1',m];t=timing('frequency_1',m)
    save=100*(1-num(r,'payload_MiB')/num(lookup['frequency_1','global'],'payload_MiB'))
    rows.append([name,ranks(r),f"{num(r,'payload_MiB'):.3f}",f"{save:.1f}\\%",f"{t['time']:.4f}",sci(t['pre'])])
text+=table('Partition sensitivity at the reference frequency, using 24 training samples and the same five queries. All methods accept the reduced fields directly. Every prescribed partition is shown.','tab:reviewer-partition','llrrrr',r'Partition & Layer ranks & Payload (MiB) & Saving & Time (s) & Max. $\rho_{\rm ROM}$',rows)
(TEX/'reviewer_parameter_partition_tables.tex').write_text('% Generated from summary_train.csv and *_timing.csv\n'+text)
rows=[]
for m in ['global','dd']:
    for r in read(OUT/'frequency_1'/f'{m}_constants.csv'):
        rows.append([('DD' if m=='dd' else 'Global'),r['j'],sci(num(r,'Khat')),sci(num(r,'Chateta')),sci(num(r,'deltahat')),sci(num(r,'Cbareta')+num(r,'deltabar')),sci(num(r,'defect'))])
text=table('Conditional error-model diagnostics at $(1,0.8761)$. Hats use the power estimate of $K$; bars use the computable norm upper bound. Diagnostic full inverse solves are excluded from timed online work.','tab:reviewer-constants','lrrrrrr',r'Model & $j$ & $\widehat K$ & $\widehat C_j\eta_j$ & $\widehat\delta_j$ & $\overline C_j\eta_j+\overline\delta_j$ & $\|\varepsilon_j\|$',rows)
text+=r'''The estimated truncation terms exceed the observed defects by factors of
20--565; the guaranteed bounds are larger still. These bounds expose
stability and training-coverage limitations but are too conservative to
predict query residuals quantitatively. Acceptance therefore uses the
full-system residual.
'''
(TEX/'reviewer_constant_table.tex').write_text('% Generated from *_constants.csv\n'+text)
# Export complete raw-derived scan, when every prescribed timing is available.
p=OUT/'summary_explore_train.csv'
if p.exists():
    scan=read(p); results=[]
    for r in scan:
        c,m=r['study'],r['variant']
        if not (OUT/c/f'{m}_timing.csv').exists():break
        t=timing(c,m);r=dict(r);r.update({f'timing_{k}':v for k,v in t.items()});results.append(r)
    if len(results)==len(scan):
        with (OUT/'complete_parameter_scan.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
print('Generated frequency, parameter/partition, and conditional-constant tables.')

# Complete development scan: no filtering by storage saving, time, or convergence.
p=OUT/'complete_parameter_scan.csv'
if p.exists():
    allrows=read(p); L={(r['study'],r['variant']):r for r in allrows}
    pairs=[]
    for c,m in L:
        if m!='global':continue
        g,d=L[c,'global'],L[c,'dd']
        if c=='two_parameter':width=.05;n=6;eta='1e-5'
        else:
            match=__import__('re').match(r'explore_w(\d+)_n(\d+)_(.*)',c)
            width=int(match[1])/100;n=int(match[2]);eta=match[3].replace('e-','1e-')
        label='relaxed' if eta=='relaxed' else sci(float(eta))
        save=100*(1-num(d,'payload_MiB')/num(g,'payload_MiB'))
        sl=timing(c,'shifted')
        pairs.append(dict(study=c,width=width,n=n,eta=eta,saving=save,dd_speedup=sl['time']/num(d,'timing_time'),global_time=num(g,'timing_time'),dd_time=num(d,'timing_time'),g=g,d=d,label=label))
    display=[]
    for x in pairs:
        g,d=x['g'],x['d']
        display.append([f"{x['width']:.2f}",rf"\({x['n']}^2\)",x['label'],f"{num(g,'payload_MiB'):.2f}/{num(d,'payload_MiB'):.2f}",f"{x['saving']:.1f}\\%",f"{x['global_time']:.3f}/{x['dd_time']:.3f}",f"{x['dd_speedup']:.2f}",f"{num(d,'timing_reps'):.0f}"])
    appendix=r'''\section{Complete two-material development scan}
\label{app:reviewer-scan}
\suppressfloats[t]
Table~\ref{tab:reviewer-full-scan} and Figure~\ref{fig:reviewer-scan}
retain all 12 development configurations. Here
\(c_{\rm left}\in[1-w,1+w]\), \(c_{\rm right}\in[0.8,0.95]\), and
training uses an \(n\times n\) grid. Thresholds apply to all three layers;
``relaxed'' denotes \((10^{-5},10^{-4},10^{-3})\). The nine development
queries combine normalized left coordinates \((-0.74,-0.06,0.74)\) with
right speeds \((0.813,0.871,0.938)\), separately from final validation.
Times use three warmed-up repetitions, or five for the original reference;
reused baseline measurements are not independent replicates.

DD savings range from \(-5.7\%\) to \(30.0\%\), with four storage
reversals. Thus training density and compression budgets affect whether
local basis savings outweigh the larger reduced operators.
'''
    appendix+=table('Complete two-material development scan. Payload is persistent ROM storage in MiB. Paired entries are Global/DD; negative savings mean larger DD payload. The speedup is baseline shifted-ILU time divided by DD time; values below one favor the baseline. Rep. is the number of recorded repeats per query.','tab:reviewer-full-scan','rrlrrrrr',r'$w$ & $n^2$ & Threshold & Payload G/DD & Saving & Time G/DD (s) & SL/DD & Rep.',display)
    (TEX/'reviewer_complete_scan.tex').write_text('% Generated from complete_parameter_scan.csv\n'+appendix)
    with (OUT/'parameter_scan_pairs.csv').open('w') as f:
        keys=['study','width','n','eta','saving','dd_speedup','global_time','dd_time']
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:x[k] for k in keys} for x in pairs)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(1,2,figsize=(9.2,3.8),constrained_layout=True)
    for ax,metric,title in zip(axes,['saving','dd_speedup'],['Persistent ROM saving (%)','Online speedup over shifted ILU']):
        data=np.array([[next(x[metric] for x in pairs if x['width']==width and x['n']==6 and x['eta']==eta) for width in [.02,.05,.10]] for eta in ['1e-3','1e-4','1e-5']])
        lo,hi=((-10,35) if metric=='saving' else (min(.8,float(data.min())),max(1.2,float(data.max()))))
        im=ax.imshow(data,cmap='RdYlBu',vmin=lo,vmax=hi,aspect='auto')
        for i in range(3):
            for j in range(3):ax.text(j,i,f'{data[i,j]:.1f}' if metric=='saving' else f'{data[i,j]:.2f}',ha='center',va='center',fontsize=12,color='#111111',bbox=dict(facecolor='white',edgecolor='none',alpha=.7,pad=2))
        ax.set_xticks(range(3),['0.02','0.05','0.10']);ax.set_yticks(range(3),['$10^{-3}$','$10^{-4}$','$10^{-5}$'])
        ax.set_xlabel('Left-speed half-width w');ax.set_ylabel('Common layer threshold');ax.set_title(title,fontsize=11)
        fig.colorbar(im,ax=ax,shrink=.8)
    fig.suptitle('Two-material development scan: 6 x 6 training, nine queries',fontsize=12)
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    fig.savefig(folder/'parameter_scan.png',dpi=200);fig.savefig(folder/'parameter_scan.pdf');plt.close(fig)
    # The full scan figure is retained as a supplement, not an independent test.
    target=TEX/'figures/reviewer_20260909';target.mkdir(exist_ok=True)
    __import__('shutil').copy2(folder/'parameter_scan.pdf',target/'parameter_scan.pdf')
    with (TEX/'reviewer_complete_scan.tex').open('a') as f:f.write(r'''
\input{reviewer_two_parameter_reference.tex}
\begin{figure}[htbp]\centering
\includegraphics[width=\linewidth]{figures/reviewer_20260909/parameter_scan.pdf}
\caption{Complete range--threshold cross-section at \(6\times6\) training.
Negative savings and speedups below one are retained. Additional training
sizes and the relaxed schedule are included in Table~\ref{tab:reviewer-full-scan}.}
\label{fig:reviewer-scan}
\end{figure}
\FloatBarrier
''')

# Frozen model, evaluated on the predeclared independent query set.
selection=OUT/'selection_before_holdout.json'
if selection.exists():
    import json
    selected=json.loads(selection.read_text())['selected']
    if all((OUT/selected/f'{m}_holdout.csv').exists() for m in ['global','dd','direct','shifted']):
        L={(r['study'],r['variant']):r for r in read(OUT/'summary_explore_train.csv')}
        g,d=(L[selected,m] for m in ['global','dd'])
        tg,td,dr,sl=(timing(selected,m,'holdout') for m in ['global','dd','direct','shifted'])
        save=100*(1-num(d,'payload_MiB')/num(g,'payload_MiB'))
        speedup=sl['time']/td['time']
        amort=num(d,'offline_s')/(sl['time']-td['time']) if sl['time']>td['time'] else math.inf
        development={(r['study'],r['variant']):r for r in read(OUT/'complete_parameter_scan.csv')}
        small=development['explore_w05_n6_e-3','dd'];large=development[selected,'dd']
        extra_amort=(num(large,'offline_s')-num(small,'offline_s'))/(num(small,'timing_time')-num(large,'timing_time'))
        rows=[]
        for name,r,t in [('Global',g,tg),('DD',d,td),('Direct',None,dr),('Shifted ILU',None,sl)]:
            rows.append([name,ranks(r) if r else '--',f"{num(r,'payload_MiB'):.3f}" if r else '--',f"{num(r,'offline_s'):.2f}" if r else '--',f"{t['time']:.4f}",f"{t['iterations']:.1f}",sci(t['rho'])])
        text=r'''From 12 development configurations (Appendix~\ref{app:reviewer-scan}),
we select an \(8\times8\) training grid with three \(10^{-4}\) layer
budgets. Within the stated parameter range, it gives the fastest DD
development mean among candidates with at least \(10\%\) storage savings
and acceleration over shifted ILU. The model is frozen before testing
nine independent queries fixed before the scan.
'''
        text+=table('Independent two-material validation of the frozen selected configuration. Persistent ROM payload is in MiB; all methods use the same true residual target. Times include completion and setup; offline costs are single measurements. The iteration column counts completion work, or the baseline solve.','tab:reviewer-selected','llrrrrr',r'Method & Layer ranks & Payload & Offline (s) & Online (s) & It. & Max. $\rho$',rows)
        text+=rf'''
Table~\ref{{tab:reviewer-selected}} and
Figure~\ref{{fig:halfspace-two-parameter-results}} show that DD reduces
persistent storage by \({save:.1f}\%\), from \({num(g,'payload_MiB'):.2f}\)
to \({num(d,'payload_MiB'):.2f}\) MiB, while maintaining comparable mean
online time: \({td['time']:.4f}\) s versus \({tg['time']:.4f}\) s for Global.
All nine queries reach \(10^{{-10}}\) after completion. DD is
\({speedup:.2f}\) times faster than shifted ILU, although sparse direct
solution remains faster at this two-dimensional size.

The storage benefit depends on training density and compression budgets:
the original \(6\times6\), \(10^{{-5}}\) configuration increases DD
payload by \(4.9\%\), owing to larger reduced operators. The full scan,
including these adverse cases, is retained in
Appendix~\ref{{app:reviewer-scan}} and Table~\ref{{tab:reviewer-two-parameter}}.
'''
        (TEX/'reviewer_selected_validation.tex').write_text('% Generated from frozen selection and *_holdout.csv\n'+text)
        metrics=dict(selected=selected,saving_percent=save,dd_vs_shifted_speedup=speedup,dd_break_even_vs_shifted=amort,global_metrics=tg,dd_metrics=td,direct_metrics=dr,shifted_metrics=sl)
        metrics['direct_metrics']['pre']=None
        (OUT/'selected_validation_summary.json').write_text(json.dumps(metrics,indent=2,allow_nan=False)+'\n')

# The separate high-frequency WaveHoltz comparison is retired from the
# manuscript. Its source/results are preserved in the dated revision archive.
