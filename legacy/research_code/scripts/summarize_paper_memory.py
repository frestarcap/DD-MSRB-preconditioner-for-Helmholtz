"""Publish measured object size and RSS, keeping negative savings visible."""
import csv
from pathlib import Path
from statistics import median
root=Path(__file__).resolve().parents[1]
out=root/'outputs/paper_memory_benchmark'
with (out/'process_memory.csv').open() as stream:
    rows=list(csv.DictReader(stream))
assert len(rows)==18, 'Require all three repetitions of both methods in all cases.'
summary=[]
labels={'halfspace':'Half-space','ecc':'ECC','waveguide':'3D waveguide'}
tex=[r'\begin{table}[t]',r'\centering',
     r'\caption{Measured deployed ROM object size and process peak RSS (MiB). Ranges cover three fresh processes.}',
     r'\label{tab:measured-memory}',r'\scriptsize',r'\setlength{\tabcolsep}{3pt}',
     r'\begin{tabular}{llrrrrr}',r'\toprule',
     r'Problem & Method & ROM & ROM saving & RSS median & RSS range & RSS saving \\',r'\midrule']
for problem in labels:
    pair={}
    for method in ('global','dd'):
        group=[r for r in rows if r['problem']==problem and r['method']==method]
        assert len(group)==3
        rss=[float(r['peak_RSS_MiB']) for r in group]
        obj=[float(r['model_MiB']) for r in group]
        assert len(set(obj))==1
        pair[method]=dict(problem=problem,method=method,model_MiB=obj[0],
                         median_RSS_MiB=median(rss),min_RSS_MiB=min(rss),max_RSS_MiB=max(rss))
    saving_obj=100*(1-pair['dd']['model_MiB']/pair['global']['model_MiB'])
    saving_rss=100*(1-pair['dd']['median_RSS_MiB']/pair['global']['median_RSS_MiB'])
    for method in ('global','dd'):
        r=pair[method]; r['object_saving_percent']=0 if method=='global' else saving_obj
        r['RSS_saving_percent']=0 if method=='global' else saving_rss
        summary.append(r)
        so='--' if method=='global' else f'{saving_obj:.1f}\\%'
        sr='--' if method=='global' else f'{saving_rss:.2f}\\%'
        tex.append(f"{labels[problem] if method=='global' else ''} & {'Global' if method=='global' else 'DD'} & "
                   f"{r['model_MiB']:.3f} & {so} & {r['median_RSS_MiB']:.1f} & "
                   f"{r['min_RSS_MiB']:.1f}--{r['max_RSS_MiB']:.1f} & {sr} \\\\")
tex += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
(root/'LaTeX_DL_468198_240419/measured_memory_table.tex').write_text('\n'.join(tex)+'\n')
with (out/'summary.csv').open('w') as stream:
    writer=csv.DictWriter(stream,fieldnames=summary[0]);writer.writeheader();writer.writerows(summary)
for r in summary: print(r)
