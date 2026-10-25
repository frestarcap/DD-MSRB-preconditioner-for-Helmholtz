"""Shared manuscript storage-table format; retain distinct basis/payload/whos scopes."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
HEADER=r'\shortstack{Global\\storage} & \shortstack{DD\\storage} & \shortstack{Storage\\saving}'
def representative_storage():
    raw=json.loads((ROOT/'outputs/waveguide_whos_20260913/results/w48_whos.json').read_text())
    models={r['method']:r for r in raw['rows']}
    lines=[r'\setlength{\belowcaptionskip}{6pt}',
           r'\caption{Waveguide layerwise basis storage at \(\omega=48\), in MiB.}\label{tab:3d-storage}',
           r'\setlength{\tabcolsep}{2.5pt}\renewcommand{\arraystretch}{1.12}',
           r'\begin{tabular}{@{}ccrrr@{}}\toprule',
           r'Layer & \shortstack{Block\\ranks} & '+HEADER+r' \\',r'\midrule']
    for j,(g,d) in enumerate(zip(models['global']['layers'],models['dd']['layers']),1):
        gb,db=g['basis_bytes']/2**20,d['basis_bytes']/2**20
        lines.append(f'{j} & '+rf'\({int(d["rank"])}^{{*}}\)'+f' & {gb:.2f} & {db:.2f} & {100*(1-db/gb):.1f}'+r'\% \\')
    gb=sum(r['basis_bytes'] for r in models['global']['layers'])/2**20
    db=sum(r['basis_bytes'] for r in models['dd']['layers'])/2**20
    lines += [f'Total & -- & {gb:.2f} & {db:.2f} & {100*(1-db/gb):.1f}'+r'\% \\',
              r'\bottomrule\end{tabular}',r'\par\smallskip\small\raggedright',
              r'\({}^{*}\)Sum of the 37 local block ranks; global ranks are \((7,6,5)\).',
              r'Global/DD payload: \(1522.43/1084.29\) MiB (28.8\% saving); MATLAB object size: \(1522.44/1168.98\) MiB (23.2\%).',
              r'Payload counts one shared index map; the measured MATLAB object retains per-layer indices. Full-order and MG data are excluded.','']
    return '\n'.join(lines)
def waveguide_frequency(rows):
    lines=[r'\begin{table}[t]\centering\small\setlength{\tabcolsep}{4pt}',
           r'\caption{Waveguide frequency study. (a) ROM storage in MiB: payload counts one shared index map, while MATLAB R2024a objects retain per-layer indices. Both exclude full-order and completion data; object size is logical storage, not RSS. (b) Global/DD costs; online means cover three queries timed once. Recovered offline timing at \(\omega=60\) is omitted.}',
           r'\label{tab:3d-frequency}',r'\begin{tabular}{rlrrr}\toprule',
           r'\multicolumn{5}{l}{(a) ROM storage (MiB)} \\',
           r'$\omega$ & Metric & '+HEADER+r' \\',r'\midrule']
    for row in rows:
        for metric,suffix in [('Payload','_MiB'),(r'MATLAB object','_whos_MiB')]:
            g,d=row['global'+suffix],row['dd'+suffix]
            lines.append(f'{row["omega"]} & {metric} & {g:.2f} & {d:.2f} & {100*(1-d/g):.1f}'+r'\% \\')
    lines += [r'\bottomrule\end{tabular}',r'\par\medskip',
              r'\begin{tabular}{rccc}\toprule',r'\multicolumn{4}{l}{(b) Solver costs} \\',
              r'$\omega$ & Offline (s) & Mean online (s) & Shifted MG mean online (s) \\',r'\midrule']
    for r in rows:
        offline='---' if r['global_offline_s'] is None else f"{r['global_offline_s']:.1f}/{r['dd_offline_s']:.1f}"
        lines.append(f"{r['omega']} & {offline} & {r['global_online_s']:.2f}/{r['dd_online_s']:.2f} & {r['shifted_online_s']:.2f}"+r' \\')
    lines += [r'\bottomrule\end{tabular}\end{table}','']
    return '\n'.join(lines)
