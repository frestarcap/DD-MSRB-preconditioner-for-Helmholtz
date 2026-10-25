#!/usr/bin/env python3
"""Write JCP frequency tables and extract paired diagnostics from frozen evidence."""
from pathlib import Path
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'outputs/waveguide_frequency_final_20260913'
OUT = ROOT / 'outputs/jcp_dd_presentation_20260914'


def main(table_target=None, table_only=False):
    rows = list(csv.DictReader((DATA / 'waveguide_frequency.csv').open()))
    diagnostic = json.loads((DATA / 'layer_diagnostics.json').read_text())
    assert [int(r['omega']) for r in rows] == [24, 36, 48, 60]
    lines = [r'% Generated from the frozen waveguide_frequency.csv cohort.',
             r'\begin{table}[!htbp]\centering\normalsize',
             r'\setlength{\belowcaptionskip}{6pt}',
             r'\caption{Waveguide frequency study: (a) calculated ROM payload; '
             r'(b) measured MATLAB object size, both in MiB.}',
             r'\label{tab:3d-frequency}\par\medskip']
    for i, (title, global_key, dd_key) in enumerate([
        ('(a) Calculated payload (MiB)', 'global_MiB', 'dd_MiB'),
        (r'(b) Measured MATLAB object (MiB)', 'global_whos_MiB', 'dd_whos_MiB')]):
        lines += [r'\begin{minipage}[t]{0.485\linewidth}\centering\vspace{0pt}',
                  title + r'\par\smallskip',
                  r'\setlength{\tabcolsep}{2pt}',
                  r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}rrrr@{}}\toprule',
                  r'$\omega$ & Global & DD & Saving \\', r'\midrule']
        for r in rows:
            g, d = float(r[global_key]), float(r[dd_key])
            lines.append(f"{r['omega']} & {g:.2f} & {d:.2f} & {100*(1-d/g):.1f}"
                         + r'\% \\')
        lines += [r'\bottomrule\end{tabular*}', r'\end{minipage}' + (r'\hfill' if i == 0 else '')]
    lines += [r'\end{table}', '', r'\begin{table}[!htbp]\centering\normalsize',
              r'\caption{Waveguide frequency study: offline construction and mean total online times in seconds.}',
              r'\label{tab:3d-frequency-cost}\par\smallskip',
              r'\setlength{\tabcolsep}{3pt}',
              r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}rrrrrr@{}}\toprule',
              r'$\omega$ & \shortstack{DD\\offline (s)} & \shortstack{Global\\offline (s)} '
              r'& \shortstack{DD\\online (s)} & \shortstack{Global\\online (s)} '
              r'& \shortstack{Shifted MG\\online (s)} \\', r'\midrule']
    for r in rows:
        vals = [f"{float(r[k]):.{digits}f}" if r[k] else '---'
                for k, digits in [('dd_offline_s', 1), ('global_offline_s', 1),
                                  ('dd_online_s', 2), ('global_online_s', 2), ('shifted_online_s', 2)]]
        lines.append(' & '.join([r['omega']] + vals) + r' \\')
    lines += [r'\bottomrule\end{tabular*}', r'\end{table}', '']
    (Path(table_target) if table_target is not None else ROOT / 'JCP/jcp_waveguide_frequency.tex').write_text('\n'.join(lines))

    if table_only:
        return
    OUT.mkdir(exist_ok=True)
    evidence = []
    for r in rows:
        omega = int(r['omega'])
        queries = [d for d in diagnostic if d['omega'] == omega]
        assert len(queries) == 3
        evidence.append(dict(omega=omega, payload_saving_pct=float(r['saving_pct']),
                             whos_saving_pct=float(r['whos_saving_pct']),
                             online_saving_pct=100*(1-float(r['dd_online_s'])/float(r['global_online_s'])),
                             queries=[dict(mu=q['mu'],
                                           global_post_rom=q['global_rom_history'][-1],
                                           dd_post_rom=q['dd_rom_history'][-1],
                                           global_completion=q['global_completion_iterations'],
                                           dd_completion=q['dd_completion_iterations'],
                                           global_third_factor=q['global_third_layer_factor'],
                                           dd_third_factor=q['dd_third_layer_factor']) for q in queries]))
    (OUT / 'paired_dd_evidence.json').write_text(json.dumps(evidence, indent=2)+'\n')
    (OUT / 'evidence_sha256.json').write_text(json.dumps({str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                                       for p in [DATA/'waveguide_frequency.csv', DATA/'layer_diagnostics.json']}, indent=2)+'\n')
    print('JCP_WAVEGUIDE_TABLE_PASS')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--table-only', action='store_true')
    args = parser.parse_args()
    main(table_only=args.table_only)
