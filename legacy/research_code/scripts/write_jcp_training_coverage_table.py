#!/usr/bin/env python3
"""Generate JCP's training-coverage table from the saved nine-query study."""
import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'outputs/revision_20260907/halfspace/summary_timing.csv'
TARGET = ROOT / 'JCP/jsc_halfspace_table.tex'


def generate():
    with SOURCE.open() as stream:
        data = {row['candidate']: row for row in csv.DictReader(stream)}
    lines = [
        r'\begin{table}[t]\centering\normalsize',
        r'\setlength{\belowcaptionskip}{6pt}',
        r'\caption{Training coverage: global/DD over nine fixed queries.}\label{tab:jsc-halfspace-reuse}',
        r'\setlength{\tabcolsep}{3pt}',
        r'\begin{tabular}{rlrrrrr}\toprule',
        r'$n_s$ & Method & Payload (MiB) & Savings & Offline (s) & Time (s) & Max. $\rho_{\rm ROM}$ \\',
        r'\midrule',
    ]
    for count in (6, 12, 24):
        global_payload = float(data[f'global_n{count}_eta-5']['payload_MiB'])
        for method, label in [('global', 'Global MSRB'), ('dd', 'DD-MSRB')]:
            row = data[f'{method}_n{count}_eta-5']
            assert int(row['training_count']) == count
            assert int(row['repetitions']) == 5
            assert float(row['true_residual_max']) <= 1e-12
            saving = 100 * (1 - float(row['payload_MiB']) / global_payload)
            mantissa, exponent = f"{float(row['precompletion_residual_max']):.2e}".split('e')
            cells = [str(count), label, f"{float(row['payload_MiB']):.3f}",
                     rf'\({saving:.1f}\%\)',
                     f"{float(row['offline_s']):.2f}", f"{float(row['online_mean_s']):.4f}",
                     rf'\({mantissa}\times10^{{{int(exponent)}}}\)']
            lines.append(' & '.join(cells) + r' \\')
    lines.append(r'\bottomrule\end{tabular}\end{table}')
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Verify without writing.')
    args = parser.parse_args()
    text = generate()
    if args.check:
        assert TARGET.read_text() == text, f'Generator differs: {TARGET}'
        print('Training-coverage table matches the saved CSV.')
    else:
        TARGET.write_text(text)
