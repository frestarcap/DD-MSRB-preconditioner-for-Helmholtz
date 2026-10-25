#!/usr/bin/env python3
"""Generate the two active JCP reviewer tables; preserve JSC independently."""
from pathlib import Path
import argparse
import csv
import json
import statistics

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'outputs/reviewer_20260909'
TEMPLATES = ROOT / 'scripts/templates/jcp'


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def sci(value):
    mantissa, exponent = f'{value:.2e}'.split('e')
    return rf'\({mantissa}\times10^{{{int(exponent)}}}\)'


def ranks(row):
    return row['ranks'].replace('[', r'\((').replace(']', r')\)').replace(' ', ',')


def timing(case, method, suffix='timing'):
    rows = read(DATA / case / f'{method}_{suffix}.csv')
    assert all(float(r['true_residual']) <= 1e-10 for r in rows)
    groups = {}
    for r in rows:
        groups.setdefault((r['c_left'], r['c_right']), []).append(r)
    assert len({len(g) for g in groups.values()}) == 1
    if suffix == 'holdout':
        assert len(groups) == 9 and all(len(g) == 5 for g in groups.values())
    return dict(time=statistics.mean(float(r['total_s']) for r in rows),
                iterations=statistics.mean(float(r['completion_iterations']) for r in rows),
                rho=max(float(r['true_residual']) for r in rows),
                pre=max(float(r['post_ROM_residual']) for r in rows))


def render(name, rows, values=None):
    text = (TEMPLATES / (name + '.in')).read_text()
    text = text.replace('@@ROWS@@', '\n'.join(' & '.join(row) + r' \\' for row in rows))
    for key, value in (values or {}).items():
        text = text.replace('@@' + key + '@@', value)
    assert '@@' not in text
    return text


def generate():
    train = {(r['study'], r['variant']): r for r in read(DATA / 'summary_train.csv')}
    rows = []
    for method, label in [('global', 'Global'), ('dd', 'Material/depth'),
                          ('merged', 'Merged PML'), ('signed', 'Signed/depth'),
                          ('cartesian', 'Uniform tiles')]:
        r = train['frequency_1', method]; t = timing('frequency_1', method)
        saving = 100 * (1 - float(r['payload_MiB']) / float(train['frequency_1', 'global']['payload_MiB']))
        rows.append([label, ranks(r), f"{float(r['payload_MiB']):.3f}",
                     f'{saving:.1f}'+r'\%', f"{t['time']:.4f}", sci(t['pre'])])
    outputs = {'reviewer_parameter_partition_tables.tex': render('reviewer_parameter_partition_tables.tex', rows)}
    selected = json.loads((DATA / 'selection_before_holdout.json').read_text())['selected']
    train = {(r['study'], r['variant']): r for r in read(DATA / 'summary_explore_train.csv')}
    g, d = (train[selected, method] for method in ['global', 'dd'])
    times = {method: timing(selected, method, 'holdout') for method in ['global', 'dd', 'direct', 'shifted']}
    rows = []
    for method, label, r in [('global', 'Global', g), ('dd', 'DD', d),
                              ('direct', 'Direct', None), ('shifted', 'Shifted ILU', None)]:
        t = times[method]
        rows.append([label, ranks(r) if r else '--', f"{float(r['payload_MiB']):.3f}" if r else '--',
                     f"{float(r['offline_s']):.2f}" if r else '--', f"{t['time']:.4f}",
                     f"{t['iterations']:.1f}", sci(t['rho'])])
    values = dict(SAVING=f"{100*(1-float(d['payload_MiB'])/float(g['payload_MiB'])):.1f}",
                  GLOBAL_PAYLOAD=f"{float(g['payload_MiB']):.2f}", DD_PAYLOAD=f"{float(d['payload_MiB']):.2f}",
                  DD_TIME=f"{times['dd']['time']:.4f}", GLOBAL_TIME=f"{times['global']['time']:.4f}",
                  SPEEDUP=f"{times['shifted']['time']/times['dd']['time']:.2f}")
    outputs['reviewer_selected_validation.tex'] = render('reviewer_selected_validation.tex', rows, values)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Compare without writing manuscript files.')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'JCP')
    args = parser.parse_args()
    for name, text in generate().items():
        target = args.output_dir / name
        if args.check:
            assert target.read_text() == text, f'Generator differs: {target}'
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text)
        print(f'PASS: {name}')


if __name__ == '__main__':
    main()
