#!/usr/bin/env python3
"""Draw the three reviewer-response claims from archived numerical evidence.

No solves, model selection, or manuscript edits are performed. PNGs are for
inspection; SVGs retain vector graphics. Raw input hashes accompany the plots.
"""
from pathlib import Path
import csv
import hashlib
import json
import statistics as stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'outputs/reviewer_20260909'
OUT = DATA / 'figures/claim_visuals_20260909'
OUT.mkdir(parents=True, exist_ok=True)
INPUTS = set()
BLUE, ORANGE, GRAY = '#2864AD', '#D95B2C', '#637383'
GREEN, RED = '#27867B', '#B84A48'
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10.5,
    'axes.titlesize': 11.5, 'axes.titleweight': 'bold',
    'axes.labelsize': 10.5, 'legend.fontsize': 9,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.axisbelow': True, 'svg.fonttype': 'none',
})


def read(path):
    INPUTS.add(path)
    with path.open() as f:
        return list(csv.DictReader(f))


def value(row, key):
    return float(row[key])


def finish(fig, name, title, subtitle, note):
    fig.suptitle(title, x=.5, y=.987, fontsize=17, fontweight='bold')
    fig.text(.5, .947, subtitle, ha='center', va='top', fontsize=10.5, color='#445160')
    fig.text(.5, .014, note, ha='center', va='bottom', fontsize=9, color='#445160')
    fig.canvas.draw()
    fig.savefig(OUT / f'{name}.png', dpi=210, facecolor='white')
    fig.savefig(OUT / f'{name}.svg', facecolor='white')
    plt.close(fig)


def panel(ax, title, log=False):
    ax.set_title(title, loc='left', pad=12, y=1.01)
    if log:
        ax.set_yscale('log')
    ax.grid(axis='y', alpha=.20, linewidth=.7)


def frequency_figure():
    high = DATA / 'high_frequency_waveholtz'
    summary = read(high / 'summary_paper.csv')
    lookup = {(r['scale'], r['method']): r for r in summary}
    assert len(lookup) == 6
    scales = ['1', '2', '4']
    omega = np.array([53.5, 107, 214])
    baseline = [read(high / f'global_scale{s}_queries.csv') for s in scales]
    prefixes = read(high / 'prefix_scale4.csv')
    assert len(prefixes) == 2
    for rows in baseline:
        assert len(rows) == 3
        assert all(value(r, 'baseline_true_residual') <= 1e-8 for r in rows)
    assert all(value(r, 'true_residual') <= 1e-8 for r in prefixes)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8.1))
    fig.subplots_adjust(left=.09, right=.965, bottom=.135, top=.85, hspace=.50, wspace=.29)
    ax = axes[0, 0]
    panel(ax, '(a) Persistent ROM storage', log=True)
    for method, color, marker, label in [('global', BLUE, 'o', 'Global'), ('dd', ORANGE, 's', 'DD')]:
        y = [value(lookup[s, method], 'payload_MiB') for s in scales]
        ax.plot(omega, y, marker=marker, color=color, lw=2, ms=6, label=label)
        ax.annotate(f'{y[-1]:.2f} MiB', (omega[-1], y[-1]), xytext=(-8, 8 if method=='global' else -19),
                    textcoords='offset points', ha='right', color=color, fontsize=10)
    saving = 100 * (1 - value(lookup['4', 'dd'], 'payload_MiB') / value(lookup['4', 'global'], 'payload_MiB'))
    ax.text(.04, .93, f'{saving:.1f}% less at $\\omega=214$', transform=ax.transAxes,
            va='top', color=ORANGE, fontweight='bold', fontsize=12)
    ax.set_ylim(9, 850)
    ax.set_yticks([10, 30, 100, 300, 600])
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.set_ylabel('Persistent payload (MiB; log scale)')
    ax.legend(loc='lower right')

    ax = axes[0, 1]
    panel(ax, '(b) Verified final Helmholtz residual', log=True)
    for method, color, marker, label in [('global', BLUE, 'o', 'Global: 3 ROM steps'), ('dd', ORANGE, 's', 'DD: 3 ROM steps')]:
        y = [value(lookup[s, method], 'max_true_residual') for s in scales]
        assert max(y) <= 1e-8
        ax.plot(omega, y, marker=marker, color=color, lw=1.8, ms=5, label=label)
    ax.plot(omega, [max(value(r, 'baseline_true_residual') for r in rows) for rows in baseline],
            marker='^', color=GRAY, lw=1.8, ms=5, label='WaveHoltz baseline')
    ax.axhline(1e-8, color='#303030', ls='--', lw=1)
    ax.text(58, 1.45e-8, 'True residual target: $10^{-8}$', fontsize=9.5)
    ax.set_ylim(1e-14, 1e-6)
    ax.set_ylabel('Worst residual over 3 queries')
    ax.legend(loc='upper left', bbox_to_anchor=(0, .78), framealpha=.95)

    ax = axes[1, 0]
    panel(ax, '(c) WaveHoltz convergence work')
    queries = [value(r, 'parameter') for r in baseline[0]]
    for mu, style, marker, color in zip(queries, ['-', '--', ':'], ['o', 's', '^'], ['#384C63', '#7D6AA5', '#25968B']):
        y = [next(value(r, 'baseline_iterations') for r in rows if value(r, 'parameter') == mu) for rows in baseline]
        ax.plot(omega, y, marker=marker, color=color, ls=style, lw=1.8, ms=5, label=rf'$c_{{right}}={mu}$')
        ax.annotate(f'{y[-1]:.0f}', (214, y[-1]), xytext=(5, -2), textcoords='offset points', fontsize=9)
    ax.axhline(800, color='#303030', ls='--', lw=1)
    ax.text(58, 811, 'Native iteration limit: 800', fontsize=9)
    ax.set_ylim(0, 875)
    ax.set_ylabel('Native GMRES iterations')
    ax.legend(loc='upper left', bbox_to_anchor=(0, .87), fontsize=8.5)

    ax = axes[1, 1]
    panel(ax, '(d) Frozen first-layer completion at 214')
    base = next(r for r in baseline[-1] if abs(value(r, 'parameter')-.8761) < 1e-12)
    p = {r['method']: r for r in prefixes}
    vals = [value(base, 'baseline_iterations'), value(p['global'], 'completion_iterations'), value(p['dd'], 'completion_iterations')]
    ax.bar([0, 1, 2], vals, width=.60, color=[GRAY, BLUE, ORANGE])
    for x, n in enumerate(vals):
        ax.text(x, n+15, f'{n:.0f}', ha='center', fontweight='bold')
    ax.set_xticks([0, 1, 2], ['WaveHoltz\nfrom zero', 'Global layer 1\n+ WaveHoltz', 'DD layer 1\n+ WaveHoltz'], fontsize=9)
    ax.set_ylim(0, 750)
    ax.set_ylabel('Native GMRES iterations')
    ax.text(.5, .96, '$c_{right}=0.8761$; all final true residuals $<3\\times10^{-11}$',
            ha='center', va='top', transform=ax.transAxes, fontsize=9)
    for ax in [axes[0, 0], axes[0, 1], axes[1, 0]]:
        ax.set_xlim(45, 232)
        ax.set_xticks(omega, ['53.5', '107', '214'])
        ax.set_xlabel(r'Angular frequency $\omega$')
    finish(fig, 'high_frequency_claim',
           'High-frequency reuse remains effective through $\\omega=214$',
           '35,721 / 142,129 / 570,025 unknowns | fixed 11.84 PPW | retrained at each frequency',
           '24 training speeds, 3 queries at each frequency. Panel (d) freezes layer 1 of the existing model.\n'
           'All fields are verified using true Helmholtz residuals; native iteration counts increase with frequency.')


def validation_figure():
    selection_path = DATA / 'selection_before_holdout.json'
    INPUTS.add(selection_path)
    selection = json.loads(selection_path.read_text())
    name = selection['selected']
    data = {m: read(DATA / name / f'{m}_holdout.csv') for m in ['global', 'dd', 'direct', 'shifted']}
    summary = read(DATA / 'summary_explore_train.csv')
    lookup = {r['variant']: r for r in summary if r['study'] == name}
    queries = list(dict.fromkeys((value(r, 'c_left'), value(r, 'c_right')) for r in data['dd']))
    assert len(queries) == 9
    assert set(queries) == {(l, r) for l in selection['holdout_left'] for r in selection['holdout_right']}
    grouped = {}
    for method, rows in data.items():
        assert len(rows) == 45 and all(value(r, 'true_residual') <= 1e-10 for r in rows)
        grouped[method] = [[r for r in rows if (value(r, 'c_left'), value(r, 'c_right')) == q] for q in queries]
        assert all(len(g) == 5 for g in grouped[method])
    with (OUT / 'validation_query_key.csv').open('w') as f:
        w = csv.writer(f); w.writerow(['query_id', 'c_left', 'c_right'])
        w.writerows((i+1, *q) for i, q in enumerate(queries))

    fig, axes = plt.subplots(2, 2, figsize=(10, 8.1))
    fig.subplots_adjust(left=.09, right=.97, bottom=.14, top=.85, hspace=.49, wspace=.29)
    ax = axes[0, 0]
    panel(ax, '(a) Persistent ROM storage')
    payloads = [value(lookup[m], 'payload_MiB') for m in ['global', 'dd']]
    ax.bar([0, 1], payloads, color=[BLUE, ORANGE], width=.55)
    for x, n in enumerate(payloads):
        ax.text(x, n+1.5, f'{n:.2f}', ha='center', fontweight='bold')
    saving = 100*(1-payloads[1]/payloads[0])
    ax.text(.95, .93, f'{saving:.1f}% less', transform=ax.transAxes, ha='right', va='top',
            color=ORANGE, fontsize=13, fontweight='bold')
    ax.set_ylim(0, 82)
    ax.set_xticks([0, 1], ['Global', 'DD'])
    ax.set_ylabel('Persistent payload (MiB)')

    ax = axes[0, 1]
    panel(ax, '(b) Complete online solve time')
    methods = ['direct', 'shifted', 'global', 'dd']
    colors = ['#A5ADB4', GRAY, BLUE, ORANGE]
    means = [stats.mean(value(r, 'total_s') for r in data[m]) for m in methods]
    ax.bar(range(4), means, color=colors, width=.63, alpha=.87)
    for i, m in enumerate(methods):
        points = [stats.mean(value(r, 'total_s') for r in group) for group in grouped[m]]
        ax.scatter(i + np.linspace(-.20, .20, 9), points, s=15, color='#233344', alpha=.7, zorder=3)
        ax.text(i, means[i]+.025, f'{means[i]:.4f}', ha='center', fontweight='bold', fontsize=9.5,
                bbox=dict(facecolor='white', alpha=.85, edgecolor='none', pad=1))
    speedup = means[1]/means[3]
    ax.text(.5, .95, f'Shifted ILU / DD = {speedup:.2f}x', transform=ax.transAxes,
            ha='center', va='top', color=ORANGE, fontsize=12, fontweight='bold')
    ax.set_ylim(0, .72)
    ax.set_xticks(range(4), ['Direct', 'Shifted ILU', 'Global', 'DD'], fontsize=9)
    ax.set_ylabel('Mean time (s); dots = query means')

    ax = axes[1, 0]
    panel(ax, '(c) Verified residuals at all 9 queries', log=True)
    x = np.arange(1, 10)
    for method, color, marker, label in [('global', BLUE, 'o', 'Global'), ('dd', ORANGE, 's', 'DD')]:
        pre = [max(value(r, 'post_ROM_residual') for r in group) for group in grouped[method]]
        final = [max(value(r, 'true_residual') for r in group) for group in grouped[method]]
        ax.plot(x, pre, color=color, marker=marker, mfc='white', ls='--', ms=5, lw=1.3, label=f'{label}: after ROM')
        ax.plot(x, final, color=color, marker=marker, ms=5, lw=1.6, label=f'{label}: final')
    ax.axhline(1e-10, color='#333333', ls=':', lw=1.4)
    ax.text(1.1, 1.7e-10, 'Final target: $10^{-10}$', fontsize=9)
    ax.set_ylim(1e-11, 2e-4)
    ax.set_ylabel('True relative Helmholtz residual')
    ax.legend(loc='center', bbox_to_anchor=(.5, .38), fontsize=8.5, ncol=2, columnspacing=.8, framealpha=.95)

    ax = axes[1, 1]
    panel(ax, '(d) Krylov work at the same queries')
    for method, color, marker, label in [('shifted', GRAY, '^', 'Shifted ILU baseline'), ('global', BLUE, 'o', 'Global completion'), ('dd', ORANGE, 's', 'DD completion')]:
        y = [stats.mean(value(r, 'completion_iterations') for r in group) for group in grouped[method]]
        ax.plot(x, y, color=color, marker=marker, ms=5, lw=1.7, label=label)
    ax.set_ylim(0, 34)
    ax.set_ylabel('GMRES iterations')
    ax.legend(loc='upper center', bbox_to_anchor=(.5, .78), fontsize=8.5)
    for ax in axes[1]:
        ax.set_xticks(x)
        ax.set_xlabel('Independent validation query ID')
        ax.set_xlim(.6, 9.4)
    finish(fig, 'independent_validation_claim',
           'Independent two-material validation',
           '$8\\times8$ training grid | common layer threshold $10^{-4}$ | 9 independent queries, 5 repeats each',
           'Online times include ROM work, residual checks, preconditioner setup and completion; training is excluded.\n'
           'DD offline construction: 14.32 s. Sparse direct remains faster at this 2D size. Query coordinates accompany the figure.')
    return saving, speedup


def complete_scan_figure():
    rows = read(DATA / 'parameter_scan_pairs.csv')
    assert len(rows) == 12
    negative = sum(value(r, 'saving') < 0 for r in rows)
    assert negative == 4
    selected_name = json.loads((DATA / 'selection_before_holdout.json').read_text())['selected']
    selected = next(i for i, r in enumerate(rows) if r['study'] == selected_name)
    labels = []
    for i, r in enumerate(rows):
        eta = 'relaxed' if r['eta'] == 'relaxed' else rf'$10^{{{int(np.log10(value(r, "eta")))}}}$'
        labels.append(f'{i+1:02d}   {value(r,"width"):.2f}   {r["n"]}x{r["n"]}   {eta}')
    y = np.arange(12)
    saving = np.array([value(r, 'saving') for r in rows])
    speedup = np.array([value(r, 'dd_speedup') for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(11, 7.4), sharey=True)
    fig.subplots_adjust(left=.245, right=.945, bottom=.13, top=.84, wspace=.23)
    for ax in axes:
        ax.axhspan(selected-.46, selected+.46, color='#FFF0D6', zorder=0)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', length=0)
        ax.grid(axis='x', alpha=.18)
        ax.set_ylim(11.6, -.7)
    ax = axes[0]
    ax.set_title('(a) DD persistent storage saving', loc='left', pad=14)
    colors = [RED if z < 0 else GREEN for z in saving]
    colors[selected] = ORANGE
    ax.barh(y, saving, color=colors, height=.58)
    ax.axvline(0, color='#333333', lw=1)
    for i, v in enumerate(saving):
        ax.text(v + (.65 if v>=0 else -.55), i, f'{v:+.1f}%', va='center',
                ha='left' if v>=0 else 'right', color=colors[i], fontweight='bold' if i==selected else 'normal')
    ax.set_xlim(-15, 37)
    ax.set_xlabel('100 x (1 - DD payload / Global payload)')
    ax.set_yticks(y, labels, fontsize=10)
    ax.text(-.04, 1.045, 'Case: w / grid / threshold', transform=ax.transAxes,
            fontsize=9.5, color='#445160', ha='right')
    ax = axes[1]
    ax.set_title('(b) DD speedup over shifted ILU', loc='left', pad=14)
    ax.barh(y, speedup, color=[ORANGE if i==selected else '#8194A7' for i in y], height=.58)
    ax.axvline(1, color='#333333', ls='--', lw=1.2)
    for i, v in enumerate(speedup):
        ax.text(v+.03, i, f'{v:.2f}x', va='center', color=ORANGE if i==selected else '#314457',
                fontweight='bold' if i==selected else 'normal')
    ax.set_xlim(0, 2.17)
    ax.set_xlabel('Shifted ILU time / DD time')
    ax.tick_params(axis='y', labelleft=False)
    finish(fig, 'complete_parameter_scan_claim',
           'All 12 development configurations are retained',
           f'{negative} cases increase DD payload (red) | highlighted case {selected+1:02d} was selected before independent validation',
           'Nine development queries per case: 3 timing repeats for new cases; case 01 retains 5.\n'
           'Relaxed thresholds = ($10^{-5}$, $10^{-4}$, $10^{-3}$). These development speedups are separate from the 1.74x independent result.')


if __name__ == '__main__':
    frequency_figure()
    saving, speedup = validation_figure()
    complete_scan_figure()
    INPUTS.add(Path(__file__).resolve())
    manifest = {}
    for path in sorted(INPUTS):
        with path.open('rb') as f:
            manifest[str(path.relative_to(ROOT))] = hashlib.file_digest(f, 'sha256').hexdigest()
    (OUT / 'figure_provenance.json').write_text(json.dumps(dict(
        inputs_sha256=manifest, independent_DD_saving_percent=saving,
        independent_shifted_to_DD_speedup=speedup,
        new_numerical_solves=False, manuscript_edited=False), indent=2)+'\n')
    print(f'Three PNG/SVG figure pairs generated in {OUT}')
