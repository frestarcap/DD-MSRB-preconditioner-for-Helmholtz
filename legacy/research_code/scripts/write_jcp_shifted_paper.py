#!/usr/bin/env python3
"""Publish verified frequency-domain tables and plots to the JCP project only."""
from pathlib import Path
import csv
import os
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'outputs/jcp_shifted_20261022'  # SS-ROM reuses its space in every ROM step
FIELDS = ROOT / 'outputs/jcp_frequency_revision_20260914'
PAPER = ROOT / 'JCP_CAS'  # active manuscript; tables are inline, so the table goes to DATA
FIG = PAPER / 'figures/jcp_shifted'
FIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(DATA / 'matplotlib_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.patches import Rectangle, Ellipse


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


SUMMARY = {(r['problem'], r['method']): r for r in read(DATA / 'summary.csv')}
HIST = read(DATA / 'paper_histories.csv')
PLAIN = read(DATA / 'paper_plain_gmres.csv')
LABELS = {'global': 'Global MSRB', 'dd': 'DD-MSRB', 'nr': 'NT-ROM',
          'fixed_pod': 'SS-ROM', 'shifted': 'Shifted ILU-GMRES', 'direct': 'Sparse direct'}
COLORS = {'global': '#d90d0d', 'dd': '#0d7340', 'nr': '#2659bf',
          'fixed_pod': '#d9730d', 'shifted': '#262626'}
MARKERS = {'global': 'o', 'dd': 's', 'nr': 'D', 'fixed_pod': '^', 'shifted': 'v'}


def write_halfspace_table(target=None):
    half = [f'% Generated from {DATA.relative_to(ROOT)}/summary.csv.',
            r'\centering\normalsize\setlength{\tabcolsep}{2.5pt}',
            r'\setlength{\belowcaptionskip}{6pt}',
            r'\caption{Half-space online cost (s) over the 23 test queries. It.\ is the mean number of SL steps per query (GMRES iterations for SL); Speedup is the median time of SL divided by that of each method.}',
            r'\label{tab:helm-cost}',
            r'\begin{tabular}{@{}lrrrr@{}}\toprule',
            r'Method & Mean & Median & It. & Speedup \\', r'\midrule']
    names = dict(LABELS, shifted='SL', direct='Direct')
    for method in ['global', 'dd', 'nr', 'fixed_pod', 'shifted', 'direct']:
        s = SUMMARY['halfspace', method]
        iters = f"{float(s['mean_completion_iterations']):.2f}" if method != 'direct' else '--'
        speedup = float(SUMMARY['halfspace', 'shifted']['median_total_s']) / float(s['median_total_s'])
        half.append(f"{names[method]} & {float(s['mean_total_s']):.4f} & "
                    f"{float(s['median_total_s']):.4f} & {iters} & "
                    f"${speedup:.1f}" + r'\times$ \\')
    half += [r'\bottomrule\end{tabular}', '']
    (Path(target) if target is not None else DATA / 'jcp_halfspace_timing.tex').write_text('\n'.join(half))


def write_tables():
    write_halfspace_table()
    # Current ECC tables belong to write_ecc_budget_paper.py (uniform5 cohort).


def save(fig, name):
    for ext in ['pdf', 'png']:
        fig.savefig(FIG / f'{name}.{ext}', dpi=250, bbox_inches='tight')
    plt.close(fig)


def residual(ax, problem, query):
    methods = ['global', 'dd', 'nr', 'fixed_pod', 'shifted'] if problem == 'halfspace' else ['global', 'dd', 'shifted']
    for method in methods:
        rows = [r for r in HIST if r['problem'] == problem and r['method'] == method and int(r['query']) == query]
        x = np.array([int(r['step']) for r in rows]); y = np.array([float(r['relative_residual']) for r in rows])
        ax.semilogy(x, y, color=COLORS[method], marker=MARKERS[method],
                    markevery=np.unique(np.linspace(0, len(x)-1, min(12, len(x))).round().astype(int)),
                    ms=3, mfc='white', lw=1.1, label=LABELS[method])
        ax.plot(x[-1], y[-1], 's', color=COLORS[method], ms=4)
    plain = [r for r in PLAIN if r['problem'] == problem]
    ax.semilogy([int(r['step']) for r in plain], [float(r['relative_residual']) for r in plain],
                ':', color='#660080', lw=1.2, label='Plain Helmholtz GMRES')
    tol = 1e-12 if problem == 'halfspace' else 1e-10
    ax.axhline(tol, ls='--', color='.55', lw=.6)
    ax.set(xlabel='ROM steps + SL steps', ylabel='Relative residual',
           ylim=(1e-15 if problem == 'halfspace' else 1e-11, 2))
    # Use the completion range for comparison; the raw plain diagnostic retains 100 steps.
    xmax = 40 if problem == 'halfspace' else 25
    ax.set_xlim(-.6, xmax)
    ax.set_xticks(np.arange(0, xmax + 1, 10 if problem == 'halfspace' else 5))
    ax.legend(fontsize=7, loc='upper right', bbox_to_anchor=(1, .9 if problem == 'halfspace' else 1),
              framealpha=1, handlelength=1.4)
    ax.grid(alpha=.2)


def write_halfspace_convergence():
    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 3.6), layout='constrained')
    for panel in (ax, bx):
        panel.set_box_aspect(1)
    residual(ax, 'halfspace', 10)
    ax.set_title('(a) Convergence', loc='left', fontsize=10)
    rows = read(ROOT / 'outputs/halfspace_dd_paper/layer_storage.csv')
    for method, offset in [('global', -.18), ('dd', .18)]:
        y = [float(r[method+'_entries'])*16/2**20 for r in rows]
        bx.bar(np.arange(1, 4)+offset, y, .34, color=COLORS[method], label='Global' if method == 'global' else 'DD')
    for j, row in enumerate(rows, 1):
        top = float(row['global_entries'])*16/2**20
        bx.text(j, top + .3, f"{float(row['saved_percent']):.1f}% saved",
                ha='center', va='bottom', fontsize=8)
    bx.set(xticks=[1, 2, 3], xlabel='Layer', ylabel='Basis storage (MiB)', ylim=(0, 13))
    bx.set_title('(b) Layerwise storage', loc='left', fontsize=10)
    bx.legend(frameon=False, fontsize=8, loc='upper left')
    bx.grid(axis='y', alpha=.2); bx.set_axisbelow(True)
    save(fig, 'halfspace_convergence_storage')


def write_figures():
    write_halfspace_convergence()
    fig, ax = plt.subplots(figsize=(3.8, 3.15), layout='constrained')
    residual(ax, 'ecc', 1)
    save(fig, 'ecc_residuals')
    fig, ax = plt.subplots(figsize=(6.4, 2.5), layout='constrained')
    regional = {'global': np.array([578125, 743552])*16/2**20,
                'dd': np.array([414750, 486384])*16/2**20}
    for method, off in [('global', .18), ('dd', -.18)]:
        bars = ax.barh(np.array([1, 0])+off, regional[method], .31,
                       color=COLORS[method], label='Global' if method == 'global' else 'DD')
        ax.bar_label(bars, fmt='%.3f', padding=4, fontsize=9)
    for j, y in enumerate([1, 0]):
        saving = 100*(1-regional['dd'][j]/regional['global'][j])
        ax.text(13.8, y, f'{saving:.1f}% saved', ha='right', va='center', fontsize=9)
    ax.set(yticks=[1, 0], yticklabels=['Physical domain', 'PML'], xlim=(0, 14),
           ylim=(-.55, 1.7), xlabel='Basis storage (MiB)', xticks=np.arange(0, 13, 2))
    ax.legend(frameon=False, loc='upper right', ncol=2, fontsize=9)
    ax.grid(axis='x', alpha=.2); ax.set_axisbelow(True)
    save(fig, 'halfspace_regional_storage')
    for problem, field in [('halfspace', 'halfspace_global_field'), ('ecc', 'ecc_dd_field')]:
        x = np.loadtxt(FIELDS / (problem+'_x.csv'), delimiter=',')
        y = np.loadtxt(FIELDS / (problem+'_y.csv'), delimiter=',')
        u = np.loadtxt(FIELDS / (field+'.csv'), delimiter=',')
        assert u.shape == (len(y), len(x))
        fig, ax = plt.subplots(figsize=(3.5, 3.15), layout='constrained')
        if problem == 'ecc':
            X, Y = np.meshgrid(x, y); a=.12*np.sqrt(.907218592905774); b=.12/np.sqrt(.907218592905774)
            u=np.ma.masked_where(((X-.5)/a)**2+((Y-.5)/b)**2<1, u)
        vmax=float(np.max(np.abs(u)))
        im=ax.pcolormesh(x, y, u, shading='nearest', cmap='RdBu_r',
                         norm=SymLogNorm(linthresh=vmax*.01, vmin=-vmax, vmax=vmax), rasterized=True)
        ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, ls='--', lw=.8, ec='.3'))
        if problem == 'halfspace':
            ax.plot([.5,.5],[0,1],':',color='.3',lw=.8)
        else:
            ax.add_patch(Ellipse((.5,.5),2*a,2*b,facecolor='white',edgecolor='.3',lw=.7))
        ax.set(xlabel='$x$', ylabel='$y$', aspect='equal')
        ax.set_xlim(x[0], x[-1]); ax.set_ylim(y[0], y[-1])
        cb=fig.colorbar(im, ax=ax, shrink=.85, label=r'$\mathrm{Re}\,u$')
        ticks=np.array([-1, -.1, 0, .1, 1])*vmax
        cb.set_ticks(ticks, labels=[f'{t:.1e}' if t else '0' for t in ticks])
        cb.ax.tick_params(labelsize=8)
        save(fig, field)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tables-only', action='store_true')
    args = parser.parse_args()
    write_tables()
    if not args.tables_only:
        write_figures()
    print('JCP_SHIFTED_PAPER_ASSETS_PASS')
