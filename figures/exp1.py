"""Experiment 1 (half-space): figures and tables from ``<data>/exp1``."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Rectangle
from matplotlib.ticker import LogLocator, MaxNLocator, NullLocator
from scipy.io import loadmat

from style import (COLORS, NAMES, chart, field_axes, panel_style, per_query_summary,
                   query_time_panel, real_field, save, sci, storage_panel)

# Residual-history palette of the method comparison (Figure 2, right).
HISTORY_LABELS = {'global': 'Global MSRB', 'dd': 'DD-MSRB', 'nt': 'NT-ROM',
                  'ss': 'SS-ROM', 'shifted': 'Shifted ILU-GMRES'}
HISTORY_COLORS = {'global': '#d90d0d', 'dd': '#0d7340', 'nt': '#2659bf',
                  'ss': '#d9730d', 'shifted': '#262626'}
HISTORY_MARKERS = {'global': 'o', 'dd': 's', 'nt': 'D', 'ss': '^', 'shifted': 'v'}
FREQUENCIES = [53.5, 80.25, 107, 214]
TABLE1 = [('global', 'Global MSRB'), ('dd', 'DD-MSRB'), ('nt', 'NT-ROM'), ('ss', 'SS-ROM'),
          ('shifted', 'Shifted ILU-GMRES'), ('direct', 'Direct')]


def fields(data, out):
    d = loadmat(data / 'field_omega53.5.mat')
    x, y = d['x'].ravel(), d['y'].ravel()
    real_field(d['U'], x, y, out, 'halfspace', tuple(d['source'].ravel()))
    fig, ax, cax = field_axes()
    # Low-saturation blue-grays: subdomain 1 is lightest, subdomain 10 darkest.
    colors = ['#f7f9fb', '#e3e8ed', '#ced6de', '#b8c4ce', '#9fadb9',
              '#8593a1', '#6b7784', '#525b65', '#393f46', '#20242a']
    im = ax.imshow(d['labels'], origin='lower', extent=[x[0], x[-1], y[0], y[-1]],
                   cmap=ListedColormap(colors), norm=BoundaryNorm(np.arange(.5, 11), 10),
                   interpolation='nearest')
    ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, ec='black', lw=.8))
    ax.plot([.5, .5], [0, 1], 'k--', lw=.8)
    ax.set(xlabel='$x$', ylabel='$y$', xticks=[0, .5, 1], yticks=[0, .5, 1])
    fig.colorbar(im, cax=cax, ticks=np.arange(1, 11))
    save(fig, out, 'halfspace_partition')
    fig, ax = chart()
    for j in (1, 2, 3):
        v = d[f'spectrum_{j}'].ravel()
        ax.semilogy(np.arange(1, len(v) + 1), np.maximum(v, 1e-16), '-o', ms=3, lw=1.2,
                    label=f'Iteration {j}')
    ax.set(xlabel='Mode index', ylabel='Singular value')
    ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
    ax.yaxis.set_major_locator(LogLocator(numticks=6))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.legend(frameon=False)
    save(fig, out, 'halfspace_spectra')
    w = loadmat(data / 'field_omega214.mat')
    assert w['rho'].item() <= 1e-10
    real_field(w['U'], w['x'].ravel(), w['y'].ravel(), out, 'omega214', w['source'].ravel())


def residuals(data, out):
    h = pd.read_csv(data / 'residual_histories_query10.csv')
    fig, ax = chart()
    for method in HISTORY_LABELS:
        t = h[h.method == method]
        x, y = t.step.to_numpy(), t.relative_residual.to_numpy()
        ax.semilogy(x, y, color=HISTORY_COLORS[method], marker=HISTORY_MARKERS[method],
                    markevery=np.unique(np.linspace(0, len(x) - 1, min(12, len(x))).round().astype(int)),
                    ms=3, mfc='white', lw=1.1, label=HISTORY_LABELS[method])
        ax.plot(x[-1], y[-1], 's', color=HISTORY_COLORS[method], ms=4)
    p = h[h.method == 'plain_gmres']
    ax.semilogy(p.step, p.relative_residual, ':', color='#660080', lw=1.2, label='Plain Helmholtz GMRES')
    ax.axhline(1e-12, ls='--', color='.55', lw=.6)
    ax.set(xlabel='ROM steps + SL steps', ylabel='Relative residual', ylim=(1e-15, 2))
    ax.set_xlim(-.6, 40)
    ax.set_xticks(np.arange(0, 41, 10))
    ax.legend(fontsize=7, loc='upper right', bbox_to_anchor=(1, .9),
              framealpha=1, handlelength=1.4)
    ax.grid(alpha=.2)
    save(fig, out, 'halfspace_residuals')


def cost(data, out):
    t = pd.read_csv(data / 'online_timings.csv')
    query_time_panel(per_query_summary(t), out, 'halfspace_representative_time_time', legend='middle')
    s = pd.read_csv(data / 'storage.csv').set_index('method')
    cols = ['basis_MiB', 'payload_MiB', 'object_MiB']
    storage_panel(s.loc['global', cols], s.loc['dd', cols], ['Basis', 'Payload', 'Object'],
                  out, 'halfspace_storage_total')


def frequency(data, out):
    s = pd.read_csv(data / 'frequency_storage.csv')
    values = [s[s.method == m].sort_values('omega').payload_MiB.to_numpy() for m in ('global', 'dd')]
    storage_panel(*values, [f'{w:g}' for w in FREQUENCIES], out, 'halfspace_frequency_storage',
                  xlabel=r'Angular frequency $\omega$')
    t = pd.read_csv(data / 'frequency_timings.csv')
    assert (t.repetitions == 5).all()
    fig, ax = chart()
    for m in COLORS:
        p = t[t.method == m].pivot(index='omega', columns='c_right', values='mean_total_s')
        avg = p.mean(axis=1)
        ax.errorbar(p.index.to_numpy(), avg, yerr=[avg - p.min(axis=1), p.max(axis=1) - avg],
                    color=COLORS[m], marker='o', capsize=3, label=NAMES[m])
    ax.set(yscale='log', xlabel=r'Angular frequency $\omega$', ylabel='Total online time (s)',
           xticks=FREQUENCIES, xticklabels=['53.5', '80.25', '107', '214'])
    ax.legend(frameon=False, fontsize=7)
    save(fig, out, 'halfspace_frequency_time')


def figures(data, out):
    data = Path(data)
    with panel_style():
        fields(data, out)
        residuals(data, out)
        cost(data, out)
        frequency(data, out)


def table_cost(data):
    """Table 1: online cost over the 23 test queries."""
    t = pd.read_csv(Path(data) / 'online_timings.csv')
    g = t.groupby('method')
    median = g.total_s.median()
    lines = [r'\begin{tabular}{@{}lrrrr@{}}\toprule', r'Method & Mean & Median & It. & Speedup \\',
             r'\midrule']
    for method, label in TABLE1:
        it = '--' if method == 'direct' else f'{g.sl_steps.mean()[method]:.2f}'
        lines.append(f'{label} & {g.total_s.mean()[method]:.4f} & {median[method]:.4f} & {it} & '
                     f'${median["shifted"] / median[method]:.1f}' + r'\times$ \\')
    return '\n'.join(lines + [r'\bottomrule\end{tabular}']) + '\n'


def table_coverage(data):
    """Table 2: training coverage n_s = 6, 12, 24 over nine fixed queries."""
    c = pd.read_csv(Path(data) / 'training_coverage.csv')
    lines = [r'\begin{tabular}{rlrrrrr}\toprule',
             r'$n_s$ & Method & Payload (MiB) & Saving & Offline (s) & Time (s) & Max. $\rho_{\rm ROM}$ \\',
             r'\midrule']
    for n in (6, 12, 24):
        g = c[(c.n_s == n) & (c.method == 'global')].iloc[0]
        for method, label in [('global', 'Global MSRB'), ('dd', 'DD-MSRB')]:
            r = c[(c.n_s == n) & (c.method == method)].iloc[0]
            saving = 100 * (1 - r.payload_MiB / g.payload_MiB)
            lines.append(' & '.join([str(n), label, f'{r.payload_MiB:.3f}', rf'\({saving:.1f}\%\)',
                                     f'{r.offline_s:.2f}', f'{r.online_mean_s:.4f}',
                                     sci(r.max_rho_rom)]) + r' \\')
    return '\n'.join(lines + [r'\bottomrule\end{tabular}']) + '\n'


def table_partitions(data):
    """Table 3: partition sensitivity over five fixed queries."""
    p = pd.read_csv(Path(data) / 'partitions.csv').set_index('variant')
    names = [('global', 'Global'), ('material_depth', 'Material/depth'), ('merged', 'Merged PML'),
             ('signed_depth', 'Signed/depth'), ('uniform4x4', 'Uniform tiles')]
    lines = [r'\begin{tabular}{llrrrr}\toprule',
             r'Partition & Iteration-wise ranks & Payload (MiB) & Saving & Time (s) & Max. $\rho_{\rm ROM}$ \\',
             r'\midrule']
    for key, label in names:
        r = p.loc[key]
        ranks = '(' + ','.join(r.ranks.strip('[]').split()) + ')'
        saving = 100 * (1 - r.payload_MiB / p.loc['global', 'payload_MiB'])
        lines.append(f'{label} & \\({ranks}\\) & {r.payload_MiB:.3f} & {saving:.1f}\\% & '
                     f'{r.mean_total_s:.4f} & {sci(r.max_rho_rom)}' + r' \\')
    return '\n'.join(lines + [r'\bottomrule\end{tabular}']) + '\n'


def tables(data):
    return {'table1_halfspace_cost': table_cost(data),
            'table2_training_coverage': table_coverage(data),
            'table3_partitions': table_partitions(data)}
