"""Experiment 2 (elliptic-obstacle scattering): figures and table from ``<data>/exp2``."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import NullLocator
from scipy.io import loadmat

from style import (chart, field_axes, panel_style, per_query_summary, query_time_panel,
                   real_field, save, storage_panel)

# The residual panel keeps the style it was published with.
RESIDUAL_RC = {'font.size': 9, 'pdf.fonttype': 42, 'axes.spines.top': False,
               'axes.spines.right': False}
RESIDUAL_STYLE = [('global', 'Global MSRB', '#2879ad'), ('dd', 'DD-MSRB', '#c64c32'),
                  ('shifted', 'Shifted ILU-GMRES', '#444444')]
TABLE = [('global', 'Global MSRB'), ('dd', 'DD-MSRB'), ('shifted', 'Shifted ILU-GMRES'),
         ('direct', 'Sparse direct')]
NEAR_RADIUS = .24


def field_and_partition(data, out):
    d = loadmat(data / 'field.mat')
    x, y = d['x'].ravel(), d['y'].ravel()
    assert d['rho'].item() <= 1e-12
    real_field(d['U'], x, y, out, 'ecc_field', obstacle_ratio=d['aspect_ratio'].item())
    X, Y = np.meshgrid(x, y)
    physical = (X > 0) & (X < 1) & (Y > 0) & (Y < 1)
    region = np.full(X.shape, 3)
    region[physical] = 2
    region[physical & (np.hypot(X - .5, Y - .5) <= NEAR_RADIUS)] = 1
    fig, ax, cax = field_axes()
    im = ax.imshow(region, origin='lower', extent=[x[0], x[-1], y[0], y[-1]],
                   cmap=ListedColormap(['#c8deed', '#f4f0e8', '#c8c8c8']),
                   norm=BoundaryNorm([.5, 1.5, 2.5, 3.5], 3), interpolation='nearest')
    ax.set(xlabel='$x$', ylabel='$y$', xticks=[0, .5, 1], yticks=[0, .5, 1])
    fig.colorbar(im, cax=cax, ticks=[1, 2, 3])
    save(fig, out, 'ecc_partition')


def spectra(data, out):
    s = pd.read_csv(data / 'tail_spectra.csv')
    r = pd.read_csv(data / 'rank_storage.csv')
    ranks = r[r.method == 'global'].sort_values('iteration').dimension.to_numpy()
    fig = plt.figure(figsize=(4.4, 3.8))
    ax = fig.add_axes([.18, .14, .65, .65 * 4.4 / 3.8])
    ax.grid(which='major', alpha=.18)
    for j, k in zip([1, 2, 3], ranks):
        t = s[s.iteration == j]
        line, = ax.semilogy(t['rank'], t['tail'].clip(lower=1e-16), label=f'Iteration {j}')
        ax.plot(k, t.loc[t['rank'] == k, 'tail'].iloc[0], 'o', color=line.get_color())
    ax.axhline(1e-5, color='.4', ls='--', lw=.8)
    ax.set(xlabel='Retained rank', ylabel='Relative Frobenius tail', ylim=(1e-10, 2))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.legend(frameon=False)
    save(fig, out, 'ecc_spectra')


def residuals(data, out):
    h = pd.read_csv(data / 'residual_histories_query1.csv')
    fig, ax = plt.subplots(figsize=(3.8, 3.15), layout='constrained')
    for method, label, color in RESIDUAL_STYLE:
        t = h[h.method == method]
        ax.semilogy(t.step, t.relative_residual, 'o-', color=color, ms=3, mfc='white', lw=1, label=label)
        ax.plot(t.step.iloc[-1], t.relative_residual.iloc[-1], 's', color=color, ms=4)
    ax.axhline(1e-12, color='.5', ls='--', lw=.7)
    ax.set(xlabel='ROM steps + SL steps', ylabel='Relative residual', ylim=(1e-15, 2))
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(alpha=.2)
    save(fig, out, 'ecc_residuals', dpi=220, bbox_inches='tight')


def cost(data, out):
    t = pd.read_csv(data / 'online_timings.csv')
    query_time_panel(per_query_summary(t), out, 'ecc_time_time', legend='top')
    s = pd.read_csv(data / 'storage.csv').set_index('method')
    cols = ['basis_MiB', 'payload_MiB', 'object_MiB']
    storage_panel(s.loc['global', cols], s.loc['dd', cols], ['Basis', 'Payload', 'Object'],
                  out, 'ecc_storage_total')


def figures(data, out):
    data = Path(data)
    with panel_style():
        field_and_partition(data, out)
        spectra(data, out)
        cost(data, out)
    with panel_style(RESIDUAL_RC):
        residuals(data, out)


def table_cost(data):
    """Online cost over the 22 test geometries (five repetitions per query)."""
    t = pd.read_csv(Path(data) / 'online_timings.csv')
    m = t.groupby('method')[['ROM_s', 'setup_s', 'solve_s', 'total_s', 'sl_steps']].mean()
    lines = [r'\begin{tabular}{lrrrrrr}\toprule',
             r'Method & ROM (s) & Setup (s) & Solve (s) & Total (s) & It. & Speedup \\', r'\midrule']
    for method, label in TABLE:
        r = m.loc[method]
        it = '--' if method == 'direct' else f'{r.sl_steps:.2f}'
        speedup = m.loc['shifted', 'total_s'] / r.total_s
        lines.append(f'{label} & {r.ROM_s:.4f} & {r.setup_s:.4f} & {r.solve_s:.4f} & {r.total_s:.4f} & '
                     f'{it} & $ {speedup:.2f} \\times $ ' + r'\\')
    return '\n'.join(lines + [r'\bottomrule\end{tabular}']) + '\n'


def tables(data):
    return {'table4_ecc_cost': table_cost(data)}
