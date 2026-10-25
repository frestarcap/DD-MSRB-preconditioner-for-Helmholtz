"""Experiment 3 (3D Gaussian waveguide): figures and tables from ``<data>/exp3``.

The geometry renderings (profile isosurfaces and the partition) are drawn by
``waveguide_geometry.m`` in MATLAB.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import SymLogNorm
from matplotlib.ticker import NullLocator
from scipy.io import loadmat

from style import COLORS, NAMES, chart, panel_style, save, storage_panel

OMEGAS = [24, 36, 48, 60]
SHIFTED = 'Shifted MG-GMRES'
VALIDATION_KEY = {'global': 'global3', 'dd': 'dd3', 'shifted': 'shifted_mg'}
VALIDATION_STYLE = {'global3': ('#2365A0', 'o', r'Global ($k=3$)'),
                    'global2': ('#C27816', 's', r'Global ($k=2$)'),
                    'dd3': ('#238562', '^', r'DD ($k=3$)'),
                    'shifted_mg': ('#414141', 'D', SHIFTED)}
VALIDATION_ROWS = [('global3', r'Global (\(k=3\))'), ('global2', r'Global (\(k=2\))'),
                   ('dd3', r'DD (\(k=3\))'), ('shifted_mg', SHIFTED)]


def field(data, out):
    s = loadmat(data / 'field_omega48_mu0.317.mat')
    x, y, v = s['x'].ravel(), s['y'].ravel(), s['real']
    assert v.shape == (len(x), len(y))
    norm = SymLogNorm(linthresh=.02, vmin=-1, vmax=1)
    cmap = plt.get_cmap('RdBu_r')
    xx, yy = np.meshgrid(x, y, indexing='ij')
    fig = plt.figure(figsize=(7.2, 2.8))
    ax = fig.add_axes([-.04, .08, .90, .86], projection='3d', computed_zorder=False)
    ax.plot_surface(xx, yy, np.full_like(xx, .5), facecolors=cmap(norm(v / np.max(abs(v)))),
                    rstride=1, cstride=1, shade=False, antialiased=False, rasterized=True,
                    zorder=2, clip_on=False)
    for z in [0, 1]:
        ax.plot([0, 3, 3, 0, 0], [0, 0, 1, 1, 0], [z] * 5, color='.4', lw=.65, zorder=3, clip_on=False)
    for x0 in [0, 3]:
        for y0 in [0, 1]:
            ax.plot([x0] * 2, [y0] * 2, [0, 1], color='.4', lw=.65, zorder=3, clip_on=False)
    ax.scatter([.2], [.5], [.5], marker='*', s=35, color='black', depthshade=False, zorder=4)
    ax.set(xlabel='$x$', ylabel='$y$', zlabel='$z$', zlim=(0, 1), xticks=[0, 1, 2, 3], yticks=[0, 1],
           zticks=[0, .5, 1])
    ax.set_box_aspect((3.35, 1.35, 1), zoom=1.65)
    ax.view_init(elev=22, azim=-75)
    ax.tick_params(pad=0, labelsize=9)
    ax.xaxis.labelpad = 0; ax.yaxis.labelpad = 0; ax.zaxis.labelpad = 0
    ax.grid(False)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.fill = False
        axis.pane.set_edgecolor('white')
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=fig.add_axes([.87, .20, .018, .60]),
                      ticks=[-1, -.1, 0, .1, 1])
    cb.ax.tick_params(labelsize=8)
    save(fig, out, 'waveguide_field_48')


def residuals(data, out):
    """ROM steps from the zero initial guess followed by SL steps; steps beyond 20 compressed."""
    h = pd.read_csv(data / 'residual_histories_omega48_mu0.317.csv')
    q = pd.read_csv(data / 'frequency_queries.csv')
    q = q[(q.omega == 48) & (q.mu == .317)].set_index('method')
    fold, squeeze = 20, 13.5
    forward = lambda t: np.where(t <= fold, t, fold + (t - fold) / squeeze)
    inverse = lambda u: np.where(u <= fold, u, fold + (u - fold) * squeeze)
    fig, ax = chart()
    ax.set_xscale('function', functions=(forward, inverse))
    ax.axvspan(fold, 352, color='.94', lw=0, zorder=0)
    for m in ['shifted', 'global', 'dd']:
        history = h[h.method == m].relative_residual.to_numpy()
        r = q.loc[m]
        assert r.true_residual <= 1e-6
        steps = np.arange(len(history))
        marks = [i for i in steps if i <= fold or (i - fold) % 10 == 0]
        ax.semilogy(steps, history, color=COLORS[m], lw=1.1, marker='o', ms=2.5, markevery=marks,
                    label=SHIFTED if m == 'shifted' else NAMES[m])
        ax.plot(steps[-1], r.true_residual, 'o', color=COLORS[m], ms=4.5)
        ax.annotate(str(int(r.sl_steps)), (steps[-1], r.true_residual), xytext=(0, -13),
                    textcoords='offset points', ha='center', fontsize=8)
    ax.axvline(3, color='.4', ls=':', lw=.9)
    ax.axhline(1e-6, color='.4', ls='--', lw=.8)
    ax.text(2.6, 2.2e-6, 'ROM', ha='right', fontsize=8)
    ax.text(3.5, 2.2e-6, 'SL', ha='left', fontsize=8)
    ax.text(200, 2.6e-6, 'compressed scale', ha='center', fontsize=7, color='.45')
    ax.set(xlabel='ROM steps + SL steps', ylabel='Relative residual', xlim=(-.5, 352), ylim=(4e-7, 2),
           xticks=[0, 3, 10, 20, 100, 200, 300])
    ax.yaxis.set_minor_locator(NullLocator())
    ax.legend(frameon=False, loc='upper right')
    save(fig, out, 'waveguide_residuals')


def validation(data, out):
    v = pd.read_csv(data / 'validation_omega48.csv')
    fig, ax = chart()
    for method, (color, marker, label) in VALIDATION_STYLE.items():
        t = v[v.method == method].sort_values('mu')
        ax.plot(t.mu, t.mean_total_s, color=color, lw=.9, label=label)
        for r in t.itertuples():
            ax.plot(r.mu, r.mean_total_s, marker=marker, ms=4,
                    mfc=color if r.group == 'common' else 'white', mec=color)
            if pd.notna(r.sd_total_s):
                ax.errorbar(r.mu, r.mean_total_s, yerr=r.sd_total_s, fmt='none', ecolor=color, capsize=2)
    ax.set(xlabel=r'Query parameter $\mu$', ylabel='Total online time (s)', xlim=(.2, .59))
    ax.legend(frameon=False, fontsize=7, loc='best')
    save(fig, out, 'waveguide_validation_time')


def online_times(data):
    """Online time per (omega, mu, method); omega=48 uses the repeated ten-query timings."""
    q = pd.read_csv(data / 'frequency_queries.csv')
    v = pd.read_csv(data / 'validation_omega48.csv')
    common = v[v.group == 'common'].set_index(['mu', 'method']).mean_total_s
    q['paper_online_s'] = [common[(r.mu, VALIDATION_KEY[r.method])] if r.omega == 48 else r.online_s
                           for r in q.itertuples()]
    return q


def storage(data, out):
    s = pd.read_csv(data / 'storage_omega48.csv').set_index('method')
    cols = ['basis_MiB', 'payload_MiB', 'object_MiB']
    storage_panel(s.loc['global', cols], s.loc['dd', cols], ['Basis', 'Payload', 'Object'],
                  out, 'waveguide_storage_total')
    f = pd.read_csv(data / 'frequency_storage.csv').sort_values('omega')
    storage_panel(f.global_payload_MiB, f.dd_payload_MiB, [str(w) for w in OMEGAS], out,
                  'waveguide_frequency_storage', xlabel=r'Angular frequency $\omega$',
                  ylabel='Payload (MiB)', saving_over_pair=True)


def frequency_time(data, out):
    q = online_times(data)
    x = np.arange(len(OMEGAS))
    fig, ax = chart()
    for m, off in [('global', -.055), ('dd', .055), ('shifted', 0)]:
        v = np.array([q[(q.omega == w) & (q.method == m)].paper_online_s.to_numpy() for w in OMEGAS])
        avg = v.mean(axis=1)
        ax.errorbar(x + off, avg, yerr=[avg - v.min(axis=1), v.max(axis=1) - avg], color=COLORS[m],
                    marker='o', capsize=3, label=SHIFTED if m == 'shifted' else NAMES[m])
    ax.set(yscale='log', ylabel='Total online time (s)')
    ax.legend(frameon=False, loc='lower right')
    ax.set(xlabel=r'Angular frequency $\omega$', xticks=x, xticklabels=[str(w) for w in OMEGAS])
    save(fig, out, 'waveguide_frequency_time')


def figures(data, out):
    data = Path(data)
    with panel_style():
        field(data, out)
        residuals(data, out)
        validation(data, out)
        storage(data, out)
        frequency_time(data, out)


def table_validation(data):
    """Online cost at omega=48 by query group (three common, seven added, all ten)."""
    v = pd.read_csv(Path(data) / 'validation_omega48.csv')
    groups = [v[v.group == 'common'], v[v.group == 'added'], v]
    lines = [r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrrrr@{}}\toprule',
             r'& \multicolumn{2}{c}{Three common queries} & \multicolumn{2}{c}{Seven added queries} '
             r'& \multicolumn{2}{c}{All ten queries} \\',
             r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){6-7}',
             r'Method & Online (s) & SL steps & Online (s) & SL steps & Online (s) & SL steps \\',
             r'\midrule']
    for method, label in VALIDATION_ROWS:
        cells = [label]
        for g in groups:
            t = g[g.method == method]
            cells += [f'{t.mean_total_s.mean():.2f}', f'{t.mean_sl_steps.mean():.1f}']
        lines.append(' & '.join(cells) + r' \\')
    return '\n'.join(lines + [r'\bottomrule\end{tabular*}']) + '\n'


def table_frequency(data):
    """rho_ROM, SL steps and online time over the three common queries; total training time."""
    q = online_times(Path(data))
    off = pd.read_csv(Path(data) / 'offline_times.csv').set_index(['omega', 'method']).offline_total_s

    def sci1(x):
        mantissa, exponent = f'{x:.1e}'.split('e')
        return rf'\({mantissa}\times10^{{{int(exponent)}}}\)'

    lines = [r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}rrrrrrrrrrr@{}}\toprule',
             r'& \multicolumn{2}{c}{\(\rho_{\rm ROM}\)} & \multicolumn{3}{c}{SL steps} & '
             r'\multicolumn{3}{c}{Online (s)} & \multicolumn{2}{c}{Offline (s)} \\',
             r'\cmidrule(lr){2-3}\cmidrule(lr){4-6}\cmidrule(lr){7-9}\cmidrule(l){10-11}',
             r'$\omega$ & Global & DD & Global & DD & Baseline & Global & DD & Baseline & Global & DD \\',
             r'\midrule']
    for w in OMEGAS:
        t = q[q.omega == w].groupby('method')
        cells = [str(w), sci1(t.rho_rom.mean()['global']), sci1(t.rho_rom.mean()['dd'])]
        cells += [f'{t.sl_steps.mean()[m]:.1f}' for m in ['global', 'dd', 'shifted']]
        cells += [f'{t.paper_online_s.mean()[m]:.2f}' for m in ['global', 'dd', 'shifted']]
        cells += [f'{off[(w, m)]:.0f}' for m in ['global', 'dd']]
        lines.append(' & '.join(cells) + r' \\')
    return '\n'.join(lines + [r'\bottomrule\end{tabular*}']) + '\n'


def tables(data):
    return {'table5_waveguide_validation': table_validation(data),
            'table6_waveguide_frequency': table_frequency(data)}
