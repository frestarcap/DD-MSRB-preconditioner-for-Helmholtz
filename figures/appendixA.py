"""Appendix A (residual composition and multigrid response): figures from ``<data>/appendixA``."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from scipy.io import loadmat

from style import COLORS, NAMES, chart, panel_style, save

OMEGAS = [24, 36, 48, 60]


def slices(data, out):
    """|r| on z=0.5 before (r0) and after the ROM steps, with subdomain interfaces in white."""
    d = loadmat(data / 'residual_slices_omega48.mat')
    x, y, h = d['x'].ravel(), d['y'].ravel(), d['spacing'].ravel()
    labels, n, p = d['subdomain_labels'], d['physical_grid_size'].ravel(), d['pml_cells'].ravel()
    i, j = np.arange(len(x))[:, None], np.arange(len(y))[None, :]
    physical = (i >= p[0]) & (i < p[0] + n[0]) & (j >= p[1]) & (j < p[1] + n[1])
    segments = []
    for a, b in zip(*np.where((labels[1:] != labels[:-1]) & physical[1:] & physical[:-1])):
        xx = (x[a] + x[a + 1]) / 2
        segments.append([(xx, y[b] - h[1] / 2), (xx, y[b] + h[1] / 2)])
    for a, b in zip(*np.where((labels[:, 1:] != labels[:, :-1]) & physical[:, 1:] & physical[:, :-1])):
        yy = (y[b] + y[b + 1]) / 2
        segments.append([(x[a] - h[0] / 2, yy), (x[a] + h[0] / 2, yy)])
    for name in ('global', 'dd'):
        fig, axs = plt.subplots(2, 1, figsize=(4.4, 3.8))
        fig.subplots_adjust(left=.12, right=.78, bottom=.14, top=.96, hspace=.46)
        for ax, field in zip(axs, ['r0', 'rafter']):
            values = d[f'{name}_{field}'] / np.sqrt(np.prod(h))
            assert values.shape == (len(x), len(y))
            im = ax.imshow(np.log10(np.maximum(values, 1e-5)).T, origin='lower',
                           extent=[x[0] - h[0] / 2, x[-1] + h[0] / 2, y[0] - h[1] / 2, y[-1] + h[1] / 2],
                           cmap='magma', vmin=-4, vmax=1, interpolation='nearest', rasterized=True)
            ax.add_collection(LineCollection(segments, colors='white', linewidths=.25, alpha=.35))
            ax.add_patch(Rectangle((0, 0), 3, 1, fill=False, ec='#bdbdbd', ls='--', lw=.65))
            ax.set(xticks=[0, 1, 2, 3], yticks=[0, .5, 1], ylabel='$y$')
        axs[1].set_xlabel('$x$')
        cb = fig.colorbar(im, cax=fig.add_axes([.82, .20, .025, .64]), ticks=[-4, -3, -2, -1, 0, 1])
        cb.ax.tick_params(labelsize=8)
        save(fig, out, 'residual_slices_' + name)


def diagnostics(data, out):
    rows = pd.read_csv(data / 'residual_components.csv')
    for mode in ['roughness', 'response']:
        fig, ax = chart()
        for m in ['global', 'dd']:
            if mode == 'roughness':
                groups = [rows[(rows.method == m) & (rows.omega == w)].residual_roughness_transverse_over_omega2
                          for w in OMEGAS]
                assert all(len(g) == 3 for g in groups)
                ax.plot(OMEGAS, [g.mean() for g in groups], '-o', ms=4, color=COLORS[m], label=NAMES[m])
                for w, g in zip(OMEGAS, groups):
                    ax.scatter([w] * len(g), g, s=10, alpha=.4, color=COLORS[m])
            else:
                for comp, style in [('pre', '-o'), ('coarse', ':^'), ('combined', '--s')]:
                    vals = [rows[(rows.method == m) & (rows.omega == w)]['mg_' + comp + '_optimal_contraction'].mean()
                            for w in OMEGAS]
                    ax.plot(OMEGAS, vals, style, ms=4, color=COLORS[m],
                            label=f'{m}: {comp if comp != "combined" else "full cycle"}')
        ax.set(xlabel=r'$\omega$', xticks=OMEGAS,
               ylabel=(r'$\|\nabla_\perp r\|^2/(\omega^2\|r\|^2)$' if mode == 'roughness'
                       else r'$\kappa_i$ (smaller is better)'))
        if mode == 'response':
            ax.set_ylim(0, 1.05)
        ax.legend(frameon=False, fontsize=7, ncol=1 if mode == 'roughness' else 2)
        save(fig, out, 'residual_' + mode)


def figures(data, out):
    data = Path(data)
    with panel_style():
        slices(data, out)
        diagnostics(data, out)


def tables(data):
    return {}
