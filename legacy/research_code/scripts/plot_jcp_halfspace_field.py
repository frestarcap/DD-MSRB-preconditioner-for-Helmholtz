#!/usr/bin/env python3
"""Redraw JCP Fig. 7 from its frozen DD field, preserving all amplitudes."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.patches import Rectangle
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'outputs/halfspace_figures_20260910/omega214_field.mat'
PUB = ROOT / 'JCP/figures/halfspace_extensions'
OUT = ROOT / 'outputs/halfspace_field_contrast_20260922'
OUT.mkdir(exist_ok=True)
d = loadmat(SOURCE)
u = d['U']; x = d['x'].ravel(); y = d['y'].ravel(); source = d['source'].ravel()
assert float(d['rho'].item()) <= 1e-10
assert np.max(np.abs(source - [.2, .5])) < 1 / 504
v = u / np.max(np.abs(u))
limit = float(np.max(np.abs(v.real)))
threshold = .01
plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 11,
                     'pdf.fonttype': 42, 'axes.spines.top': False,
                     'axes.spines.right': False})
fig, axs = plt.subplots(1, 2, figsize=(8, 3.8), layout='constrained')
for index, ax in enumerate(axs):
    if index == 0:
        values = v.real
        options = dict(cmap='RdBu_r', norm=SymLogNorm(linthresh=threshold,
                       linscale=1, vmin=-limit, vmax=limit, base=10))
        title = r'(a) $\mathrm{Re}(u)/\|u\|_\infty$'
    else:
        values = np.log10(np.maximum(abs(v), 1e-6))
        options = dict(cmap='magma', vmin=-6, vmax=0)
        title = r'(b) $\log_{10}(|u|/\|u\|_\infty)$'
    im = ax.imshow(values, origin='lower', extent=[x[0], x[-1], y[0], y[-1]],
                   interpolation='nearest', rasterized=True, **options)
    ax.add_patch(Rectangle((0, 0), 1, 1, fill=False,
                          ec='black' if index == 0 else 'limegreen', lw=.8))
    ax.plot([.5, .5], [0, 1], '--', color='.3' if index == 0 else 'white', lw=.75)
    ax.scatter(*source, s=12, c='white', edgecolors='black', linewidths=.6)
    ax.set(xlabel='$x$', ylabel='$y$', title=title, xticks=[0, .5, 1], yticks=[0, .5, 1])
    cb = fig.colorbar(im, ax=ax, shrink=.87, pad=.02)
    if index == 0:
        cb.set_ticks([-.5, -.1, -.01, 0, .01, .1, .5],
                     labels=['−0.5', '−0.1', '−0.01', '0', '0.01', '0.1', '0.5'])
        cb.ax.set_title('symlog', fontsize=8, pad=6)
    else:
        cb.set_ticks([-6, -5, -4, -3, -2, -1, 0])
    cb.ax.tick_params(labelsize=9)
for ext in ['pdf', 'png']:
    fig.savefig(PUB / f'omega214_solution.{ext}', dpi=600 if ext == 'pdf' else 250,
                bbox_inches='tight', pad_inches=.04)
plt.close(fig)
(OUT / 'provenance.json').write_text(json.dumps({
    'source': str(SOURCE.relative_to(ROOT)),
    'sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    'true_residual': float(d['rho'].item()),
    'real_color_norm': 'SymLogNorm', 'linthresh': threshold,
    'symmetric_limits': [-limit, limit], 'amplitude_clipping': False,
    'magnitude_log10_floor': -6, 'pdf_raster_dpi': 600,
}, indent=2) + '\n')
print('Fig. 7 regenerated; frozen data and full real-amplitude range preserved.')
