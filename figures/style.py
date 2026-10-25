"""Shared plotting style and panel builders for the paper figures.

Every figure is drawn from the files in a data directory (``paper_data/`` or a
fresh ``results/`` export with the same schema).  The style values reproduce the
published panels exactly; change them only together with the reference images.
"""
from contextlib import contextmanager
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import SymLogNorm
from matplotlib.patches import Ellipse, Rectangle
from matplotlib.ticker import MaxNLocator

# Style of the single-panel layout used by most figures.
PANEL_RC = {'font.family': 'serif', 'font.size': 10, 'pdf.fonttype': 42,
            'axes.spines.top': False, 'axes.spines.right': False,
            'axes.linewidth': .7, 'legend.fontsize': 8}
COLORS = {'global': '#BF3030', 'dd': '#23794C', 'shifted': '#667085', 'direct': '#3768A0'}
NAMES = {'global': 'Global MSRB', 'dd': 'DD-MSRB', 'shifted': 'Shifted ILU-GMRES',
         'direct': 'Sparse direct'}


@contextmanager
def panel_style(rc=None):
    """Draw from matplotlib defaults plus ``rc`` (PANEL_RC by default)."""
    with matplotlib.rc_context():
        matplotlib.rcdefaults()
        matplotlib.use('Agg')
        plt.rcParams.update(PANEL_RC if rc is None else rc)
        yield


def save(fig, out, name, dpi=400, **kwargs):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f'{name}.pdf'
    fig.savefig(path, dpi=dpi, **kwargs)
    plt.close(fig)
    return path


def chart():
    fig, ax = plt.subplots(figsize=(4.2, 3.9), layout='constrained')
    ax.set_box_aspect(1)
    ax.grid(which='major', alpha=.18)
    ax.set_axisbelow(True)
    return fig, ax


def ratio_note(ax):
    ax.text(.98, .97, 'Above 1: DD faster', transform=ax.transAxes,
            ha='right', va='top', fontsize=8,
            bbox=dict(facecolor='white', edgecolor='none', alpha=.9, pad=2))


def storage_panel(global_values, dd_values, categories, out, stem, xlabel=None, ylabel=None,
                  saving_over_pair=False):
    """Paired global/DD bars with the DD saving printed above each pair."""
    g, d = np.asarray(global_values, float), np.asarray(dd_values, float)
    assert len(g) == len(d) == len(categories) and np.all(g > 0)
    fig, ax = chart()
    half_gap = .25 if len(categories) > 3 else .18
    maximum = max(g.max(), d.max())
    for method, values, offset in [('global', g, -half_gap), ('dd', d, half_gap)]:
        bars = ax.bar(np.arange(len(g)) + offset, values, .34,
                      color=COLORS[method], label=NAMES[method])
        ax.bar_label(bars, fmt='%.2f', fontsize=8, padding=3)
    for i, saving in enumerate(100 * (1 - d / g)):
        x, y = ((i, max(g[i], d[i]) + maximum * .08) if saving_over_pair
                else (i + half_gap, d[i] + maximum * .11))
        ax.text(x, y, f'{saving:.1f}%\nsaved', ha='center', va='bottom', fontsize=8)
    ax.set(xticks=np.arange(len(g)), xticklabels=categories,
           ylabel=ylabel or 'Persistent storage (MiB)', ylim=(0, maximum * 1.4))
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    ax.grid(axis='x', visible=False)
    ax.legend(frameon=False, loc='upper left')
    return save(fig, out, stem)


def field_axes():
    fig = plt.figure(figsize=(4.4, 3.8))
    ax = fig.add_axes([.13, .14, .65, .65 * 4.4 / 3.8])
    cax = fig.add_axes([.81, .14, .028, .65 * 4.4 / 3.8])
    return fig, ax, cax


def real_field(u, x, y, out, stem, source_xy=None, obstacle_ratio=None):
    """Normalized real part on a symmetric-log scale over the PML-extended grid."""
    if obstacle_ratio is not None:
        X, Y = np.meshgrid(x, y)
        a, b = .12 * np.sqrt(obstacle_ratio), .12 / np.sqrt(obstacle_ratio)
        u = np.ma.masked_where(((X - .5) / a)**2 + ((Y - .5) / b)**2 < 1, u)
    u = u / np.max(np.abs(u))
    limit = np.max(np.abs(u.real))
    fig, ax, cax = field_axes()
    im = ax.imshow(u.real, origin='lower', extent=[x[0], x[-1], y[0], y[-1]],
                   interpolation='nearest', rasterized=True, cmap='RdBu_r',
                   norm=SymLogNorm(linthresh=.01, vmin=-limit, vmax=limit))
    ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, ec='.3', lw=.8))
    if obstacle_ratio is None:
        ax.plot([.5, .5], [0, 1], '--', color='.3', lw=.7)
    else:
        ax.add_patch(Ellipse((.5, .5), 2 * a, 2 * b, facecolor='white', edgecolor='.3', lw=.7))
    if source_xy is not None:
        ax.scatter(*source_xy, s=13, c='white', edgecolors='black', linewidths=.6)
    ax.set(xlabel='$x$', ylabel='$y$', xticks=[0, .5, 1], yticks=[0, .5, 1])
    cb = fig.colorbar(im, cax=cax)
    cb.set_ticks([-.5, -.1, -.01, 0, .01, .1, .5],
                 labels=['−0.5', '−0.1', '−0.01', '0', '0.01', '0.1', '0.5'])
    cb.ax.tick_params(labelsize=8)
    return save(fig, out, stem + '_real')


def query_time_panel(summary, out, stem, legend):
    """Mean +/- SD online time per test query; ``summary`` has method, query, mean, std, count."""
    assert (summary['count'] == 5).all()
    fig, ax = chart()
    for m in COLORS:
        d = summary[summary.method == m]
        ax.errorbar(d['query'], d['mean'], yerr=d['std'], color=COLORS[m],
                    marker='s' if m == 'dd' else 'o', ms=3, lw=1, capsize=2, label=NAMES[m])
    ax.set(yscale='log', xlabel='Test query', ylabel='Total online time (s)')
    ax.xaxis.set_major_locator(MaxNLocator(6, integer=True))
    if legend == 'top':
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom, bottom * (top / bottom)**(1 / .78))
        ax.legend(frameon=False, fontsize=7, loc='upper center', ncol=2,
                  columnspacing=1.2, handlelength=2)
    else:
        ax.legend(frameon=False, fontsize=7, loc='center', bbox_to_anchor=(.5, .70),
                  ncol=2, columnspacing=1.2, handlelength=2)
    return save(fig, out, stem)


def per_query_summary(timings):
    """Collapse repeated timings to mean/std/count per method and query."""
    t = timings[timings.method.isin(list(COLORS))]
    return t.groupby(['method', 'query']).total_s.agg(['mean', 'std', 'count']).reset_index()


def sci(x, digits=2):
    mantissa, exponent = f'{x:.{digits}e}'.split('e')
    return rf'\({mantissa}\times10^{{{int(exponent)}}}\)'
