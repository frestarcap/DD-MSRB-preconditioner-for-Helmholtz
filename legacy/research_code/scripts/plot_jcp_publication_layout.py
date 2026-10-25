#!/usr/bin/env python3
"""Regenerate title-free, individually typeset JCP panels from frozen evidence.

Run code/export_jcp_layout_data.m first. No solver is called here; figures are
drawn from numerical arrays, never cropped from an existing publication image.
LaTeX owns panel letters and captions. Inputs and their hashes are recorded.
"""
from pathlib import Path
import hashlib
import io
import json
import os
import tempfile
import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm, ListedColormap, BoundaryNorm
from matplotlib.patches import Rectangle, Ellipse
from matplotlib.collections import LineCollection
from matplotlib.ticker import LogLocator, NullLocator, MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'JCP_CAS/figures/layout_20260925'  # active manuscript (JCP/ is archived)
AUDIT = ROOT / 'outputs/jcp_layout_20260925'
INPUTS = set()
HALFSPACE_TIMINGS = 'outputs/jcp_shifted_20261022/paper_timings.csv'  # SS-ROM reuses its space
# Experiment 1 at omega=53.5: main DD model [44 80 123], retimed with the partition variants.
HALFSPACE_RETIME = 'outputs/halfspace_default_retime_20261024'
WAVEGUIDE_VALIDATION = 'outputs/jcp_revision_experiments_20260914/cloud/analysis/summary.json'
STORAGE_AUDIT = []
COLORS = {'global': '#BF3030', 'dd': '#23794C', 'shifted': '#667085', 'direct': '#3768A0'}
NAMES = {'global': 'Global MSRB', 'dd': 'DD-MSRB', 'shifted': 'Shifted ILU-GMRES', 'direct': 'Sparse direct'}
plt.rcParams.update({'font.family': 'serif', 'font.size': 10, 'pdf.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.linewidth': .7, 'legend.fontsize': 8})


def source(relative):
    p = ROOT / relative
    assert p.is_file(), p
    INPUTS.add(p)
    return p


def csv(relative):
    return pd.read_csv(source(relative))


def js(relative):
    return json.loads(source(relative).read_text())


def save(fig, name):
    # Fixed canvases keep the plotting areas of adjacent field panels equal.
    # Atomic replacement also avoids partially rewritten files on iCloud.
    for directory, ext, dpi in [(OUT, 'pdf', 400), (AUDIT, 'png', 140)]:
        stream = io.BytesIO()
        fig.savefig(stream, format=ext, dpi=dpi)
        with tempfile.NamedTemporaryFile(dir=directory, delete=False) as tmp:
            temporary = Path(tmp.name)
            tmp.write(stream.getvalue())
        os.replace(temporary, directory / (name + '.' + ext))
    plt.close(fig)


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


def storage_panel(global_values, dd_values, categories, stem, layerwise=False, xlabel=None, ylabel=None,
                  saving_over_pair=False):
    """One format for all three experiments; retain negative savings."""
    g, d = np.asarray(global_values, float), np.asarray(dd_values, float)
    assert len(g) == len(d) == len(categories) and np.all(g > 0)
    fig, ax = chart()
    # Keep the same canvas and type scale for all storage panels: LaTeX
    # includes the frequency plot and representative panels at equal widths.
    # Four groups need more separation within each pair for the value labels.
    half_gap = .25 if len(categories) > 3 else .18
    maximum = max(g.max(), d.max())
    for method, values, offset in [('global', g, -half_gap), ('dd', d, half_gap)]:
        bars = ax.bar(np.arange(len(g))+offset, values, .34,
                      color=COLORS[method], label=NAMES[method])
        ax.bar_label(bars, fmt='%.2f', fontsize=8, padding=3)
    for i, saving in enumerate(100*(1-d/g)):
        # Above the DD bar, or above the taller bar of the pair when values differ widely.
        x, y = (i, max(g[i], d[i])+maximum*.08) if saving_over_pair else (i+half_gap, d[i]+maximum*.11)
        ax.text(x, y, f'{saving:.1f}%\nsaved', ha='center', va='bottom', fontsize=8)
        STORAGE_AUDIT.append(dict(panel=stem, category=categories[i],
                                  global_MiB=g[i], dd_MiB=d[i], saved_percent=saving))
    ax.set(xticks=np.arange(len(g)), xticklabels=categories,
           ylabel=ylabel or ('Basis storage (MiB)' if layerwise else 'Persistent storage (MiB)'),
           ylim=(0, maximum*1.4))
    if layerwise or xlabel is not None:
        ax.set_xlabel('Iteration' if layerwise else xlabel)
    ax.grid(axis='x', visible=False)
    ax.legend(frameon=False, loc='upper left')
    save(fig, stem)


def halfspace_frequency_storage():
    training = pd.concat([
        csv('outputs/reviewer_20260909/summary_train.csv'),
        csv('outputs/reviewer_20260909/summary_train_frequency_4.csv')])
    frequencies = [53.5, 80.25, 107, 214]
    values = []
    for method in ['global', 'dd']:
        rows = training[(training.variant == method) &
                        training.study.str.startswith('frequency_')].sort_values('omega')
        assert np.array_equal(rows.omega.to_numpy(), frequencies)
        assert np.allclose(rows.payload_MiB,
                           rows.basis_MiB + rows.operators_MiB + rows.index_MiB)
        values.append(rows.payload_MiB.to_numpy(copy=True))
    retime = csv(HALFSPACE_RETIME + '/summary.csv').set_index('variant')
    assert np.isclose(values[0][0], retime.loc['global', 'payload_MiB'])
    values[1][0] = retime.loc['dd', 'payload_MiB']
    storage_panel(*values, [f'{omega:g}' for omega in frequencies],
                  'halfspace_frequency_storage', xlabel=r'Angular frequency $\omega$')


def field_axes():
    fig = plt.figure(figsize=(4.4, 3.8))
    ax = fig.add_axes([.13, .14, .65, .65 * 4.4 / 3.8])
    cax = fig.add_axes([.81, .14, .028, .65 * 4.4 / 3.8])
    return fig, ax, cax


def fields(u, x, y, stem, source_xy=None, obstacle_ratio=None):
    if obstacle_ratio is not None:
        X, Y = np.meshgrid(x, y)
        a, b = .12*np.sqrt(obstacle_ratio), .12/np.sqrt(obstacle_ratio)
        u = np.ma.masked_where(((X-.5)/a)**2+((Y-.5)/b)**2 < 1, u)
    u = u / np.max(np.abs(u))
    limit = np.max(np.abs(u.real))
    for part in ['real', 'magnitude']:
        fig, ax, cax = field_axes()
        if part == 'real':
            values = u.real
            options = dict(cmap='RdBu_r', norm=SymLogNorm(linthresh=.01, vmin=-limit, vmax=limit))
        else:
            values = np.log10(np.maximum(np.abs(u), 1e-6))
            options = dict(cmap='magma', vmin=-6, vmax=0)
        im = ax.imshow(values, origin='lower', extent=[x[0], x[-1], y[0], y[-1]],
                       interpolation='nearest', rasterized=True, **options)
        ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, ec='.3' if part == 'real' else 'limegreen', lw=.8))
        if obstacle_ratio is None:
            ax.plot([.5, .5], [0, 1], '--', color='.3' if part == 'real' else 'white', lw=.7)
        else:
            ax.add_patch(Ellipse((.5,.5), 2*a, 2*b, facecolor='white', edgecolor='.3', lw=.7))
        if source_xy is not None:
            ax.scatter(*source_xy, s=13, c='white', edgecolors='black', linewidths=.6)
        ax.set(xlabel='$x$', ylabel='$y$', xticks=[0, .5, 1], yticks=[0, .5, 1])
        cb = fig.colorbar(im, cax=cax)
        if part == 'real':
            cb.set_ticks([-.5, -.1, -.01, 0, .01, .1, .5], labels=['−0.5', '−0.1', '−0.01', '0', '0.01', '0.1', '0.5'])
        else:
            cb.set_ticks(np.arange(-6, 1))
        cb.ax.tick_params(labelsize=8)
        save(fig, stem + '_' + part)


def halfspace():
    d = loadmat(source('outputs/jcp_layout_20260925/halfspace_plot_data.mat'))
    x, y = d['x'].ravel(), d['y'].ravel()
    previous = np.loadtxt(source('outputs/jcp_frequency_revision_20260914/halfspace_global_field.csv'), delimiter=',')
    assert np.allclose(d['U'].real, previous, rtol=1e-12, atol=1e-18)
    fields(d['U'], x, y, 'halfspace', (25/126, .5))
    fig, ax, cax = field_axes()
    # Low-saturation blue-grays: block 1 is lightest, block 10 darkest.
    colors = ['#f7f9fb', '#e3e8ed', '#ced6de', '#b8c4ce', '#9fadb9',
              '#8593a1', '#6b7784', '#525b65', '#393f46', '#20242a']
    im = ax.imshow(d['labels'], origin='lower', extent=[x[0], x[-1], y[0], y[-1]],
                   cmap=ListedColormap(colors), norm=BoundaryNorm(np.arange(.5, 11), 10), interpolation='nearest')
    ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, ec='black', lw=.8))
    ax.plot([.5, .5], [0, 1], 'k--', lw=.8)
    ax.set(xlabel='$x$', ylabel='$y$', xticks=[0, .5, 1], yticks=[0, .5, 1])
    fig.colorbar(im, cax=cax, ticks=np.arange(1, 11))
    save(fig, 'halfspace_partition')
    fig, ax = chart()
    for j, spectrum in enumerate(d['spectra'].ravel(), 1):
        v = spectrum.ravel()
        ax.semilogy(np.arange(1, len(v)+1), np.maximum(v, 1e-16), '-o', ms=3, lw=1.2, label=f'Iteration {j}')
    ax.set(xlabel='Mode index', ylabel='Singular value')
    ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
    ax.yaxis.set_major_locator(LogLocator(numticks=6))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.legend(frameon=False)
    save(fig, 'halfspace_spectra')
    d = loadmat(source('outputs/halfspace_figures_20260910/omega214_field.mat'))
    assert d['rho'].item() <= 1e-10
    fields(d['U'], d['x'].ravel(), d['y'].ravel(), 'omega214', d['source'].ravel())
    # Reuse the established residual renderer, with the same frozen cohort.
    import write_jcp_shifted_paper as shifted
    source('outputs/jcp_shifted_20261022/paper_histories.csv')
    source('outputs/jcp_shifted_20261022/paper_plain_gmres.csv')
    fig, ax = chart()
    shifted.residual(ax, 'halfspace', 10)
    save(fig, 'halfspace_residuals')
    s = csv('outputs/halfspace_dd_paper/layer_storage.csv')
    storage_panel(s.global_entries*16/2**20, s.dd_entries*16/2**20,
                  ['1','2','3'], 'halfspace_layer_storage', layerwise=True)
    totals = csv('outputs/halfspace_dd_paper/main_results.csv').set_index('variant')
    measured = csv('outputs/paper_memory_benchmark/summary.csv')
    measured = measured[measured.problem=='halfspace'].set_index('method')
    values = []
    for method, variant in [('global','global'),('dd','slabs1-aligned-sym8')]:
        r = totals.loc[variant]
        assert np.isclose(r.basis_bytes, s[method+'_entries'].sum()*16)
        assert r.rom_payload_bytes == r.basis_bytes+r.operator_bytes+r.partition_bytes
        values.append([r.basis_bytes/2**20, r.rom_payload_bytes/2**20, measured.loc[method,'model_MiB']])
    storage_panel(*values, ['Basis','Payload','Object'], 'halfspace_storage_total')


def ecc():
    prefix = 'outputs/ecc_dd_budget_20260919/'
    x = np.loadtxt(source(prefix+'paper_x.csv'), delimiter=',')
    y = np.loadtxt(source(prefix+'paper_y.csv'), delimiter=',')
    d = loadmat(source('outputs/jcp_layout_20260925/ecc_plot_data.mat'))
    assert np.allclose(d['x'].ravel(), x) and np.allclose(d['y'].ravel(), y)
    previous = np.loadtxt(source(prefix+'paper_dd_field.csv'), delimiter=',')
    assert np.allclose(d['U'].real, previous, rtol=1e-12, atol=1e-18)
    assert d['residual'].item() <= 1e-12
    fields(d['U'], x, y, 'ecc_field', obstacle_ratio=d['parameter'].item())
    X, Y = np.meshgrid(x, y)
    physical = (X > 0) & (X < 1) & (Y > 0) & (Y < 1)
    region = np.full(X.shape, 3)
    region[physical] = 2
    region[physical & (np.hypot(X-.5, Y-.5) <= .24)] = 1
    fig, ax, cax = field_axes()
    im = ax.imshow(region, origin='lower', extent=[x[0], x[-1], y[0], y[-1]],
                   cmap=ListedColormap(['#c8deed', '#f4f0e8', '#c8c8c8']),
                   norm=BoundaryNorm([.5, 1.5, 2.5, 3.5], 3), interpolation='nearest')
    ax.set(xlabel='$x$', ylabel='$y$', xticks=[0, .5, 1], yticks=[0, .5, 1])
    fig.colorbar(im, cax=cax, ticks=[1, 2, 3])
    save(fig, 'ecc_partition')
    s = csv(prefix+'paper_spectra.csv')
    fig = plt.figure(figsize=(4.4, 3.8))
    ax = fig.add_axes([.18, .14, .65, .65 * 4.4 / 3.8])
    ax.grid(which='major', alpha=.18)
    for j, k in zip([1, 2, 3], [25, 42, 42]):
        t = s[s.layer == j]
        line, = ax.semilogy(t['rank'], t['tail'].clip(lower=1e-16), label=f'Iteration {j}')
        ax.plot(k, t.loc[t['rank'] == k, 'tail'].iloc[0], 'o', color=line.get_color())
    ax.axhline(1e-5, color='.4', ls='--', lw=.8)
    ax.set(xlabel='Retained rank', ylabel='Relative Frobenius tail', ylim=(1e-10, 2))
    ax.yaxis.set_minor_locator(NullLocator()); ax.legend(frameon=False)
    save(fig, 'ecc_spectra')
    audit = csv(prefix+'uniform5_tail_audit.csv')
    storage = csv(prefix+'uniform5_storage.csv').set_index('method')
    layers = [audit.loc[audit.tag=='uniform5_'+m, 'basis_entries'].to_numpy()*16/2**20
              for m in ['global','dd']]
    totals = [storage.loc['uniform5_'+m, ['basis_MiB','payload_MiB','MATLAB_object_MiB']].to_numpy(dtype=float)
              for m in ['global','dd']]
    assert all(np.isclose(v.sum(), t[0]) for v, t in zip(layers, totals))
    storage_panel(*layers, ['1','2','3'], 'ecc_storage_layers', layerwise=True)
    storage_panel(*totals, ['Basis','Payload','Object'], 'ecc_storage_total')


def waveguide():
    reports = []
    for omega in [24, 36, 48, 60]:
        folder = ('remote_recovery_w60' if omega == 60 else f'remote_v7/full_w{omega}_p8_t4_b2_sr40_or40_sm40_om40_c12000')
        d = js(f'outputs/scnet_3d_20260909/{folder}/report.json')
        assert d['all_converged']
        reports.append(d)
    d = reports[2]
    q = next(q for q in d['tests'] if q['mu'] == .317)
    norm = SymLogNorm(linthresh=.02, vmin=-1, vmax=1)
    cmap = plt.get_cmap('RdBu_r')
    for report in reports:
        s = report['field_slice']; v = np.asarray(s['real'])
        assert v.shape == (len(s['x']), len(s['y'])) and s['mu'] == .317 and s['z'] == .5
        xx, yy = np.meshgrid(s['x'], s['y'], indexing='ij')
        fig = plt.figure(figsize=(7.2, 2.8))
        ax = fig.add_axes([-.04, .08, .90, .86], projection='3d', computed_zorder=False)
        ax.plot_surface(xx, yy, np.full_like(xx, .5), facecolors=cmap(norm(v/np.max(abs(v)))),
                        rstride=1, cstride=1, shade=False, antialiased=False, rasterized=True, zorder=2, clip_on=False)
        for z in [0, 1]:
            ax.plot([0, 3, 3, 0, 0], [0, 0, 1, 1, 0], [z]*5, color='.4', lw=.65, zorder=3, clip_on=False)
        for x0 in [0, 3]:
            for y0 in [0, 1]: ax.plot([x0]*2, [y0]*2, [0, 1], color='.4', lw=.65, zorder=3, clip_on=False)
        ax.scatter([.2], [.5], [.5], marker='*', s=35, color='black', depthshade=False, zorder=4)
        ax.set(xlabel='$x$', ylabel='$y$', zlabel='$z$', zlim=(0, 1), xticks=[0, 1, 2, 3], yticks=[0, 1], zticks=[0, .5, 1])
        ax.set_box_aspect((3.35, 1.35, 1), zoom=1.65); ax.view_init(elev=22, azim=-75)
        ax.tick_params(pad=0, labelsize=9)
        ax.xaxis.labelpad=0; ax.yaxis.labelpad=0; ax.zaxis.labelpad=0
        ax.grid(False)
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]: axis.pane.fill=False; axis.pane.set_edgecolor('white')
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=fig.add_axes([.87, .20, .018, .60]), ticks=[-1, -.1, 0, .1, 1])
        cb.ax.tick_params(labelsize=8)
        save(fig, f'waveguide_field_{int(report["omega"])}')
    # ROM steps (from the zero initial guess) followed by SL steps, as in the 2D panels.
    # Steps 0-20 keep full scale; later steps are compressed so the whole history fits.
    histories = {}
    for m in ['shifted', 'global', 'dd']:
        r=q[m]; sl=np.asarray(r['history'])
        assert len(sl)==r['iterations']+1 and r['true_residual'] <= 1e-6
        if m == 'shifted':
            histories[m] = sl
        else:
            rom = np.asarray(r['rom_history']); assert rom[0] == 1 and np.isclose(rom[-1], sl[0])
            histories[m] = np.concatenate([rom, sl[1:]])
    labels = {m: 'Shifted MG-GMRES' if m=='shifted' else NAMES[m] for m in histories}
    fold, squeeze = 20, 13.5
    forward = lambda t: np.where(t <= fold, t, fold + (t - fold) / squeeze)
    inverse = lambda u: np.where(u <= fold, u, fold + (u - fold) * squeeze)
    fig, ax = chart()
    ax.set_xscale('function', functions=(forward, inverse))
    ax.axvspan(fold, 352, color='.94', lw=0, zorder=0)
    for m, history in histories.items():
        r = q[m]; steps = np.arange(len(history))
        marks = [i for i in steps if i <= fold or (i - fold) % 10 == 0]
        ax.semilogy(steps, history, color=COLORS[m], lw=1.1, marker='o', ms=2.5, markevery=marks, label=labels[m])
        ax.plot(steps[-1], r['true_residual'], 'o', color=COLORS[m], ms=4.5)
        ax.annotate(str(r['iterations']), (steps[-1], r['true_residual']), xytext=(0, -13),
                    textcoords='offset points', ha='center', fontsize=8)
    ax.axvline(3, color='.4', ls=':', lw=.9); ax.axhline(1e-6, color='.4', ls='--', lw=.8)
    ax.text(2.6, 2.2e-6, 'ROM', ha='right', fontsize=8); ax.text(3.5, 2.2e-6, 'SL', ha='left', fontsize=8)
    ax.text(200, 2.6e-6, 'compressed scale', ha='center', fontsize=7, color='.45')
    ax.set(xlabel='ROM steps + SL steps', ylabel='Relative residual', xlim=(-.5, 352), ylim=(4e-7, 2),
           xticks=[0, 3, 10, 20, 100, 200, 300])
    ax.yaxis.set_minor_locator(NullLocator()); ax.legend(frameon=False, loc='upper right')
    save(fig,'waveguide_residuals')
    objects = js('outputs/waveguide_whos_20260913/results/w48_whos.json')
    assert objects['validated']
    objects = {r['method']: r for r in objects['rows']}
    layer_values, total_values = [], []
    for method in ['global','dd']:
        r = objects[method]
        layer_values.append(np.array([layer['basis_bytes'] for layer in r['layers']])/2**20)
        assert np.isclose(layer_values[-1].sum(), d[method+'_basis_MiB'])
        assert r['numeric_bytes_one_shared_map'] == r['formula_persistent_bytes']
        assert np.isclose(r['formula_persistent_bytes']/2**20, d[method+'_persistent_MiB'])
        total_values.append([r['basis_whos_bytes']/2**20,
                             r['formula_persistent_bytes']/2**20, r['model_MiB']])
    storage_panel(*layer_values, ['1','2','3'], 'waveguide_layer_storage', layerwise=True)
    storage_panel(*total_values, ['Basis','Payload','Object'], 'waveguide_storage_total')
    rows=csv('outputs/waveguide_frequency_final_20260913/waveguide_frequency.csv')
    assert np.allclose(100*(1-rows.dd_MiB/rows.global_MiB), rows.saving_pct)
    storage_panel(rows.global_MiB, rows.dd_MiB, ['24','36','48','60'], 'waveguide_frequency_storage',
                  xlabel=r'Angular frequency $\omega$', ylabel='Payload (MiB)', saving_over_pair=True)
    # At omega=48 the online times are the repeated timings of the ten-query study.
    repeated = {q['parameter']: q['methods'] for q in js(WAVEGUIDE_VALIDATION)['per_query']
                if q['query_kind'] == 'original_repeated'}
    key = {'global': 'global3', 'dd': 'dd3', 'shifted': 'shifted_mg'}
    def online(i, test, m):
        return repeated[test['mu']][key[m]]['mean_total_s'] if i == 2 else test[m]['seconds']
    x=np.arange(4)
    for mode in ['time','ratio']:
        fig,ax=chart()
        if mode=='time':
            for m,off in [('global',-.055),('dd',.055),('shifted',0)]:
                v=np.array([[online(i,q,m) for q in d['tests']] for i,d in enumerate(reports)]);avg=v.mean(axis=1)
                ax.errorbar(x+off,avg,yerr=[avg-v.min(axis=1),v.max(axis=1)-avg],color=COLORS[m],marker='o',capsize=3,label='Shifted MG-GMRES' if m=='shifted' else NAMES[m])
            ax.set(yscale='log',ylabel='Total online time (s)');ax.legend(frameon=False,loc='lower right')
        else:
            for mu in [.237,.317,.483]:
                v=[next(q for q in d['tests'] if q['mu']==mu) for d in reports]
                ax.plot(x,[online(i,q,'global')/online(i,q,'dd') for i,q in enumerate(v)],'-o',label=rf'$\mu={mu:.3f}$')
            ax.axhline(1,color='.4',ls='--',lw=.8);ax.set(ylabel=r'$T_{\mathrm{Global}}/T_{\mathrm{DD}}$',ylim=(0,2.05));ax.legend(frameon=False,loc='lower right')
            ratio_note(ax)
        ax.set(xlabel=r'Angular frequency $\omega$',xticks=x,xticklabels=['24','36','48','60'])
        save(fig,'waveguide_frequency_'+mode)


def residual_slices():
    d=js('outputs/dd_highfreq_mechanism_20260913/results/interfaces_44244813/w48/slices/slices.json')
    assert d['validated'] and d['omega']==48
    records={r['variant']:r for r in d['records']};g=d['geometry']
    x=np.asarray(g['x']).ravel();y=np.asarray(g['y']).ravel();h=np.asarray(g['spacing']).ravel()
    labels=np.asarray(g['block_labels']);n=np.asarray(g['physical_grid_size']).ravel();p=np.asarray(g['pml_cells']).ravel()
    physical=(np.arange(len(x))[:,None]>=p[0])&(np.arange(len(x))[:,None]<p[0]+n[0])&(np.arange(len(y))[None,:]>=p[1])&(np.arange(len(y))[None,:]<p[1]+n[1])
    segments=[]
    for i,j in zip(*np.where((labels[1:]!=labels[:-1])&physical[1:]&physical[:-1])):
        xx=(x[i]+x[i+1])/2;segments.append([(xx,y[j]-h[1]/2),(xx,y[j]+h[1]/2)])
    for i,j in zip(*np.where((labels[:,1:]!=labels[:,:-1])&physical[:,1:]&physical[:,:-1])):
        yy=(y[j]+y[j+1])/2;segments.append([(x[i]-h[0]/2,yy),(x[i]+h[0]/2,yy)])
    for variant,name in [('GGG','global'),('DDD','dd')]:
        fig,axs=plt.subplots(2,1,figsize=(4.4,3.8))
        fig.subplots_adjust(left=.12,right=.78,bottom=.14,top=.96,hspace=.46)
        for ax,field in zip(axs,['r0','rafter']):
            values=np.asarray(records[variant]['fields'][field]['normalized_abs_slice'])/np.sqrt(np.prod(h))
            assert values.shape==(len(x),len(y))
            im=ax.imshow(np.log10(np.maximum(values,1e-5)).T,origin='lower',extent=[x[0]-h[0]/2,x[-1]+h[0]/2,y[0]-h[1]/2,y[-1]+h[1]/2],cmap='magma',vmin=-4,vmax=1,interpolation='nearest',rasterized=True)
            ax.add_collection(LineCollection(segments,colors='white',linewidths=.25,alpha=.35))
            ax.add_patch(Rectangle((0,0),3,1,fill=False,ec='#bdbdbd',ls='--',lw=.65))
            ax.set(xticks=[0,1,2,3],yticks=[0,.5,1],ylabel='$y$')
        axs[1].set_xlabel('$x$')
        cb=fig.colorbar(im,cax=fig.add_axes([.82,.20,.025,.64]),ticks=[-4,-3,-2,-1,0,1]);cb.ax.tick_params(labelsize=8)
        save(fig,'residual_slices_'+name)


def timing_time_panel(data, stem):
    assert (data['count']==5).all()
    fig,ax=chart()
    for m in COLORS:
        d=data[data.method==m]
        ax.errorbar(d['query'],d['mean'],yerr=d['std'],color=COLORS[m],marker='s' if m=='dd' else 'o',ms=3,lw=1,capsize=2,label=NAMES[m])
    ax.set(yscale='log',xlabel='Test query',ylabel='Total online time (s)')
    ax.xaxis.set_major_locator(MaxNLocator(6,integer=True))
    if stem == 'ecc_time':
        # Reserve the upper band for the legend, above all curves/error bars,
        # while retaining the canvas and square axes used by the paired panel.
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom, bottom * (top / bottom)**(1 / .78))
        ax.legend(frameon=False, fontsize=7, loc='upper center', ncol=2,
                  columnspacing=1.2, handlelength=2)
    elif stem == 'halfspace_representative_time':
        # Two-column legend in the empty band between sparse direct and shifted ILU.
        ax.legend(frameon=False, fontsize=7, loc='center', bbox_to_anchor=(.5, .70),
                  ncol=2, columnspacing=1.2, handlelength=2)
    else:
        ax.legend(frameon=False, fontsize=7)
    save(fig,stem+'_time')


def timing_panels():
    for stem in ['halfspace_representative_time', 'ecc_time', 'halfspace_two_parameter_time']:
        if stem == 'halfspace_representative_time':
            # Re-timed half-space cohort; same per-query summary as plot_2d_paired_times.py.
            t=csv(HALFSPACE_TIMINGS); t=t[(t.problem=='halfspace') & t.method.isin(list(COLORS))]
            data=t.groupby(['method','query']).total_s.agg(['mean','std','count']).reset_index()
        else:
            data=csv('outputs/paired_times_20260921/'+stem+'_query_summary.csv')
        timing_time_panel(data, stem)
        g=data[data.method=='global'].set_index('query')['mean'];d=data[data.method=='dd'].set_index('query')['mean']
        fig,ax=chart();ratio=g/d
        ax.plot(ratio.index,ratio,'o-',ms=3,color='#75529A');ax.axhline(1,color='.4',ls='--',lw=.8)
        ax.set(xlabel='Test query',ylabel=r'$T_{\mathrm{Global}}/T_{\mathrm{DD}}$');ax.margins(y=.18)
        ax.xaxis.set_major_locator(MaxNLocator(6,integer=True))
        ratio_note(ax)
        save(fig,stem+'_ratio')
    data=csv('outputs/paired_times_20260921/halfspace_frequency_query_summary.csv')
    retimed=pd.concat([csv(f'{HALFSPACE_RETIME}/{m}_timing.csv') for m in COLORS])
    retimed=retimed.groupby(['variant','c_right']).total_s.agg(['mean','count']).reset_index().assign(scale=1.0)
    assert (retimed['count']==5).all()
    data=pd.concat([data[data.scale!=1.0],retimed[data.columns]])
    fig,ax=chart()
    for m in COLORS:
        t=data[data.variant==m].pivot(index='scale',columns='c_right',values='mean');avg=t.mean(axis=1)
        ax.errorbar(t.index.to_numpy()*53.5,avg,yerr=[avg-t.min(axis=1),t.max(axis=1)-avg],color=COLORS[m],marker='o',capsize=3,label=NAMES[m])
    ax.set(yscale='log',xlabel=r'Angular frequency $\omega$',ylabel='Total online time (s)',xticks=[53.5,80.25,107,214],xticklabels=['53.5','80.25','107','214']);ax.legend(frameon=False,fontsize=7)
    save(fig,'halfspace_frequency_time')
    g=data[data.variant=='global'].pivot(index='scale',columns='c_right',values='mean');d=data[data.variant=='dd'].pivot(index='scale',columns='c_right',values='mean')
    fig,ax=chart()
    for speed in g.columns:ax.plot(g.index.to_numpy()*53.5,g[speed]/d[speed],'-o',ms=3,label=f'{speed:.4f}')
    ax.axhline(1,color='.4',ls='--',lw=.8);ax.set(xlabel=r'Angular frequency $\omega$',ylabel=r'$T_{\mathrm{Global}}/T_{\mathrm{DD}}$',xticks=[53.5,80.25,107,214],xticklabels=['53.5','80.25','107','214']);ax.legend(title=r'$c_{\rm right}$',frameon=False,fontsize=7,loc='lower right');ax.margins(y=.2)
    ratio_note(ax)
    save(fig,'halfspace_frequency_ratio')
    halfspace_frequency_storage()
    selection=js('outputs/reviewer_20260909/selection_before_holdout.json')
    fig,ax=chart()
    ax.add_patch(Rectangle((-.25,-.25),1.5,1.5,fc='#e4e4e4',ec='black',lw=.8))
    for x0,col in [(0,'#dae7f5'),(.5,'#fae5d4')]:ax.add_patch(Rectangle((x0,0),.5,1,fc=col,ec='black',lw=.8))
    ax.plot([.5,.5],[0,1],'k--',lw=.8);ax.scatter(.2,.5,s=15,c='black');ax.text(.2,.4,'source',ha='center',fontsize=9)
    ax.text(.25,.8,r'$c_{\rm left}$',ha='center');ax.text(.75,.8,r'$c_{\rm right}$',ha='center');ax.text(.5,1.12,'PML',ha='center')
    ax.set(xlim=(-.3,1.3),ylim=(-.3,1.3),aspect='equal',xlabel='$x$',ylabel='$y$',xticks=[0,.5,1],yticks=[0,.5,1]);ax.grid(False)
    save(fig,'two_parameter_geometry')
    fig,ax=chart();a,b=np.meshgrid(np.linspace(.95,1.05,8),np.linspace(.8,.95,8))
    ax.scatter(a,b,s=15,facecolors='white',edgecolors='.5',label=r'$8\times8$ training')
    a,b=np.meshgrid(selection['holdout_left'],selection['holdout_right']);ax.scatter(a,b,s=32,c='#6b3a8a',marker='x',label='Validation')
    ax.set(xlabel=r'$c_{\rm left}$',ylabel=r'$c_{\rm right}$',xlim=(.943,1.057),ylim=(.791,.959))
    ax.legend(loc='upper center',bbox_to_anchor=(.5,1.16),ncol=2,frameon=False,fontsize=8)
    save(fig,'two_parameter_samples')


def validation_panels():
    from write_jcp_3d_validation import validate
    d=js(WAVEGUIDE_VALIDATION);validate(d)
    queries=sorted(d['per_query'],key=lambda q:q['parameter'])
    styles={'global3':('#2365A0','o',r'Global ($k=3$)'), 'global2':('#C27816','s',r'Global ($k=2$)'), 'dd3':('#238562','^',r'DD ($k=3$)'), 'shifted_mg':('#414141','D','Shifted MG-GMRES')}
    for mode in ['time','ratio']:
        fig,ax=chart()
        for method in (styles if mode=='time' else ['global3','global2']):
            color,marker,label=styles[method]
            xs=[q['parameter'] for q in queries]
            ys=[q['methods'][method]['mean_total_s']/(q['methods']['dd3']['mean_total_s'] if mode=='ratio' else 1) for q in queries]
            ax.plot(xs,ys,color=color,lw=.9,label=label)
            for q,x,y in zip(queries,xs,ys):
                filled=q['query_kind']=='original_repeated'
                ax.plot(x,y,marker=marker,ms=4,mfc=color if filled else 'white',mec=color)
                sd=q['methods'][method]['sample_sd_total_s']
                if mode=='time' and sd is not None:ax.errorbar(x,y,yerr=sd,fmt='none',ecolor=color,capsize=2)
        if mode=='ratio':ax.axhline(1,color='.5',ls='--',lw=.8)
        ax.set(xlabel=r'Query parameter $\mu$',ylabel='Total online time (s)' if mode=='time' else r'$T_{\mathrm{Global}}/T_{\mathrm{DD3}}$',xlim=(.2,.59))
        ax.legend(frameon=False,fontsize=7,loc='lower right' if mode=='ratio' else 'best')
        if mode=='ratio':
            ax.margins(y=.18)
            ratio_note(ax)
        save(fig,'waveguide_validation_'+mode)


def residual_diagnostics():
    rows=csv('outputs/dd_highfreq_mechanism_20260913/analysis/20260913_220451_151478/final_layer_components.csv')
    ws=[24,36,48,60]
    for mode in ['roughness','response']:
        fig,ax=chart()
        for m in ['global','dd']:
            if mode=='roughness':
                groups=[rows[(rows.method==m)&(rows.omega==w)].residual_roughness_transverse_over_omega2 for w in ws]
                assert all(len(g)==3 for g in groups)
                ax.plot(ws,[g.mean() for g in groups],'-o',ms=4,color=COLORS[m],label=NAMES[m])
                for w,g in zip(ws,groups):ax.scatter([w]*len(g),g,s=10,alpha=.4,color=COLORS[m])
            else:
                for comp,style in [('pre','-o'),('coarse',':^'),('combined','--s')]:
                    vals=[rows[(rows.method==m)&(rows.omega==w)]['mg_'+comp+'_optimal_contraction'].mean() for w in ws]
                    ax.plot(ws,vals,style,ms=4,color=COLORS[m],label=f'{m}: {comp if comp!="combined" else "full cycle"}')
        ax.set(xlabel=r'$\omega$',xticks=ws,ylabel=r'$\|\nabla_\perp r\|^2/(\omega^2\|r\|^2)$' if mode=='roughness' else r'$\kappa_i$ (smaller is better)')
        if mode=='response':ax.set_ylim(0,1.05)
        ax.legend(frameon=False,fontsize=7,ncol=1 if mode=='roughness' else 2)
        save(fig,'residual_'+mode)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--halfspace-frequency-storage-only', action='store_true',
                        help='Redraw only the half-space frequency payload bars.')
    parser.add_argument('--ecc-time-only', action='store_true',
                        help='Redraw only the ECC online-time panel.')
    args = parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);AUDIT.mkdir(parents=True,exist_ok=True)
    if args.ecc_time_only:
        timing_time_panel(csv('outputs/paired_times_20260921/ecc_time_query_summary.csv'),
                          'ecc_time')
        print('ECC online-time panel regenerated from the frozen timing summary.')
        return
    if args.halfspace_frequency_storage_only:
        halfspace_frequency_storage()
        pd.DataFrame(STORAGE_AUDIT).to_csv(AUDIT/'halfspace_frequency_storage_values.csv',index=False)
        print('Half-space frequency payload bars regenerated from the frozen training summaries.')
        return
    halfspace();ecc();waveguide();residual_slices()
    timing_panels();validation_panels();residual_diagnostics()
    manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(INPUTS)}
    (AUDIT/'plot_sources.json').write_text(json.dumps(manifest,indent=2)+'\n')
    pd.DataFrame(STORAGE_AUDIT).to_csv(AUDIT/'storage_panel_values.csv',index=False)
    print(f'JCP publication panels regenerated from {len(INPUTS)} frozen inputs.')


if __name__=='__main__':main()
