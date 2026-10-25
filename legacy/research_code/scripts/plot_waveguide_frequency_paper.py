"""Publication figures: completed four-frequency cohort and disclosed recovery settings."""
from pathlib import Path
import csv
from paper_storage_tables import representative_storage, waveguide_frequency
import hashlib
import json
import tempfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.cm import ScalarMappable

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/waveguide_frequency_final_20260913'
FIG = ROOT / 'LaTeX_DL_468198_240419/figures/waveguide_frequency'
DATA = ROOT / 'outputs/scnet_3d_20260909'


def save_figure(fig, path, **kwargs):
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=path.suffix, delete=False) as f:
        temporary = Path(f.name)
    try:
        fig.savefig(temporary, **kwargs)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    audit = json.loads((DATA / 'verification.json').read_text())
    assert audit['uniform_complete'] and not audit['invalid_complete_reports']
    accepted = {r['job_id'] for r in audit['cases'] if r['accepted']}
    profile = json.loads((DATA / 'profiles/beta2_coarse12000_restart40.json').read_text())
    whos_path = ROOT / 'outputs/waveguide_whos_20260913/summary.json'
    whos = json.loads(whos_path.read_text())
    assert whos['validated'] and whos['matlab_version'].endswith('(R2024a)')
    for filename, expected in whos['source_sha256'].items():
        assert hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() == expected
    whos_rows = {r['omega']: r for r in whos['rows']}

    reports = []
    rows = []
    hashes = {str(whos_path.relative_to(ROOT)): hashlib.sha256(whos_path.read_bytes()).hexdigest()}
    for omega in [24, 36, 48, 60]:
        folder = DATA / f'remote_v7/full_w{omega}_p8_t4_b2_sr40_or40_sm40_om40_c12000'
        if omega == 60:
            folder = DATA / 'remote_recovery_w60'
        path = folder / 'report.json'
        d = json.loads(path.read_text())
        if omega < 60:
            assert str(d['metadata']['job_id']) in accepted
            assert d['metadata']['node'] == 'g02r1n09'
            assert d['solver_config'] == profile and not d['training_resumed']
        else:
            expected = dict(profile, snapshot_maxit=120)
            assert d['metadata']['job_id'] == '44057113'
            assert d['solver_config'] == expected and d['training_resumed']
        assert d['all_converged'] and d['snapshot_max_relres'] <= 1e-7
        assert [q['mu'] for q in d['tests']] == [.237, .317, .483]
        for q in d['tests']:
            for m in ['global', 'dd', 'shifted']:
                assert q[m]['flag'] == 0 and q[m]['true_residual'] <= 1e-6
        config = json.loads((folder / 'configuration.json').read_text())['cfg']
        assert config['layer_tail_tolerances'] == [.003, .008, .02]
        assert config['num_layers'] == 3 and config['train_count'] == 8
        assert config['dd_rank_strategy'] == 'marginal_storage'
        with next(folder.glob('helmholtz_3d_candidates_*.csv')).open() as f:
            csvrows = list(csv.DictReader(f))
        d['offline'] = {m: float(next(r['offline_s'] for r in csvrows if r['method'] == m)) for m in ['global', 'dd']}
        row = {'omega': omega, 'N': int(d['N']), 'global_MiB': d['global_persistent_MiB'], 'dd_MiB': d['dd_persistent_MiB'],
               'saving_pct': 100 * (1 - d['dd_persistent_MiB'] / d['global_persistent_MiB'])}
        for m in ['global', 'dd', 'shifted']:
            row[m + '_online_s'] = np.mean([q[m]['seconds'] for q in d['tests']])
            row[m + '_completion_iters'] = np.mean([q[m]['iterations'] for q in d['tests']])
            row[m + '_max_residual'] = max(q[m]['true_residual'] for q in d['tests'])
        row.update({m + '_offline_s': d['offline'][m] if omega < 60 else None for m in ['global', 'dd']})
        measured = whos_rows[omega]
        for m in ['global', 'dd']:
            assert abs(measured[m + '_formula_MiB'] - row[m + '_MiB']) < 1e-8
            row[m + '_whos_MiB'] = measured[m + '_whos_MiB']
        row['whos_saving_pct'] = measured['whos_saving_pct']

        rows.append(row)
        reports.append(d)
        for source in [path, folder / 'configuration.json', next(folder.glob('helmholtz_3d_candidates_*.csv'))]:
            hashes[str(source.relative_to(ROOT))] = hashlib.sha256(source.read_bytes()).hexdigest()
    with (OUT / 'waveguide_frequency.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (OUT / 'waveguide_frequency_sources.json').write_text(json.dumps(hashes, indent=2) + '\n')
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    colors = {'global': '#BF3030', 'dd': '#23794C', 'shifted': '#667085'}
    labels = {'global': 'Global TA-ROM', 'dd': 'DD-TA-ROM', 'shifted': 'Shifted MG-GMRES'}
    layer_diagnostics = []
    for d in reports:
        for q in d['tests']:
            entry = {'omega': d['omega'], 'mu': q['mu']}
            for m in ['global', 'dd']:
                h = q[m]['rom_history']
                assert len(h) == 4
                entry[m + '_rom_history'] = h
                entry[m + '_third_layer_factor'] = h[2] / h[3]
                entry[m + '_completion_iterations'] = q[m]['iterations']
            layer_diagnostics.append(entry)
    (OUT / 'layer_diagnostics.json').write_text(json.dumps(layer_diagnostics, indent=2) + '\n')
    fig = plt.figure(figsize=(9, 6.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, .88])
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])]
    x = np.arange(4)
    for m, offset in [('global', -.18), ('dd', .18)]:
        bars = axes[0].bar(x + offset, [r[m + '_whos_MiB'] for r in rows], .34, color=colors[m], label=labels[m])
    for i, r in enumerate(rows):
        axes[0].text(i, r['global_whos_MiB'] + 100, f"{r['whos_saving_pct']:.1f}%\nsaved", ha='center', va='bottom', fontsize=10)
    axes[0].set(ylabel='Deployed ROM object (MiB)', ylim=(0, 5300), title=r'(a) MATLAB object size', xticks=x, xticklabels=['24', '36', '48', '60'])
    axes[0].legend(frameon=False, fontsize=9, loc='upper left')
    for m, offset in [('global', -.055), ('dd', .055), ('shifted', 0)]:
        v = np.array([[q[m]['seconds'] for q in d['tests']] for d in reports])
        avg = v.mean(axis=1)
        axes[1].errorbar(x + offset, avg, yerr=[avg - v.min(axis=1), v.max(axis=1) - avg], color=colors[m], marker='o', capsize=4, label=labels[m])
    axes[1].set(yscale='log', ylabel='Total online time (s)', title='(b) Three-query mean and range', xticks=x, xticklabels=['24', '36', '48', '60'])
    axes[1].legend(frameon=False, fontsize=8, loc='lower right')
    query_ratios = []
    for mu, color, marker in [(.237, '#3768A0', 'o'), (.317, '#C47A21', 's'), (.483, '#75529A', '^')]:
        values = []
        for d in reports:
            q = next(q for q in d['tests'] if q['mu'] == mu)
            ratio = q['global']['seconds'] / q['dd']['seconds']
            values.append(ratio)
            query_ratios.append(dict(omega=d['omega'], mu=mu, global_over_dd=ratio,
                                     global_s=q['global']['seconds'], dd_s=q['dd']['seconds']))
        axes[2].plot(x, values, color=color, marker=marker, lw=1.6, label=rf'$\mu={mu:.3f}$')
    axes[2].axhline(1, color='.3', ls='--', lw=1)
    axes[2].set(ylabel=r'$T_{\mathrm{Global}}/T_{\mathrm{DD}}$', ylim=(0, 2.05),
                title='(c) Paired online time ratios: all three queries', xticks=x, xticklabels=['24', '36', '48', '60'])
    axes[2].text(.99, .96, 'Above 1: DD faster', ha='right', va='top', transform=axes[2].transAxes, fontsize=9)
    axes[2].legend(frameon=False, fontsize=9, loc='lower right', ncol=3)
    (OUT / 'query_time_ratios.json').write_text(json.dumps(query_ratios, indent=2) + '\n')
    for ax in axes:
        ax.set_xlabel(r'Angular frequency $\omega$')
        ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    for ext in ['pdf', 'png']:
        save_figure(fig, FIG / f'frequency_storage_time.{ext}', dpi=220)
    plt.close(fig)

    # Perspective views of the exported central plane, not a volume interpolation.
    norm = SymLogNorm(linthresh=.02, vmin=-1, vmax=1, base=10)
    cmap = plt.get_cmap('RdBu_r')
    fig = plt.figure(figsize=(9, 10.5))
    for i, d in enumerate(reports):
        s = d['field_slice']; real = np.asarray(s['real'])
        assert real.shape == (len(s['x']), len(s['y']))
        assert s['mu'] == .317 and s['z'] == .5
        xx, yy = np.meshgrid(s['x'], s['y'], indexing='ij')
        ax = fig.add_axes([.02, .75 - .245 * i, .78, .245], projection='3d', computed_zorder=False)
        ax.plot_surface(xx, yy, np.full_like(xx, s['z']),
                        facecolors=cmap(norm(real / np.max(np.abs(real)))),
                        rstride=1, cstride=1, shade=False, antialiased=False,
                        rasterized=True, zorder=2, clip_on=False)
        for z in (0, 1):
            ax.plot([0, 3, 3, 0, 0], [0, 0, 1, 1, 0], [z]*5, color='.4', lw=.65, zorder=3, clip_on=False)
        for x0 in (0, 3):
            for y0 in (0, 1):
                ax.plot([x0, x0], [y0, y0], [0, 1], color='.4', lw=.65, zorder=3, clip_on=False)
        ax.scatter([.2], [.5], [.5], marker='*', s=40, color='black', depthshade=False, zorder=4)
        ax.set(xlabel='$x$', ylabel='$y$', zlabel='$z$', zlim=(0, 1),
               xticks=[0, 1, 2, 3], yticks=[0, 1], zticks=[0, .5, 1])
        ax.set_box_aspect((3.35, 1.35, 1), zoom=1.55)
        ax.view_init(elev=22, azim=-75)
        ax.tick_params(pad=0, labelsize=9)
        ax.xaxis.labelpad=0; ax.yaxis.labelpad=0; ax.zaxis.labelpad=0
        ax.grid(False)
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.pane.fill=False
            axis.pane.set_edgecolor('white')
        fig.text(.06, .982 - .245*i,
                 f"({chr(97+i)})  $\\omega={int(d['omega'])}$,  $N_h={int(d['N']):,}$", fontsize=12)
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap),
                     cax=fig.add_axes([.85, .20, .025, .60]), ticks=[-1, -.1, -.01, 0, .01, .1, 1])
    cb.set_label(r'$\mathrm{Re}(u)/\max|\mathrm{Re}(u)|$')
    for ext in ['pdf', 'png']:
        save_figure(fig, FIG / f'frequency_solution_3d.{ext}', dpi=240, bbox_inches='tight', pad_inches=.04)
    plt.close(fig)

    d48 = next(d for d in reports if d['omega'] == 48)
    q48 = next(q for q in d48['tests'] if q['mu'] == .317)
    s = d48['field_slice']; real = np.asarray(s['real'])
    xx, yy = np.meshgrid(s['x'], s['y'], indexing='ij')
    fig = plt.figure(figsize=(10, 3.8))
    ax = fig.add_axes([.015, .10, .48, .78], projection='3d', computed_zorder=False)
    ax.plot_surface(xx, yy, np.full_like(xx, s['z']), facecolors=cmap(norm(real / np.max(np.abs(real)))),
                    rstride=1, cstride=1, shade=False, antialiased=False, rasterized=True, zorder=2, clip_on=False)
    for z0 in (0, 1):
        ax.plot([0, 3, 3, 0, 0], [0, 0, 1, 1, 0], [z0]*5, color='.4', lw=.65, zorder=3, clip_on=False)
    for x0 in (0, 3):
        for y0 in (0, 1):
            ax.plot([x0, x0], [y0, y0], [0, 1], color='.4', lw=.65, zorder=3, clip_on=False)
    ax.scatter([.2], [.5], [.5], marker='*', s=40, color='black', depthshade=False, zorder=4)
    ax.set(xlabel='$x$', ylabel='$y$', zlabel='$z$', zlim=(0, 1), xticks=[0, 1, 2, 3], yticks=[0, 1], zticks=[0, .5, 1])
    ax.set_box_aspect((3.35, 1.35, 1), zoom=1.1); ax.view_init(elev=22, azim=-75)
    ax.tick_params(pad=0, labelsize=9)
    ax.xaxis.labelpad=0; ax.yaxis.labelpad=0; ax.zaxis.labelpad=0
    ax.grid(False)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.fill=False; axis.pane.set_edgecolor('white')
    fig.text(.06, .94, r'(a) DD field at $\omega=48$, $\mu=0.317$', fontsize=11)
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=fig.add_axes([.11, .09, .29, .025]), orientation='horizontal', ticks=[-1, -.1, 0, .1, 1])
    cb.set_label(r'$\mathrm{Re}(u)/\max|\mathrm{Re}(u)|$', fontsize=9); cb.ax.tick_params(labelsize=8)
    ax = fig.add_axes([.62, .20, .35, .66])
    for m in ['shifted', 'global', 'dd']:
        r = q48[m]; hist = np.asarray(r['history'])
        assert len(hist) == r['iterations'] + 1
        ax.semilogy(np.arange(len(hist)), hist, color=colors[m], lw=1.8, label=labels[m])
        ax.plot(r['iterations'], r['true_residual'], 'o', color=colors[m], ms=5)
        ax.annotate(str(r['iterations']), (r['iterations'], r['true_residual']),
                    xytext=(0, -15), textcoords='offset points', ha='center', fontsize=9, color=colors[m])
    ax.axhline(1e-6, color='.25', ls='--', lw=.8)
    ax.set(xlabel='SL steps', ylabel='Relative residual',
           ylim=(4e-7, 2), title=r'(b) SL steps at $\mu=0.317$')
    ax.grid(axis='y', alpha=.2); ax.legend(frameon=False, fontsize=8, loc='upper right')
    for ext in ['pdf', 'png']:
        save_figure(fig, FIG / f'omega48_representative.{ext}', dpi=240, bbox_inches='tight', pad_inches=.04)
    plt.close(fig)


    def sci(v):
        a, e = f'{v:.2e}'.split('e')
        return rf'\({a}\times10^{{{int(e)}}}\)'
    lines = [r'\begin{table}[!htbp]\centering\small',
             r'\caption{Representative waveguide query at \(\omega=48,\mu=0.317\). Times include ROM application, MG setup, and completion. All methods use the same true-residual target \(10^{-6}\); each solve is timed once.}',
             r'\label{tab:3d-online}', r'\begin{tabular}{lrrrr}\toprule',
             r'Method & Completion it. & Online (s) & Final residual & Speedup \\', r'\midrule']
    for m in ['shifted', 'global', 'dd']:
        r = q48[m]
        lines.append(f"{labels[m]} & {r['iterations']} & {r['seconds']:.2f} & {sci(r['true_residual'])} & {q48['shifted']['seconds']/r['seconds']:.2f}" + r' \\')
    lines += [r'\bottomrule\end{tabular}\end{table}', '']
    (ROOT / 'LaTeX_DL_468198_240419/waveguide_representative_online.tex').write_text('\n'.join(lines))
    row48 = next(r for r in rows if r['omega'] == 48)
    idx = d48['N'] * 8 / 2**20
    gs = d48['global_persistent_MiB'] - d48['global_basis_MiB']
    ds = d48['dd_persistent_MiB'] - d48['dd_basis_MiB'] - idx
    metrics = [('Basis', d48['global_basis_MiB'], d48['dd_basis_MiB']),
               ('Affine operators', gs, ds), ('Shared index map', 0, idx),
               ('Persistent ROM payload', d48['global_persistent_MiB'], d48['dd_persistent_MiB']),
               (r'Measured MATLAB object', row48['global_whos_MiB'], row48['dd_whos_MiB'])]
    # Shared five-column format; content stays beside the layer-storage figure.
    (ROOT / 'LaTeX_DL_468198_240419/waveguide_representative_storage.tex').write_text(representative_storage())
    (OUT / 'representative48_metrics.json').write_text(json.dumps(dict(
        query=q48, storage=metrics, relative_difference=s['relative_difference_to_shifted']), indent=2) + '\n')

    # Preserve the omega48 layerwise diagnostic and its established figure label.
    d = next(d for d in reports if d['omega'] == 48)
    layer_source = DATA / 'figures/uniform_layers/layer_storage_metrics.csv'
    with layer_source.open() as f:
        layer_rows = [r for r in csv.DictReader(f) if r['job_id'] == str(d['metadata']['job_id'])]
    assert len(layer_rows) == 6
    hashes[str(layer_source.relative_to(ROOT))] = hashlib.sha256(layer_source.read_bytes()).hexdigest()
    (OUT / 'waveguide_frequency_sources.json').write_text(json.dumps(hashes, indent=2) + '\n')
    fig, ax = plt.subplots(figsize=(3.15, 2.95), constrained_layout=True)
    for m, off in [('global', -.18), ('dd', .18)]:
        vals = [float(next(r['basis_MiB'] for r in layer_rows if r['method'] == m and int(r['layer']) == k)) for k in [1, 2, 3]]
        assert abs(sum(vals) - d[m + '_basis_MiB']) < .015001
        bars = ax.bar(np.arange(3) + off, vals, .34, color=colors[m], label=('Global' if m == 'global' else 'DD'))
        annotations = ax.bar_label(bars, fmt='%.1f', fontsize=9, padding=3)
        if m == 'dd':
            annotations[-1].set_position((0, 14))
        if m == 'dd':
            bars[-1].set_hatch('///'); bars[-1].set_edgecolor('#153f2c')
    ax.set(xticks=np.arange(3), xticklabels=['Layer 1', 'Layer 2', 'Layer 3'],
                ylabel='Basis (MiB)', ylim=(0, 770))
    ax.legend(frameon=False, fontsize=9, ncol=2, loc='upper center')
    ax.tick_params(labelsize=9)
    ax.yaxis.label.set_size(10)
    ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    for ext in ['pdf', 'png']:
        save_figure(fig, FIG / f'omega48_storage_convergence.{ext}', dpi=240)
    plt.close(fig)
    (OUT / 'waveguide_frequency_table.tex').write_text(waveguide_frequency(rows))
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
