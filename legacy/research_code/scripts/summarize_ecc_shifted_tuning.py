"""Report paired ECC timings, preserving unfavorable pilot evidence."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 'ecc_shifted_tuning_20260919'
LABELS = {'shifted': 'Shifted ILU', 'direct': 'Sparse direct', 'old_global': 'Original TA-ROM',
          'old_dd': 'Original DD-TA-ROM', 'global_n96_d3_eta1e-06': 'Adjusted TA-ROM',
          'dd_n96_d3_eta1e-06': 'Adjusted DD-TA-ROM'}
METHODS = list(LABELS)
summaries = {}
lines = ['# ECC shifted-Laplacian tuning', '',
         'Same grid, frequency, parameter interval, shifted ILUTP (beta 0.2, drop tolerance 1e-3), '
         'GMRES restart 40, and full-domain residual target 1e-10. '
         'Adjusted models use 96 training samples, three layers, and tails (1e-6,1e-6,1e-6).', '',
         'Every timing includes ROM work, conditional shifted-matrix/ILU setup, completion, '
         'and the final true-residual evaluation. Common full-order assembly, loading, and '
         'offline training are excluded. Serial MATLAB R2025a, one computational thread; '
         'one warm-up and five measured repetitions with rotating method order.', '']
fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
for col, cohort in enumerate(('confirm', 'validate')):
    rows = pd.read_csv(OUT / f'{cohort}_timings.csv')
    assert (rows.residual <= 1e-10).all()
    assert rows.groupby(['method', 'query']).size().eq(5).all()
    summary = pd.read_csv(OUT / f'{cohort}_summary.csv').set_index('method').loc[METHODS]
    summaries[cohort] = summary
    n = rows['query'].nunique()
    title = f'{n} original queries' if cohort == 'confirm' else f'{n} fresh queries'
    lines += [f'## {title}', '', '| Method | Online (s) | Completion steps | Speedup | ROM accepted | Max field error |',
              '|---|---:|---:|---:|---:|---:|']
    for name, r in summary.iterrows():
        lines.append(f'| {LABELS[name]} | {r.total_s:.5f} | {r.iterations:.2f} | {r.speedup:.2f}x | '
                     f'{int(r.no_setup_queries)}/{n} | {r.max_error:.2e} |')
    lines += ['', 'Sparse direct has no ROM stage; its “ROM accepted” count only denotes absence of ILU setup.', '']
    ax = axes[0, col]
    y = np.arange(len(METHODS)); left = np.zeros(len(METHODS))
    for key, label, color in [('ROM_s', 'ROM', '#3377aa'), ('setup_s', 'ILU setup', '#b6bcc3'),
                              ('solve_s', 'Completion / direct', '#dd8844')]:
        vals = summary[key].to_numpy()
        ax.barh(y, vals, left=left, color=color, label=label)
        left += vals
    ax.set_yticks(y, [LABELS[x] for x in METHODS]); ax.invert_yaxis()
    ax.set_xlabel('Mean online time (s)'); ax.set_title(title); ax.grid(axis='x', alpha=.2)
    if col == 0: ax.legend(loc='lower right', fontsize=8)
    ax = axes[1, col]
    for name, color in [('shifted','#555555'),('global_n96_d3_eta1e-06','#3377aa'),
                         ('dd_n96_d3_eta1e-06','#c44d55')]:
        t = rows[rows.method == name].groupby('parameter').mean(numeric_only=True).sort_index()
        ax.plot(t.index, t.total_s, '.-', label=LABELS[name], color=color)
    ax.set_xlabel('Ellipse aspect ratio'); ax.set_ylabel('Mean online time (s)')
    ax.grid(alpha=.2); ax.legend(fontsize=8)
fig.suptitle('ECC: training coverage and efficient DD assembly', fontsize=15)
fig.savefig(OUT / 'comparison.png', dpi=190)
fig.savefig(OUT / 'comparison.pdf')
plt.close(fig)
storage = pd.read_csv(OUT / 'storage.csv')
g, d = storage.iloc[0], storage.iloc[1]
lines += ['## Storage and offline cost', '',
          f'- Global/DD basis: {g.basis_MiB:.3f}/{d.basis_MiB:.3f} MiB; '
          f'DD reduction {(1-d.basis_MiB/g.basis_MiB)*100:.1f}%.',
          f'- Basis plus one shared index map: {g.payload_MiB:.3f}/{d.payload_MiB:.3f} MiB; '
          f'DD reduction {(1-d.payload_MiB/g.payload_MiB)*100:.1f}%.',
          f'- Measured offline training: {g.offline_s:.2f}/{d.offline_s:.2f} s. '
          'This is excluded from online speedups. Storage is not peak process memory.', '',
          '## Validation', '',
          'The selected configuration was fixed before the fresh cohort was evaluated. '
          'Fresh seeds start at 20260919, with disjoint training, development, and validation queries. '
          'Geometry repair can map draws onto old points; overlaps are rejected before solver evaluation '
          'until 22 fresh geometries are collected (validation_candidates.csv). '
          'Every final field is checked against a direct reference. Production replay and block assembly '
          'checks are saved in production_verification.csv and assembly_verification.csv.', '']
for cohort, s in summaries.items():
    lines.append(f'- {cohort}: maximum true residual {s.max_residual.max():.3e}; '
                 f'DD time saving versus adjusted Global {(1-s.loc[METHODS[-1],"total_s"]/s.loc[METHODS[-2],"total_s"])*100:.1f}%.')
accuracy = pd.read_csv(OUT / 'accuracy_summary.csv').set_index('method').loc[METHODS]
arows = pd.read_csv(OUT / 'accuracy_timings.csv')
assert (arows.residual <= 1e-12).all()
assert arows.groupby(['method', 'query']).size().eq(3).all()
lines += ['', '## Higher-accuracy crosscheck', '',
          'Same 22 validation geometries and frozen models; all methods use a uniformly '
          'tightened 1e-12 residual target, one warm-up and three repetitions. '
          'This check follows the larger DD field error observed at the original target.', '',
          '| Method | Online (s) | Steps | Speedup | Max field error |', '|---|---:|---:|---:|---:|']
for name, r in accuracy.iterrows():
    lines.append(f'| {LABELS[name]} | {r.total_s:.5f} | {r.iterations:.2f} | {r.speedup:.2f}x | {r.max_error:.2e} |')
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True,
                         gridspec_kw={'width_ratios':[1.8,1]})
y=np.arange(len(METHODS)); left=np.zeros(len(METHODS))
for key,label,color in [('ROM_s','ROM','#3377aa'),('setup_s','ILU setup','#b6bcc3'),
                        ('solve_s','Completion / direct','#dd8844')]:
    values=accuracy[key].to_numpy()
    axes[0].barh(y,values,left=left,color=color,label=label); left+=values
axes[0].set_yticks(y,[LABELS[x] for x in METHODS]);axes[0].invert_yaxis()
axes[0].set_xlabel('Mean online time (s)');axes[0].grid(axis='x',alpha=.2)
axes[0].legend(loc='lower right',fontsize=8)
axes[0].set_title('22 fresh geometries; true residual target 1e-12')
bars=axes[1].bar(['TA-ROM','DD-TA-ROM'],[g.payload_MiB,d.payload_MiB],color=['#3377aa','#c44d55'])
axes[1].bar_label(bars,fmt='%.2f MiB',padding=3)
axes[1].set_ylim(0,51);axes[1].set_ylabel('Basis + shared index map (MiB)')
axes[1].set_title('Adjusted model storage');axes[1].grid(axis='y',alpha=.2)
fig.suptitle(f'Uniform higher accuracy: TA-ROM {accuracy.loc[METHODS[-2],"speedup"]:.2f}x, '
             f'DD-TA-ROM {accuracy.loc[METHODS[-1],"speedup"]:.2f}x',fontsize=14)
fig.savefig(OUT/'accuracy_comparison.png',dpi=190)
fig.savefig(OUT/'accuracy_comparison.pdf')
plt.close(fig)
(OUT / 'results.md').write_text('\n'.join(lines)+'\n')
paths = ['code/run_ecc_shifted_tuning.m','code/benchmark_ecc_shifted_tuning.m',
         'code/verify_ecc_shifted_tuning.m','code/rom_reduced_space_utils.m',
         'code/waveholtz_2d_pml_scattering_ecc_correction_rom.m','code/rom_verified_snapshot_step.m',
         'code/shifted_laplacian_gmres.m','scripts/reproduce_ecc_shifted_tuning.sh',
         'scripts/summarize_ecc_shifted_tuning.py']
hashes = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}
(OUT/'provenance.json').write_text(json.dumps({'source_sha256':hashes},indent=2)+'\n')
print((OUT / 'results.md').read_text())
