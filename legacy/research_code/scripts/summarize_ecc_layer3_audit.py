"""Summarize the frozen Experiment 2 layer ablation; no paper files edited."""
from pathlib import Path
import csv
import hashlib
import json
import statistics as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/ecc_layer3_audit_20260911'


def read(name):
    with (OUT / name).open() as f:
        return list(csv.DictReader(f))


def mean(rows, key):
    return st.mean(float(r[key]) for r in rows)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    residual = read('residuals.csv')
    training = read('training_residuals.csv')
    storage = read('storage.csv')
    completion = read('completion.csv')
    assert len(residual) == 22 and len(training) == 24 and len(completion) == 44
    assert len({(r['method'], r['query'], r['layers']) for r in completion}) == 44
    assert all(float(r['final_update']) < 1e-8 for r in completion)
    assert all(float(r['prefix_replay_error']) < 1e-10 for r in completion)
    summary = {}
    for m in ['global', 'dd']:
        rr = [r for r in residual if r['method'] == m]
        tr = [r for r in training if r['method'] == m]
        cr = [r for r in completion if r['method'] == m]
        sr = [r for r in storage if r['method'] == m]
        d = summary[m] = {
            'mean_rho2': mean(rr, 'rho2'),
            'mean_rho3': mean(rr, 'rho3'),
            'mean_residual_decrease_pct': 100 * (1 - mean(rr, 'rho3') / mean(rr, 'rho2')),
            'median_query_decrease_pct': st.median(float(r['decrease_pct']) for r in rr),
            'training_mean_decrease_pct': 100 * (1 - mean(tr, 'rho3') / mean(tr, 'rho2')),
            'third_basis_MiB': float(sr[2]['basis_MiB']),
            'saved_mean_layer3_seconds': mean(rr, 'saved_layer3_seconds'),
            'representative_rho2': float(rr[0]['rho2']),
            'representative_rho3': float(rr[0]['rho3']),
        }
        for depth in [2, 3]:
            rows = [r for r in cr if int(r['layers']) == depth]
            d[f'L{depth}'] = {
                'payload_MiB': sum(float(r['basis_MiB']) for r in sr[:depth]) + float(sr[0]['shared_map_MiB']),
                'mean_completion_iters': mean(rows, 'completion_iters'),
                'mean_online_seconds': mean(rows, 'total_seconds'),
                'max_final_residual': max(float(r['final_residual']) for r in rows),
                'max_final_error': max(float(r['final_error']) for r in rows),
                'max_prefix_replay_error': max(float(r['prefix_replay_error']) for r in rows),
            }
        change = []
        for k in range(1, 12):
            rows = {int(r['layers']): r for r in cr if int(r['query']) == k}
            change.append(int(rows[3]['completion_iters']) - int(rows[2]['completion_iters']))
        d['third_layer_completion_iteration_changes'] = change
        d['queries_fewer_equal_more_iters_with_L3'] = [sum(v < 0 for v in change), sum(v == 0 for v in change), sum(v > 0 for v in change)]

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.5), constrained_layout=True)
    for col, m in enumerate(['global', 'dd']):
        rr = [r for r in residual if r['method'] == m]
        x = [float(r['ratio']) for r in rr]
        for depth, color, marker in [(2, '#2878B5', 'o'), (3, '#D95319', 's')]:
            axes[0, col].semilogy(x, [float(r[f'rho{depth}']) for r in rr], color=color, marker=marker, markersize=4, label=f'{depth} layers')
            cr = sorted([r for r in completion if r['method'] == m and int(r['layers']) == depth], key=lambda r: int(r['query']))
            axes[1, col].plot(x, [int(r['completion_iters']) for r in cr], color=color, marker=marker, markersize=4, label=f'{depth} layers')
        axes[0, col].set_title('Global TA-ROM' if m == 'global' else 'DD-TA-ROM')
        axes[0, col].set_ylabel('Post-ROM true relative residual')
        axes[0, col].set_ylim(1e-5, 1e-1)
        axes[1, col].set_ylabel('WaveHoltz completion iterations')
        for row in range(2):
            axes[row, col].set_xlabel(r'Ellipse axis ratio $\alpha$')
            axes[row, col].grid(alpha=.2)
            axes[row, col].legend(frameon=False)
    fig.savefig(OUT / 'layer2_vs_layer3.png', dpi=190)
    plt.close(fig)
    source = ROOT / 'outputs/helm_trajectory_evidence/review_repaired_20260905/ecc_residual_ablation_paper.mat'
    paths = [source, ROOT / 'code/run_ecc_layer3_audit.m', ROOT / 'code/waveholtz_2d_pml_scattering_ecc_correction_rom.m', Path(__file__)]
    for name in ['residuals.csv', 'training_residuals.csv', 'completion.csv', 'storage.csv']:
        paths.append(OUT / name)
    summary['source_sha256'] = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    text = ['# Experiment 2: necessity of the third ROM layer', '',
            'Frozen-model audit, 2026-09-11. Workflow: ECC/elliptic-obstacle scattering.', '',
            'The two-layer models are exact prefixes of the current paper models. Training samples, '
            'basis truncations in layers 1–2, all eleven unseen geometries, PML, lift, and the '
            'WaveHoltz relative-update stopping criterion (1e-8) are unchanged. Both prefixes '
            'are replayed through the original ECC kernel without retraining. Replayed fields '
            'are checked against archived iterates; three-layer completion counts must match '
            'the original cohort. True residuals include inactive obstacle nodes.', '',
            '| Method | Mean residual, L2 | Mean residual, L3 | Decrease | Median per-query decrease |',
            '|---|---:|---:|---:|---:|']
    for m in ['global', 'dd']:
        d = summary[m]
        text.append(f"| {m} | {d['mean_rho2']:.6e} | {d['mean_rho3']:.6e} | {d['mean_residual_decrease_pct']:.2f}% | {d['median_query_decrease_pct']:.2f}% |")
    text += ['', '| Method | Layers | Payload (MiB) | Mean completion iterations | Mean online time (s)* | Maximum final residual |',
             '|---|---:|---:|---:|---:|---:|']
    for m in ['global', 'dd']:
        for depth in [2, 3]:
            d = summary[m][f'L{depth}']
            text.append(f"| {m} | {depth} | {d['payload_MiB']:.4f} | {d['mean_completion_iters']:.3f} | {d['mean_online_seconds']:.4f} | {d['max_final_residual']:.3e} |")
    text += ['', '*Exploratory timing: one fresh pass per method/prefix/query, serial MATLAB '
             'with one computational thread; four-case order rotated between queries. No warm-up '
             'or repeated timing cohort. Online time includes ROM, frequency-to-wave lift, and '
             'completion; it excludes setup, loading, assembly and verification. Do not compare '
             'these wall times directly with the paper’s four-worker timing cohort or infer small '
             'speed differences. Iteration counts and true residuals are the primary ablation evidence.', '',
             '## Interpretation', '',
             'The third layer improves the mean residual modestly, with a weaker benefit for DD. '
             'FGMRES minimizes over an expanded search space, so a reduced residual by itself '
             'does not establish that the extra layer is worthwhile. Its compressed inverse action '
             'must add a useful direction for the residual left by the first two layers. A concentrated '
             'snapshot spectrum measures compressibility, not online residual reduction.', '',
             'At alpha=0.97440976247239, the third layer reduces the Global residual from '
             '0.04692 to 0.04369 and DD from 0.03617 to 0.03520. This query dominates the mean; '
             'the third layer does not resolve this difficult geometry. The mean reductions '
             'on the twelve training samples are larger than on the tests: '
             f"{summary['global']['training_mean_decrease_pct']:.1f}% for Global and "
             f"{summary['dd']['training_mean_decrease_pct']:.1f}% for DD. This is evidence of limited "
             'held-out utility under the current layer-3 compression, not proof that all possible '
             'third-layer constructions are ineffective.', '']
    for m in ['global', 'dd']:
        d = summary[m]
        text.append(f"- {m}: L3 changes completion iterations by {d['third_layer_completion_iteration_changes']} "
                    f"(queries 1–11, L3 minus L2); fewer/equal/more in {d['queries_fewer_equal_more_iters_with_L3']}. "
                    f"Its extra basis payload is {d['third_basis_MiB']:.4f} MiB.")
    text += ['', '## Recommendation', '',
             'Use two layers for both methods in the main Experiment 2 if the aim is a compact '
             'configuration with comparable complete-solve performance. Keep this two-versus-three '
             'ablation as evidence for the choice. The third layer is not required for convergence '
             'on any of the eleven tests. Global has exactly the same aggregate completion count; '
             'DD saves only six iterations across all eleven queries (0.42%). These small average '
             'changes do not establish a meaningful wall-time advantage.', '',
             'Removing layer 3 reduces Global payload by 14.3% and DD payload by 1.3%. With both '
             'methods using two layers, DD still saves 36.4% of persistent payload, compared with '
             '44.7% for three layers. Any manuscript adoption should update layer counts, spectra, '
             'storage accounting and completion statistics together, and refresh repeated timings '
             'before replacing the published timing table. The conclusion is specific to the '
             'current frozen three-layer construction; a differently trained or compressed third '
             'layer has not been ruled out.', '']
    text += ['', '## Reproduction', '', '```sh',
             'matlab -singleCompThread -batch "addpath(\'code\');run_ecc_layer3_audit(\'completion\');"',
             'outputs/reviewer_20260909/tooling_venv/bin/python scripts/summarize_ecc_layer3_audit.py', '```', '',
             'Data and figure: `outputs/ecc_layer3_audit_20260911/`. '
             'Source and CSV hashes are recorded in `summary.json`. '
             'The manuscript and published figures were not changed.', '']
    (ROOT / 'md/ecc_layer3_audit_20260911.md').write_text('\n'.join(text))
    print(json.dumps({k: v for k, v in summary.items() if k != 'source_sha256'}, indent=2))


if __name__ == '__main__':
    main()
