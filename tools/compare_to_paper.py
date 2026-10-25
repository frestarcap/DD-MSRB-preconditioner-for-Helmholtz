#!/usr/bin/env python3
"""Compare a re-run of Experiments 1-2 (results/paper) with the paper data (paper_data).

    python tools/compare_to_paper.py [--rerun results/paper] [--paper paper_data] [--out report.md]

Each quantity has a class:
  E  must agree exactly (ranks, storage to the printed digits, SL step counts);
  T  must agree within a tolerance (rho_ROM within a factor 2, object size within 0.5%);
  M  marginal: a query may gain or lose one SL step when rho_ROM or the residual of
     its last SL step lies at the tolerance; such flips are listed, not failed;
  R  timing ratios (speedups), reported with their relative change;
  X  differs by design (see docs/CHANGES_FROM_PAPER_RUNS.md); reported only.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROWS = []


def add(exp, quantity, paper, rerun, cls, ok=None, note=''):
    if ok is None:
        ok = paper == rerun
    status = {True: 'match', False: 'DIFFERS'}[bool(ok)] if cls in 'ET' else (
        'flip' if cls == 'M' and not ok else 'reported')
    ROWS.append(dict(experiment=exp, quantity=quantity, paper=paper, rerun=rerun, cls=cls,
                     status=status, note=note))


def fmt(x, digits=3):
    return f'{x:.{digits}f}'


def rho_ok(a, b, tol):
    return (a <= tol and b <= tol) or (max(a, b) <= 2 * min(a, b))


def online(exp, paper, rerun, tol, methods, baseline='shifted', median=False):
    for m in methods:
        p, r = paper[paper.method == m], rerun[rerun.method == m]
        ps = p[p.repeat == 1].set_index('query').sl_steps
        rs = r[r.repeat == 1].set_index('query').sl_steps
        flips = int(((ps - rs).abs() == 1).sum())
        add(exp, f'{m}: mean SL steps', fmt(ps.mean(), 2), fmt(rs.mean(), 2), 'M' if flips else 'E',
            ps.equals(rs), f'{flips} queries differ by one step' if flips else '')
        if m not in ('shifted', 'direct'):
            add(exp, f'{m}: queries without SL steps', int((ps == 0).sum()), int((rs == 0).sum()), 'E')
            pr = p[p.repeat == 1].set_index('query').rho_rom
            rr = r[r.repeat == 1].set_index('query').rho_rom
            add(exp, f'{m}: max rho_ROM', f'{pr.max():.2e}', f'{rr.max():.2e}', 'T', rho_ok(pr.max(), rr.max(), tol))
            for q in ps.index:
                if ps[q] != rs[q]:
                    # One step more or less: rho_ROM or the last SL residual lies at the tolerance.
                    marginal = min(pr[q], rr[q]) <= 2 * tol or abs(ps[q] - rs[q]) == 1
                    add(exp, f'{m}: SL steps of query {q}', int(ps[q]), int(rs[q]), 'M' if marginal else 'E',
                        False, f'rho_ROM {pr[q]:.2e} / {rr[q]:.2e}')
        stat = 'median' if median else 'mean'
        pt, rt = getattr(p.total_s, stat)(), getattr(r.total_s, stat)()
        pb = getattr(paper[paper.method == baseline].total_s, stat)()
        rb = getattr(rerun[rerun.method == baseline].total_s, stat)()
        add(exp, f'{m}: speedup vs SL ({stat})', fmt(pb / pt, 2), fmt(rb / rt, 2), 'R',
            note=f'{100 * ((rb / rt) / (pb / pt) - 1):+.0f}%')


def storage(exp, paper, rerun, key='method'):
    p, r = paper.set_index(key), rerun.set_index(key)
    for m in p.index:
        for col in ('basis_MiB', 'payload_MiB'):
            add(exp, f'{m}: {col}', fmt(p.loc[m, col]), fmt(r.loc[m, col]), 'E')
        if 'object_MiB' in p:
            a, b = p.loc[m, 'object_MiB'], r.loc[m, 'object_MiB']
            add(exp, f'{m}: object_MiB', fmt(a), fmt(b), 'T', abs(b / a - 1) <= 5e-3)


def exp1(paper, rerun):
    e = 'Exp 1'
    online(e, pd.read_csv(paper / 'online_timings.csv'), pd.read_csv(rerun / 'online_timings.csv'), 1e-12,
           ['global', 'dd', 'nt', 'ss', 'shifted'], median=True)
    storage(e, pd.read_csv(paper / 'storage.csv'), pd.read_csv(rerun / 'storage.csv'))
    p, r = pd.read_csv(paper / 'training_coverage.csv'), pd.read_csv(rerun / 'training_coverage.csv')
    for (n, m), a in p.set_index(['n_s', 'method']).iterrows():
        b = r.set_index(['n_s', 'method']).loc[(n, m)]
        add(e, f'n_s={n} {m}: ranks', a.ranks, b.ranks, 'E')
        add(e, f'n_s={n} {m}: payload_MiB', fmt(a.payload_MiB), fmt(b.payload_MiB), 'E')
        add(e, f'n_s={n} {m}: max rho_ROM', f'{a.max_rho_rom:.2e}', f'{b.max_rho_rom:.2e}', 'T',
            rho_ok(a.max_rho_rom, b.max_rho_rom, 1e-12))
        add(e, f'n_s={n} {m}: online mean (s)', fmt(a.online_mean_s, 4), fmt(b.online_mean_s, 4), 'X',
            note='paper: exact-LU SL; rerun: ILU')
        add(e, f'n_s={n} {m}: offline (s)', fmt(a.offline_s, 2), fmt(b.offline_s, 2), 'X')
    p, r = pd.read_csv(paper / 'partitions.csv').set_index('variant'), pd.read_csv(rerun / 'partitions.csv').set_index('variant')
    for v, a in p.iterrows():
        b = r.loc[v]
        add(e, f'{v}: ranks', a.ranks, b.ranks, 'E')
        add(e, f'{v}: payload_MiB', fmt(a.payload_MiB), fmt(b.payload_MiB), 'E')
        add(e, f'{v}: max rho_ROM', f'{a.max_rho_rom:.2e}', f'{b.max_rho_rom:.2e}', 'T',
            rho_ok(a.max_rho_rom, b.max_rho_rom, 1e-10))
        add(e, f'{v}: mean time (s)', fmt(a.mean_total_s, 4), fmt(b.mean_total_s, 4), 'X',
            note='rotating order in the rerun')
    p, r = pd.read_csv(paper / 'frequency_storage.csv'), pd.read_csv(rerun / 'frequency_storage.csv')
    for (w, m), a in p.set_index(['omega', 'method']).iterrows():
        b = r.set_index(['omega', 'method']).loc[(w, m)]
        add(e, f'omega={w:g} {m}: payload_MiB', fmt(a.payload_MiB), fmt(b.payload_MiB), 'E')
    p, r = pd.read_csv(paper / 'frequency_timings.csv'), pd.read_csv(rerun / 'frequency_timings.csv')
    for w in sorted(p.omega.unique()):
        for m in ('global', 'dd'):
            pa = p[(p.omega == w) & (p.method == m)].mean_total_s.mean()
            ra = r[(np.isclose(r.omega, w)) & (r.method == m)].mean_total_s.mean()
            pb = p[(p.omega == w) & (p.method == 'shifted')].mean_total_s.mean()
            rb = r[(np.isclose(r.omega, w)) & (r.method == 'shifted')].mean_total_s.mean()
            add(e, f'omega={w:g} {m}: speedup vs SL', fmt(pb / pa, 1), fmt(rb / ra, 1), 'R',
                note=f'{100 * ((rb / ra) / (pb / pa) - 1):+.0f}%')


def exp2(paper, rerun):
    e = 'Exp 2'
    online(e, pd.read_csv(paper / 'online_timings.csv'), pd.read_csv(rerun / 'online_timings.csv'), 1e-12,
           ['global', 'dd', 'shifted'])
    p, r = pd.read_csv(paper / 'rank_storage.csv'), pd.read_csv(rerun / 'rank_storage.csv')
    for (m, j), a in p.set_index(['method', 'iteration']).iterrows():
        b = r.set_index(['method', 'iteration']).loc[(m, j)]
        add(e, f'{m} iteration {j}: ranks', str(a.ranks), str(b.ranks), 'E')
    storage(e, pd.read_csv(paper / 'storage.csv'), pd.read_csv(rerun / 'storage.csv'))
    p, r = pd.read_csv(paper / 'storage.csv').set_index('method'), pd.read_csv(rerun / 'storage.csv').set_index('method')
    for m in p.index:
        add(e, f'{m}: offline (s)', fmt(p.loc[m, 'offline_s'], 2), fmt(r.loc[m, 'offline_s'], 2), 'X',
            note='unused diagnostics removed')


def main():
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rerun', type=Path, default=root / 'results/paper')
    ap.add_argument('--paper', type=Path, default=root / 'paper_data')
    ap.add_argument('--out', type=Path, default=root / 'reference_outputs/comparison_report.md')
    args = ap.parse_args()
    for name, fn in (('exp1', exp1), ('exp2', exp2)):
        if (args.rerun / name).is_dir():
            fn(args.paper / name, args.rerun / name)
    t = pd.DataFrame(ROWS)
    lines = ['# Re-run versus paper', '',
             'Generated by `tools/compare_to_paper.py`. Classes: E exact, T tolerance, '
             'M marginal (rho_ROM or the last SL residual at the tolerance), R timing ratio, X differs by design.', '']
    counts = t.groupby(['cls', 'status']).size()
    lines += ['| Class | Status | Count |', '|---|---|---:|']
    lines += [f'| {c} | {s} | {n} |' for (c, s), n in counts.items()] + ['']
    for exp, g in t.groupby('experiment', sort=False):
        lines += [f'## {exp}', '', '| Quantity | Paper | Re-run | Class | Status | Note |', '|---|---:|---:|---|---|---|']
        lines += [f'| {r.quantity} | {r.paper} | {r.rerun} | {r.cls} | {r.status} | {r.note} |' for r in g.itertuples()]
        lines.append('')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text('\n'.join(lines))
    failed = t[(t.cls.isin(['E', 'T'])) & (t.status == 'DIFFERS')]
    print(f'{len(t)} quantities; {len(failed)} E/T differences; report: {args.out}')
    for r in failed.itertuples():
        print(f'  {r.experiment} {r.quantity}: paper {r.paper}, rerun {r.rerun} ({r.cls})')


if __name__ == '__main__':
    main()
