"""Read only spectra/metadata (or recompute SVDs of saved snapshots) for JCP.

Run in an environment with h5py and NumPy. No inverse solves or trajectory
retraining. Explicit manifest entries define the manuscript coverage.
"""
import argparse
import json
import struct
import time
from pathlib import Path
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/greedy_dp_paper_20260921'


def hex64(value):
    return struct.pack('>d', float(value)).hex()


def scalar(d):
    return float(d[()].ravel()[0])


def field(f, group, key, i):
    d = group[key]
    return f[d[()].ravel()[i]]


def chars(d):
    return ''.join(chr(int(v)) for v in d[()].ravel())


def local_manifest():
    entries = []
    def add(tag, group, source, kind, **kwargs):
        entries.append(dict(tag=tag, group=group, source=str(ROOT/source), kind=kind, **kwargs))
    add('halfspace_main', 'halfspace_main', Path('outputs/halfspace_dd_refinement/review_repaired_20260905/slabs1-aligned-sym8.mat'), 'artifact')
    for ns in (6, 12, 24):
        add(f'halfspace_n{ns}', 'halfspace_training', Path(f'outputs/revision_20260907/halfspace/dd_n{ns}_eta-5.mat'), 'r')
    reviewer = ROOT/'outputs/reviewer_20260909'
    for scale in ('1', '1.5', '2', '4'):
        add(f'halfspace_frequency_{scale}', 'halfspace_frequency', Path(f'outputs/reviewer_20260909/frequency_{scale}/dd.mat'), 'reviewer')
    for variant in ('merged', 'signed', 'cartesian'):
        add(f'halfspace_partition_{variant}', 'halfspace_partition', Path(f'outputs/reviewer_20260909/frequency_1/{variant}.mat'), 'reviewer')
    for directory in sorted([reviewer/'two_parameter'] + list(reviewer.glob('explore_*'))):
        add(directory.name, 'halfspace_two_parameter', directory/'dd.mat', 'reviewer')
    add('ecc_uniform5', 'ecc_main', Path('outputs/ecc_dd_budget_20260919/uniform5_dd.mat'), 'ecc')
    for label, point in zip(('half', 'reference', 'double', 'quadruple'), ('P1','P2','P3','P4')):
        add(f'waveguide_threshold_{point}', 'waveguide_threshold', Path(f'outputs/helmholtz_3d_candidates/large/gaussian_waveguide_threshold_20260908_{label}_training.mat'), '3d')
    return entries


def export_one(entry):
    records = []
    with h5py.File(entry['source']) as f:
        kind = entry['kind']
        path = dict(artifact='artifact/layers', r='r/correction_rom/layers',
                    reviewer='model', ecc='model/layers', **{'3d':'models/dd/layers'})[kind]
        for j, ref in enumerate(f[path][()].ravel()):
            layer = f[ref]
            sp = layer['space'] if 'space' in layer else layer['basis_space']
            n, ranks, spectra, names, constraints = [], [], [], [], []
            t0 = time.perf_counter()
            if kind == 'ecc':
                for name, basis_name in zip(('near','far','pml'), ('Un','Uf','Up')):
                    m = sp['block_meta'][name]
                    spectra.append(m['singular_values'][()].ravel().tolist())
                    n.append(sp[basis_name].shape[1]); ranks.append(sp[basis_name].shape[0]); names.append(name)
                    assert np.isinf(scalar(m['rank_cap']))
                    assert m['rank_tail_tol'].attrs.get('MATLAB_empty')
                    assert chars(m['rank_rule']).startswith('marginal_storage')
                eta = float(f['cfg/corr_rank_rtols'][()].ravel()[j])
                origin = 'saved local singular values'
            else:
                blocks = sp['blocks']
                count = blocks['k'].size
                eta = None
                if kind == 'reviewer':
                    eta = float(f['c/eta'][()].ravel()[j])
                    raw = field(f, f['audit'], 'E', j)[()]
                    et = raw['real'] + 1j*raw['imag']
                    del raw
                for b in range(count):
                    nb = int(scalar(field(f, blocks, 'count', b)))
                    n.append(nb); ranks.append(int(scalar(field(f, blocks, 'k', b))))
                    names.append(chars(field(f, blocks, 'name', b)) if 'name' in blocks else f'block{b+1}')
                    if kind == 'reviewer':
                        ix = field(f, blocks, 'idx', b)[()].ravel().astype(np.int64)-1
                        sv = np.linalg.svd(et[:, ix].T, compute_uv=False)
                        spectra.append(sv.tolist())
                    else:
                        m = field(f, blocks, 'meta', b)
                        eta_b = scalar(m['rank_global_tail_tol'])
                        assert eta is None or eta == eta_b
                        eta = eta_b
                        spectra.append(m['singular_values'][()].ravel().tolist())
                        cap = scalar(m['rank_cap'])
                        local = None if m['rank_tail_tol'].attrs.get('MATLAB_empty') else scalar(m['rank_tail_tol'])
                        constraints.append(dict(cap=None if np.isinf(cap) else cap, local_tail=local,
                                                policy=chars(m['rank_rule'])))
                origin = 'recomputed SVD of frozen audit.E (no trajectory retraining)' if kind == 'reviewer' else 'saved local singular values'
                if kind == 'reviewer':
                    del et
            assert eta is not None
            for c in constraints:
                assert c['cap'] is None and c['local_tail'] is None, (entry['tag'], c)
                assert c['policy'].startswith('marginal_storage'), (entry['tag'], c)
            rec = dict(tag=entry['tag'], group=entry['group'], source=entry['source'], layer=j+1,
                       n=n, saved_ranks=ranks, names=names, eta=eta, eta_hex=hex64(eta),
                       svals=spectra, svals_hex=[[hex64(v) for v in sv] for sv in spectra],
                       spectrum_origin=origin, export_seconds=time.perf_counter()-t0,
                       constraints=constraints)
            records.append(rec)
            print(f"EXPORTED {entry['tag']} L{j+1} B={len(n)} rank={sum(ranks)} {rec['export_seconds']:.3f}s", flush=True)
    return records


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest');p.add_argument('--output', default=str(OUT/'local_inputs.json'))
    args=p.parse_args()
    entries=json.loads(Path(args.manifest).read_text()) if args.manifest else local_manifest()
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    (output.parent/(output.stem+'_manifest.json')).write_text(json.dumps(entries,indent=2)+'\n')
    records=[]
    for e in entries:
        records.extend(export_one(e))
        output.write_text(json.dumps(dict(records=records),indent=2)+'\n')
    print(f'PAPER_INPUT_EXPORT_PASS models={len(entries)} layers={len(records)}')


if __name__ == '__main__':
    main()
