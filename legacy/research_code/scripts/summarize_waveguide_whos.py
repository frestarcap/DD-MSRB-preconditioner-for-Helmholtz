"""Validate MATLAB object measurements against the four accepted frequency reports."""
from pathlib import Path
import csv
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/waveguide_whos_20260913'
RESULTS = OUT / 'results'


def main():
    rows, hashes, versions = [], {}, set()
    for omega in [24, 36, 48, 60]:
        path = RESULTS / f'w{omega}_whos.json'
        measured = json.loads(path.read_text())
        assert measured['validated'] and len(measured['rows']) == 2
        report_dir = ROOT / 'outputs/scnet_3d_20260909' / (
            f'remote_v7/full_w{omega}_p8_t4_b2_sr40_or40_sm40_om40_c12000'
            if omega < 60 else 'remote_recovery_w60')
        report_path = report_dir / 'report.json'
        report = json.loads(report_path.read_text())
        source_hash_file = RESULTS / f'w{omega}_source_sha256.txt'
        source_hashes = dict(line.split(maxsplit=1)[::-1] for line in source_hash_file.read_text().splitlines())
        source_hashes = {k.strip(): v for k, v in source_hashes.items()}
        report_hashes = [v for k, v in source_hashes.items() if k.endswith('/report.json')]
        assert report_hashes == [hashlib.sha256(report_path.read_bytes()).hexdigest()]
        assert measured['source'] in source_hashes
        pair = {r['method']: r for r in measured['rows']}
        row = {'omega': omega, 'N': int(report['N'])}
        for method in ['global', 'dd']:
            r = pair[method]
            versions.add(r['matlab_version'])
            assert r['matlab_release'] == '2024a' and r['computer'] == 'GLNXA64'
            assert r['job_id'] == '44214697' and r['omega'] == omega and r['N'] == report['N']
            assert len(r['layers']) == 3
            assert r['formula_persistent_bytes'] == r['numeric_bytes_one_shared_map']
            assert r['formula_persistent_bytes'] == r['stored_training_persistent_bytes']
            assert abs(r['formula_persistent_bytes'] / 2**20 - report[method + '_persistent_MiB']) < 1e-8
            assert r['basis_whos_bytes'] / 2**20 == report[method + '_basis_MiB']
            assert r['index_whos_bytes_all_layers'] == 3 * r['index_whos_bytes_one_map']
            assert r['model_whos_bytes'] == r['numeric_bytes_per_layer_indices'] + r['container_and_auxiliary_bytes']
            row[method + '_formula_MiB'] = r['formula_persistent_bytes'] / 2**20
            row[method + '_whos_MiB'] = r['model_whos_bytes'] / 2**20
            row[method + '_index_all_layers_MiB'] = r['index_whos_bytes_all_layers'] / 2**20
            row[method + '_container_auxiliary_MiB'] = r['container_and_auxiliary_bytes'] / 2**20
        row['formula_saving_pct'] = 100 * (1 - row['dd_formula_MiB'] / row['global_formula_MiB'])
        row['whos_saving_pct'] = 100 * (1 - row['dd_whos_MiB'] / row['global_whos_MiB'])
        resource = (RESULTS / f'w{omega}_inspection_resources.txt').read_text()
        assert 'Exit status: 0' in resource
        row['inspection_process_peak_RSS_MiB'] = int(re.search(r'Maximum resident set size \(kbytes\):\s*(\d+)', resource)[1]) / 1024
        rows.append(row)
        for p in [path, source_hash_file, report_path, RESULTS / f'w{omega}_inspection_resources.txt']:
            hashes[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    assert len(versions) == 1
    accounting = (OUT / 'accounting.txt').read_text()
    assert '44214697|COMPLETED|0:0|' in accounting
    assert 'WAVEGUIDE_WHOS_COHORT_PASS' in (OUT / 'stdout.log').read_text()
    manifest = {'validated': True, 'matlab_version': versions.pop(), 'job_id': '44214697',
                'protocol': 'Logical deployed-object whos; actual per-layer index fields retained; not physical RSS.',
                'rss_scope': 'Inspection process loading both stored models; not a solver benchmark or Global/DD comparison.',
                'rows': rows, 'source_sha256': hashes}
    (OUT / 'summary.json').write_text(json.dumps(manifest, indent=2) + '\n')
    with (OUT / 'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
