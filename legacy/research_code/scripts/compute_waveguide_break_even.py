#!/usr/bin/env python3
"""Break-even query counts for the waveguide frequency study (Table tab:3d-frequency-cost).

N = offline cost / (mean shifted-MG online time - mean ROM online time), for two
offline measures: the reported wall-clock time, with the eight training solves
of each iteration run concurrently, and the aggregate time of all training solves
(serial equivalent: wall-clock + sum over iterations of sum(t_k) - max(t_k)).
Per-solve times come from the job logs; DD reuses the global first-iteration
snapshots, as in the offline times.  At omega = 60 the training ran on four workers
and spans two jobs: 43935834 accepted seven first-iteration solves and stopped at
mu = 0.6 under the 40-cycle limit; 44057113 reused those seven and computed the rest.
Only the aggregate (sum of the accepted training-solve times) is defined there.
Online times are means over the three common queries; at omega = 48 they are the
repeated timings of the ten-query study (Table 5).  The script also writes the body
of the frequency table (tab:3d-frequency).
"""
import csv
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / 'outputs/scnet_3d_20260909/remote_v7/logs'
FREQ = ROOT / 'outputs/waveguide_frequency_final_20260913/waveguide_frequency.csv'
OUT = ROOT / 'outputs/waveguide_break_even_20261023'
JOBS = {'24': '43868638', '36': '43864386', '48': '43864387'}
VALIDATION = ROOT / 'outputs/jcp_revision_experiments_20260914/cloud/analysis/summary.json'
REPORTS = {'24': 'remote_v7/full_w24_p8_t4_b2_sr40_or40_sm40_om40_c12000',
           '36': 'remote_v7/full_w36_p8_t4_b2_sr40_or40_sm40_om40_c12000',
           '48': 'remote_v7/full_w48_p8_t4_b2_sr40_or40_sm40_om40_c12000',
           '60': 'remote_recovery_w60'}
W60_FIRST = ROOT / 'outputs/experiment23_revision_20260911/omega60_status.log'  # job 43935834
W60_RECOVERY = ROOT / 'outputs/omega60_final_revision_20260913/omega60_stdout.log'  # job 44057113


def training(log):
    text = log.read_text()
    parts = re.split(r'--- Training (GLOBAL|DD) trajectory-aware ROM ---', text)
    out = {}
    for style, body in zip(parts[1::2], parts[2::2]):
        layers, cur = [], []
        for line in body.splitlines():
            m = re.match(r'snapshot mu=\S+ it=\d+ true=\S+ wall=([0-9.]+)s', line)
            if m:
                cur.append(float(m.group(1)))
            elif line.startswith('layer ') and 'snapshots' in line:
                layers.append(cur)
                cur = []
        wall = float(re.search(r'%s offline ([0-9.]+)s' % style, body).group(1))
        out[style.lower()] = (layers, wall)
    if not out['dd'][0][0]:
        out['dd'][0][0] = out['global'][0][0]
    return out


def omega60():
    first = [float(w) for true, w in re.findall(r'snapshot mu=\S+ it=\d+ true=(\S+) wall=([0-9.]+)s', W60_FIRST.read_text())
             if float(true) <= 1e-7]
    assert len(first) == 7
    t = training(W60_RECOVERY)
    out = {}
    for method in ['dd', 'global']:
        layers, _ = t[method]
        assert [len(layer) for layer in layers] == [1, 8, 8]
        out[method] = sum(first) + sum(map(sum, layers))
    return out


def online_means(freq):
    out = {o: {m: float(r[m + '_online_s']) for m in ['global', 'dd', 'shifted']} for o, r in freq.items()}
    common = [q for q in json.load(VALIDATION.open())['per_query'] if q['query_kind'] == 'original_repeated']
    assert sorted(q['parameter'] for q in common) == [0.237, 0.317, 0.483]
    for m, key in [('global', 'global3'), ('dd', 'dd3'), ('shifted', 'shifted_mg')]:
        out['48'][m] = sum(q['methods'][key]['mean_total_s'] for q in common) / 3
    return out


def sci(x):
    mantissa, exponent = f'{x:.1e}'.split('e')
    return rf'\({mantissa}\times10^{{{int(exponent)}}}\)'


def frequency_table(online, offline):
    lines = []
    for omega in ['24', '36', '48', '60']:
        tests = json.load((ROOT / 'outputs/scnet_3d_20260909' / REPORTS[omega] / 'report.json').open())['tests']
        rho = {m: sum(t[m]['rom_residual'] for t in tests) / 3 for m in ['global', 'dd']}
        steps = {m: sum(t[m]['iterations'] for t in tests) / 3 for m in ['global', 'dd', 'shifted']}
        cells = [omega, sci(rho['global']), sci(rho['dd'])]
        cells += [f'{steps[m]:.1f}' for m in ['global', 'dd', 'shifted']]
        cells += [f'{online[omega][m]:.2f}' for m in ['global', 'dd', 'shifted']]
        cells += [f'{offline[(omega, m)]:.0f}' for m in ['global', 'dd']]
        lines.append(' & '.join(cells) + r' \\')
    return '\n'.join(lines) + '\n'


def main():
    freq = {r['omega']: r for r in csv.DictReader(FREQ.open())}
    online = online_means(freq)
    rows = []
    for omega, job in JOBS.items():
        r, t = freq[omega], training(LOGS / f'helm3d_{job}.out')
        for method in ['dd', 'global']:
            layers, wall = t[method]
            assert abs(wall - float(r[method + '_offline_s'])) < 0.01
            assert all(len(layer) == 8 for layer in layers)
            serial = wall + sum(sum(layer) - max(layer) for layer in layers)
            saving = online[omega]['shifted'] - online[omega][method]
            rows.append({'omega': omega, 'method': method, 'offline_wall_s': wall,
                         'offline_serial_s': serial, 'online_saving_s': saving,
                         'break_even_wall': math.ceil(wall / saving),
                         'break_even_serial': math.ceil(serial / saving)})
    for method, serial in omega60().items():
        saving = online['60']['shifted'] - online['60'][method]
        rows.append({'omega': '60', 'method': method, 'offline_wall_s': '',
                     'offline_serial_s': serial, 'online_saving_s': saving,
                     'break_even_wall': '', 'break_even_serial': math.ceil(serial / saving)})
    OUT.mkdir(exist_ok=True)
    with (OUT / 'break_even.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    table = frequency_table(online, {(r['omega'], r['method']): r['offline_serial_s'] for r in rows})
    (OUT / 'frequency_table.tex').write_text(table)
    for row in rows:
        print(row)
    print(table)


if __name__ == '__main__':
    main()
