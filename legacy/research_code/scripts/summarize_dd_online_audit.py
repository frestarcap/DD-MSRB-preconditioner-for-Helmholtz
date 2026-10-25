#!/usr/bin/env python3
"""Validate and summarize the separate paired DD implementation timing cohort."""
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/dd_online_audit_20260910'
rows = list(csv.DictReader((OUT / 'paired_timings.csv').open()))
checks = list(csv.DictReader((OUT / 'equivalence.csv').open()))
assert len(rows) == 400 and len(checks) == 40
assert all(x['bitwise_equal'] == '1' for x in checks)
assert all(float(x['true_residual']) <= 1e-10 for x in rows)
groups = defaultdict(list)
for row in rows:
    groups[(row['scale'], row['variant'], row['optimized'])].append(row)
summary = []
for (scale, variant, optimized), group in groups.items():
    assert len(group) == 25
    assert len({(x['c_right'], x['repetition']) for x in group}) == 25
    no_completion = [x for x in group if float(x['completion_iterations']) == 0]
    completed = sorted({float(x['c_right']) for x in group if float(x['completion_iterations']) > 0})
    assert completed == ([0.8053] if scale == '4' else [])
    summary.append(dict(
        omega=53.5*float(scale), variant=variant, optimized=int(optimized),
        mean_online_s=statistics.mean(float(x['total_s']) for x in group),
        mean_prefix_s=statistics.mean(float(x['prefix_s']) for x in group),
        mean_no_completion_online_s=statistics.mean(float(x['total_s']) for x in no_completion),
        completion_queries=completed,
        maximum_true_residual=max(float(x['true_residual']) for x in group)))
summary.sort(key=lambda x: (x['omega'], x['optimized'], x['variant']))
(OUT/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
with (OUT/'summary.csv').open('w') as stream:
    writer=csv.DictWriter(stream, fieldnames=list(summary[0]))
    writer.writeheader();writer.writerows(summary)
for omega in [53.5,80.25,107,214]:
    for opt in [0,1]:
        selected={x['variant']:x for x in summary if x['omega']==omega and x['optimized']==opt}
        g,d=selected['global'],selected['dd']
        print(f"omega={omega:g} optimized={opt}: Global={g['mean_online_s']:.6f}s "
              f"DD={d['mean_online_s']:.6f}s DD/Global={d['mean_online_s']/g['mean_online_s']:.4f}; "
              f"no completion G/DD={g['mean_no_completion_online_s']:.6f}/{d['mean_no_completion_online_s']:.6f}s")
print('PASS: 400 timings, 40 bitwise-equal solution/Arnoldi comparisons; all final residuals <= 1e-10.')
