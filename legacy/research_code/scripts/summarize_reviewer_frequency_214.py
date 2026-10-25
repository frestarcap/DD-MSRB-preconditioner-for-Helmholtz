#!/usr/bin/env python3
"""Validate and summarize all Global/DD omega-214 algebraic timing samples."""
import csv
import json
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/reviewer_20260909'
FOLDER = OUT / 'frequency_4'
QUERIES = [0.8053, 0.8297, 0.8761, 0.9229, 0.9431]


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def main():
    training = {r['variant']: r for r in read(OUT / 'summary_train_frequency_4.csv')}
    assert set(training) == {'global', 'dd'}
    for row in training.values():
        assert int(row['N']) == 570025 and int(row['training_count']) == 24
        assert float(row['max_tail']) <= 1.001e-5
        assert float(row['max_snapshot_residual']) < 1e-9
    audit = read(FOLDER / 'model_validation.csv')
    assert len(audit) == 18
    assert all(float(r['affine_projection_defect']) < 1e-10 and
               float(r['orthogonality_defect']) < 1e-10 and
               float(r['snapshot_residual']) < 1e-9 for r in audit)
    details = []
    aggregate = {}
    for method in ('global', 'dd', 'direct', 'shifted'):
        rows = read(FOLDER / f'{method}_timing.csv')
        assert len(rows) == 25
        assert all(float(r['c_left']) == 1 and float(r['true_residual']) <= 1e-10 for r in rows)
        for mu in QUERIES:
            group = [r for r in rows if float(r['c_right']) == mu]
            assert sorted(int(r['repetition']) for r in group) == [1, 2, 3, 4, 5]
            details.append(dict(method=method, c_right=mu,
                mean_online_s=mean(float(r['total_s']) for r in group),
                sd_online_s=stdev(float(r['total_s']) for r in group),
                mean_completion_iterations=mean(float(r['completion_iterations']) for r in group),
                max_true_residual=max(float(r['true_residual']) for r in group)))
        aggregate[method] = dict(
            mean_online_s=mean(float(r['total_s']) for r in rows),
            mean_completion_setup_s=mean(float(r['completion_setup_s']) for r in rows),
            mean_completion_iterations=mean(float(r['completion_iterations']) for r in rows),
            max_true_residual=max(float(r['true_residual']) for r in rows),
            max_within_query_sd_s=max(r['sd_online_s'] for r in details if r['method'] == method))
        if method in training:
            aggregate[method].update({k:float(training[method][k]) for k in
                ('basis_MiB','operators_MiB','index_MiB','payload_MiB','object_MiB','offline_s')})
            aggregate[method]['ranks'] = training[method]['ranks']
            aggregate[method]['queries_needing_completion'] = sum(
                r['mean_completion_iterations'] > 0 for r in details if r['method'] == method)
    g, d = aggregate['global'], aggregate['dd']
    per_query = {(r['method'],r['c_right']):r for r in details}
    reduced_queries = [mu for mu in QUERIES if all(
        per_query[m,mu]['mean_completion_iterations']==0 for m in ('global','dd'))]
    for method in ('global','dd'):
        aggregate[method]['mean_online_s_on_common_no_completion_queries'] = (
            mean(per_query[method,mu]['mean_online_s'] for mu in reduced_queries)
            if reduced_queries else None)
    result = dict(methods=aggregate,
        dd_storage_saving_percent=100*(1-d['payload_MiB']/g['payload_MiB']),
        dd_online_time_reduction_percent=100*(1-d['mean_online_s']/g['mean_online_s']),
        dd_faster_query_count=sum(per_query['dd',mu]['mean_online_s'] <
            per_query['global',mu]['mean_online_s'] for mu in QUERIES),
        common_no_completion_queries=reduced_queries,
        verified_timing_samples=100, query_count=5, repetitions_per_query=5)
    with (FOLDER / 'per_query_summary.csv').open('w') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(details[0]))
        writer.writeheader(); writer.writerows(details)
    (FOLDER / 'comparison_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    print('FREQUENCY_4_TIMING_VALIDATION_PASS: 100 recorded samples')


if __name__ == '__main__':
    main()
