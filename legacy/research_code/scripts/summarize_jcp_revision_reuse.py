#!/usr/bin/env python3
"""Summarize the frozen fixed-space reuse comparison without external packages."""
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics as st

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/jcp_revision_experiments_20260914/reuse'


def read(name):
    with (OUT / name).open() as f:
        return list(csv.DictReader(f))


def write(name, rows):
    with (OUT / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def average(rows, field):
    return st.mean(float(x[field]) for x in rows)


def main():
    rows = read('paper_timings.csv')
    expected_methods = {'shifted', 'direct', 'global', 'dd', 'solution_pod7',
                        'solution_pod12', 'solution_pod16', 'solution_pod20',
                        'solution_pod24', 'pooled_pod_dd', 'pooled_pod_global'}
    expected_keys = {(m, q, rep) for m in expected_methods
                     for q in range(1, 24) for rep in (1, 2, 3)}
    keys = [(x['method'], int(x['query']), int(x['repeat'])) for x in rows]
    assert len(rows) == 23 * 11 * 3 == 759
    assert len(set(keys)) == len(keys) and set(keys) == expected_keys
    for x in rows:
        assert math.isfinite(float(x['true_residual'])) and float(x['true_residual']) <= 1e-12
        assert math.isfinite(float(x['relative_error']))
        assert math.isfinite(float(x['total_s'])) and float(x['total_s']) > 0
        for field in ('parameter', 'ROM_s', 'setup_s', 'solve_s', 'completion_iterations', 'flag', 'factor_bytes'):
            assert math.isfinite(float(x[field]))
        if x['method'] == 'direct':
            assert math.isnan(float(x['post_ROM_residual']))
        else:
            assert math.isfinite(float(x['post_ROM_residual']))
    offline = read('offline_timings.csv')
    offline_keys = [(x['method'], int(x['repeat'])) for x in offline]
    assert len(offline) == 9 * 3 and len(set(offline_keys)) == 27
    assert set(offline_keys) == {(m, rep) for m in expected_methods - {'shifted', 'direct'} for rep in (1, 2, 3)}
    assert all(math.isfinite(float(x['offline_total_s'])) and float(x['offline_total_s']) > 0 for x in offline)
    storage = {x['method']: x for x in read('storage.csv')}
    checks = read('paper_prefix_checks.csv')
    assert len(checks) == 46
    assert {(x['method'], int(x['query'])) for x in checks} == {(m, q) for m in ('global', 'dd') for q in range(1, 24)}
    assert all(math.isfinite(float(x['prefix_field_difference'])) and float(x['prefix_field_difference']) < 1e-9 for x in checks)
    assert all(math.isfinite(float(x['post_ROM_residual'])) for x in checks)
    methods = list(dict.fromkeys(x['method'] for x in rows))
    preferred = ['shifted', 'direct', 'global', 'dd', 'solution_pod7',
                 'solution_pod12', 'solution_pod16', 'solution_pod20',
                 'solution_pod24', 'pooled_pod_dd', 'pooled_pod_global']
    methods = [m for m in preferred if m in methods]
    summary = []
    for m in methods:
        r = [x for x in rows if x['method'] == m]
        o = [x for x in offline if x['method'] == m]
        s = storage.get(m, {})
        per_query = {}
        for x in r:
            per_query.setdefault(x['query'], []).append(x)
        for group in per_query.values():
            assert len({x['completion_iterations'] for x in group}) == 1
        repeat_means = [average([x for x in r if int(x['repeat']) == rep], 'total_s')
                        for rep in (1, 2, 3)]
        item = dict(method=m, ranks=s.get('ranks', ''),
                    basis_MiB=float(s.get('basis_MiB', 0)),
                    payload_MiB=float(s.get('payload_MiB', 0)),
                    matlab_object_MiB=float(s.get('matlab_object_MiB', 0)),
                    offline_mean_s=average(o, 'offline_total_s') if o else 0,
                    offline_sd_s=st.stdev(float(x['offline_total_s']) for x in o) if len(o) > 1 else 0,
                    online_mean_s=average(r, 'total_s'),
                    online_median_s=st.median(float(x['total_s']) for x in r),
                    online_repeat_mean_sd_s=st.stdev(repeat_means),
                    max_query_timing_sd_s=max(st.stdev(float(x['total_s']) for x in g) for g in per_query.values()),
                    completion_mean=average(r, 'completion_iterations'),
                    completion_max=max(int(x['completion_iterations']) for x in r),
                    completed_without_FOM=sum(int(g[0]['completion_iterations']) == 0 for g in per_query.values()) if m != 'direct' else '',
                    true_residual_max=max(float(x['true_residual']) for x in r),
                    relative_error_max=max(float(x['relative_error']) for x in r),
                    post_ROM_residual_max=max(float(x['post_ROM_residual']) for x in r),
                    query_count=len(per_query), repeats=3)
        assert item['true_residual_max'] <= 1e-12
        assert all(int(x['flag']) == 0 for x in r)
        summary.append(item)
    lookup = {x['method']: x for x in summary}
    for x in summary:
        for base in ('direct', 'shifted'):
            saving = lookup[base]['online_mean_s'] - x['online_mean_s']
            x[f'break_even_vs_{base}_queries'] = math.floor(x['offline_mean_s'] / saving) + 1 if saving > 0 and x['method'] != base else ''
    write('summary.csv', summary)
    costs = [dict(method=x['method'], queries=n,
                  offline_s=x['offline_mean_s'], mean_online_s=x['online_mean_s'],
                  estimated_total_s=x['offline_mean_s'] + n * x['online_mean_s'])
             for x in summary for n in (1, 5, 10, 25, 50, 100, 250, 500)]
    write('total_costs.csv', costs)
    validation = dict(measured_solves=len(rows), prefix_checks=len(checks),
                      max_prefix_field_difference=max(float(x['prefix_field_difference']) for x in checks),
                      max_true_residual=max(x['true_residual_max'] for x in summary),
                      max_relative_error=max(x['relative_error_max'] for x in summary),
                      completion_counts_identical_across_repeats=True,
                      all_flags_zero=True, all_residuals_and_errors_finite=True,
                      full_query_method_repeat_coverage=True, all_timings_finite=True,
                      full_offline_repeat_coverage=True)
    (OUT / 'validation.json').write_text(json.dumps(validation, indent=2) + '\n')
    manifest = json.loads((OUT / 'manifest.json').read_text())
    source_paths = [ROOT / 'code/run_jcp_revision_reuse_benchmark.m',
                    ROOT / 'scripts/summarize_jcp_revision_reuse.py',
                    ROOT / 'code/helm_waveholtz_correction_rom.m',
                    ROOT / 'code/rom_reduced_space_utils.m',
                    ROOT / 'code/rom_arnoldi_step.m',
                    ROOT / 'code/build_shifted_laplacian_preconditioner.m',
                    ROOT / 'code/shifted_laplacian_gmres.m']
    source_paths += [Path(manifest[k]) for k in ('source_cohort', 'source_global', 'source_dd')]
    provenance = {}
    for path in source_paths:
        h = hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                h.update(chunk)
        provenance[str(path)] = dict(bytes=path.stat().st_size, sha256=h.hexdigest())
    (OUT / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    lines = ['# 半空间强复用对照：原始结果汇报', '',
             '此实验仅新增证据，不修改 JCP/JSC 稿件。冻结原半空间 24 个训练参数、23 个插值查询、35,721 个未知量与频率 53.5。', '',
             '## 设计', '',
             '- Global/DD 保持原三层训练模型。新 fixed solution-POD 固定扫描 7、12、16、20、24 维，其中 24 维保留所有训练解方向；所有扫描结果完整报告。',
             '- pooled-POD 将 Global 三层共 72 个逆作用快照逐列归一化后做一次 SVD，构造单个固定 Galerkin 空间。它使用同一训练信息，预算事先由 DD/Global 的持久数值 payload 决定，选最大可容纳整数秩；没有根据测试残差选择秩。',
             '- 数值 payload = 复数双精度基 + 两个仿射降阶矩阵 + DD 共用索引。MATLAB 对象字节另行报告，不包含离线快照、full-order 矩阵或在线工作数组。固定空间的整数秩不能精确填满预算。',
             '- 全部方法使用 shifted-ILUTP（β=0.2，droptol=1e-3）completion，GMRES restart=40、maxit=20，统一真残差 1e-12。每个查询包含所需的预条件器 setup；ROM 已达标时不构造 ILU。',
             '- 本地 MATLAB，单计算线程、串行查询，一次预热加三次计时，方法顺序轮换；计时排除了文件载入和共同全矩阵组装。另对 Global/DD 各三次完整重新训练，仅用于离线计时；正式在线仍使用冻结模型。',
             '- solution-POD 离线包含 24 次直接训练解、SVD 和降阶矩阵组装；pooled-POD 完整计入 Global 三层 TA 训练时间，再加汇总/归一化/SVD 与降阶组装。', '',
             '## 结果', '',
             '| 方法 | 秩 | payload MiB | 离线 s | 平均在线 s | 平均 completion it. | 无 completion 查询 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for x in summary:
        lines.append(f"| {x['method']} | {x['ranks']} | {x['payload_MiB']:.3f} | {x['offline_mean_s']:.3f} | {x['online_mean_s']:.6f} | {x['completion_mean']:.3f} | {x['completed_without_FOM']} |")
    lines += ['', f"全部 {len(rows)} 次计时求解通过；最大真残差 {validation['max_true_residual']:.3e}，最大对直接参考解相对误差 {validation['max_relative_error']:.3e}。{len(checks)} 次冻结 Global/DD 前缀检查的最大场差异 {validation['max_prefix_field_difference']:.3e}。", '',
              '## 对“强复用对照不足”的回应', '']
    for m in ('solution_pod20', 'solution_pod24'):
        x = lookup[m]
        lines.append(f"- {m}：{x['completed_without_FOM']}/23 个查询无 completion，平均在线 {x['online_mean_s']:.6f} s，离线 {x['offline_mean_s']:.3f} s，payload {x['payload_MiB']:.3f} MiB；相对 Global 在线降低 {100*(1-x['online_mean_s']/lookup['global']['online_mean_s']):.1f}%，payload 降低 {100*(1-x['payload_MiB']/lookup['global']['payload_MiB']):.1f}%。")
    lines += ['- 这份结果应完整纳入后续研究判断。7 维 fixed-POD 只能展示严格截断后的解空间，不足以代表有竞争力的固定空间复用方法；当前低频二维算例不能仅据其对比宣称逐层 TA 优于充分训练的固定 solution-POD。',
              '- pooled-POD26/36 采用预先固定的逐列归一化汇总策略；其结果只反映这一构造规则，不能推广为所有单一固定空间都会较差。它包含 TA 离线训练成本，解释时必须与仅使用训练解的 solution-POD 分开。', '',
              '## 总成本口径', '',
              'total_costs.csv 给出离线均值 + 查询数 × 在线均值；summary.csv 给出相对 direct/shifted 的估算摊销查询数。此处是同一本地单线程环境的墙钟时间模型，依赖当前参数族、重复查询分布及实现；不包含全矩阵组装和模型载入。', '',
              '## 证据边界', '',
              '完整 solution-POD 与 pooled-POD 是比原 7 维基线更充分的固定空间对照。pooled-POD 仍依赖 TA 训练，因此它主要检验在线逐层使用是否必要，不是与 TA 无关的廉价离线替代品。此实验没有实现 GCRO-DR/recycling，也不能单凭低频二维结果确认高频三维优势。', '',
              'solution_spectrum.csv 显示 σ16/σ1=4.83e-14、σ17/σ1=4.48e-15，而第18–24模态约为7.31e-16到3.75e-16。POD20/24包含接近浮点噪声的尾部方向；它们在这23个插值点上的实测通过仍然成立，但这些尾部方向不能直接解释为稳定物理模态。POD16本身也在22/23个查询无需completion，完整报告保留这一更小空间及其单个补全开销。', '',
              '原始记录：paper_timings.csv、offline_timings.csv；模型和全部最终场：models.mat、paper_results.mat；预注册规则：manifest.json；检查：validation.json；源文件校验：provenance.json。', '']
    (OUT / 'results.md').write_text('\n'.join(lines))
    print(json.dumps(validation, indent=2))


if __name__ == '__main__':
    main()
