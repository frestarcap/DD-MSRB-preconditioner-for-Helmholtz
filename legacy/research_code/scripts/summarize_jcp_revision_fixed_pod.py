"""Summarize the independent cached-solution omega48 fixed-POD experiment."""
import csv
import hashlib
import json
import math
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/jcp_revision_experiments_20260914/fixed3d'


def read_csv(path):
    with path.open() as file:
        return list(csv.DictReader(file))


def f(row, key):
    return float(row[key])


def seq(value):
    return value if isinstance(value, list) else [value]


def main():
    source = OUT / 'results'
    prep = json.loads((source / 'prepare_complete.json').read_text())
    done = json.loads((source / 'production_complete.json').read_text())
    rows = read_csv(source / 'timings.csv')
    cache = read_csv(source / 'cache_validation.csv')
    input_hashes = dict(reversed(line.split(maxsplit=1)) for line in (OUT / 'input_hashes.txt').read_text().splitlines())
    measured = [r for r in rows if r['phase'] == 'measured']
    warm = [r for r in rows if r['phase'] == 'warmup']
    assert prep['passed'] and len(cache) == 8 and len(rows) == 12 and len(measured) == 9 and len(warm) == 3
    assert {int(r['sample']) for r in cache} == set(range(1, 9))
    assert all(abs(f(r, 'parameter') - (.2 + (int(r['sample']) - 1) * .4 / 7)) < 1e-14 for r in cache)
    assert done['measured_solves'] == 9 and done['warmup_solves'] == 3
    names = ['global_pod8', 'dd_pod8', 'dd3']
    assert {(r['method'], f(r, 'parameter')) for r in measured} == {(n, q) for n in names for q in [.237, .317, .483]}
    assert {r['method'] for r in warm} == set(names) and all(f(r, 'parameter') == .317 for r in warm)
    assert len({r['node'] for r in rows}) == 1 and len({r['job_id'] for r in rows}) == 1
    with (OUT / 'audit/slurm_status.txt').open() as file:
        slurm = {r['JobID']: r for r in csv.DictReader(file, delimiter='|')}
    assert all(slurm[str(job)]['State'] == 'COMPLETED' and slurm[str(job)]['ExitCode'] == '0:0' for job in [prep['job_id'], done['job_id']])
    model_check = (OUT / 'prepared_model_hash_check_after.txt').read_text().splitlines()
    assert len(model_check) == 3 and all(line.endswith(': OK') for line in model_check)
    assert {Path(line[:-4]).name for line in model_check} == {'global_pod8.mat', 'dd_pod8.mat', 'dd3.mat'}
    for stage in ['prepare', 'production']:
        audit = OUT / 'audit' / stage
        before = (audit / 'input_before.txt').read_text()
        after = (audit / 'input_after.txt').read_text()
        assert before == after and len(before.splitlines()) == 10
        assert all(line.endswith(': OK') for line in before.splitlines())
        assert (audit / 'matlab_exit_code.txt').read_text().strip() == '0'
    for r in rows:
        assert all(math.isfinite(f(r, k)) for k in ['rom_s', 'setup_s', 'completion_s', 'true_check_s', 'total_s', 'true_residual', 'precompletion_residual', 'completion_iterations'])
        assert all(f(r, k) >= 0 for k in ['rom_s', 'setup_s', 'completion_s', 'true_check_s', 'total_s'])
        assert f(r, 'total_s') >= sum(f(r, k) for k in ['rom_s', 'setup_s', 'completion_s', 'true_check_s'])
        assert f(r, 'converged') == 1 and f(r, 'true_residual') <= 1e-6
    assert all(math.isfinite(f(r, 'recomputed_true_residual')) and f(r, 'recomputed_true_residual') <= 1e-7 * (1 + 1e-6) for r in cache)
    models = {m['method']: m for m in prep['models']}
    assert seq(models['global_pod8']['layer_ranks']) == [8]
    assert max(seq(models['dd_pod8']['block_ranks'])) <= 8
    assert all(models[n]['snapshot_projection_error'] < 1e-11 and models[n]['operator_projection_error'] < 1e-10 for n in names[:2])
    prefix = prep['dd3_prefix_checks']
    assert len(prefix) == 3 and {p['parameter'] for p in prefix} == {.237, .317, .483}
    assert all(math.isfinite(p['observed']) and abs(p['observed'] - p['archived']) <= max(1e-11, abs(p['archived']) * 1e-5) for p in prefix)
    summary = []
    for name in names:
        rs = [r for r in measured if r['method'] == name]
        meta = models[name]
        summary.append(dict(method=name, layer_ranks=json.dumps(meta['layer_ranks']), basis_MiB=meta['basis_MiB'],
                            payload_MiB=meta['payload_MiB'], model_object_MiB=meta['model_object_MiB'],
                            construction_svd_s=meta['construction_svd_s'], construction_reduced_assembly_s=meta['construction_reduced_assembly_s'],
                            mean_online_s=st.mean(f(r, 'total_s') for r in rs), mean_iterations=st.mean(f(r, 'completion_iterations') for r in rs),
                            max_true_residual=max(f(r, 'true_residual') for r in rs), max_post_ROM_residual=max(f(r, 'precompletion_residual') for r in rs),
                            accepted=sum(f(r, 'true_residual') <= 1e-6 for r in rs)))
    with (OUT / 'summary.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=summary[0].keys())
        writer.writeheader(); writer.writerows(summary)
    look = {s['method']: s for s in summary}; dd = look['dd3']
    passed = sum(f(r, 'true_residual') <= 1e-6 for r in rows)
    lines = ['# 高频三维强固定空间对照：简要结果', '',
             f"ω=48，N=5,543,003；生产节点 `{rows[0]['node']}`，job `{rows[0]['job_id']}`。原三查询，每方法一次正式测量；"
             '测量前在μ=.317各完成一次完整预热。方法顺序轮换，比较来自同一个节点。本结果是配对诊断，未进行重复计时，不能给出统计稳健的提速结论。', '',
             '## 设计与验收', '',
             '训练信息仅来自原有8个训练参数的第一层逆作用缓存，没有重解训练样本，没有使用测试解选择秩。'
             '缓存求解的是共同归一化右端项 b/||b||，保留原解之间的相对大小；不存在逐列解范数归一化。', '',
             '- Global-POD8：保留8个数值非零训练方向，纯Galerkin初值。',
             '- DD-POD8：原37块逐块做解POD，仅按1e-13相对奇异值删除数值零方向；每块至多8维，全局耦合Galerkin初值。',
             '- DD3：原冻结三层trajectory-aware DD-ROM，replay先与历史三点残差核验。', '',
             '三种初值之后使用同一shifted-MG completion，β=2，coarse cap=12000，CGS2 MEX，restart40/max40；'
             '所有完成结果以原始Helmholtz方程真实残差≤1e-6验收。缓存训练解另以1e-7验收。', '',
             '完整在线时间包含ROM、必要的MG setup、completion和最终真实残差检查；共同全阶矩阵组装及模型加载不计入。', '',
             '## 存储与新构造成本', '',
             '| 方法 | 秩 | 基MiB | payload MiB | MATLAB对象MiB | 新SVD s | 新降阶算子组装 s |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for s in summary:
        a = '-' if s['construction_svd_s'] is None else f"{s['construction_svd_s']:.3f}"
        b = '-' if s['construction_reduced_assembly_s'] is None else f"{s['construction_reduced_assembly_s']:.3f}"
        lines.append(f"| {s['method']} | {s['layer_ranks']} | {s['basis_MiB']:.3f} | {s['payload_MiB']:.3f} | {s['model_object_MiB']:.3f} | {a} | {b} |")
    cost = prep['training_cost']
    lines += ['', 'payload计算复数基、两个持久降阶算子和一次共享索引。对象大小由MATLAB whos测量，均不是完整进程内存。', '',
              f"原8个训练解的历史并行阶段墙钟时间为 {cost['historical_shared_first_wall_s']:.3f} s，"
              f"原worker求解墙钟时间之和为 {cost['historical_sum_worker_wall_s']:.3f} s。"
              '原训练采用8个worker、每worker4线程，新构造使用单进程8线程，因此旧解成本、新构造成本和缓存读取必须分列。'
              '历史shared_first和stage1 snapshot wall描述同一阶段，不能相加；本次缓存读取不能充作完整离线成本。', '',
              '## 每查询正式结果', '',
              '| μ | 方法 | post-ROM残差 | MG迭代数 | ROM s | setup s | completion s | 完整时间 s | 最终真实残差 |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for q in [.237, .317, .483]:
        for name in names:
            r = next(r for r in measured if f(r, 'parameter') == q and r['method'] == name)
            lines.append(f"| {q} | {name} | {f(r,'precompletion_residual'):.3e} | {int(f(r,'completion_iterations'))} | "
                         f"{f(r,'rom_s'):.3f} | {f(r,'setup_s'):.3f} | {f(r,'completion_s'):.3f} | {f(r,'total_s'):.3f} | {f(r,'true_residual'):.3e} |")
    lines += ['', '## 对强固定空间质疑的回应', '']
    for n in names[:2]:
        s = look[n]
        lines.append(f"{n}：相对冻结DD3，payload减少 {100*(1-s['payload_MiB']/dd['payload_MiB']):.2f}%；"
                     f"三查询平均完整时间 {s['mean_online_s']:.3f} s（DD3为{dd['mean_online_s']:.3f} s），"
                     f"平均completion迭代数 {s['mean_iterations']:.3f}（DD3为{dd['mean_iterations']:.3f}）。")
        lines.append('')
    best = min(summary, key=lambda s: s['mean_online_s'])
    by_pair = {(r['method'], f(r, 'parameter')): r for r in measured}
    ratios = [f(by_pair['dd_pod8', q], 'total_s') / f(by_pair['dd3', q], 'total_s') for q in [.237, .317, .483]]
    fewer_iterations = [f(by_pair['dd_pod8', q], 'completion_iterations') < f(by_pair['dd3', q], 'completion_iterations') for q in [.237, .317, .483]]
    lines += [f"这组诊断中平均完整时间最低的是 `{best['method']}`。DD-POD8在 {sum(v < 1 for v in ratios)}/3 个查询中完整时间较短，"
              f"在 {sum(fewer_iterations)}/3 个查询中completion迭代较少；按μ=.237/.317/.483顺序，DD-POD8/DD3完整时间比为 "
              + '/'.join(f'{v:.4f}' for v in ratios) + '。', '']
    if all(v < 1 for v in ratios) and all(fewer_iterations):
        lines += ['这三个预设查询中，固定局部解空间同时具有更小payload和更少completion迭代，三次配对完整计时也均较短。'
                  '因此本例没有显示三层轨迹训练超过完整固定局部解空间的额外收益；应把固定局部solution-POD纳入正式基线，'
                  '收窄本例对trajectory-aware优势的表述。单次测量仍不足以支持统计稳健的提速结论。', '']
    else:
        lines += ['应保留每个查询的相反结果，不能用三点平均排名声称一种训练方式普遍优越。'
                  '本结论限于当前固定空间和三点单次诊断。', '']
    global_warm = next(r for r in warm if r['method'] == 'global_pod8')
    global_measured = by_pair['global_pod8', .317]
    lines += [f"计时边界的一个实际例子：μ=.317的Global-POD8预热与正式求解均为 "
              f"{int(f(global_warm, 'completion_iterations'))} 次completion迭代，完整时间分别为 "
              f"{f(global_warm, 'total_s'):.3f} 和 {f(global_measured, 'total_s'):.3f} s。"
              '因此时间差应与确定记录的迭代和存储结果一起报告。', '',
              'DD-POD与Global-POD使用同8个训练解，局部空间可独立组合各区域系数，因此其总维数更大。'
              '这个对照检验固定局部解空间能否在本组查询替代三层轨迹空间。两者同时改变训练信息、层数和局部秩/存储，'
              '不能据此因果分离这些因素的作用，也不构成特定局部秩分配评分优越性的证明。', '',
              '## 验证及证据', '',
              f"包含预热的 {passed}/{len(rows)} 次求解满足真实残差，最大值 {max(f(r,'true_residual') for r in rows):.5e}；"
              f"8个缓存重新核验最大残差 {max(f(r,'recomputed_true_residual') for r in cache):.5e}。",
              'prepare同时核验全训练快照空间捕获、投影算子耦合以及冻结DD3历史replay。'
              '原10个源文件哈希在两个作业前后检查；3个部署模型在生产前冻结哈希，并于生产后再次核对，全部一致。'
              '科学输入与依赖版本保留，大型部署MAT留在远端独立目录。', '',
              f"训练源MAT SHA-256：`{input_hashes[prep['source']]}`。",
              f"原报告JSON SHA-256：`{input_hashes[prep['source_report']]}`。", '',
              '原始CSV/JSON见 `results/`；资源、退出码与源hash检查见 `audit/`；'
              '完整输入清单见 `input_hashes.txt`，冻结部署模型清单及生产后核验见 '
              '`prepared_model_hashes.txt` 和 `prepared_model_hash_check_after.txt`；Slurm日志见 `logs/`。论文未修改。', '']
    (OUT / 'results.md').write_text('\n'.join(lines))
    manifest = {str(p.relative_to(OUT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file() and p.name != 'validation.json' and p.suffix in {'.json', '.csv', '.txt', '.out', '.err'}}
    validation = dict(measured_solves=len(measured), warmup_solves=len(warm), all_accepted=passed == 12,
                      max_true_residual=max(f(r, 'true_residual') for r in rows), all_same_node=True,
                      cache_count=len(cache), full_numerical_global_rank=8,
                      fixed_dd_rank_max=max(seq(models['dd_pod8']['block_ranks'])), prepared_model_hashes_unchanged=True,
                      source_hashes_unchanged=True, slurm_completed_zero_exit=True, sha256=manifest)
    (OUT / 'validation.json').write_text(json.dumps(validation, indent=2) + '\n')
    assert done['all_converged'] and validation['all_accepted']
    print('JCP_FIXED48_REPORT_PASS')


if __name__ == '__main__':
    main()
