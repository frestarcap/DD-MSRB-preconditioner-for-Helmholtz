"""Write the manuscript-wide frozen-spectrum DP audit report."""
from collections import defaultdict
from pathlib import Path
import csv
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/greedy_dp_paper_20260921'


def main():
    data=json.loads((OUT/'results.json').read_text());layers=data['results']
    models=list(csv.DictReader((OUT/'models.csv').open()))
    by_tag={r['tag']:r for r in models}
    groups=defaultdict(list)
    for r in models:groups[r['group']].append(r)
    assert len(models)==32 and len(layers)==92
    assert all(r['dp_tail']<=r['eta'] and r['dp_entries']<=r['greedy_entries'] for r in layers)
    assert all(r['saved_entries']<=r['gap_bound_entries']+1e-8 for r in layers)
    max_model=max(models,key=lambda r:float(r['savings_percent']))
    max_layer=max(layers,key=lambda r:r['savings_percent'])
    names={
        'halfspace_main':'半空间主算例', 'halfspace_training':'半空间训练数量（6/12/24）',
        'halfspace_frequency':'半空间频率（53.5/80.25/107/214）',
        'halfspace_partition':'半空间补充分区（merged/signed/tiles）',
        'halfspace_two_parameter':'半空间双参数完整扫描', 'ecc_main':'ECC 当前主配置',
        'waveguide_frequency':'三维波导四个频率', 'waveguide_threshold':'附录波导阈值 P1–P4'}
    lines=['# 当前 JCP 全部 DD 实验的 Greedy / 精确 DP 审计（2026-09-21）','',
        '## 结论','',
        '对当前 JCP 三个主实验及其训练数量、分区、频率、双参数和附录阈值研究，'
        '共检查 32 份冻结模型、92 个层分配记录。无需重新训练；所有比较都固定原有的每层快照、分区和误差预算。','',
        '**没有发现很大的存储差距。** 按每个模型各层存储求和，DP 相比 Greedy 还能减少的基存储为 '
        f"0–{float(max_model['savings_percent']):.6f}%；最大值来自波导阈值 P2。"
        '若以最优存储为分母，Greedy 的最大超额为 1.019261%。','',
        f"92 层中 {sum(r['saved_entries']==0 for r in layers)} 层的 Greedy 已经存储最优；"
        f"逐层最大可节省比例为 {max_layer['savings_percent']:.6f}%，来自 ECC 第一层。",
        '', '本文的“节省”统一定义为 `(S_g - S_DP) / S_g`，不是 DD 相对 Global 的节省。'
        '目标只计复数双精度局部基的条目数（16 bytes/entry），不优化投影矩阵、索引、'
        '运行峰值内存或在线时间。','',
        '## 全部研究覆盖','',
        '| 研究 | 保存模型数 | 层数 | DP 可再节省总基存储 |',
        '|---|---:|---:|---:|']
    for group in names:
        rs=groups[group];values=[float(r['savings_percent']) for r in rs]
        interval=f'{min(values):.4f}%–{max(values):.4f}%' if len(rs)>1 else f'{values[0]:.4f}%'
        lines.append(f"| {names[group]} | {len(rs)} | {sum(int(r['layers']) for r in rs)} | {interval} |")
    lines += ['',
        '分区表中的 material/depth 模型就是半空间 frequency_1，已在频率组检查，'
        '没有重复计入补充分区。双参数最终验证采用扫描中已检查的 `explore_w05_n8_e-4`。'
        '波导追加参数、重复计时和误差/MG诊断使用同一批冻结频率模型，不产生新的秩分配。'
        'Global POD 只有一个块，其最小满足预算的截断秩已直接给出存储最优解；'
        'NR、固定 POD 和全阶基线不属于本次局部 Greedy 分配目标。','',
        '## 主实验的存储数值','',
        '| 模型 | Greedy basis MiB | 精确 DP basis MiB | DP 可再节省 |',
        '|---|---:|---:|---:|']
    for tag,label in [('halfspace_main','半空间主算例'),('ecc_uniform5','ECC 主算例'),
                      ('waveguide_w24','波导 ω=24'),('waveguide_w36','波导 ω=36'),
                      ('waveguide_w48','波导 ω=48'),('waveguide_w60','波导 ω=60')]:
        r=by_tag[tag]
        lines.append(f"| {label} | {float(r['greedy_basis_MiB']):.6f} | {float(r['dp_basis_MiB']):.6f} | {float(r['savings_percent']):.6f}% |")
    lines += ['',
        'ω=60 的绝对减少量最大，为 14.009476 MiB，但相对基存储仅 0.655654%。'
        '此模型各层总秩从 `[121,174,195]` 变为 `[120,175,197]`；存储更少也可能'
        '伴随更大的总降阶维数。ECC 第一层的 `[27,8,6] -> [29,7,6]` 也是这种情况。','',
        '## 不重新训练能回答什么','',
        '本次精确回答：对论文已经生成的每一层快照，当前 Greedy 相比满足同一谱尾预算的'
        '最小基存储分配，多存了多少。已保存局部谱的模型可直接使用；'
        '19 份半空间补充模型的部署对象删去了局部谱，但保存了 `audit.E`，'
        '因此对其 57 层重新计算局部 SVD。没有重新求解任何训练逆作用，也没有重新生成轨迹。','',
        '表中的 DP 多层合计是静态合计。若真正以 DP 替换训练过程的第一层，'
        '后续层快照通常会变化，需要完整重训才能测新模型的总存储、在线迭代和时间。'
        '这次没有将静态最优分配当成一个已经验证的重新训练模型。','',
        '## 精确 DP 与验证','',
        '成本状态保存最小丢弃尾能量；同成本保留最小尾，随后删除被更低成本、'
        '更小或相同尾能量支配的状态。所有能量由 binary64 奇异值精确平方得到，'
        '再用公共二进制分母转成 Python 整数，避免能量量化和紧容差下的大数消减。', '',
        '为减少状态数，使用 Greedy 的 LP 松弛给出严格删枝界。设最后模式能量/成本为 '
        '`a*, n*`，`T_b(r)` 为块尾能量，定义','',
        '```text\nP_b(r) = a* n_b r + n* T_b(r)\nm_b = min_r P_b(r)\nW = a* S_g + n* tau - sum_b m_b\n```','',
        '任何满足 `S<=S_g` 且 `D<=tau` 的整数解，都必须满足 '
        '`sum_b(P_b(r_b)-m_b)<=W`。因每项非负，可安全删除超出这个界的块选项及部分状态。'
        '这不会删除最优解；不是基于误差容差或经验阈值的近似删枝。','',
        '- 203 个已知反例、极小尾能量、零谱和固定种子随机问题，均通过完整秩组合枚举。',
        '- 92/92 层的精确算术 Greedy 秩与冻结模型完全一致，包括重新计算 SVD 的 57 层。',
        '- 3/10/16 块实例与无 LP 删枝的原始精确成本 DP 核对；三块问题另有直接秩枚举。',
        '- 所有 37 块实例与反向块顺序 DP、独立分支枚举核对；后者不使用成本状态表或支配删枝。',
        '- 92/92 层满足原预算，且实际存储差距均不超过 Greedy 的 LP 加性界。','',
        f"本轮带严格界的 DP 最大保留前沿为 {max(max(r['dp_stats']['frontiers']) for r in layers)} 个状态。"
        f"本机单次全体 DP 内核计时合计 {sum(r['dp_seconds'] for r in layers):.6f} 秒，"
        f"单层最大 {max(r['dp_seconds'] for r in layers):.6f} 秒；"
        '不包含输入读取、SVD、精确整数转换、独立核验或程序启动。这不是正式性能基准，'
        '但表明本批谱分配规模很小，精确 DP 在这些实例上可行。','',
        '“精确”针对保存或重算得到的浮点局部谱；不声称恢复原快照的无限精度奇异值。'
        '重复早期层仍作为相应配置的记录保留，92 层不应被称为 92 个独立问题族。','',
        '## 完整模型表','',
        '| 配置 | 层数/块数 | Greedy basis MiB | DP basis MiB | 可再节省 |',
        '|---|---:|---:|---:|---:|']
    for r in models:
        lines.append(f"| {r['tag']} | {r['layers']}/{r['blocks']} | {float(r['greedy_basis_MiB']):.6f} | {float(r['dp_basis_MiB']):.6f} | {float(r['savings_percent']):.6f}% |")
    lines += ['', '## 数据、资源与复现','',
        '论文范围依据 `JCP/elsarticle-template-num.tex` 及其 `reviewer_supplement.tex`、'
        '`reviewer_complete_scan.tex`、`reviewer_parameter_partition_tables.tex`、'
        '`waveguide_threshold_appendix.tex` 等输入文件。所有精确输入和结果位于 '
        '`outputs/greedy_dp_paper_20260921/`。','',
        '- `local_inputs.json` / `remote_inputs.json`：局部谱 binary64 原码、块大小、预算、原秩和源路径。',
        '- `local_inputs_manifest.json`：本地 28 份模型的显式清单。',
        '- `results.json` / `layers.csv`：92 层的 Greedy/DP 秩、谱尾、LP 证书和核验方式。',
        '- `models.csv`：32 份模型静态合计。输入 SHA-256 保存在 `results.json`。',
        '- `full_dp.log`：逐层通过记录；`job_accounting.txt` / `resource_usage.txt`：超算资源记录。','',
        '三维频率谱由 <cluster> Slurm 作业 **44852123** 只读导出：节点 `a02r3n19`，'
        '申请 8 CPUs/28 GiB，MATLAB R2024a 使用 1 计算线程；作业 `COMPLETED / 0:0`，'
        '用时 3 分 50 秒，Slurm batch MaxRSS 6500564 KiB（约 6.20 GiB）。'
        '远端目录 `${REMOTE_RUN_ROOT}/greedy_dp_paper_20260921`。', '',
        '```sh\n# 本地提取；环境需 h5py、NumPy\nOPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \\\n  outputs/greedy_dp_paper_20260921/tooling/bin/python scripts/export_paper_rank_inputs.py\n'
        '# remote_inputs.json 由已记录的只读 Slurm 提取作业产生\n'
        'python3 scripts/audit_paper_greedy_dp.py \\\n  outputs/greedy_dp_paper_20260921/local_inputs.json \\\n  outputs/greedy_dp_paper_20260921/remote_inputs.json\n'
        'python3 scripts/summarize_paper_greedy_dp.py\n```','',
        '本轮没有修改生产分配器、训练模型、论文或数值图；未运行在线性能试验，也未运行 LaTeX 编译。']
    text='\n'.join(lines)+'\n'
    (OUT/'report.md').write_text(text)
    (ROOT/'md/greedy_dp_all_jcp_experiments_20260921.md').write_text(text)
    verification=dict(models=32,layers=92,greedy_rank_match=92,budget_pass=92,
        optimal_greedy_layers=sum(r['saved_entries']==0 for r in layers),
        max_model_savings_percent=float(max_model['savings_percent']),
        max_layer_savings_percent=max_layer['savings_percent'],
        arithmetic='exact integer arithmetic on binary64 local spectra',
        source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in [OUT/'local_inputs.json',OUT/'remote_inputs.json',OUT/'results.json']})
    (OUT/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    print(json.dumps(verification,indent=2))


if __name__=='__main__':main()
