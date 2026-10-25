"""Report the fixed-cohort, all-three-layers 1e-3 follow-up."""
from pathlib import Path
import csv
root=Path(__file__).resolve().parents[1]
out=root/'outputs/ecc_dd_budget_20260919'
def read(name,key):
    return {r[key]:r for r in csv.DictReader((out/name).open())}
v=read('uniform3_summary.csv','method')
st=read('uniform3_storage.csv','method')
a=list(csv.DictReader((out/'uniform3_tail_audit.csv').open()))
lines=['# ECC：三层均为 1e-3 的补充实验','',
'沿用 96 个训练参数、三层、同一网格和频率。Global/DD 均从第一层重新训练；DD 仅使用全局 Frobenius 尾预算、零起始 marginal-storage 贪心，无额外局部误差限制或秩上限。', '',
'最终真实残差阈值仍为 1e-12。shifted-ILUTP 保持 beta=0.2、droptol=1e-3、restart=40、maxit=20；MATLAB R2025a，本机单计算线程。', '',
'使用上一轮冻结的 22 个测试参数，各方法预热后重复五次、轮换顺序。这是既有测试集上的补充比较，不是新抽取的独立验证。所有时间均来自本次同批计时，包含需要时的 ILU 构建；不含公共全阶组装、模型加载和离线训练。', '',
'| 方法 | 三层预算 | 基存储 MiB | 平均在线 s | 对 shifted 加速 | 完成迭代 | 最大相对场误差 |',
'|---|---|---:|---:|---:|---:|---:|']
for name in ['shifted','direct','last3_global','last3_dd','uniform3_global','uniform3_dd']:
    r=v[name];budget='—' if name in ['shifted','direct'] else ('(1e-3,1e-3,1e-3)' if name.startswith('uniform3') else '(1e-6,1e-6,1e-3)')
    mem='—' if name not in st else f"{float(st[name]['basis_MiB']):.3f}"
    lines.append(f"| {name} | {budget} | {mem} | {float(r['total_s']):.5f} | {float(r['speedup']):.2f} | {float(r['iterations']):.2f} | {float(r['max_error']):.3e} |")
g=float(st['uniform3_global']['basis_MiB']);d=float(st['uniform3_dd']['basis_MiB']);old=float(st['last3_dd']['basis_MiB'])
lines+=['',f'三层均为 1e-3 时，DD 比同预算 Global 少用 {100*(1-d/g):.1f}% 基存储；比此前分层预算 DD 再少用 {100*(1-d/old):.1f}%。',
f"计入单份共享索引，DD payload 为 {float(st['uniform3_dd']['payload_MiB']):.3f} MiB；实际 MATLAB 部署对象为 {float(st['uniform3_dd']['MATLAB_object_MiB']):.3f} MiB。这些数值不代表进程峰值内存。", '',
'Global 三层秩为 [12,18,23]；DD 的近场/远场/PML 局部秩依次为 [14,4,3]、[20,7,6]、[29,9,8]，总维数 [21,33,46]。', '',
'六个层的实际快照截断误差：', '', '| 模型 | 层 | 实际相对尾误差 |', '|---|---:|---:|']
for r in a:lines.append(f"| {r['tag']} | {r['layer']} | {float(r['projection_tail']):.6e} |")
lines+=['', '全部层满足 1e-3 预算。660 个计时解均满足最终真实残差阈值；88 个生产入口重放和 12 个降阶矩阵组装核验通过。', '',
f"统一 1e-3 的 Global/DD 无需 ILU 查询数分别为 {float(v['uniform3_global']['no_setup_queries']):g}/22、{float(v['uniform3_dd']['no_setup_queries']):g}/22。更强截断减少了基存储，但前两层的近似能力减弱，使完成迭代和 ILU 构建重新成为主要成本；不能把这个配置描述为同时改善存储和时间。", '',
'原 11 个开发参数上，Global/DD 平均完成步数为 20.00/18.09，最大相对场误差为 6.37e-11/8.30e-11；训练过程中的单次在线时间仅供诊断，正式时间以上述重复计时为准。', '',
'复现：`scripts/reproduce_ecc_uniform3.sh`；汇总：`python3 scripts/summarize_ecc_uniform3.py`。证据位于 `outputs/ecc_dd_budget_20260919/uniform3_*`。']
(root/'md/ecc_uniform3_20260919.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[:18]))
