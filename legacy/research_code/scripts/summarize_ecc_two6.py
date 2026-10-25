"""Report fixed-cohort two-layer ablation and measured third-layer costs."""
from pathlib import Path
import csv
root=Path(__file__).resolve().parents[1];out=root/'outputs/ecc_dd_budget_20260919'
def read(f,key):return {r[key]:r for r in csv.DictReader((out/f).open())}
v=read('two6_summary.csv','method');st=read('two6_storage.csv','method')
lines=['# ECC 两层 1e-6 与第三层消融','',
'两种表示均重新训练两层，96 个训练参数、严格全局尾预算、无局部约束或秩上限。Global 秩 [30,46]；DD 各层近场/远场/PML 秩 [31,9,8]、[50,20,13]，总维数 [48,83]。四层实际投影尾误差均满足 1e-6。', '',
'固定先前22个查询，预热后每组5次串行配对计时并轮换顺序；此为既有测试集上的后续消融，不是新独立验证。所有方法真实残差要求1e-12，shifted-ILUTP beta=.2、drop=1e-3、restart40、maxit20不变。本地MATLAB R2025a单计算线程。在线时间含ROM、条件ILU构建和完成求解，不含共同全阶组装、模型加载及离线训练。', '',
'| 方法 | 基 MiB | 在线 s | 对 shifted 加速 | 完成迭代 | 免 ILU 查询 | 最大相对场误差 |','|---|---:|---:|---:|---:|---:|---:|']
for name in ['shifted','direct','two6_global','two6_dd','last3_global','last3_dd']:
 r=v[name];mem='—' if name not in st else f"{float(st[name]['basis_MiB']):.3f}"
 lines.append(f"| {name} | {mem} | {float(r['total_s']):.5f} | {float(r['speedup']):.2f} | {float(r['iterations']):.2f} | {float(r['no_setup_queries']):g}/22 | {float(r['max_error']):.3e} |")
d=float(st['two6_dd']['basis_MiB']);g=float(st['two6_global']['basis_MiB']);d3=float(st['last3_dd']['basis_MiB'])
lines+=['',f'两层DD比同预算Global节省 {100*(1-d/g):.1f}% 基存储；比三层DD少用 {100*(1-d/d3):.1f}%。两层DD在线耗时是三层DD的 {float(v["two6_dd"]["total_s"])/float(v["last3_dd"]["total_s"]):.2f} 倍。', '',
'第三层对收敛不是必要条件：两层仍可由相同完成求解器达到目标。但当前未复用每个参数的ILU因子，第三层使更多查询免去ILU构建，其速度价值不能仅由减少的迭代步数衡量。若业务可复用同一矩阵的ILU因子，应另做摊销比较，不能直接套用本批时间结论。', '',
'| DD 阶段 | 两层 s | 三层 s |','|---|---:|---:|']
for field in ['ROM_s','setup_s','solve_s']:lines.append(f'| {field} | {float(v["two6_dd"][field]):.5f} | {float(v["last3_dd"][field]):.5f} |')
lines+=['', '全部660个计时解通过残差验收；88个生产入口重放及10个降阶矩阵组装检查通过。模型存储不等同于进程峰值内存。', '',
'关于不截断：目前结果支持训练信息保留量与成本需要权衡，尚未直接测试零截断。单层保存96个长度21025的复双精度全局向量需约30.8MiB，但不截断可能使训练轨迹较早breakdown，不能假设三层各保留96维并当作实测成本。论文宜写成“截断控制离线信息转化为在线预条件空间的成本”，不宜写成“全保留必然不可承受”。', '',
'复现：scripts/reproduce_ecc_two6.sh；汇总：python3 scripts/summarize_ecc_two6.py。模型/审计/计时/生产核验均存于 outputs/ecc_dd_budget_20260919/two6_*。']
(root/'md/ecc_two6_20260919.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
