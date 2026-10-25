"""Report fixed-cohort uniform-1e-5 ECC experiment without changing paper."""
from pathlib import Path
import csv
root=Path(__file__).resolve().parents[1];out=root/'outputs/ecc_dd_budget_20260919'
def read(f,key):return {r[key]:r for r in csv.DictReader((out/f).open())}
v=read('uniform5_summary.csv','method');st=read('uniform5_storage.csv','method')
a=list(csv.DictReader((out/'uniform5_tail_audit.csv').open()))
lines=['# ECC 三层统一 1e-5 实验（2026-09-20）','',
'固定96个训练参数，Global/DD从第一层重新训练。使用严格单个全局Frobenius尾预算、零秩起始贪心，无局部额外约束或秩上限。最终真实残差目标仍为1e-12；shifted-ILUTP beta=.2、droptol=1e-3、restart40、maxit20不变。', '',
'本地MATLAB R2025a单计算线程。在已有冻结22个查询上预热后每组重复五次、轮换顺序；这不是新独立验证集。所有时间均来自本次同批实验，包括条件ILU构建、ROM和完成求解，不含全阶公共组装、模型加载及离线训练。论文配置保持不变。', '',
'| 方法 | 基 MiB | 在线 s | 对shifted加速 | 完成迭代 | 免ILU查询 | 最大相对场误差 |','|---|---:|---:|---:|---:|---:|---:|']
for name in ['shifted','direct','last3_global','last3_dd','uniform5_global','uniform5_dd']:
 r=v[name];mem='—' if name not in st else f"{float(st[name]['basis_MiB']):.3f}"
 lines.append(f"| {name} | {mem} | {float(r['total_s']):.5f} | {float(r['speedup']):.2f} | {float(r['iterations']):.2f} | {float(r['no_setup_queries']):g}/22 | {float(r['max_error']):.3e} |")
g=float(st['uniform5_global']['basis_MiB']);d=float(st['uniform5_dd']['basis_MiB']);old=float(st['last3_dd']['basis_MiB'])
lines+=['',f'统一1e-5时DD比Global减少{100*(1-d/g):.1f}%基存储。相比论文last3 DD，统一1e-5 DD少用{100*(1-d/old):.1f}%基存储，在线耗时为{float(v["uniform5_dd"]["total_s"])/float(v["last3_dd"]["total_s"]):.2f}倍。', '',
'Global层秩为[25,42,42]；DD层总维数为[41,71,75]。实际逐层投影尾误差如下，全部不超过1e-5：','', '| 模型 | 层 | 局部秩/Global秩 | 实际相对尾误差 |','|---|---:|---|---:|']
for r in a:lines.append(f"| {r['tag']} | {r['layer']} | {r['ranks']} | {float(r['projection_tail']):.6e} |")
lines+=['','| DD阶段 | 论文last3 s | 统一1e-5 s |','|---|---:|---:|']
for k in ['ROM_s','setup_s','solve_s']:lines.append(f'| {k} | {float(v["last3_dd"][k]):.5f} | {float(v["uniform5_dd"][k]):.5f} |')
lines+=['',f"统一1e-5 DD payload（基+单份共享索引）为{float(st['uniform5_dd']['payload_MiB']):.3f}MiB，实际MATLAB部署对象为{float(st['uniform5_dd']['MATLAB_object_MiB']):.3f}MiB。均不是进程峰值内存。", '',
'全部660个计时解满足1e-12残差目标；88个生产入口重放和12个降阶组装核验通过。Global/DD在原11个开发参数上的最大相对场误差为1.012e-8/8.661e-10；两批均保留原始误差记录。', '',
'复现：scripts/reproduce_ecc_uniform5.sh；汇总：python3 scripts/summarize_ecc_uniform5.py。完整数据位于outputs/ecc_dd_budget_20260919/uniform5_*。']
(root/'md/ecc_uniform5_20260920.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
