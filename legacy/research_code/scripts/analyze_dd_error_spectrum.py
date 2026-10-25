"""Summarize saved full-volume MATLAB Fourier diagnostics without changing inputs."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/dd_error_spectrum_20260920'
src=OUT/'results/spectra.json'
s=json.loads(src.read_text());rows=s['records']
def get(w,m,f,window):
 a=[r for r in rows if (r['omega'],r['method'],r['field'],r['window'])==(w,m,f,window)]
 assert len(a)==1
 return a[0]
records=[]
for r in rows:
 assert abs(sum(r['radial_fraction'])-1)<1e-10
 assert abs(sum(r['theta_fraction'])-1)<1e-10
 assert r['parseval_relative_error']<1e-10
 records.append({k:r[k] for k in ['omega','method','field','window','reference_rho','physical_energy','windowed_energy','radial_ge1p5','grid_high_any','parseval_relative_error']}|{'axial_ge1p5':r['axis'][0]['ge1p5']})
with (OUT/'spectral_summary.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=records[0].keys());writer.writeheader();writer.writerows(records)
plt.rcParams.update({'font.family':'serif','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(2,2,figsize=(10,7),layout='constrained')
for col,w in enumerate([24,48]):
 for m,color in [('global','#2369a0'),('dd','#d05c24')]:
  for field,style,label in [('new_error','-','error'),('residual','--','residual')]:
   r=get(w,m,field,'hann_xyz');x=np.array(r['radial_lower_edges'])+.025;y=np.array(r['radial_fraction'])
   axs[0,col].semilogy(x,np.maximum(y,1e-18),style,color=color,label=f'{"Global" if m=="global" else "DD"}: {label}',lw=1.5)
   t=np.array(r['theta_lower_edges']);p=np.array(r['theta_fraction']);tail=np.cumsum(p[::-1])[::-1]
   axs[1,col].semilogy(t,np.maximum(tail,1e-18),style,color=color,lw=1.5)
 axs[0,col].axvline(1.5,color='gray',lw=.8,ls=':')
 axs[0,col].set(xlim=(0,4),ylim=(1e-10,1),title=rf'$\omega={w}$: physical-wavenumber spectrum',xlabel=r'$|k|/\omega$',ylabel='Fraction per radial bin (width 0.05)')
 axs[0,col].legend(frameon=False,fontsize=8)
 axs[1,col].axvline(.5,color='gray',lw=.8,ls=':')
 axs[1,col].set(xlim=(0,1),ylim=(1e-10,1.2),title=rf'$\omega={w}$: grid-scale spectral tail',xlabel=r'Cutoff $t$ for $\max_j |k_j h_j|/\pi$',ylabel=r'Energy fraction at or above $t$')
for ax in axs.flat:ax.grid(alpha=.18)
fig.savefig(OUT/'error_residual_spectra.png',dpi=200);fig.savefig(OUT/'error_residual_spectra.svg')
lines=['# DD / Global 完整三维误差频谱分析','',
'对象：冻结的三层模型，主查询 μ=0.317，ω=24（低频对照）与48（高频）。使用已有约1e-10原方程相对残差的严格计算参考解；误差为 e=u_ref−u_ROM。此计算参考不是经认证的精确解。新分析直接对完整三维物理域复数场作FFT，而非对中心切片或模长图作FFT。','',
'## 结论','',
'1. 两个已测主查询中，DD计算误差的高频尾部比Global更强；方向在三种窗函数下保持一致。但这不等于DD剩余误差以高频为主。',
'2. ω=48三维Hann窗下，DD物理高频误差占2.019%，网格高频误差仅0.1953%；相应残差占87.68%和67.92%。误差与残差的频谱不可互换。',
'3. DD网格高频误差占比在三种窗下为ω=24: 0.139–0.730%，ω=48: 0.195–0.398%；窗函数影响具体数值，但不改变DD高于Global的方向。',
'4. Global三维Hann窗的极小尾部对参考收紧敏感，不宜突出巨大的DD/Global倍数。ω=48 DD尾部对参考收紧稳定（参考变化高频范数约为新误差高频范数的0.17%），仍不是经认证的误差界。',
'5. 这次直接验证了更强的高频误差尾部，但没有证明其主导加速，也没有测量平滑误差传播算子的smoothing factor。正文仍建议 residual composition → more favorable for MG；附录可补充 stronger high-frequency tails in the computed error。结论范围限于ω=24,48的主查询，不外推到ω=36,60或所有参数。',
'',
'## 定义与验证','',
'- 同时分析误差 e 与交接残差 r；保留原始复相位。剔除PML后进行矩形窗、三维可分离Hann窗、仅轴向Hann窗三种分析。',
'- 物理波数高频：|k|/ω≥1.5；它是背景速度为1的描述性阈值，不是本问题的色散关系。',
'- 网格高频：至少一个方向 |k_j h_j|≥π/2，对应标准2倍粗化下粗网格不能无混叠表示的频带。非周期异质PML算子的Fourier分量不是其特征模态，这不是局部Fourier平滑因子分析。',
'- 所有比例以各自窗口后的物理域欧氏能量为分母。检验了Parseval恒等式与频带能量和；已知单Fourier模态用于验证低/高频分类。',
'', '## 三维Hann窗结果','',
'| ω | 方法 | 误差：物理高频 (%) | 残差：物理高频 (%) | 误差：网格高频 (%) | 残差：网格高频 (%) |',
'|---|---|---:|---:|---:|---:|']
for w in [24,48]:
 for m in ['global','dd']:
  e=get(w,m,'new_error','hann_xyz');r=get(w,m,'residual','hann_xyz')
  lines.append(f"| {w} | {m} | {100*e['radial_ge1p5']:.6g} | {100*r['radial_ge1p5']:.6g} | {100*e['grid_high_any']:.6g} | {100*r['grid_high_any']:.6g} |")
lines+=['','## 窗函数与参考解敏感性','','| ω | 方法 | 窗 | 新误差网格高频 (%) | 旧误差网格高频 (%) | 参考变化的高频范数 / 新误差高频范数 |','|---|---|---|---:|---:|---:|']
for w in [24,48]:
 for m in ['global','dd']:
  for window in ['rectangular','hann_xyz','hann_x']:
   e=get(w,m,'new_error',window);old=get(w,m,'old_error',window);d=get(w,m,'reference_change',window)
   ratio=np.sqrt((d['windowed_energy']*d['grid_high_any'])/(e['windowed_energy']*e['grid_high_any']))
   lines.append(f"| {w} | {m} | {window} | {100*e['grid_high_any']:.6g} | {100*old['grid_high_any']:.6g} | {ratio:.6g} |")
lines+=['','参考变化是从原约1e-6参考收紧到约1e-10参考带来的差异，用来检查敏感性；它不是新参考误差的严格上界。','',
'![三维Hann窗误差与残差频谱](error_residual_spectra.png)','',
'## 可复现性','',f"- MATLAB {s['matlab']}，Slurm作业 {s['job']}；2 CPUs / 6 GiB，仅后处理，无训练或求解器修改。",'- MATLAB入口：`code/run_dd_error_spectrum_saved.m`；汇总绘图：`scripts/analyze_dd_error_spectrum.py`。',
'- `results/spectra.json` 和 `results/spectra.mat` 保存全部频谱；`spectral_summary.csv` 保存48行指标。',f'- spectra.json SHA256: `{hashlib.sha256(src.read_bytes()).hexdigest()}`。']
(OUT/'report.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[12:23]))
