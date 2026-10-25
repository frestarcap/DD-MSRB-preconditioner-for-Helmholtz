#!/usr/bin/env python3
"""Read additive interface job outputs; write a new CSV/report/figure package.

Multiple --input-root arguments combine disjoint frequency cohorts. Duplicate
frequencies are rejected, even if file content matches. --results-root points
to the diagnostic results directory to verify actual fixed-work iteration
counts and residual-vector norms against normalized.json.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path


def array(x):
    if not isinstance(x, list):
        return [x]
    return [z for item in x for z in array(item)]


def numeric(x):
    return None if x is None else float(x)


def close(a, b, message, rtol=2e-8, atol=1e-12):
    assert abs(a-b) <= atol+rtol*max(abs(a), abs(b)), message


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def pct(x):
    return '—' if x is None else f'{100*x:.3f}%'


def num(x):
    return '—' if x is None else f'{x:.3g}'


def load_cohorts(roots, results_root):
    cohorts={}; sources={}
    for root in roots:
        paths=[root] if root.is_file() else sorted(root.rglob('interfaces.json'))
        assert paths, f'No interfaces.json found in {root}'
        for path in paths:
            report=json.loads(path.read_text())
            assert report['validated'] and report['definition_version']=='dd_residual_interfaces_v1'
            omega=int(report['omega'])
            assert omega not in cohorts, f'Duplicate frequency {omega}: explicitly select one cohort.'
            records={row['variant']:row for row in report['records']}
            assert len(records)==len(report['records'])==6
            assert set(records)=={'GG','GGG','DD','DDD','GGD','DDG'}
            assert all(abs(row['mu']-.317)<1e-10 for row in records.values())
            assert report['geometry']['mask_names'][1]=='physical_interblock'
            normalized={}
            if results_root:
                np=results_root/Path(report['vectors_dir']).name/'normalized.json'
                assert np.is_file(), f'Missing fixed-work evidence {np}'
                entries=json.loads(np.read_text()); sources[str(np)]=hashlib.sha256(np.read_bytes()).hexdigest()
                normalized={r['variant']:r for r in entries if abs(r['mu']-.317)<1e-10}
                assert set(records).issubset(normalized)
                for label,row in records.items():
                    nr=normalized[label]
                    assert nr['omega']==omega
                    assert nr['iteration_count']<=nr['fixed_step_cap']
                    close(row['rafter']['energy_total'],nr['true_relative_residual']**2,
                          f'Fixed-work vector norm mismatch {omega}/{label}',atol=1e-28)
            capture_path=path.parent/'refined_space_capture.json'
            capture=None
            if capture_path.exists():
                capture=json.loads(capture_path.read_text())
                assert capture['validated'] and capture['definition_version']=='dd_refined_space_capture_v1'
                assert capture['omega']==omega and abs(capture['mu']-.317)<1e-10
                assert capture['training_file']==report['training_file']
                assert capture['vectors_dir']==report['vectors_dir']
                assert {t['variant'] for t in capture['targets']}=={'GG','DD','GGG','DDD'}
                assert len(capture['targets'])==4
                sources[str(capture_path)]=hashlib.sha256(capture_path.read_bytes()).hexdigest()
            sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            cohorts[omega]=(report,records,normalized,capture,path)
    return dict(sorted(cohorts.items())),sources


def extract(cohorts):
    all_rows=[]; edges=[]; changes=[]; captures=[]
    for omega,(report,records,normalized,capture,path) in cohorts.items():
        geom=report['geometry']
        for variant,row in records.items():
            for field in ('r0','e0','rafter'):
                s=row[field]
                for k,mask in enumerate(geom['mask_names']):
                    all_rows.append(dict(omega=omega,mu=row['mu'],variant=variant,field=field,mask=mask,
                        energy_total=s['energy_total'],energy_physical=s['energy_physical'],
                        interface_energy_fraction=numeric(s['energy_fraction'][k]),
                        interface_volume_fraction=geom['volume_fraction'][k],
                        interface_enrichment=numeric(s['enrichment'][k]),
                        physical_interface_energy_fraction=numeric(s['physical_energy_fraction'][k]),
                        physical_interface_volume_fraction=geom['physical_volume_fraction'][k],
                        physical_interface_enrichment=numeric(s['physical_enrichment'][k]),
                        actual_completion_steps=(normalized[variant]['iteration_count'] if field=='rafter' and normalized else None),
                        old_computed_reference=(field=='e0'),source_json=str(path)))
                ed=array(s['edge_mean_squared_difference']); es=array(s['edge_mean_squared_difference_over_physical_mean_power'])
                eg=array(s['edge_mean_squared_gradient_over_physical_mean_power'])
                edges.append(dict(omega=omega,mu=row['mu'],variant=variant,field=field,
                    physical_crossblock_edge_count=sum(geom['physical_edge_count_by_axis'][0]),
                    physical_withinblock_edge_count=sum(geom['physical_edge_count_by_axis'][1]),
                    crossblock_mean_squared_difference=numeric(ed[0]),withinblock_mean_squared_difference=numeric(ed[1]),
                    crossblock_difference_over_mean_field_power=numeric(es[0]),withinblock_difference_over_mean_field_power=numeric(es[1]),
                    crossblock_gradient_over_mean_field_power=numeric(eg[0]),withinblock_gradient_over_mean_field_power=numeric(eg[1]),
                    crossblock_to_withinblock_ratio=numeric(s['interblock_to_withinblock_difference_ratio'])))
            if variant in ('GGG','DDD'):
                before=row['r0']['physical_energy_fraction'][1];after=row['rafter']['physical_energy_fraction'][1]
                initial_interface_unit=row['r0']['mask_energy'][1]/row['r0']['energy_total']
                after_interface_unit=row['rafter']['mask_energy'][1]
                retained=(after_interface_unit/initial_interface_unit if initial_interface_unit>0 else None)
                changes.append(dict(omega=omega,mu=row['mu'],variant=variant,
                    physical_interface_volume_fraction=geom['physical_volume_fraction'][1],
                    initial_physical_interface_fraction=before,after_physical_interface_fraction=after,
                    fraction_change=after-before,
                    initial_physical_interface_enrichment=row['r0']['physical_enrichment'][1],
                    after_physical_interface_enrichment=row['rafter']['physical_enrichment'][1],
                    initial_interface_energy_on_unit_rhs_scale=initial_interface_unit,
                    after_interface_energy_on_unit_rhs_scale=after_interface_unit,
                    interface_energy_retained_ratio=retained,
                    interface_energy_net_reduction_fraction=(1-retained if retained is not None else None),
                    fixed_work_true_relative_residual=math.sqrt(row['rafter']['energy_total']),
                    actual_completion_steps=normalized[variant]['iteration_count'] if normalized else None,
                    iteration_count_independently_verified=bool(normalized)))
        if capture:
            for t in capture['targets']:
                for space in ('global','dd'):
                    s=t[space]
                    close(s['captured_energy_fraction']+s['relative_best_approximation_error']**2,1,
                          f'Projection energy identity {omega}/{t["variant"]}/{space}')
                    assert s['orthogonal_energy_closure_error']<1e-9
                    captures.append(dict(omega=omega,mu=capture['mu'],common_target=t['variant'],
                        prefix_depth=t['prefix_depth'],candidate_third_space=space,
                        refined_error_norm=t['refined_error_norm'],reference_true_relative_residual=capture['reference_true_residual'],
                        reference_change_over_refined_error=t['reference_change_over_refined_error'],
                        captured_energy_fraction=s['captured_energy_fraction'],
                        relative_best_approximation_error=s['relative_best_approximation_error'],
                        orthogonal_energy_closure_error=s['orthogonal_energy_closure_error'],
                        maximum_basis_gram_identity_deviation=max(array(capture['spaces'][space]['basis_gram_identity_deviation'])),
                        source_json=str(path.parent/'refined_space_capture.json')))
    changes.sort(key=lambda r:(r['omega'],0 if r['variant']=='GGG' else 1))
    return all_rows,edges,changes,captures


def make_report(out,cohorts,rows,edges,changes,captures):
    native=[r for r in rows if r['variant'] in ('GGG','DDD') and r['mask']=='physical_interblock']
    lookup={(r['omega'],r['variant'],r['field']):r for r in native}
    lines=['# DD 物理区内部界面与严格参考误差诊断','',
        '这里把界面定义为：相邻节点属于不同 DD 块，且两个节点都位于物理区域；界面包含该边的两个端点。Global 与 DD 使用完全相同的界面掩膜。外部边界和物理/PML 过渡不混入此表；另外两类掩膜保留在完整 CSV。','',
        '“能量”仅指二范数平方。下表能量比例以整个物理区域为分母；体积比例以物理节点数为分母。富集度等于两者之比，均匀平方振幅对应 1。由于界面厚度固定为一层网格，跨频率比较必须同时看体积比例。','',
        '| ω | 字段 | 界面体积 | Global 界面能量 | DD 界面能量 | Global 富集度 | DD 富集度 |',
        '|---:|:---|---:|---:|---:|---:|---:|']
    labels={'r0':'ROM 后残差','e0':'ROM 解误差（原参考）','rafter':'固定步数后残差'}
    for omega in cohorts:
        for field in ('r0','e0','rafter'):
            g=lookup[omega,'GGG',field];d=lookup[omega,'DDD',field]
            lines.append(f"| {omega} | {labels[field]} | {pct(g['physical_interface_volume_fraction'])} | {pct(g['physical_interface_energy_fraction'])} | {pct(d['physical_interface_energy_fraction'])} | {num(g['physical_interface_enrichment'])} | {num(d['physical_interface_enrichment'])} |")
    lines+=['','## 固定工作量之后还剩什么','',
        '`rafter` 是单位范数初始残差经过 completion 后的剩余向量；其原始范数与 r0 不在同一尺度上。界面比例、富集度和按物理平均平方振幅归一化的相邻差仍可比较。仅看比例变化不足以量化界面残差衰减，下面再用完整向量范数将初始和最终界面能量统一到同一个尺度。','',
        '| ω | 方法 | 核验步数 | ROM 后界面占比 | completion 后界面占比 | 最终归一化残差 |',
        '|---:|:---|---:|---:|---:|---:|']
    for r in changes:
        lines.append(f"| {r['omega']} | {'Global' if r['variant']=='GGG' else 'DD'} | {r['actual_completion_steps'] if r['actual_completion_steps'] is not None else '未独立核验'} | {pct(r['initial_physical_interface_fraction'])} | {pct(r['after_physical_interface_fraction'])} | {r['fixed_work_true_relative_residual']:.4g} |")
    lines+=['','设 Γ 为相同物理内部界面掩膜，初始方向为 r0/‖r0‖，则界面能量保留比是 ‖rafter[Γ]‖² / (‖r0[Γ]‖²/‖r0‖²)。等价地，它等于完整归一化最终残差的平方乘以最终/初始**全域分母**界面能量比例之比；不能直接把物理条件比例代入而漏掉物理能量变化。','',
        '| ω | 方法 | 单位初值尺度的初始界面能量 | 最终界面能量 | 界面能量保留比 | 界面能量净下降 |',
        '|---:|:---|---:|---:|---:|---:|---:|']
    for r in changes:
        lines.append(f"| {r['omega']} | {'Global' if r['variant']=='GGG' else 'DD'} | {r['initial_interface_energy_on_unit_rhs_scale']:.5g} | {r['after_interface_energy_on_unit_rhs_scale']:.5g} | {num(r['interface_energy_retained_ratio'])} | {pct(r['interface_energy_net_reduction_fraction'])} |")
    lines+=['','这是固定工作量 GMRES 对界面限制向量产生的**实测净变化**；并不等于界面对收敛贡献了一个独立百分比。迭代可能在界面内外重新分配残差，不能把各区域当作独立演化的模态。']
    lines+=['','下面比较每条物理跨块边的平均平方差与每条物理块内边的平均平方差之比；两类边均按自己的边数归一化，避免边数差造成假象。','',
        '| ω | 字段 | Global 跨块/块内比 | DD 跨块/块内比 |',
        '|---:|:---|---:|---:|']
    edge_lookup={(r['omega'],r['variant'],r['field']):r for r in edges}
    for omega in cohorts:
        for field in ('r0','e0','rafter'):
            g=edge_lookup[omega,'GGG',field];d=edge_lookup[omega,'DDD',field]
            lines.append(f"| {omega} | {labels[field]} | {num(g['crossblock_to_withinblock_ratio'])} | {num(d['crossblock_to_withinblock_ratio'])} |")
    lines+=['','这些差异反映离散场的局部变化，不自动代表不连续性或某种连续谱模态。按物理平均平方振幅归一化的跨块/块内均值、间距缩放梯度以及各边数也保留在 `physical_edge_metrics.csv`。','',
        '## 同一个严格参考误差目标，两种第三层空间','',
        '下面对 GG 或 DD 两层前缀后的**同一个**误差向量，分别求其在 Global 第三层和 DD 第三层中的最佳欧氏正交近似。误差由保存的 e0 加上新旧参考解之差得到；没有重新求解或重新训练。该量衡量空间能表示什么，不等于实际 Galerkin 修正、FGMRES 步长或 completion 收益。','',
        '| ω | 共同目标 | 参考解真实残差 | Global 第三层捕获 | DD 第三层捕获 |',
        '|---:|:---|---:|---:|---:|']
    cap={(r['omega'],r['common_target'],r['candidate_third_space']):r for r in captures}
    for omega in cohorts:
        for target in ('GG','DD'):
            if (omega,target,'global') in cap:
                g=cap[omega,target,'global'];d=cap[omega,target,'dd']
                lines.append(f"| {omega} | {target} 前缀后误差 | {g['reference_true_relative_residual']:.3g} | {pct(g['captured_energy_fraction'])} | {pct(d['captured_energy_fraction'])} |")
    if not captures:lines+=['','所选后处理目录尚无严格参考空间捕获结果；本报告不补造数值。']
    lines+=['','GGG/DDD 最终原生误差的同类捕获结果、Gram 正交性检查和参考解变化相对误差大小也保留在 `refined_space_capture.csv`。严格参考仍是计算参考，并非已认证的解误差界。','',
        '## 可以据此解释到哪一步','']
    for omega in cohorts:
        g=lookup[omega,'GGG','r0'];d=lookup[omega,'DDD','r0']
        afterg=lookup[omega,'GGG','rafter'];afterd=lookup[omega,'DDD','rafter']
        lines.append(f"- ω={omega}：DD 初始物理内部界面残差占比 {pct(d['physical_interface_energy_fraction'])}，Global 为 {pct(g['physical_interface_energy_fraction'])}；固定工作量后分别为 {pct(afterd['physical_interface_energy_fraction'])} 和 {pct(afterg['physical_interface_energy_fraction'])}。这些是实测位置分量，需结合完整归一化残差和 MG 作用判断其可消除性。")
    lines+=['','残差 r=Ae 与解误差 e 的空间分布不能互相替代。界面与介质/波导结构相关，因而空间富集并不是独立随机对照；即使观察到 DD 界面残差更集中，也只支持与局部空间边界有关的解释，不能仅凭这项诊断确定因果或推广到全部高频参数。','',
        '**参考质量区分：** 界面表中的 e0 使用原始 shifted 求解参考，可能对参考误差敏感；严格参考仅用于单独的空间捕获表以及相应切片。当前脚本没有把原参考的界面误差冒充严格参考结果。','',
        '数据来源、输入 SHA256、频率去重与固定步数核验见 `verification.json`。']
    (out/'interface_analysis_zh.md').write_text('\n'.join(lines)+'\n')


def plot(out,rows):
    os.environ.setdefault('MPLCONFIGDIR',str(out/'mplconfig'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected=[r for r in rows if r['variant'] in ('GGG','DDD') and r['mask']=='physical_interblock']
    ws=sorted({r['omega'] for r in selected}); idx={(r['omega'],r['variant'],r['field']):r for r in selected}
    fig,axes=plt.subplots(2,3,figsize=(12.5,6.8),layout='constrained',sharex=True)
    titles=['ROM residual','Field error (original reference)','Residual after fixed work']
    for j,field in enumerate(('r0','e0','rafter')):
        for variant,name,color,marker in [('GGG','Global','#2474a8','o'),('DDD','DD','#dc6d2d','s')]:
            axes[0,j].plot(ws,[100*idx[w,variant,field]['physical_interface_energy_fraction'] for w in ws],marker=marker,color=color,label=name)
            axes[1,j].plot(ws,[idx[w,variant,field]['physical_interface_enrichment'] for w in ws],marker=marker,color=color,label=name)
        axes[0,j].plot(ws,[100*idx[w,'GGG',field]['physical_interface_volume_fraction'] for w in ws],color='.4',ls='--',label='Interface volume')
        axes[1,j].axhline(1,color='.4',ls='--',lw=1)
        axes[0,j].set_title(titles[j]);axes[1,j].set_xlabel(r'$\omega$')
        axes[0,j].set_ylim(bottom=0);axes[1,j].set_ylim(bottom=0)
        for a in axes[:,j]:a.set_xticks(ws);a.grid(alpha=.18)
    axes[0,0].set_ylabel('Physical energy at interfaces (%)');axes[1,0].set_ylabel('Interface enrichment')
    axes[0,0].legend(fontsize=9)
    fig.suptitle('Physical internal DD interfaces: identical masks for both methods',fontsize=12)
    fig.savefig(out/'physical_internal_interfaces.png',dpi=170);fig.savefig(out/'physical_internal_interfaces.pdf');plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-root',action='append',type=Path,required=True)
    p.add_argument('--results-root',type=Path)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--no-plot',action='store_true')
    args=p.parse_args(); assert not args.output_dir.exists(),'Refusing to overwrite analysis outputs.'
    cohorts,sources=load_cohorts(args.input_root,args.results_root)
    rows,edges,changes,captures=extract(cohorts)
    args.output_dir.mkdir(parents=True)
    write_csv(args.output_dir/'interface_all_masks.csv',rows)
    write_csv(args.output_dir/'native_physical_interfaces.csv',[r for r in rows if r['variant'] in ('GGG','DDD') and r['mask']=='physical_interblock'])
    write_csv(args.output_dir/'physical_edge_metrics.csv',edges)
    write_csv(args.output_dir/'normalized_direction_changes.csv',changes)
    write_csv(args.output_dir/'refined_space_capture.csv',captures)
    make_report(args.output_dir,cohorts,rows,edges,changes,captures)
    if not args.no_plot:plot(args.output_dir,rows)
    verification=dict(validated=True,frequencies=list(cohorts),duplicate_frequency_policy='reject',
        strict_reference_frequencies=[w for w,(_,_,_,c,_) in cohorts.items() if c],
        completion_steps_independently_verified=bool(args.results_root),
        input_sha256=sources,source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        row_counts=dict(all_masks=len(rows),edge_metrics=len(edges),normalized_changes=len(changes),strict_capture=len(captures)))
    (args.output_dir/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    print(json.dumps(verification,ensure_ascii=False))


if __name__=='__main__':main()
