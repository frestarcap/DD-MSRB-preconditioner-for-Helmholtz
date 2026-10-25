#!/usr/bin/env python3
"""Produce a short audit from real reports, including cases still incomplete."""
import argparse
import csv
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]/'outputs/scnet_3d_20260909')
    root=parser.parse_args().root
    plan=json.loads((root/'expected_cases.json').read_text())
    extra_file=root/'extra_runs.json'
    extras=json.loads(extra_file.read_text()) if extra_file.exists() else []
    accounting=root/'accounting.psv'
    jobs={}
    if accounting.exists():
        with accounting.open() as f:
            for row in csv.DictReader(f,delimiter='|'):
                jobs[row['JobID']]=row
    lines=['# 三维高频实验核验','',
           f"更新于 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}。",'',
           '每个正式算例申请 64 CPU / 192 GiB；8 个 worker，每个计算线程上限为 4。',
           '离线快照要求真实残差 ≤ 1e-7，在线求解要求 ≤ 1e-6。三维完成器为 shifted-MG。',
           '各频率分别训练三层 ROM，再验证该频率下的非训练参数查询。','',
           '| 算例 | ω | 未知量 | 作业状态 | 数值验收 |',
           '| --- | ---: | ---: | --- | --- |']
    reports=[]
    cases=[]
    for original in plan['cases']:
        case=dict(original,version=plan['version'],job_id=f"{plan['array_id']}_{original['array_index']}")
        cases.append(case)
    cases.extend(e for e in extras if e['kind']=='full')
    for case in cases:
        local=root/('remote_'+case['version'])
        path=local/case['tag']/'report.json'
        filename=f"helm3d_sweep_{plan['array_id']}_{case['array_index']}.out" if 'array_index' in case else f"helm3d_{case['job_id']}.out"
        log=local/'logs'/filename
        job=jobs.get(str(case['job_id']),{})
        status=job.get('State','尚未取得调度状态')
        # Slurm compresses waiting array elements into a single row.
        if not job and 'array_index' in case:
            for key,row in jobs.items():
                match=re.fullmatch(rf"{plan['array_id']}_\[(\d+)-(\d+)%\d+\]",key)
                if match and int(match[1])<=case['array_index']<=int(match[2]):
                    status=row['State']; break
        acceptance='尚无完整结果'
        if status.startswith('CANCELLED'):
            replacements=[e for e in extras if e['kind']=='full' and e['problem']==case['problem'] and e['omega']==case['omega']]
            if replacements:
                acceptance='未启动，由 '+str(replacements[-1]['job_id'])+' 替代'
        if path.exists():
            d=json.loads(path.read_text())
            d['report_path']=str(path.resolve())
            logtext=log.read_text() if log.exists() else ''
            cpu=re.search(r'^Model name:\s+(.+)$',logtext,re.M)
            mhz=re.search(r'^CPU max MHz:\s+([\d.]+)',logtext,re.M)
            d['cpu_model']=cpu[1] if cpu else '未取得'
            d['cpu_max_GHz']=f"{float(mhz[1])/1000:g}" if mhz else '未取得'
            csv_path=path.parent/f"helmholtz_3d_candidates_{case['tag']}.csv"
            d['offline_seconds']={}
            if csv_path.exists():
                with csv_path.open() as f:
                    for row in csv.DictReader(f):
                        if row['method'] in ('global','dd'):
                            d['offline_seconds'][row['method']]=float(row['offline_s'])
            if 'solver_config' not in d:
                cfg=json.loads((path.parent/'configuration.json').read_text())['cfg']
                d['solver_config']={key:cfg[key] for key in ('shift_beta','snapshot_restart','gmres_restart','mg_coarse_max_dofs')}
            reports.append(d)
            residuals=[q[m]['true_residual'] for q in d['tests'] for m in ('shifted','global','dd')]
            passed=d['all_converged'] and max(residuals)<=1e-6 and d['snapshot_max_relres']<=1e-7
            acceptance=f"通过，max ρ={max(residuals):.2e}" if passed else '未通过（数据保留）'
        else:
            if log.exists():
                logtext=log.read_text()
                checkpoints=re.findall(r'CHECKPOINT (GLOBAL|DD) layer (\d+) saved',logtext)
                failures=re.findall(r'Snapshot mu=([\d.]+) failed true residual ([\deE.+-]+)',logtext)
                if failures:
                    mu,rho=failures[-1]
                    acceptance=f'训练未通过，μ={mu}，ρ={rho}'
                elif checkpoints:
                    style,layer=checkpoints[-1]
                    acceptance=f'{style} 第 {layer}/3 层已保存，待完整验收'
                recorded=re.findall(r'^snapshot mu=[\d.]+ it=\d+ true=([\deE.+-]+)',logtext,re.M)
                if recorded:
                    count=sum(float(value)<=1e-7 for value in recorded)
                    acceptance+=f'；日志已记录 {count} 次快照通过'
        name='波导' if case['problem']=='waveguide' else '半空间'
        if case.get('label'): name+='（'+case['label']+'）'
        lines.append(f"| {name} | {case['omega']} | {case['N']:,} | {status} | {acceptance} |")
    if reports:
        lines+=['','## 已得到的指标','',
                '| 作业 | 算例 | ω | Global / DD 基 (MiB) | 基节约 | 总 ROM 存储节约 | DD 在线加速范围 |',
                '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
        for d in reports:
            speed=[q['shifted']['seconds']/q['dd']['seconds'] for q in d['tests']
                   if q['shifted']['true_residual']<=1e-6 and q['dd']['true_residual']<=1e-6]
            ratio=f'{min(speed):.2f}–{max(speed):.2f}×' if speed else '未取得共同收敛计时'
            persistent=100*(1-d['dd_persistent_MiB']/d['global_persistent_MiB'])
            name='半空间' if d.get('problem_name')=='affine_halfspace' else '波导'
            lines.append(f"| {d['metadata']['job_id']} | {name} | {d['omega']} | {d['global_basis_MiB']:.2f} / {d['dd_basis_MiB']:.2f} | {d['basis_saving_percent']:.1f}% | {persistent:.1f}% | {ratio} |")
        lines+=['','总 ROM 存储为基、降阶矩阵及一次共享分区索引的逻辑部署量；不等同于整个作业峰值 RSS。',
                '在线加速比较同一查询和真实残差容限，包含预条件器设置、ROM 步与完成求解。',
                '公共算子及移位矩阵装配位于各方法计时区间之外；MG 设置实测一次，并分别计入需要 MG 的方法。',
                '参数域内最小 PPW 固定约 9.93；源宽度随网格缩放。这组数据用于离散求解器的高频测试。']
        lines+=['','| 完整结果作业 | β | snapshot / online restart | 直接粗解阈值 |',
                '| --- | ---: | ---: | ---: |']
        for d in reports:
            cfg=d['solver_config']
            lines.append(f"| {d['metadata']['job_id']} | {cfg['shift_beta']} | {cfg['snapshot_restart']} / {cfg['gmres_restart']} | {cfg['mg_coarse_max_dofs']} |")
        lines+=['','| 完整结果作业 | 节点 | CPU 型号 | lscpu 最高频率 (GHz) |',
                '| --- | --- | --- | ---: |']
        for d in reports:
            lines.append(f"| {d['metadata']['job_id']} | {d['metadata']['node']} | {d['cpu_model']} | {d['cpu_max_GHz']} |")
        lines+=['','每个查询的加速比来自同一节点上的方法对照；跨频率的绝对耗时还需控制硬件。',
                '原 v4 波导24位于7285H /2.5 GHz节点；v7统一配置的波导24/36/48安排在',
                '同一g02r1n09节点（7285 /2.0 GHz），原结果仍保留。']
        lines+=['','## 训练与逐参数成本','',
                '每个案例为 8 个训练参数、3 个非训练查询 × 3 种方法，共 9 次在线求解。',
                '下表保留全部查询；每种方法每个查询计时一次，不是重复计时统计。','',
                '| 作业 | Global / DD 离线 (s) | 完整数值流程 (s) | DD 估计摊平查询数 | DD 与基线的全场相对差异（μ=.317） |',
                '| --- | ---: | ---: | ---: | ---: |']
        for d in reports:
            offline=d['offline_seconds']
            timing=f"{offline['global']:.2f} / {offline['dd']:.2f}" if offline else '未取得'
            resumed='（续算）' if d['training_resumed'] else ''
            saved=sum(q['shifted']['seconds']-q['dd']['seconds'] for q in d['tests'])/len(d['tests'])
            breakeven=str(math.ceil(offline['dd']/saved)) if offline and saved>0 and d['all_converged'] and not d['training_resumed'] else '不估计'
            lines.append(f"| {d['metadata']['job_id']}{resumed} | {timing} | {d['elapsed_seconds']:.2f} | {breakeven} | {d['field_slice']['relative_difference_to_shifted']:.3e} |")
        lines+=['','离线列按各模型独立训练记账，共享首层快照的求解费用仍计入 DD；',
                '完整流程耗时是实际共享快照、建立两种模型并完成在线对照的墙钟时间，另含数据输出。',
                '摊平查询数按 DD 独立离线费用除以 3 个查询的平均在线节省、再向上取整估计；',
                '它依赖这 3 个参数的代表性，不是已实测完成相应数量查询的总耗时。','',
                '| 作业 | μ | shifted / Global / DD 在线 (s) | 完成迭代（同序） | DD 加速 | 最大真实残差 |',
                '| --- | ---: | ---: | --- | ---: | ---: |']
        for d in reports:
            for q in d['tests']:
                methods=[q[m] for m in ('shifted','global','dd')]
                times=' / '.join(f"{r['seconds']:.3f}" for r in methods)
                iterations=' / '.join(str(r['iterations']) for r in methods)
                maximum=max(r['true_residual'] for r in methods)
                speed=f"{methods[0]['seconds']/methods[2]['seconds']:.2f}×" if all(r['true_residual']<=1e-6 for r in (methods[0],methods[2])) else '未共同收敛'
                lines.append(f"| {d['metadata']['job_id']} | {q['mu']:.3f} | {times} | {iterations} | {speed} | {maximum:.3e} |")
        lines+=['','Global / DD 的完成迭代之前各有 3 个 ROM 步；在线耗时已包含这 3 步。',
                '两个查询接近训练参数，不能将其大加速倍数外推为整个参数域的保证。']
    probes=[e for e in extras if e['kind']=='probe']
    if probes:
        lines+=['','## 困难参数诊断','',
                '| 作业 | 状态 | 同一 RHS 的参数扫描 |','| --- | --- | --- |']
        for e in probes:
            path=root/('remote_'+e['version'])/e['tag']/'probe.json'
            result='等待完整扫描'
            if path.exists():
                d=json.loads(path.read_text()); count=len(d['rows']) if isinstance(d['rows'],list) else 1
                result=f"{d['passing_count']} / {count} 个配置达到原 1e-7 容限"
            lines.append(f"| {e['job_id']} | {jobs.get(str(e['job_id']),{}).get('State','待取得状态')} | {result} |")
        for e in probes:
            path=root/('remote_'+e['version'])/e['tag']/'probe.json'
            if not path.exists(): continue
            d=json.loads(path.read_text()); rows=d['rows'] if isinstance(d['rows'],list) else [d['rows']]
            lines+=['',f"作业 {e['job_id']} 的全部配置（同一右端项，μ={d['baseline']['mu']:.6f}）：",'',
                    '| β | restart | 粗解阈值 | 迭代 | 真实残差 | 验收 |',
                    '| ---: | ---: | ---: | ---: | ---: | --- |']
            for r in rows:
                rho=f"{r['true_residual']:.3e}" if r['true_residual'] is not None else '未取得'
                lines.append(f"| {r['shift_beta']} | {r['restart']} | {r['mg_coarse_max_dofs']} | {r['iterations']} | {rho} | {'通过' if r['converged'] else '未通过'} |")
            figure=root/'figures/solver_probes'/e['tag']/'solver_probe.png'
            if figure.exists():
                lines+=['',f'![全部求解器配置的收敛与代价](<{figure.resolve()}>)',
                        '', '扫描耗时来自同一节点并发运行的不同配置；保留失败项，不视为独立重复计时。']
    lines+=['','## 作业资源记录','',
            '| 作业 | 状态 / 退出码 | 实际时长 | Slurm MaxRSS |',
            '| --- | --- | --- | --- |']
    for key,row in jobs.items():
        if '.batch' not in key:
            continue
        lines.append(f"| {key} | {row['State']} / {row['ExitCode']} | {row['Elapsed']} | {row.get('MaxRSS') or '运行中，待结算'} |")
    scaling_paths=sorted(root.glob('**/scaling.json'))
    if scaling_paths:
        scaling=json.loads(scaling_paths[-1].read_text()); rows=scaling['rows']
        base=next(r for r in rows if r['workers']==1 and r['threads']==1 and r['backend']=='cgs2_mex')
        best=min(rows,key=lambda r:r['wall_seconds'])
        matlab=next(r for r in rows if r['workers']==8 and r['threads']==1 and r['backend']=='cgs2')
        blas=next(r for r in rows if r['workers']==8 and r['threads']==1 and r['backend']=='cgs2_mex')
        lines+=['','## 并行校准','',
            f"单 worker / 单线程 BLAS 路径为 {base['wall_seconds']:.3f} s；{best['workers']}×{best['threads']} 为 {best['wall_seconds']:.3f} s，约 {base['wall_seconds']/best['wall_seconds']:.2f} 倍。",
            f"固定 8×1 时 MATLAB 切片路径为 {matlab['wall_seconds']:.3f} s、BLAS 路径为 {blas['wall_seconds']:.3f} s，约 {matlab['wall_seconds']/blas['wall_seconds']:.2f} 倍。",
            f"同一批 8 个 ω={scaling['omega']:g} 任务，每个布局计时一次，排除池启动；不外推成 64 倍扩展。"]
    lines+=['','## 数据与图形','',
            '初始配置/校准脚本异常及被回填时限中止的预备作业保留在各版本日志和运行记录中。']
    source_bundle=root/'delivery/helm3d_v7_source.tar.gz'
    if source_bundle.exists():
        lines+=['',f'[与超算 v7 哈希一致的部署源码包](<{source_bundle.resolve()}>)，包含所选求解器配置和提交说明。']
    verification=root/'verification.json'
    if verification.exists():
        lines+=['',f'[保存证据交叉核验记录及输入哈希](<{verification.resolve()}>)；以记录中的核验时间为准。']
    for d in reports:
        lines+=['',f"[作业 {d['metadata']['job_id']} 的完整指标 JSON](<{d['report_path']}>)。"]
    for folder in ('','halfspace'):
        figure=root/'figures'/folder/'frequency_summary.png'
        if figure.exists():
            label='半空间' if folder else '波导'
            lines+=['',f'![{label}的完整结果汇总](<{figure.resolve()}>)']
    for relative,label in [('uniform_profile/frequency_summary.png','同配置同节点的频率比较'),
                           ('uniform_profile/workflow_cost.png','同配置同节点的训练、在线、完整流程及内存成本'),
                           ('layer_storage/layer_storage.png','各层存储及不利层')]:
        figure=root/'figures'/relative
        if figure.exists(): lines+=['',f'![{label}](<{figure.resolve()}>)']
    for remote in dict.fromkeys([plan['remote']]+[e['remote'] for e in extras]):
        lines+=['',f"远端完整 MAT 数据：`{remote}/outputs/scnet_3d_20260909/`。"]
    lines+=['']
    out=root/'results_audit_zh.md'; out.write_text('\n'.join(lines))
    print(out)


if __name__=='__main__': main()
