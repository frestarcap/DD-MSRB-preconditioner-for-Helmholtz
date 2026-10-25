"""Write the human-readable report from independently checked pilot results."""
from pathlib import Path
import csv,json,statistics
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/dd_parameter_mechanism_20260915'

def readcsv(name):return list(csv.DictReader((OUT/name).open()))
def avg(xs):return statistics.mean(xs)
def num(x):return float(x)
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])+'\n'
def link(name):return f'[{name}](<{OUT/name}>)'

v=json.loads((OUT/'validation.json').read_text());assert v['passed']
s=readcsv('model_summary.csv');d=readcsv('directions.csv');iv=readcsv('interventions.csv');comm=readcsv('commutators.csv')
span=readcsv('span_closure.csv');toy=json.loads((OUT/'span_identity_toy.json').read_text());assert toy['passed']
fields=readcsv('field_response.csv');assert len(fields)==48
training=readcsv('training_summary.csv')
assert len(span)==69
labels={1:'波导 ω12',2:'波导 ω6',3:'波导 ω9',4:'加长波导 ω12',5:'Gaussian lens ω12',6:'细网格波导 ω12'}
sections=[]
sections.append('# 参数扩展、固定 DD 与三层 TA，以及 DD/MG 互补：小规模三维结果\n')
sections.append('本轮只增加独立实验、结果和数学说明，未修改论文及原有求解器。首先固定协议，再进行小网格验证；参数主试验完成后，公开补充更细的单层截断和远离训练节点的查询。所有不利结果均保留。\n')
pairrows=[];wins={'fixed':0,'TA3':0};total={'fixed':0,'TA3':0}
for cid in range(1,7):
    values=[]
    for fam in ['fixed_t0e+00','TA3_t3e-03']:
        dd=[r for r in d if int(r['case_id'])==cid and r['method']=='dd_'+fam]
        gg=[r for r in d if int(r['case_id'])==cid and r['method']=='global_'+fam]
        assert len(dd)==len(gg)==2
        ratio=avg(num(r['normalized20']) for r in dd)/avg(num(r['normalized20']) for r in gg)
        values.append(f'{ratio:.3f}')
        family=fam.split('_')[0]
        for x in dd:
            y=next(y for y in gg if num(y['mu'])==num(x['mu']))
            wins[family]+=num(x['normalized20'])<num(y['normalized20']);total[family]+=1
    pairrows.append([labels[cid]]+values)
sections.append('## 1. 主要结论\n')
sections.append('1. **参数扩到 [0,1.4] 后，没有观察到三层 DD 在已测存储—完成工作曲线上形成明确的成本反转。** 训练数随范围从5增到9、15时，适当保留更多方向的固定 DD 仍很有竞争力；范围扩大但训练数维持5时，两者都明显退化。部分更紧的TA配置能减少少量迭代，但需要显著更多存储，应逐点比较，不能写成TA没有作用。\n')
sections.append(f'2. **DD 的归一化残差方向优势得到跨案例复现。** 固定 POD 中 {wins["fixed"]}/{total["fixed"]} 个成对查询、TA中 {wins["TA3"]}/{total["TA3"]} 个成对查询，DD的20步剩余残差更小。这与总完成迭代的胜负不同：很小的Global交接残差仍可更早达到目标。\n')
sections.append('   **但场误差的相对消除速度没有同样的统一优势。** 独立复算表明，DD的残差快速下降往往并不意味着其场误差也按更快比例下降。“DD误差更容易消除”需要限定为交接残差；绝对场误差大小另行比较。\n')
sections.append('3. **更重要的数学发现是当前纯逆作用实现的完整训练空间封闭性。** 固定分区、精确训练逆作用且ROM层之间没有全阶动作时，后续所有TA空间均包含在首批训练解的完整局部空间中。TA可以重新压缩、组织和使用这些方向，但不能在该完整空间之外增加方向。这一结论有明确条件，不适用于直接照搬Tang–Peng含恒等作用的SA结构。完整证明见 '+link('theory.md')+'。\n')
sections.append('## 2. 规模、资源和公平性\n')
case_rows=[]
for c in v['cases']:
    if c['case_id']>6:continue
    case_rows.append([c['case_id'],labels[c['case_id']],f"{c['N']:,}",c['training_solves'],c['online_solves'],f"{c['elapsed_s']/60:.2f}"])
sections.append(table(['编号','三维模型','未知量','训练参考解','在线求解次数','案例墙钟/分钟'],case_rows))
sections.append(f'合计 {v["formal_solves"]} 次正式在线求解，另有8次小网格试跑；单位残差探针、MG干预和严格参考解另计。所有正式解均核对原方程相对残差≤1e-6，训练参考解≤2e-11，后续逆作用快照缺陷≤1e-7。运行环境为本地MATLAB R2025a；主套件4线程、补充试验4线程，最多两进程共8线程，无并行池。两套件后期有重叠，因此秒数只作诊断，不声称统计意义上的加速比。\n')
sections.append('波导三频率使用同一网格、同一物理源宽度；细网格保持源宽度和PML物理厚度。最高扩展参数的最短波长约有8.1个网格点，本轮讨论离散系统性能。长波导保持网格间距并将长度从3改为6；lens改变介质形状。较小的coarse cap=1000与原大型生产配置不同，不能直接比较秒数或将小规模结果外推为ω48/60验证。\n')
sections.append('部署payload按复双精度基、两个稠密仿射约化矩阵、共享一次DD索引计算；不是进程RSS。保存的诊断MAT还包含训练快照，文件体积不等于部署存储。共同参考解成本与新POD/重放成本分开记录；近breakdown导致的额外逆作用求解全部计入训练检查，未隐藏为免费操作。\n')
sections.append('## 3. 参数范围与单层/三层成本\n')
sections.append(table(['组别','参数区间','训练数','原始查询'],[
 ['narrow5','[0.2,0.6]',5,'0.314, 0.494'],['medium9','[0,0.8]',9,'0.228, 0.588'],
 ['wide15','[0,1.4]',15,'0.399, 1.029'],['wide5','[0,1.4]',5,'0.399, 1.029']]))
sections.append('前三组采样间距都为0.1；第四组保持5个样本。四组的训练并集只有17个参数。固定DD初始扫描尾容差0.03、0.003、0.0003及完整数值秩，再补1e-5、1e-7；TA分别以0.03、0.003、0.0003为首层容差，后续比例固定为8/3、20/3。每个TA配置沿自己的轨迹重训。\n')
rr=[]
for cohort in ['narrow5','medium9','wide15','wide5']:
    a=next(r for r in s if int(r['case_id'])==1 and r['cohort']==cohort and r['method']=='dd_fixed_t0e+00')
    b=next(r for r in s if int(r['case_id'])==1 and r['cohort']==cohort and r['method']=='dd_TA3_t3e-03')
    z=next(r for r in s if int(r['case_id'])==1 and r['cohort']==cohort and r['method']=='dd_TA3_t3e-04')
    rr.append([cohort,f"{num(a['payload_MiB']):.2f} / {num(a['mean_iterations']):.1f}",f"{num(b['payload_MiB']):.2f} / {num(b['mean_iterations']):.1f}",f"{num(z['payload_MiB']):.2f} / {num(z['mean_iterations']):.1f}"])
sections.append(table(['组别','完整固定 DD：MiB / 平均迭代','TA首层0.003：MiB / 平均迭代','TA首层0.0003：MiB / 平均迭代'],rr))
sections.append(f'![所有已测截断档位，连线仅引导阅读，不代表已测连续前沿](<{OUT/"cost_tradeoff.png"}>)\n')
sections.append('宽范围的完整固定空间确实更大，但三层空间也增长。不能只比较完整固定空间与较松三层配置的存储：它们的完成工作并不相同。完整扫描及预先指定0/10/20/40/80步门槛下的已测最低存储见 '+link('sampled_cost_envelope.csv')+'；未测试的精确等预算最优配置没有被声称已经找到。\n')
sections.append('### 远离训练节点的补充查询\n')
sections.append('原始wide查询0.399恰好非常接近训练点0.4。为检查节点邻近性影响，补充[.25,.55]、[.25,.65]、[.45,1.15]、[.45,1.15]；前三组为0.1网格单元中点，稀疏wide5使用相同宽范围查询。所有旧模型冻结；没有把测试参考解用于训练。下表单独报告，不与原始查询混合。\n')
rr=[]
for cohort in ['narrow5','medium9','wide15','wide5']:
    values=[]
    for method in ['dd_fixed_t1e-07','dd_fixed_t0e+00','dd_TA3_t3e-04']:
        a=next(r for r in s if int(r['case_id'])==102 and r['cohort']==cohort and r['method']==method)
        values.append(f"{num(a['payload_MiB']):.2f} / {num(a['max_iterations']):.0f}")
    rr.append([cohort]+values)
sections.append(table(['组别','固定1e-7：MiB / 最差迭代','完整固定：MiB / 最差迭代','紧TA：MiB / 最差迭代'],rr))
sections.append('这组补充区分了两件事：训练点附近表现过好会夸大绝对精度；但适当截断的固定空间是否仍比三层更划算，必须在新的查询上重新检查。完整原始数据见 '+link('model_summary.csv')+'。\n')
sections.append('### 训练成本与信息复用\n')
tr=[]
for cohort in ['narrow5','medium9','wide15','wide5']:
    items=[next(r for r in training if int(r['case_id'])==1 and r['cohort']==cohort and r['method']==m) for m in ['dd_fixed_t0e+00','dd_TA3_t3e-03','dd_TA3_t3e-04']]
    tr.append([cohort,f"{num(items[0]['common_reference_source_s']):.1f}"]+[f"{num(r['model_build_source_s']):.2f} / {r['fallback_solves']}" for r in items])
sections.append(table(['组别','共享训练参考解/秒','固定：额外秒 / 补解数','中央TA：额外秒 / 补解数','紧TA：额外秒 / 补解数'],tr))
sections.append('后续快照优先使用经原方程验证的Arnoldi逆作用递推，故中央TA的额外训练可以很便宜。较紧TA在接近breakdown时触发补解，额外训练成本明显增加。上述秒数为本次来源计时，不能作为稳定的性能比；补解数和使用的信息量更可核查。扩参数主试验只做17个初始参考解，整套六案例共42个初始训练参考解及63次补充逆作用求解。中点查询只增加测试参考解，不增加训练信息。\n')
sections.append('## 4. DD 的方向是否更适合 shifted-MG\n')
sections.append('将各方法自己的交接残差统一成单位范数，保持同一原算子A、同一MG和20步上限。下表是两查询平均剩余范数的DD/Global比，低于1表示DD方向更容易处理。\n')
sections.append(table(['案例','固定 POD','TA3'],pairrows))
sections.append(f'![DD与Global的同幅度残差方向比较](<{OUT/"direction_ratios.png"}>)\n')
sections.append('一次MG作用被分成前平滑、粗网格和后平滑，数值和与原作用一致；完整修正的最优标量方向收缩与GMRES第一步一致。固定POD与TA均展现DD优势，说明该现象不专属于轨迹训练。\n')
rr=[]
for beta,sweeps in [(2,2),(.5,2),(1,2),(2,0),(2,4)]:
    ratios=[];dds=[];ggs=[]
    for cid in range(1,7):
        a=next(r for r in iv if int(r['case_id'])==cid and r['method']=='dd_TA3_t3e-03' and num(r['beta'])==beta and num(r['sweeps'])==sweeps)
        b=next(r for r in iv if int(r['case_id'])==cid and r['method']=='global_TA3_t3e-03' and num(r['beta'])==beta and num(r['sweeps'])==sweeps)
        ratios.append(num(a['rho20'])/num(b['rho20']));dds.append(num(a['rho20']));ggs.append(num(b['rho20']))
    rr.append([beta,f'{sweeps}+{sweeps}',f'{avg(dds):.3f}',f'{avg(ggs):.3f}',f'{avg(ratios):.3f}'])
sections.append(table(['shift β','前/后平滑','DD-TA平均20步残差','Global-TA平均20步残差','成对比值平均'],rr))
sections.append('去掉平滑会让该V-cycle成为秩受限的粗空间作用，通常两者都接近停滞。它支持平滑参与互补，但不是同成本候选算法，更不能单独证明平滑解释全部加速。更多平滑也不保证同样20步更好；β变化同时影响移位近似和MG质量。\n')
sections.append('### 残差快降不等于场误差按相同比例快降\n')
fr=[]
for cid in range(1,7):
    vv=[]
    for fam in ['fixed_t0e+00','TA3_t3e-03']:
        dd=[r for r in fields if int(r['case_id'])==cid and r['method']=='dd_'+fam]
        gg=[r for r in fields if int(r['case_id'])==cid and r['method']=='global_'+fam]
        vv.append(f"{avg(num(r['field_contraction20']) for r in dd)/avg(num(r['field_contraction20']) for r in gg):.3f}")
    fr.append([labels[cid]]+vv)
sections.append('下表把每种方法的场误差先按自身起始范数归一化，再比较20步剩余比例的DD/Global比；大于1表示DD场误差按比例下降得更慢。参考场来自严格原方程求解；这些是计算参考误差估计，不是认证误差界。\n')
sections.append(f"所有48个探针中，参考解残差/ROM交接残差的最大比为{max(num(r['reference_residual_over_handoff']) for r in fields):.3e}。该检查避免把明显的参考残差当成ROM残差，但不代替场误差认证。\n")
sections.append(table(['案例','固定 POD：场误差收缩比 DD/G','TA3：场误差收缩比 DD/G'],fr))
sections.append('例如原波导ω12，固定DD的20步残差收缩明显更强，但场误差收缩比约为Global的1.06倍。它的绝对场误差仍更小：ROM交接时约为Global的0.205倍，20步后约为0.218倍。加长波导TA也类似：交接场误差约0.486倍，20步后约0.638倍，优势来自较好的起点，并非更快的场误差相对收缩。\n')
sections.append('数学上 r_k=p_k(AM)r_0，e_k=p_k(MA)e_0，而p_k按残差范数最优。差分对短尺度成分的放大，使“残差中占很多”与“场误差中占很多”成为不同判断。这把此前的误差组成解释进一步收准：DD可以留下易处理的界面残差，同时仍留下并不更快消退的场误差尾部。数据见 '+link('field_response.csv')+'。\n')
sections.append('## 5. 界面项解释的收获和限制\n')
rr=[]
for r in comm:
    rr.append([labels[int(r['case_id'])], '固定' if 'fixed' in r['method'] else 'TA',
      f"{num(r['bulk_over_total']):.2g}",f"{num(r['interface_over_total']):.2g}",f"{num(r['cosine']):.5f}",
      f"{num(r['interface20']):.3f}"])
sections.append(table(['案例','构造','体内项/总残差','交换子项/总残差','交叉余弦','交换子单位残差20步'],rr))
sections.append('分解使用 r=b−Σχ_B A v_B−Σ[A,χ_B]v_B，并检验重构。交换子局限于跨块模板，给出了界面残差的直接来源；差分中的h⁻²放大与Jacobi对角逆的O(h²)尺度也提供了与平滑匹配的解释。\n')
sections.append('但TA的这两项常远大于总残差、几乎完全抵消。这个结果很重要：**不能把交换子项能量直接称为TA残差中“易消除误差的占比”**。分解还依赖全局快照延拓的选择。分别求解分量会产生不同GMRES多项式，因此分量收缩不构成原求解的可加因果贡献。相比只描述“高频误差”，新解释更具体，也明确暴露了它的限制。\n')
sections.append('## 6. 完整训练空间封闭性的核验\n')
maxsnap=max(num(r['snapshot_relative_outside_full_solution_space']) for r in span)
maxbasis=max(num(r['basis_relative_outside_full_solution_space']) for r in span)
maxcol=max(num(r['max_individual_snapshot_relative_outside']) for r in span)
sections.append(f'对{len(span)}个层/参数组记录，后续逆作用快照矩阵在完整首层空间之外的最大相对Frobenius范数为{maxsnap:.3e}；单个快照的最大相对范数为{maxcol:.3e}，基矩阵的最大相对分量为{maxbasis:.3e}。有限精度泄漏主要应结合近breakdown及严格快照复核解释，不能视为额外物理信息。另一个独立96维直接求解代数检查把纯逆作用的第三层泄漏压到6.72e-13；加入恒等作用后封闭性消失。\n')
sections.append('这里的证明只覆盖ROM前缀，不包括后续MG完成场。完整固定空间的最佳逼近不劣于其子空间，但实际Galerkin投影不等于最小残差投影，所以不推出固定Galerkin在所有查询上获胜。TA的可辩护价值需要落在压缩成本、数值稳定性、顺序使用或具体交接效果上，而不是在当前条件下增加完整局部训练空间之外的信息。\n')
sections.append('## 7. 与此前高频诊断的关系\n')
sections.append('此前ω24/36/48/60冻结模型已经给出“交接幅度×方向收缩”、DD界面残差富集和较强前平滑响应。本轮新增了同网格不同频率、加长传播路径、lens与细网格的复现，固定/TA两种构造，shift和平滑干预，以及交换子分解与空间封闭性证明。旧的ω48/60大规模性能结果未重跑，也未改写。\n')
sections.append('尚未证明异质PML算子下的统一收敛界，未穷尽秩分配和固定投影形式，未证明所有参数范围与分区都有DD优势。当前证据更适合支持“局部表示改变了与完成求解器匹配的残差方向”，并把TA的额外价值作为需要准确界定的问题。\n')
sections.append('## 8. 复现与数据\n')
sections.append('原始套件源码与11个依赖保存在source/；补充套件保存在source_refinement/，分别有SHA-256清单，收尾逐项复核。原始模型/参考解位于results/case_*/。主要入口为code/run_dd_parameter_mechanism_pilot.m；cases1–6为原套件，101补细固定截断，102做中点查询。分析入口scripts/analyze_dd_parameter_mechanism_pilot.py；空间核验code/verify_dd_trajectory_span.m；独立代数检查scripts/check_trajectory_span_identity.py。\n')
sections.append('检查文件：'+link('validation.json')+'、'+link('case_validation.csv')+'、'+link('training_summary.csv')+'、'+link('span_closure.csv')+'。完整数学推导：'+link('theory.md')+'。\n')
(OUT/'report.md').write_text('\n'.join(sections))
print(OUT/'report.md')
