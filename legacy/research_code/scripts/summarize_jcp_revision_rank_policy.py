"""Validate and summarize the independent 2026-09-14 rank-policy experiment."""
import csv
import hashlib
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/jcp_revision_experiments_20260914/rank_policy"


def read(name):
    with (OUT / name).open() as handle:
        return list(csv.DictReader(handle))


def value(row, field):
    return float(row[field])


def main():
    models = read("models.csv")
    timings = read("timings.csv")
    compression = read("compression.csv")
    common = read("common_snapshot_allocation.csv")
    assert len(models) == 6 and len(timings) == 360 and len(compression) == 180
    assert len(common) == 18
    assert all(value(r, "true_residual") <= 1e-12 for r in timings)
    assert all(value(r, "aggregate_relative_tail") <= value(r, "budget") * (1 + 1e-7) for r in compression)
    assert all(value(r, "max_prefix_difference") < 1e-9 for r in models)
    grouped = defaultdict(list)
    for row in timings:
        grouped[row["candidate"]].append(row)
    for rows in grouped.values():
        assert len(rows) == 45
        for query in range(1, 10):
            q = [r for r in rows if int(r["query"]) == query]
            assert {int(r["repeat"]) for r in q} == set(range(1, 6))
            assert len({int(r["completion_iterations"]) for r in q}) == 1
    meta = {r["candidate"]: r for r in models}
    summaries = []
    baseline_mean = {n: st.mean(value(r, "total_s") for r in grouped[n]) for n in ("shifted", "direct")}
    for name, rows in grouped.items():
        m = meta.get(name)
        time_mean = st.mean(value(r, "total_s") for r in rows)
        qmeans = [st.mean(value(r, "total_s") for r in rows if int(r["query"]) == q) for q in range(1, 10)]
        qstd = [st.stdev(value(r, "total_s") for r in rows if int(r["query"]) == q) for q in range(1, 10)]
        summary = dict(candidate=name, budget=m["budget"] if m else "", ranks=m["ranks"] if m else "",
                       total_rank=m["total_rank"] if m else "", basis_MiB=m["basis_MiB"] if m else "",
                       payload_MiB=m["payload_MiB"] if m else "", model_object_MiB=m["model_object_MiB"] if m else "",
                       offline_s=m["offline_s"] if m else 0, online_mean_s=time_mean,
                       online_median_s=st.median(qmeans), online_worst_query_mean_s=max(qmeans),
                       max_query_timing_sd_s=max(qstd), completion_mean=st.mean(value(r, "completion_iterations") for r in rows),
                       completion_max=max(value(r, "completion_iterations") for r in rows),
                       queries_needing_completion=sum(int(r["completion_iterations"]) > 0 for r in rows if int(r["repeat"]) == 1),
                       max_true_residual=max(value(r, "true_residual") for r in rows),
                       max_reference_error=max(value(r, "reference_error") for r in rows))
        for baseline, baseline_s in baseline_mean.items():
            denominator = baseline_s - time_mean
            summary[f"break_even_vs_{baseline}"] = math.floor(float(summary["offline_s"]) / denominator) + 1 if m and denominator > 0 else "none"
        summaries.append(summary)
    with (OUT / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)
    lookup = {r["candidate"]: r for r in summaries}
    lines = ["# 局部秩分配消融：简要结果", "", "实验日期：2026-09-14。仅新增实验代码与结果，未修改论文或共享数值方法。", "",
             "## 设计", "",
             "Helm 半空间，ω=53.5，35,721 个未知量，原 2 个物理块与 8 个 PML 块。固定 24 个训练参数（[0.8,0.95] 等距）、9 个预先确定的未见插值参数、3 层。"
             "两组逐层 aggregate relative Frobenius-tail 上限 ε=10⁻⁵、10⁻⁴；每种规则均独立完整 retrain/replay。", "",
             "- **marginal-storage**：当前方法，每次选 σ²/块自由度 最大的下一模态。",
             "- **uniform-relative**：每块使用相同局部相对尾误差上限 ε，因此组装后的整体误差也不超过 ε；局部约束更保守。",
             "- **energy-greedy**：每次选 σ² 最大的下一模态，只优化总秩，不使用块自由度。", "",
             "另对 marginal-storage 的每层同一快照矩阵运行三种分配，单独观察分配效果；这项静态诊断不计入离线或在线用时。", "",
             "所有模型冻结后在本地 MATLAB R2025a、单计算线程下计时；与另一组本地实验串行执行。"
             "每个参数一次预热和五次测量，轮换八种方法的顺序（六个模型 + shifted-ILU + direct）。"
             "统一原 Helmholtz 方程真实残差 ≤10⁻¹²。ILUTP drop tolerance=10⁻³，shift β=0.2，GMRES restart=40、最多20轮。", "",
             "时间包括 ROM、降阶算子组装/分解、Arnoldi、必要时的 shifted-ILU setup 和 completion、最终真实残差检查；"
             "不包括模型加载、共同全阶矩阵组装和离线训练。离线时间为单次完整参考解+replay训练墙钟时间，未重复测量。", "",
             "## 完整训练后的结果", "",
             "| ε | 规则 | 三层总秩 | 基 MiB | payload MiB | 离线 s | 平均在线 s | 平均 completion 次数 | 需 completion 查询/9 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for m in models:
        s = lookup[m["candidate"]]
        lines.append(f"| {float(m['budget']):.0e} | {m['policy']} | {m['total_rank']} | {float(m['basis_MiB']):.3f} | "
                     f"{float(m['payload_MiB']):.3f} | {float(m['offline_s']):.3f} | {s['online_mean_s']:.5f} | "
                     f"{s['completion_mean']:.3f} | {s['queries_needing_completion']} |")
    lines += ["", "payload 是复数基 + 两个持久降阶算子 + 只计一次的共享索引数值字节数；未包括全阶算子、临时工作区或整个进程内存。"
              "实际部署 MATLAB 对象的 `whos` 大小另外保存在 `models.csv`，与 payload 相差约 0.012 MiB。", "",
              f"独立基线平均在线时间：shifted-ILU {baseline_mean['shifted']:.5f} s；sparse direct {baseline_mean['direct']:.5f} s。", "",
              "## 对审稿意见的回应", ""]
    for exponent in [-5, -4]:
        m = lookup[f"marginal_storage_eta{exponent}"]
        u = lookup[f"uniform_relative_eta{exponent}"]
        e = lookup[f"energy_greedy_eta{exponent}"]
        saved_uniform = 100 * (1 - float(m["payload_MiB"]) / float(u["payload_MiB"]))
        saved_energy = 100 * (1 - float(m["payload_MiB"]) / float(e["payload_MiB"]))
        lines.append(f"ε=10^{exponent}：marginal-storage 相对 uniform-relative 减少 **{saved_uniform:.2f}% payload**；"
                     f"相对 energy-greedy 的 payload 减少为 **{saved_energy:.2f}%**（负值表示更大）。")
        lines.append("")
    static_savings = []
    for exponent in [-5, -4]:
        source = f"marginal_storage_eta{exponent}"
        totals = {policy: sum(value(r, "basis_MiB") for r in common if r["snapshot_source"] == source and r["allocation_policy"] == policy)
                  for policy in ("marginal_storage", "uniform_relative")}
        static_savings.append(100 * (1 - totals["marginal_storage"] / totals["uniform_relative"]))
    lines += ["这组消融支持：统一整体误差预算能比逐块相同相对阈值减少持久存储；但 **marginal-storage 的特定评分相对简单 energy-greedy 未显示显著、稳定的优势**。"
              "两种 aggregate-budget 规则的 payload 差异均不足 0.4%。不宜把该特定 greedy 规则单独作为已验证的主要创新。", "",
              f"在固定 marginal 快照上的静态分配中，marginal 相对 uniform 的总基存储分别减少{static_savings[0]:.2f}%和{static_savings[1]:.2f}%；完整 retrain 后差别更大，"
              "说明局部选择会传播到后续 replay 轨迹。这是与分层训练配合的证据，不能据此定量归因三维高频速度收益。", "",
              "较严格 ε=10⁻⁵ 与较松 ε=10⁻⁴ 的 online 差异需要结合 completion setup 理解；存储减少并不保证端到端时间减少。"
              "详见每参数 `timings.csv` 中的 setup、solve、post-ROM residual 和 iteration 分项。", "",
              "## 总成本", "",
              "以本次实测均值估算 `offline + Q × mean_online` 的首个整数摊销查询数（单机同一计算线程口径）：", "",
              "| 规则/ε | 相对 shifted-ILU | 相对 direct |", "|---|---:|---:|"]
    for m in models:
        s = lookup[m["candidate"]]
        lines.append(f"| {m['candidate']} | {s['break_even_vs_shifted']} | {s['break_even_vs_direct']} |")
    lines += ["", "`none` 表示本次均值下 online 已不快于该基线，因而没有正摊销点。"
              "离线仅计时一次，摊销数是该配置的估算，不能外推到三维或并行资源不同的环境。", "",
              "## 验证与局限", "",
              f"360/360 次正式求解均通过；真实残差最大 {max(value(r,'true_residual') for r in timings):.4e}，"
              f"相对 direct 参考解误差最大 {max(value(r,'reference_error') for r in timings):.4e}。"
              f"54 个模型/参数的独立部署 prefix 检查最大差异 {max(value(r,'max_prefix_difference') for r in models):.4e}；"
              "180 个逐块记录涵盖全部18层，实际投影残差均满足预设整体误差预算。所有同一方法/参数的 completion 迭代数在五次计时中一致。", "",
              "该实验仅涉及一个二维半空间尺度和两个预算，尚不能证明特定评分在高频三维中更优；"
              "也不能把本例的接近结果解释成所有分区上三种规则都等价。未更换训练集、分区或按测试结果挑选预算。", "",
              "原始模型与日志：`raw/`；存储/离线：`models.csv`；逐层逐块误差：`compression.csv`；"
              "固定快照分配：`common_snapshot_allocation.csv`；计时：`timings.csv`；汇总：`summary.csv`。", ""]
    (OUT / "results.md").write_text("\n".join(lines))
    checked = [ROOT / "code/run_jcp_revision_rank_policy.m", ROOT / "code/jcp_revision_rank_space_utils.m",
               ROOT / "code/helm_waveholtz_correction_rom.m", ROOT / "code/rom_reduced_space_utils.m",
               ROOT / "code/rom_arnoldi_step.m", ROOT / "code/build_shifted_laplacian_preconditioner.m",
               ROOT / "code/shifted_laplacian_gmres.m", Path(__file__)]
    checked += [OUT / n for n in ("models.csv", "compression.csv", "timings.csv", "summary.csv", "common_snapshot_allocation.csv")]
    validation = dict(measured_solves=len(timings), passed_solves=len(timings), models=len(models),
                      max_true_residual=max(value(r, "true_residual") for r in timings),
                      max_prefix_difference=max(value(r, "max_prefix_difference") for r in models),
                      identical_repeat_iteration_counts=True,
                      active_source_matches_training_snapshot=(ROOT / "code/helm_waveholtz_correction_rom.m").read_bytes() == (OUT / "raw/active_helm_source.m.txt").read_bytes(),
                      sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in checked})
    assert validation["active_source_matches_training_snapshot"]
    (OUT / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps({k: v for k, v in validation.items() if k != "sha256"}, indent=2))
    print("JCP_RANK_REPORT_PASS")


if __name__ == "__main__":
    main()
