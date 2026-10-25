#!/usr/bin/env python3
"""Audit current JCP total-cost claims using frozen, full-precision source rows.

This is a reanalysis, not a new solver experiment. It writes only below the
revision output directory, and never regenerates manuscript tables.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs/jcp_revision_experiments_20260914/cost"
SOURCES: dict[str, dict] = {}
CHECKS: list[str] = []


def source(relative: str | Path) -> Path:
    path = ROOT / relative
    relative = str(path.relative_to(ROOT))
    if relative not in SOURCES:
        SOURCES[relative] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }
    return path


def rows(relative):
    with source(relative).open(newline="") as stream:
        return list(csv.DictReader(stream))


def document(relative):
    return json.loads(source(relative).read_text())


def write_csv(path, data):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), (actual, expected)


def repeated_query_means(data, repeats, query_keys, repeat_key, tolerance):
    groups = defaultdict(list)
    for row in data:
        assert float(row["true_residual"]) <= tolerance
        if "flag" in row:
            assert int(row["flag"]) == 0
        key = tuple(row[k] for k in query_keys)
        groups[key].append(row)
    result = {}
    for key, group in groups.items():
        assert sorted(int(r[repeat_key]) for r in group) == list(range(1, repeats + 1))
        result[";".join(key)] = mean(float(r["total_s"]) for r in group)
    return result


def collect():
    studies = []
    training = rows("outputs/reviewer_20260909/summary_train.csv")
    training += rows("outputs/reviewer_20260909/summary_train_frequency_4.csv")
    train = {(r["study"], r["variant"]): r for r in training}
    paired = rows("outputs/dd_online_audit_20260910/paired_timings.csv")
    baselines = rows("outputs/dd_online_audit_20260910/baseline_timings.csv")
    assert len(paired) == 400 and len(baselines) == 200
    consolidated = rows("outputs/paper_restructure_20260910/consolidated_online_summary.csv")
    expected = {(float(r["omega"]), r["method"]): float(r["mean_online_s"]) for r in consolidated}
    for scale in (1, 1.5, 2, 4):
        case = f"frequency_{scale:g}"
        t = {m: train[case, m] for m in ("global", "dd")}
        online = {}
        for method in ("global", "dd", "direct", "shifted"):
            raw = paired if method in t else baselines
            selected = [r for r in raw if float(r["scale"]) == scale
                        and r["variant"] == method and r.get("optimized", "1") == "1"]
            online[method] = repeated_query_means(selected, 5, ["c_right"], "repetition", 1e-10)
            assert len(online[method]) == 5
            close(mean(online[method].values()), expected[53.5 * scale, method])
        studies.append(dict(
            study=f"halfspace_frequency_{53.5 * scale:g}", omega=53.5 * scale,
            N=int(t["global"]["N"]), online=online,
            offline={m: float(t[m]["offline_s"]) for m in t}, repeats=5,
            resource_note="Local M4 Max / R2025a; serial offline; one-thread online; separate timing batches",
            scope_note="Frozen frequency models; direct includes fresh factorization; offline is one in-memory construction measurement",
            offline_kind="measured_complete_in_memory_single_run",
        ))
    CHECKS.append("All 400 selected frequency measurements pass 1e-10; raw means reproduce the current consolidated JCP source to 1e-12.")

    online = {}
    for method in ("global", "dd", "direct", "shifted"):
        data = rows(f"outputs/reviewer_20260909/two_parameter/{method}_timing.csv")
        online[method] = repeated_query_means(data, 5, ["c_left", "c_right"], "repetition", 1e-10)
        assert len(online[method]) == 9
    studies.append(dict(
        study="halfspace_two_parameter_development", omega=53.5, N=35721,
        online=online, offline={m: float(train["two_parameter", m]["offline_s"]) for m in ("global", "dd")},
        repeats=5, resource_note="Local M4 Max / R2025a; serial; historical batch",
        scope_note="36 training points; nine development queries, not the independent selected-configuration holdout",
        offline_kind="measured_complete_in_memory_single_run",
    ))

    current = rows("outputs/jcp_shifted_20260914/paper_timings.csv")
    assert len(current) == 910
    old_train = rows("outputs/halfspace_dd_refinement/review_repaired_20260905/summary.csv")
    old_offline = {r["variant"]: float(r["offline_s"]) for r in old_train}
    current_summary = rows("outputs/jcp_shifted_20260914/summary.csv")
    current_expected = {(r["problem"], r["method"]): float(r["mean_total_s"]) for r in current_summary}
    provenance = document("outputs/jcp_shifted_20260914/provenance.json")
    timing_entry = next(r for r in provenance["outputs_and_analysis"] if r["path"].endswith("paper_timings.csv"))
    assert SOURCES[timing_entry["path"]]["sha256"] == timing_entry["sha256"]
    for problem, omega, n, count, tolerance in [("halfspace", 53.5, 35721, 23, 1e-12), ("ecc", 18, 21025, 11, 1e-10)]:
        online = {}
        for method in ("global", "dd", "direct", "shifted"):
            selected = [r for r in current if r["problem"] == problem and r["method"] == method]
            online[method] = repeated_query_means(selected, 5, ["parameter"], "repeat", tolerance)
            assert len(online[method]) == count
            close(mean(online[method].values()), current_expected[problem, method])
        offline = ({"global": old_offline["global"], "dd": old_offline["slabs1-aligned-sym8"]}
                   if problem == "halfspace" else {"global": None, "dd": None})
        studies.append(dict(
            study=f"{problem}_representative_current", omega=omega, N=n, online=online,
            offline=offline, repeats=5,
            resource_note="Local M4 Max / R2025a; one-thread serial current online; archived offline resource match not established",
            scope_note=("Archived complete in-memory offline cost of the exact frozen model, including verified-snapshot replacements; current online is a later shifted-ILU batch"
                        if problem == "halfspace" else "Two-layer truncation of archived models; complete standalone two-layer offline cost not recorded (prepare_ecc_two_layer_paper sets NaN)"),
            offline_kind=("archived_frozen_model_single_run_cross_batch" if problem == "halfspace" else "not_recorded"),
        ))
    CHECKS.append("Current half-space/ECC selected raw rows reproduce summary.csv; all selected repeats pass their problem's residual target. ECC offline remains missing.")

    frozen = document("outputs/waveguide_frequency_final_20260913/waveguide_frequency_sources.json")
    for filename, expected_hash in frozen.items():
        source(filename)
        assert SOURCES[filename]["sha256"] == expected_hash, filename
    frozen_rows = {int(r["omega"]): r for r in rows("outputs/waveguide_frequency_final_20260913/waveguide_frequency.csv")}
    for omega in (24, 36, 48, 60):
        folder = f"outputs/scnet_3d_20260909/remote_v7/full_w{omega}_p8_t4_b2_sr40_or40_sm40_om40_c12000"
        if omega == 60:
            folder = "outputs/scnet_3d_20260909/remote_recovery_w60"
        report = document(f"{folder}/report.json")
        cfg = document(f"{folder}/configuration.json")["cfg"]
        assert report["all_converged"] and report["snapshot_max_relres"] <= 1e-7
        assert cfg["train_count"] == 8 and cfg["num_layers"] == 3
        online = {}
        for method in ("global", "dd", "shifted"):
            values = {}
            for q in report["tests"]:
                result = q[method]
                assert result["flag"] == 0 and result["true_residual"] <= 1e-6
                values[f"{q['mu']:.3f}"] = result["seconds"]
            assert set(values) == {"0.237", "0.317", "0.483"}
            online[method] = values
            close(mean(values.values()), float(frozen_rows[omega][method + "_online_s"]))
        csv_path = next((ROOT / folder).glob("helmholtz_3d_candidates_*.csv"))
        candidates = rows(csv_path.relative_to(ROOT))
        offline = {}
        for method in ("global", "dd"):
            values = {float(r["offline_s"]) for r in candidates if r["method"] == method}
            assert len(values) == 1
            offline[method] = values.pop() if omega < 60 else None
            if omega < 60:
                assert not report["training_resumed"]
                close(offline[method], float(frozen_rows[omega][method + "_offline_s"]))
        studies.append(dict(
            study=f"waveguide_3d_{omega}", omega=omega, N=int(report["N"]), online=online,
            offline=offline, repeats=1,
            resource_note=f"Hygon 7285 / R2024a; 64 allocated CPUs; {cfg['parallel_workers']} training workers x 4 threads; serial online capped at 8 threads; node {report['metadata']['node']}",
            scope_note="Three queries timed once; mixed parallel resource wall-clock estimate, not CPU cost; direct baseline not measured",
            offline_kind="measured_parallel_wall_time_single_run" if omega < 60 else "incomplete_resumed_training_omitted",
        ))
    CHECKS.append("All frozen 3D source hashes verified; 36 raw online solves pass 1e-6; recomputed online/offline values match the current frozen table. Recovered omega60 offline is deliberately omitted.")
    for path in ["scripts/analyze_jcp_revision_cost.py", "scripts/write_reviewer_20260909_evidence.py",
                 "scripts/plot_waveguide_frequency_paper.py", "code/prepare_ecc_two_layer_paper.m",
                 "code/run_jcp_shifted_experiments.m", "agent_runs/2026-09-10_halfspace_paper_restructure.md",
                 "agent_runs/2026-09-11_experiment23_paper_revision.md", "agent_runs/2026-09-14_jcp_shifted_experiments.md",
                 "agent_runs/2026-09-05_reviewer_point_by_point_audit.md"]:
        source(path)
    return studies


def break_even(offline, saving):
    # Even with a missing offline measurement, nonpositive saving excludes a
    # positive break-even for every nonnegative training cost.
    if saving <= 0:
        return "no_finite_break_even", None, None
    if offline is None:
        return "offline_missing", None, None
    crossing = offline / saving
    return "finite", crossing, max(1, math.floor(crossing) + 1)


def analyze(studies, out):
    costs, query_deltas, curves = [], [], []
    for s in studies:
        averages = {m: mean(v.values()) for m, v in s["online"].items()}
        strongest = min((m for m in averages if m in ("direct", "shifted")), key=averages.get)
        for method in ("global", "dd"):
            offline = s["offline"][method]
            for baseline in (m for m in averages if m in ("direct", "shifted")):
                assert s["online"][method].keys() == s["online"][baseline].keys()
                deltas = {q: s["online"][baseline][q] - value for q, value in s["online"][method].items()}
                saving = mean(deltas.values())
                status, crossing, integer_q = break_even(offline, saving)
                loo = [break_even(offline, mean(v for qq, v in deltas.items() if qq != q)) for q in deltas]
                finite_loo = [entry[1] for entry in loo if entry[0] == "finite"]
                row = dict(
                    study=s["study"], omega=s["omega"], N=s["N"], method=method,
                    baseline=baseline, strongest_measured_baseline=baseline == strongest,
                    evidence_kind="reanalysis_of_frozen_existing_measurements",
                    queries=len(deltas), repetitions_per_query=s["repeats"],
                    offline_s=offline, online_mean_s=averages[method], baseline_mean_s=averages[baseline],
                    saving_per_query_s=saving, break_even_status=status,
                    break_even_real_Q=crossing, first_strictly_faster_integer_Q=integer_q,
                    paired_faster_queries=sum(v > 0 for v in deltas.values()),
                    loo_min_finite_break_even_Q=min(finite_loo) if finite_loo else None,
                    loo_max_finite_break_even_Q=max(finite_loo) if finite_loo else None,
                    loo_no_finite_count=sum(entry[0] == "no_finite_break_even" for entry in loo),
                    offline_kind=s["offline_kind"], resource_note=s["resource_note"], scope_note=s["scope_note"],
                )
                costs.append(row)
                for q, delta in deltas.items():
                    query_deltas.append(dict(study=s["study"], method=method, baseline=baseline,
                                             query=q, ROM_online_mean_s=s["online"][method][q],
                                             baseline_online_mean_s=s["online"][baseline][q], saving_s=delta))
                if baseline == strongest:
                    for count in (1, 5, 10, 25, 50, 100, 250, 500, 1000):
                        total = offline + count * averages[method] if offline is not None else None
                        base_total = count * averages[baseline]
                        curves.append(dict(study=s["study"], method=method, baseline=baseline, Q=count,
                                           ROM_offline_s=offline, ROM_online_total_s=count * averages[method],
                                           ROM_total_s=total, baseline_total_s=base_total,
                                           baseline_over_ROM_total=base_total / total if total else None))
    write_csv(out / "all_baseline_costs.csv", costs)
    selected = [r for r in costs if r["strongest_measured_baseline"]]
    write_csv(out / "strongest_baseline_costs.csv", selected)
    write_csv(out / "paired_query_savings.csv", query_deltas)
    write_csv(out / "query_count_costs.csv", curves)
    return selected


def report(selected, out, has_plot):
    lines = ["# JCP 补强实验：既有数据的总成本审计", "",
             "本表是冻结数据的重新计算，不是新数值求解；不包含本轮其他实验尚未完成的结果。所有均值从原始 CSV/JSON 的全精度逐查询数据计算，并与当前稿件使用的汇总来源核对。", "",
             "计算约定：每个查询先对重复计时取均值，再对查询等权平均。"
             "总成本 C_ROM(Q)=T_offline+Q×mean(T_online)，C_base(Q)=Q×mean(T_base)。"
             "当 ΔT=mean(T_base−T_online)>0 且离线成本已知时，实数交点 Q*=T_offline/ΔT；"
             "表中整数为首次严格更快的查询数 floor(Q*)+1。ΔT≤0 时对任何非负离线成本均无摊销点。"
             "共同矩阵组装、模型加载、归档 I/O 未计入；逐查询 direct 新分解及需要时的 completion/setup 已计入。", "",
             "每个案例选择本批次已测 full-order 方法中平均在线时间最小者作为基线；三维只有 shifted-MG，不能解释为它优于所有潜在直接/迭代方法。", "",
             "| 既有案例 | 方法 | 基线 | 离线 s | 在线 s | 基线在线 s | 首次更快 Q | 逐点更快 |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    for r in selected:
        offline = "未记录" if r["offline_s"] is None else f"{r['offline_s']:.3f}"
        q = str(r["first_strictly_faster_integer_Q"]) if r["break_even_status"] == "finite" else ("无" if r["break_even_status"] == "no_finite_break_even" else "未知")
        lines.append(f"| {r['study']} | {r['method']} | {r['baseline']} | {offline} | {r['online_mean_s']:.6f} | {r['baseline_mean_s']:.6f} | {q} | {r['paired_faster_queries']}/{r['queries']} |")
    lines += ["", "解释与限制：", "",
              "- 二维频率案例使用 2026-09-10 的 optimized=1 ROM 计时与刷新基线；旧 frequency_4/*_timing.csv 是另一批次，不能混入当前比较。",
              "- 三维 ω=24/36/48 的 Global/DD 离线使用 8 个进程、每进程 4 线程；在线串行、8 线程上限。表中交点是这套并行安排下的墙钟情景估算，不是等 CPU 时、能耗或云费用交点。尚未提供统一资源的离线/在线计时与 CPU 用量。",
              "- ω=60 的训练从缓存恢复，完整独立离线成本缺失，因而不计算总成本或摊销点。",
              "- 当前代表性 half-space 的旧离线模型与新 shifted-ILU 在线批次不同；保留旧测量用于初步情景分析，需重测一致训练流程/资源才能作为严格总成本证据。ECC 的两层模型由已存模型截断得到，独立两层离线成本未记录；即使离线免费，其当前平均在线时间仍超过 direct。",
              "- Q 代表每个参数需要一次新求解、总体查询分布与当前样本相同的情景；重复相同参数时的分解/解缓存不在此模型内。摊销曲线不证明未知查询分布也有相同收益。",
              "- 同一查询的重复计时不是独立参数样本。paired_query_savings.csv 逐点保存配对差；all_baseline_costs.csv 中 leave-one-query-out 仅用于敏感性检查，不能当成置信区间。", ""]
    critical = next(r for r in selected if r["study"] == "halfspace_frequency_214" and r["method"] == "dd")
    lines += [f"ω=214 的 DD 均值交点为 {critical['break_even_real_Q']:.2f} 次，首次严格更快为 {critical['first_strictly_faster_integer_Q']} 次。"
              f"删去唯一需要 completion 的困难查询后，交点为 {critical['loo_min_finite_break_even_Q']:.2f} 次；"
              f"删去任一其他查询则得到 {critical['loo_no_finite_count']} 个不存在正摊销点的留一子集。该结果对困难查询比例敏感，不能只报告约 471 次而省略分布条件。", "",
              "对本轮补强实验的最小统计建议：冻结模型后按预先确定的查询区间验证；逐点报告 Global/DD/基线完整时间、重复计时散布和真实残差；"
              "用查询作为配对单位统计胜率、时间差与 completion 差；三维新结果应在同一节点/线程配置下独立成表，待离线资源匹配后再更新总成本。", "",
              "来源与复核：source_manifest.json 保存所有输入 SHA-256；validation.json 保存源表对齐与验收记录。主结果 CSV 为 strongest_baseline_costs.csv；"
              "all_baseline_costs.csv 另保留 shifted 基线等其他已测比较；query_count_costs.csv 给出 Q=1–1000 的情景值。", ""]
    if has_plot:
        lines += ["![既有测量的总墙钟情景](total_cost.png)", "",
                  "图中三维子图使用不同离线/在线并行资源，二维子图采用分开批次的历史离线与优化在线。图为情景计算，不是新增计时结果。", ""]
    (out / "cost_report.md").write_text("\n".join(lines))


def plot(selected, out):
    os.environ.setdefault("MPLCONFIGDIR", str(out / "matplotlib_cache"))
    os.environ.setdefault("XDG_CACHE_HOME", str(out / "system_cache"))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return False
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.1))
    configurations = [("halfspace_frequency_53.5", "2D half-space, omega = 53.5", 200),
                      ("halfspace_frequency_214", "2D half-space, omega = 214", 1000),
                      ("waveguide_3d_48", "3D waveguide, omega = 48\nParallel-offline wall-clock scenario", 30)]
    for ax, (study, title, maximum) in zip(axes, configurations):
        values = [r for r in selected if r["study"] == study]
        q = np.linspace(0, maximum, 400)
        ax.plot(q, q * values[0]["baseline_mean_s"], color="#52525b", label=values[0]["baseline"])
        for method, color in [("global", "#b64135"), ("dd", "#19767d")]:
            row = next(r for r in values if r["method"] == method)
            ax.plot(q, row["offline_s"] + q * row["online_mean_s"], color=color, label=method.upper())
            star = row["break_even_real_Q"]
            ax.scatter([star], [star * row["baseline_mean_s"]], color=color, s=22)
        ax.set(title=title, xlabel="Number of queries Q", ylabel="Total wall time (s)", xlim=(0, maximum), ylim=(0, None))
        ax.grid(alpha=.18)
        ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"total_cost.{ext}", dpi=190, bbox_inches="tight")
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    studies = collect()
    selected = analyze(studies, args.output)
    has_plot = plot(selected, args.output)
    report(selected, args.output, has_plot)
    (args.output / "source_manifest.json").write_text(json.dumps(SOURCES, indent=2) + "\n")
    (args.output / "validation.json").write_text(json.dumps(dict(
        passed=True, evidence_kind="reanalysis_of_frozen_existing_measurements",
        analysis_python=sys.executable,
        study_count=len(studies), strongest_baseline_comparisons=len(selected),
        formulas={"ROM_total": "offline_s + Q * online_mean_s", "baseline_total": "Q * baseline_mean_s",
                  "break_even_real": "offline_s / (baseline_mean_s - online_mean_s) when denominator > 0",
                  "first_strictly_faster_integer": "max(1, floor(break_even_real) + 1)"},
        checks=CHECKS, chart_generated=has_plot,
    ), indent=2) + "\n")
    print(f"JCP_REVISION_COST_AUDIT_PASS studies={len(studies)} comparisons={len(selected)} output={args.output}")


if __name__ == "__main__":
    main()
