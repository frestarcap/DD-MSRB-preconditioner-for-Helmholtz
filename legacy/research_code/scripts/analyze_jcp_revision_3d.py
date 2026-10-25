#!/usr/bin/env python3
"""Summarize the frozen omega-48 validation only after all 10 queries finish.

Original queries: three measured ROM repetitions plus one full warm-up each.
New queries and shifted-MG baselines: single measurements. Never count warm-ups
or repetitions as independent parameter samples. This script launches no solver.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLOUD = ROOT / "outputs/jcp_revision_experiments_20260914/cloud"
METHODS = ("global3", "dd3", "global2", "shifted_mg")
DEPTHS = {"global3": 3, "dd3": 3, "global2": 2, "shifted_mg": 0}
TRAIN = [.2 + .4 * i / 7 for i in range(8)]
FRACTIONS = [.31, .67, .43, .59, .37, .71, .53]
QUERIES = [.237, .317, .483] + [TRAIN[i] + (TRAIN[i + 1] - TRAIN[i]) * f for i, f in enumerate(FRACTIONS)]
TOL = 1e-6


def require(condition, message):
    if not condition:
        raise ValueError(message)


def near(a, b, absolute=1e-12, relative=1e-10):
    return math.isclose(float(a), float(b), abs_tol=absolute, rel_tol=relative)


def numbers(value):
    if isinstance(value, list):
        return [float(x) for x in value]
    return [float(value)]


def truth(value):
    return value is True or str(value).lower() in ("true", "1")


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def recorded_input_hashes(before, after, index, protocol):
    require(before.read_bytes() == after.read_bytes(), f"q{index}: input before/after SHA-256 records differ")
    hashes = {}
    for line in before.read_text().splitlines():
        match = re.fullmatch(r"([0-9a-fA-F]{64})[ \t]+\*?(.+)", line)
        require(match is not None, f"q{index}: malformed input SHA-256 record")
        digest, filename = match.groups()
        require(filename not in hashes, f"q{index}: duplicate input SHA-256 path {filename}")
        hashes[filename] = digest.lower()
    require(set((protocol["source"], protocol["source_report"])).issubset(hashes), f"q{index}: frozen training/report input hashes missing")
    return hashes


def consistent_input_hashes(queries):
    shared = {}
    for q in queries:
        for filename, digest in q["recorded_frozen_source_sha256"].items():
            require(filename not in shared or shared[filename] == digest,
                    f"q{q['index']}: cross-query input SHA-256 mismatch for {filename}")
            shared[filename] = digest
    return shared


def load_query(folder, index, allow_fixture=False):
    paths = [folder / name for name in ("protocol.json", "complete.json", "histories.json", "timings.csv",
                                      "input_sha256_before.txt", "input_sha256_after.txt")]
    require(all(p.is_file() for p in paths), f"q{index}: required files missing")
    protocol, complete, histories = [json.loads(p.read_text()) for p in paths[:3]]
    require(allow_fixture or not protocol.get("synthetic_fixture", False), "Synthetic data cannot be accepted as production evidence")
    input_hashes = recorded_input_hashes(paths[4], paths[5], index, protocol)
    require(complete.get("all_converged") is True, f"q{index}: no production all_converged=true marker")
    require(all(complete.get(k) == value for k, value in protocol.items()), f"q{index}: protocol/complete metadata mismatch")
    require(protocol["query_index"] == index and near(protocol["mu"], QUERIES[index - 1]), f"q{index}: unexpected query")
    require(protocol["omega"] == 48 and protocol["N"] == 5543003, f"q{index}: unexpected problem")
    for key, expected in [("all_queries", QUERIES), ("train_parameters", TRAIN), ("stratified_fractions", FRACTIONS)]:
        require(len(protocol[key]) == len(expected) and all(near(a, b) for a, b in zip(protocol[key], expected)), f"q{index}: {key} mismatch")
    require(all(abs(protocol["mu"] - t) > 1e-10 for t in TRAIN), f"q{index}: a training point was used")
    repeats = 3 if index <= 3 else 1
    require(protocol["repetitions"] == repeats and protocol["baseline_repetitions"] == 1, f"q{index}: repetition protocol mismatch")
    require(protocol["query_kind"] == ("original_repeated" if index <= 3 else "new_stratified"), f"q{index}: query kind mismatch")
    require(protocol["full_warmup_each_rom"] is (index <= 3), f"q{index}: warm-up protocol mismatch")
    require(protocol["assembly_and_load_excluded"] and protocol["per_solve_mg_setup_included"], f"q{index}: timing scope mismatch")
    require(protocol["shift_beta"] == 2 and protocol["thread_cap"] == 8, f"q{index}: expected shift=2 and thread_cap=8")
    gm = protocol["gmres_options"]
    require(gm["restart"] == 40 and gm["maxit"] == 40 and near(gm["tol"], TOL), f"q{index}: GMRES protocol mismatch")
    require(gm["orthogonalization"] == "cgs2_mex" and gm["fixed_linear_preconditioner"], f"q{index}: solver implementation mismatch")
    mg = protocol["mg_options"]
    require(mg["mg_coarse_max_dofs"] == 12000 and mg["mg_max_levels"] == 6, f"q{index}: MG hierarchy mismatch")
    require(mg["mg_pre_sweeps"] == mg["mg_post_sweeps"] == 2 and near(mg["mg_jacobi_weight"], .6), f"q{index}: MG smoother mismatch")
    require(mg["mg_cycles_per_apply"] == 1, f"q{index}: MG cycle mismatch")
    checks = {v["method"]: v for v in protocol["frozen_replay_checks"]}
    require(set(checks) == set(METHODS[:3]), f"q{index}: missing frozen replay checks")
    for method, value in checks.items():
        require(value["depth"] == DEPTHS[method], f"q{index}/{method}: wrong ROM depth")
        if index <= 3 and DEPTHS[method] == 3:
            require(near(value["post_rom_residual"], value["archived_post_rom_residual"], absolute=1e-11, relative=1e-5), f"q{index}/{method}: archived trajectory mismatch")
    raw = read_csv(paths[3])
    expected = {(m, r) for m in METHODS[:3] for r in range(0 if index <= 3 else 1, repeats + 1)} | {("shifted_mg", 1)}
    actual = [(r["method"], int(r["repetition"])) for r in raw]
    require(len(actual) == len(expected) and set(actual) == expected, f"q{index}: missing, duplicate or extra timing rows")
    measured = sum(int(r["repetition"]) > 0 for r in raw)
    warm = len(raw) - measured
    require(complete["recorded_solve_count"] == measured and complete["warmup_solve_count"] == warm, f"q{index}: recorded solve count mismatch")
    hist = {(r["method"], int(r["repetition"])): r for r in histories}
    require(len(histories) == len(hist) == len(expected) and set(hist) == expected, f"q{index}: histories missing/duplicated")
    for row in raw:
        method, rep = row["method"], int(row["repetition"])
        require(int(row["query_index"]) == index and near(row["parameter"], protocol["mu"]), f"q{index}: row query mismatch")
        require(row["node"] == protocol["node"], f"q{index}: row node mismatch")
        rho = float(row["true_residual"])
        require(math.isfinite(rho) and 0 <= rho <= TOL and truth(row["converged"]) and int(row["flag"]) == 0, f"q{index}/{method}/{rep}: failed solve")
        for key in ("total_s", "rom_s", "setup_s", "completion_s", "persistent_MiB"):
            value = float(row[key])
            require(math.isfinite(value) and value >= 0, f"q{index}/{method}: invalid {key}")
        require(float(row["total_s"]) > 0, f"q{index}/{method}: nonpositive timing")
        require(float(row["total_s"]) + 1e-6 >= sum(float(row[k]) for k in ("rom_s", "setup_s", "completion_s")), f"q{index}/{method}: timing components exceed total")
        h = hist[method, rep]
        rh, ch = numbers(h["rom_history"]), numbers(h["completion_history"])
        require(all(math.isfinite(v) and v >= 0 for v in rh + ch), f"q{index}/{method}: invalid residual history")
        require(len(ch) == int(row["completion_iterations"]) + 1, f"q{index}/{method}: completion history length mismatch")
        require(near(ch[0], row["precompletion_residual"], absolute=1e-11, relative=1e-5), f"q{index}/{method}: handoff mismatch")
        require(near(ch[-1], rho, absolute=1e-11, relative=1e-4), f"q{index}/{method}: terminal residual mismatch")
        require(near(rh[0], 1) and near(rh[-1], row["precompletion_residual"], absolute=1e-11, relative=1e-5), f"q{index}/{method}: ROM history mismatch")
        if method != "shifted_mg":
            require(len(rh) == DEPTHS[method] + 1, f"q{index}/{method}: unexpected early ROM breakdown")
            require(near(row["persistent_MiB"], checks[method]["persistent_MiB"]), f"q{index}/{method}: storage changed")
            require(near(rh[-1], checks[method]["post_rom_residual"], absolute=1e-11, relative=1e-5), f"q{index}/{method}: replay check mismatch")
    return {"index": index, "protocol": protocol, "rows": raw, "paths": paths,
            "recorded_frozen_source_sha256": input_hashes}


def accounting(cloud, queries):
    paths = sorted(cloud.glob("jobs_accounting*.txt"), key=lambda p: p.stat().st_mtime_ns)
    records = {}
    parsed_files = []
    for path in paths:
        lines = path.read_text().splitlines()
        starts = [i for i, line in enumerate(lines) if "State" in line and "ExitCode" in line and "|" in line and "JobID" in line]
        if not starts:
            continue
        parsed_files.append(str(path))
        for row in csv.DictReader(lines[starts[0]:], delimiter="|"):
            jid = row.get("JobIDRaw", row.get("JobID", ""))
            if jid:
                records[jid.strip()] = row
    logs = []
    logdir = cloud / "logs"
    if logdir.is_dir():
        for path in logdir.rglob("*"):
            if path.is_file() and path.suffix in (".out", ".log", ".txt"):
                text = path.read_text(errors="replace")
                if "Model name" in text:
                    logs.append((path, text))
    matched, resource_paths = [], []
    for q in queries:
        p = q["protocol"]
        candidates = [str(p["job_id"])]
        if p.get("array_job_id"):
            candidates.append(f"{p['array_job_id']}_{q['index']}")
        row = next((records[key] for key in candidates if key in records), None)
        batch = next((records[key + ".batch"] for key in candidates if key + ".batch" in records), None)
        state = row.get("State", "").strip() if row else "unavailable"
        exit_code = row.get("ExitCode", "").strip() if row else ""
        require(not any(state.startswith(v) for v in ("FAILED", "CANCELLED", "OUT_OF_MEMORY", "TIMEOUT", "NODE_FAIL")), f"q{q['index']}: accounting reports {state}")
        if state == "COMPLETED":
            require(exit_code == "0:0", f"q{q['index']}: nonzero Slurm exit {exit_code}")
        resource_path = cloud / "results" / f"production_q{q['index']}" / "resources.txt"
        resource = {}
        if resource_path.is_file():
            resource_paths.append(str(resource_path))
            text = resource_path.read_text(errors="replace")
            for key, pattern in [
                ("maximum_resident_KiB", r"Maximum resident set size \(kbytes\):\s*(\d+)"),
                ("user_seconds", r"User time \(seconds\):\s*([\d.]+)"),
                ("system_seconds", r"System time \(seconds\):\s*([\d.]+)"),
                ("exit_status", r"Exit status:\s*(\d+)"),
            ]:
                value = re.search(pattern, text)
                if value:
                    resource[key] = float(value.group(1))
            require(resource.get("exit_status", 0) == 0, f"q{q['index']}: GNU time reports nonzero process exit")
        cpu_model, cpu_log = None, None
        for logpath, text in logs:
            marker = re.search(rf"FROZEN_QUERY_VALIDATED index={q['index']}\b[^\n]*mode=production", text)
            filename_match = p.get("array_job_id") and re.search(rf"{re.escape(str(p['array_job_id']))}[_-]{q['index']}\D", logpath.name)
            if marker or filename_match:
                cpu = re.search(r"^\s*Model name\s*:\s*(.+)$", text, flags=re.MULTILINE)
                if cpu:
                    cpu_model, cpu_log = cpu.group(1).strip(), str(logpath)
        matched.append(dict(query_index=q["index"], job_id=p["job_id"], node=p["node"],
                            state=state, exit_code=exit_code, allocation=row, batch=batch,
                            cpu_model=cpu_model, cpu_model_log=cpu_log, gnu_time=resource,
                            memory_note="GNU time peak RSS and Slurm batch MaxRSS are separate measurements with different sampling; never substitute or equate them"))
    return {"files": [str(p) for p in paths], "parsed_files": parsed_files,
            "resource_files": resource_paths, "cpu_log_files": sorted({r["cpu_model_log"] for r in matched if r["cpu_model_log"]}),
            "cpu_models": sorted({r["cpu_model"] for r in matched if r["cpu_model"]}),
            "all_ten_completed_exit_zero": len(matched) == 10 and all(r["state"] == "COMPLETED" and r["exit_code"] == "0:0" for r in matched),
            "queries": matched}


def statistics(queries):
    by_query = []
    for q in queries:
        methods = {}
        for method in METHODS:
            rs = [r for r in q["rows"] if r["method"] == method and int(r["repetition"]) > 0]
            times = [float(r["total_s"]) for r in rs]
            sd = stdev(times) if len(times) > 1 else None
            methods[method] = dict(
                measured_repetitions=len(rs), timing_kind="repeated" if len(rs) > 1 else "single_measurement",
                mean_total_s=mean(times), sample_sd_total_s=sd, CV_total=sd / mean(times) if sd is not None else None,
                min_total_s=min(times), max_total_s=max(times),
                mean_rom_s=mean(float(r["rom_s"]) for r in rs),
                mean_setup_s=mean(float(r["setup_s"]) for r in rs),
                mean_completion_s=mean(float(r["completion_s"]) for r in rs),
                mean_completion_iterations=mean(int(r["completion_iterations"]) for r in rs),
                min_completion_iterations=min(int(r["completion_iterations"]) for r in rs),
                max_completion_iterations=max(int(r["completion_iterations"]) for r in rs),
                max_true_residual=max(float(r["true_residual"]) for r in rs),
                persistent_MiB=float(rs[0]["persistent_MiB"]),
            )
        paired = {m: dict(baseline_over_DD_mean_time=methods[m]["mean_total_s"] / methods["dd3"]["mean_total_s"],
                          DD_minus_baseline_mean_time_s=methods["dd3"]["mean_total_s"] - methods[m]["mean_total_s"],
                          DD_minus_baseline_mean_completion_iterations=methods["dd3"]["mean_completion_iterations"] - methods[m]["mean_completion_iterations"])
                  for m in ("global3", "global2", "shifted_mg")}
        by_query.append(dict(query_index=q["index"], parameter=q["protocol"]["mu"],
                             query_kind=q["protocol"]["query_kind"], node=q["protocol"]["node"], methods=methods,
                             DD_paired_comparisons=paired))
    cohorts = {}
    for name, chosen in [("original_three_repeated", by_query[:3]), ("new_seven_single", by_query[3:]), ("all_ten_equal_query_weight", by_query)]:
        methods = {m: {"mean_total_s": mean(q["methods"][m]["mean_total_s"] for q in chosen),
                       "mean_completion_iterations": mean(q["methods"][m]["mean_completion_iterations"] for q in chosen),
                       "max_true_residual": max(q["methods"][m]["max_true_residual"] for q in chosen)} for m in METHODS}
        comparisons = {}
        for baseline in ("global3", "global2", "shifted_mg"):
            comparison = dict(
                baseline=baseline,
                DD_time_reduction_percent=100 * (1 - methods["dd3"]["mean_total_s"] / methods[baseline]["mean_total_s"]),
                baseline_over_DD_mean_time=methods[baseline]["mean_total_s"] / methods["dd3"]["mean_total_s"],
                geometric_mean_paired_speedup=math.exp(mean(math.log(q["DD_paired_comparisons"][baseline]["baseline_over_DD_mean_time"]) for q in chosen)),
                DD_faster_query_count=sum(q["methods"]["dd3"]["mean_total_s"] < q["methods"][baseline]["mean_total_s"] for q in chosen),
                DD_fewer_completion_query_count=sum(q["methods"]["dd3"]["mean_completion_iterations"] < q["methods"][baseline]["mean_completion_iterations"] for q in chosen),
                DD_minus_baseline_mean_completion_iterations=methods["dd3"]["mean_completion_iterations"] - methods[baseline]["mean_completion_iterations"],
            )
            comparisons[baseline] = comparison
        cohorts[name] = dict(query_count=len(chosen), methods=methods, DD_comparisons=comparisons)
    return by_query, cohorts


def plot(summary, out):
    os.environ.setdefault("MPLCONFIGDIR", str(out / "matplotlib_cache"))
    os.environ.setdefault("XDG_CACHE_HOME", str(out / "system_cache"))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    qs = sorted(summary["per_query"], key=lambda q: q["parameter"])
    colors = {"global3": "#b64135", "dd3": "#19767d", "global2": "#9d7431", "shifted_mg": "#606069"}
    labels = {"global3": "Global3", "dd3": "DD3", "global2": "Global2", "shifted_mg": "Shifted MG (single run)"}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    x = [q["parameter"] for q in qs]
    for method in METHODS:
        y = [q["methods"][method]["mean_total_s"] for q in qs]
        err = [q["methods"][method]["sample_sd_total_s"] or 0 for q in qs]
        axes[0].errorbar(x, y, yerr=err, marker="o", ms=4, capsize=3, lw=1.2, color=colors[method], label=labels[method])
    for baseline in ("global3", "global2"):
        ratios = [q["methods"][baseline]["mean_total_s"] / q["methods"]["dd3"]["mean_total_s"] for q in qs]
        axes[1].plot(x, ratios, "o-", color=colors[baseline], label=f"{labels[baseline]} / DD3")
    axes[1].axhline(1, color=".4", ls="--", lw=1)
    for ax in axes:
        for q in qs:
            if q["query_index"] <= 3:
                ax.axvline(q["parameter"], color=".4", lw=.8, ls=":", alpha=.35)
        ax.grid(alpha=.18)
        ax.legend(frameon=False, fontsize=9)
    axes[0].legend(frameon=False, fontsize=9, ncol=4, loc="upper left")
    axes[0].set(yscale="log", ylabel="Complete online time (s)", title="Frozen omega = 48: 10 unseen parameters\nDotted guides: original queries, ROM mean +/- sample SD of 3 runs")
    axes[1].set(xlabel="Parameter mu", ylabel="Paired mean-time ratio", title="Above 1: DD3 is faster; new-query timings are single runs")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"online_validation.{ext}", dpi=190, bbox_inches="tight")
    plt.close(fig)
    return True


def report(summary, out):
    lines = ["# 冻结三维 ω=48 模型：新增参数与重复计时", "",
             "数值记录已完整核验：10 个未见参数、58 次正式求解及 9 次完整预热全部通过原方程真实残差 1e-6。"
             "原有三点对每种 ROM 各重复 3 次；新增七点每种 ROM 各运行 1 次；shifted-MG 在全部十点均只测 1 次。", "",
             "统计先在每个参数内部取重复均值，再对参数等权平均。误差条仅表示同一参数重复计时的样本标准差，"
             "新增点及基线单次记录没有计时方差估计。本次不重训模型；公共组装与载入排除，逐次所需 MG setup、ROM 和 completion 包含在完整在线时间。", "",
             "| 参数组 | 方法 | 等权平均在线 s | 平均 completion | 最大真实残差 |",
             "|---|---|---:|---:|---:|"]
    names = {"original_three_repeated": "原3点：ROM重复3次", "new_seven_single": "新增7点：单次", "all_ten_equal_query_weight": "全部10点"}
    for name, cohort in summary["cohorts"].items():
        for method, r in cohort["methods"].items():
            lines.append(f"| {names[name]} | {method} | {r['mean_total_s']:.3f} | {r['mean_completion_iterations']:.2f} | {r['max_true_residual']:.3e} |")
    lines += ["", "| 参数组 | DD3 相对对象 | 平均时间减少 | 配对几何平均加速比 | 更快参数数 | completion 更少参数数 |",
              "|---|---|---:|---:|---:|---:|"]
    for name, cohort in summary["cohorts"].items():
        for method, r in cohort["DD_comparisons"].items():
            lines.append(f"| {names[name]} | {method} | {r['DD_time_reduction_percent']:.2f}% | {r['geometric_mean_paired_speedup']:.3f} | {r['DD_faster_query_count']}/{cohort['query_count']} | {r['DD_fewer_completion_query_count']}/{cohort['query_count']} |")
    lines += ["", "原三点的重复计时散布（基线只有单次，不列 CV）：", "",
              "| μ | 方法 | 平均 s | 最小–最大 s | CV | completion 最小–最大 |", "|---:|---|---:|---:|---:|---:|"]
    for q in summary["per_query"][:3]:
        for method in METHODS[:3]:
            r = q["methods"][method]
            lines.append(f"| {q['parameter']:.6f} | {method} | {r['mean_total_s']:.3f} | {r['min_total_s']:.3f}–{r['max_total_s']:.3f} | {100*r['CV_total']:.2f}% | {r['min_completion_iterations']}–{r['max_completion_iterations']} |")
    storage = summary["storage_payload_MiB"]
    lines += ["", f"持久 payload：Global3={storage['global3']:.3f} MiB，DD3={storage['dd3']:.3f} MiB，"
              f"Global2={storage['global2']:.3f} MiB。DD3 比 Global3 减少 {100*(1-storage['dd3']/storage['global3']):.2f}%；"
              f"相对 Global2 的 payload 差为 {100*(storage['dd3']/storage['global2']-1):+.2f}%。"
              "Global2 是冻结 Global 模型的两层深度对照，实际存储差在此明确给出；不把它称为严格等存储的重新优化方法。", "",
              f"计算节点：{', '.join(summary['nodes'])}；每个查询内各方法使用同一节点，线程上限为 {summary['thread_caps']}。"
              f"已解析 CPU 型号：{', '.join(summary['accounting']['cpu_models']) or '日志尚不可用'}。"
              "CPU 存在 7285/7285H 等变体时，十点算术平均墙钟表示本批任务总体，不能作为严格相同硬件上的耗时均值；"
              "主要配对结论使用每查询内的时间比、胜率和这些时间比的几何平均（exp(mean(log(T_base/T_DD))))。"
              "Slurm 全部十项 COMPLETED/0:0 核验：" + ("已通过。" if summary["accounting"]["all_ten_completed_exit_zero"] else "尚未取得全部可解析终态记录；数值完成记录和资源退出核验分别列示。"), "",
              "GNU time 的进程最大 RSS 与 Slurm batch 的 MaxRSS 分别保存在 summary.json，不相互替代；采样和统计口径不同可产生明显差异。", "",
              "结论边界：新增七点覆盖七个相邻训练区间，样本仍是固定的一维参数族；单次计时无法给出硬件噪声置信区间。"
              "本表独立于旧三维计时批次，不直接合并均值或据此重估离线摊销。Global2 对照用于检查层数及近似存储水平，不能替代一种独立且充分调优的复用算法。", "",
             "完整逐点统计、校验计数、求解配置、节点和原始文件 SHA-256 均保存在 summary.json。", ""]
    lines += ["所有十个作业的 input_sha256_before.txt/after.txt 齐全且逐字节一致；跨作业相同源路径的记录哈希一致。"
              "已验证训练模型与原始报告均包含在记录中，二十份哈希记录文件本身也纳入本次来源清单。", ""]
    if summary["chart_generated"]:
        lines += ["![在线验证](online_validation.png)", ""]
    (out / "report.md").write_text("\n".join(lines))


def analyze(cloud, out, make_plot=True, allow_fixture=False):
    queries = [load_query(cloud / "results" / f"production_q{i}", i, allow_fixture) for i in range(1, 11)]
    frozen_source_hashes = consistent_input_hashes(queries)
    for q in queries[1:]:
        for key in ("source", "source_report", "matlab", "thread_cap", "mg_options", "gmres_options"):
            require(q["protocol"][key] == queries[0]["protocol"][key], f"Inconsistent frozen source/runtime/configuration: {key}")
    raw = [r for q in queries for r in q["rows"]]
    require(sum(int(r["repetition"]) > 0 for r in raw) == 58 and sum(int(r["repetition"]) == 0 for r in raw) == 9, "Expected 58 measured plus 9 warm-up solves")
    per_query, cohorts = statistics(queries)
    storage = {m: per_query[0]["methods"][m]["persistent_MiB"] for m in METHODS[:3]}
    require(all(near(q["methods"][m]["persistent_MiB"], storage[m]) for q in per_query for m in storage), "Frozen payload changed across queries")
    account = accounting(cloud, queries)
    for q, resource in zip(per_query, account["queries"]):
        q["cpu_model"] = resource["cpu_model"]
    paths = ([p for q in queries for p in q["paths"]] + list(cloud.glob("jobs_accounting*.txt"))
             + [Path(p) for p in account["resource_files"] + account["cpu_log_files"]] + [Path(__file__).resolve()])
    summary = dict(
        complete=True, numerical_validation_passed=True,
        validation=dict(all_ten_before_after_hash_files_present=True,
                        all_ten_before_after_records_byte_identical=True,
                        cross_query_recorded_source_hashes_consistent=True,
                        frozen_training_and_report_paths_recorded_for_every_query=True,
                        thread_cap_eight_for_every_query=True,
                        recorded_frozen_source_sha256=frozen_source_hashes),
        evidence_kind="synthetic_fixture_only" if allow_fixture else "new_frozen_model_solver_measurements",
        omega=48, N=5543003, query_count=10, measured_solves=58, warmup_solves=9,
        all_recorded_true_residuals_pass=True, max_true_residual_including_warmups=max(float(r["true_residual"]) for r in raw),
        original_query_count=3, new_query_count=7, aggregation="per-query repetition mean, then equal weights across queries",
        storage_payload_MiB=storage, per_query=per_query, cohorts=cohorts,
        nodes=sorted({q["protocol"]["node"] for q in queries}), thread_caps=sorted({q["protocol"]["thread_cap"] for q in queries}),
        protocol=queries[0]["protocol"], accounting=account,
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        analysis_python=sys.executable, chart_generated=False,
    )
    out.mkdir(parents=True, exist_ok=True)
    if make_plot:
        summary["chart_generated"] = plot(summary, out)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    report(summary, out)
    return summary


def progress(cloud, out):
    entries = []
    for i in range(1, 11):
        folder = cloud / "results" / f"production_q{i}"
        try:
            q = load_query(folder, i)
            entries.append(dict(query_index=i, status="numerical_records_valid", recorded_solves=sum(int(r["repetition"]) > 0 for r in q["rows"])))
        except (ValueError, KeyError, json.JSONDecodeError, OSError) as error:
            entries.append(dict(query_index=i, status="pending_or_invalid", detail=str(error)))
    value = dict(complete=False, mode="progress_only", valid_query_records=sum(r["status"] == "numerical_records_valid" for r in entries),
                 expected_queries=10, entries=entries, note="Progress only; no full-cohort conclusion or final report is generated.")
    out.mkdir(parents=True, exist_ok=True)
    (out / "progress.json").write_text(json.dumps(value, indent=2) + "\n")
    print(f"JCP_REVISION_3D_PROGRESS valid_query_records={value['valid_query_records']}/10; no full-cohort validation claimed")


def self_test():
    """Small temporary algebra fixture; never accepted as production evidence."""
    with tempfile.TemporaryDirectory(prefix="jcp_revision_3d_fixture_") as temporary:
        cloud = Path(temporary)
        for index in range(1, 11):
            folder = cloud / "results" / f"production_q{index}"
            folder.mkdir(parents=True)
            checks = [dict(method=m, depth=DEPTHS[m], post_rom_residual=.001,
                           archived_post_rom_residual=.001, persistent_MiB={"global3": 100., "dd3": 70., "global2": 65.}[m]) for m in METHODS[:3]]
            protocol = dict(synthetic_fixture=True, omega=48, N=5543003, query_index=index, mu=QUERIES[index-1],
                            all_queries=QUERIES, train_parameters=TRAIN, stratified_fractions=FRACTIONS,
                            repetitions=3 if index <= 3 else 1, baseline_repetitions=1, full_warmup_each_rom=index <= 3,
                            query_kind="original_repeated" if index <= 3 else "new_stratified", assembly_and_load_excluded=True,
                            per_solve_mg_setup_included=True, shift_beta=2, thread_cap=8, node="synthetic-node",
                            job_id=str(index), array_job_id="", source="synthetic_training.mat", source_report="synthetic_report.json", matlab="fixture",
                            gmres_options=dict(restart=40,maxit=40,tol=TOL,orthogonalization="cgs2_mex",fixed_linear_preconditioner=True),
                            mg_options=dict(mg_coarse_max_dofs=12000,mg_max_levels=6,mg_pre_sweeps=2,mg_post_sweeps=2,mg_jacobi_weight=.6,mg_cycles_per_apply=1),
                            frozen_replay_checks=checks)
            raw, histories = [], []
            for method in METHODS:
                reps = [1] if method == "shifted_mg" else range(0 if index <= 3 else 1, protocol["repetitions"]+1)
                for rep in reps:
                    total = (10+index+rep) * {"global3":1,"dd3":.7,"global2":1.05,"shifted_mg":3}[method]
                    n = {"global3":60,"dd3":40,"global2":50,"shifted_mg":100}[method]
                    pre = 1 if method == "shifted_mg" else .001
                    rh = [1] if method == "shifted_mg" else [1] + [.001]*(DEPTHS[method])
                    ch = [pre * (5e-7/pre)**(j/n) for j in range(n+1)]
                    raw.append(dict(query_index=index,parameter=QUERIES[index-1],method=method,repetition=rep,node="synthetic-node",
                                    total_s=total,rom_s=.1,setup_s=.2,completion_s=total-.4,completion_iterations=n,
                                    precompletion_residual=pre,true_residual=5e-7,converged=1,flag=0,
                                    persistent_MiB=next((v["persistent_MiB"] for v in checks if v["method"]==method),0)))
                    histories.append(dict(method=method,repetition=rep,rom_history=rh,completion_history=ch))
            complete = dict(protocol,all_converged=True,recorded_solve_count=sum(r["repetition"]>0 for r in raw),warmup_solve_count=sum(r["repetition"]==0 for r in raw))
            for name,value in [("protocol",protocol),("complete",complete),("histories",histories)]:
                (folder/f"{name}.json").write_text(json.dumps(value))
            hash_text = "".join(f"{hashlib.sha256(name.encode()).hexdigest()}  {name}\n"
                                for name in (protocol["source"], protocol["source_report"]))
            for name in ("input_sha256_before.txt", "input_sha256_after.txt"):
                (folder/name).write_text(hash_text)
            with (folder/"timings.csv").open("w",newline="") as stream:
                writer=csv.DictWriter(stream,fieldnames=list(raw[0]));writer.writeheader();writer.writerows(raw)
        summary=analyze(cloud,cloud/"analysis",make_plot=False,allow_fixture=True)
        require(summary["measured_solves"]==58 and summary["warmup_solves"]==9,"Fixture count error")
        for cohort in summary["cohorts"].values():
            require(near(cohort["DD_comparisons"]["global3"]["DD_time_reduction_percent"],30),"Fixture reduction error")
            require(near(cohort["DD_comparisons"]["global3"]["geometric_mean_paired_speedup"],1/.7),"Fixture geometric-mean speedup error")
        require(near(summary["cohorts"]["all_ten_equal_query_weight"]["methods"]["global3"]["mean_total_s"],16.8),"Repetitions improperly weighted as queries")
        require(summary["per_query"][3]["methods"]["dd3"]["CV_total"] is None,"Single run given a false CV")
        try:
            load_query(cloud/"results/production_q1",1)
            raise AssertionError("Synthetic fixture accepted as production")
        except ValueError as error:
            require("Synthetic" in str(error),"Unexpected fixture rejection")
        before=cloud/"results/production_q10/input_sha256_before.txt"
        after=cloud/"results/production_q10/input_sha256_after.txt"
        original_hash_record=before.read_text()
        after.write_text(original_hash_record.replace(original_hash_record[:64],"0"*64,1))
        try:
            load_query(cloud/"results/production_q10",10,allow_fixture=True)
            raise AssertionError("Changed before/after input hash accepted")
        except ValueError as error:
            require("before/after" in str(error),"Unexpected hash-preservation rejection")
        before.write_text(after.read_text())
        try:
            consistent_input_hashes([load_query(cloud/"results/production_q1",1,True),
                                     load_query(cloud/"results/production_q10",10,True)])
            raise AssertionError("Cross-query source drift accepted")
        except ValueError as error:
            require("cross-query" in str(error),"Unexpected source-consistency rejection")
        before.write_text(original_hash_record);after.write_text(original_hash_record)
        # A success marker is insufficient when its raw records disagree.
        bad_path=cloud/"results/production_q10/complete.json"
        bad=json.loads(bad_path.read_text());bad["recorded_solve_count"]=3
        bad_path.write_text(json.dumps(bad))
        try:
            load_query(cloud/"results/production_q10",10,allow_fixture=True)
            raise AssertionError("Inconsistent completion count accepted")
        except ValueError as error:
            require("count mismatch" in str(error),"Unexpected malformed-fixture rejection")
    print("JCP_REVISION_3D_SYNTHETIC_ALGEBRA_PASS: temporary fixture deleted; no real solver validation implied")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cloud",type=Path,default=DEFAULT_CLOUD)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--partial",action="store_true",help="Write progress.json only; never claim complete validation")
    parser.add_argument("--no-plot",action="store_true")
    parser.add_argument("--self-test",action="store_true",help="Temporary synthetic fixture: only checks summary algebra")
    args=parser.parse_args()
    if args.self_test:
        self_test();return
    out=args.output or args.cloud/"analysis"
    if args.partial:
        progress(args.cloud,out);return
    try:
        summary=analyze(args.cloud,out,not args.no_plot)
    except (ValueError,KeyError,json.JSONDecodeError,OSError) as error:
        raise SystemExit(f"JCP_REVISION_3D_INCOMPLETE_OR_INVALID: {error}; no complete report generated") from error
    print(f"JCP_REVISION_3D_NUMERICAL_AUDIT_PASS queries=10 measured=58 warmup=9 slurm_all_exit_zero={summary['accounting']['all_ten_completed_exit_zero']} output={out}")


if __name__=="__main__":
    main()
