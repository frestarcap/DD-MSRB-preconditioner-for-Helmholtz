#!/usr/bin/env python3
"""Validate and summarize the frozen JCP shifted-ILU comparison."""
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("JCP_SHIFTED_OUT", ROOT / "outputs/jcp_shifted_20260914"))
os.environ.setdefault("MPLCONFIGDIR", str(OUT / "matplotlib_cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NAMES = {"shifted": "Shifted ILU-GMRES", "direct": "Sparse direct",
         "global": "Global TA-ROM", "dd": "DD-TA-ROM", "nr": "NR-ROM",
         "fixed_pod": "Fixed solution POD"}
ORDER = ["shifted", "direct", "global", "dd", "nr", "fixed_pod"]
COLORS = {"shifted": "#374151", "direct": "#a855f7", "global": "#c0392b",
          "dd": "#007c91", "nr": "#3972bc", "fixed_pod": "#d88914"}


def read(name):
    with (OUT / name).open() as f:
        return list(csv.DictReader(f))


def run():
    rows = read("paper_timings.csv")
    checks = read("paper_prefix_checks.csv")
    groups = defaultdict(list)
    paired = defaultdict(list)
    for r in rows:
        assert int(r["flag"]) == 0
        assert float(r["true_residual"]) <= float(r["tolerance"])
        groups[r["problem"], r["method"]].append(r)
        paired[r["problem"], r["method"], int(r["query"])].append(r)
    assert len(rows) == 23 * 6 * 5 + 11 * 4 * 5
    assert len(checks) == 23 * 4 + 11 * 2
    assert max(float(r["prefix_field_difference"]) for r in checks) < 1e-9
    for key, rs in paired.items():
        assert sorted(int(r["repeat"]) for r in rs) == list(range(1, 6)), key
        assert len({r["completion_iterations"] for r in rs}) == 1, key
    summary = []
    for problem in ["halfspace", "ecc"]:
        for method in ORDER:
            rs = groups.get((problem, method))
            if not rs:
                continue
            item = {"problem": problem, "method": method,
                    "queries": len(rs) // 5, "repeats": 5}
            for field in ["total_s", "ROM_s", "setup_s", "solve_s", "completion_iterations"]:
                item["mean_" + field] = mean(float(r[field]) for r in rs)
            item["median_total_s"] = median(float(r["total_s"]) for r in rs)
            item["max_true_residual"] = max(float(r["true_residual"]) for r in rs)
            item["max_relative_error"] = max(float(r["relative_error"]) for r in rs)
            item["zero_completion_queries"] = sum(
                int(qrs[0]["completion_iterations"]) == 0
                for (p, m, _), qrs in paired.items() if p == problem and m == method)
            for base in ["shifted", "direct"]:
                baseline = mean(float(r["total_s"]) for r in groups[problem, base])
                item["speedup_vs_" + base] = baseline / item["mean_total_s"]
            summary.append(item)
    with (OUT / "summary.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    (OUT / "validation.json").write_text(json.dumps({
        "passed": True, "measured_solves": len(rows), "prefix_checks": len(checks),
        "max_prefix_field_difference": max(float(r["prefix_field_difference"]) for r in checks),
        "five_repetitions_per_query": True, "iteration_counts_repeat_exactly": True,
    }, indent=2) + "\n")
    make_plots(paired, summary)
    make_report(summary, rows, checks)
    print(json.dumps(summary, indent=2))


def make_plots(paired, summary):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.bbox": "tight"})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for ax, problem, title in zip(axes, ["halfspace", "ecc"],
                                 ["Experiment 1: half-space", "Experiment 2: elliptic scattering"]):
        for method in ORDER:
            vals = [(k[2], rs) for k, rs in paired.items() if k[:2] == (problem, method)]
            if not vals:
                continue
            vals.sort()
            x = [float(rs[0]["parameter"]) for _, rs in vals]
            y = [mean(float(r["total_s"]) for r in rs) for _, rs in vals]
            err = [stdev(float(r["total_s"]) for r in rs) for _, rs in vals]
            ax.errorbar(x, y, yerr=err, fmt="o-", ms=3, lw=1.3, capsize=2,
                        color=COLORS[method], label=NAMES[method])
        ax.set_yscale("log")
        ax.set_xlabel("Right-hand wavespeed" if problem == "halfspace" else "Ellipse aspect ratio")
        ax.set_ylabel("Complete online solve time (s)")
        ax.set_title(title)
        ax.grid(alpha=.2, which="both")
        ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(OUT / f"online_times.{ext}", dpi=200)
    plt.close(fig)

    histories = read("paper_histories.csv")
    plain = read("paper_plain_gmres.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for ax, problem, query, tol in zip(axes, ["halfspace", "ecc"], [10, 1], [1e-12, 1e-10]):
        for method in ORDER:
            hs = [r for r in histories if r["problem"] == problem and
                  r["method"] == method and int(r["query"]) == query]
            if not hs:
                continue
            x = [int(r["step"]) for r in hs]
            y = [float(r["relative_residual"]) for r in hs]
            depth = int(hs[0]["ROM_depth"])
            ax.semilogy(x, y, "-", lw=1.2, color=COLORS[method], label=NAMES[method])
            ax.scatter(x[:depth + 1], y[:depth + 1], s=20, color=COLORS[method])
            ax.scatter(x[-1], y[-1], s=25, marker="s", color=COLORS[method])
        hs = [r for r in plain if r["problem"] == problem]
        ax.semilogy([int(r["step"]) for r in hs],
                    [float(r["relative_residual"]) for r in hs], ":", color="#888888", label="Plain GMRES (100 steps)")
        ax.axhline(tol, color="#666666", ls="--", lw=.8)
        ax.set_xlabel("ROM steps + GMRES iterations")
        ax.set_ylabel(r"$\|b-Au\|_2/\|b\|_2$")
        ax.set_title("Half-space: c = 0.8622" if problem == "halfspace" else "Scattering: ratio = 0.9072")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(OUT / f"representative_residuals.{ext}", dpi=200)
    plt.close(fig)


def make_report(summary, rows, checks):
    lines = ["# JCP shifted-Laplacian experiments — 2026-09-14", "",
        "Frozen original training sets, test points and Global/DD bases. "
        "All new solves use the frequency-domain Helmholtz–PML matrices. "
        "The NR control is reconstructed through the existing equal-rank trainer; "
        "fixed POD uses the saved seven-dimensional Galerkin space.", "",
        "Shifted-ILU: beta 0.2, ILUTP drop tolerance 1e-3, restart 40, at most 20 cycles, "
        "two-pass orthogonalization. Half-space target 1e-12; scattering target 1e-10. "
        "MATLAB R2025a, Apple M4 Max, one computational thread, serial queries, "
        "one warm-up plus five measured repeats with rotating method order.", "",
        "Online time includes ROM work, reduced assembly/solve, orthogonalization, "
        "shifted-matrix construction, ILU setup when completion is needed, completion "
        "and final true-residual evaluation. It excludes common full-order assembly, "
        "model/data loading and offline training. Each query builds its own required "
        "preconditioner; it is not shared across methods or repetitions.", "",
        "| Problem | Method | Mean online (s) | Median (s) | Mean completion iterations | "
        "Speedup vs shifted | Speedup vs direct | Max true residual | Max field error |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s in summary:
        lines.append(f"| {s['problem']} | {NAMES[s['method']]} | {s['mean_total_s']:.5f} | "
                     f"{s['median_total_s']:.5f} | {s['mean_completion_iterations']:.2f} | "
                     f"{s['speedup_vs_shifted']:.2f} | {s['speedup_vs_direct']:.2f} | "
                     f"{s['max_true_residual']:.2e} | {s['max_relative_error']:.2e} |")
    lines += ["", "Timing decomposition:", "",
        "| Problem | Method | ROM (s) | Shifted setup (s) | Solve (s) | Queries needing no completion |",
        "|---|---|---:|---:|---:|---:|"]
    for s in summary:
        zero = f"{s['zero_completion_queries']}/{s['queries']}" if s['method'] not in ['shifted', 'direct'] else "—"
        lines.append(f"| {s['problem']} | {NAMES[s['method']]} | {s['mean_ROM_s']:.5f} | "
                     f"{s['mean_setup_s']:.5f} | {s['mean_solve_s']:.5f} | {zero} |")
    lines += ["", f"Validation: all {len(rows)} measured solves meet their true-residual targets; "
        f"{len(checks)} prefix comparisons preserve the saved fields to relative difference below "
        f"{max(float(r['prefix_field_difference']) for r in checks):.2e}. "
        "Completion iteration counts are identical across the five repeats for every method/query.", "",
        "![Online times](online_times.png)", "",
        "Error bars show the sample standard deviation of repeated timings at each query. "
        "The timing figures compare methods in this batch only; historical WaveHoltz "
        "timings used a different execution arrangement and should not be treated as a matched speedup.", "",
        "![Representative residuals](representative_residuals.png)", "",
        "ROM residuals and terminal squares are evaluated in the original frequency-domain system. "
        "Intermediate GMRES values are Arnoldi residual estimates; the solver recomputes the "
        "true residual at acceptance and cycle boundaries. Iteration counts mix stages of different "
        "cost and should be read together with the timing plot.", "",
        "The initial ECC smoke used 1e-8; its field error reached approximately 1e-6, "
        "so the final batch uniformly tightened ECC to 1e-10 without changing bases. "
        "No manuscript text, published figures or historical experiment files were overwritten. "
        "These are online comparisons, not an offline-amortization claim.", ""]
    (OUT / "results.md").write_text("\n".join(lines))


if __name__ == "__main__":
    run()
