#!/usr/bin/env python3
"""Publish the accepted omega=48 frozen-model validation from summary.json only.

Run with a Python environment containing matplotlib, e.g.:
    /opt/homebrew/bin/python3.12 scripts/write_jcp_3d_validation.py

No experiments are run and no historical timing files are read. The input must
describe all ten accepted production queries. Arithmetic means and paired
geometric speedups are recomputed from its per-query records before publication.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/jcp_revision_experiments_20260914/cloud/analysis/summary.json"
RECORD_DIR = ROOT / "outputs/jcp_revision_experiments_20260914/jcp_validation_publication"
METHODS = ("global3", "global2", "dd3", "shifted_mg")
BASELINES = ("global3", "global2", "shifted_mg")
LABELS = {
    "global3": "Global (3 layers)",
    "global2": "Global (2 layers)",
    "dd3": "DD (3 layers)",
    "shifted_mg": "Shifted MG-GMRES",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def equal(actual, expected, label):
    require(math.isfinite(actual) and math.isfinite(expected), f"Nonfinite {label}")
    require(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12),
            f"Inconsistent {label}: {actual!r} versus {expected!r}")


def validate(data):
    """Check the accepted production schema and independently recompute statistics."""
    for field in ("complete", "numerical_validation_passed",
                  "all_recorded_true_residuals_pass"):
        require(data.get(field) is True, f"Production validation missing: {field}")
    require(data.get("evidence_kind") == "new_frozen_model_solver_measurements",
            "Only the real frozen-model production measurements can be published")
    require(data.get("omega") == 48 and data.get("N") == 5543003,
            "Unexpected frequency or system size")
    require(data.get("query_count") == 10 and data.get("original_query_count") == 3
            and data.get("new_query_count") == 7, "All ten queries are required")
    require(data.get("measured_solves") == 58 and data.get("warmup_solves") == 9,
            "Expected 58 measured solves plus nine excluded warmups")
    require(data.get("thread_caps") == [8], "Expected eight-thread query jobs")
    for field in ("all_ten_before_after_hash_files_present",
                  "all_ten_before_after_records_byte_identical",
                  "cross_query_recorded_source_hashes_consistent",
                  "frozen_training_and_report_paths_recorded_for_every_query",
                  "thread_cap_eight_for_every_query"):
        require(data["validation"].get(field) is True, f"Input acceptance missing: {field}")
    require(data["accounting"].get("all_ten_completed_exit_zero") is True,
            "Ten successful production job exits are required")
    require(0 <= data["max_true_residual_including_warmups"] <= 1e-6,
            "A recorded true residual exceeded tolerance")
    protocol = data["protocol"]
    require(protocol["assembly_and_load_excluded"] is True and
            protocol["per_solve_mg_setup_included"] is True,
            "Unexpected online timing scope")
    equal(protocol["gmres_options"]["tol"], 1e-6, "GMRES tolerance")
    queries = sorted(data["per_query"], key=lambda q: q["query_index"])
    require([q["query_index"] for q in queries] == list(range(1, 11)),
            "Query indices must be exactly 1 through 10")
    cvs = []
    count = 0
    for query in queries:
        original = query["query_index"] <= 3
        require(query["query_kind"] == ("original_repeated" if original else "new_stratified"),
                "Unexpected query cohort")
        equal(query["parameter"], protocol["all_queries"][query["query_index"] - 1],
              "query parameter")
        require(query.get("node") and query.get("cpu_model"), "Missing query hardware")
        for method in METHODS:
            values = query["methods"][method]
            repetitions = 3 if original and method != "shifted_mg" else 1
            require(values["measured_repetitions"] == repetitions, "Unexpected repeats")
            count += repetitions
            mean = values["mean_total_s"]
            require(mean > 0 and values["min_total_s"] <= mean <= values["max_total_s"],
                    "Invalid timing mean or range")
            require(0 <= values["max_true_residual"] <= 1e-6, "Query failed residual target")
            require(values["max_true_residual"] <= data["max_true_residual_including_warmups"],
                    "Inconsistent maximum residual")
            if repetitions == 3:
                sd = values["sample_sd_total_s"]
                require(sd is not None and sd >= 0, "Missing repeated-run standard deviation")
                equal(sd / mean, values["CV_total"], "sample coefficient of variation")
                cvs.append(100 * sd / mean)
            else:
                require(values["sample_sd_total_s"] is None and values["CV_total"] is None,
                        "Single measurements must not have error bars")
                equal(mean, values["min_total_s"], "single timing minimum")
                equal(mean, values["max_total_s"], "single timing maximum")
            if method != "shifted_mg":
                equal(values["persistent_MiB"], data["storage_payload_MiB"][method],
                      "frozen payload")
        for baseline in BASELINES:
            equal(query["methods"][baseline]["mean_total_s"] /
                  query["methods"]["dd3"]["mean_total_s"],
                  query["DD_paired_comparisons"][baseline]["baseline_over_DD_mean_time"],
                  "per-query paired speedup")
    require(count == 58 and len(cvs) == 9, "Unexpected formal timing/error-bar count")

    cohorts = {}
    selections = {
        "original_three_repeated": queries[:3],
        "new_seven_single": queries[3:],
        "all_ten_equal_query_weight": queries,
    }
    for name, selected in selections.items():
        stored = data["cohorts"][name]
        require(stored["query_count"] == len(selected), "Inconsistent cohort count")
        methods = {}
        for method in METHODS:
            methods[method] = {}
            for field in ("mean_total_s", "mean_completion_iterations"):
                value = statistics.mean(q["methods"][method][field] for q in selected)
                equal(value, stored["methods"][method][field], f"{name} {method} {field}")
                methods[method][field] = value
        comparisons = {}
        for baseline in BASELINES:
            paired = [q["methods"][baseline]["mean_total_s"] /
                      q["methods"]["dd3"]["mean_total_s"] for q in selected]
            geometric = math.exp(statistics.mean(math.log(ratio) for ratio in paired))
            faster = sum(ratio > 1 for ratio in paired)
            fewer = sum(q["methods"]["dd3"]["mean_completion_iterations"] <
                        q["methods"][baseline]["mean_completion_iterations"] for q in selected)
            reference = stored["DD_comparisons"][baseline]
            equal(geometric, reference["geometric_mean_paired_speedup"], "paired geometric mean")
            require(faster == reference["DD_faster_query_count"] and
                    fewer == reference["DD_fewer_completion_query_count"],
                    "Inconsistent paired win counts")
            comparisons[baseline] = {"geometric_mean_paired_speedup": geometric,
                                     "DD_faster_query_count": faster,
                                     "DD_fewer_completion_query_count": fewer}
        cohorts[name] = {"query_count": len(selected), "methods": methods,
                         "DD_comparisons": comparisons}
    return {"passed": True, "cohorts_recomputed_from_per_query": cohorts,
            "formal_solve_count_checked": count, "ROM_errorbar_count": len(cvs),
            "repeated_ROM_CV_percent_range": [min(cvs), max(cvs)],
            "max_true_residual_including_warmups": data["max_true_residual_including_warmups"],
            "payload_MiB": data["storage_payload_MiB"],
            "DD_payload_saving_percent": {
                baseline: 100 * (1 - data["storage_payload_MiB"]["dd3"] /
                                 data["storage_payload_MiB"][baseline])
                for baseline in ("global3", "global2")},
            "checks_scope": "Internal acceptance flags and per-query algebra in summary.json; "
                            "this publication generator does not reread raw solver files."}


def write_table(data, target, source_hash):
    lines = [
        "% Generated by scripts/write_jcp_3d_validation.py; do not edit numerical cells.",
        f"% Numeric input SHA256: {source_hash}",
        r"\begin{table}[!htbp]\centering\normalsize",
        r"\setlength{\belowcaptionskip}{6pt}",
        r"\caption{Waveguide at $\omega=48$: online cost over the ten test parameters, "
        r"by query group. Each group weights its queries equally, and the times "
        r"exclude matrix assembly and model loading.}",
        r"\label{tab:3d-validation}\par\medskip",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrrrr@{}}\toprule",
        r"& \multicolumn{2}{c}{Three common queries} & \multicolumn{2}{c}{Seven added queries} "
        r"& \multicolumn{2}{c}{All ten queries} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){6-7}",
        r"Method & Online (s) & SL steps & Online (s) & SL steps & Online (s) & SL steps \\",
        r"\midrule",
    ]
    for method in METHODS:
        cells = [LABELS[method]]
        for group in ("original_three_repeated", "new_seven_single", "all_ten_equal_query_weight"):
            values = data["cohorts"][group]["methods"][method]
            cells.extend([f"{values['mean_total_s']:.2f}",
                          f"{values['mean_completion_iterations']:.1f}"])
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule\end{tabular*}", r"\end{table}", ""])
    target.write_text("\n".join(lines))


def write_figure(data, target_stem, source_hash):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MultipleLocator

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 10,
        "axes.titlesize": 10, "legend.fontsize": 8.5, "axes.linewidth": 0.65,
        "xtick.direction": "in", "ytick.direction": "in",
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
    })
    styles = {
        "global3": dict(color="#2365A0", marker="o", linestyle="-"),
        "global2": dict(color="#C27816", marker="s", linestyle="--"),
        "dd3": dict(color="#238562", marker="^", linestyle="-"),
        "shifted_mg": dict(color="#414141", marker="D", linestyle=":"),
    }
    queries = sorted(data["per_query"], key=lambda q: q["parameter"])
    xs = [q["parameter"] for q in queries]
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.3), sharex=True,
                             gridspec_kw={"height_ratios": [1.2, 1]})
    fig.subplots_adjust(left=0.1, right=0.985, top=0.915, bottom=0.095, hspace=0.31)
    legend = [Line2D([], [], **styles[m], markersize=4, linewidth=1,
                     label=LABELS[m]) for m in METHODS]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.55, 1),
               ncol=4, frameon=False, columnspacing=1.25, handlelength=2.2)
    for method in METHODS:
        style = styles[method]
        ys = [q["methods"][method]["mean_total_s"] for q in queries]
        axes[0].plot(xs, ys, color=style["color"], linestyle=style["linestyle"], linewidth=0.9)
        for q, x, y in zip(queries, xs, ys):
            original = q["query_kind"] == "original_repeated"
            axes[0].plot(x, y, marker=style["marker"], markersize=4.7,
                         markerfacecolor=style["color"] if original else "white",
                         markeredgecolor=style["color"], markeredgewidth=0.9)
            sd = q["methods"][method]["sample_sd_total_s"]
            if sd is not None:
                axes[0].errorbar(x, y, yerr=sd, fmt="none", ecolor=style["color"],
                                 elinewidth=0.8, capsize=2.4, capthick=0.8)
    axes[0].set_title("(a) Total online time", loc="left", pad=8)
    axes[0].set_ylabel("Time (s)")
    axes[0].set_ylim(bottom=0)
    axes[0].yaxis.set_major_locator(MultipleLocator(250))
    for baseline in ("global3", "global2"):
        style = styles[baseline]
        ratios = [q["methods"][baseline]["mean_total_s"] /
                  q["methods"]["dd3"]["mean_total_s"] for q in queries]
        axes[1].plot(xs, ratios, color=style["color"], linestyle=style["linestyle"], linewidth=0.9)
        for q, x, ratio in zip(queries, xs, ratios):
            axes[1].plot(x, ratio, marker=style["marker"], markersize=4.7,
                         markerfacecolor=style["color"] if q["query_kind"] == "original_repeated" else "white",
                         markeredgecolor=style["color"], markeredgewidth=0.9)
    axes[1].axhline(1, color="0.55", linestyle=(0, (3, 3)), linewidth=0.65, zorder=0)
    axes[1].set_title("(b) Paired Global/DD online-time ratios", loc="left", pad=8)
    axes[1].set_ylabel(r"$t_{\mathrm{Global}}/t_{\mathrm{DD3}}$")
    axes[1].set_xlabel(r"Query parameter $\mu$")
    axes[1].set_ylim(bottom=0.95)
    axes[1].yaxis.set_major_locator(MultipleLocator(0.25))
    axes[1].xaxis.set_major_locator(MultipleLocator(0.05))
    axes[1].set_xlim(0.2, 0.59)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="0.9", linewidth=0.5)
        axis.set_axisbelow(True)
    subject = f"Numeric input summary.json SHA256 {source_hash}"
    fig.savefig(target_stem.with_suffix(".pdf"), metadata={
        "Title": "Frozen-model waveguide validation at omega=48",
        "Subject": subject, "Creator": "scripts/write_jcp_3d_validation.py",
        "CreationDate": None, "ModDate": None,
    })
    fig.savefig(target_stem.with_suffix(".png"), metadata={"Description": subject})
    plt.close(fig)
    return matplotlib.__version__


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Validate without generating artifacts")
    parser.add_argument("--table-only", action="store_true", help="Regenerate only the table, preserving figure edits")
    args = parser.parse_args()
    raw = SOURCE.read_bytes()
    data = json.loads(raw)
    source_hash = hashlib.sha256(raw).hexdigest()
    checked = validate(data)
    if args.check_only:
        print(json.dumps({"source_sha256": source_hash, **checked}, indent=2))
        return
    RECORD_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(RECORD_DIR / "mplconfig"))
    os.environ.setdefault("XDG_CACHE_HOME", str(RECORD_DIR / "cache"))
    figure_dir = ROOT / "JCP/figures/waveguide_validation"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table = ROOT / "JCP/jcp_waveguide_validation.tex"
    stem = figure_dir / "omega48_validation"
    write_table(data, table, source_hash)
    if args.table_only:
        print("JCP_VALIDATION_TABLE_PASS")
        return
    matplotlib_version = write_figure(data, stem, source_hash)
    caption = (
        r"Frozen-model waveguide validation at $\omega=48$. (a) Total online time. "
        r"Filled markers identify the three original query locations; open markers identify "
        r"the seven additional queries. Error bars show one sample standard deviation about "
        r"the mean of three measured runs only for ROM methods at the original queries. "
        r"Shifted MG at all ten queries and every method at the additional queries are timed once. "
        r"(b) Per-query Global-to-DD time ratios relative to three-layer DD, using the means in (a); "
        r"values above one favor DD. All methods at a query run on the same node with eight "
        r"threads; nodes vary across queries. Lines connect query measurements to guide the eye."
    )
    artifacts = [table, stem.with_suffix(".pdf"), stem.with_suffix(".png")]
    record = {
        "numeric_source": str(SOURCE.relative_to(ROOT)), "source_sha256": source_hash,
        "generator": str(Path(__file__).resolve().relative_to(ROOT)),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "matplotlib_version": matplotlib_version,
        "formulas": {
            "cohort_mean": "sum(per_query_repeat_mean) / number_of_queries",
            "paired_geometric_speedup": "exp(mean(log(t_method(query) / t_DD3(query))))",
            "CV_percent": "100 * sample_sd_total_s / mean_total_s",
            "payload_saving_percent": "100 * (1 - payload_DD3 / payload_baseline)",
            "errorbars": "mean_total_s +/- sample_sd_total_s; original three ROM queries only",
        },
        "numeric_validation": checked,
        "artifact_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in artifacts},
        "suggested_figure_label": "fig:3d-validation",
        "suggested_figure_caption_latex": caption,
        "visual_encoding": {"filled_markers": "original query locations (MG remains single)",
                            "open_markers": "seven additional query locations",
                            "errorbars": "9 original-query ROM mean +/- sample SD bars",
                            "lower_panel": "20 paired Global/DD ratios; no ratio errorbars"},
    }
    record_path = RECORD_DIR / "validation.json"
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"numeric_validation_passed": True, "source_sha256": source_hash,
                      "artifacts": [str(p.relative_to(ROOT)) for p in artifacts],
                      "validation_record": str(record_path.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
