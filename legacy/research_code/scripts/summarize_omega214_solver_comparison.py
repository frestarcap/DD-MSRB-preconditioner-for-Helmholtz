#!/usr/bin/env python3
"""Summarize the matched omega-214 zero-start solver comparison (stdlib only)."""
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "outputs/omega214_solver_comparison_20260910"
    samples = []
    for stage in ("algebraic", "waveholtz"):
        with (out / f"{stage}_samples.csv").open() as stream:
            samples.extend(row for row in csv.DictReader(stream)
                           if int(row["repetition"]) > 0)
    assert len(samples) == 21, "Expected 18 repeated algebraic and 3 WaveHoltz samples."
    assert all(float(row["true_residual"]) <= 1e-10 for row in samples)
    assert all(int(row["flag"]) == 0 for row in samples)
    field_errors = [float(row["relative_field_difference_from_direct"])
                    for row in samples if row["method"] != "direct"]
    assert all(math.isfinite(value) and value < 1e-8 for value in field_errors)
    groups = {}
    for row in samples:
        groups.setdefault((float(row["parameter"]), row["method"]), []).append(row)
    summary = []
    for (mu, method), group in sorted(groups.items()):
        times = [float(row["total_s"]) for row in group]
        summary.append({
            "parameter": mu, "method": method, "repetitions": len(group),
            "mean_total_s": mean(times),
            "std_total_s": stdev(times) if len(times) > 1 else None,
            "mean_setup_s": mean(float(row["setup_s"]) for row in group),
            "mean_solve_s": mean(float(row["solve_s"]) for row in group),
            "mean_recovery_s": mean(float(row["recovery_s"]) for row in group),
            "mean_iterations": mean(float(row["iterations"]) for row in group),
            "max_true_residual": max(float(row["true_residual"]) for row in group),
            "max_relative_field_difference_from_direct": (
                max(float(row["relative_field_difference_from_direct"]) for row in group)
                if method != "direct" else 0),
        })
    with (out / "summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    by_method = {}
    for method in ("waveholtz", "shifted_laplacian", "direct"):
        group = [row for row in summary if row["method"] == method]
        assert len(group) == 3
        by_method[method] = {
            "mean_total_s": mean(row["mean_total_s"] for row in group),
            "mean_iterations": mean(row["mean_iterations"] for row in group),
            "max_true_residual": max(row["max_true_residual"] for row in group),
        }
    wh = by_method["waveholtz"]["mean_total_s"]
    sl = by_method["shifted_laplacian"]["mean_total_s"]
    direct = by_method["direct"]["mean_total_s"]
    archived_file = root / "outputs/reviewer_20260909/high_frequency_waveholtz/global_scale4_queries.csv"
    with archived_file.open() as stream:
        archived = {float(row["parameter"]): row for row in csv.DictReader(stream)}
    iterations_match = all(
        int(row["iterations"]) == int(archived[float(row["parameter"])]["baseline_iterations"])
        for row in samples if row["method"] == "waveholtz")
    assert iterations_match, "Fresh WaveHoltz iteration counts differ from the paper cohort."
    result = {"per_method": by_method, "waveholtz_over_shifted_time": wh/sl,
              "shifted_over_direct_time": sl/direct,
              "waveholtz_over_direct_time": wh/direct,
              "max_relative_field_difference_from_direct": max(field_errors),
              "waveholtz_iteration_counts_match_paper": iterations_match,
              "recorded_samples": len(samples), "all_true_residuals_pass": True}
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
