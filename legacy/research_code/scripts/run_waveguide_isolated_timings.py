#!/usr/bin/env python3
"""Measure each distinct frozen waveguide model in a fresh MATLAB process.

Training data are reused unchanged. Run serially, with no active training or
other MATLAB benchmark. The output subdirectory is an independent timing
cohort; earlier shared-process measurements remain available for diagnosis.
"""
from pathlib import Path
import csv
import hashlib
import json
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/waveguide_threshold_tradeoff_20260908"
OUT = SOURCE / "isolated_timing"
MATLAB = shutil.which("matlab")
LABELS = ("quarter", "half", "reference", "double", "quadruple")


def mquote(value):
    return "'" + str(value).replace("'", "''") + "'"


def run_matlab(expression, log_name):
    assert MATLAB, "MATLAB executable is required"
    with (OUT / log_name).open("w") as log:
        subprocess.run(
            [MATLAB, "-singleCompThread", "-batch",
             f"addpath({mquote(ROOT / 'code')}); {expression}"],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True,
        )


def validate_csv(path):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 6, (path, "expected three repetitions of two queries")
    assert {(float(r["parameter"]), int(r["repetition"])) for r in rows} == {
        (mu, rep) for mu in (0.317, 0.483) for rep in (1, 2, 3)
    }
    assert all(0 <= float(r["true_residual"]) <= 1e-6 for r in rows)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    for label in LABELS:
        for method in ("global", "dd"):
            path = OUT / f"{label}_{method}.mat"
            target = SOURCE / path.name
            assert target.is_file(), target
            if not path.exists():
                path.symlink_to(target)
            assert path.resolve() == target.resolve()

    # The reversed-order control started with Global in a fresh process and
    # collected exactly the prescribed warm-up plus three repetitions.
    control = SOURCE / "order_check/reference_global_lazy_timings.csv"
    canonical = OUT / "reference_global_lazy_timings.csv"
    reference_source = "fresh isolated reference_global job"
    if not canonical.exists() and control.exists():
        assert digest(SOURCE / "reference_global.mat") == digest(
            SOURCE / "order_check/reference_global.mat")
        validate_csv(control)
        shutil.copy2(control, canonical)
        reference_source = str(control)
    if canonical.exists():
        validate_csv(canonical)
        if (OUT / "protocol.json").exists():
            reference_source = json.loads((OUT / "protocol.json").read_text())[
                "reference_global_timing_source"]
    (OUT / "protocol.json").write_text(json.dumps({
        "training_directory": str(SOURCE),
        "execution": "one fresh MATLAB process per distinct deployed model; serial",
        "warmups_per_query": 1,
        "recorded_repetitions_per_query": 3,
        "parameters": [0.317, 0.483],
        "reference_global_timing_source": reference_source,
        "global_reuse": "complete model, RHS, geometry and online configuration equality required",
        "excluded_timing_cohort": "parent-directory shared-process measurements",
    }, indent=2))

    cases = [("reference_global", "baseline"), ("reference_global", "lazy"), ("quarter_dd", "lazy"),
             ("half_dd", "lazy"), ("reference_dd", "lazy"),
             ("double_global", "lazy"), ("double_dd", "lazy"),
             ("quadruple_global", "lazy"), ("quadruple_dd", "lazy")]
    for stem, policy in cases:
        output = OUT / f"{stem}_{policy}_timings.csv"
        if output.exists():
            validate_csv(output)
            print(f"Reusing complete isolated timing: {stem} {policy}", flush=True)
            continue
        print(f"ISOLATED TIMING START {stem} {policy}", flush=True)
        expression = (f"run_jsc_3d_deployment({mquote(OUT / (stem + '.mat'))},"
                      f"{mquote(policy)},[],3,false);")
        run_matlab(expression, f"{stem}_{policy}.log")
        validate_csv(output)
        print(f"ISOLATED TIMING DONE {stem} {policy}", flush=True)

    # Explicitly certify the Global rank plateau before sharing any samples.
    reference = mquote(OUT / "reference_global.mat")
    expression = f"a=load({reference},'model','rhs','problem_cfg','cfg'); "
    for label in ("quarter", "half"):
        expression += (
            f"b=load({mquote(OUT / (label + '_global.mat'))},'model','rhs','problem_cfg','cfg'); "
            "assert(isequaln(a.model,b.model) && isequaln(a.rhs,b.rhs) && "
            "isequaln(a.problem_cfg,b.problem_cfg)); "
            "ignore={'output_tag','layer_tail_tolerances'}; "
            "assert(isequaln(rmfield(a.cfg,ignore),rmfield(b.cfg,ignore))); "
            f"copyfile({mquote(canonical)},{mquote(OUT / (label + '_global_lazy_timings.csv'))}); "
            f"fprintf('Complete Global model equivalence verified: {label} = reference\\n'); "
        )
    expression += f"run_waveguide_threshold_tradeoff('plot',[],{mquote(OUT)});"
    run_matlab(expression, "collect_and_plot.log")
    print(f"Complete: {OUT / 'comparison.csv'}", flush=True)


if __name__ == "__main__":
    main()
