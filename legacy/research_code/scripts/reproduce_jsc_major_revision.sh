#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$project_root"
matlab_bin="${MATLAB_BIN:-matlab}"
phase="${1:-training}"
time_args=(-l)
if [[ "$(uname -s)" != Darwin ]]; then time_args=(-v); fi
mkdir -p outputs/revision_20260907/deployment
case "$phase" in
  training)
    "$matlab_bin" -batch "addpath('code'); run_jsc_lens_tradeoff;"
    "$matlab_bin" -singleCompThread -batch "addpath('code'); run_jsc_pc_study('screen'); run_jsc_pc_study('verified'); run_jsc_halfspace_study('train');"
    ;;
  timing)
    # Run after exploratory jobs have finished. No parallel background jobs.
    "$matlab_bin" -singleCompThread -batch "addpath('code'); for c={'edge','balanced','tight','simple_tight','middle','simple_middle'}, prepare_jsc_3d_deployment(c{1}); end"
    for candidate in edge balanced tight simple_tight middle simple_middle; do
      for method in global dd; do
        stem="outputs/revision_20260907/deployment/lens_${candidate}_${method}"
        /usr/bin/time "${time_args[@]}" "$matlab_bin" -singleCompThread -batch \
          "addpath('code'); run_jsc_3d_deployment('${stem}.mat','lazy');" \
          > "${stem}_lazy_process.log" 2>&1
      done
    done
    for method in global dd; do
      stem="outputs/revision_20260907/deployment/lens_tight_${method}"
      /usr/bin/time "${time_args[@]}" "$matlab_bin" -singleCompThread -batch \
        "addpath('code'); run_jsc_3d_deployment('${stem}.mat','eager');" \
        > "${stem}_eager_process.log" 2>&1
    done
    "$matlab_bin" -singleCompThread -batch "addpath('code'); run_jsc_halfspace_study('timing'); run_jsc_pc_study('timing');" \
      > outputs/revision_20260907/timings_2d.log 2>&1
    ;;
  test)
    "$matlab_bin" -singleCompThread -batch "addpath('code'); test_jsc_revision_helpers; test_rom_verified_snapshot_step;"
    ;;
  finalize)
    "$matlab_bin" -singleCompThread -batch "addpath('code'); run_jsc_final_checks;"
    "${PYTHON_BIN:-python3}" scripts/write_jsc_revision_evidence.py
    ;;
  *) echo "Usage: $0 [training|timing|finalize|test]" >&2; exit 2 ;;
esac
