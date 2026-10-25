#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
MATLAB_BIN="${MATLAB_BIN:-matlab}"
phase="${1:-all}"
case "$phase" in
  train)
    "$MATLAB_BIN" -singleCompThread -batch "addpath('code'); run_ecc_shifted_tuning('pilot'); run_ecc_shifted_tuning('refine');"
    ;;
  screen)
    "$MATLAB_BIN" -singleCompThread -batch "addpath('code'); benchmark_ecc_shifted_tuning('screen'); benchmark_ecc_shifted_tuning('refine');"
    ;;
  confirm)
    "$MATLAB_BIN" -singleCompThread -batch "addpath('code'); benchmark_ecc_shifted_tuning('confirm',[96,3,1e-6,3]); benchmark_ecc_shifted_tuning('validate',[96,3,1e-6,3]);"
    ;;
  verify)
    "$MATLAB_BIN" -singleCompThread -batch "addpath('code'); test_block_reduced_assembly; test_halfspace_refined_partition; verify_ecc_shifted_tuning;"
    ;;
  accuracy)
    "$MATLAB_BIN" -singleCompThread -batch "addpath('code'); benchmark_ecc_shifted_tuning('accuracy',[96,3,1e-6,3]);"
    ;;
  report)
    "${PYTHON_BIN:-python3}" scripts/summarize_ecc_shifted_tuning.py
    ;;
  all)
    for step in train screen confirm verify accuracy report; do bash "$0" "$step"; done
    ;;
  *)
    echo "Usage: $0 [all|train|screen|confirm|verify|accuracy|report]" >&2
    exit 2
    ;;
esac
