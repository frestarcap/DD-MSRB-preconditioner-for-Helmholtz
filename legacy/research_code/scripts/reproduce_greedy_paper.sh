#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CASE_NAME="${1:-all}"
MATLAB_BIN="${MATLAB_BIN:-matlab}"
cd "${ROOT_DIR}"
case "${CASE_NAME}" in
  halfspace)
    TASK="prepare_halfspace_dd_paper('assemble'); test_halfspace_dd_paper;" ;;
  ecc)
    TASK="if isempty(gcp('nocreate')), parpool('Processes',4); end; run_pc_ecc_residual_ablation_experiment('ecc','paper',struct('output_tag','review_repaired_20260905')); redraw_paper_output_figures('ecc');" ;;
  pc)
    TASK="if isempty(gcp('nocreate')), parpool('Processes',4); end; run_pc_ecc_residual_ablation_experiment('pc','paper',struct('output_tag','review_repaired_20260905')); refresh_pc_review_baseline; redraw_paper_output_figures('pc');" ;;
  redraw)
    TASK="prepare_halfspace_dd_paper('redraw'); redraw_paper_output_figures('ecc'); redraw_paper_output_figures('pc');" ;;
  all)
    TASK="prepare_halfspace_dd_paper('assemble'); test_halfspace_dd_paper; if isempty(gcp('nocreate')), parpool('Processes',4); end; run_pc_ecc_residual_ablation_experiment('both','paper',struct('output_tag','review_repaired_20260905')); refresh_pc_review_baseline; prepare_greedy_ecc_pc_paper; redraw_paper_output_figures('ecc'); redraw_paper_output_figures('pc');" ;;
  *) echo "Usage: $0 [all|halfspace|ecc|pc|redraw]" >&2; exit 2 ;;
esac
"${MATLAB_BIN}" -batch "addpath('code'); ${TASK}"
