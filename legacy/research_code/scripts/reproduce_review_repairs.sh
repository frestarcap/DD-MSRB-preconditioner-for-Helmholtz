#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
MATLAB_BIN="${MATLAB_BIN:-matlab}"
case "${1:-tests}" in
  tests) TASK="test_rom_verified_snapshot_step; test_pml_heterogeneous_symbol;" ;;
  halfspace) TASK="run_halfspace_dd_refinement('paper',struct('tag','review_repaired_20260905','variant_names',{{'global','slabs1-aligned-sym8'}},'solver_cfg',struct('fp_tol',1e-12,'online_helm_accept_tol',1e-12)));" ;;
  ecc) TASK="parpool('Processes',4); run_pc_ecc_residual_ablation_experiment('ecc','paper',struct('output_tag','review_repaired_20260905'));" ;;
  pc) TASK="parpool('Processes',4); run_pc_ecc_residual_ablation_experiment('pc','paper',struct('output_tag','review_repaired_20260905')); refresh_pc_review_baseline;" ;;
  holdout3d) TASK="run_3d_frozen_holdout_audit;" ;;
  audit) TASK="audit_revision_evidence('repaired'); test_review_saved_evidence; write_revision_audit_tables;" ;;
  *) echo "Usage: $0 [tests|halfspace|ecc|pc|holdout3d|audit]" >&2; exit 2 ;;
esac
"${MATLAB_BIN}" -batch "addpath('code'); ${TASK}"
