#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CASE_NAME="${1:-all}"
MATLAB_BIN="${MATLAB_BIN:-matlab}"

case "${CASE_NAME}" in
  lens|waveguide|all) ;;
  *)
    echo "Usage: $0 [lens|waveguide|all]" >&2
    exit 2
    ;;
esac

cd "${ROOT_DIR}"
if [[ "${CASE_NAME}" == "all" ]]; then
  MATLAB_COMMAND="run_waveholtz_3d_forward_timing('lens',cfg); run_waveholtz_3d_forward_timing('waveguide',cfg);"
else
  MATLAB_COMMAND="run_waveholtz_3d_forward_timing('${CASE_NAME}',cfg);"
fi

"${MATLAB_BIN}" -batch "addpath('code'); cfg=struct( ...
  'timing_repeats',1,'gmres_probe_steps',4, ...
  'run_full_waveholtz_solve',true,'waveholtz_tol',1e-7, ...
  'output_tag','large_full_waveholtz_tol1e7'); ${MATLAB_COMMAND}"
