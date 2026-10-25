#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-pilot}"
MATLAB_BIN="${MATLAB_BIN:-matlab}"
case "${MODE}" in
  smoke|pilot|paper|redraw) ;;
  *) echo "Usage: $0 [smoke|pilot|paper|redraw]" >&2; exit 2 ;;
esac
cd "${ROOT_DIR}"
if [[ "${MODE}" == "redraw" ]]; then
  "${MATLAB_BIN}" -batch "addpath('code'); plot_halfspace_dd_refinement('paper');"
else
  "${MATLAB_BIN}" -batch \
    "addpath('code'); run_halfspace_dd_refinement('${MODE}'); if strcmp('${MODE}','paper'), plot_halfspace_dd_refinement('paper'); end"
fi
