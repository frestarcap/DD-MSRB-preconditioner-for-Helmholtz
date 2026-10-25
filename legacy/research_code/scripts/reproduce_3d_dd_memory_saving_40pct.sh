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
"${MATLAB_BIN}" -batch \
  "addpath('code'); run_helmholtz_3d_40pct_reproduction('${CASE_NAME}');"
