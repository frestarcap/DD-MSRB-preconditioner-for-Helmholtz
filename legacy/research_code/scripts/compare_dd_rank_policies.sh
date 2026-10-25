#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CASE_NAME="${1:-all}"
RUN_TAG="${2:-policy_crosscheck_20260905}"
MATLAB_BIN="${MATLAB_BIN:-matlab}"
case "${CASE_NAME}" in
  all|halfspace|waveguide) ;;
  *) echo "Usage: $0 [all|halfspace|waveguide] [output-tag]" >&2; exit 2 ;;
esac
if [[ ! "${RUN_TAG}" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "Output tag must contain only letters, digits, underscores, and hyphens." >&2
  exit 2
fi
cd "${ROOT_DIR}"
"${MATLAB_BIN}" -batch \
  "addpath('code'); run_dd_rank_policy_comparison('${CASE_NAME}','${RUN_TAG}');"
