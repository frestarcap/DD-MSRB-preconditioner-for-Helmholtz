#!/usr/bin/env bash
# Accepted Experiment 1: 28.6% basis / 23.4% ROM payload saving.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MATLAB_BIN="${MATLAB_BIN:-matlab}"
case "${1:-assemble}" in
  assemble|redraw)
    "$MATLAB_BIN" -batch "addpath('code'); prepare_halfspace_dd_paper('${1:-assemble}');"
    ;;
  *) printf 'Usage: %s [assemble|redraw]\n' "$0" >&2; exit 2 ;;
esac
