#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
out=outputs/marmousi_20260920/data
mkdir -p "$out"
if [[ ! -f "$out/vp_marmousi_bi" ]]; then
  curl -fL --retry 2 --max-time 120 \
    https://raw.githubusercontent.com/devitocodes/data/bbb6bc99ccf91b0f6451bd4b7f46353382edf4d0/Simple2D/vp_marmousi_bi \
    -o "$out/vp_marmousi_bi"
fi
expected=0f72aca4ffc47707d9e3e2970ccd3f604bc4e2e70a5497273a4d3786748f4c83
actual=$(shasum -a 256 "$out/vp_marmousi_bi" | cut -d ' ' -f 1)
[[ "$actual" == "$expected" ]] || { echo 'Marmousi checksum mismatch' >&2; exit 1; }
phase="${1:-paired}"
case "$phase" in smoke|pilot|strong|paired|compact) ;; *) echo 'Use smoke, pilot, strong, paired, or compact' >&2; exit 2;; esac
"${MATLAB_BIN:-matlab}" -singleCompThread -batch \
  "addpath('code'); run_marmousi_rom_experiment('$phase');"
