#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Reuses the frozen 22-query cohort and last3 reference models.
matlab -singleCompThread -batch "addpath('code'); run_ecc_dd_budget_study('uniform5_train'); benchmark_ecc_dd_budget('uniform5'); verify_ecc_dd_budget('uniform5');"
