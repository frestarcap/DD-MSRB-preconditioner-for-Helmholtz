#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Uses the existing frozen 22-query cohort and last3 comparison models.
matlab -singleCompThread -batch "addpath('code'); run_ecc_dd_budget_study('two6_train'); benchmark_ecc_dd_budget('two6'); verify_ecc_dd_budget('two6');"
