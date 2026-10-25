#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Requires the frozen validate_results.mat cohort from the strict DD study.
matlab -singleCompThread -batch "addpath('code'); run_ecc_dd_budget_study('uniform3_train'); benchmark_ecc_dd_budget('uniform3'); verify_ecc_dd_budget('uniform3');"
