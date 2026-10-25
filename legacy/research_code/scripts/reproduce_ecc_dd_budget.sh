#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
matlab -singleCompThread -batch "addpath('code'); run_ecc_dd_budget_study('audit'); run_ecc_dd_budget_study('train'); run_ecc_dd_budget_study('screen'); run_ecc_dd_budget_study('confirm'); run_ecc_dd_budget_study('validate'); verify_ecc_dd_budget; test_marginal_storage_budget; test_block_reduced_assembly; test_halfspace_refined_partition;"
python3 scripts/summarize_ecc_dd_budget.py
