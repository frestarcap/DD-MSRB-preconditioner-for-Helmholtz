#!/bin/bash
#SBATCH --job-name=helm3d_sweep
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --time-min=04:00:00
#SBATCH --array=0-3%1
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# One 64-CPU allocation at a time. Every case has independent resumable data.
set -euo pipefail
HELM_CASE_OMEGAS=(24 48 36 48)
HELM_CASE_PROBLEMS=(waveguide halfspace waveguide waveguide)
exec bash scripts/scnet_3d.slurm full \
    "${HELM_CASE_OMEGAS[$SLURM_ARRAY_TASK_ID]}" 8 4 \
    "${HELM_CASE_PROBLEMS[$SLURM_ARRAY_TASK_ID]}"
