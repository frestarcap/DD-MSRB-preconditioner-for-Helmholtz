#!/bin/bash
#SBATCH --job-name=helm3d_w60
#SBATCH --partition=tyhcnormal
#SBATCH --constraint=64core
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=224G
#SBATCH --time=16:00:00
#SBATCH --time-min=16:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -eo pipefail
export LC_ALL=C LANG=C
for module_root in /public/software/modules /opt/hpc/software/modules; do
    if [[ -d "$module_root" ]]; then module use "$module_root"; fi
done
source ${MATLAB_ENV_SCRIPT}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HELM_OMEGA="${1:-60}" HELM_WORKERS="${2:-4}" HELM_THREADS="${3:-4}"
export HELM_SOLVER_OPTIONS_FILE="${4:-profiles/beta2_coarse12000_restart40.json}"
hostname
lscpu | head -25
free -m
date -u
MATLAB_EXE=${MATLAB_EXE}
/usr/bin/time -v "$MATLAB_EXE" -nodesktop -nosplash -nojvm -r \
"try; addpath('code'); maxNumCompThreads(min(8,str2double(getenv('SLURM_CPUS_PER_TASK')))); build_helm_3d_hpc_mex; solver_options=jsondecode(fileread(getenv('HELM_SOLVER_OPTIONS_FILE'))); run_scnet_3d_experiment('full',str2double(getenv('HELM_OMEGA')),str2double(getenv('HELM_WORKERS')),str2double(getenv('HELM_THREADS')),'waveguide',solver_options); catch ME; disp(getReport(ME,'extended','hyperlinks','off')); exit(1); end; exit(0);"
