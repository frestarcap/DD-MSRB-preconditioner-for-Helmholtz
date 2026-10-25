#!/bin/bash
#SBATCH --job-name=helm3d
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=192G
#SBATCH --time=08:00:00
#SBATCH --time-min=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -eo pipefail
export LC_ALL=C LANG=C
module use /public/software/modules /opt/hpc/software/modules
source ${MATLAB_ENV_SCRIPT}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HELM_ACTION="${1:-verify}" HELM_OMEGA="${2:-24}"
export HELM_WORKERS="${3:-8}" HELM_THREADS="${4:-1}"
export HELM_PROBLEM="${5:-waveguide}"
export HELM_SOLVER_OPTIONS_FILE="${6:-}"
hostname
lscpu | head -25
free -m
date -u
MATLAB_EXE=${MATLAB_EXE}
/usr/bin/time -v "$MATLAB_EXE" -nodesktop -nosplash -nojvm -r \
"try; addpath('code'); maxNumCompThreads(min(8,str2double(getenv('SLURM_CPUS_PER_TASK')))); build_helm_3d_hpc_mex; solver_options=struct(); if ~isempty(getenv('HELM_SOLVER_OPTIONS_FILE')); solver_options=jsondecode(fileread(getenv('HELM_SOLVER_OPTIONS_FILE'))); end; if strcmp(getenv('HELM_ACTION'),'kernel'); disp('SCNET_3D_KERNEL_PASS'); elseif strcmp(getenv('HELM_ACTION'),'qualify'); run_scnet_3d_experiment('verify',6,8,1); run_scnet_3d_scaling; elseif strcmp(getenv('HELM_ACTION'),'scaling'); run_scnet_3d_scaling; else; run_scnet_3d_experiment(getenv('HELM_ACTION'),str2double(getenv('HELM_OMEGA')),str2double(getenv('HELM_WORKERS')),str2double(getenv('HELM_THREADS')),getenv('HELM_PROBLEM'),solver_options); end; catch ME; disp(getReport(ME,'extended','hyperlinks','off')); exit(1); end; exit(0);"
