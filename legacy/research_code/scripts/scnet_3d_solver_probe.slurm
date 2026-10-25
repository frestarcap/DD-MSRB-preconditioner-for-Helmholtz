#!/bin/bash
#SBATCH --job-name=helm3d_probe
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=192G
#SBATCH --time=02:00:00
#SBATCH --time-min=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -eo pipefail
export LC_ALL=C LANG=C
module use /public/software/modules /opt/hpc/software/modules
source ${MATLAB_ENV_SCRIPT}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HELM_PROBE_SNAPSHOT="${1:?Pass the retained snapshot MAT file}"
export HELM_PROBE_TAG="${2:-halfspace_w48_solver_probe}"
MATLAB_EXE=${MATLAB_EXE}
/usr/bin/time -v "$MATLAB_EXE" -nodesktop -nosplash -nojvm -r \
"try; addpath('code'); maxNumCompThreads(8); build_helm_3d_hpc_mex; run_scnet_3d_solver_probe(getenv('HELM_PROBE_SNAPSHOT'),getenv('HELM_PROBE_TAG')); catch ME; disp(getReport(ME,'extended','hyperlinks','off')); exit(1); end; exit(0);"
