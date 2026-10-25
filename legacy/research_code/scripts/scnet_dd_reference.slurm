#!/bin/bash
#SBATCH --job-name=dd_reference
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=56G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -eo pipefail
export LC_ALL=C LANG=C
module use /public/software/modules /opt/hpc/software/modules
source ${MATLAB_ENV_SCRIPT}
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export REF_SOURCE="$1" REF_RESULTS="$2" REF_OUT="$3"
/usr/bin/time -v -o "${REF_OUT}_resources.txt" timeout --signal=TERM --kill-after=30s 6900s \
 ${MATLAB_EXE} \
 -nodesktop -nosplash -nojvm -r \
 "try; addpath('code'); maxNumCompThreads(8); refine_dd_mechanism_reference(getenv('REF_SOURCE'),getenv('REF_RESULTS'),getenv('REF_OUT'),0.317,1e-10); catch ME; disp(getReport(ME,'extended','hyperlinks','off')); exit(1); end; exit(0);"
sha256sum "$REF_SOURCE" "$REF_RESULTS" > "${REF_OUT}/input_sha256.txt"
