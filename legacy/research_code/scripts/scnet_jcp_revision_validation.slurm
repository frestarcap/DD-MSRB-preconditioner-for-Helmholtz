#!/bin/bash
#SBATCH --job-name=jcp_frozen48
#SBATCH --partition=<partition>
#SBATCH --constraint=64core
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --time-min=04:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -eo pipefail
export LC_ALL=C LANG=C
module use /public/software/modules /opt/hpc/software/modules
source ${MATLAB_ENV_SCRIPT}
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export JCP_QUERY_INDEX="${SLURM_ARRAY_TASK_ID:-2}" JCP_MODE="${1:-production}"
export JCP_SOURCE="$(awk -F '\t' '$1==48 {print $2}' sources.tsv)"
export JCP_REPORT="$(awk -F '\t' '$1==48 {print $3}' sources.tsv)"
export JCP_OUT="results/${JCP_MODE}_q${JCP_QUERY_INDEX}"
mkdir -p "$JCP_OUT"
hostname
lscpu | head -25
sha256sum "$JCP_SOURCE" "$JCP_REPORT" > "$JCP_OUT/input_sha256_before.txt"
/usr/bin/time -v -o "$JCP_OUT/resources.txt" \
 timeout --signal=TERM --kill-after=30s 14100s \
 ${MATLAB_EXE} \
 -nodesktop -nosplash -nojvm -r \
 "try; addpath('code'); addpath('kernel'); assert(strcmp(version('-release'),'2024a')); maxNumCompThreads(8); if strcmp(getenv('JCP_MODE'),'verify'); test_helm_3d_hpc; end; run_jcp_revision_3d_validation(getenv('JCP_SOURCE'),getenv('JCP_REPORT'),getenv('JCP_OUT'),str2double(getenv('JCP_QUERY_INDEX')),getenv('JCP_MODE')); catch ME; disp(getReport(ME,'extended','hyperlinks','off')); exit(1); end; exit(0);"
sha256sum "$JCP_SOURCE" "$JCP_REPORT" > "$JCP_OUT/input_sha256_after.txt"
cmp "$JCP_OUT/input_sha256_before.txt" "$JCP_OUT/input_sha256_after.txt"
