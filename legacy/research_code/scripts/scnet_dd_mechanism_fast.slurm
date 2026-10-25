#!/bin/bash
#SBATCH --job-name=dd_mechanism
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -eo pipefail
export LC_ALL=C LANG=C
module use /public/software/modules /opt/hpc/software/modules
source ${MATLAB_ENV_SCRIPT}
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export MECH_SOURCE="$1" MECH_RESULTS="$2" MECH_OUT="$3" MECH_MODE="${4:-full}"
mkdir -p "$MECH_OUT"
sha256sum "$MECH_SOURCE" "$MECH_RESULTS" > "$MECH_OUT/input_sha256_before.txt"
/usr/bin/time -v -o "$MECH_OUT/resources.txt" timeout --signal=TERM --kill-after=30s 13800s \
 ${MATLAB_EXE} \
 -nodesktop -nosplash -nojvm -r \
 "try; addpath('code'); addpath('kernel'); assert(exist('helm_gmres_cgs2_mex','file')==3); test_helm_3d_hpc;  maxNumCompThreads(8); run_dd_highfreq_mechanism_fast(getenv('MECH_SOURCE'),getenv('MECH_RESULTS'),getenv('MECH_OUT'),getenv('MECH_MODE')); catch ME; disp(getReport(ME,'extended','hyperlinks','off')); exit(1); end; exit(0);"
sha256sum "$MECH_SOURCE" "$MECH_RESULTS" > "$MECH_OUT/input_sha256_after.txt"
cmp "$MECH_OUT/input_sha256_before.txt" "$MECH_OUT/input_sha256_after.txt"
