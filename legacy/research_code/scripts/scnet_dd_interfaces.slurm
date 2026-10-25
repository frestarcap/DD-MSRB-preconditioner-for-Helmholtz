#!/bin/bash
#SBATCH --job-name=dd_mech_interfaces
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -eo pipefail
export LC_ALL=C LANG=C
module use /public/software/modules /opt/hpc/software/modules
source ${MATLAB_ENV_SCRIPT}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
while IFS=$'\t' read -r w src result cpu mem; do
 export INTERFACE_SOURCE="$src" INTERFACE_DIR="results/w${w}" INTERFACE_OUT="results/w${w}/interfaces.json"
 /usr/bin/time -v -o "results/w${w}/interface_resources.txt" timeout 400s \
 ${MATLAB_EXE} \
 -nodesktop -nosplash -nojvm -r \
 "try; addpath('code'); maxNumCompThreads(4); audit_dd_residual_interfaces(getenv('INTERFACE_SOURCE'),getenv('INTERFACE_DIR'),getenv('INTERFACE_OUT')); catch ME; disp(getReport(ME,'extended','hyperlinks','off')); exit(1); end; exit(0);"
 sha256sum "results/w${w}/"*_vectors.mat > "results/w${w}/vector_sha256.txt"
done < sources.tsv
printf 'DD_INTERFACE_COHORT_PASS\n'
