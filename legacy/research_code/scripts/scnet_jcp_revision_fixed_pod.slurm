#!/bin/bash
#SBATCH --job-name=jcp_fixed48
#SBATCH --partition=<partition>
#SBATCH --constraint=64core
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --time-min=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
export LC_ALL=C LANG=C
module use /public/software/modules /opt/hpc/software/modules
source ${MATLAB_ENV_SCRIPT}
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export JCPF_MODE="${1:-prepare}"
export JCPF_SOURCE="$(awk -F '\t' '$1==48 {print $2}' sources.tsv)"
export JCPF_REPORT="$(awk -F '\t' '$1==48 {print $3}' sources.tsv)"
export JCPF_OUT="results"
mkdir -p "$JCPF_OUT" "audit/$JCPF_MODE"
hostname
lscpu | sed -n '1,25p'
sha256sum -c input_hashes.txt > "audit/$JCPF_MODE/input_before.txt"
sha256sum code/*.m kernel/* scripts/scnet_jcp_revision_fixed_pod.slurm > "audit/$JCPF_MODE/dependency_hashes.txt"
jcpf_timeout=14100
if [ "$JCPF_MODE" = prepare ]; then jcpf_timeout=1650; fi
set +e
/usr/bin/time -v -o "audit/$JCPF_MODE/resources.txt" \
 timeout --signal=TERM --kill-after=30s "$jcpf_timeout"s \
 ${MATLAB_EXE} \
 -nodesktop -nosplash -nojvm -r \
 "try; addpath('code'); addpath('kernel'); assert(strcmp(version('-release'),'2024a')); maxNumCompThreads(8); run_jcp_revision_3d_fixed_pod(getenv('JCPF_SOURCE'),getenv('JCPF_REPORT'),getenv('JCPF_OUT'),getenv('JCPF_MODE')); catch ME; disp(getReport(ME,'extended','hyperlinks','off')); exit(1); end; exit(0);"
jcpf_exit=$?
set -e
sha256sum -c input_hashes.txt > "audit/$JCPF_MODE/input_after.txt"
printf '%s\n' "$jcpf_exit" > "audit/$JCPF_MODE/matlab_exit_code.txt"
exit "$jcpf_exit"
