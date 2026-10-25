#!/bin/bash
#SBATCH --job-name=helm3d_whos
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -eo pipefail
export LC_ALL=C LANG=C
module use /public/software/modules /opt/hpc/software/modules
source ${MATLAB_ENV_SCRIPT}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
MATLAB_EXE=${MATLAB_EXE}
mkdir -p results
hostname
sha256sum code/*.m scripts/scnet_waveguide_whos.slurm sources.tsv > results/audit_source_sha256.txt
while IFS=$'\t' read -r omega model_source report_source; do
    export WHOS_SOURCE="$model_source" WHOS_REPORT="$report_source" WHOS_OMEGA="$omega"
    sha256sum "$model_source" "$report_source" > "results/w${omega}_source_sha256.txt"
    /usr/bin/time -v -o "results/w${omega}_inspection_resources.txt" \
        timeout --signal=TERM --kill-after=20s 1200s "$MATLAB_EXE" \
        -nodesktop -nosplash -nojvm -r \
        "try; addpath('code'); assert(strcmp(version('-release'),'2024a')); maxNumCompThreads(1); measure_waveguide_rom_whos(getenv('WHOS_SOURCE'),fullfile('results',['w' getenv('WHOS_OMEGA') '_whos.json']),getenv('WHOS_REPORT')); catch ME; disp(getReport(ME,'extended','hyperlinks','off')); exit(1); end; exit(0);"
done < sources.tsv
printf 'WAVEGUIDE_WHOS_COHORT_PASS\n'
