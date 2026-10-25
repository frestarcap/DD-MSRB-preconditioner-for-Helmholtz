#!/bin/bash
#SBATCH --job-name=dd_interfaces_flexible
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Usage: sbatch scripts/scnet_dd_interfaces_flexible.slurm MANIFEST.tsv [NEW_OUTPUT_DIR]
# Relative paths in the manifest are relative to SLURM_SUBMIT_DIR.
# This script neither modifies nor deletes any existing result directory.
set -euo pipefail
export LC_ALL=C LANG=C
cd "${SLURM_SUBMIT_DIR:?Submit from the diagnostic working directory}"
interface_manifest="${1:-interfaces_fast_manifest.tsv}"
interface_audit_dir="${2:-results/interfaces_${SLURM_JOB_ID}}"
test -f "$interface_manifest"
test ! -e "$interface_audit_dir"

# Check the entire cohort before creating any audit outputs.
while IFS=$'\t' read -r frequency training vectors complete expected_count reference_dir; do
    case "$frequency" in ''|\#*) continue ;; esac
    test -f "$training"
    test -d "$vectors"
    test -f "$complete"
    test -n "$expected_count"
    if test -n "$reference_dir" && test "$reference_dir" != '-'; then
        test -f "$reference_dir/summary.json"
        test -f "$reference_dir/refined_reference.mat"
        test -f "$reference_dir/global_refined_error.mat"
        test -f "$reference_dir/dd_refined_error.mat"
    fi
done < "$interface_manifest"

mkdir "$interface_audit_dir"
cp "$interface_manifest" "$interface_audit_dir/manifest.tsv"
sha256sum code/audit_dd_residual_interfaces.m code/export_dd_mechanism_slices.m \
    code/audit_dd_refined_space_capture.m > "$interface_audit_dir/audit_source_sha256.txt"
printf 'job_id=%s\nnode=%s\ncpus=%s\n' "$SLURM_JOB_ID" "$HOSTNAME" "$SLURM_CPUS_PER_TASK" > "$interface_audit_dir/job.txt"
set +u
module use /public/software/modules /opt/hpc/software/modules
source ${MATLAB_ENV_SCRIPT}
set -u
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4

while IFS=$'\t' read -r frequency training vectors complete expected_count reference_dir; do
    case "$frequency" in ''|\#*) continue ;; esac
    frequency_audit_dir="$interface_audit_dir/w${frequency}"
    mkdir "$frequency_audit_dir"
    export INTERFACE_SOURCE="$training" INTERFACE_DIR="$vectors"
    export INTERFACE_OUT="$frequency_audit_dir/interfaces.json"
    export INTERFACE_COMPLETE="$complete" INTERFACE_EXPECTED_COUNT="$expected_count"
    export INTERFACE_FREQUENCY="$frequency"
    export INTERFACE_REFERENCE="${reference_dir:--}"
    shopt -s nullglob
    interface_vectors=("$vectors/"*_vectors.mat)
    test "${#interface_vectors[@]}" -eq "$expected_count"
    interface_inputs=("$training" "$complete" "${interface_vectors[@]}")
    if test "$INTERFACE_REFERENCE" != '-'; then
        interface_inputs+=("$INTERFACE_REFERENCE/summary.json" "$INTERFACE_REFERENCE/refined_reference.mat" \
            "$INTERFACE_REFERENCE/global_refined_error.mat" "$INTERFACE_REFERENCE/dd_refined_error.mat")
    fi
    sha256sum "${interface_inputs[@]}" > "$frequency_audit_dir/input_sha256_before.txt"
    /usr/bin/time -v -o "$frequency_audit_dir/resources.txt" \
      timeout --signal=TERM --kill-after=20s 800s \
      ${MATLAB_EXE} \
      -nodesktop -nosplash -nojvm -r \
      "try; addpath('code'); maxNumCompThreads(4); completed=jsondecode(fileread(getenv('INTERFACE_COMPLETE'))); assert(completed.validated && strcmp(completed.mode,'full')); assert(completed.omega==str2double(getenv('INTERFACE_FREQUENCY'))); report=audit_dd_residual_interfaces(getenv('INTERFACE_SOURCE'),getenv('INTERFACE_DIR'),getenv('INTERFACE_OUT')); assert(numel(report.records)==str2double(getenv('INTERFACE_EXPECTED_COUNT'))); assert(isequal(sort({report.records.variant}),sort({'GG','GGG','DD','DDD','GGD','DDG'}))); assert(all(abs([report.records.mu]-.317)<1e-10)); destination=fileparts(getenv('INTERFACE_OUT')); reference=getenv('INTERFACE_REFERENCE'); if strcmp(reference,'-'), export_dd_mechanism_slices(getenv('INTERFACE_SOURCE'),getenv('INTERFACE_DIR'),fullfile(destination,'slices')); else, export_dd_mechanism_slices(getenv('INTERFACE_SOURCE'),getenv('INTERFACE_DIR'),fullfile(destination,'slices'),reference); audit_dd_refined_space_capture(getenv('INTERFACE_SOURCE'),getenv('INTERFACE_DIR'),reference,fullfile(destination,'refined_space_capture.json')); end; fprintf('DD_INTERFACE_FREQUENCY_PASS omega=%g\n',report.omega); catch ME; disp(getReport(ME,'extended','hyperlinks','off')); exit(1); end; exit(0);" \
      > "$frequency_audit_dir/matlab.log" 2>&1
    sha256sum "${interface_inputs[@]}" > "$frequency_audit_dir/input_sha256_after.txt"
    cmp "$frequency_audit_dir/input_sha256_before.txt" "$frequency_audit_dir/input_sha256_after.txt"
done < "$interface_manifest"
printf 'DD_INTERFACE_COHORT_PASS\n' | tee "$interface_audit_dir/complete.txt"
