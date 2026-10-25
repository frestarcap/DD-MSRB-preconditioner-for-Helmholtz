#!/bin/bash
# Fetch only lightweight results/logs. Full MAT models remain on the cluster.
set -euo pipefail
export LC_ALL=C LANG=C
REMOTE_RUN=${1:-${REMOTE_RUN_ROOT}/helm3d_20260909_v1}
LOCAL_RESULTS=${2:-outputs/scnet_3d_20260909/remote_v1}
mkdir -p "$LOCAL_RESULTS/logs"
rsync -az "<cluster>:$REMOTE_RUN/logs/" "$LOCAL_RESULTS/logs/"
rsync -az "<cluster>:$REMOTE_RUN/source_manifest.json" "$LOCAL_RESULTS/"
rsync -az --include='*/' --include='*.json' --include='*.csv' --exclude='*' \
    "<cluster>:$REMOTE_RUN/outputs/scnet_3d_20260909/" "$LOCAL_RESULTS/"
