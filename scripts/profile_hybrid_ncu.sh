#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "usage: $0 WATER_LAB_EXECUTABLE [OUTPUT_BASENAME]" >&2
    exit 2
fi

executable=$1
output=${2:-/tmp/bounded-force-particle}
ncu=/opt/nvidia/nsight-compute/2025.4.0/ncu

if [[ ! -x "$ncu" ]]; then
    echo "Nsight Compute 2025.4.0 was not found at $ncu" >&2
    exit 1
fi

# RmProfilingAdminOnly=1 on this workstation, so collection needs sudo. This
# selects one post-warmup fluid-force launch; Compute replay time is not FPS.
sudo "$ncu" --set basic --kernel-name-base function \
    --kernel-name 'regex:particle_forces_kernel' \
    --launch-skip 10 --launch-count 1 \
    --export "$output" --force-overwrite \
    "$executable" --profile 1 --warmups 10 \
    --width 64 --height 64 --drive-box

echo "Open $output.ncu-rep with /opt/nvidia/nsight-compute/2025.4.0/ncu-ui"
