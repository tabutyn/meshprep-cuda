#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "usage: $0 WATER_LAB_EXECUTABLE [OUTPUT_BASENAME]" >&2
    exit 2
fi

executable=$1
output=${2:-bounded-force-water}

# Capture the finite-mass rectangle approaching and settling into the droplet.
# Nsight time includes instrumentation overhead and is not an FPS measurement.
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
    --force-overwrite=true --output="$output" \
    "$executable" --profile 180 --warmups 10 \
    --width 64 --height 64 --drive-box

nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum "$output.nsys-rep"
