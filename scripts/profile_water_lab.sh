#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 WATER_LAB_EXECUTABLE" >&2
    exit 2
fi

"$1" --profile 120 --width 960 --height 720 --substeps 8
nsys profile --trace=cuda,nvtx,osrt --sample=none --force-overwrite=true \
    --output=water-lab-nsys "$1" --profile 1 --width 960 --height 720 --substeps 8
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum water-lab-nsys.nsys-rep
