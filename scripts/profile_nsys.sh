#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 BENCHMARK SCENE.obj" >&2
    exit 2
fi
nsys profile --trace=cuda,nvtx,osrt --sample=none --force-overwrite=true \
    --output=parallel-mater-nsys "$1" --obj "$2" --operation hierarchy --warmups 1 --iterations 1
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum parallel-mater-nsys.nsys-rep
