#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 BENCHMARK SCENE.obj" >&2
    exit 2
fi
ncu --set full --kernel-name-base demangled --launch-skip 0 --launch-count 20 \
    --export parallel-mater-ncu --force-overwrite "$1" --obj "$2" \
    --operation hierarchy --warmups 0 --iterations 1
