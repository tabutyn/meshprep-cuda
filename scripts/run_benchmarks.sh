#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "usage: $0 BUILD_DIR [OUTPUT.csv]" >&2
    exit 2
fi
build_dir="$1"
output="${2:-results.csv}"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
benchmark="${build_dir}/parallel-mater-benchmark"

: > "${output}"
for scene in sibenik sponza sanmiguel; do
    "${benchmark}" --obj "${project_root}/benchmarks/data/mesh/${scene}.obj" \
        --operation hierarchy --warmups 5 --iterations 30 | tee -a "${output}"
done
