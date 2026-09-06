#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 TEST_EXECUTABLE" >&2
    exit 2
fi
sanitizer="${COMPUTE_SANITIZER:-}"
if [[ -z "${sanitizer}" ]]; then
    sanitizer="$(command -v compute-sanitizer || true)"
fi
if [[ -z "${sanitizer}" && -x /usr/local/cuda/bin/compute-sanitizer ]]; then
    sanitizer=/usr/local/cuda/bin/compute-sanitizer
fi
if [[ -z "${sanitizer}" ]]; then
    echo "compute-sanitizer not found; set COMPUTE_SANITIZER" >&2
    exit 2
fi
for tool in memcheck racecheck initcheck synccheck; do
    "${sanitizer}" --tool "${tool}" --error-exitcode 99 "$1"
done
