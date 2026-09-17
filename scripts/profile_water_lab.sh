#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 WATER_LAB_EXECUTABLE" >&2
    exit 2
fi

"$1" --profile 120 --warmups 10 --width 960 --height 720
"$1" --profile 180 --warmups 10 --width 960 --height 720 --drive-box
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
"$script_dir/profile_hybrid_nsys.sh" "$1" water-lab-nsys
