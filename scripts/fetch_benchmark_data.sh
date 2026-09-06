#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dependency_dir="${project_root}/benchmarks/third_party/cuda-lbvh"
data_dir="${project_root}/benchmarks/data/mesh"
revision="605802671beb6473b74a43552168f61e63af46db"

if [[ ! -d "${dependency_dir}/.git" ]]; then
    git clone https://github.com/nolmoonen/cuda-lbvh.git "${dependency_dir}"
fi
git -C "${dependency_dir}" fetch --depth 1 origin "${revision}"
git -C "${dependency_dir}" checkout --detach "${revision}"
mkdir -p "${data_dir}"
unzip -n "${dependency_dir}/scenes/sibenik.zip" -d "${data_dir}"
unzip -n "${dependency_dir}/scenes/sponza.zip" -d "${data_dir}"
7z x "${dependency_dir}/scenes/sanmiguel.zip.001" "-o${data_dir}" -aos
(cd "${project_root}" && sha256sum -c benchmarks/data/SHA256SUMS)
