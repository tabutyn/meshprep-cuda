#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${script_dir}/.." && pwd)"
blender_executable="${BLENDER_BIN:-/usr/local/bin/blender}"

if [[ ! -x "${blender_executable}" ]]; then
  blender_executable="$(command -v blender || true)"
fi
if [[ -z "${blender_executable}" || ! -x "${blender_executable}" ]]; then
  echo "Blender was not found. Set BLENDER_BIN to its executable." >&2
  exit 1
fi

exec "${blender_executable}" --background \
  --python "${repo_dir}/tools/blender/softbody_asset_pipeline.py" -- \
  reproduce \
  --glb "${repo_dir}/assets/softbody/checker_cylinder.glb" \
  --asset "${repo_dir}/assets/softbody/checker_cylinder.msb" \
  --texture "${repo_dir}/assets/softbody/checker_cylinder_checker.png" \
  --preview "${repo_dir}/assets/softbody/checker_cylinder_preview.png" \
  --relax-iterations 24
