# Physics library release checklist

The physics layer should ship as a new minor release. It must not change the
published v0.1 geometry contracts or present the eight gallery scenes as stable
library APIs.

## Required before tagging

- [x] Separate `ParallelMater::physics` target and gallery-independent public header.
- [x] Movable ownership, `Status` error boundary, explicit stream contract, and
  borrowed device views.
- [x] Complete-step and prepare/couple/finish APIs with a custom CUDA contact example.
- [x] Fresh local Release build and full 15-test CUDA run on compute capability 8.6.
- [x] CPU-only CTest run where device tests skip and host contracts still execute.
- [x] Installed-package export includes `ParallelMater::physics`.
- [x] Asset ownership and deterministic regeneration documented.
- [x] Compute Sanitizer memcheck, racecheck, initcheck, and synccheck on the new
  `meshprep-physics-tests` executable.
- [ ] CI passes with CUDA 12.6 and 13.1 after the new target is pushed.
- [ ] Run the repository formatter once `clang-format` is available on the
  release workstation or CI image (it was not installed during this audit).
- [x] Run a fresh installed-package consumer against the exact staged prefix.
- [x] Define and run sustained soft-body stability fixtures independent of gallery goals.
- [x] Audit public names, option ranges, ABI expectations, and `.msb` format compatibility.
- [x] Produce an initial performance and memory baseline for the public solver.
- [x] Review all repository assets and source headers for licensing/provenance.
- [x] Use `v0.2.0-alpha.1` for the first public physics prerelease; retain
  `v0.1.x` as the stable geometry-only line until the physics contract settles.

Publishing, committing, and pushing remain explicit maintainer actions.
The CMake project version stays at the last tagged version until the dedicated
release-preparation change, avoiding an untagged checkout that identifies as a
published release.
