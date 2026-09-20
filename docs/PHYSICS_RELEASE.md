# Physics library release checklist

The physics layer should ship as a new minor release. It must not change the
published v0.1 geometry contracts or present the example gallery recipes as stable
library APIs.

## Required before tagging

- [x] Separate `ParallelMater::physics` target and gallery-independent public header.
- [x] Movable ownership, `Status` error boundary, explicit stream contract, and
  borrowed device views.
- [x] Complete-step and prepare/couple/finish APIs with a custom CUDA contact example.
- [x] Fresh local Release build and complete 17-test CUDA CTest run on compute capability 8.6.
- [x] CPU-only CTest run where device tests skip and host contracts still execute.
- [x] Installed-package export includes `ParallelMater::physics`.
- [x] Asset ownership and deterministic regeneration documented.
- [x] Compute Sanitizer memcheck, racecheck, initcheck, and synccheck on the
  public physics, smoke, and gallery-runtime executables.
- [x] CI passes with CUDA 12.6 and 13.1 after the new target is pushed
  ([build 35522367518](https://github.com/tabutyn/meshprep-cuda/actions/runs/35522367518)).
- [x] Run `clang-format` over the installed library, public examples, package
  consumer, and public runtime tests; keep the isolated historical app out of
  formatting-only release churn.
- [x] Run a fresh installed-package consumer against the exact staged prefix.
- [x] Define and run sustained soft-body stability fixtures independent of gallery goals.
- [x] Audit public names, option ranges, ABI expectations, and `.msb` format compatibility.
- [x] Refresh the performance and memory baseline after the private-core split.
- [x] Review all repository assets and source headers for licensing/provenance.
- [x] Use `v0.2.0-alpha.1` for the first public physics prerelease; retain
  `v0.1.x` as the stable geometry-only line until the physics contract settles.

Publishing and tagging remain explicit maintainer actions.
The CMake project version stays at the last tagged version until the dedicated
release-preparation change, avoiding an untagged checkout that identifies as a
published release.
