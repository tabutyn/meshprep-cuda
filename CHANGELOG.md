# Changelog

## Unreleased

- Split the physics umbrella into self-contained Fluid, Cloth, Rope,
  SoftBody, RigidBody, Smoke, and coupling headers and move the fixed-topology
  implementation into private library sources.
- Add solver-neutral analytic colliders and deterministic sorted constraint
  batches for cross-solver coupling without floating-point contact atomics.
- Separate SoftBody, Fluid, Cloth, Rope, and Smoke telemetry collection from
  frame advancement so asynchronous submission performs no diagnostic
  readback.
- Keep gallery recipes, controls, objectives, progression, and visualization
  state out of the installed package contract.
- Rename the public package to ParallelMater with canonical
  `<parallel_mater/...>` headers, `parallel_mater` namespace spelling,
  `ParallelMater::` CMake targets, and `libparallel-mater-*` artifacts. Keep the
  former names as a temporary source-compatibility layer.
- Restore the original eight-context native application as
  `parallel-mater-lab`; it now links the exported gallery library instead of
  recompiling the simulation core privately.
- Add a gallery-independent `ParallelMater::physics` target and `SoftBody` API with
  fixed stepping, manual CUDA coupling, fracture statistics, material controls,
  and borrowed node/bond/surface views.
- Make tests, examples, benchmarks, capture tools, the eight-scene gallery API,
  and the native OpenGL app opt-in; the default build now contains only the
  geometry and general physics libraries.
- Add an installed-package physics target, standalone runtime contract test,
  and custom CUDA ground-contact example.
- Add a native CUDA/OpenGL water lab with a 40,500-triangle geodesic droplet.
- Add a precomputed CSR surface graph and multi-substep zero-gravity spring/pressure solver.
- Add per-frame hierarchy rebuilding, clickable ray-picked impulses, and refractive entry/exit ray tracing.
- Add headless stage profiling, NVTX ranges, Nsight evidence, and a live performance title.

## 0.1.0 — 2026-09-06

- Introduce deterministic face/vertex normal preprocessing with sharp-edge fan splitting.
- Introduce deterministic eight-way breadth-first AABB hierarchy construction.
- Add RAII device outputs, reusable workspace, synchronous stream-aware status API, and CMake package export.
- Add CPU-reference fixtures, 100-run determinism tests, sanitizer scripts, scale benchmarks, and profiling recipes.
