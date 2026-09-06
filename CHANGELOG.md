# Changelog

## Unreleased

- Add a native CUDA/OpenGL water lab with a 40,500-triangle geodesic droplet.
- Add a precomputed CSR surface graph and multi-substep zero-gravity spring/pressure solver.
- Add per-frame hierarchy rebuilding, clickable ray-picked impulses, and refractive entry/exit ray tracing.
- Add headless stage profiling, NVTX ranges, Nsight evidence, and a live performance title.

## 0.1.0 — 2026-09-06

- Introduce deterministic face/vertex normal preprocessing with sharp-edge fan splitting.
- Introduce deterministic eight-way breadth-first AABB hierarchy construction.
- Add RAII device outputs, reusable workspace, synchronous stream-aware status API, and CMake package export.
- Add CPU-reference fixtures, 100-run determinism tests, sanitizer scripts, scale benchmarks, and profiling recipes.
