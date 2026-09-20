# ParallelMater

Deterministic CUDA C++ geometry processing and composable GPU physics.

ParallelMater is an MIT-licensed C++20/CUDA library for applications that keep
geometry and simulation state on an NVIDIA GPU. The installed package contains:

- deterministic face/vertex normals and eight-way AABB hierarchies;
- independent owning `Fluid`, `Cloth`, `Rope`, `SoftBody`, `RigidBody`, and
  `Smoke` solvers;
- a common frame/substep protocol with synchronous and asynchronous completion;
- solver-neutral analytic colliders and deterministic constraint batches;
- deterministic contact painting for particles, deformable surfaces, and
  application-parameterized rigid/cloth textures; and
- device-resident borrowed views for application-defined coupling and rendering.

Renderers, input, authored levels, objectives, presets, and progression are not
part of the installed API. The repository includes a headless 15-recipe gallery
that composes only the public owners, plus an optional native CUDA/OpenGL lab for
interactive visualization of current and historical experiments.

## Build

Requirements are Linux, CMake 3.25+, a C++20 compiler, CUDA Toolkit 12.6+, and a
supported NVIDIA GPU.

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DPARALLEL_MATER_BUILD_TESTS=ON \
  -DPARALLEL_MATER_BUILD_EXAMPLES=ON \
  -DPARALLEL_MATER_BUILD_GALLERY=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Omit `CMAKE_CUDA_COMPILER` when `nvcc` is already on `PATH`; otherwise point it
at the selected toolkit as shown.

Add `-DPARALLEL_MATER_BUILD_LAB=ON` for the GLFW/OpenGL application:

```bash
cmake --build build --target parallel-mater-lab
./build/parallel-mater-lab
```

The build-tree gallery example demonstrates the public composition boundary:

```bash
./build/parallel-mater-gallery-example
```

## Package use

```cmake
find_package(ParallelMater CONFIG REQUIRED)
target_link_libraries(my_target PRIVATE
  ParallelMater::geometry
  ParallelMater::physics)
```

```cpp
#include <parallel_mater/fluid.hpp>
#include <parallel_mater/frame.hpp>

using namespace parallel_mater::physics;

Fluid fluid;
FluidOptions material;
Status status = fluid.initialize(initial_particles, material, stream);
if (status) {
    Completion completion;
    status = fluid.advance_async(
        FrameOptions{1.0F / 60.0F, 4U, {0.0F, -9.81F, 0.0F}},
        completion, stream);
    if (status) status = completion.wait();
}
```

Every solver owns its allocations and is movable but not copyable. Views borrow
device pointers and must not outlive or overlap mutation of their owner.
Applications that couple multiple solvers call `begin_frame`, then
`prepare_substep`, enqueue coupling work, call `finish_substep`, and finally
`finish_frame`. Telemetry readback is a separate explicit operation.

## Public components

- `ParallelMater::geometry`: normal generation, hierarchy build/refit, and
  reusable workspace/output owners.
- `ParallelMater::physics`: frame protocol, completion tokens, colliders,
  deterministic constraint batches, and all six owning solvers.
- `parallel-mater-example-gallery`: build-tree-only public-API compositions;
  never installed or exported.
- `parallel-mater-lab`: optional native visualization and interaction client.

Start with [`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md), then use the
complete one-sentence public inventory in
[`docs/API_INVENTORY.md`](docs/API_INVENTORY.md). Detailed contracts are in
[`docs/API.md`](docs/API.md), and the package layering is described in
[`docs/SIMULATION_API_ARCHITECTURE.md`](docs/SIMULATION_API_ARCHITECTURE.md).
The current deletion/migration priorities are recorded in
[`docs/CODE_REDUCTION_AUDIT.md`](docs/CODE_REDUCTION_AUDIT.md).

## Correctness and determinism

Geometry ordering is defined by stable CUB sorting and scans. Same-GPU calls
with identical inputs produce identical hierarchy topology, primitive order,
normal indexing, and deterministic constraint reduction order. Floating-point
bit identity across GPU architectures or compiler versions is not promised.

Release validation includes CPU references, seeded fixtures, same-GPU replay,
CUDA runtime tests, installed-package consumers, Compute Sanitizer, and CUDA
12.6/13.1 CI. Hardware-specific performance results remain explicitly local.

## Performance

The published geometry baseline on an RTX 3050 Ti Laptop GPU and CUDA 13.1 is
documented in [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md). The public soft-body
baseline is in [`docs/PHYSICS_PERFORMANCE.md`](docs/PHYSICS_PERFORMANCE.md).
Profiler and benchmark results are evidence for a specific build and workload,
not universal speedup claims.

## Scope

ParallelMater currently targets Linux and CUDA. It does not provide a stable C
ABI, Python binding, CPU fallback, Windows backend, WebGPU backend, arbitrary
mesh traversal API, incompressible CFD solver, or general rigid-body collision
world. Collider support is deliberately solver-specific and documented by each
owner.

## License

Project source is available under the [MIT License](LICENSE). Benchmark scenes
and reference implementations are not redistributed; see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
