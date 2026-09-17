# meshprep-cuda

Deterministic CUDA C++ mesh preprocessing: face/vertex normals and eight-way AABB hierarchy construction.

`meshprep-cuda` is a small C++20 library for pipelines that already keep geometry on the GPU. It turns caller-owned indexed triangle meshes or primitive AABBs into reusable, device-resident normal and hierarchy outputs. Stable CUB sorting replaces atomic scatter order, so topology, primitive permutation, and corner-normal indices are repeatable on the same supported environment.

The repository also includes a [native CUDA/OpenGL water lab](apps/water_lab/README.md).
It now opens as a tilt-controlled peg course: steer a 3,000-particle soft water
ball with WASD or arrows. Its physical and smooth render skins both use the
frequency-10, 2,000-triangle sphere by default. One fixed tick runs per displayed frame, rebuilding
both hierarchies. The original dynamic-rectangle experiment remains under
`--scene lab`. See the [course notes and measurements](docs/TILT_COURSE.md).

Rectangle-edge contact is covered by a deterministic 1,980-frame headless
experiment. The retained smooth contact shell reduced measured corner vibration
by 83.6% without increasing penetration or skin-physics time; methodology,
failed candidates, limitations, and reproduction commands are in
[`docs/CONTACT_STABILITY.md`](docs/CONTACT_STABILITY.md).

Soft-body fracture is also capture-driven. The saved 245-frame failure now
replays from authored rest and gates supported triangle area, edge length,
retained surface, orientation, and finiteness. The current solver reduces
broken bonds from 4,230 to 483 and keeps structurally supported triangles within
0.6039–1.3759× rest area, with a 1.2780× maximum edge and no supported
inversions, while retaining at least 94.75% of the surface. Fractured bonds no
longer hide their adjacent material triangles; the headless audit still
measures supported and detached geometry separately. The combined `V` view
shows internal voxels and colors broken lattice connections red. See
[`docs/SOFT_BODY_STABILITY.md`](docs/SOFT_BODY_STABILITY.md).

## Why this project exists

The implementation began as working production geometry code with two concrete defects: identical builds emitted different topology, and a two-dimensional launch failed on the 9,963,191-triangle San Miguel scene. The current design makes ordering part of the API contract and uses flattened launches throughout. The refactor is intentionally preserved in Git history.

## Highlights

- normalized face normals and area-weighted vertex normals;
- caller-provided sharp-edge splitting, including duplicate and non-manifold inputs;
- stable `(vertex, triangle, corner)` adjacency and `(parent, octant)` partitioning with CUB;
- breadth-first eight-way hierarchy with contiguous child and primitive ranges;
- generic device-AABB hierarchy input for particles and non-triangle primitives;
- reusable movable RAII workspace and outputs, with no exceptions across the API;
- validation for non-finite coordinates, indices, sharp-edge endpoints, and 32-bit limits;
- seeded CPU-reference tests, 100-run determinism checks, Compute Sanitizer gates, and a 10M-triangle scale test;
- CMake install/export for `find_package(meshprep-cuda CONFIG REQUIRED)`.

## Experimental simulation/render contract

The installable `<meshprep/simulation.hpp>` header defines borrowed CUDA render
views, a validated fixed-step descriptor, composable component flags, and the
eight numbered gallery recipes used by the native app. The separately exported
`meshprep::meshprep-simulation` target owns and steps those recipes headlessly;
it has no GLFW, OpenGL, ray-tracer, or HUD dependency. Native and headless
recipes share one preset factory; optional gravity and solver-iteration
overrides are explicit, and `resolved_physics()` reports what was selected.
The gallery now has a widened 15,000-particle hemispherical bowl with a dynamic sphere,
closed-box rigid-sphere/cloth and rigid-sphere/soft-body examples, a combined
sphere/particle/catching-cloth scene, a torque-driven water wheel with compliant
axle-to-rim soft crosses, a load-bearing procedural soft sphere rolling over
ground cloth into hanging cloth, and a rigid sphere tethered to a central post
by a procedural rope. The native `P`
panel exposes active particle count and physical water-skin detail where
applicable. These simulation examples remain experimental; the core geometry
library's contracts are separate.

```bash
./build/meshprep-simulation-contexts
./build/meshprep-water-lab --context 6
```

See [`examples/simulation_contexts.cu`](examples/simulation_contexts.cu) for
consumer code and
[`docs/SIMULATION_API_ARCHITECTURE.md`](docs/SIMULATION_API_ARCHITECTURE.md)
for the extraction boundary and current experimental limitations.

## Requirements

- Linux;
- CMake 3.25 or newer;
- a C++20 compiler;
- CUDA Toolkit 12.6 or newer and a supported NVIDIA GPU.

CUDA 13.1 on compute capability 8.6 is the runtime-validated configuration for v0.1.0. CI compile-checks CUDA 12.6 and 13.1.

## Build and test

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j
ctest --test-dir build --output-on-failure
./scripts/run_sanitizers.sh build/meshprep-tests
```

The test and sanitizer commands need a CUDA-capable host. Tests cover smooth and sharp meshes, a sharp cube, disconnected fans, duplicate edges, a non-manifold edge, degenerate faces, identical centroids, invalid inputs, and seeded triangle soup.

## API

```cpp
meshprep::Workspace workspace;
meshprep::NormalOutput normals;
meshprep::Hierarchy hierarchy;

meshprep::Status normal_status = meshprep::compute_normals(
    mesh, sharp_edges, workspace, normals, stream);
meshprep::Status hierarchy_status = meshprep::build_hierarchy(
    mesh, {.max_leaf_size = 8}, workspace, hierarchy, stream);

meshprep::Status particle_hierarchy_status = meshprep::build_hierarchy(
    meshprep::DeviceAabbView{device_bounds, particle_count},
    {.max_leaf_size = 8}, workspace, hierarchy, stream);
```

See [`examples/basic.cu`](examples/basic.cu) for a complete upload/use example and [`docs/API.md`](docs/API.md) for ownership and synchronization contracts.

```mermaid
flowchart LR
    M[DeviceMeshView] --> V[validate]
    V --> N[normal pipeline]
    V --> C[centroids + triangle bounds]
    N --> NO[NormalOutput]
    C --> P[stable parent/octant partitions]
    P --> H[BFS nodes + leaf permutation]
    H --> B[bottom-up bounds]
    B --> HO[Hierarchy]
    W[Workspace] -. reused scratch .-> N
    W -. reused scratch .-> P
```

Calls are synchronous before return in v0.1.0. Inputs and outputs stay on the device. Output objects own their allocations; views do not. One `Workspace` must not be used concurrently by multiple calls.

## Performance snapshot

On an RTX 3050 Ti Laptop GPU (4 GB, compute capability 8.6), CUDA 13.1, five warmups and 30 samples:

| Scene | Triangles | Median | p5–p95 | Median throughput | Workspace | Output |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Sibenik | 73,564 | 2.349 ms | 2.290–2.479 ms | 31.3 Mtri/s | 15.5 MiB | 5.9 MiB |
| Sponza | 262,267 | 6.708 ms | 5.789–6.810 ms | 39.1 Mtri/s | 55.1 MiB | 21.0 MiB |
| San Miguel | 9,963,191 | 239.265 ms | 238.910–240.019 ms | 41.6 Mtri/s | 2.04 GiB | 798 MiB |

The deterministic implementation is slower than the legacy code and the pinned `cuda-lbvh` reference on the small published scenes. These algorithms also emit different hierarchy contracts, so the figures are not interchangeable. See [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) for methodology, comparisons, and limitations.

## Scope

v0.1.0 does not provide a C ABI, Python binding, CPU fallback, Windows support, public traversal API, dynamic mesh updates, or bundled third-party scenes. Vertices, triangles, and AABB primitives must remain within documented 32-bit limits. Determinism is guaranteed for repeated calls with identical inputs, options, CUDA software, and GPU architecture; floating-point values are compared to documented tolerances across environments.

## License

Project source is available under the [MIT License](LICENSE). Benchmark scenes and reference implementations are not redistributed and retain their own licenses; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
