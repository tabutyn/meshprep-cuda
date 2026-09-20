# API contract

The public geometry API is declared in `<parallel_mater/geometry.hpp>` under
namespace `parallel_mater`.

Installed-package consumers must enable CUDA as a CMake project language so CUDA headers and runtime link flags are available.

## General physics

`<parallel_mater/physics.hpp>` exposes the gallery-independent
`parallel_mater::physics::SoftBody` class. Link `ParallelMater::physics`. The class loads a
versioned `.msb` fixed-topology volumetric lattice, owns all CUDA allocations,
and supports up to 32 translated instances sharing one asset topology.

`SoftBodyOptions` controls the fixed timestep, substeps, deterministic Jacobi
constraint iterations, mass, stiffness, damping, fracture threshold, speed and
projection bounds, hierarchy leaf size, optional non-bonded node collision,
and per-instance origins. It does not contain a scene, camera, gameplay goal,
or renderer.

Initialization validates these public ranges: instances `[1,32]`, substeps
`[1,32]`, constraint iterations `[1,256]`, stiffness `[100,160000]`, spring
damping ratio `[0,4]`, velocity damping `[0,30]`, maximum projection fraction
`(0,1]`, velocity response `[0,1]`, fracture persistence `[1,64]`, maximum
speed `[0.5,30]`, strength multiplier `[0.0625,64]`, ground friction `[0,50]`,
and hierarchy leaf size `[1,32]`. Timestep, mass, stiffness, break strain, and
strength must be finite and positive; active origins and gravity must be finite.

`step(gravity, timings, stream)` is the shortest complete update. Applications
with custom collision or coupling use this sequence:

1. `begin_frame()`;
2. for every substep, `prepare_substep(dt, gravity)`, launch application CUDA
   kernels using `nodes()`, then `finish_substep(dt, gravity)`;
3. `finish_frame(timings)`.

Prediction clears `external_impulses` and `position_corrections`. Coupling
kernels write an impulse in N·s and/or a positional correction for each node.
`finish_substep` consumes those arrays, applies graph constraints and fracture,
and advances the body.

`nodes()`, `bonds()`, and `surface()` return borrowed device views. The bond
array is shared topology; `bond_active` is indexed by instance and permanently
changes after fracture until `reset()`. The surface view includes positions,
normals, UVs, active triangle flags, and the refitted hierarchy. Reacquire all
views after a step, reset, initialization, or move.

All owning operations catch implementation exceptions and return `Status`.
Calls are synchronous before return in this release. Asset loading occurs only
during initialization; no file I/O or device allocation occurs in `step()`.
See `examples/soft_body.cu` for a minimal custom ground collider.

The C++ API offers source compatibility within a tagged minor line; a stable C
ABI is not promised. `.msb` assets have their own checked format version. This
release accepts version 1 little-endian assets and rejects unknown versions,
truncated data, invalid graph indices, malformed CSR, and invalid render
bindings rather than attempting forward-compatible interpretation.

## Optional gallery simulation/render views

`<parallel_mater/game.hpp>` is the CUDA-free configuration and campaign entry
point. It provides fifteen `LevelDefinition` records, fluent
`SimulationConfig`, validation, goal evaluation, and `Campaign`. The catalog
is deliberately component-oriented: the five individual simulations come
first, followed by fluid pairs, then the remaining cloth, soft-body, and rope
pairs. Recipes are selected in the native browser or by stable textual slugs;
they are not coupled to keyboard keys.

`<parallel_mater/gallery.hpp>` provides the owning CUDA adapter. Its movable
`GallerySimulation` has `initialize`/`create`, fixed `step`, and `reset`
operations returning `Status`; no exception crosses the public boundary.
`ParticleRenderView`, `SurfaceRenderView`, `RigidBodyRenderView`,
`LatticeRenderView`, and `FrameRenderView` borrow device allocations and must
be reacquired after every step.

```cpp
parallel_mater::sim::GallerySimulation simulation;
auto status = parallel_mater::sim::SimulationBuilder(
        parallel_mater::sim::ExampleContext::water_rope)
    .particles(40'000)
    .rope_nodes(96)
    .iterations(4)
    .build(simulation);

if (status) status = simulation.step();
auto frame = simulation.render_view();
```

Omitted builder values retain the recipe preset. `bridge_grid(columns, rows)`
and `cylinder_grid(columns, rows)` expose the two adjustable procedural fields;
the lower-level `GallerySimulationOptions` exposes the same overrides.
`resize_particles()` changes the deterministic active
prefix without reallocating inside the reserved capacity. `resolved_physics()`
reports the actual timestep, iterations, gravity, and course preset.

`FrameRenderView::lattices` contains every simulated node plus shared local
bond endpoints/rest lengths and per-instance activity. Offset a local endpoint
by `instance * nodes_per_instance`, render only active bonds, and reacquire the
view after stepping because solver buffers may swap. The gallery API performs
no rendering and has no GLFW or OpenGL types.

## Views and ownership

`DeviceMeshView` borrows CUDA device pointers to `float3` positions and indexed `uint3` triangles. `DeviceAabbView` borrows a device array of ordered, finite `Aabb` values. `SharpEdgeView` borrows device `uint2` pairs. The caller owns these buffers and must keep them valid until the call returns.

`Workspace`, `NormalOutput`, and `Hierarchy` own their device allocations. They are movable, not copyable. Capacity is retained and reused. A workspace or output must not participate in overlapping calls.

`NormalOutput` contains one normalized face normal per triangle, a compact array of area-weighted vertex normals, and one compact-normal index per triangle corner. Degenerate triangles emit a zero face normal, contribute no area vector, and increment `degenerate_triangle_count`.

`Hierarchy` contains breadth-first nodes and a stable permutation of input primitive IDs. A branch owns the contiguous range `[first_child, first_child + child_count)`. A leaf owns `[first_primitive, first_primitive + primitive_count)` in `primitive_indices()`. Mesh input returns triangle IDs; AABB input returns AABB indices. `primitive_count()` reports the input count retained when the hierarchy was built.

`refit_hierarchy(DeviceAabbView, Hierarchy&, stream)` updates leaf and branch bounds while preserving node topology and primitive order. Its AABB count must match the hierarchy's original primitive count. It is intended for moving primitives whose partition remains useful.

`refit_hierarchy_unchecked_async` is the advanced hot-loop form. It requires finite, ordered bounds emitted earlier on the same stream, enqueues without validation or synchronization, and relies on the caller's later stream boundary for completion and asynchronous CUDA errors. The water lab uses it between solver iterations; external data should use the checked refit.

## Ordering and sharp edges

Adjacency is sorted by `(vertex, triangle, corner)`. Smooth incident triangle fans are connected through shared edges. A canonical edge `(min(a,b), max(a,b))` in `SharpEdgeView` breaks that connection. Duplicate and reversed edge records have the same effect as one record. Stable vertex order and the smallest incident corner in each component define compact normal IDs.

Hierarchy construction recursively classifies triangle centroids or AABB centers against each parent segment's mean centroid. Octant order is numeric `[0, 7]`; CUB radix sorting preserves primitive-ID order within equal keys. Identical centroids use a stable rank fallback so subdivision terminates.

## Streams and errors

Normal computation, hierarchy construction, and checked refitting synchronize the supplied stream before returning. The explicitly named unchecked-async refit is enqueue-only. A null stream uses CUDA's default-stream semantics.

No public operation throws. `Status::code`, `Status::cuda_error`, and `Status::message` report validation, allocation, CUDA, or invariant failures. A failed call resets output statistics to zero; retained allocation capacity remains reusable.

## Validation

Calls reject null or empty mesh/AABB buffers, counts outside v0.1's 32-bit domain, non-finite coordinates, unordered AABB minima/maxima, out-of-range triangle indices, null sharp-edge storage with a nonzero count, self edges, out-of-range edge endpoints, and a hierarchy leaf capacity outside `[1, 32]`.

## Compatibility names

The former `<meshprep/...>` headers and `meshprep` namespace remain as a
temporary source-compatibility layer. They refer to the same implementation;
new applications should use the ParallelMater names above.
