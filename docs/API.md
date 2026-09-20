# API contract

The public geometry API is declared in `<parallel_mater/geometry.hpp>` under
namespace `parallel_mater`.

Installed-package consumers must enable CUDA as a CMake project language so CUDA headers and runtime link flags are available.

## General physics

Link `ParallelMater::physics` and include `<parallel_mater/physics.hpp>` or one
of the focused solver headers. `Fluid`, `Cloth`, `Rope`, `SoftBody`,
`RigidBody`, and `Smoke` independently own their state; none selects scenes,
objectives, controls, or rendering policy.

`FrameOptions` divides one fixed physical interval into equal substeps.
Each owner follows `begin_frame`, `prepare_substep`, coupling work,
`finish_substep`, and `finish_frame`. `advance()` is the synchronous
convenience driver. `advance_async()` records a movable `Completion`; call
`ready()` to poll or `wait()` to create a completion boundary and report
deferred CUDA failures. The mature soft-body and smoke backends still perform
internal timing/statistics readbacks, so those two may synchronize before
recording the token; fluid completion is enqueue-only.
Statistics downloaded by an async frame are stable after its token completes.
`Fluid`, `Cloth`, `Rope`, and `SoftBody` expose the same
`PointCouplingView`; its impulse buffer is writable only during a prepared
substep. `RigidBody` gathers finite-mass reactions through `apply_force` and
`apply_torque` in the same phase.

`Fluid` owns a deterministic sorted-cell particle system and exposes one
writable impulse per particle. `Cloth` and `Rope` generate dedicated spring
graphs and expose the same node/bond/surface views as `SoftBody`. `RigidBody`
integrates finite-mass translation and rotation and accepts forces and torques
during its prepared coupling phase. `Smoke` keeps its analytic advection and
one-way aerodynamic coupling model.

`SoftBody` loads a versioned `.msb` fixed-topology volumetric lattice, owns all
CUDA allocations, and supports up to 256 translated instances sharing one
asset topology. It accepts either a filesystem path or `SoftBodyAssetView`;
the memory overload parses and copies bytes before returning and performs no
filesystem access.

`SoftBodyOptions` controls the fixed timestep, substeps, deterministic Jacobi
constraint iterations, mass, stiffness, damping, fracture threshold, speed and
projection bounds, hierarchy leaf size, optional non-bonded node collision,
and per-instance origins. It does not contain a scene, camera, gameplay goal,
or renderer.

Initialization validates these public ranges: instances `[1,256]`, substeps
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
Synchronous convenience calls wait before return; asynchronous calls leave the
completion boundary with the token. Asset loading occurs only during
initialization; no file I/O or device allocation occurs in a simulation frame.
See `examples/soft_body.cu` for a minimal custom ground collider.

The C++ API offers source compatibility within a tagged minor line; a stable C
ABI is not promised. `.msb` assets have their own checked format version. This
release accepts version 1 little-endian assets and rejects unknown versions,
truncated data, invalid graph indices, malformed CSR, and invalid render
bindings rather than attempting forward-compatible interpretation.

## Example-only recipe gallery

The recipe catalog, `GallerySimulation`, builder, and render aggregation views
live under `examples/gallery`. They are compiled only when
`PARALLEL_MATER_BUILD_GALLERY=ON` and are deliberately absent from the install
tree and `ParallelMaterTargets.cmake`. They demonstrate composition; they are
not a second public simulation abstraction. Gameplay objectives and automatic
progression remain in `apps/water_lab`.

The complete installed-symbol inventory is in
[`API_INVENTORY.md`](API_INVENTORY.md); the gallery reading order is in
[`GALLERY_APP_STUDY.md`](GALLERY_APP_STUDY.md).

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
