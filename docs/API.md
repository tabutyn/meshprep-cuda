# API contract

The public API is declared in `<meshprep/meshprep.hpp>` under namespace `meshprep`.

Installed-package consumers must enable CUDA as a CMake project language so CUDA headers and runtime link flags are available.

## Experimental simulation/render views

`<meshprep/simulation.hpp>` adds the `meshprep::sim` namespace. Its component
flags and numbered `ExampleContext` catalog describe the native integration
fixtures without making the renderer depend on their concrete solvers.
`ParticleRenderView`, `SurfaceRenderView`, `RigidBodyRenderView`, and
`FrameRenderView` borrow device memory; they never allocate or transfer it.
Reacquire them after every simulation step because an owning backend may move
its allocations. `valid(FixedStepOptions)` checks the positive, finite
timestep contract.

The installed `meshprep::meshprep-simulation` target provides the movable
PIMPL `GallerySimulation`. Its `initialize`/`create`, fixed `step`, and `reset`
operations return `Status`; `render_view()` borrows its current CUDA arrays.
CUDA and geometry-library failures raised inside the adapted solvers retain
their original `StatusCode` and `cuda_error` at this noexcept boundary.
Unexpected implementation exceptions report `internal_error` with
`cudaSuccess`; callers must not interpret that pair as a successful CUDA call.
Context 4 requires a caller-provided converted cylinder `.msb`; the bowl,
cloth, water wheel and its soft cross, rigid spheres, soft-sphere, and rope examples
are procedural. This remains an experimental
composition API rather than a stable ABI for each underlying solver. See
`examples/simulation_contexts.cu` and `docs/SIMULATION_API_ARCHITECTURE.md`.

Each context selects one complete physics preset. The course uses four
iterations and gravity `(0, -7.2, 0)`. Contexts 2–8 use four iterations and
Earth gravity `(0, -9.81, 0)` by default; context 2 deliberately uses twice
that magnitude. Context 7 also starts under vertical Earth gravity; camera-relative
arrow input supplies any horizontal component used to produce rolling.
`GallerySimulationOptions::solver_iterations_override`, `gravity_override`,
`particle_count_override` (256–100,000 active particle IDs), and
`physical_skin_frequency_override` (2–45), and `rope_node_count_override`
(8–512, context 8) are optional: leaving them empty selects that
recipe, while assigning a value is an explicit departure from it.
`GallerySimulation::resize_particles()` adjusts the active prefix in place;
growth reinitializes new IDs at deterministic HCP spawn positions, and shrinkage
removes IDs from the tail. `reset()` restores positions for the current active
count; it does not undo a resize. Reserved particle storage is sized to the
larger of the recipe default and the requested override, up to 100,000 IDs;
there is no arbitrary-ID deletion or unbounded spawning in this version. Context 6
recycles particles after they leave the lower collector to the high inlet; its
wheel shell follows the lower-left inlet-to-outlet path and leaves openings at
nine and six o'clock.
`resolved_physics()` reports the exact timestep, iteration count, gravity, and
preset class actually in use. The installed backend and native gallery both
use the same recipe factory, so their default and overridden values do not
silently diverge.

`FrameRenderView::lattices` exposes the complete simulated graph for volume and
fracture visualization: borrowed node positions/flags, shared local bond
endpoints/rest lengths, and per-instance bond activity. Render only live bonds;
local endpoints are offset by `instance * nodes_per_instance`. The authored post has
1,000 nodes, including 500 interior nodes and 70 pinned foundation nodes. The
procedural soft sphere has 1,000 nodes; context 7 merges it with a 576-node
hanging cloth and a 1,600-node horizontal ground cloth. Context 6 uses two
procedural connected 1,155-node, three-layer crosses. Each has a pinned 3x3x3
axle volume and four three-wide rim anchors on a common axis. Reacquire these views after stepping
because the position buffers swap. No extra device allocation or download is
performed by this getter.
Context 8 exposes a procedural rope lattice and render tube plus three rigid
views: its finite-mass sphere, ground, and center post.

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
