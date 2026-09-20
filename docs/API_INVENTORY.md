# Public API inventory

This first-pass inventory covers the installed headers in `parallel_mater/` and
describes each remaining public type and free function in one sentence; the
native gallery's objectives and progression are intentionally excluded because
they are application code.

As of this pass the installed headers contain 2 enums, 36 structures, 10
classes, and 10 free functions/templates; the installed implementation contains
56 private CUDA kernels. Gallery recipes and render aggregation are excluded.

## Geometry

### Enums

| API | Description |
| --- | --- |
| `StatusCode` | Categorizes success, invalid input, unsupported size, allocation, CUDA, and internal failures without throwing exceptions. |

### Structures

| API | Description |
| --- | --- |
| `Status` | Returns a library status code, optional CUDA error, and static diagnostic message from API operations. |
| `DeviceMeshView` | Borrows device arrays containing vertex positions and indexed triangles. |
| `Aabb` | Stores the minimum and maximum corners of one axis-aligned bounding box. |
| `DeviceAabbView` | Borrows a device array of primitive bounds for hierarchy construction or refitting. |
| `SharpEdgeView` | Borrows canonical vertex pairs that prevent normal smoothing across selected edges. |
| `NormalStatistics` | Reports the compact vertex-normal count and number of degenerate triangles. |
| `HierarchyOptions` | Selects the maximum number of primitives stored in a hierarchy leaf. |
| `HierarchyNode` | Describes one breadth-first hierarchy node and its contiguous child or primitive range. |
| `HierarchyStatistics` | Reports node, leaf, branch, and maximum-depth totals for a built hierarchy. |

### Owning objects

| API | Description |
| --- | --- |
| `Workspace` | Owns reusable device scratch storage shared by geometry operations. |
| `NormalOutput` | Owns device face normals, compact vertex normals, corner-normal indices, and normal statistics. |
| `Hierarchy` | Owns breadth-first nodes, stable primitive permutation, retained refit data, and hierarchy statistics. |

### Free functions

| API | Description |
| --- | --- |
| `compute_normals` | Validates an indexed mesh and builds deterministic face, vertex, and corner-normal data on a CUDA stream. |
| `build_hierarchy(DeviceMeshView, ...)` | Builds a deterministic eight-way hierarchy over indexed mesh triangles. |
| `build_hierarchy(DeviceAabbView, ...)` | Builds the same deterministic hierarchy directly over caller-provided primitive bounds. |
| `refit_hierarchy` | Recomputes hierarchy bounds synchronously without changing its topology or primitive order. |
| `refit_hierarchy_unchecked_async` | Enqueues a bounds-only refit for already validated same-stream input and leaves completion to the caller. |
| `status_code_name` | Returns a stable textual name for a `StatusCode`. |

## Common frame protocol

| API | Description |
| --- | --- |
| `FrameOptions` | Defines a fixed physical frame interval, equal substep count, and external acceleration. |
| `SubstepContext` | Identifies one ordered substep and carries its derived timestep and acceleration. |
| `PointCouplingView` | Borrows point positions, velocities, and one writable impulse row per point across fluid and deformable solvers. |
| `Completion` | Owns a reusable CUDA event that can be polled or waited for after asynchronous submission. |
| `FrameSolver` | Describes the shared begin/prepare/couple/finish structural protocol at compile time. |
| `valid(FrameOptions)` | Validates finite positive frame timing, acceleration, and a bounded substep count. |
| `substep_context` | Derives one immutable substep context from a frame and index. |
| `step_async` | Drives an uncoupled `FrameSolver` frame and records its completion token. |
| `step` | Drives an uncoupled frame and waits for its completion token. |

## Independent particle and body solvers

| API | Description |
| --- | --- |
| `FluidParticle` | Supplies one initial particle position and velocity from application memory. |
| `FluidOptions` | Configures particle size, fixed-radius interaction, mass, pressure, viscosity, damping, and speed. |
| `FluidView` | Borrows device particle state and writable external impulses during coupling. |
| `FluidStatistics` | Reports count, neighbor/finiteness diagnostics, frame index, and retained storage. |
| `Fluid` | Owns a deterministic sorted-cell CUDA particle fluid independent of boundaries and rendering. |
| `ClothOptions` | Configures a generated rectangular spring sheet and its shared deformable solver material. |
| `Cloth` | Owns a generated cloth graph with node, bond, and deformed-surface views. |
| `RopeOptions` | Configures a generated spring rope's nodes, spacing, origin, direction, and material. |
| `Rope` | Owns a generated rope graph with the common frame and coupling protocol. |
| `RigidShape` | Selects sphere or box collider metadata for a finite-mass rigid body. |
| `RigidBodyOptions` | Configures mass, inertia, shape dimensions, damping, and velocity limits. |
| `RigidBodyState` | Stores position, quaternion orientation, and linear and angular velocities. |
| `RigidBody` | Owns finite-mass rigid dynamics and gathers forces and torques during each coupling phase. |

`Fluid`, `Cloth`, `Rope`, and `RigidBody` all provide `initialize`,
`begin_frame`, `prepare_substep`, `finish_substep`, `finish_frame`,
`advance_async`, `advance`, `reset`, and state/view access appropriate to the
solver. `Cloth` and `Rope` reuse the fixed-topology graph backend without
exposing gallery recipes.

## Soft-body physics

### Structures and constants

| API | Description |
| --- | --- |
| `node_surface` | Marks a soft-body node as belonging to the renderable exterior. |
| `node_pinned` | Marks a soft-body node as kinematic rather than dynamically integrated. |
| `Bond` | Identifies two connected nodes and their rest length. |
| `SoftBodyAssetView` | Borrows an in-memory `.msb` payload that initialization validates and copies. |
| `SoftBodyOptions` | Configures topology instances, timestep, mass, spring solving, fracture, damping, collision, hierarchy, and instance placement. |
| `SoftBodyMaterial` | Groups runtime-adjustable stiffness, damping, speed, and ground-friction values. |
| `SoftBodyNodeView` | Borrows device node state and writable impulse/correction buffers for application-defined coupling kernels. |
| `SoftBodyBondView` | Borrows device positions, flags, shared bonds, and per-instance bond activity. |
| `SoftBodySurfaceView` | Borrows the deformed surface mesh, normals, UVs, active triangles, and its hierarchy. |
| `SoftBodyTimings` | Reports solver, surface-deformation, and hierarchy GPU times. |
| `SoftBodyStatistics` | Reports instance, node, bond, fracture, finite-failure, and frame totals. |

### Owning object

| API | Description |
| --- | --- |
| `SoftBody` | Owns and advances one or more instances of a fixed-topology `.msb` spring lattice on the GPU. |

### `SoftBody` operations

| API | Description |
| --- | --- |
| `create` / `initialize` | Loads a lattice from a path or memory bytes, validates it, and initializes the requested instances. |
| `step` | Advances a complete synchronous frame with optional stage timings. |
| `advance_async` / `advance` | Drive the common frame protocol with a completion token or synchronous wait. |
| `begin_frame` | Starts the manual coupling protocol and prepares per-frame solver state. |
| `prepare_substep` | Predicts one substep and exposes cleared external impulse and position-correction buffers. |
| `finish_substep` | Consumes application coupling, solves constraints and fracture, and commits one substep. |
| `finish_frame` | Completes the manual frame, rebuilds presentation data, and reports timings. |
| `reset` | Restores initial positions, velocities, and unbroken bond activity. |
| `set_material` | Changes the runtime-adjustable material values after initialization. |
| `set_substeps` | Changes the number of solver substeps within each fixed frame. |
| `set_constraint_iterations` | Changes the Jacobi constraint passes performed per substep. |
| `set_strength_multiplier` | Scales the authored fracture strength without rebuilding topology. |
| `set_node_mass` | Changes the common dynamic mass assigned to lattice nodes. |
| `initialized` | Reports whether the object currently owns a valid simulation. |
| `options` / `material` / `node_mass` | Return the resolved creation and runtime material configuration. |
| `nodes` / `bonds` / `surface` | Return borrowed device views that must be reacquired after state-changing calls. |
| `statistics` / `allocated_bytes` | Report current solver counters and retained memory. |

## Smoke physics

### Structures

| API | Description |
| --- | --- |
| `SmokeOptions` | Configures deterministic tracer capacity, emission, lifetime, buoyancy, damping, turbulence, speed, and seed. |
| `SmokeSphereCollider` | Supplies one moving analytic sphere that deflects smoke tracers. |
| `SmokeStepInput` | Supplies external acceleration and a host array of sphere colliders for one step. |
| `SmokeParticleView` | Borrows device positions, velocities, ages, and temperatures for active smoke particles. |
| `SmokeCouplingView` | Borrows body points and a writable impulse buffer for one-way aerodynamic loading. |
| `SmokeTimings` | Reports smoke integration and body-coupling GPU times. |
| `SmokeStatistics` | Reports frame, respawn, finite-failure, maximum-speed, and allocation totals. |

### Owning object

| API | Description |
| --- | --- |
| `Smoke` | Owns a deterministic GPU tracer field with analytic advection, sphere collision, and one-way body coupling. |

### `Smoke` operations

| API | Description |
| --- | --- |
| `create` / `initialize` | Validate smoke options and allocate and seed all device-resident tracer state. |
| `step` | Advances smoke synchronously with optional colliders and timing output. |
| `advance_async` / `advance` | Drive smoke through the common frame protocol and completion-token interface. |
| `couple` | Accumulates aerodynamic impulses into a smoke-specific or common point-coupling view. |
| `reset` | Reseeds the original deterministic tracer state. |
| `initialized` | Reports whether valid smoke state has been created. |
| `options` | Returns the resolved smoke configuration. |
| `particles` | Returns the current borrowed device particle view. |
| `statistics` | Returns smoke runtime and allocation counters. |

## Example-code boundary

The recipe catalog, gallery builder, aggregate render views, objectives, and
progression live in `examples/gallery` and `apps/water_lab`. They are not
installed and are intentionally absent from this inventory.

## Remaining reuse work

1. Extract the fixed-topology graph backend from the current water-lab source
   directory so the installed implementation has no app-directory dependency.
2. Replace smoke's internal timing readbacks and the mature soft-body frame
   readback with truly enqueue-only variants; their completion tokens currently
   preserve the protocol but cannot recover synchronization already performed
   inside those backends.
3. Add generic collider batches and balanced reaction gathering so common
   fluid/deformable/rigid contacts do not require application kernels.
4. Split the physics umbrella declarations into implementation-independent
   focused headers once the ABI settles; the focused headers currently forward
   to the common umbrella to guarantee identical view definitions.
5. Standardize all diagnostics as device-resident snapshots associated with a
   completion token, avoiding races when statistics are queried too early.
