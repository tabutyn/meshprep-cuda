# Public API inventory

This first-pass inventory covers the installed headers in `parallel_mater/` and
describes each remaining public type and free function in one sentence; the
native gallery's objectives and progression are intentionally excluded because
they are application code.

As of this pass the installed headers contain 5 enums, 36 structures, 7 owning
classes, 13 free functions, and 118 explicit callable declarations when class
operations are included; the implementation contains 71 private CUDA kernels.

## Geometry

### Enums

| API | Description |
| --- | --- |
| `StatusCode` | Categorizes success, invalid input, unsupported size, allocation, CUDA, and internal failures without throwing exceptions. |

### Structures

| API | Description |
| --- | --- |
| `Status` | Returns a library status code, optional CUDA error, and static diagnostic message from synchronous operations. |
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

## Soft-body physics

### Structures and constants

| API | Description |
| --- | --- |
| `node_surface` | Marks a soft-body node as belonging to the renderable exterior. |
| `node_pinned` | Marks a soft-body node as kinematic rather than dynamically integrated. |
| `Bond` | Identifies two connected nodes and their rest length. |
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
| `create` / `initialize` | Loads and validates a lattice asset, allocates device state, and initializes the requested instances. |
| `step` | Advances a complete synchronous frame with optional stage timings. |
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
| `couple` | Accumulates aerodynamic impulses into a caller-provided body view. |
| `reset` | Reseeds the original deterministic tracer state. |
| `initialized` | Reports whether valid smoke state has been created. |
| `options` | Returns the resolved smoke configuration. |
| `particles` | Returns the current borrowed device particle view. |
| `statistics` | Returns smoke runtime and allocation counters. |

## Optional gallery recipes

These declarations configure the shipped demonstrations and are not required by
the geometry, soft-body, or smoke libraries.

### Enums, structures, and constants

| API | Description |
| --- | --- |
| `Component` | Identifies solver capabilities used by an example recipe. |
| `SimulationRecipe` | Selects one of the fifteen shipped solver-composition demonstrations. |
| `SimulationRecipeInfo` | Associates a recipe with its stable slug, title, and component flags. |
| `simulation_recipes` | Defines the ordered compile-time catalog used by gallery clients. |
| `RecipeConfig` | Stores portable overrides for a selected gallery recipe and provides fluent setters. |
| `RecipeConfigError` | Categorizes invalid recipe, timestep, iteration, count, and detail overrides. |
| `operator|` | Combines two `Component` flags into one component set. |
| `has_component` | Tests whether a component set contains a requested capability. |
| `find_simulation_recipe` | Finds a catalog entry by stable textual slug. |
| `validate_recipe_config` | Validates the portable recipe configuration without allocating GPU state. |
| `recipe_config_error_message` | Converts a recipe validation error into a static diagnostic message. |

## Optional gallery simulation

### Enums, structures, and constants

| API | Description |
| --- | --- |
| `ParticleMaterial` | Labels a particle render view as fluid, smoke, or steam. |
| `ParticleRenderView` | Borrows one device particle system for renderer consumption. |
| `SurfaceRenderView` | Borrows one indexed surface, vertex normals, UVs, and active-triangle mask. |
| `RigidBodyRenderView` | Describes one borrowed rigid mesh and its translation, quaternion, and scale. |
| `lattice_node_surface` | Marks a gallery lattice node as exterior. |
| `lattice_node_pinned` | Marks a gallery lattice node as kinematic. |
| `LatticeBond` | Identifies a gallery lattice connection and its rest length. |
| `LatticeRenderView` | Borrows device lattice nodes, flags, shared bonds, and per-instance activity. |
| `FrameRenderView` | Aggregates all borrowed particle, surface, rigid-body, and lattice views for one frame. |
| `FixedStepOptions` | Stores the physical timestep used by a gallery simulation. |
| `GallerySimulationOptions` | Selects a recipe and optional solver, gravity, count, resolution, grid, and asset overrides. |
| `ResolvedPhysicsOptions` | Reports the timestep, solver iterations, gravity, and rigid-course preset actually selected. |
| `GallerySimulationStatistics` | Reports frame, particle, surface, rigid-body, failure, fracture, timing, and allocation totals. |
| `valid(FixedStepOptions)` | Checks that a timestep is finite, positive, and representable. |
| `requires_soft_body_asset` | Reports whether a recipe needs an external `.msb` asset at initialization. |

### Owning objects

| API | Description |
| --- | --- |
| `GallerySimulation` | Owns and synchronously steps one complete shipped recipe without owning a renderer or window. |
| `SimulationBuilder` | Converts fluent recipe overrides into validated `GallerySimulationOptions` and initializes a gallery simulation. |

### `GallerySimulation` operations

| API | Description |
| --- | --- |
| `create` / `initialize` | Build all solver state required by a selected gallery recipe. |
| `step` | Advance exactly one fixed gallery timestep. |
| `reset` | Restore the selected recipe to its authored initial state. |
| `resize_particles` | Change the active deterministic particle prefix within reserved capacity. |
| `initialized` | Report whether the gallery owns a valid recipe instance. |
| `recipe` / `options` / `resolved_physics` | Return the selected recipe and its requested and resolved configuration. |
| `statistics` | Return aggregate gallery runtime counters. |
| `render_view` | Return borrowed device views for presentation by an external renderer. |

### `SimulationBuilder` operations

| API | Description |
| --- | --- |
| `timestep` / `iterations` | Override the recipe's fixed step and solver iteration count. |
| `particles` / `skin_frequency` / `rope_nodes` / `cloth_detail` | Override primary simulation quantities for recipes that use those systems. |
| `bridge_grid` / `cylinder_grid` | Override procedural bridge and cylinder-field dimensions. |
| `gravity` | Override the recipe's gravity vector. |
| `soft_body_asset` | Supply the `.msb` asset path required by asset-backed recipes. |
| `config` | Return the portable recipe configuration accumulated by the builder. |
| `build` | Validate all overrides and initialize the destination gallery simulation. |

## Reusability findings

1. `SimulationRecipe`, `Component`, `RecipeConfig`, and `GallerySimulation` still
   encode the sample catalog; they should eventually move to an examples-only
   package or become non-installed targets.
2. Fluid, cloth, rope, and rigid-body behavior exists behind
   `GallerySimulation` but lacks independent owning APIs comparable to
   `SoftBody` and `Smoke`; extracting those solvers is the largest step toward
   arbitrary application composition.
3. All solvers should converge on one explicit frame/substep/coupling protocol
   so applications can exchange impulses without solver-specific orchestration.
4. Asset-backed `SoftBody` initialization should accept a validated byte view
   as well as a filesystem path so packaged applications control their own I/O.
5. The project should use a real `parallel_mater` namespace instead of exposing
   `meshprep` through a namespace alias before promising source compatibility.
6. The synchronous API is simple but limits overlap; a future asynchronous
   layer should return a completion token while retaining the current safe
   synchronous convenience calls.
7. Device views should use consistent naming and ownership vocabulary, with
   explicit validity rules and common span-like count/pointer conventions.
8. Smoke coupling is one-way and collider support is sphere-specific; generic
   collider views and balanced force exchange would make it suitable for more
   than gallery effects.
