# Simulation API architecture

ParallelMater separates installed CUDA ownership from authored examples.

## Package layers

1. `ParallelMater::geometry` provides deterministic normals, hierarchy
   construction/refitting, and reusable allocation owners.
2. `ParallelMater::physics` provides `Fluid`, `Cloth`, `Rope`, `SoftBody`,
   `RigidBody`, and `Smoke`, plus the shared frame, collider, and constraint
   interfaces.
3. `parallel-mater-example-recipes` contains names and configuration for the 15
   example compositions.
4. `parallel-mater-example-gallery` owns and composes the installed public
   solvers. It has no dependency on `apps/water_lab`.
5. `parallel-mater-native-simulations` preserves advanced authored CUDA
   experiments for the optional interactive lab. It is app-only, not installed,
   and may use specialized fixtures unavailable in the general API.
6. `parallel-mater-lab` supplies GLFW input, rendering, controls, objectives,
   captures, and progression.

The reusable library never includes files from `apps/` or `examples/`. The
private fixed-topology backend contains only graph simulation, asset loading,
surface deformation, normal generation, and hierarchy refitting. Arenas,
courses, bridge dimensions, fractured-triangle presentation, and internal
member visualization remain in app-only code.

## Ownership and stepping

Every public solver is an independent movable owner. A coupled frame is:

1. `begin_frame(frame, stream)` on every owner;
2. `prepare_substep(context, stream)` on every owner;
3. application coupling through borrowed `PointCouplingView` buffers,
   `ColliderView`, and/or `ConstraintBatch`;
4. `finish_substep(context, stream)` on every owner; and
5. `finish_frame(completion, stream)` followed by an application-selected
   completion boundary.

`advance_async` and `advance` are convenience operations for uncoupled frames.
Telemetry has a separate request/resolve lifecycle, so asynchronous advancement
does not perform diagnostic downloads.

## Generic coupling

`ColliderSet` owns device copies of solver-neutral sphere, box, plane, and
capsule descriptions. Individual solvers document which shapes they consume.

`ConstraintBatch` carries one or more stable `(point, order)` contributions,
sorts them deterministically, and gathers impulses and position corrections
into any compatible point owner without floating-point contact atomics.

The headless gallery exercises all six owners, `ColliderSet`, and
`ConstraintBatch`. It intentionally keeps its presets and recipe selection out
of the install tree.

## Current boundary

The native lab retains app-only implementations for its historical water-skin,
fracture-presentation, wheel, cage, and minigame behavior. Those implementations
are isolated in `apps/water_lab` and do not define or extend the installed API.
New reusable behavior should enter a public solver or a generic coupling type;
new authored behavior should remain in the example layer.
