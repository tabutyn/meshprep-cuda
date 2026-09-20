# Smoke API

`ParallelMater::physics::Smoke` is a deterministic CUDA tracer and
aerodynamic-coupling component. It is intentionally smaller than a CFD smoke
solver: applications get a visible advected stream, analytic buoyancy and
turbulence, finite lifetimes with deterministic respawn, sphere obstacles, and
a device-to-device force interface.

```cpp
#include <parallel_mater/smoke.hpp>

parallel_mater::physics::SmokeOptions options;
options.particle_count = 6'000;
options.emitter_center = {-2.0F, 0.0F, 0.0F};
options.initial_velocity = {3.0F, 0.0F, 0.0F};

parallel_mater::physics::Smoke smoke;
auto status = smoke.initialize(options, stream);

parallel_mater::physics::SmokeTimings timings;
if (status) status = smoke.step({}, timings, stream);

auto tracers = smoke.particles(); // borrowed device pointers
```

To load a cloth, rope, soft body, or another application-owned point system,
pass a borrowed `SmokeCouplingView` to `couple()`. The solver samples the same
analytic carrier field used by the visible tracers and adds impulses to the
caller's buffer. It never owns or assumes the topology of the receiving body.

```cpp
parallel_mater::physics::SmokeCouplingView body{
    positions,
    velocities,
    external_impulses,
    node_count,
    inverse_node_mass,
    node_radius,
    substep_dt,
};
status = smoke.couple(body, 5.0F, timings, stream);
```

The coupling pass is linear in receiving points. It does not compare every
body point with every visual tracer. Each receiving point owns one output row,
so coupling requires no floating-point atomics and has a fixed same-GPU write
order.

## Ownership and synchronization

- `Smoke` owns its CUDA allocations and is movable but not copyable.
- `SmokeParticleView` and all input coupling views are borrowed.
- Reacquire particle views after `initialize()` or `reset()`.
- Legacy `step` and `couple` calls are synchronous; the common frame protocol
  returns a `Completion` token after its internal statistics readback.
- `external_impulses` is accumulated, not cleared or replaced.
- A `SmokeCouplingView::timestep` is the receiving solver's substep; smoke
  advection keeps its own fixed `SmokeOptions::timestep`.

## Example-gallery validation

The optional gallery composes the same API in five contexts:

| Composition | Runtime assertion |
| --- | --- |
| Smoke + rigid sphere | sphere translates and rotates through the stream |
| Fluid + smoke | water prefix shrinks while buoyant steam is emitted over a hot pan |
| Cloth + smoke | pitched cloth blades transfer smoke load into damped rotor torque |
| Soft body + smoke | green bristles receive wind impulses while a rigid sphere rolls through |
| Rope + smoke | segmented rope bridge receives wind and rigid-sphere contact |

`meshprep-smoke-tests` validates finite advection, respawn, buoyancy, obstacle
wake, and non-zero coupling. `meshprep-simulation-runtime-tests` initializes
and steps every example composition against the reusable solver APIs and checks
the dynamic assertions above. The native app uses `Tab` to open the collapsible
context catalog.

## Current scope

The tracer field is not incompressible Navier–Stokes, does not voxelize arbitrary
colliders, and does not receive equal-and-opposite feedback from coupled bodies.
Those are explicit future extensions rather than hidden behavior in the gallery.
