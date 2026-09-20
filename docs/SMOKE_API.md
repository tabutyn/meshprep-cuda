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

parallel_mater::physics::ColliderSet colliders;
parallel_mater::physics::Collider sphere;
sphere.position = {0.0F, 0.2F, 0.0F};
sphere.dimensions = {0.35F, 0.0F, 0.0F};
if (status) status = colliders.update(
    std::span<const parallel_mater::physics::Collider>(&sphere, 1), stream);
if (status) status = smoke.step({}, colliders.view(), stream);
if (status) status = smoke.collect_telemetry(stream);

auto tracers = smoke.particles(); // borrowed device pointers
auto timings = smoke.telemetry().timings;
```

To load a cloth, rope, soft body, or another application-owned point system,
pass a borrowed `PointCouplingView` to `couple()`. The solver samples the same
analytic carrier field used by the visible tracers and adds impulses to the
caller's buffer. It never owns or assumes the topology of the receiving body.

```cpp
parallel_mater::physics::PointCouplingView body{
    positions,
    velocities,
    external_impulses,
    position_corrections,
    node_count,
    node_radius,
    inverse_node_mass,
};
status = smoke.couple(body, substep_dt, 5.0F, stream);
```

The coupling pass is linear in receiving points. It does not compare every
body point with every visual tracer. Each receiving point owns one output row,
so coupling requires no floating-point atomics and has a fixed same-GPU write
order.

## Ownership and synchronization

- `Smoke` owns its CUDA allocations and is movable but not copyable.
- `SmokeParticleView` and all input coupling views are borrowed.
- Reacquire particle views after `initialize()` or `reset()`.
- `step`, `advance`, and `couple` never download telemetry; asynchronous frame
  submission records `Completion` after queued simulation work.
- `collect_telemetry_async` performs the optional readback under its own token;
  `collect_telemetry` is the explicit synchronous convenience form.
- `external_impulses` is accumulated, not cleared or replaced.
- The coupling timestep is the receiving solver's substep; smoke advection
  keeps its own fixed `SmokeOptions::timestep`.

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
