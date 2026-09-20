# Getting started with ParallelMater physics

Start here after the root README. ParallelMater is a set of independent CUDA
owners, not a scene or game framework: initialize only the systems an
application needs, advance them with one `FrameOptions`, and insert contact work
between `prepare_substep` and `finish_substep`.

## One system

`examples/solvers.cu` is the shortest source tour of the owning `Fluid`,
`Cloth`, `Rope`, and `RigidBody` types. For an uncoupled owner:

```cpp
parallel_mater::physics::Fluid fluid;
auto status = fluid.initialize(initial_particles);
if (status) {
    status = fluid.advance({1.0F / 60.0F, 4U, {0.0F, -9.81F, 0.0F}}, stream);
}
```

The returned views contain borrowed device pointers. Reacquire them after
initialization, reset, or any operation documented as replacing storage.

## Two or more systems

All owners implement the same frame protocol. `advance_coupled` prepares every
owner, invokes one callback for contacts, then finishes every owner. If the
callback or a solver rejects a stage, active frame state is abandoned so the
owners can be used again. The error path drains the supplied stream; GPU work
already enqueued before the error is not rolled back.

```cpp
using namespace parallel_mater::physics;

auto status = advance_coupled(
    FrameOptions{1.0F / 60.0F, 4U, gravity},
    [&](SubstepContext step, cudaStream_t stream) noexcept {
        auto result = apply_colliders_async(fluid.point_state(), colliders.view(),
                                            step.timestep, stream);
        if (result) {
            result = apply_colliders_async(cloth.point_state(), colliders.view(),
                                           step.timestep, stream);
        }
        return result;
    },
    stream, fluid, cloth);
```

Use `ConstraintBatch` when a custom contact generator emits multiple
contributions for the same point. It stable-sorts by `(point, order)` before
gathering, so results do not depend on CUDA scheduling.

## Contact painting

`Fluid`, `Cloth`, `Rope`, `SoftBody`, and `Smoke` own one persistent linear RGBA
color per point and expose it through `point_state()`. A `ConstraintRecord` can
apply paint with the same deterministic gather that applies its impulse and
position correction. `Collider::paint_color` and `paint_amount` make
`apply_colliders_async` transfer paint whenever a point contacts a sphere, box,
plane, or capsule.

Deformable surface colors are interpolated from node colors and exposed in
`SoftBodySurfaceView::vertex_colors`. For persistent renderer textures—cloth
UVs, a rigid sphere atlas, or a box atlas—use `PaintSurface` and submit ordered
device `PaintStamp` records. The application owns the mapping from a contact to
UV coordinates; the physics API owns deterministic persistent blending.

## Existing examples are the integration suite

The repository does not replace its work with two artificial reference apps.
The existing recipe gallery composes all public owners, and the native lab keeps
the richer authored contexts:

1. `examples/gallery/recipes.hpp` — composition catalog;
2. `examples/gallery/gallery.cpp` — public owners, common stepping, analytic
   contacts, point paint, and renderer-neutral views;
3. `tests/simulation_runtime_tests.cu` — initializes and steps every gallery
   recipe;
4. `apps/water_lab/main.cpp` — optional interactive CUDA/OpenGL client.

Continue with [API.md](API.md) for contracts, [API_INVENTORY.md](API_INVENTORY.md)
for the complete public surface, and
[SIMULATION_API_ARCHITECTURE.md](SIMULATION_API_ARCHITECTURE.md) for package
boundaries.
