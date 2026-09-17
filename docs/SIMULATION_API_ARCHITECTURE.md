# Simulation API architecture

The installable `meshprep-cuda` target is currently a focused geometry library.
The water, cloth, and soft-body work should become a second layer rather than
expanding the hierarchy builder into one application-sized class.

## Installed headless boundary

`<meshprep/simulation.hpp>` provides the data-facing contract shared by the
interactive examples:

- composable component flags and seven numbered example recipes;
- borrowed device views for particles, triangle surfaces, and rigid bodies;
- a frame render view that does not own or copy CUDA memory; and
- fixed-step options plus explicit optional gravity/iteration overrides,
  independent of a window or renderer; and
- movable PIMPL ownership with `Status`-returning initialize, fixed-step, and
  reset operations.

The `meshprep::meshprep-simulation` target packages the backend without GLFW,
OpenGL, the ray tracer, HUD, or capture code. `examples/simulation_contexts.cu`
initializes and advances a real context-2 CUDA frame, and the native gallery
can launch recipes with `--context 1..7`. Device views are deliberately
read-only: a solver owns its allocations and a renderer reacquires the view
after each step.

The adapted solvers use one internal typed exception carrying a complete
`meshprep::Status`. `GallerySimulation` catches it at its noexcept package
boundary, preserving allocation-versus-CUDA classification and the concrete
`cudaError_t` without parsing diagnostic strings. Truly unexpected exceptions
remain `internal_error` with `cudaSuccess`.

Context 4 requires a caller-provided converted cylinder `.msb` path. Context
1's rigid course, the bowl, rigid spheres, cloth, water wheel and soft cross,
soft sphere, and context-8 rope are procedural.

The native app and installed PIMPL resolve recipes through one compact
app-independent factory. Context 1 keeps the course preset: `1/60 s`, four
solver iterations, and gravity `(0, -7.2, 0)`. Contexts 2–8 use a separate
Earth-gravity gallery preset `(0, -9.81, 0)` and four iterations; context 2
uses twice-Earth gravity. Context 7 starts under the same vertical Earth
gravity and rolls only after camera-relative steering supplies a horizontal
component. The factory also controls
the analytic course-collider flag, so a recipe that does not declare rigid
bodies cannot accidentally inherit invisible floor or rail collision.

`GallerySimulationOptions` does not use sentinel gravity or iteration values.
Its gravity, solver-iteration, particle-count, physical-skin-frequency, and rope-node-count
optionals are empty by default; assigning any is an explicit override. Active
particle count may also change after initialization within the capacity reserved
by the recipe or explicit count override, up to 100,000 IDs. `resolved_physics()` lets a
consumer log the exact resolved values. Contract tests compare these values
exactly with the native factory for every recipe, and runtime smoke tests step
every recipe repeatedly.

The gallery recipes are executable integration examples, not eight independent
production solver implementations. Context 2 disables particle/skin force
coupling and uses an analytic hemispherical bowl, 15,000 active particles, and
a finite-mass rigid sphere. Contexts 5 and 6 use direct deterministically gathered particle-to-cloth/cross
contact. Context 5 selects the uppermost live cloth triangle at each particle's
horizontal coordinate before its one-sided catch. Its continuous 616-node
sheet has nine distributed supports and a separate four-board rigid perimeter;
a previous-frame particle already below that surface is not
projected back through it. Context 6 uses separate 10-degree inlet and collector
ramps, a finite-inertia eight-fin wheel, a lower-left 2.5-unit quarter shell
with inlet and bottom openings, and two connected 1,155-node volumetric soft
crosses pinned at their central axle volumes and outer-rim tips. A finite
one-way barrier begins repelling approaching water before fin overlap; its
equal-and-opposite impulse reduces to signed wheel torque. Context 6 recycles
particles past the lower collector back to authored emitter positions without
changing IDs or reallocating. Cloth and voxel soft bodies share the fixed graph
solver. Context 7 merges a procedural 1,000-node free soft sphere, 576-node
hanging cloth, and 1,600-node pinned-perimeter ground cloth; soft nodes query
cloth triangles, then reactions are gathered by cloth nodes. Contact preserves
the source node's side during a swept crossing, so the hanging sheet is a
two-sided barrier. The scene starts under vertical gravity. Bounded Coulomb
friction is applied after graph projection, so it also generates rolling torque
while the user applies a horizontal gravity component.
Contexts 3–5 share one closed-volume analytic room, including its
floor and ceiling; context 5 additionally owns an invisible, ceiling-height
local particle perimeter aligned to the cloth edge. Context 3 fixes
its cloth's entire top and bottom rows, buries the lower anchors one spacing
below the floor so their collision radii do not form a hidden bar, and gives a
heavy sphere a longer approach.
Contexts 3 and 4 roll one finite-mass rigid sphere through
hanging cloth and a checkerboard soft post, respectively. These are demos, not
a general rigid/contact API.
Context 8 demonstrates the same lattice and rigid-body boundaries with a
mass-weighted bilateral endpoint constraint. Rope resolution is an explicit
8–512 node option; material stiffness, damping, iteration count, gravity, and
rigid mass remain independent physics controls.
The anchored post and course recipes use 16 fixed-topology Jacobi graph iterations per
substep. A 900-step gravity-only control exposed delayed collapse with eight;
the 16-iteration control retained its height without breaking bonds. This
solver setting is deliberately distinct from the gallery's visible physics
substep count.

`HybridDroplet` still owns unused physical-water-skin buffers in particle-only
contexts. The native timing HUD labels this residual cost `ADAPTER SKIN`, so
it cannot be mistaken for particle physics. Removing those allocations requires
extracting `ParticleSystem`, rather than lying about current memory or timing.

## Current code audit

The installed PIMPL currently adapts these existing implementations:

- `SoftBodyAsset`, its versioned loader/validator, and `SoftBodyCourse` from
  `apps/water_lab/soft_body.hpp`. `SoftBodyCourse` already uses a PIMPL and is
  the closest component to a distributable class.
- `ParticleCellView` and the deterministic particle-cell builder. They are a
  reusable broad-phase service rather than water-specific policy.
- `FluidSurface` as an optional surface reconstruction component.
- common fixed-step state, timings, and device views extracted from
  `HybridDroplet`.
- analytic contact primitives and deterministic gathered contact responses,
  once they no longer depend on a specific course layout.

The following should remain sample/application code:

- GLFW input, HUD text, camera behavior, screenshots, captures, and playback;
- `RayTracer` until it is separated from the water-lab scene composition;
- numbered-key routing and example-specific initial placement;
- the tilt-course floor, rails, goal, peg layout, and gameplay tuning presets;
- diagnostics that synchronously download whole simulations for experiments.

`HybridDroplet` is not ready to publish as the general API. It currently owns
fluid particles, physical and render skins, a rectangle body, course policy,
soft-body coupling, diagnostics, hierarchy workspaces, and timing events. Its
public header also exposes most private GPU allocations. Making this class the
API would freeze the current coupling mistakes into the package.

## Smallest concrete refactor

1. Create one internal `DeviceWorld` that owns streams, workspaces, and typed
   component arrays. Components register views; they do not own one another.
2. Extract three independent fixed-step components: `ParticleSystem`,
   `TriangleCloth`, and `SoftBody`. Each exposes `prepare`, `solve`, `finish`,
   and a read-only render view.
3. Extract one batched contact stage. It accepts component views plus collider
   views and deterministically gathers equal-and-opposite responses.
4. Build contexts by composition. The seven examples should differ only in
   component creation and initial conditions, not by branching inside kernels.
5. Keep the CUDA ray tracer in a future optional `meshprep-render-cuda` target
   so headless users do not inherit rendering.
6. Convert each numbered context into a small example executable or shared
   setup function. The interactive gallery currently composes them from the
   public recipe table; the component factories remain app-local until their
   solver contracts are ready to support.

The current experimental milestone publishes context 2 as particles, a
procedural rigid hemispherical bowl, and a dynamic sphere, exercising hierarchy
rebuild, broad phase, two-way contact, fixed stepping, installation, and an
external package consumer.

## Package and compatibility rules

- Retain `meshprep::meshprep-cuda` and its v0.1 contracts unchanged.
- Treat the simulation layer as experimental until all seven contexts pass
  sustained correctness and sanitizer tests.
- Keep CUDA pointers borrowed and device-resident. Existing timing/statistics
  readbacks make steps synchronous and are documented rather than hidden.
- Use PIMPL for owning public C++ classes. Do not expose CUB storage, event
  arrays, or concrete allocation strategies in installed headers.
- Return `meshprep::Status` at package boundaries; exceptions may be used by
  sample asset-loading code but not by per-frame solver operations.
- Version serialized assets independently from the C++ ABI.
- Install examples and test a fresh `find_package(meshprep-cuda CONFIG
  REQUIRED)` consumer after every public-header change.

All project source remains covered by the repository's MIT license. Converted
assets need their own provenance record; the API must not imply that input GLB
content is relicensed.
