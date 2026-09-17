# Continuous water and independent surface foam

`V` has exactly one alternate state in every context: filled material surfaces
or the combined particle/lattice view. The combined view shows water particles,
water-skin and soft-body wires, all internal voxels, and the lattice graph.
Broken bonds are red, but their adjacent material triangles remain visible.
Interior voxels are green,
surface voxels yellow, and pinned voxels blue. Context 1 uses continuous water
in its normal view; other contexts show their declared particles and filled
cloth/soft-body surfaces. `F` toggles foam.
The `--view particles` CLI option remains available for isolated renderer
profiling, but it is not a keyboard view. Existing physics, course controls,
skin, and fixed physical time are unchanged.

## Representation

- `apps/water_lab/fluid_surface.cu`: a bounded 48³ world-space grid follows the
  particle hierarchy bounds. At each sample, the existing hierarchy gathers
  nearby particles with compact weight `(1-r²/h²)³`. A Zhu–Bridson-style scalar
  measures distance to their weighted mean minus `h/3`; no contribution means
  exterior. The zero level joins neighboring samples into a continuous volume.
- `fluid_surface.cuh`: trilinear scalar interpolation and its analytic gradient,
  shared by rendering, foam projection, and reference tests. This scalar is
  **not a signed-distance bound** and must not be blindly sphere-traced.
- `fluid_visuals.cu`: particle-distribution normals and local relative-velocity
  variation identify upward-facing exposed samples. A sparse calm-water source
  keeps settled bowl foam visible, while local agitation increases emission;
  uniform translation alone does not count as agitation. A separate,
  deterministic 2,048-slot tracer
  pool emits small bubbles, carries them with nearby fluid velocity, projects
  them back onto the scalar surface, and retires expired/lost tracers. Bubbles
  have independent position, velocity, normal, radius, age, and lifetime.
- `water_kernels.cu`: grid traversal finds the continuous water boundary and
  brackets each cell's cubic ray restriction at derivative extrema before
  finding its first zero. Centered-difference shading normals reduce grid
  faceting; they do not change intersections or the foam attachment gradient.
  A separate foam hierarchy supplies animated local patches. Each live tracer
  deterministically instances eleven irregular analytic bubble cells in its
  current tangent plane. Cells spread, grow, and pop at staggered normalized-age
  phases; translucent centers and narrow thin-film rims preserve the water below.
  The existing opaque course and translucent outer skin participate in composition.
- `main.cpp`: the `WATER FOAM` timing includes surface reconstruction and foam
  update/hierarchy cost. GPU TOTAL also includes this work. Foam evolves while
  hidden so toggling display cannot change history; pause freezes it.

All persistent buffers are bounded and reused. This replaces the old enlarged
sphere surface and coverage-whitening branch; those are not retained as a
fourth renderer. There is no new fluid force, spring, contact solver, or water
mass transfer.

## Recording

`R` clears foam. New `M` captures use version 5. Foam records and their
deterministic emission tick retain the version-3 layout; versions 4 and 5 add
soft-body state and pending fracture damage. Replay rebuilds the derived scalar
field from each restored physics frame, then restores that frame's foam
exactly. Seeking never integrates physics or invents missing foam history.

Versions 1 and 2 still load. They have no independent tracer history and replay
with no new foam; the old v2 white-coating effect is intentionally not retained.
Version 3 adds 98,312 bytes per frame over v2 for the 2,048 records and tick
(about 35.4 MB across the 360-frame rolling host buffer). Physics payloads and
the build-specific format checks remain unchanged.

## Scope

This is a rendering reconstruction, not a new CFD solver. The closed physical
skin still controls the rollable body's shape and prevents free splashes from
detaching. Foam is a one-way visual surface tracer, not an air phase, airborne
spray, or simulated bubble collisions. A finite 48³ grid limits fine features
and thin sheets; sparse, separated water samples can still produce separated
drops. Surface projection may retire tracers that lose the water surface.
The inexpensive optical model uses one refracted interface and a bounded
absorption approximation; it does not solve multiple internal reflections or
physically validated light transport through foam.

## Why the clips are native rather than Blender renders

Blender 4.5.2 LTS is installed and was evaluated as an offline foam authoring
tool. A rendered sprite atlas would require a new texture/UV pipeline, would
carry baked lighting, and would flatten under the orbital camera. Animated mesh
caches would add loading, changing geometry, and another hierarchy workload.

The retained prototype therefore uses the same useful division as an authored
clip without importing an asset system: CUDA owns each live surface anchor, and
normalized tracer age selects a deterministic local 3D spread/pop sequence.
It remains responsive to current impacts, pauses exactly, survives capture and
replay without a new file format, and works from every camera angle. A future
Blender-authored clip can replace the procedural local offsets/radii while using
the same anchor, age, analytic-bubble, and hierarchy contract; world-space
Blender trajectories should not replace live advection.

## Reproduce

```bash
cmake --build build --target meshprep-water-lab meshprep-fluid-visual-tests meshprep-hybrid-tests -j4
ctest --test-dir build --output-on-failure
./build/meshprep-water-lab --view surface --foam on --profile 120 --warmups 10
./build/meshprep-water-lab --view particles --foam off --profile 120 --warmups 10
./build/meshprep-water-lab --view wire --foam off --profile 120 --warmups 10
```

## Local results — 2026-09-13

The subsequent spatial-query optimization reduced the continuous-surface plus
foam update from 5.3049 to 2.2889 ms median and complete GPU time from 14.4568
to 11.4546 ms. See [WATER_FOAM_OPTIMIZATION.md](WATER_FOAM_OPTIMIZATION.md) for
the repeated measurements and rejected experiments. The table below records the
earlier visual-quality baseline.

RTX 3050 Ti Laptop, CUDA 13.1.80, Release/SM86, 960×720, 10 warmups and
120 measured frames. No profiler, GUI/vsync, or rolling capture was running.
Times are elapsed CUDA-event stages and wall time, not isolated kernel busy time.

| View | Field + foam median ms | Ray trace median ms | Wall median / p95 / max ms |
|---|---:|---:|---:|
| Continuous water + animated foam patches | 5.385 | 2.835 | 22.899 / 28.600 / 30.122 |
| Continuous water, foam hidden | 5.348 | 2.367 | 22.320 / 28.194 / 29.580 |

Both runs update identical field/foam history even when display is hidden. The
visual owner reports 1,447,796 resident bytes, including its grid, foam pool,
hierarchy, workspace, and particle-normal/source data. Both ended at frame 130
with 45 active tracers and identical physics summaries. Enabling the patches
added 0.467 ms to median ray tracing and 0.579 ms to median frame wall time in
these two sequential runs. A separate 360-frame settling/tilting sequence had
46 / 32 / 80 active tracers at frames 120 / 240 / 360, with zero reported
non-finite state. Close-up before/after frames were visually inspected; the
initial regular flower layout was rejected in favor of independently sized,
hashed cells with staggered birth and disappearance.

The immediately preceding single-cap foam renderer measured 2.392 ms ray trace
and 22.668 ms wall median with the same startup settings. The animated patches
are a visual-quality change with additional cost, **not a performance
improvement**; the measured complete frame does not meet 60 FPS. Interactive
capture/window overhead is additional, and these startup timings do not bound
every impact.

## Validation

All thirteen CTests pass. Tests cover a connected bridge
between separated individual particle spheres, a CPU weighted-field reference,
analytic interpolation/gradients, cubic tangencies and hidden crossings,
no static/uniform-motion foam, agitation-driven emission, finite lifetime,
surface attachment, deterministic reset, exact pause/restore, and unchanged
physics buffers. The energetic fixture produces 23 active tracers after eight
frames; every active tracer passes the geometric attachment tolerance.

Renderer tests distinguish all views, compose an independent foam patch through
the transparent skin, and completely occlude water/foam behind a course peg.
Close-up renders were inspected after settling and tilting. A nonzero-foam v3
file roundtrip/restoration was exact; existing v1 (360 frames) and v2 (2 frames)
captures also loaded and restored.

The earlier surface/foam manager passed all four Compute Sanitizer tools. The
final animated-patch renderer also passed targeted memcheck with zero errors.
This is scoped visual-feature coverage, not a new full-program four-tool audit.

```bash
for tool in memcheck racecheck initcheck synccheck; do
  /usr/local/cuda/bin/compute-sanitizer --tool "$tool" --error-exitcode 99 \
    ./build/meshprep-fluid-visual-tests
done
MESHPREP_VISUAL_SANITIZER_SMOKE=1 /usr/local/cuda/bin/compute-sanitizer \
  --tool memcheck --kernel-name kns=render_fluid_kernel --error-exitcode 99 \
  ./build/meshprep-hybrid-tests
```

Temporary logs, screenshots, and the capture-compatibility probe for this
iteration are under `/tmp/meshprep-continuous-water-kOUI2f/`. After review,
cleanup only that exact directory:

```bash
rm -r -- /tmp/meshprep-continuous-water-kOUI2f
```
