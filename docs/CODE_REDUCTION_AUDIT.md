# Code reduction audit

This audit distinguishes reusable library code from the historical native lab.
The installed physics implementation is about 4,800 lines; the four largest
native-lab translation units alone are about 12,600 lines. The highest-value
reduction is therefore migration and deletion at the app boundary, not making
public owners cryptic.

## Highest-value deletions

1. **Retire the second simulation stack context by context.**
   `apps/water_lab/hybrid_kernels.cu` (2,376 lines) and
   `apps/water_lab/soft_body.cu` (3,368 lines) overlap the installed fluid,
   deformable, collider, and constraint concepts. Keep each authored context as
   a test, migrate its state to public owners, then delete only the legacy code
   no longer referenced. This is the only path likely to remove thousands of
   lines without hiding complexity.

2. **Move native paint to the public paint primitives.**
   `apps/water_lab/water_kernels.cu` still owns separate bowl, sphere, goal
   cloth, and bridge masks, device globals, update kernels, and synchronous
   count readbacks. `PointStateView` contact paint and `PaintSurface` now cover
   the reusable mechanism. The app should retain only contact-to-UV mappings and
   objective coverage calculation; the four storage/update implementations can
   then be removed.

3. **Choose one recipe composition layer.**
   `apps/water_lab/simulation_gallery.cpp` (669 lines) and
   `examples/gallery/gallery.cpp` (now roughly 630 lines) both translate recipe
   names into systems. The former should become native presentation/controller
   policy over the latter, not another physics constructor. Recipe-specific
   rendering can remain in the app.

4. **Delete compatibility timing entry points before the first stable physics
   tag.** `SoftBodyOptions::{timestep,substeps}`, `SmokeOptions::timestep`, the
   older `step(gravity, timings)` overloads, and `set_substeps` duplicate
   `FrameOptions`. They remain for current native/example callers but should be
   migrated and removed together so time has one source of truth.

## Mechanical duplication

- CUDA vector helpers (`add`, `subtract`, `multiply`, `dot`) are independently
  defined in 10–13 source groups. A small private `src/internal/device_math.cuh`
  would remove repetition without adding public API.
- `invalid`/`cuda_status` translation appears in eight implementation groups.
  A private status helper would centralize error classification and wording.
- `Cloth` and `Rope` repeat about 90 lines of forwarding methods. Both already
  use one `FixedTopology` implementation; a private forwarding base/helper can
  reduce boilerplate while the public types remain distinct topology builders.
- The headless gallery repeated begin/prepare/finish/complete dispatch for five
  nullable owners. A small example-only optional-owner adapter can use
  `advance_coupled`; this should not become an installed scene abstraction.

## API symmetry after this pass

The five point-based types now share `point_state()` with positions,
velocities, radius, inverse mass, optional coupling outputs, and persistent
color. All six owners share `begin_frame`, `prepare_substep`,
`finish_substep`, `finish_frame`, `abandon_frame`, `advance_async`, `advance`,
and `reset`.

Remaining asymmetry worth fixing before a stable physics tag:

- telemetry names and payloads differ (`collect_statistics` for fluid/cloth/
  rope, `collect_telemetry` for soft body/smoke, and no rigid-body snapshot);
- deformable `SoftBodyOptions` still carries compatibility timestep/substep
  fields while common advancement uses `FrameOptions`;
- rigid reactions are host force/torque calls, whereas point reactions are
  device records—an explicit device reaction batch is needed before claiming a
  fully GPU-resident two-way rigid coupling path;
- smoke's specialized direct collider path supports spheres while the generic
  common point contact supports all four analytic shapes.

## Removed in this pass

- Fake midpoint impulses in the headless gallery were removed; actual analytic
  point/collider contacts now exercise the public API.
- Gallery options for physical skin frequency, bridge dimensions, cylinder
  dimensions, and `rigid_course_preset` were removed because that backend never
  consumed them.
- Collider-set growth now preserves uploaded contents instead of silently
  replacing them.

## Do not collapse

- Keep `Fluid`, `Smoke`, and fixed-topology deformables as separate algorithms;
  a shared point view is useful, a single mega-solver is not.
- Keep gallery recipes and native controls out of the installed API.
- Keep telemetry opt-in rather than folding CPU waits back into advancement.
- Keep the existing gallery contexts as the integration/test matrix; do not
  replace them with two simplified reference applications.
