# Tilt-controlled water course

Run `./build/meshprep-water-lab`. WASD/arrows tilt gravity relative to the camera;
left-drag orbits, Ctrl-left-drag pans, wheel zooms, R resets, Space pauses, and
T toggles timings. Reach the green target beyond the eight rigid posts. The
original rectangle lab is still available with `--scene lab`.

`[` / `]` decrease/increase acceleration, particle/skin speed caps, and the
load-bearing water forces together in 1× increments from 1× to 8×, starting at
4×. The HUD and window title show the multiplier. Scaling gravity without those
forces made particles measurably compress more at high multipliers; the current
control preserves their ratio. It works while paused or while editing
parameters; replay and the rectangle lab ignore it. Reset preserves the choice.

`,` / `.` do nothing in context 1 because its posts are rigid. They remain
available in the numbered soft-body contexts.

The bracket control does not change the fixed timestep. Its iteration floor is
`max(4, multiplier)`, preserving 0.05 maximum speed-cap travel per substep. Each
displayed frame still advances exactly 1/60 second. Manually selected extra
iterations remain when above the old floor.

Control validation covers every multiplier, tilt-direction preservation,
press/repeat/release, paused/panel input, replay/lab exclusion, and capture/reset
retention. The course regression checks the analytic capped-cylinder distances,
projection normals, finite state, deterministic reset, post clearance, skin
quality, and downhill progress.

## Implementation and limits

- `obstacle_course.hpp`: one shared signed-distance definition for the board,
  rails, goal, and eight capped cylindrical posts.
- `hybrid_kernels.cu`: both particle and skin integration project against that
  rigid analytic course. Rectangle and soft-body contact kernels remain skipped
  in context 1.
- `course_rotation.cuh`: deterministic best-fit rotation for course-only soft
  shape matching. A center reduction, covariance/rotation kernel, and local
  shape update maintain a rollable, deformable shell. They run in SKIN PHYSICS.
- `water_kernels.cu`: analytic board, rail, and finite-cylinder hits compose
  with the CUDA water renderer.
- `main.cpp`: smoothed gravity (default magnitude 7.2), maximum 20-degree tilt,
  follow camera, progress/finish HUD, and capture/replay. New context-1 captures
  contain no soft-body payload; older captures still replay their water state
  while their retired deformable-post state is ignored.

The source cylinder used by contexts 4 and 6 is converted by Blender into exactly 1,000 voxels: 500
surface voxels, 50 pinned bottom voxels, and 500 interior voxels after a
deterministic 24-iteration relaxation. It has 7,704 undirected breakable bonds
and a 931-vertex/1,632-triangle render surface. The complete architecture,
reproduction steps, and Function/Form/Runtime gates are in
[SOFT_BODY_COURSE.md](SOFT_BODY_COURSE.md); the `.msb` format is documented in
[the Blender pipeline guide](../tools/blender/README.md).

This is a gameplay soft body containing repulsive particles, not validated
water CFD. Shape matching is deliberately a course-only shape-memory term;
it does not conserve liquid volume. Initial force-only, volume-only, and
radius-only prototypes flattened or folded under gravity and were removed.
The original course material used particle-boundary stiffness 2000 and force
cap 240. The faster preset scales existing forces with the larger gravity load;
no extra containment fallback was added. The normal rectangle lab is unchanged.

For the next phase, see [water cohesion and foam research](WATER_COHESION_AND_FOAM.md).
That document proposes a particle-derived surface instead of the physical
shape-preserving skin. A smaller [water/foam visual layer](WATER_VISUAL_EFFECTS.md)
is now implemented; free-liquid puddle physics remains future work.

Course contact projects physical water-skin vertices and internal particles
against the same rigid post signed distance used by the regression tests. It is
a discrete fixed-substep projection, not continuous collision detection.

## Archived deformable-post measurements — 2026-09-13

RTX 3050 Ti Laptop GPU, driver 590.44.01, CUDA 13.1.80, Release/SM86. The
960x720 startup profile used 10 warmups and 120 unprofiled samples with foam
visible. Median GPU total was 14.826 ms and median frame wall was 15.556 ms;
p95 wall was 16.679 ms, so sustained 60 FPS is not claimed. The four soft-body
stage medians were 0.765 ms physics, 0.123 ms hierarchy, 2.515 ms contact, and
0.075 ms render preparation. The detailed p5/p95/maximum table, sanitizer
scope, memory use, and 600-frame impact geometry are recorded in
[SOFT_BODY_COURSE.md](SOFT_BODY_COURSE.md).

## Analytic-peg reference baseline — 2026-09-13

These measurements came from the earlier analytic course. They are useful as a
reference, but must not be presented as fresh measurements of the current build.

The equal-radius particle search now uses a deterministic uniform-cell index
instead of per-particle traversal of the generic AABB hierarchy. At the default
4x/N4 course setting, three unprofiled 960x720 runs measured a 2.280 ms median
fluid-physics stage, 14.457 ms median GPU total, and 15.136 ms median frame wall.
All course geometry gates and the full eight-test suite pass. The p95 frame wall
is still 17.469 ms, so this is not a locked-60-FPS claim. See
[the optimization report](FLUID_PHYSICS_OPTIMIZATION.md) for the experiment
matrix, rejected changes, stability comparison, and reproduction commands.

## Historical 4× driving preset — 2026-09-13

Gravity/tilt acceleration is 4× (1.8 → 7.2), and the particle/skin speed ceilings
are 4× (3/2 → 12/8). Damping coefficients, the 30/s shape-recovery response, and
the fixed 1/60-second physical tick are unchanged. This is not a promise of
four times the travel speed in every contact or a four-times-faster solver.

The existing repulsion, spring stiffness, particle/skin boundary stiffness, and
particle/skin force caps also scale by four. This matches support forces to the
increased gravity load; otherwise particles compact and escape the soft skin.
No new collision law or recovery mechanism was introduced. Four fixed substeps
retain the original maximum travel per integrator step (0.05). The existing
0.1 travel guard now checks the substep interval instead of the whole tick.

The same 1,200-tick tilted driving fixture was used to reject cheaper or
unbalanced candidates; geometry gates were not relaxed:

| Candidate | Substeps | Maximum reported outside | Max absolute edge strain | Result |
|---|---:|---:|---:|---|
| 4× gravity/caps only | 4 | 4,624 | 0.966 | Fail containment |
| 4× horizontal acceleration/caps, original downward load | 4 | 1,646 | Not audited | Fail containment |
| Matched-force 4× preset | 2 | 1 | 2.431 | Fail containment and geometry |
| Matched-force 4× preset | 3 | 0 | 1.672 | Fail strain limit of 1.5 |
| **Matched-force 4× preset** | **4** | **0** | **1.346** | **Pass existing course gates** |

At four substeps: 786 peg-contact frames; no non-finite state; no audited point
penetration beyond tolerance; minimum relative triangle area 0.218; minimum
rotated-rest face alignment 0.233. Peak measured speed: particles 3.991, skin
5.099; droplet horizontal center-of-mass speed mean 0.695, maximum 2.584.
These are one trajectory's values, not guaranteed limits for arbitrary inputs.
Same-reset bitwise comparison is retained.

All eight CTests pass on the final build, including the unchanged rectangle
stability regression. A separate four-tick, obstacle-free test compares the
old/new presets with identical substeps: horizontal center-of-mass velocity
0.042527 → 0.170110, ratio **4.0000**. This verifies the increased early driving
response, not terminal rolling speed or completion time. Existing sanitizer
coverage below is from the original 1× course; this parameter change did not
rerun all four sanitizer tools.

Unprofiled 960×720, 10 warmups + 120 rendered startup samples, RTX 3050 Ti Laptop,
CUDA 13.1.80, Release/SM86, without GUI/vsync or rolling captures:

| Stage | Median ms | p95 ms |
|---|---:|---:|
| Fluid hierarchy | 1.109 | 1.450 |
| Skin hierarchy | 0.474 | 0.655 |
| Fluid physics | 9.867 | 13.860 |
| Skin physics | 0.333 | 0.341 |
| Physical normals | 0.847 | 1.161 |
| Render surface preparation | 1.708 | 2.016 |
| Ray trace | 1.119 | 1.164 |
| GPU total, including ray trace | 15.567 | 19.587 |
| Frame wall | 16.375 | 20.359 |

Wall maximum was 22.048 ms: this is **not sustained 60 FPS**, and interactive
capture/presentation adds overhead. The long tilted fixture's headless median
was 15.852 ms, excluding ray tracing and diagnostic downloads. Further speed
work should target particle-force neighbor traversal, not weaken geometry gates.
A second identical rendered run, after compilation/tests finished, measured
16.031 ms wall median, 20.703 ms p95, and 21.242 ms maximum (GPU median 15.236 ms).
Both runs ended with zero reported outside particles and non-finite failures.

```bash
./build/meshprep-course-tests --frames 1200 --iterations 4
./build/meshprep-water-lab --scene course --profile 120 --warmups 10
ctest --test-dir build --output-on-failure
```

The live panel starts at the bounded four-iteration floor. `--iterations 1` is
rejected for this speed preset. Use `--iterations 4` to reproduce this table.
Historical low-speed captures retain their saved settings and binary layout.
New trials/logs are under `/tmp/meshprep-course-speed4-UNEsOY/`; cleanup after
review is `rm -r -- /tmp/meshprep-course-speed4-UNEsOY`. Prototype selectors live
only in the temporary probe, not the application.

## Original 1× baseline measurements — 2026-09-13

RTX 3050 Ti Laptop GPU, driver 590.44.01, CUDA 13.1.80, Release, architecture 86.
960×720, one fixed 1/60-second tick, 10 warmups and 120 measured rendered frames.
Profilers were **not** active. No window/vsync or rolling capture during this run.

| Stage | Median ms | p95 ms |
|---|---:|---:|
| Fluid hierarchy | 0.867 | 1.156 |
| Skin hierarchy | 0.381 | 0.541 |
| Fluid physics | 1.952 | 2.324 |
| Skin physics, including shape matching | 0.081 | 0.084 |
| Physical normals | 0.155 | 0.240 |
| Render normals and hierarchy | 1.657 | 1.881 |
| Ray trace | 1.103 | 1.141 |
| GPU total, including ray trace | 6.227 | 6.798 |
| Frame wall | 6.970 | 7.520 |

Frame-wall maximum: 8.227 ms. GPU resident allocation report: 29,354,963 bytes.
Stage medians need not sum to the median total. Interactive presentation and
capture introduce additional costs; this is not a measured display FPS claim.

The 1,200-tick headless tilt test completed with 914 peg-contact frames, zero
reported outside particles and non-finite failures, and no point/peg/board
penetration beyond 0.0002 tolerance. Skin height at frame 300 was 1.390; minimum
triangle area was 0.001650 (40.1% of rest). Minimum face-normal alignment with
the best-fit-rotated rest normal was 0.335. The regression requires positive
alignment, at least 10% rest area, and at most 1.5 absolute edge strain. These
are shape-regime checks, not general proofs against self-intersection.
Maximum absolute edge strain was 0.897: this is a
deformable prototype, not a claim of small-strain physics. The outside counter
uses the existing local surface query, not an independent global hull proof.
All seven CTests pass, including legacy hierarchy, normals, hybrid and contact
experiments, course geometry, and repeat-reset bitwise motion checks.

```bash
cmake --build build -j4 --target meshprep-water-lab meshprep-course-tests
ctest --test-dir build --output-on-failure
./build/meshprep-course-tests --frames 1200
./build/meshprep-water-lab --scene course --profile 120 --warmups 10
```

The original headless driving fixture used gravity `(0,-1.8,-0.65)` throughout,
slightly stronger than the interactive tilt limit. The current faster fixture
scales that vector by four. Measurements above are historical 1× level-startup
results, not the new preset or a driving fixture.

## Original 1× sanitizer coverage

The final build reported zero memory errors and race hazards in 600-frame
checks targeting `course_*`, `integrate_particles_kernel`, and
`integrate_skin_kernel`. A 300-frame synchronization check over those kernels
also reported zero errors. A **fully unfiltered** 300-frame initialization
check reported zero errors; all runs also passed the geometry/state assertions.
These are scoped checks, not full-program coverage from all four tools.

```bash
filters=(--kernel-name kns=course_ --kernel-name kns=integrate_particles_kernel
         --kernel-name kns=integrate_skin_kernel)
for tool in memcheck racecheck; do
  /usr/local/cuda/bin/compute-sanitizer --tool "$tool" --error-exitcode 99 \
    "${filters[@]}" ./build/meshprep-course-tests
done
/usr/local/cuda/bin/compute-sanitizer --tool initcheck --error-exitcode 99 \
  ./build/meshprep-course-tests --frames 300
/usr/local/cuda/bin/compute-sanitizer --tool synccheck --error-exitcode 99 \
  "${filters[@]}" ./build/meshprep-course-tests --frames 300
```

The discarded filtered initcheck run reported untracked hierarchy writes as
uninitialized data. Do not use producer-excluding filters for that check; the
unfiltered run supersedes that report. Instrumented runtimes are not FPS data.

Temporary development artifacts and sanitizer reports are isolated in
`/tmp/meshprep-course-development-dV3CH2/`. After review, cleanup is:

```bash
rm -r -- /tmp/meshprep-course-development-dV3CH2
```

No commits, pushes, package installations, or system configuration changes.
