# Skin / post contact: connected response and exclusion experiments

## Scope and decision

This is an experimental continuation of [round 3](SKIN_SOFT_BODY_CONTACT_ROUND3.md).
No production physics changes have been promoted from this round. The stricter
`--render-contact` audit is retained in the main repository. Scratch
implementations, GPU binaries, tests, and verbose traces are under:

`/tmp/meshprep-contact-round4-Qf8Q3D/`

An experimental whole-water translation constraint reaches **zero enclosed
post voxel centers** on all three saved failure recordings without destroying
post triangles or stopping the water. This is a useful result, **not yet a
complete soft-body collision fix**: it makes the contact response more rigid,
is initially one-way, and does not establish continuous surface exclusion.
The later mass-aware, 0.004-margin variant also reaches zero enclosed **visible
post vertices** on all three recordings. Its limitations are detailed below.

## Measurement conditions

- RTX 3050 Ti Laptop GPU; CUDA 13.1.80; Release, C++20, SM86.
- Restore only the first pre-contact state, then advance every recorded input.
  Never restore later failed states or repeatedly reset the trajectory.
- Four physical substeps, 16 post spring iterations; material settings unchanged.
- Corrected outward-wound cylinder asset and disabled anchor-normal heuristic
  from round 3 for candidate runs.
- Every saved frame audited. Timings include GPU simulation and surface
  preparation, not ray tracing, display, or CPU diagnostic geometry work.
- These are single screening replays, not repeated FPS benchmarks. Baseline
  figures below are from round 3, not a new paired timing trial.

## 1. Does the connected post deliver the proposed contact movement?

The isolated GPU fixture applies a known normal displacement to one, eight,
or 32 exposed post nodes, then runs the actual post spring solver. Gravity,
the floor, water, and the controller are absent. Both a genuinely free post
(zero pins) and the normal 50-pin post are tested. Loading is along the local
outward side normal, not tangent to the surface.

For a 0.001-unit proposal and 16 spring iterations:

| Loaded nodes | Remaining node displacement / proposal | Anchored render-patch displacement / proposal |
| --- | ---: | ---: |
| 1 | 0.197 | 0.159 |
| 8 | 0.312 | 0.275 |
| 32 | 0.388 | 0.372 |

The free and pinned cases have essentially the same short-time local result:
this is not solely a pinning problem. Springs spread and partially undo the
contact displacement. That is an expected mechanical response, but the
sequential contact calculation does not solve for the resulting final gap.
The measured center-of-mass change is 90.9–92.9% of the proposed mass-average
change; the existing degree-averaged spring projections are not exactly
momentum preserving either.

Fixture: `response/apps/water_lab/contact_response_fixture.cu`.
Results: `response-corrected.csv` (space-delimited despite the suffix).
This tests a displacement response, not a complete contact Jacobian.

## 2. Is this simply a search-radius miss?

The newest original capture contains 829 outside-to-inside voxel transitions.
595 (71.8%) cross a skin face interior, rather than near an edge or vertex.
However, **all 829** have a nearest skin vertex within the existing 0.0726667
forward-query radius; the maximum nearest-vertex distance is 0.0571131.

This argues against simply increasing that radius as the main remedy. It
does not prove that the existing skin-vertex-to-post-triangle query generates
the correct constraint for every transition; the two query directions differ.
The CPU coverage analysis uses the recorded moving physical skin, with
28-step interpolation bracketing of each transition.

Evidence: `coverage/SUMMARY.md`, `coverage/outside_to_inside_all.csv`.

## 3. Can coherent movement prevent the destructive local corrections?

The prototype queries every nearby post voxel against the actual closed
physical skin, then solves contact halfspaces in the water's global
translation coordinates. It translates **both skin and internal particles**,
preserving their relative geometry. The original local forward contact and
post spring solver remain active. A 0.002-unit margin avoids the much larger
standoff produced by using the entire voxel radius.

The initial diagnostic adds translation/dt to water velocities and treats
posts as fixed for this additional correction. This is not a reciprocal,
energy-qualified contact impulse model.

| Recording | Baseline peak enclosed centers | Prototype peak enclosed centers | Broken bonds | Active post inversions | Minimum skin area/rest |
| --- | ---: | ---: | ---: | ---: | ---: |
| `205126-frame-448` | 771 | 0 | 0 | 0 | 0.885 |
| `214101-frame-236` | 717 | 0 | 0 | 0 | 0.751 |
| `220356-frame-406` | 798 | 0 | 0 | 0 | 0.859 |

Each prototype passes the existing skin-volume and active-post geometry
thresholds. On the newest recording the water-center path length is 2.483
units, including 0.440 during the final 60 frames: exclusion is not obtained
by freezing the water.

A paired newest-recording control disables only the original forward contact.
Comparing all 8,000 final post positions gives RMS difference 0.001191 and
maximum difference 0.006996; 646 nodes differ by over 0.001. Thus the forward
contact still changes the post, but only modestly. Large bending and breaking
are **not** established by these successful center-exclusion trials.

### Exact-query acceleration

The brute-force signed query was replaced by nearest-first BVH traversal and
closest-feature angle-weighted pseudonormals, following
[Bærentzen and Aanæs](https://orbit.dtu.dk/en/publications/signed-distance-computation-using-the-angle-weighted-pseudonormal/).
It requires a closed, consistently outward-oriented, nondegenerate skin.
Traversal uses deterministic ties and reports stack overflow; it does not
silently skip nodes.

The GPU/CPU fixture covers an outward tetrahedron, face/edge/vertex samples,
multi-leaf traversal, orientation reversal, and explicit overflow. It is not
an exhaustive arbitrary-folded-mesh validation. Independent review found and
corrected a corner-angle indexing error before the reported GPU test.

Newest-recording median GPU update: brute-force **30.394 ms**, fast query
**20.940 ms**. Recorded geometry metrics are unchanged. The fast prototype
remains slower than the earlier unchanged baseline (**12.238 ms**) and misses
the complete-frame 60 FPS target even before rendering.

## What the current containment gate actually guarantees

- `center_inside`: lattice voxel centers inside the closed physical water skin.
- `surface_inside`: the same test restricted to surface-flagged lattice nodes;
  **not** a test of rendered post vertices.
- `sphere_inside`: enclosed centers at least one voxel radius from the skin;
  **not** a count of all sphere/surface intersections.

Zero counts do not prove that finite-radius voxels, rendered post triangles,
or motion between frame boundaries remain outside. Separate visible-surface
checks and reciprocal-response tests are necessary before promotion.

## 4. Visible surface check: catches a false sense of success

The new `--render-contact` measurement downloads the actual GPU-deformed post
surface and tests every vertex referenced by active render triangles. It
includes restored frame zero and every advanced frame. It shares the existing
physical-skin parity/closest-distance implementation rather than using a
second geometric approximation. Failures propagate to the executable's final
status and exit code.

The unchanged newest replay has **626 visible vertices inside at peak**, with
maximum depth **0.687993** and 104,335 vertex/frame observations. Its positions
and velocities still reproduce the saved capture **bit-for-bit**, confirming
that adding the measurement did not change physics.

The one-way 0.002-margin prototype has zero enclosed voxel centers but **six
visible-vertex overlap observations**: peak two, depth 0.000980. The strict
gate correctly rejects it. Checking only voxel centers would have missed this.

The permanent audit rejects empty active render geometry, non-finite points,
sampled classification disagreements and incomplete frame coverage as well.
Its CPU analytic/active-vertex-selection self-test passes via CTest; invalid
`--render-contact --stride 2` and `--baseline-only` combinations are rejected.
CSV columns remain unchanged; visible-surface measurements are stdout rows.

## 5. Mass-aware translation response: a useful but limited model

The next prototype recomputes active-bond connected components. An anchored
component has zero translational mobility; detached components use the sum of
their actual voxel masses. Water translation uses the mass of **all** physical
skin vertices and fluid particles. Position corrections and zero-restitution
velocity impulses are shared according to these inverse masses, rather than
injecting translation/dt into water velocity.

GPU fixture checks use the actual CUDA/CUB implementation, not duplicated host
formulas: positional mass ratios, equal-and-opposite linear impulses, total
kinetic-energy nonincrease for uniform and nonuniform free-node velocities,
non-vacuous pinned response, severed bonds, and 100 randomized 1,024-node
connectivity comparisons against a CPU disjoint-set reference. All pass.
Review corrected unsafe concurrent union-find compression, a missing base
velocity in the first test's momentum assertion, and weak initial fixtures.
The tests explicitly detect that stopping component-mean closing velocity
**does not** stop every locally deforming contact point.

Compute Sanitizer **memcheck, racecheck, initcheck and synccheck all pass on
this isolated GPU fixture**. The first initcheck run found uninitialized
padding in the downloaded statistics struct; using an initialized full-width
flag fixed it. Logs preserve both the original finding (`modal-initcheck.log`)
and the clean reruns (`modal-*-fixed.log`). This is not a sanitizer pass for the
full course simulation.

At a 0.002 margin, this mass-aware candidate still has ten visible-vertex
overlap observations in the newest recording: peak one, depth 0.001543.
Increasing the margin to **0.004** covers the measured render-surface offset
in these replays, while staying well below the approximately 0.032 voxel radius.
This is an experimentally selected margin, **not** a proved bound on render
deformation or tunneling.

| Recording | Enclosed voxel centers | Enclosed visible vertices | Active post inversions | Minimum skin area/rest | GPU median ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| `205126-frame-448` | 0 | 0 | 0 | 0.886 | 22.382 |
| `214101-frame-236` | 0 | 0 | 0 | 0.879 | 22.232 |
| `220356-frame-406` | 0 | 0 | 0 | 0.870 | 22.619 |

All three have zero new broken bonds, finite states, full frame coverage and
passing measured geometry gates. Water-center paths are respectively 1.827,
2.063 and 2.007 units; newest final-60-frame travel is 0.336 units. The water is
not frozen. The fresh unchanged newest baseline median is **12.546 ms** under
the new audit, so this is not a performance win.

### Why this is not the default app physics

- This is a **three-translation-coordinate** contact model per component, not
  a complete deformable contact solve. It has no angular impulse response.
- A pinned component cannot translate in this extra solve. Local bending still
  comes from the original forward contact, so the observed interaction becomes
  more rigid. Substantial bending and breakage have not been qualified.
- Detached components lack floor/rail constraints in this extra projection.
  The isolated free-body tests deliberately omit external obstacles. Passing
  those fixtures is not a full falling-fragment course test.
- Position projection is split from velocity response; the linear impulse and
  kinetic-energy fixture does not prove preservation of full elastic energy.
- No continuous collision detection or triangle-interior exclusion guarantee;
  no complete-scene sanitizer/determinism/performance qualification yet.
- About 22–23 ms before ray tracing misses the original 60 FPS requirement.

**Next focused test:** retain the stricter regression, then replace the extra
rigid component response with a measured connected-post bending response.
Require both exclusion and meaningful post deflection under load; a fixed
obstacle that merely prevents overlap is not sufficient. Separately constrain
detached component floor/rail contact and measure angular response. Only then
optimize the repeated query/component reductions and qualify full replays.

## Reproduction

From the scratch `coupled/` directory:

```bash
cmake -S . -B build-r4 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DPARALLEL_MATER_BUILD_LAB=OFF -DPARALLEL_MATER_BUILD_TESTS=OFF \
  -DPARALLEL_MATER_BUILD_EXAMPLES=OFF -DPARALLEL_MATER_BUILD_BENCHMARKS=OFF
cmake --build build-r4 --target meshprep-soft-body-capture-audit -j2
env MESHPREP_DISABLE_POST_ANCHOR_HINT=1 \
  MESHPREP_COHERENT_CONTACT=2 MESHPREP_COHERENT_RADIUS=0.002 \
  MESHPREP_COHERENT_FAST=1 \
  ./build-r4/meshprep-soft-body-capture-audit \
  /tmp/meshprep-hybrid-captures/capture-20260915-220356-frame-406 \
  --asset assets/softbody/checker_cylinder_outward.msb \
  --resimulate-saved-start --skin-contact --stride 1 \
  --csv /tmp/meshprep-contact-round4-Qf8Q3D/repeat-newest.csv
```

The middle recording's first successful trial used the brute-force query;
the oldest and newest rows use the accelerated query. Verbose logs are named
`coherent-small-second.log`, `coherent-fast-first.log`, and
`coherent-fast-newest.log`. `MESHPREP_NO_FORWARD_CONTACT=1` selects the paired
diagnostic control, not a shipping feature.

For the final mass-aware trials, use `MESHPREP_COHERENT_CONTACT=3`,
`MESHPREP_COHERENT_RADIUS=0.004`, and
`MESHPREP_AUDIT_RENDER_CONTACT=1` with the same command. Logs are
`modal-4mm-visible-{first,second,newest}.log` and matching CSVs. The temporary
environment selector is confined to scratch; no such selector was added to
the main simulation.

The retained production measurement can be reproduced without experimental
physics:

```bash
./build/meshprep-soft-body-capture-audit \
  /tmp/meshprep-hybrid-captures/capture-20260915-220356-frame-406 \
  --resimulate-saved-start --render-contact --stride 1
```

An unchanged solver is expected to **fail** this regression. Evidence is in
`production-visible-baseline.log`; this failure is not an execution crash.

Isolated response fixture, from scratch `response/`:

```bash
cmake --build build-contact-response --target meshprep-modal-contact-workspace-fixture -j2
./build-contact-response/meshprep-modal-contact-workspace-fixture
for check in memcheck racecheck initcheck synccheck; do
  /usr/local/cuda-13.1/bin/compute-sanitizer --tool "$check" --error-exitcode 99 \
    ./build-contact-response/meshprep-modal-contact-workspace-fixture || break
done
```

Cleanup, **only after preserving any desired evidence**:

```bash
rm -r -- /tmp/meshprep-contact-round4-Qf8Q3D
```

This removes only this round's scratch work, not the original captures.
