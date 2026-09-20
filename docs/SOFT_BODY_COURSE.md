# Deformable-post obstacle course

The tilt course replaces its eight analytic capsule pegs with eight instances of
one authored soft-body cylinder. Each instance has exactly 1,000 simulated
voxels, a fixed spring-neighbor graph whose bonds can break, and a UV-mapped
triangle surface driven by the surface voxels. The objective is readable game
physics: the water droplet can bend a post at ordinary course speed and fracture
it after a sufficiently hard impact.

This document describes the implemented architecture and the final local
correctness/performance measurements. The numbers below are measurements on one
RTX 3050 Ti Laptop GPU, not portable FPS or speedup claims.

## Asset and instance data

The committed checker-cylinder asset contains:

| Property | Per post | Eight-post course |
|---|---:|---:|
| Physics voxels | 1,000 | 8,000 |
| Surface voxels | 500 | 4,000 |
| Pinned bottom voxels | 50 | 400 |
| Undirected breakable bonds | 7,704 | 61,632 |
| Sorted CSR neighbor entries | 15,408 | 123,264 |
| Render vertices | 931 | 7,448 |
| Render triangles | 1,632 | 13,056 |

The voxel graph is allocated once. Breaking a connection changes one active
byte associated with its stable edge ID; it does not rebuild adjacency. Each
render vertex stores four surface-voxel candidates and normalized weights. At
runtime, the strongest candidate becomes one material anchor and two bonded
candidates define a deterministic local frame. Corotational skinning preserves
the authored surface at rest while rotating its offset with the material:

```text
render position = current dominant anchor
                + current frame * inverse(rest frame)
                  * (authored rest position - rest dominant anchor)
```

The bottom 50 surface voxels remain fixed to the board. This is a deliberate
gameplay boundary condition rather than a general rigid-body joint.

## Runtime architecture

`SoftBodyCourse` owns all eight instances in combined device buffers. Its
public frame protocol is `begin_frame`, then one
`prepare_substep`/water-contact/`finish_substep` sequence per water substep, and
finally `finish_frame`.

Within a substep:

1. Free voxel positions are predicted from the previous position and damped
   velocity. In the integrated course, post gravity is deliberately zero:
   pinned base voxels attach each post to the board and water contact supplies
   the external load, while the independently integrated water follows the
   player-controlled gravity vector.
2. Dominant-anchor corotational skinning updates the authored render surface,
   emits its triangle bounds, and refits the hierarchy used by contact and
   rendering. It avoids averaging positions across separated fracture pieces.
3. Every physical water-skin vertex traverses the deformed post hierarchy for
   its closest active triangle. A triangle is eligible only inside a finite
   support derived from the contact thickness and bounded per-substep water
   travel; a remote triangle cannot act as an infinite signed plane.
4. Contact is one unilateral, inverse-mass-weighted PBD constraint. The water
   vertex and the unpinned voxels in the triangle's render bindings use the same
   multiplier, so correction is mass-balanced between the two bodies.
5. Triangle barycentrics and four-voxel render bindings produce at most twelve
   voxel proposals per water vertex. Records are stable-sorted and reduced by
   voxel. Proposals sharing a voxel are deterministically averaged before their
   bounded correction is applied; floating-point contact atomics do not choose
   the result or multiply the response by contact density.
6. One CUDA thread then gathers the active, sorted CSR neighborhood of one
   voxel. Eight Jacobi iterations project active bonds toward their authored
   rest lengths, distributing the contact displacement through the connected
   lattice before velocity is finalized.
7. Velocity is reconstructed from the accepted substep displacement, clamped,
   and damped along active bond directions. A final fracture pass disables
   bonds only after residual post-solve strain persists for eight substeps.
   Render triangles leave later contact and smooth ray queries only when an
   anchor/frame support bond actually breaks. Area, edge length, and orientation
   remain independent measurements and never hide bad geometry.

The final deformed triangle hierarchy participates in primary rays, water
reflection/refraction rays, and course occlusion. Runtime shading interpolates
the GLB UVs and evaluates the checker pattern from them. The embedded bitmap is
retained in the source GLB for authoring inspection; the CUDA renderer does not
sample that image directly.

Important code locations:

- [`fixed_topology.hpp`](../src/internal/fixed_topology.hpp): private asset contract, views,
  runtime options, state, timings, and `SoftBodyCourse` API.
- [`fixed_topology.cu`](../src/internal/fixed_topology.cu): validation, deterministic
  spring gather, fracture, skinning, hierarchy updates, and state transfer.
- [`hybrid_kernels.cu`](../apps/water_lab/hybrid_kernels.cu): water/post triangle
  contact and deterministic reaction gathering.
- [`water_kernels.cu`](../apps/water_lab/water_kernels.cu): hierarchy traversal,
  UV interpolation, checker shading, and two-sided triangle rendering.
- [`softbody_asset_pipeline.py`](../tools/blender/softbody_asset_pipeline.py):
  Blender GLB generation, voxelization, relaxation, graph creation, render
  bindings, validation, and deterministic `.msb` serialization.

## Fracture control

`,` divides and `.` multiplies the post strength multiplier by `1.25` per key
event. Gallery presets choose their own starting strength; the experimental
interactive range is `0.0625x` through `64x`. The HUD reports the current value and cumulative
broken-bond count. Strength scales the bond-break strain threshold only. Broken
bonds stay visible in the lattice diagnostic, but no authored material triangle
is hidden; its existing vertex bindings keep it attached by an edge or vertex.
It intentionally does not change spring stiffness, so a
fracture experiment does not silently change solver stability. The upper range
is intended for stress experiments and is not a calibrated material model.

Breaking is irreversible during a run. `R` resets positions, velocities, active
bonds, and active render triangles. It retains the selected strength multiplier,
matching the acceleration control's reset behavior.

`[` and `]` still change course motion from 1x through 8x. Gravity, particle/skin
speed caps, and the water forces supporting that gravity scale together, while
the stable substep floor follows the multiplier. This keeps acceleration tests
from also becoming unintended fluid-compression tests.

## Blender conversion and reproduction

Regenerate the example GLB, checker image, preview, runtime asset, manifest, and
determinism check with Blender 4.5 or newer:

```bash
./scripts/generate_softbody_assets.sh
```

Convert another closed, watertight GLB with:

```bash
/usr/local/bin/blender --background \
  --python tools/blender/softbody_asset_pipeline.py -- \
  convert --input model.glb --output model.msb \
  --voxels 1000 --relax-iterations 24
```

The converter transforms Blender coordinates to right-handed, Y-up runtime
coordinates. It selects deterministic surface samples, fills the interior from
a fixed three-dimensional grid, and runs 24 deterministic short-range repulsion
iterations while the perimeter samples remain fixed. That relaxation avoids
very close interior pairs before the static neighbor graph is generated.

The reproduction command converts twice and requires byte-identical `.msb`
files. It also validates finite data, counts, indices, reciprocal sorted CSR,
canonical unique bonds, connectivity, normalized surface-only bindings, and
nondegenerate render triangles. The complete v1 binary layout is documented in
[`tools/blender/README.md`](../tools/blender/README.md).

Current asset identity:

```text
checker_cylinder.msb SHA-256
9cd9f824903b4a2e761f90230aa5474d24700d989e59f6d47bcfc33462deeffd
```

## Measurement and gates

Use the following commands after building for the target GPU:

```bash
./scripts/generate_softbody_assets.sh
cmake --build build -j --target meshprep-soft-body-tests meshprep-course-tests meshprep-water-lab
ctest --test-dir build --output-on-failure
./build/meshprep-water-lab --scene course --profile 120 --warmups 10
```

The course profile reports `soft_body_physics`, `soft_body_hierarchy`,
`soft_body_contact`, and `soft_body_render` independently. `soft_body_hierarchy`
is the deformed triangle-hierarchy refit; `soft_body_render` is skinning,
triangle-activity, and bound emission. Profile output also records instance,
voxel, edge, broken-bond, finite-failure, and resident-memory statistics.

### Function gate

The gate requires all of the following:

- exactly 1,000 finite voxels per post and all pinned voxels fixed;
- the isolated moderate-impact fixture produces measurable post displacement
  without broken bonds at the default strength;
- the harder, otherwise identical fixture produces at least one broken bond;
- mass-balanced water/post coupling keeps the water skin finite, outward
  oriented, and above its minimum-area gate during the scripted impact;
- reset restores the authored positions and repairs all fracture state;
- deterministic reset/replay produces the same active-edge state on the same
  supported GPU.

Status: **pass**. The deterministic rest, moderate-bend, hard-fracture, reset,
state-round-trip, and 600-frame integrated-course fixtures pass. The integrated
run reported zero escaped particles and finite failures, 0.381373 maximum water
edge strain, 0.746476 minimum relative triangle area, and positive 0.817822
minimum face alignment. Sustained contact began fracturing the struck posts at
frame 115; the high-energy course is therefore exercising deformation and
fracture rather than timing an inert obstacle.

### Form gate

The Blender fixed-camera preview verifies an upright, grounded cylinder, clear
checker wrapping, and readable proportions. Runtime review must additionally
show that the checker surface follows bending without separating from its voxel
collision surface, and that disabled triangles make a fracture visible.

Status: **pass for the authored asset and observed runtime integration**. The
fixed-camera preview verifies the source asset, and the launched native course
shows the UV checker surface following deformation and exposing fractures. The
automated tests validate triangle-activity state and rest-pose ray occlusion,
but do not yet pixel-certify a deformed fracture sequence. The subjective look
of extreme fractures remains an interactive art-polish task, not a correctness
claim.

Preview: [`checker_cylinder_preview.png`](../assets/softbody/checker_cylinder_preview.png).

### Runtime gate

The runtime gate requires:

- all relevant unit/integration tests pass on the target CUDA environment;
- zero finite failures in the bend and break fixtures;
- no remaining invisible analytic peg collision or shadow geometry;
- 10 warmups and 120 unprofiled measured frames at 960x720;
- median, p5, p95, and maximum for all four soft-body timing rows, GPU total,
  and frame wall time;
- pair the non-colliding startup profile with the separate 600-frame driven
  regression, whose broken-bond and water-geometry gates prevent an inert or
  invalid implementation from passing the overall runtime gate.

Status: **pass on the measured local environment**: RTX 3050 Ti Laptop GPU,
driver 590.44.01, CUDA 13.1.80, Release, SM 8.6. At 960x720 with 10 warmups and
120 unprofiled samples (foam visible), the startup workload measured:

| Stage | Median ms | p5 ms | p95 ms | Maximum ms |
|---|---:|---:|---:|---:|
| Soft-body physics | 0.7650 | 0.7544 | 0.7897 | 0.8335 |
| Soft-body hierarchy | 0.1232 | 0.1195 | 0.1394 | 0.1577 |
| Soft-body contact | 2.5149 | 2.4276 | 2.5754 | 2.5856 |
| Soft-body render preparation | 0.0746 | 0.0664 | 0.0942 | 0.1936 |
| GPU total, including all water/render work | 14.8260 | 14.4572 | 15.9207 | 16.3558 |
| Frame wall | 15.5564 | 15.1664 | 16.6790 | 17.2319 |

The four soft-body medians total 3.4777 ms. The median frame is below the
16.667 ms 60 Hz budget, while p95 exceeds it by 0.012 ms, so this is not a
locked-60-FPS claim. The soft-body allocation is 6,638,000 bytes.

The separate 600-frame driven-contact fixture measured 0.739 ms soft physics,
0.123 ms soft hierarchy, 4.198 ms contact, and 0.071 ms soft render preparation
at the median. Its 11.500 ms reported aggregate excludes ray tracing, visual
fluid generation, and diagnostic downloads and must not be compared directly
with the rendered frame total.

Full final memcheck, initcheck, and synccheck runs report zero errors. A targeted
racecheck over 500 launches of the project-owned prediction, constraint,
damping, contact-application, fracture, skinning, activity, and bounds kernels
reports zero hazards. The unrestricted racecheck was not completed and is not
claimed.

Removing an unused surface-voxel hierarchy preserved the scripted geometry
metrics while reducing the isolated eight-post solver from about 0.69 ms to
0.62 ms, reducing the four soft-stage startup median from 3.6329 ms to
3.4777 ms, and reducing soft-body resident storage from 8,170,272 to 6,638,000
bytes. Contacts now use the sole deformed render-triangle hierarchy.

## Known limitations

- This is a deterministic gameplay spring lattice, not FEM, a constitutive
  material model, or a validated fracture simulation.
- The graph has fixed adjacency. Bonds break, but fragments do not create new
  contacts or bonds and no remeshing occurs.
- Surface fracture disables stretched triangles; it can expose holes but does
  not generate interior fracture faces or preserve watertight fragments.
- Bottom pinning is baked into the asset. Moving or freely falling soft bodies
  need a different attachment/rigid-body model.
- Contact is between the water's physical skin vertices and deformed post
  triangles. Continuous swept edge/interior certification for arbitrary impact
  speeds is not claimed.
- One source material/UV set is preserved. The runtime currently evaluates a
  checker shader from UVs rather than importing the GLB material graph.
- The v1 course loader requires exactly 1,000 voxels per source asset and at
  most eight instances.
