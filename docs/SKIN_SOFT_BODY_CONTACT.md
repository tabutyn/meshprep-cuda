# Water skin / soft-body contact experiment

Capture: `capture-20260915-205126-frame-448` (September 15, 2026, 20:51 local).
Context 1: the rolling water sphere overlaps a deformable post. The required
contact direction is that soft-body voxels remain outside the closed water skin.

## Reproducible input

- Original recording: `/tmp/meshprep-hybrid-captures/capture-20260915-205126-frame-448/`.
- Binary SHA-256: `aa6457eca1d5815014c2bda3fce424773c08fb46760f1d911ea61f900eb896f8`.
- 360 recorded frames, physics frames 89–448; 10,000 water particles;
  1,002 physical skin vertices / 2,000 triangles; 8,000 soft-body voxels.
- Binary format is v6. The old manifest incorrectly said v5; new manifests
  derive the version from the binary header.
- Hardware: RTX 3050 Ti Laptop GPU (4 GB), driver 590.44.01.
- Build: Release, CUDA 13.1.80, compute architecture 86, no device-debug build.
- Temporary source variants and traces:
  `/tmp/meshprep-skin-contact-20260915-LfaRyB/`.

Recorded geometry is measured directly. Resimulations restore the earliest
recorded state, then advance with each subsequent frame's saved gravity,
material settings, and control inputs. The capture stores soft-body strength
and dynamic state but does not store all `SoftBodyOptions`; those settings
must match the context-1 recipe. Replay agreement is measured before comparing
physics changes. A saved frame represents the state after its physics step.

## Measurements

The primary measurement is soft voxel centers inside the actual closed physical
skin, with minimum point-to-triangle distance providing penetration depth.
Surface voxels are also counted separately. This uses the deformed triangle
mesh, rather than an assumed sphere or the existing forward-contact counter.

Each candidate is also checked for skin triangle area/volume, non-finite state,
post speed, broken bonds, and physics time. CPU geometry analysis and downloads
are reported separately from physics. A smaller overlap obtained by collapsing
or exploding either body is a failed candidate. The target is zero interior
soft voxels; partial improvements must retain their remaining penetration.

## Five hypotheses

| Trial | Proposed cause | Isolated change |
| --- | --- | --- |
| H1 | Bounded search abandons penetrations outside its contact band | Double the movement term in the forward search radius |
| H2 | The small correction cap resolves overlap slower than it is created | Increase only the per-contact transfer cap fourfold |
| H3 | Averaging reactions weakens shared-voxel response | Sum reactions, retaining the final correction cap |
| H4 | Spring iterations undo the contact correction | Apply the positional contact correction after spring projection |
| H5 | Sampling only water vertices misses post voxels crossing triangle interiors | Add reverse voxel-to-water-triangle contact with deterministic reactions |

The runtime currently only queries water vertices against the post render
triangles. Existing tests checked that some contacts occur, but did not check
whether soft voxels remain outside the water skin. This missing invariant is
the central measurement added by this experiment.

## Results

The full original recording first contains a post voxel at physics frame 254.
The peak is **771 interior voxel centers at frame 417**, of which 748 complete
voxel spheres are inside. Maximum center-to-exit distance is **0.694188**.
The first saved frame has no interior post voxels. Replaying from that frame
reproduces the recorded particle, skin, and post positions and velocities
bit-for-bit across all 359 transitions in the final baseline run.

Exploratory trials advance all 359 transitions; contact geometry is sampled
every fourth saved frame. Triangle validity and fracture metrics cover every
frame. “Inside samples” sums interior voxel counts over those 90 observations;
it is not a count of unique voxels. All trials remain finite.

| Trial | Inside samples | Max depth | New broken bonds | Active triangle inversions (observations) | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Baseline | 22,414 | 0.694167 | 1 | 0 | Known collision failure |
| H1 wider search | 21,029 | 0.685040 | 17 | 4 | Reject: geometry regression |
| H2 larger transfer | 12,115 | 0.591948 | 523 | 331 | Reject: severe fracture/distortion |
| H3 summed reaction | 14,213 | 0.603029 | 342 | 216 | Reject: severe fracture/distortion |
| H4 contact after springs | 22,922 | 0.688419 | 33 | 0 | Reject: more overlap and higher speed |
| H5 reverse contact | 2,597 | 0.175955 | 646 | 118 | Reject: coverage improves, response destabilizes |

H5 reduces summed penetration depth by 97.3%, confirming the importance of
reverse collision coverage. Its positional correction was also converted to
velocity. H5b removes that conversion while retaining the position barrier:

- Inside samples: 298; peak 23 at frame 285.
- Maximum depth: 0.050613; summed depth: 3.340567, down 99.94% from baseline.
- No broken bonds or inverted triangles; minimum active triangle area ratio
  0.761795 and maximum edge ratio 1.219567.
- GPU simulation mean 13.599 ms, versus 10.999 ms in the paired baseline.
- Containment still fails: this is a stable partial improvement, not complete
  exclusion of soft-body voxels.

H5c then raises only the position-correction limit to one maximum-speed
substep displacement. Full-frame measurement finds 11 interior center samples
(peak 2), maximum depth 0.019266, no breaks or inversions. However, minimum
active triangle area falls to 0.251214, below the 0.5 geometry threshold.
The larger correction is rejected.

H5d adds a closing-velocity impulse to H5b. It regresses to 507 broken bonds,
118 active inversion observations, and maximum penetration 0.200322. Reject.
H5e instead removes the original forward contact when the reverse pass runs.
It has no breaks or inversions but peaks at 731 interior voxels, with summed
depth 19108.492556. Reverse contact cannot simply replace the forward pass.
Reject. These refinements also used full-frame measurements.

## Final full-frame comparison

H5b was then rerun twice with **every frame measured**, not just every fourth
frame. The original five hypotheses were implemented by Terra agents in
isolated source copies; none was installed in the app during these tests.

| Metric | Baseline | H5b run 1 | H5b run 2 |
| --- | ---: | ---: | ---: |
| Peak inside voxel centers | 771 | 23 | 23 |
| Total inside-center observations | 90,777 | 1,188 | 1,188 |
| Maximum center-to-exit depth | 0.694188 | 0.052703 | 0.052703 |
| Sum of inside-center depths | 22019.100981 | 13.426162 | 13.426162 |
| New broken bonds | 1 | 0 | 0 |
| Active triangle inversions | 0 | 0 | 0 |
| Minimum active triangle area / rest | 0.635747 | 0.761795 | 0.761795 |
| Maximum active edge / rest | 1.432481 | 1.219567 | 1.219567 |
| GPU simulation median (ms) | 11.266 | 14.059 | 14.089 |
| GPU simulation p95 (ms) | 13.369 | 14.633 | 14.593 |
| GPU simulation mean (ms) | 11.674 | 13.599 | 13.570 |
| Simulation call wall mean (ms) | 11.808 | 13.734 | 13.701 |
| Strict exclusion gate | FAIL | FAIL | FAIL |

H5b reduces peak inside count by 97.0%, maximum depth by 92.4%, and summed
depth by 99.94%. Its two CSVs agree in every non-timing field at the recorded
precision; this is repeatability of measurements, not a bitwise comparison of
the two candidate trajectories. Baseline replay was independently checked
with byte comparisons of all recorded positions and velocities.

Timings are unprofiled CUDA-event simulation measurements over the 359 replay
steps. They include soft-body work and render-surface deformation/hierarchy
updates, but exclude ray tracing/presentation, initialization, diagnostic
downloads, and CPU geometry analysis. This is a contact replay,
not a 120-frame steady-state FPS benchmark. The interactive app was suspended
during GPU measurements. Baseline time varied between separate batches;
the full-frame comparison above gives about a 25% median regression for H5b.

**Decision: no physics candidate is promoted.** H5b is the best stable partial
result, but still permits post voxels inside the water and costs more. The
stronger alternatives either damage geometry or lose the improvement.
Production physics remains byte-for-byte unchanged from the start of this
experiment. Kept changes are the capture-v6 audit, pre-contact replay/drift
checks, CPU measurement self-test, strict exclusion reporting, and manifest
version correction. No experiment selector or extra collision kernel is
added to the app.

## What the evidence supports next

The existing contact counter sees at most 0.078666 while the actual closed
skin contains voxels 0.694188 from the exit. Vertex-to-post queries alone do
not enforce post-to-water exclusion. Reverse triangle coverage is therefore
the leading direction, not another stiffness increase.

The next isolated experiment should retain both contact directions but use a
shared response for the water triangle and lattice nodes, rather than adding
an independent ejection velocity or multiplying each pair's impulse. H5b gives
the post its inverse-mass-weighted correction, but subsequently averages the
skin's shares, so the resulting two-sided corrections are not strictly reciprocal;
its nearest-face sign is also not the audit's closed-mesh parity test. These
are limitations to resolve, not a validated production contact law. Evaluate
the existing zero-inside and geometry gates first, then optimize the candidate
search. Do not relax the gates to accept permanent embedding.

## Reproduction and artifacts

From the repository root, build and check the CPU reference:

```bash
cmake --build build --target meshprep-soft-body-capture-audit meshprep-water-lab -j4
ctest --test-dir build -R '^meshprep-skin-contact-reference$' --output-on-failure
```

Measure recorded states without a GPU:

```bash
./build/meshprep-soft-body-capture-audit \
  /tmp/meshprep-hybrid-captures/capture-20260915-205126-frame-448 \
  --baseline-only --skin-contact --stride 1 \
  --csv /tmp/meshprep-skin-contact-20260915-LfaRyB/captured-final.csv
```

Replay the current solver from the first, pre-contact recorded frame:

```bash
./build/meshprep-soft-body-capture-audit \
  /tmp/meshprep-hybrid-captures/capture-20260915-205126-frame-448 \
  --resimulate-saved-start --skin-contact --stride 1 \
  --csv /tmp/meshprep-skin-contact-20260915-LfaRyB/replay-final.csv
```

Both currently return **exit 2**, the expected measured exclusion failure,
not a crash. Exit 1 is an execution/parser error; exit 77 skips GPU replay
when no device is available. A clean subsampled run is inconclusive, never a
strict pass. The winding check samples the first AABB candidate on selected
frames; it is not an exhaustive winding or self-intersection validation.
Recorded post-render geometry uses the historical weighted reconstruction;
candidate comparisons above use actual GPU-deformed render positions in both
the baseline replay and changed replay.

Temporary full traces are `reverse-H0-full1/`, `reverse-H5b-full1/`, and
`reverse-H5b-full2/` beneath the experiment directory. Source variants are
`forward-variants/`, `reverse-variant/`, and `reverse-only/`. To reproduce H5b
while those scratch artifacts exist:

```bash
MESHPREP_REVERSE_CONTACT=2 \
  /tmp/meshprep-skin-contact-20260915-LfaRyB/reverse-variant/build-h5/meshprep-soft-body-capture-audit \
  /tmp/meshprep-hybrid-captures/capture-20260915-205126-frame-448 \
  --resimulate-saved-start --skin-contact --stride 1 \
  --csv /tmp/meshprep-skin-contact-20260915-LfaRyB/h5b-repeat.csv
```

The selector exists only in that scratch build, not production. Temporary
artifacts are not a durable archive; preserve them elsewhere before cleaning
if further comparison is needed. This exact cleanup removes only experiment
copies, logs, and CSVs, **not the original capture**:

```bash
rm -rf -- /tmp/meshprep-skin-contact-20260915-LfaRyB
```

Verification: audit and interactive app build successfully; all 14 CTest tests
pass, including the new analytic measurement reference. Those green tests do
not imply the capture passes: the full recorded exclusion audit returns exit
2. No candidate was promoted or sanitizer-qualified in this experiment.
