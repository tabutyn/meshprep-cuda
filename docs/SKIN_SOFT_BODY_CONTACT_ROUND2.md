# Water skin / soft-body contact: second capture

This experiment continues [the first contact audit](SKIN_SOFT_BODY_CONTACT.md).
The required outcome remains that post voxels do not cross the closed water
skin. A reduced overlap that damages either mesh is not a successful fix.

Follow-up: [round 3 implementation audit and measured trials](SKIN_SOFT_BODY_CONTACT_ROUND3.md)
tests normals, render-map contact response, interleaving, and connected mobility.

## Input and method

- Capture: `/tmp/meshprep-hybrid-captures/capture-20260915-214101-frame-236/`.
- SHA-256: `b28e79cd5e3d87a3a115d19db1cfe91f7d11712884ff227d72fb32c4beb01992`.
- Context 1, capture v6, 236 post-step states (physics frames 1–236).
- 10,000 fluid particles, 1,002 physical skin vertices / 2,000 triangles,
  8,000 post voxels. Initial course preset: four substeps per 1/60-second frame.
- Experiment copies and traces: `/tmp/meshprep-skin-contact-round2-QBCm09/`.
- RTX 3050 Ti Laptop GPU, Release CUDA 13.1, SM86.

The CPU audit uses the actual deformed physical water triangles: ray parity
for inside classification and closest-triangle distance for depth. Every frame
is measured. Winding is an explicitly sampled cross-check, not a proof of
self-intersection freedom. Candidate post geometry uses GPU-deformed render
positions. Saved-start replay restores only the first frame, before contact,
then advances the recorded inputs without restoring subsequent states.

Production physics is unchanged at the start of this experiment. The previous
best candidate, H5b, remains a scratch reference: reverse post-to-skin triangle
queries plus bounded position correction, without injecting correction/dt
into velocity. New trials each change one feature of that reference.

## Five hypotheses

| Trial | Suspected cause | Isolated test |
| --- | --- | --- |
| A | A single reverse pass leaves a contact residual | Four contact-only Jacobi sweeps per physical substep, recomputing contacts |
| B | Averaging only skin reactions makes the response asymmetric | Scale each shared multiplier by contact degree on both sides, then sum skin reactions |
| C | The nearest face's normal misclassifies an edge/fold contact | Use closed-mesh parity and signed closest-feature distance |
| D | Shape restoration injects velocity that drives skin back into posts | Remove only shape-restoration velocity injection, retaining its bounded positional correction |
| E | Independent reverse post corrections damage its connected lattice | Apply reverse positional correction to skin only; retain normal forward post dynamics |

E is deliberately non-reciprocal and diagnostic, not a proposed physical law.
No variant changes the recorded controls, gravity, physical time, or material
settings. More contact sweeps in A are solver iterations, not extra elapsed
physical time. All candidate selectors exist only in temporary source copies.

## Captured failure

The first inside voxel occurs at frame **150**. Peak inside centers: **717**
at frame 217. Maximum center-to-exit distance: **0.657536** at frame 215.
There are 35,735 inside-center observations across the recording, including
32,371 observations of complete voxel spheres inside. No non-finite state is
observed. Minimum physical skin triangle area / rest is 0.358247.

The recorded post state has 341 broken bonds and 148 active inversion
observations (29 unique triangles). Historical weighted-render reconstruction
also flags severe area distortion. Candidate decisions use the current
GPU-render deformation in both baseline replay and candidate replay, not
that historical reconstruction.

## Measured trials

The unchanged solver reproduces all recorded particle, skin, and post positions
and velocities bit-for-bit. Its runtime-deformed post mesh has 142 active
triangle inversion observations and minimum active area ratio 0.084416. These
runtime metrics differ from historical weighted reconstruction, but both
identify the failure. The pre-contact restored state is not a frozen failed
state.

Every trial below starts from that same frame and uses the same subsequent
inputs. Counts of broken bonds are cumulative; inversion counts are active
triangle/frame observations, not unique triangles.

| Trial | Peak inside voxels | Max depth | Broken bonds | Active inversions | Min post area / rest | Min skin area / rest | GPU median ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Unchanged solver | 717 | 0.657536 | 341 | 142 | 0.084416 | 0.358247 | 11.330 |
| Previous H5b reference | 414 | 0.486590 | 266 | 168 | 0.070566 | 0.127193 | 13.345 |
| A: four contact sweeps | 7 | 0.044195 | 0 | 0 | 0.210127 | 0.478889 | 15.846 |
| B: symmetric degree normalization | 676 | 0.620042 | 317 | 97 | 0.050857 | 0.395929 | 13.202 |
| C: mesh-parity sign | 403 | 0.478513 | 296 | 152 | 0.039685 | 0.183423 | 13.330 |
| D: no shape velocity injection | 4 | 0.016420 | 0 | 0 | 0.745611 | 0.005012 | 12.046 |
| E: reverse correction moves only skin | 516 | 0.546020 | 257 | 120 | 0.105306 | 0.158954 | 13.132 |
| D-half: half shape velocity contribution | 576 | 0.557119 | 299 | 135 | 0.013256 | 0.180500 | 12.984 |

None passes. A reduces summed penetration from 6762.665312 to 1.514550 but
over-compresses some post triangles. B and C do not improve the previous best
reference enough to resolve contact or geometry. E rejects the idea that
holding the post fixed during the reverse correction alone is sufficient.

D's small overlap is especially misleading: its water volume falls to
1.134210 (initial volume approximately 1.75), while some skin triangles reach
0.5% of their original area. Stable-looking post geometry does not make this
a stable coupled simulation. The audit now includes explicit skin-collapse
checks so a future zero-overlap result cannot hide this failure.

The bounded D-half follow-up retains the same positional shape correction
and half the added velocity. It preserves water volume better (1.705350 to
1.804400) but fails with 576 inside voxels, 299 broken bonds, and 135 active
inversion observations. No further tuning sweep was performed.

Timings are unprofiled CUDA-event totals over 235 advanced frames. They include
GPU simulation and render-surface preparation, not ray tracing/presentation,
initialization, diagnostic downloads, or CPU triangle analysis. The app is
temporarily suspended during GPU measurements. These are contact-trajectory
measurements, not a steady-state frame-rate benchmark. The interactive app
was resumed after collection. Candidates were screened once on this capture;
none qualified for repeated performance measurements or promotion, so timing
differences alone are not treated as validated optimizations.

## Decision and retained changes

**All six candidate changes are rejected. Production physics is unchanged.**
They were implemented only in scratch builds; no experimental branch, new
ejection pass, or extra contact iteration was added to the app. No shader,
material setting, physical timestep, or existing user work was discarded.

The retained code is a small extension of the existing optional capture audit:

- Minimum physical skin triangle area must remain at least 10% of rest area.
- Signed skin volume must remain within 80–120% of the first measured frame.
- These join the existing zero-inside, finite-state, and full-frame-coverage
  checks; sparse sampling cannot establish a pass.
- The CPU self-test checks valid geometry and rejects collapsed area, volume
  loss, and non-finite input.

These are explicitly **gross-collapse diagnostic limits**, not a claim of
physical accuracy, good triangle quality, or absence of self-intersections.
The recorded failure still fails for penetration and post geometry. Candidate
D's measured area and volume also fail the new skin-quality thresholds.

The audit builds and its CPU reference/quality tests pass. No candidate was
sanitizer-qualified or promoted. This round does not claim the collision bug
is fixed.

## Next falsifiable experiment

The evidence supports testing **interleaved contact and lattice constraints**,
not simply more ejection strength. Currently the post finishes its spring
iterations before the reverse contact candidate moves its individual nodes.
More reverse sweeps reduce overlap but damage local triangle shape. Removing
shape-restoration velocity protects the post only by sacrificing the water.

A narrow next test would put the same reciprocal contact constraints between
lattice projection iterations, then reconstruct velocities once from that
combined solve. Compare against A at the same total number of spring/contact
evaluations and the same elapsed physical time. Require zero inside voxels,
valid post geometry, preserved skin volume/area, and passing both captures.
This remains a hypothesis, not a guarantee that a larger solver will work.

The local mass split is relevant but not sufficient: post inverse voxel mass
is 50 versus approximately 0.167 for a barycentrically interpolated water
triangle, so a triangle-interior contact initially assigns about 99.7% of its
position correction to one post voxel. That omits the response of its anchored
neighbors. However, E shows that simply shifting all correction to water does
not solve the problem. An unmeasured effective-mass formula was therefore not
added to production.

## Reproduce and clean up

From the repository root:

```bash
cmake --build build --target meshprep-soft-body-capture-audit -j4
ctest --test-dir build -R '^meshprep-skin-contact-reference$' --output-on-failure
./build/meshprep-soft-body-capture-audit \
  /tmp/meshprep-hybrid-captures/capture-20260915-214101-frame-236 \
  --resimulate-saved-start --skin-contact --stride 1 \
  --csv /tmp/meshprep-skin-contact-round2-QBCm09/replay.csv
```

The replay should return **exit 2** for the measured failure, not crash.
Use `--baseline-only` instead of `--resimulate-saved-start` to inspect recorded
geometry without a GPU. Source copies and commands for the candidates:

```bash
# A/B/C: MESHPREP_REVERSE_CONTACT=5/6/7, respectively.
MESHPREP_REVERSE_CONTACT=5 \
  /tmp/meshprep-skin-contact-round2-QBCm09/build-round2/meshprep-soft-body-capture-audit \
  /tmp/meshprep-hybrid-captures/capture-20260915-214101-frame-236 \
  --resimulate-saved-start --skin-contact --stride 1

# D/E/D-half: MESHPREP_ROUND2_VARIANT=5/6/7, respectively.
MESHPREP_REVERSE_CONTACT=2 MESHPREP_ROUND2_VARIANT=5 \
  /tmp/meshprep-skin-contact-round2-QBCm09/shape-variants/build-round2-audit/meshprep-soft-body-capture-audit \
  /tmp/meshprep-hybrid-captures/capture-20260915-214101-frame-236 \
  --resimulate-saved-start --skin-contact --stride 1
```

These selectors exist only in scratch builds. Logs/CSVs are named
`baseline-replay`, `h5b-reference`, `A`, `B`, `C`, `D`, `E`, and `Dhalf` under
the experiment directory. Their audit predates the added gross-collapse
guard, but records the area/volume values used to reject D. `final-recorded`
uses the updated guard. Temporary traces are not a durable archive. After
preserving anything needed, this exact cleanup removes only round-two scratch
source copies, builds, and reports, **not either original capture**:

```bash
rm -rf -- /tmp/meshprep-skin-contact-round2-QBCm09
```
