# Skin / post contact: normal, response, and solver-order experiments

## Decision

**No candidate fixes exclusion without another failure. No experimental
physics, selectors, or assets from this round were promoted into the app.**
The application physics remains unchanged. The strongest diagnostic lead is
the difference between a contact's assumed local voxel mobility and the
connected post's response after its spring solve. Artificially making the post
heavier at contact is not a physically consistent solution.

This follows [round 2](SKIN_SOFT_BODY_CONTACT_ROUND2.md). Terra agents implemented
isolated candidates; a separate Terra reviewer and the main agent checked the
math and implementation. The main agent ran GPU trials sequentially.

## Reproduction and measurement

- Primary input: `/tmp/meshprep-hybrid-captures/capture-20260915-220356-frame-406/`.
- Capture SHA-256: `34429be0a609164cfe6986ebb7965e08a27f23b7f69a44a9b9d34b72eeecbf45`.
- Context 1, 360 saved states, physics frames 47–406; 359 advanced frames.
- 10,000 fluid particles, 1,002 physical skin vertices / 2,000 triangles,
  8,000 post voxels; four physical substeps and 16 post spring iterations.
- RTX 3050 Ti Laptop GPU; CUDA 13.1.80, Release, SM86.
- Scratch sources, binaries, host tests, CSVs, and logs:
  `/tmp/meshprep-contact-fixes-9cOjzw/`.

Replay restores only the **first, pre-contact state**, then advances recorded
inputs. It does not restart from the failed final state or snap back to later
recorded positions. The unchanged baseline reproduces recorded positions and
velocities bit-for-bit. Every frame is audited (`--stride 1`).

Inside counts use closed physical-skin triangle queries, not a fitted sphere.
Depth is the enclosed voxel center's nearest-skin distance. Geometry metrics
use actual GPU-deformed post render vertices in replays. Active inversions are
triangle/frame observations, not unique triangles. Finite checks pass for all
trials below, but that alone is not a stability pass.

GPU times include simulation and surface preparation, not ray tracing,
presentation, diagnostic downloads, or CPU geometry analysis. These are single
screening replays, not repeated performance benchmarks. H3's repeated contacts
and refits are counted inside the soft-body stage; its total is comparable,
but named phase timings are not directly comparable. No qualifying candidate
was found for sanitizer/repeated-performance qualification.

## Primary results

All rows after N0 use the outward asset and disable the anchor-normal hint (N1).
The mobility trials leave spring integration/material masses unchanged and
alter only contact response; they are diagnostic counterfactuals.

| Trial | Peak inside voxels | Max depth | Broken bonds | Active inversions | Min post area/rest | Min skin area/rest | GPU median ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Unchanged baseline | 798 | 0.691662 | 19 | 23 | 0.327687 | 0.295324 | 12.238 |
| N0: outward asset, old hint retained | 779 | 0.693062 | 20 | 26 | 0.306302 | 0.289851 | 12.392 |
| N1: outward asset, hint removed | 788 | 0.687743 | 19 | 26 | 0.319644 | 0.515770 | 12.401 |
| Translation-only render/contact map | 788 | 0.689871 | 16 | 67 | 0.031276 | 0.537512 | 12.231 |
| Render-map contact Jacobian | 698 | 0.606727 | 490 | 319 | 0.071032 | 0.510574 | 12.763 |
| H3: interleaved contacts, final contact | 137 | 0.292642 | 746 | 201 | 0.025800 | 0.060069 | 19.394 |
| H3 revised: final spring spreading | 123 | 0.267837 | 628 | 246 | 0.033785 | 0.065533 | 16.430 |
| Contact mobility ×0.01, diagnostic | 59 | 0.144756 | 0 | 0 | 0.781759 | 0.657689 | 11.009 |
| Zero post contact response, diagnostic | 28 | 0.062196 | 0 | 0 | 0.781759 | 0.644271 | 11.066 |

The two mobility diagnostics reduce peak inside counts by 92.6% and 96.5%,
respectively, without the post destruction seen in H3. Skin volume remains
above 1.74945 / 1.74601 respectively, compared with initial 1.75614. Neither
reaches zero overlap, and neither is eligible for promotion: they substitute
an artificial contact mobility rather than solving the actual connected-body
response. A shorter trajectory through overlap also changes workload, so the
lower times are not an independently validated optimization.

## What was implemented and what each result means

### 1. Normal orientation: a real bug, not the entire collision failure

The exporter reverses triangle winding on conversion `(x,y,z) -> (x,z,-y)`.
That transform has determinant +1 and should preserve winding. All 1,632
canonical cylinder triangles are inward in the original asset. The scratch
asset corrects them; all 1,632 pass the cylinder-specific outward-normal check.

The runtime also uses the dominant binding anchors as an outward hint. Those
anchors are not guaranteed to lie below each render triangle. A CPU rest-pose
check leaves 280 triangle normals inward after that heuristic; near-zero hints
are additionally sensitive to rounding. Merely reversing winding does not fix
the heuristic's nonzero-sign decisions.

N1 removes that heuristic for the solid contact path, retaining the existing
cloth handling. It eliminates the false initial deep contact: baseline/N0
first report 0.0769052 penetration at frame 155; N1 first reports 0.00088541 at
frame 159. That is a concrete improvement in contact classification, but later
volume exclusion still fails.

N1 was also replayed against both earlier captures:

| Capture | Baseline peak inside | N1 peak inside | Baseline/N1 broken | Baseline/N1 active inversions |
| --- | ---: | ---: | ---: | ---: |
| `205126-frame-448` | 771 | 783 | 1 / 2 | 0 / 0 |
| `214101-frame-236` | 717 | 743 | 341 / 326 | 142 / 88 |

Therefore normal correction is necessary cleanup for a future complete fix,
but it is not being presented or installed as an exclusion fix by itself.

### 2. Contact response must match render deformation

The render map uses an anchor and two support nodes to rotate each authored
surface offset. Existing contacts distribute correction only through anchor
weights. The translation-only diagnostic makes the old contact mapping exact
by removing rotation from **both** visible and collision geometry. It is a
representation change, not a valid fix for rotating/bending solids; it fails.

The full candidate shares the renderer's corotation helper with contact and
central-differences its normal Jacobian over at most nine unique nodes, with
epsilon 1e-4. It uses `w_skin + sum(w_i * |J_i|^2)` for effective inverse mass,
`sum(J_i dot v_i)` for surface normal velocity, and `-w_i * J_i * lambda` for
reaction corrections. Closest feature, barycentrics, and normal are frozen
during this local linearization; this is not a derivative through feature
switching.

Its host test checks rigid-translation invariance, non-anchor sensitivity,
and the actual nonlinear gap change after applying a small correction. It
passes, and independent review confirms signs and mass terms. The replay
still damages the post. Existing multi-contact averaging and subsequent
spring projection remain outside this isolated Jacobian correction.

### 3. Contact/spring order matters, but repetition is insufficient

H3 refreshes the deformed contact surface and hierarchy, regenerates contacts,
and applies their reactions after every four spring iterations. Velocities
and fracture are evaluated once after the solve. The first version makes five
contact evaluations per substep including the initial one, ending in contact.
Review flagged that the final correction has no subsequent spring spreading.
The revised version ends with four spring iterations instead (four contact
evaluations total). Both fail, despite markedly lower overlap. This comparison
also changes work count; it is not an equal-work proof about ordering alone.

### 4. Local contact mobility does not represent connected resistance

For an illustrative triangle-interior contact with three distinct free anchors,
`w_skin=0.5` and the barycentric post inverse mass is about 16.667. The current
0.006416 transfer cap then gives the skin only about **0.000187** correction
per contact. This is an isolated-constraint upper bound, not a measured average.
The post receives most of the proposal, then shared-node averaging and spring
projection change its actual displacement.

Scaling only contact post mobility tests this mismatch without claiming to
repair it. Its much better result supports investigating the connected
response, not shipping a magic 100× mass change. Zero mobility still leaves
28 enclosed centers: contact transfer limits, vertex-only sampling, and
discrete collision coverage also remain possible contributors.

## Implementation review and corrections

Failures were not simply accepted as proof against the ideas:

- Earlier reverse-contact candidates averaged one side but applied full
  corrections on the other. Their response was not fully reciprocal.
- Earlier reverse corrections ran after post springs/fracture; that did not
  test a coupled solve. Their hierarchy was refreshed later, not immediately.
- An earlier parity candidate lacked a half-open shared-edge rule; it is not
  an exact classification guarantee.
- This round's first H3 ended with an unspread contact correction. A revised
  candidate was built and measured; it still fails.
- The new Jacobian initially emitted records for zero-derivative support
  nodes. Those zero records diluted other corrections because gathering
  divides by record count. They were removed **before** the reported GPU run.
- The extracted deformation helper initially omitted some finite checks.
  The main agent restored the original checks before the reported GPU run.

One review warning about asset wiring was resolved by verifying that every N1
command explicitly passes `--asset`; it was not an actual experiment failure.
Default app wiring would need changing if this asset were promoted.

The remaining failed trials cannot fairly be blamed solely on Terra's coding.
There were implementation defects, including in the main agent's first H3,
but reviewed/revised candidates still fail. The retained averaging and
constraint-order assumptions are architectural limitations.

## Reproducible commands

From the production repository, the unchanged replay is:

```bash
./build/meshprep-soft-body-capture-audit \
  /tmp/meshprep-hybrid-captures/capture-20260915-220356-frame-406 \
  --resimulate-saved-start --skin-contact --stride 1 \
  --csv /tmp/meshprep-contact-fixes-9cOjzw/baseline-newest.csv
```

Example N1 command, from `/tmp/meshprep-contact-fixes-9cOjzw/normals`:

```bash
MESHPREP_DISABLE_POST_ANCHOR_HINT=1 \
  ./build-normals/meshprep-soft-body-capture-audit \
  /tmp/meshprep-hybrid-captures/capture-20260915-220356-frame-406 \
  --asset assets/softbody/checker_cylinder_outward.msb \
  --resimulate-saved-start --skin-contact --stride 1 \
  --csv /tmp/meshprep-contact-fixes-9cOjzw/n1-newest.csv
```

Other scratch roots/flags, always retaining the N1 flag and explicit asset:

- `jacobian/build-translation-audit`: `MESHPREP_TRANSLATION_SKINNING_EXPERIMENT=1`.
- `jacobian-exact/build-exact`: full render-map Jacobian; no additional flag.
- `solver/build-solver`: `MESHPREP_INTERLEAVE_CONTACTS=1`; add
  `MESHPREP_FINAL_CONTACT=1` to reproduce the original final-contact variant.
- `solver/build-solver`: `MESHPREP_POST_MOBILITY=0.01` or `0`, without interleaving.

All builds use `-DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86`
and `-DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc`. CPU validation passed:
`./build/meshprep-soft-body-capture-audit --skin-contact-self-test` and the
scratch `build-exact/meshprep-soft-body-skinning-host-test`. Failed trials
return exit 2; that is a measured gate failure, not an executable crash.

Keep the scratch directory for inspection. When no longer needed, this removes
only this round's disposable builds/traces, not any original capture:

```bash
rm -rf -- /tmp/meshprep-contact-fixes-9cOjzw
```

## Next bounded experiment

Use a small supported lattice/contact fixture to measure **actual gap change
after gathering and spring projection**, versus the change predicted by the
contact multiplier. Compare one and many contacts, free and pinned support,
and check momentum balance. Then test a contact update that uses that connected
response, preserving the original masses and time advancement. Do not add
another ejection pass or tune mass until this fixture explains the discrepancy.

Separately test swept edge/face coverage in that fixture; no tested candidate
here establishes complete continuous collision detection. Correct winding,
render-map derivatives, reciprocal gathering, and coverage are distinct
requirements. A change must satisfy both exclusion and geometry before any
runtime optimization or app promotion.
