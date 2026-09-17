<!-- SPDX-License-Identifier: MIT -->

# Soft-body capture audit

`meshprep-soft-body-capture-audit` reads water-lab capture versions 4 through 6 and
measures deformation without opening a window. Historical frames are
reconstructed with their recorded four-weight binding model. Fresh replays
download the actual runtime render positions, so changes to GPU deformation
cannot accidentally be audited with an obsolete host formula.

The default input is the directory named by
`/tmp/meshprep-hybrid-captures/LAST_CAPTURE.txt`:

```bash
cmake --build build --target meshprep-soft-body-capture-audit -j2
./build/meshprep-soft-body-capture-audit \
  --csv /tmp/meshprep-hybrid-captures/latest-soft-body-audit.csv
```

Pass a capture directory or `capture.bin` as the first argument to select an
older recording. `--baseline-only` performs the file and geometry audit
without requiring a GPU.

For water-skin/post exclusion, add `--skin-contact --stride 1`. This measures
post voxels inside the recorded physical water triangles, not an assumed
sphere or the forward-contact counter. `--resimulate-saved-start` restores
only the first recorded frame and then advances all later recorded inputs;
it never restores the final failed state or snaps to later saved positions.
Use a recording whose first frame precedes contact. Position/velocity drift
and bitwise agreement against the recording are reported separately, so an
unchanged solver's replay can be checked before testing a candidate.

The optional containment gate requires zero inside voxel centers throughout
the measured run and a conservative skin-quality guard: every sampled physical
triangle must retain area/rest-area at least `0.10`, and signed volume must
remain within `[0.80, 1.20]` of the first measured frame. These are deliberate
gross-collapse safeguards, not general material-quality thresholds.
Subsampled `--stride` runs can discover failures but cannot establish a strict
pass. The winding-number cross-check is sampled, not an exhaustive proof of
inside classification or self-intersection freedom.
For a stricter GPU replay, add `--render-contact`. This also enables
`--skin-contact` and checks the actual deformed post vertices referenced by
active render triangles, including restored frame zero. It emits
`RENDER_CONTACT` rows and a separate `RENDER_CONTACT_GATE`; enclosed visible
vertices, empty active render geometry, non-finite points, classification disagreement, or incomplete
coverage fail the final result. These rows are written to stdout; the existing
CSV schema is unchanged. Diagnostic downloads and analysis are outside GPU
physics timings. This option requires `--stride 1` and cannot be combined with
`--baseline-only`.

`surface_inside` in the older voxel audit means **surface-flagged voxel
centers**, not visible mesh vertices. `sphere_inside` counts wholly enclosed
voxel-radius spheres, not every sphere/skin intersection. Neither that audit
nor `--render-contact` proves absence of triangle-interior intersections or
tunneling between frames. See [round 4](SKIN_SOFT_BODY_CONTACT_ROUND4.md) for
the stricter check's motivation and experiment results.

See [the September 15 contact experiment](SKIN_SOFT_BODY_CONTACT.md) for the
baseline, five hypotheses, rejected candidates, and reproduction commands.

The first `METRICS` row describes the states actually stored in the capture.
The second is a new simulation that starts at authored rest and applies each
recorded frame's runtime options, gravity, rectangle force/torque, and
soft-body strength. It never restores the captured final state. This distinction
keeps a corrupt or exploded snapshot from becoming the simulation's initial
condition.

The geometry measurements include maximum and p99 triangle area relative to
rest, maximum render-edge length relative to rest, the fraction of all triangle
samples outside the diagnostic `[0.5, 2.0]` range, maximum penetration, breaks
per frame, voxel speed, active-triangle fraction, and explicit/reported
non-finite failures. Every captured voxel position and velocity is checked,
including voxels that no render binding references. Active (rendered/collision) triangles are also measured
separately: minimum, maximum, and p99 area ratio; abnormal fraction; and maximum
edge ratio.

Triangle orientation is measured relative to a nearby interior voxel selected
in the rest pose. The signed tetrahedron volume between that deformation-following
material anchor and the triangle must retain its rest sign. Unlike comparing a
current normal directly with a world-space rest normal, this test is invariant
under rigid translation and rotation. The report includes total and active
inversion counts. The optional CSV contains the same measurements per frame so the
first growth event can be correlated with contact and fracture. The summary also
reports unique inverted triangle IDs and an `INVERSION_INSTANCES` row that groups
unique active flips by soft-body instance.

The result has three independent gates:

- `finite` requires every voxel position and velocity to be finite and requires
  no derived-geometry, fluid-reported, or soft-body-reported non-finite failures.
- `triangle_normal` requires every supported area ratio in `[0.5, 2.0]`, every
  active render-edge ratio at most `2.0`, and at least 90% of render triangles
  active throughout the run.
- `orientation` requires zero material-relative inversions among supported
  rendering/collision triangles. Inversions in already inactive fracture
  bindings remain an unrestricted diagnostic because those bindings no longer
  participate in geometry queries.

`--baseline-only` applies all three gates to the recorded states. A full run reports
the captured gates as historical evidence but returns success or failure from
the fresh-rest resimulation, which tests the currently built solver. Therefore
an unstable historical capture can remain a useful regression input after the
current solver starts passing it.

Version 4 captures have complete positions, velocities, broken-edge flags, and
render activity, but do not contain pending fracture-damage counters. Their
stored geometry is auditable; subsequent fracture timing after a restored v4
snapshot is not exactly reproducible. Version 5 records those counters. Fresh
rest resimulation does not depend on either version's saved final state.

For the 2026-09-13 v4 failure capture (245 frames), the captured mesh reached a
maximum triangle-area ratio of `1510.75` and a maximum render-edge ratio of
`84.09`. Among triangles still marked active, area reached `709.22` and edge
length reached `58.12`; the active minimum area was `0.0646`. All finite
counters stayed zero, so the finite gate passes while the triangle-normal gate
fails. It also contains 19,981 triangle-frame inversions, of which 3,756 were
still active (317 unique active triangles, all on instance 2).

The fresh-rest current-code resimulation passes all three gates. Supported area
is `[0.6039, 1.3759]`, supported p99 area is `1.0009`, supported maximum edge
ratio is `1.2780`, and the minimum supported fraction is `0.9475`. It breaks
483 bonds rather than 4,230. It records 16 inversion observations only after
the corresponding support bond is inactive, and zero supported inversions.
Across all triangles, including inactive fracture seams, maximum area is
`3.8350`, p99 area is `1.0073`, and maximum edge ratio is `4.8477`. Both
unrestricted and supported geometry remain in the report.

Triangle activity is intentionally not a geometry-quality filter. The runtime
never deactivates a triangle because its area, edge length, or orientation
failed a gate; it deactivates only when an actual lattice bond used by an
anchor or local corotational frame fractures. This makes the supported gates
independent measurements rather than circular consequences of the mask.
