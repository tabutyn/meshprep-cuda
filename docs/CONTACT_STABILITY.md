# Rectangle-edge temporal stability

The captured deep-fold failure exposed a closest-point BC-edge inequality bug,
now corrected and covered by host/CUDA regressions. Historical comparisons
below predate that correction. See [the geometry regression and analytical
hypotheses](FOLD_STABILITY_ANALYSIS.md) before using their parameter rankings
as evidence about the remaining instability.

`meshprep-contact-experiment` is a deterministic, headless test for the visibly
excited skin vertices around the dynamic rectangle. It advances exactly 1,980
fixed `1/60 s` physics ticks without rendering:

1. settle for 600 ticks;
2. press the rectangle from `(1.45, 0, 0)` to `(0.72, 0, 0)` over 180 ticks;
3. hold for 180 ticks;
4. slide to `(0.72, 0.30, 0)` over 180 ticks, moving an edge through loaded skin;
5. hold for 300 ticks;
6. withdraw over 180 ticks; and
7. recover for 360 ticks.

The test uses the finite-mass rectangle and its normal target controller. It
does not teleport the collider or infer stability from a rendered image.

## Measurement improvements

The original scalar metric sampled whichever vertices happened to be in the
edge region on each frame. That could make a quieter result merely by changing
which vertices were counted. The durable test now builds a fixed union of every
vertex entering the pressed-face edge region during contact, then evaluates the
same vertices throughout each phase.

For both that edge patch and the complete 1,002-vertex physical skin it reports:

- full-vector RMS, p99, peak, and worst-vertex velocity after subtracting a
  centered 15-frame moving average;
- the tangential component separately from the OBB-normal component; and
- press, face-hold, edge-slide, edge-hold, withdrawal, and recovery phases
  separately.

It also records exact analytic OBB penetration, CPU triangle/OBB intersections,
contact-local triangle area and edge stretch relative to the settled state,
rest-radial orientation reversals (a fold diagnostic, not a general inversion
or self-intersection test), shell skips, actual contact-normal angle changes, particle
contact-owner changes, per-vertex box impulse and pair work, all force-cap hit
counts, escapes, non-finite state, persistent recovery time, and simulation time
separately from diagnostic time. Global area/stretch extrema are still logged,
but do not gate contact because their worst values occur during initial settling.

Each run writes `frames.csv` and `skin.bin` under `/tmp`. `skin.bin` version 2
is little-endian: four `uint32_t` values (magic `0x4d504353`, version, frame
count, vertex count), followed per frame by all position `float3` values, all
box-force `float3` values, and all accumulated box-impulse `float3` values.

## Hypothesis results

All comparisons use one physics iteration and identical inputs unless noted.
The reference is the former sharp analytic OBB, damping `8`, and nearest-vertex
particle reaction. Timings are unprofiled CUDA-event measurements from the same
Release build; diagnostic downloads and CPU analysis are excluded.

### 1. Round the analytic box

Radii `0.01`, `0.02`, and `0.04` replaced the sharp OBB by a rounded-box SDF.
This dramatically reduced contact-normal jumps, but did not materially reduce
temporal excitation:

| Radius | Maximum normal change | Press p99 change | Edge-hold p99 change | Other result |
|---:|---:|---:|---:|---|
| 0.01 | 8.33 deg | +0.2% | -5.8% | effectively unchanged |
| 0.02 | 4.52 deg | -11.5% | +5.5% | worst edge vertex +47% |
| 0.04 | 2.45 deg | -12.7% | -5.3% | hold vibration grew (`1.154x`); contact impulse -4.7% |

Conclusion: the visible instability was not primarily caused by the OBB edge
normal discontinuity. Rounding also changes the physical collider and unloads
its corners. The rounded-box branch and selector were removed.

### 2. Distribute particle pressure over a triangle

The old contact assigned a particle's complete skin reaction to the nearest
vertex. The candidate finds the closest triangle incident to that vertex,
evaluates its closest point, and distributes the equal reaction to all three
vertices with barycentric weights. Reactions remain stable-sorted and reduced
by vertex, so there are no floating-point force atomics.

At damping `8`, this cut fixed-patch edge-hold RMS by 71.7% and edge-hold p99 by
68.3%, but made the initial press RMS 17.0% worse. This identified two separate
effects: discontinuous point loading dominates the settled chatter, while the
box contact itself needs more damping during impact.

### 3. Increase damping or use more substeps

| Candidate | Press p99 change | Edge-hold p99 change | Total physics time change | Decision |
|---|---:|---:|---:|---|
| Damping 16 | -24.7% | -8.6% | noise-level | insufficient |
| Damping 32 | -41.6% | -4.2% | noise-level | impact improved; hold did not |
| Damping 64 | -50.7% | -9.2% | noise-level | combine with distributed load |
| 2 substeps | -2.5% | -24.2% | +52% | remove |
| 4 substeps | -6.1% | -38.9% | +159% | remove |

More substeps are a broad, expensive way to reduce chatter and do little for the
press transient. They remain an interactive debugging control, not the selected
solution. Damping `64` plus distributed reactions is the smallest candidate
that passes both transient and stationary stability gates.

## Selected result

| Metric | Former default | Selected | Change |
|---|---:|---:|---:|
| Dynamic edge-region vibration RMS | 0.00400270 | 0.00132768 | **-66.8%** |
| Fixed-patch press vector RMS | 0.00908244 | 0.00603027 | **-33.6%** |
| Fixed-patch press p99 | 0.0408353 | 0.0266124 | **-34.8%** |
| Fixed-patch edge-hold RMS | 0.00591544 | 0.00124309 | **-79.0%** |
| Fixed-patch edge-hold p99 | 0.0206470 | 0.00461028 | **-77.7%** |
| Fixed-patch edge-hold peak | 0.0448484 | 0.0183738 | -59.0% |
| Final/initial hold vibration | 0.910960 | 0.333374 | decays faster |
| Maximum vertex penetration | 0.00444844 | 0.00144297 | **-67.6%** |
| Frames with triangle/OBB intersection | 777 | 745 | -4.1% |
| Peak intersecting triangles | 85 | 51 | -40.0% |
| Contact minimum area/settled area | 0.852770 | 0.853622 | +0.1% |
| Contact maximum edge/settled edge | 1.02565 | 1.02421 | -0.1% |
| Persistent recovery | 1.633 s | 1.600 s | -2.0% |
| Median skin physics | 0.060416 ms | 0.063488 ms | +0.003072 ms |
| Median total physics GPU | about 2.22 ms | about 2.34 ms | about +0.12 ms |

There were zero inversions, force-cap hits, outside particles, or non-finite
values. Total contact impulse stayed essentially constant (1,447.8 versus
1,450.1), so the quieter result was not obtained by removing contact. Five
complete selected-state traces were bit-identical (SHA-256
`add8ff0f56d663bd32f14e835ed435093ffad653ad22a1505c65a0da2e510634`).

The stiff stress preset (`Particle Repel = 12`, `Skin Spring = 740`) still fails:
the selected contact reduces its vibration and penetration, but as many as 1,206
particles leave the skin. It remains an unsupported material regime rather than
a relaxed acceptance test.

## Deep-fold experiment

The shallow edge trajectory did not reproduce the later reported failure. The
same harness now accepts `--pressed-x`; setting it to `0.45` loads the left box
face deeply enough to form a fold before the vertical edge slide. The finite-
mass box reaches about `x=0.585`, rather than teleporting to the target.

The instrumented former implementation exposed the dominant discontinuity.
Particle contact normals were flipped toward each vertex's undeformed spherical
direction. Nine such reversals occurred in one fold run. The largest associated
internal contact torque was `12.86`, face-hold edge vibration rose to `0.03712`,
and slide vibration rose to `0.03015`. A deformed fold has no reason to remain
aligned with the original sphere, so the override was removed. Degenerate
interpolated normals now fall back to the current triangle normal.

A relative-normal damper was then added between each particle and the
barycentrically interpolated skin velocity. The force is
`max(0, k * violation + c * relative_normal_velocity)` and retains the existing
force cap and deterministic reaction gather. A sweep selected `c=12` as the
best balanced value:

| Candidate | Face-hold RMS | Slide RMS | Edge-hold RMS | Max penetration | Recovery | Decision |
|---|---:|---:|---:|---:|---:|---|
| Former rest-sphere orientation | 0.037122 | 0.030151 | 0.003809 | 0.017350 | 2.800 s | remove |
| Current orientation, no new damping | 0.005096 | 0.002577 | 0.003770 | 0.002578 | 2.783 s | orientation fix passes |
| Damping 8 | 0.003815 | 0.001917 | 0.002401 | 0.002658 | 2.800 s | improved |
| **Damping 12** | **0.003660** | **0.001841** | **0.002215** | **0.002690** | **2.800 s** | **selected** |
| Damping 16 | 0.003594 | 0.001737 | 0.002346 | 0.002715 | 2.800 s | worse hold/peak balance |
| Damping 28 | 0.008419 | 0.003611 | 0.003348 | 0.002793 | 2.800 s | overdamped response becomes noisy |

Across five selected runs, every primary metric was identical (`0%` variation).
Relative to the former fold implementation, face-hold RMS improved `90.1%`,
slide RMS `93.9%`, stationary edge-hold RMS `41.8%`, and maximum penetration
`84.5%`. Contact-local minimum area improved from `0.3990` to `0.7925`, maximum
edge stretch fell from `1.9313` to `1.0336`, outside-particle count fell from
two to zero, and no force cap or non-finite state occurred.

Several attractive hypotheses failed and their production branches were
removed:

- Aligning force exactly with the closest-feature vector eliminated measured
  contact torque, but increased owner changes from about 50,000 to 240,000 and
  worsened face-hold and slide vibration relative to the simpler orientation
  fix.
- Searching all 2,000 triangles selected the opposite sheet in a tight fold.
  It drove 5,512 particles outside, raised physics time to `38.4 ms`, and did
  not recover. Any future global query must preserve surface-side continuity;
  Euclidean nearest distance is insufficient.
- Tapering particle-pair damping at the support radius increased edge-hold RMS
  to `0.00547` and prevented recovery.
- Reducing particle/skin center clearance from `0.05` to `0.025` made the
  visible layer thinner, but raised slide RMS from `0.00184` to `0.00540`,
  increased recovery from `2.80` to `4.65 s`, and allowed one particle outside.

The harness also measures whether the loaded edge is actually backed by fluid.
A skin vertex is supported when a particle center lies within the `0.12`
particle support radius. The selected run maintained `100%` support; the worst
nearest-particle distance improved from `0.0854` without particle/skin damping
to `0.0754` with it. Particle circulation RMS around the four pressed box edges
changed only from `0.02535` to `0.02509`, but peak circulation fell `32.5%`
(`0.2454` to `0.1657`). Thus missing neighbors at the force-support scale was
not the measured cause in this trajectory. The apparent empty layer is partly
the intentional `0.05` center clearance around particles rendered with radius
`0.0225`.

Same-batch unprofiled timing changed from `2.284` to `2.315 ms` median total
physics (`+1.3%`); median skin physics remained `0.063488 ms`. Timings from
earlier batches under different concurrent GPU load are not compared.

## Regression gate

The executable freezes the former-default measurements above as its comparison
reference. `PASS` requires:

- at least 30% lower dynamic corner-vibration RMS;
- at least 30% lower fixed-patch press vector RMS;
- final/initial hold vibration no greater than `1.0`;
- penetration and intersection frames no more than 5% above the reference;
- contact-local minimum relative area and maximum relative stretch no worse
  than 5%;
- persistent recovery no slower than one `1/60 s` tick beyond the reference;
- no inversion, force-cap hit, particle escape, or non-finite state;
- on the recorded RTX 3050 Ti, median skin time within the larger of 5% or
  `0.02 ms` above the reference (reported but not gated on other GPUs);
- nonzero contact and at most 5% variation in primary metrics across repeats.

CTest runs one repetition. Five repetitions are the reproducibility audit.
The negative control `--box-damping 8` exits with `status=FAIL`: its distributed
load improves the stationary hold, but press RMS is `0.0106302`, worse than both
the selected threshold and former default.

The complete CTest suite passes 3/3. Compute Sanitizer memcheck, initcheck, and
synccheck report zero errors for `meshprep-hybrid-tests`; a focused one-frame
racecheck covering the changed force-generation/reduction path reports zero
hazards. The attempted full 240-frame racecheck exceeded the runner lifetime and
is not counted as a pass.

## Reproduce

Measured on an NVIDIA GeForce RTX 3050 Ti Laptop GPU (compute capability 8.6),
driver 590.44.01, CUDA Toolkit 13.1.80, CMake 3.25.1, Release build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j --target meshprep-contact-experiment
./build/meshprep-contact-experiment --runs 5 --iterations 1 \
  --material default --output /tmp/meshprep-contact-experiment
```

The deep-fold gate used for the latest measurements is:

```bash
./build/meshprep-contact-experiment --runs 5 --iterations 1 \
  --pressed-x 0.45 --label fold-selected \
  --output /tmp/meshprep-contact-experiment
```

Use `--verbose` for per-run metrics in addition to the concise summary. The
stiff stress audit is:

```bash
./build/meshprep-contact-experiment --runs 1 --iterations 1 \
  --material stiff --output /tmp/meshprep-contact-experiment
```

It intentionally exits nonzero while that unsupported regime fails. Remove the
temporary traces with this exact command:

```bash
rm -r /tmp/meshprep-contact-experiment/
```

The exploratory deep-fold matrix from this audit is isolated separately:

```bash
rm -r /tmp/meshprep-fold-experiment/
```
