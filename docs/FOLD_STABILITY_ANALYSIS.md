# Captured fold: geometry regression and next analytical tests

## Confirmed correction

Capture `capture-20260912-171242-frame-645610` exposed an incorrect BC-edge
region test in the particle/triangle closest-point calculation. The test used
`d5 <= d6`; it requires `d5 >= d6`. The fall-through extrapolated the triangle's
interior coordinates, generating negative weights and weights greater than one.
Those weights affected contact position, normal, interpolated velocity, and
the distribution of particle reaction forces to skin vertices.

The production helper now lives in
[`triangle_contact.cuh`](../apps/water_lab/triangle_contact.cuh), shared by the
simulation and [`triangle_contact_tests.cu`](../tests/triangle_contact_tests.cu).
This extracts the existing calculation without adding a second implementation
to the simulation. The only changed geometry rule is the BC inequality.

Two exact captured float-coordinate fixtures are embedded as hexadecimal
literals, so tests do not depend on `/tmp` recordings:

| Input physics frame | Particle / triangle | Former weights | Correct BC weights |
|---|---|---|---|
| 645514 | 2158 / 1996 | (-7.9784, 7.6817, 1.2968) | (0, 0.474862, 0.525138) |
| 645528 | 1564 / 1993 | (-17.6590, 27.6722, -9.0132) | (0, 0.794428, 0.205572) |

The reference projects onto the triangle plane when the projection is inside,
otherwise minimizes distance over three closed segments, in double precision.
It does not repeat the production Voronoi-region branches. Tests cover 1,590
queries: captured inputs, all seven closest-feature regions, six vertex
permutations (including winding reversal), and seeded queries around the two
thin captured triangles. Host and CUDA results must satisfy:

- finite weights, each in [0,1] within `1e-5`, summing to one within `1e-5`;
- reconstructed closest point within `2e-5` world units of the reference;
- sum of magnitudes of reactions distributed from a 55-unit input at most
  `55.001`, excluding hidden amplification through opposite vertex forces.

Red/green verification: the former condition fails fixture zero with a
`0.119214` closest-point error and `932.627` total reaction magnitude. The
corrected implementation passes all queries on host and RTX 3050 Ti CUDA.
The hybrid suite and existing 1,980-tick contact-stability test pass. All four
Compute Sanitizer tools report zero errors for the new geometry test.

```bash
cmake -S . -B build
cmake --build build -j4 --target meshprep-triangle-contact-tests \
  meshprep-hybrid-tests meshprep-contact-experiment meshprep-water-lab
ctest --test-dir build \
  -R 'meshprep-(triangle-contact|hybrid-tests|contact-stability)' --output-on-failure
```

Verified with CUDA 13.1, Release, compute capability 8.6. These are geometry
and existing-scene regressions; the corrected interactive capture trajectory
has not yet been simulated end-to-end.

## Establish the corrected baseline first

Start before contact, for example capture index 185 (physics frame 645436),
and apply subsequent recorded control forces and torque at the saved timestep.
Do not begin from an already folded end state to judge prevention. Measure
the actual achieved box motion because the corrected water reaction may move
the dynamic box differently. Later repeat with recorded targets through the
controller to check the interactive behavior.

Record vertex speed tails and cap hits, reaction magnitudes, triangle area,
edge stretch, nonadjacent triangle intersections, and independently classified
particle containment. Run identical inputs repeatedly; time physics separately
from CPU diagnostics. Use the corrected run as the baseline for every new law.

The old capture's `particles_outside` counter uses the same local sign as the
force law. Its 2,246 reported outside particles do not independently prove
2,246 actual escapes. Likewise, a face pointing toward the mesh centroid or
neighboring face normals differing by more than 90 degrees can indicate valid
concavity. Neither alone proves inversion. Signed volume becomes an algebraic
quantity rather than a reliable occupied-volume measure after self-crossing.

## Ranked hypotheses, each with a discriminating test

### 1. Many contacts make the shared skin response too stiff for 60 Hz

With fixed normals and weights, multiple particle springs and dampers act on
the same skin degrees of freedom. Individually stable coefficients need not
give a stable combined update. For an isolated, active, unsaturated fixture of
N identical particles on one skin vertex, let `a = 1/mp + N/ms`. Without global
drag the semi-implicit update has the stability condition
`h*h*k*a + 2*h*c*a < 4`, with positive stiffness and damping.

For current defaults `h=1/60`, `k=220`, `c=12`, `mp=1`, `ms=2`, the left side is
`0.461111 * (1 + N/2)`. One contact is stable; 16 give `4.15`, beyond the limit.
Including the existing global drags (particle `0.4`, skin `1.2`) in a three-state
linear model gives spectral radius `1.063596` at N=16: a perturbation grows
and changes sign. This is an analytical counterexample with fixed active
contacts around a balanced preload, not a measurement of the droplet. Real
triangle weights, other forces, contact opening, and caps change the result.

Reproduce that calculation (NumPy is an analysis tool, not an app dependency):

```python
import numpy as np
h, k, c, mp, ms = 1/60, 220., 12., 1., 2.
for N in (1, 8, 16, 32):
    # State: relative displacement, particle velocity, skin velocity.
    p = np.exp(-.4*h) * np.array([-h*k/mp, 1-h*c/mp, h*c/mp])
    s = np.exp(-1.2*h) * np.array([h*N*k/ms, h*N*c/ms, 1-h*N*c/ms])
    A = np.stack((np.array([1., 0., 0.]) + h*(p-s), p, s))
    print(N, max(abs(np.linalg.eigvals(A))))
```

Next test: assemble the frozen local contact/spring/damping Jacobian on the
first excited patch of the corrected replay, using its actual masses and
barycentric weights. Measure eigenvalues of the actual timestep update and
verify predicted growth with small perturbations. Compare 1/60 and 1/120 and
vary damping alone. Only if growing modes match the observed excitation, test
an aggregate implicit contact response. Increasing explicit damping blindly
can make this instability worse. The general stiffness/timestep motivation is
discussed by [Baraff and Witkin](https://www.cs.cmu.edu/~baraff/papers/sig98.pdf);
the fixture and numeric values above are derived from this app's equations.

### 2. Independent force and speed caps break action/reaction balance

Particle reactions are emitted before the aggregate particle force is capped;
the gathered skin force is capped separately. The box also has acceleration
and speed caps. Valid barycentric weights do not repair these later mismatches.
Five aligned 55-unit contacts on one vertex request 275 units on each side;
capping only the skin at 240 leaves a 35-unit internal-force residual.

Test: construct a closed particle/skin fixture, disable drag/controller, and
compare total momentum before and after one step against the exact uncapped
impulse ledger. Repeat just below and above each cap. In the real replay,
account separately for controller and drag impulses. If caps occur only after
the first fold, they are an amplifier rather than its initiator. If confirmed,
test a common admitted contact impulse applied to both sides, then measure
whether reduced contact strength causes more penetration.

### 3. Smooth physics normals introduce spurious internal torque

The particle force follows an interpolated vertex normal; its opposite reaction
acts at barycentric contact point q. The contact's net torque is
`(q - p) cross (f*n)`, which vanishes only when n is parallel to p-q. A smooth
normal at a folded edge generally need not satisfy that condition. Under rigid
rotation the relative-normal damper can also respond to unchanged geometry.

Test: use one known closest-feature contact and prescribe a normal tilted by
angle theta. Compare measured torque to `f*|p-q|*sin(theta)`, then impose rigid
rotation and measure damping work. On corrected capture states, compare this
torque with the patch's angular-momentum change. Nonzero torque establishes
a conservation defect, not automatically energy injection. Only if it is
large enough to explain the spin, test a geometric distance-gradient force
with an independently correct side sign; keep smooth normals for rendering.

### 4. Folded-surface ownership or side classification is discontinuous

The search still considers only triangles incident to the nearest vertex. Its
local interpolated-normal sign enables recovery at any distance when positive.
Correct arithmetic does not guarantee a globally closest feature or a correct
inside/outside sign in concave regions.

Test on a closed, nonintersecting concave fixture with known occupancy: compare
exact triangle distances and independent inside/outside classification while
a particle moves continuously across owner boundaries. Measure force-angle
jumps and signed-distance discontinuities; then sample the corrected capture.
[libigl documents separate distance/sign methods](https://libigl.github.io/dox/signed__distance_8h.html)
useful for an independent reference. Self-intersecting states need to be
flagged before treating a sign oracle as an unambiguous physical interior.
If confirmed, repair side/feature handling before adding owner memory, which
could retain an incorrect contact.

### 5. Springs permit folding, while particle clearance can empty thin pockets

An isolated two-triangle hinge can rotate about its shared edge without
changing any of its edge lengths. The skin's edge springs therefore supply no
local bending penalty for that mode; this does not imply a zero-energy global
collapse mode of the entire closed sphere. There is also no skin self-contact.
Two opposing `0.05` particle exclusion layers leave no force-free particle-center
region in a fluid-side gap narrower than `0.10`.

Test hinge energy versus angle analytically. For the real fold, measure
nonadjacent sheet distance/intersection and active outward particle traction
per skin area. Separate inaccessible fluid-side pockets from exterior dents.
Only if these precede the excitation, test one targeted term: weak bending
for curvature collapse, self-contact for sheet crossing, or revised clearance
for empty thin pockets. A global volume force alone cannot guarantee local
filling or prevent crossings.

Prior parameter comparisons in `CONTACT_STABILITY.md` used the defective
closest-point function. They remain historical observations; their causal
rankings require a corrected baseline before reusing them to choose a law.
For a new candidate, retain the original 30% vibration-improvement target,
check per-run geometry/containment and recovery, and report runtime changes.
Analytical agreement selects an experiment; it does not replace these gates.
