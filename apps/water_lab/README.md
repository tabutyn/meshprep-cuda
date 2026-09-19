# CUDA water tilt course

The native CUDA/OpenGL app now opens as a playable peg course. A 3,000-particle
fluid droplet inside a spring-connected skin rests on a long board. Tilt gravity
with the keyboard, roll the droplet through eight rigid orange posts,
and reach the goal near the far end.

Water is visible by default as a continuous particle-derived surface, with
reflection/refraction and independent, advected foam patches. Each patch plays
a short 3D bubble growth/spread/pop sequence instead of drawing one white bead.
See [implementation and tests](../../docs/WATER_VISUAL_EFFECTS.md).

The simulation advances exactly one fixed `1/60 s` tick per displayed frame. If
rendering is slow, simulated time lags. It does not launch catch-up ticks. The
course starts at the authored 8× preset, with force, speed, and iteration
limits scaled together. Steering reaches at most 20 degrees and smoothly
returns to level. Bracket controls change acceleration, particle/skin speed
caps, load-bearing material forces, and the substep floor together without
changing the fixed physical time. Matching the force/gravity ratio prevents the
extra gravity at 8x from simply compacting the fluid more than at 1x.

## Course

The board extends from `z=1.8` to `z=-10.8`, with containment rails at
`x=+/-3.4`. Eight capped, rigid cylinders form the slalom. The circular goal is
centered at `z=-9.4` with radius `0.85`. Floor, rails, and posts share one
analytic collision description; the ray tracer uses the same post centers,
radii, and heights. Context 1 therefore has no voxel lattice, fracture pass,
soft-body hierarchy, or water-to-soft-body gather. Soft-body development
continues independently in contexts 4, 6, and 7.

The low visible rails mark full-height containment planes. Context 1 resets to
3,000 particles, physical/render skin frequency `10`/`10`, and particle
repulsion `20`; the live repel range is `0`–`120`. Course-only material settings
start with particle–skin stiffness `8000`, skin
spring stiffness `560`, and particle/skin force caps `960`. Bracket motion scales
these forces with gravity so equilibrium compression is approximately consistent
between multipliers. Historical 4× testing found that fewer than four iterations
failed the water geometry gates, so course mode retains four as its minimum. See
[course measurements](../../docs/TILT_COURSE.md),
[fluid-physics optimization results](../../docs/FLUID_PHYSICS_OPTIMIZATION.md),
[foam-query optimization results](../../docs/WATER_FOAM_OPTIMIZATION.md), and
[puddle/foam research](../../docs/WATER_COHESION_AND_FOAM.md).

The camera begins behind and above the droplet. It follows the droplet while
preserving orbit and pan changes. The compact on-screen panel contains only
state plus the `Z/X/C/V/K/B/R/T/P/L/M/[ ]` shortcuts; course progress, tilt text,
the old title, and verbose mouse instructions were removed.

## Numbered API gallery

Keys `1` through `9` switch at a frame boundary between composable minigames.
Completing a displayed objective holds the completion card for 1.5 seconds and
then advances automatically; direct number-key selection remains available.
They use the same public recipe catalog as `<parallel_mater/game.hpp>`:

| Key | Fixture |
| --- | --- |
| `1` | Current water-skin obstacle course |
| `2` | Paint the bowl through persistent per-pixel water contact; four cylinder pegs and an invisible continuation above the rim keep water contained |
| `3` | Start farther from a smooth-shaded cloth carrying a procedural `GOAL` texture; win on the first damaged connection |
| `4` | Paint a checker-textured sphere blue by rolling it through twenty simple blue hanging cylinders below a low open grate |
| `5` | Pilot a boat and 20,000 shallow-spawned particles through two rails into a transparent green goal volume above nine equivalent cloth cells |
| `6` | Roll a checker sphere across the compact outer wheel while 20,000 water particles spawn over five seconds and drive its soft crosses |
| `7` | Roll a blue 1,000-voxel soft sphere into a farther green `GOAL` cloth; contact paints it blue and damage wins |
| `8` | Wrap the checker sphere around the post while a Y branch suspends a glass sphere inside a dodecahedral rope cage |
| `9` | Cross a dynamic MxN rope bridge; edit both dimensions in `P` to benchmark larger grids |

Contexts 2, 5, and 6 have no invisible membrane force. Contexts 2 and 5 use
deterministically reduced particle reactions to move their finite-mass rigid
spheres. Context 5 keeps one cloth perfectly horizontal and pins every node
along four support lines per direction, producing nine flexible cells in a
visible square beam grid. Its closed perimeter and two low visible snake rails
use ceiling-height collision volumes; all nine cloth cells share identical
physics and the goal is an independent transparent volume. Contexts 5 and 6 use direct
particle-to-cloth/soft-cross contact with deterministic vertex gathering. The current
`HybridDroplet` owner still carries a hidden adapter skin allocation and a
small hierarchy/normal cost even where it is not a declared component. The
timing table exposes that cost. Context 1 uses its authored 8× course preset;
context 2 uses `(0,-19.62,0)`, and contexts 3–9 use Earth gravity
`(0,-9.81,0)`. Non-course recipes default to four iterations. No gallery
recipe inherits invisible course rails.
`K` toggles an alternative raster path: particles are projected, stably sorted
back-to-front, and drawn as round semi-transparent billboards. It skips the
implicit surface reconstruction and ray/sphere particle traversal, making it a
useful fallback for thin streams and high particle counts.
Context 3 defaults to friction `5`; cloth contact and every room plane exchange
bounded tangential linear/angular impulse with the sphere, so a rolling ball can
climb briefly when its rotating surface grips a wall.
Contexts 1–9 accept camera-relative arrow-key gravity tilt. Context 7 starts
from rest under vertical gravity; its default friction is `10`, the live range
is `0`–`50`, and friction is applied after graph projection so it remains active
while steering supplies a horizontal gravity component. Ground cloth provides
a visible deformable rolling surface before the two-sided hanging barrier. Its
pinned perimeter rests on the room floor while the free interior spans a
2.6 m square opening and can droop 1.25 m into a finite rendered pit.
Context 8 uses one pinned Y rope, structural and bend links, a direct
mass-weighted attachment for the checker sphere, and a second branch ending in
a dodecahedral cage around a glass surface. The attachment is solved after the
rope graph, so the committed rope endpoint limits sphere travel. Ground contact is
resolved before the mass-shared endpoint attachment, whose upper-hemisphere
direction prevents the rope from pulling a floor-supported sphere downward.
`L` changes rope
node count from 16 to 512 and resets the scene;
`P` exposes solver iterations, bond strength, spring stiffness/damping, drag,
speed, gravity, ground friction, and rigid-sphere mass.
Context 2 starts at 20,000 particles with repulsion `50`. Its bowl is red until
individual persistent paint texels receive actual particle contact; contacted
texels turn blue. Particle/debug view draws native points rather than ray
tracing thousands of water spheres. Its regression records free-surface height in
two unobstructed annuli,
rim escape, finite state, and sphere motion rather than assuming the added
rigid body preserves the earlier particle-only equilibrium.
Context 6 keeps the fluid-driven central fin wheel and moves both soft crosses
and their outer rims outward along the axle, clear of the feed and fins. The
former ladder, tow rope, pulleys, and artificial sphere lift were removed. Two
thin horizontal platforms sit 0.1 m below the wheel crown on the front stage
plane, one on each side, with a central gap exposing the collidable outer rim.
The rigid sphere starts on the left platform. Particle IDs still
recycle at the downhill sink.
Context 5 uses a 5,248-node detail-3 graph as one continuous sheet. Eight cells
are fixed cloth-covered supports; the far-right goal cell retains its live,
tearable interior. Contact uses the uppermost live cloth triangle under
each particle's X/Z point, rather than the closest triangle in 3-D, plus the
triangle's actual normal for the side-wall response. The contact transfers
reactions into cloth voxels. A swept-side guard prevents particles already
below the cloth from being projected through it. The sustained regression now
also includes the dynamic sphere and closed box, so earlier particle-only
counts are not presented as results for this scene.
The cylinder-curtain fixture (4) starts at 8× bond strength and spring stiffness
`80,000`. Each of twenty top caps is fixed just below the ceiling and its bottom is free;
lower the
strength in `P` or with `,` when testing fracture. The wheel crosses use a 16×
break threshold and prescribed axle/rim anchors with dynamic interior nodes.
The hanging-cloth fixtures pin every node of the top hanger row. Contexts 3 and
7 also pin the complete bottom row; context 3 buries it one spacing below the floor, so the sheet crosses the floor
without exposing the anchors as an invisible collision bar. Its heavy sphere
transfers equal-and-opposite reactions to the cloth rather than moving as a
prescribed obstacle.
Context 4 aims the ball through the lower, unsupported part of the hanging
column. The graph uses 16 Jacobi constraint iterations per substep. This is not
a validated fracture-material model.
Context 3 samples impact fracture before spring projection, so raising solve
iterations cannot erase the tear signal. Broken graph edges no longer hide
surface triangles. Context 3 expands its render surface to private triangle
vertices; after a structural side breaks, the face preserves its authored size
and hangs from a surviving edge, or from one vertex when no edge survives.

Cloth and soft bodies start with filled checker surfaces in every context.
`V` switches to one combined view: water particles, green foam particles,
water-skin wires, soft-body surface wires, and the internal lattice. Context 5
uses a faint triangle wire instead of its dense spring lattice so water remains
visible. Green points are interior
voxels, yellow points are surface voxels, and blue points are pinned. Broken
bonds turn red in the lattice view, while every authored surface triangle
remains visible so a tear cannot make cloth material disappear.
Live internal bonds are projected as shaded thin four-sided boxes. Members that
touch an interior voxel are bright green; surface-only members are orange.
All contexts open in their normal filled view; `V` enters the shared diagnostic
wire/particle/lattice view.
The `T` table shows only stages actually run by the selected recipe; it also
labels residual particle-only adapter work. `GPU TOTAL` sums its non-overlapping
CUDA/render stages. `FRAME WALL` covers the update, ray trace, and diagnostic
downloads up to the OpenGL draw; texture upload, overlay drawing, compositor
presentation, and any vsync wait are not included. These live samples are not
a median FPS benchmark.

Controls:

- `WASD` or arrow keys in contexts 1–9: tilt gravity relative to the camera. When the physics panel is
  open, use `WASD` for steering because arrows edit its selected value.
- `[` / `]` in context 1: decrease / increase acceleration, particle/skin speed caps, and
  their load-bearing material forces together by 1× from 1× to 8×, starting
  at 8× in the water course. Other gallery contexts use their own gravity
  control in `P` and ignore these keys. The
  iteration floor follows
  `max(4, multiplier)` to retain bounded travel per substep; manually selected
  extra iterations are preserved. The HUD shows both values. Works while paused
  and with the parameter panel open. Reset retains the selected multiplier;
  playback does not accept these keys. The fixed `1/60 s` physical tick is
  unchanged.
- `,` / `.` in soft-body contexts: divide / multiply fracture strength by
  `1.25` per key event, clamped to
  `0.0625×` through `64×`. This scales the bond-break strain threshold;
  the surface remains attached to its existing vertex bindings after a bond
  breaks. It does not make the spring solver stiffer. Reset repairs broken
  bonds while retaining the selected multiplier.
- Left-drag: orbit.
- `Ctrl` + left-drag: pan.
- Wheel: zoom.
- `R`: reset the droplet, course progress, camera follow, and tilt.
- `Space`: pause.
- `T`: toggle timings.
- `V`: toggle filled surfaces and the combined particle/lattice view in the
  current context. Context 1 shows water particles and its skin wireframe;
  soft-body contexts additionally show simulated voxels and live bonds.
- `1` through `9`: select the nine API gallery examples listed above.
- `F`: toggle surface foam.
- `P`: toggle context-relevant live material and force parameters, including
  gravity, solver iterations, soft-body stiffness/damping/drag/speed, rigid mass,
  bond strength, context-7 ground friction (`0`–`50`), and foam emission,
  size, and lifetime. Water physics supports 1–16 iterations; particle repel
  supports `0`–`120`, and course boundary force supports up to `64,000`.
- `L`: toggle reset-on-change simulation quantities. Particle contexts allow up
  to 100,000 particles, water skins expose physical and render frequencies, and
  deformables expose up to 256 spring solves. Contexts 3 and 5 expose cloth
  detail from 1× through 8× while preserving physical cloth dimensions;
  context 8 exposes 8–512 rope nodes. Every accepted change rebuilds the authored
  scene so high-resolution stress tests do not inherit stale state.
  Soft spring stiffness maps monotonically across `100`–`160,000`; `40,000` is
  the exact full-response point, while higher values perform proportionally
  more Jacobi passes instead of disappearing into a stiffness clamp.
- `M`: pause and save the preceding six seconds under
  `/tmp/meshprep-hybrid-captures/`. Capture v6 includes foam history and the
  active soft bodies' voxel positions, velocities, strength, bond activity,
  pending per-bond fracture damage, and render-triangle activity. Context 1
  has no soft-body payload.
- `Z`, `X`, `C`, `B`: toggle skin normals, obstacle forces,
  particle-to-skin forces, and spring forces.
- `Esc`: quit.

## Lab scene

The rectangle experiment remains available as:

```bash
./build/parallel-mater-lab --scene lab
```

Lab controls retain Shift + left-drag for the dynamic rectangle target and
`Q`/`E` for rectangle torque. The rectangle has finite mass and receives the
equal-and-opposite reactions from skin contact. It is rendered as a wireframe,
while collision uses its complete oriented-box volume.

Both scenes use the same force model:

1. Fluid particles find nearby particles through the rebuilt hierarchy and
   apply bounded short-range repulsion and radial damping.
2. Particles near the boundary exchange equal-and-opposite forces with the
   closest physical skin triangle.
3. The fixed skin graph supplies one-ring springs and damping. A barycentric
   embedding drives the render surface from the physical skin; context 1 now
   defaults both meshes to frequency 10 (2,000 triangles each), while `L` can
   independently raise render detail.
4. Particles and skin use finite masses, force and speed caps, exponential drag,
   and semi-implicit Euler integration.
5. Course mode adds rotation-aware soft shape matching so gravity does not
   collapse the gas-and-spring droplet. A CUB center/covariance reduction finds
   the current best-fit rotation, then a local kernel gently pulls the skin
   toward that rotated shape. The droplet can roll and dent without being tied
   to a world-space orientation. This is a gameplay constraint rather than
   water incompressibility or validated fluid dynamics, and its work is charged
   to `SKIN PHYSICS`.
6. Soft-body contexts separately use fixed spring graphs, fracture state,
   bound render surfaces, and hierarchy refits. None of that work is active in
   the rigid context-1 course.

Press `P` to change the iteration count, particle repulsion and damping,
particle-boundary force, skin springs and damping, force caps, and the lab
rectangle coefficients. Course mode hides the rectangle-only rows. Runtime
changes apply on the next tick without a reset.

## Build and run

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DPARALLEL_MATER_BUILD_LAB=ON -DPARALLEL_MATER_BUILD_TESTS=ON
cmake --build build -j --target parallel-mater-lab meshprep-hybrid-tests
ctest --test-dir build --output-on-failure

./scripts/generate_softbody_assets.sh

./build/parallel-mater-lab
./build/parallel-mater-lab --context 7
./build/parallel-mater-lab --scene lab
./build/parallel-mater-lab --scene course --profile 120 --warmups 10
./build/parallel-mater-lab --scene lab --profile 180 --warmups 10 --drive-box
```

The timing HUD reports fluid and skin hierarchy work, fluid physics, skin
physics, normals, render preparation, ray tracing, GPU total, and frame wall
time. Soft-body contexts additionally report `SOFT PHYSICS`, `SOFT HIERARCHY`,
`SOFT CONTACT`, and `SOFT RENDER`, plus live fracture strength and broken-bond
count. `GPU TOTAL` includes ray tracing. The lab also shows rectangle physics;
course mode hides inactive rows.

## Capture and replay

Press `M` immediately after an interesting event. Replay the resulting capture
without advancing physics:

```bash
./build/parallel-mater-lab --replay /tmp/meshprep-hybrid-captures/capture-TIMESTAMP-frame-N
```

Replay uses `Space` to play or pause recorded frames, Left/Right to step one
frame, Shift + Left/Right to step 60 frames, and `R` to return to the first frame.
Capture v6 stores the scene and gravity plus any active soft-body state. Older
context-1 captures still replay their water, camera, and control history; their
retired deformable-post payload is ignored. Captures written before the course
was added load as lab recordings; their shorter options prefix is read into the
current defaults.

The headless rectangle contact experiment remains available for stability work:

```bash
./build/meshprep-contact-experiment --runs 5 --iterations 1 \
  --material default --output /tmp/meshprep-contact-experiment
```

Its metrics and interpretation are documented in
[`docs/CONTACT_STABILITY.md`](../../docs/CONTACT_STABILITY.md).
