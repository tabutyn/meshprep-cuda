# From a rolling soft body to puddles and foam

Research brief, 2026-09-13. The full free-liquid architecture below remains a
proposal. A [continuous particle-derived surface and separate foam tracers](WATER_VISUAL_EFFECTS.md)
are implemented without changing the existing skin/particle physics.
Cohesion, free topology, airborne spray, and submerged bubble dynamics are not implemented.

## What is stopping a puddle now?

The course is repulsive particles inside a closed spring mesh. In
`apps/water_lab/hybrid_kernels.cu`, `particle_forces_kernel` computes particle
repulsion and particle/skin reactions; `course_shape_kernel` pulls the mesh
toward its rotated rest sphere. That last term deliberately makes a rollable
game object. Adding attraction inside it cannot release droplets or change
the fixed skin topology. Stronger attraction could instead leave an empty shell.

For free water, replace that physical shell with a **render-only surface
derived from the fluid** after a particle-only fixture passes. Do not retain
two competing mechanisms for keeping water together. Preserve the current
course while testing this replacement; do not add another coupled solver.

## Attractive layers: cohesion, pressure, and wetting

Use the existing finite-radius particle neighbor search. The useful conceptual
layers are short-range repulsion to resist crowding, weaker attraction farther
out to keep water together, and zero force beyond the support radius. They are
distance bands, not permanent spring connections. Start from the current
spacing 0.045 and support 0.12 as experimental scales, not molecular units.

Implement smooth, bounded pair interactions, gathering from immutable state
with symmetric pair contributions. Avoid a singular inverse-power attraction.
But **cohesion alone is insufficient**: Akinci et al. show clustering/stringy
artifacts and combine cohesion with curvature-based surface tension and density
correction. Separate water/water cohesion from water/solid adhesion; the latter
controls wetting, not bulk volume. [Akinci et al., 2013](https://cg.informatik.uni-freiburg.de/publications/2013_SIGGRAPHASIA_surfaceTensionAdhesion.pdf)

My recommendation: first measure a particle-only drop on a level floor with
repulsion alone, then add one cohesion law. If bulk density cannot remain
bounded without stiff forces and tiny steps, replace the pressure model with
a fixed-iteration density solve. PBF gives a compact reference; it does not
justify assuming this implementation will automatically be stable or fast.
[Macklin and Müller, 2013](https://mmacklin.com/pbf_sig_preprint.pdf)

Measure settled footprint/height, bulk density p5/median/p95, volume estimate,
detached-particle fraction, residual kinetic energy, obstacle penetration,
same-seed repeatability, and GPU time. Check drop/settle, two-puddle merging,
and a controlled peg impact separately. Publish baseline variability before
choosing numerical pass thresholds. A quiet collapsed clump is not a pass.

## Make particles look like continuous water

Build a compact-support scalar field from nearby particles and render its
isosurface. Reuse the hierarchy for spatial queries and the existing water
reflection/refraction shading. Test a simple isotropic field first;
anisotropic kernels are a later option for smoother sheets and streams.
Rendering smoothing must not move the simulated particles.
[Yu and Turk, 2010](https://faculty.cc.gatech.edu/~turk/my_papers/sph_surfaces.pdf)

A world-space surface suits the existing secondary reflection/refraction
rays. Benchmark extracted triangles versus direct field intersection; a
density field is not a signed distance, so unqualified sphere-tracing steps
would miss surfaces. Measure reconstruction, normal generation, ray tracing,
flicker, and volume independently of physics. No promised frame rate yet.

## Splash foam that floats and fades

Use a fixed-capacity GPU pool of **secondary visual particles**, not another
water-pressure system. Emit near disturbed free surfaces, using local relative
velocity/energy and surface information. Uniform fast translation alone should
not create foam. The reference framework distinguishes airborne spray,
surface foam carried by the fluid, and submerged bubbles with buoyancy/drag;
it omits feedback onto the primary fluid.
[Ihmsen et al., 2012](https://cg.informatik.uni-freiburg.de/publications/2012_CGI_sprayFoamBubbles.pdf)

For the first slice, only surface foam is necessary: sample fluid velocity,
advect each tracer, and keep it close to the **local moving free surface**.
Use the surface field/normal rather than a constant world Y. If bubbles are
added later, their buoyancy follows opposite gravity as the course tilts.
Give foam finite seeded lifetimes, fade opacity near expiry, and compact/recycle
the pool deterministically. Foam must collide with obstacles and disappear
when it loses the surface; never delete water to spawn it. Render white,
rough, mostly opaque flecks, not refractive blue water spheres.

The authors' maintained tooling exposes lifetime, buoyancy, drag, and emission
controls, useful for comparison rather than as mandatory new dependencies.
[SPlisHSPlasH FoamGenerator](https://github.com/InteractiveComputerGraphics/SPlisHSPlasH/blob/master/doc/FoamGenerator.md)

Test no emission from a resting or uniformly translating puddle, emission
during impact, surface-following error, complete decay after emission stops,
obstacle leakage, pool saturation, and separate update/render time. Keep
foam-to-water feedback, bubble collisions, and a full air phase out of scope.

## Smallest implementation sequence

1. Particle-only settling fixture; pressure/cohesion A/B measurements.
2. Particle-derived water surface; retire course shape matching/physical skin
   from the new liquid mode only when the fixture passes.
3. Surface-foam pool, emission, advection, expiry, and rendering.

Each step gets one reproducible recording and a before/after table. A failed
candidate is removed, not retained behind a growing collection of fallbacks.
Faster impacts must be tested separately: four times the actual speed means
sixteen times kinetic energy at unchanged mass, so old-speed stability is not
evidence of high-speed stability.
