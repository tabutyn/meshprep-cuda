# Simulation API architecture

Parallel Mater separates reusable CUDA simulation ownership from the native
gallery application.

## Layers

1. `ParallelMater::geometry` builds normals and deterministic hierarchies.
2. `ParallelMater::physics` installs independent owning `Fluid`, `Cloth`,
   `Rope`, `SoftBody`, `RigidBody`, and `Smoke` solvers plus the shared frame
   protocol and completion token.
3. `parallel-mater-example-recipes` is a build-tree-only catalog of fifteen
   demonstrations.
4. `parallel-mater-example-gallery` adapts those recipes behind example-only
   aggregate render views.
5. `parallel-mater-lab` supplies GLFW input, OpenGL/CUDA rendering, HUD,
   capture/playback, authored objectives, progression, and minigame controls.

The gallery recipes do not have independent solver implementations. They
compose Water, Cloth, Softbody, Rope, and Smoke with rigid fixtures. Catalog
order groups individual simulations first, then fluid pairs, then the remaining
cloth, soft-body, and rope pairs:

| Recipe | Components |
| --- | --- |
| Water | fluid + rigid |
| Cloth | cloth + rigid |
| Softbody | softbody + rigid |
| Rope | rope + rigid |
| Smoke | smoke + rigid |
| Water-Cloth | fluid + water skin + rigid |
| Water-Softbody | fluid + softbody + rigid |
| Water-Rope | fluid + rope + rigid |
| Fluid-Smoke | fluid + smoke |
| Cloth-Softbody | cloth + softbody + rigid |
| Cloth-Rope | cloth tiles + rope links + rigid |
| Cloth-Smoke | cloth + smoke + rigid |
| Softbody-Rope | softbody + rope bridge + rigid |
| Softbody-Smoke | softbody + smoke + rigid |
| Rope-Smoke | rope + smoke + rigid |

## Ownership and stepping

The installed solver owners each own their state and are movable, not copyable.
They share an ordered prepare/couple/finish substep protocol; applications
exchange impulses through borrowed views without selecting a gallery recipe.
`Completion` provides the asynchronous boundary, while `advance()` waits.

The example-only `GallerySimulation` owns all device memory required by one
authored composition and is movable, not copyable.
`step()` advances exactly one fixed timestep and returns only after the current
public synchronous contract is satisfied. Render views borrow that memory and
must be reacquired after every step or reset.

Particle, surface, rigid-body, and lattice views are data interfaces rather
than rendering commands. Applications may rasterize, ray trace, debug draw, or
ignore any view without changing solver behavior.

The active particle prefix may be resized within reserved capacity. Lattice
topology is fixed until reset: local bond endpoints are shared by every
instance, while activity is stored per instance. Broken bonds remain broken
until reset.

## Recipe-specific authored controls

- Water-Rope uses a closed tank, one vertically authored pinned rope, and a
  heavy finite-mass treasure chest in 40,000 particles. The native app
  translates the pinned head, scales rope rest lengths for reeling, latches
  the hook on contact, and completes only after the chest reaches the top.
- Cloth-Rope procedurally builds an M x N tile floor from 6 x 6-node tiles.
  Four attachment samples are inset from each edge's corners, and every rope
  is one direct tile-to-tile bond with no rope-to-rope joints.
- Softbody-Rope uses 16 x 16-node cloth tiles. Dense structural and shear bonds
  simulate each tile but remain presentation-hidden; only four direct links
  across each neighboring edge render as ropes.
- Rope exposes two equal-size rigid spheres and one lattice. The second sphere
  is physically contacted by face-braced D12 nodes and constrained by its 12
  live face half-spaces; no deforming proxy surface is part of the public view.
- Cloth-Rope and Softbody-Rope apply gravity to the rigid sphere and zero body
  gravity to the structure, isolating rigid-to-deformable load transfer.
- Softbody accepts cylinder row/column overrides. The instance count is
  exactly `rows * columns` and the cylinders become thinner so the authored
  occupied volume stays approximately constant.
- Cloth scales node mass with tessellation density, pins its authored support
  rows, and separates passive load strength from rigid-impact fracture. This
  prevents own-weight tearing while preserving impact damage.

## Error boundary

Construction validates counts, ranges, finite values, and required assets.
Installed initialization, stepping, reset, and resizing return `Status`; implementation
exceptions do not cross the public boundary. CUDA failures retain their CUDA
error code. The native and headless example galleries call the same recipe
factory so their default physics values remain identical, but that factory is
not installed with the library.

See `examples/simulation_contexts.cu` for a renderer-independent client and
`apps/water_lab/README.md` for the interactive catalog and controls.
