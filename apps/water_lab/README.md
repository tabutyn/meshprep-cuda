# Parallel Mater native gallery

`parallel-mater-lab` is the interactive CUDA/OpenGL client for the installed
Parallel Mater simulation API. The application owns camera, input, objectives,
and rendering; `GallerySimulation` and the component solvers own CUDA state.

## Context catalog

The catalog is organized as the five individual simulations, all fluid pairs,
then the remaining cloth, soft-body, and rope pairs:

| Context | Current fixture |
| --- | --- |
| Water | Fluid in the painted hemisphere bowl |
| Cloth | Rigid sphere and breakable hanging cloth |
| Softbody | Configurable N x M field of hanging blue cylinders |
| Rope | Two equal finite-mass spheres, post, Y rope, and eight-corner cage with twelve slack rope midpoints |
| Smoke | Rolling rigid sphere and a turbulent grey wake |
| Water-Cloth | Water-skin obstacle course |
| Water-Softbody | Water-driven wheel, soft crosses, and 32 rigid outer rungs |
| Water-Rope | 40k-particle fishing tank, reelable hook, and heavy treasure chest |
| Fluid-Smoke | Heated shallow water emitting buoyant steam |
| Cloth-Softbody | Soft sphere, ground cloth, and goal cloth |
| Cloth-Rope | 4 x 10 floor of 6 x 6 tiles with four inset four-segment ropes per edge |
| Cloth-Smoke | Four pitched cloth windmill blades driven by smoke |
| Softbody-Rope | Soft sphere on 16 x 16 cloth tiles with four four-segment ropes per edge |
| Softbody-Smoke | Rolling sphere, green soft bristles, and a crosswind |
| Rope-Smoke | Rigid sphere crossing a wind-loaded rope bridge |

In Water-Rope, left/right move the pinned fishing head and up/down reel the
rope. Touching the chest with the hook latches it; winning requires reeling the
heavy chest to the top. Both the rope and chest participate in the water solve.

In Cloth-Rope and Softbody-Rope, gravity loads only the rigid sphere. The
cloth/rope structure has zero body gravity, so all visible deformation is a
measurable reaction to the sphere rather than sag from its own weight.

## Controls

- `Tab`: open the collapsible, system-color-coded context catalog. Arrow keys
  select a recipe and `Enter` opens it.
- Arrow keys or `WASD`: apply the context's authored control. In Water-Rope,
  arrows control the fishing head and rope length instead of gravity.
- `P`: physics/material controls. Cloth-Softbody exposes the source soft-body
  mass independently of the goal cloth; Cloth-Rope exposes rigid-sphere mass.
- `L`: simulation quantities. Softbody exposes cylinder columns and rows;
  changing either rebuilds the same occupied volume with thinner cylinders.
- `V`: particle/lattice debug view.
- `K`: billboard-water rendering.
- `T`: timings, `R`: reset, `M`: capture, `Esc`: quit.

The simulation advances one fixed `1/60 s` tick per displayed frame. A slow
render therefore slows physical time instead of launching catch-up ticks.

## Build and run

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DPARALLEL_MATER_BUILD_LAB=ON -DPARALLEL_MATER_BUILD_GALLERY=ON
cmake --build build -j --target parallel-mater-lab
./build/parallel-mater-lab
```

The CUDA regression suite is separate from rendering:

```bash
ctest --test-dir build --output-on-failure \
  -R 'meshprep-simulation-(api|runtime|gallery)-tests'
```
