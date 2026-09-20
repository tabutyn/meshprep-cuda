# Gallery and native-lab study guide

ParallelMater has two deliberately different example layers.

## Headless public-API gallery

Read these files in order:

1. `examples/gallery/recipes.hpp` defines the 15 composition recipes.
2. `examples/gallery/gallery.hpp` defines renderer-neutral borrowed views.
3. `examples/gallery/gallery.cpp` creates and advances public `Fluid`, `Cloth`,
   `Rope`, `SoftBody`, `RigidBody`, and `Smoke` owners through the common frame
   protocol. It also demonstrates common analytic contacts and persistent point
   paint; focused physics tests cover custom deterministic constraint batches.
4. `examples/simulation_contexts.cu` is the smallest executable consumer.
5. `tests/simulation_runtime_tests.cu` initializes and advances every recipe.

The gallery target has no include path or source dependency on
`apps/water_lab`. It is build-tree-only and is not installed.

## Native CUDA/OpenGL lab

`apps/water_lab/main.cpp` owns the window, camera, controls, objectives,
captures, HUD, and rendering. `parallel-mater-gallery-runtime` contains visual
reconstruction and ray-tracing kernels. `parallel-mater-native-simulations`
contains specialized historical fixtures such as the coupled water skin,
authored wheel, rope cage, and fracture presentation.

Those specialized fixtures are intentionally app-only. They are preserved so
the interactive experiments remain available, but they are no longer compiled
into `ParallelMater::physics` and should not be copied into new consumers.

## Build and inspect

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DPARALLEL_MATER_BUILD_GALLERY=ON \
  -DPARALLEL_MATER_BUILD_LAB=ON
cmake --build build --target \
  parallel-mater-gallery-example parallel-mater-lab
./build/parallel-mater-gallery-example
./build/parallel-mater-lab
```

In the native lab, `Tab` opens the recipe catalog, `P` edits physics values,
`L` edits simulation quantities, `V` toggles diagnostic rendering, and `T`
toggles timings.
