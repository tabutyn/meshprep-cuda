# ParallelMater native gallery study guide

`parallel-mater-lab` is the original native CUDA/OpenGL application, now with
fifteen selectable recipes. It is intentionally a consumer of the installable ParallelMater targets rather
than a second simulation implementation.

## Build boundary

The dependency direction is:

```text
parallel-mater-lab
  -> parallel-mater-gallery-runtime (ray-tracing/fluid presentation kernels)
  -> parallel-mater-example-gallery  (example-only simulation compositions)
  -> ParallelMater::physics          (independent owning solvers)
  -> ParallelMater::geometry         (normals and hierarchy)
```

The target wiring is in `CMakeLists.txt`. The app-local runtime contains only
`fluid_surface.cu`, `fluid_visuals.cu`, and `water_kernels.cu`; cloth, soft-body,
context composition, geometry processing, and hierarchy code come through the
library targets.

## Reading order

1. `examples/gallery/recipes.hpp` declares the example recipe catalog and
   component configuration.
2. `examples/gallery/gallery.hpp` declares the headless example composition
   and its borrowed render views.
3. `examples/gallery/recipes.cpp` implements recipe validation without CUDA or a
   window system.
4. `examples/gallery/gallery.cpp` adapts the scene recipes to the CUDA simulation
   components and exposes `GallerySimulation`.
5. `apps/water_lab/campaign.cpp` owns objectives and auto-progression solely for
   the example gallery.
6. `apps/water_lab/main.cpp` is the native renderer and interaction layer; the
   Tab browser selects a pending recipe that is rebuilt at the frame boundary.
7. `apps/water_lab/simulation_gallery.cpp` contains the advanced native fixture
   assembly still needed by the renderer. It is compiled into
   `parallel-mater-example-gallery`, not the installed API.

For the smallest headless consumer, start with
`examples/simulation_contexts.cu`; it uses the build-tree example headers and
the `parallel-mater-example-gallery` target.

The public `GallerySimulation` is deliberately smaller than the native app. It
owns headless fixed-step simulation and borrowed CUDA render views. Camera,
GLFW input, HUD, captures, ray tracing, and visual debug modes stay in the app.
This keeps game policy out of the general physics API while leaving the full
gallery available as studied integration code.

## Build and inspect

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DPARALLEL_MATER_BUILD_GALLERY=ON \
  -DPARALLEL_MATER_BUILD_LAB=ON
cmake --build build -j --target parallel-mater-lab
./build/parallel-mater-lab
```

Press `1` through `9` to switch contexts. Use `P` for physics controls, `L` for
simulation quantities, `V` for the combined diagnostic view, and `T` for stage
timings.

Only the real `parallel_mater` namespace is installed; the former
`<meshprep/...>` compatibility headers and namespace alias have been removed.
