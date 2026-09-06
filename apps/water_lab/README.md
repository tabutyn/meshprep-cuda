# CUDA water lab

A native, clickable zero-gravity water-droplet experiment built directly on `meshprep-cuda`.

The mesh is a frequency-45 geodesic icosphere with exactly 40,500 triangles and 20,252 approximately equidistant vertices. Its 60,750 undirected edges are welded once on the CPU and uploaded as a CSR neighbor graph. Each physics substep launches one thread per vertex; threads gather adjacent positions and velocities without write conflicts. Edge springs provide surface tension, edge-relative damping controls ringing, and a device-resident volume estimate supplies pressure.

Every frame performs this pipeline:

```text
8 × surface substep → rebuild eight-way hierarchy and bounds → refractive ray trace → OpenGL display
```

The ray tracer traverses `meshprep::Hierarchy` iteratively. Primary rays find the water entry surface, refract at an index of refraction of 1.333, traverse the hierarchy again to find the exit surface, and refract back into a procedural star field. Beer–Lambert absorption gives the droplet its blue-green depth.

## Build and launch

Requirements beyond the library are OpenGL and GLFW 3.3.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j --target meshprep-water-lab
./build/meshprep-water-lab
```

Controls:

- left click: ray-pick the current mesh and push a localized impulse into the surface;
- `Space`: pause physics;
- `R`: restore the sphere;
- `+` / `-`: increase or decrease physics substeps;
- `Esc`: quit.

## Profile

Headless mode uses ten warmups, reports median/p5/p95 CUDA-event times for physics, hierarchy construction, and ray tracing, and records end-to-end wall time:

```bash
./build/meshprep-water-lab --profile 120 --width 960 --height 720 --substeps 8
./scripts/profile_water_lab.sh build/meshprep-water-lab
```

The physics solver is a deliberately compact interactive model, not a validated CFD solver. Its pressure direction assumes the droplet remains star-shaped around its center of mass. Self-collision, topology changes, and viscosity fields are outside the current lab.

Measured results and the Nsight kernel breakdown are in [`docs/WATER_LAB_PERFORMANCE.md`](../../docs/WATER_LAB_PERFORMANCE.md).
