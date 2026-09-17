# Nsight guide for the bounded-force lab

The Release build already uses `--generate-line-info`, which maps device work
back to source without device-debug optimization loss. Nsight Systems 2025.5.2
and Nsight Compute 2025.4.0 are installed on the development workstation.

## 1. Establish an unprofiled baseline

```bash
./build/meshprep-water-lab --profile 180 --warmups 10 \
  --width 960 --height 720 --drive-box
```

Use these CUDA-event and wall-clock numbers for performance claims. Profilers
add overhead, and Compute may replay one launch several times.

## 2. Read the frame in Nsight Systems

```bash
./scripts/profile_hybrid_nsys.sh ./build/meshprep-water-lab /tmp/bounded-force
/opt/nvidia/nsight-systems/2025.5.2/host-linux-x64/nsys-ui \
  /tmp/bounded-force.nsys-rep
```

In the timeline:

1. Expand the application thread, CUDA API, CUDA GPU, and NVTX rows.
2. Find `bounded_force/frame`; one range is exactly one `1/60 s` simulation
   tick. There are no nested catch-up ticks.
3. Compare kernel spans with CPU gaps. A `cudaStreamSynchronize` at the end is
   required because the renderer and UI consume the completed state.
4. Group GPU kernels by name. Do not add nested range totals together.
5. Confirm that no `cudaMalloc` occurs inside repeated frames.

The live HUD separates fluid hierarchy, skin hierarchy, fluid physics, skin
physics, rectangle physics, physical normals, and detailed render preparation.
Systems explains launch count and CPU/GPU scheduling inside those totals.

## 3. Explain one kernel in Nsight Compute

The supplied command captures one post-warmup `particle_forces_kernel` launch:

```bash
./scripts/profile_hybrid_ncu.sh ./build/meshprep-water-lab /tmp/bounded-force-particle
/opt/nvidia/nsight-compute/2025.4.0/ncu-ui \
  /tmp/bounded-force-particle.ncu-rep
```

GPU performance counters require administrator permission on this workstation,
so the script elevates only the `ncu` collection command. Open the report as the
normal user.

Review these sections in order:

- **GPU Speed of Light:** whether compute or memory pipelines are saturated.
- **Launch Statistics:** grid size, registers per thread, shared/local memory,
  and theoretical occupancy.
- **Warp State Statistics:** divergence and long-latency stalls caused by the
  two hierarchy traversals.
- **Memory Workload Analysis:** cache hit rates and excess local-memory traffic
  from traversal stacks.
- **Source:** correlate expensive instructions with neighbor and closest-skin
  loops in [`particle_forces_kernel`](../apps/water_lab/hybrid_kernels.cu#L176).

Then edit `--kernel-name` in the script to inspect `skin_spring_forces_kernel`,
`skin_rectangle_forces_kernel`, `embed_render_surface_kernel`, or
`render_kernel`. Capture only one or a few launches; a full-application Compute
run is extremely slow and cannot be interpreted as frame rate.

See [the kernel brief](KERNEL_PROFILING_BRIEF.md) for every input, output, and
source location.
