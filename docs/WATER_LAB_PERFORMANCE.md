# Water lab performance

## Workload

- frequency-45 geodesic sphere;
- 20,252 vertices, 40,500 triangles, and 60,750 undirected graph edges;
- eight zero-gravity surface iterations per frame;
- complete deterministic hierarchy and AABB rebuild every frame;
- 960×720 primary rays, with a second hierarchy traversal through water after a hit;
- pinned device-to-host render-target transfer included in wall time;
- NVIDIA GeForce RTX 3050 Ti Laptop GPU, compute capability 8.6, CUDA 13.1.

The standalone profile performs ten warmups followed by 120 measured frames. A localized impulse is applied during warmup so measurements cover a moving, deformed surface.

```bash
./build/meshprep-water-lab --profile 120 --width 960 --height 720 --substeps 8
```

| Stage | Median | p5 | p95 |
| --- | ---: | ---: | ---: |
| Eight physics substeps | 1.425 ms | 1.408 ms | 1.858 ms |
| Hierarchy and bounds rebuild | 1.333 ms | 1.221 ms | 1.594 ms |
| Refractive ray trace | 1.861 ms | 1.724 ms | 2.037 ms |
| Complete headless frame wall time | 5.255 ms | 5.023 ms | 6.216 ms |

The median frame retains 11.41 ms of the 16.67 ms 60 FPS budget. Hierarchy throughput is 30.4 Mtri/s for this small dynamic mesh. OpenGL texture upload and desktop presentation are not included in headless wall time; the interactive stage timings remain visible in the window title.

## Nsight Systems evidence

The instrumented command traces 20 measured frames plus ten warmups:

```bash
nsys profile --trace=cuda,nvtx,osrt --sample=none \
  --output=water-lab-nsys ./build/meshprep-water-lab \
  --profile 20 --width 960 --height 720 --substeps 8
nsys stats --report cuda_gpu_kern_sum,nvtx_sum water-lab-nsys.nsys-rep
```

Instrumentation raises the measured median frame to 5.58 ms. Aggregate kernel evidence:

| Kernel or group | Mean per launch | GPU share | Interpretation |
| --- | ---: | ---: | --- |
| Refractive render kernel | 1.921 ms | 42.2% | Dominant complete-frame kernel |
| Center accumulation | 115.8 µs | 20.4% | Atomic global constraint, once per substep |
| Volume accumulation | 77.7 µs | 13.7% | Atomic triangle reduction, once per substep |
| CSR surface step | 9.43 µs | 1.7% | All vertices and adjacent springs, once per substep |
| Hierarchy CUB radix kernels | 6.89 µs average | 7.3% | Multiple launches per hierarchy level |

The requested all-vertex neighbor solver is not the physics bottleneck: eight launches total roughly 75 µs of kernel execution. Global center/volume constraints dominate physics. A follow-up optimization should replace their atomics with reusable CUB reductions or fuse block reductions before increasing substep count.

Compute Sanitizer memcheck reports zero errors, and racecheck reports zero hazards across the physics, impulse, hierarchy, and ray-tracing path.

## Limits

This is an interactive surface model rather than CFD. Edge springs model tension; a global pressure correction resists volume loss. The pressure direction assumes a star-shaped droplet. There is no self-collision, remeshing, viscosity field, or topology change. Ray traversal uses a fixed 128-entry local stack sized comfortably above the measured hierarchy depth.
