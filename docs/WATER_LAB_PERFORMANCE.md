# Water lab performance

## Workload

- two frequency-32 geodesic spheres moving toward contact;
- 20,484 total vertices, 40,960 total triangles, and 61,440 undirected graph edges;
- eight zero-gravity surface iterations per frame;
- continuous vertex/OBB collision and eight triangle/OBB contact projections per surface iteration;
- public `parallel_mater::compute_normals` face, smoothing-group, and vertex-normal pipeline every frame;
- complete deterministic hierarchy and AABB rebuild every frame;
- 960×720 primary rays with barycentrically interpolated normals, an analytic solid box, and a second hierarchy traversal through water after a hit;
- pinned device-to-host render-target transfer included in wall time;
- NVIDIA GeForce RTX 3050 Ti Laptop GPU, compute capability 8.6, CUDA 13.1.

The standalone profile performs ten warmups followed by 120 measured frames. Both droplets retain their initial inward velocity, so measurements cover moving geometry.

```bash
./build/meshprep-water-lab --profile 120 --width 960 --height 720 --substeps 8
```

| Stage | Median | p5 | p95 |
| --- | ---: | ---: | ---: |
| Eight physics/collision substeps | 2.237 ms | 2.213 ms | 2.842 ms |
| Face and vertex normals | 0.304 ms | 0.285 ms | 0.378 ms |
| Hierarchy and bounds rebuild | 1.343 ms | 1.291 ms | 1.580 ms |
| Smooth refractive ray trace | 1.848 ms | 1.830 ms | 2.403 ms |
| Complete headless frame wall time | 6.397 ms | 6.264 ms | 7.756 ms |

The median frame retains 10.270 ms of the 16.67 ms 60 FPS budget. Hierarchy throughput is 30.501 Mtri/s for this small dynamic mesh. The normals output contains 20,484 vertex normals and reports zero degenerate triangles. The final collision audit reports zero vertices inside and zero triangles intersecting the aside OBB. OpenGL texture upload, timing overlay, and desktop presentation are not included in headless wall time.

## Moving-box stress profile

`--stress-box` drives the OBB through the droplets while translating and rotating it, and audits vertex containment and triangle overlap after every frame. The solver uses eight contact iterations for a stationary box and additional iterations while it moves. With the reduced 0.012-unit contact shell, the final signed-distance volume pass reported a maximum of zero contained vertices throughout the 190-frame run. Triangle/OBB overlap is reported separately because vertex containment and continuous triangle-surface separation are different contracts. A low-resolution 180-frame correctness run produced:

| Stage | Median | p5 | p95 |
| --- | ---: | ---: | ---: |
| Physics and moving-box collision | 3.506 ms | 3.378 ms | 4.378 ms |
| Face and vertex normals | 0.307 ms | 0.280 ms | 0.423 ms |
| Hierarchy and bounds rebuild | 1.366 ms | 1.246 ms | 1.710 ms |
| Smooth refractive ray trace | 0.436 ms | 0.368 ms | 0.522 ms |
| Complete headless frame wall time | 5.690 ms | 5.435 ms | 6.950 ms |

The moving-box median retains 10.977 ms of the 60 FPS budget at the reduced stress resolution. The vertex-volume invariant passes every audited frame. Highly stretched triangles can still intersect the box even when all three vertex centers are outside; this run found up to 286 such triangle intersections. The output reports that unresolved surface metric rather than conflating it with vertex containment.

```bash
./build/meshprep-water-lab --profile 180 --width 160 --height 120 \
  --substeps 8 --stress-box
```

## Prototype fusion event

A 420-frame 64×64 run reached the first cross-component vertex contact. Runtime surgery removed both incident fans, aligned and stitched the equal-size boundary rings, rebuilt and validated the closed-manifold host graph, and uploaded the inactive topology buffers in 19.878 ms wall time. Triangle and edge counts remained 40,960 and 61,440. `parallel_mater::compute_normals` subsequently reported 20,482 active normal groups because the two removed vertex IDs remain inactive in the fixed 20,484-slot pool.

This one-time host implementation satisfies the 33.33 ms correctness target for a 30 Hz event, but not the 2 ms stretch target or a continuous 30 Hz graph-update design. Its measurement includes device position download, contact search, surgery, adjacency reconstruction, validation, and upload. The next phase moves detection and graph construction to GPU double buffers and reports graph work independently each second frame.

## Nsight Systems evidence

The instrumented command traces 20 measured frames plus ten warmups:

```bash
nsys profile --trace=cuda,nvtx,osrt --sample=none \
  --output=water-lab-nsys ./build/meshprep-water-lab \
  --profile 20 --width 960 --height 720 --substeps 8
nsys stats --report cuda_gpu_kern_sum,nvtx_sum water-lab-nsys.nsys-rep
```

Instrumentation raises the measured median frame to 8.31 ms. Aggregate kernel evidence:

| Kernel or group | Mean per launch | GPU share | Interpretation |
| --- | ---: | ---: | --- |
| Smooth refractive render kernel | 1.996 ms | 31.2% | Dominant complete-frame kernel |
| Triangle/OBB projection | 21.2 µs | 21.2% | Eight deterministic Jacobi passes per substep |
| Center accumulation | 122.5 µs | 15.3% | Atomic global constraint, once per substep |
| Volume accumulation | 82.4 µs | 10.3% | Atomic triangle reduction, once per substep |
| CSR surface step + swept vertex collision | 12.2 µs | 1.5% | All vertices and adjacent springs, once per substep |
| `parallel_mater::compute_normals` NVTX range | 0.365 ms median | 4.3% | Validation, corner sort, fan grouping, and normal emission |

The requested all-vertex neighbor solver remains inexpensive: eight launches total roughly 98 µs. The collision projection adds about 1.36 ms of raw kernel execution per frame, while global center/volume atomics remain the other substantial physics cost. The general-purpose normal API costs more than a topology-specific gather kernel would, but it exercises the library's deterministic corner-normal contract and remains under 0.3 ms without instrumentation.

Compute Sanitizer memcheck reports zero errors on the two-droplet physics, normal, hierarchy, and ray-tracing path. Earlier single-droplet racecheck runs reported zero hazards. Full sanitizer coverage must be repeated after dynamic topology is implemented.

## Limits

This is an interactive surface model rather than CFD. Edge springs model tension; a global pressure correction resists volume loss. Separate per-component volume constraints are required before fusion is physically correct. Only the first two-component fusion is implemented; there is no general self-collision, remeshing, viscosity field, puncture, or split yet. Ray traversal uses a fixed 128-entry local stack sized comfortably above the measured hierarchy depth.
