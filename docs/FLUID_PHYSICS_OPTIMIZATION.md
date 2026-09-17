# Fluid physics optimization

Measured 2026-09-13 on an NVIDIA GeForce RTX 3050 Ti Laptop GPU, driver
590.44.01, CUDA 13.1.80, Release build for SM 86. Rendered measurements use
960x720, 10 warmups, 120 samples, the course scene at four physics iterations,
surface view, and foam enabled. Profilers were not active unless explicitly
stated.

## Result

The particle-force stage was dominated by 10,000 independent traversals of a
generic particle AABB hierarchy on every substep. All particles have the same
finite support radius, so the retained implementation builds a deterministic
uniform-cell index with cell width equal to that radius. CUB stably sorts
`(cell key, particle ID)` records; each particle then scans the 27 cells that
can overlap its support sphere. The existing meshprep hierarchy remains the
rendering and validation hierarchy.

The cell index preserves exact distance tests, self exclusion, directed force
evaluation, stable particle IDs, and fixed neighbor-cell/particle order. It
adds 240,000 resident bytes. Its sort is included in `REBUILD FLUID HIERARCHY`,
not hidden in `FLUID PHYSICS`.

| Measurement | Before | After | Change |
|---|---:|---:|---:|
| Fluid physics median, original | 9.8668 ms | 2.2804 ms | **-76.9%** |
| Fluid physics median, final exact BVH | 8.4044 ms | 2.2804 ms | **-72.9%** |
| Fluid hierarchy median | 0.9120 ms | 1.1827 ms | +0.2707 ms |
| GPU total median | 20.6144 ms | 14.4568 ms | **-29.9%** |
| Frame wall median | 21.3358 ms | 15.1358 ms | **-29.1%** |
| Resident allocation | 29,355,195 B | 29,595,195 B | +240,000 B |

The hierarchy, GPU-total, wall, and second fluid rows use the final exact-BVH
implementation after its successful traversal-order changes, so they isolate
the cell-index change. The original fluid row is the median of three independent
baseline runs and shows the complete retained improvement.

Final repeated measurements:

| Stage | Repeat medians | Median p5 | Median p95 | Median max |
|---|---|---:|---:|---:|
| Fluid hierarchy | 1.1827, 1.1822, 1.1950 ms | 1.0781 | 1.3408 | 1.4039 |
| Fluid physics | 2.2780, 2.2804, 2.2953 ms | 2.0685 | 3.3987 | 3.5881 |
| GPU total | 14.4568, 14.4168, 14.7005 ms | 12.6207 | 16.8159 | 17.6596 |
| Frame wall | 15.1358, 15.0802, 15.3847 ms | 13.3596 | 17.4690 | 18.3604 |

The p95 still exceeds a 60 Hz frame. This result establishes a median below
16.667 ms; it is not a claim of locked 60 FPS.

## Hypotheses tested

| Change | Fluid median | Decision |
|---|---:|---|
| Original exact BVH traversal | 9.8668 ms | baseline |
| Schedule particles in spatial order | 9.5606 ms | retain principle; cell order now supplies it |
| Seed exact skin search from prior contact triangle | 8.9564 ms | retain |
| Visit nearest skin child first | 8.4044 ms | retain |
| Skip statistics on discarded substeps | 9.8579 ms at that point | remove: noise only |
| Reduce traversal stack from 128 to 64 entries | 8.4284 ms | remove: no improvement |
| Launch force kernel with 128 instead of 256 threads | 8.6902 ms | remove: 3.4% regression |
| Uniform particle cells plus exact distance test | 2.2804 ms | retain |
| Split pair and skin work into separate kernels | 2.8088 ms | remove: 23.2% regression |

The rejected implementations and their selectors were removed rather than
left as dormant fallback branches.

## Stability and correctness

The 1,200-frame, four-iteration course gate passed three times with bit-identical
reported physical results on the same GPU. Each run reported zero outside
particles and zero non-finite values. The former and retained results compare
as follows; the exact course trajectory changes because the deterministic
neighbor summation order changed.

| Gate | Exact BVH | Uniform cells |
|---|---:|---:|
| Maximum edge strain | 1.346291 | 1.346307 |
| Minimum triangle area | 0.000736666 | 0.000863167 |
| Minimum relative triangle area | 0.218333 | 0.243695 |
| Minimum face alignment | 0.233436 | 0.233179 |
| Maximum outside particles | 0 | 0 |
| Particle/board minimum clearance | 0.003873 | 0.002922 |
| Particle/peg minimum clearance | 0.002198 | 0.002196 |

All eight CTests pass, including the headless rectangle-contact stability gate.
Compute Sanitizer memcheck reports zero errors for `meshprep-hybrid-tests`.

## Reproduction

```bash
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/meshprep-course-tests --frames 1200 --iterations 4
./build/meshprep-water-lab --scene course --view surface --foam on \
  --profile 120 --warmups 10
/usr/local/cuda-13.1/bin/compute-sanitizer --tool memcheck \
  --error-exitcode=99 ./build/meshprep-hybrid-tests
```

The post-change Nsight Systems capture is
`/tmp/meshprep-cell-search-audit.nsys-rep`. Profiler-instrumented times are
intentionally excluded from the performance table.
