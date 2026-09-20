# Water foam optimization

Measured 2026-09-13 on an NVIDIA GeForce RTX 3050 Ti Laptop GPU, driver
590.44.01, CUDA 13.1.80, Release/SM86. Each result is the median of three
independent unprofiled runs at 960x720, with 10 warmups and 120 measured course
frames, four physics iterations, continuous-surface rendering, and foam enabled.

`WATER FOAM` covers scalar-field reconstruction, distribution normals and foam
source, foam advection/emission, audits, and the foam render-hierarchy build. It
does not include tracing the resulting bubbles; that work is in `RAYTRACE`.

## Retained result

| Measurement | Before | After | Change |
|---|---:|---:|---:|
| Water foam median | 5.3049 ms | 2.2889 ms | **-56.9%** |
| GPU total median | 14.4568 ms | 11.4546 ms | **-20.8%** |
| Frame wall median | 15.1358 ms | 12.1333 ms | **-19.8%** |

Final repeated measurements:

| Stage | Repeat medians | Median p5 | Median p95 | Median max |
|---|---|---:|---:|---:|
| Water foam | 2.2682, 2.2889, 2.3001 ms | 1.2607 | 2.8877 | 3.0916 |
| Ray trace | 2.7396, 2.7074, 2.7143 ms | 2.4908 | 3.0147 | 3.1371 |
| GPU total | 11.4546, 11.3917, 11.4883 ms | 11.0659 | 12.5178 | 12.8595 |
| Frame wall | 12.1333, 12.0993, 12.2249 ms | 11.7429 | 13.1904 | 13.5920 |

All three final runs produced 45 active foam records at tick 130 and the same
physical summary: zero outside particles, zero non-finite values, 146.01 average
neighbors, 241 maximum neighbors, mean skin radius 0.74993, and 2.2923% radial
standard deviation.

## Changes kept

### Constant-time hierarchy count validation

The fluid-grid initializer previously used one CUDA thread to walk every
hierarchy node and recount all leaf primitives. The hierarchy builder already
retains this invariant. `parallel_mater::Hierarchy::primitive_count()` now exposes
that host metadata, and `FluidSurface::update` checks it before launching.
Removing the redundant serial device walk reduced the foam median from 5.3049
to 4.9117 ms: **0.3932 ms, or 7.4%**.

### Reuse the simulation's sorted particle cells

The simulation already builds a deterministic CUB-sorted `(cell key, particle
ID)` index after final particle integration. A borrowed `ParticleCellView` now
lets rendering reuse it without allocation, copying, sorting, or ownership
transfer. Scalar-field samples, distribution/source calculations, and live foam
velocity samples search the 27 potentially overlapping cells and retain their
original exact distance tests and weights. Distribution work is scheduled in
cell order for locality but written by stable particle ID.

The original hierarchy path remains as the empty-view fallback for independent
fixtures. The cell view is rejected if its position pointer, count, or actual
cell width does not match the query. An oracle regression compares the complete
cell-generated scalar field and distribution output against the exact hierarchy
path within `2e-5`.

In the immediate A/B measurements, cell reuse reduced the post-audit foam
median from 4.9117 to 2.2584 ms: **2.6533 ms, or 54.0%**. Final repeated timing
landed at 2.2889 ms; the small difference is normal run-to-run variation.

## Changes rejected and removed

| Experiment | Result | Decision |
|---|---|---|
| Compute foam source only for the roughly 2,048 candidate slots | 2.6665 ms foam, +18.1% | Removed. Random candidate order lost enough spatial locality to outweigh less work. |
| Refit the existing 2,048-record foam hierarchy | 1.8923 ms foam, but 5.2201 ms ray trace and 13.6103 ms GPU total | Removed. Initially co-located inactive records produced highly overlapping refit nodes. |
| Merge the surface and visual audit synchronization | 2.2649 ms foam; 11.3702 ms GPU total | Removed. No improvement beyond measurement noise because hierarchy construction remains a required host-decision boundary. |
| Compile separate cell and hierarchy kernel variants | 2.2483 ms foam, about -0.4% | Removed. The gain was below repeat variability and did not justify duplicate launch branches. |

No dormant experiment selectors or experimental policies were retained.

## Validation and reproduction

All eight CTests pass. The visual test exercises both the optimized production
path and the independent hierarchy fallback. Compute Sanitizer memcheck reports
zero errors for `meshprep-hybrid-tests`.

```bash
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/meshprep-water-lab --scene course --view surface --foam on \
  --profile 120 --warmups 10
/usr/local/cuda-13.1/bin/compute-sanitizer --tool memcheck \
  --error-exitcode=99 ./build/meshprep-hybrid-tests
```

Profiler-instrumented runtime is deliberately excluded from the timing table.
