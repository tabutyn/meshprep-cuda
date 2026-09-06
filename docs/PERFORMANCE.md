# Performance report

## Question

Can stable GPU ordering fix nondeterministic topology and the 10M-triangle launch failure without hiding cost or changing the original eight-way spatial-partition idea?

The answer for v0.1.0 is yes for correctness and scale, with a measurable runtime and memory cost.

## Environment and method

- NVIDIA GeForce RTX 3050 Ti Laptop GPU, 4096 MiB, compute capability 8.6;
- driver 590.44.01; CUDA Toolkit 13.1.80;
- Linux 6.1 x86-64; Release build;
- OBJ inputs uploaded once before measurement;
- output and reusable workspace allocated by warmups;
- five warmups, 30 recorded samples;
- CUDA events measure each synchronous operation; `steady_clock` records end-to-end call wall time;
- percentiles use inclusive linear interpolation.

Commands:

```bash
./scripts/fetch_benchmark_data.sh
./scripts/run_benchmarks.sh build results.csv
python3 benchmarks/summarize.py results.csv
```

Scenes are not distributed by this project. Their hashes appear in `benchmarks/data/MANIFEST.md`.

## v0.1 hierarchy results

| Scene | Triangles | median | p5 | p95 | median throughput | Nodes | Max depth |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Sibenik | 73,564 | 2.349 ms | 2.290 ms | 2.479 ms | 31.3 Mtri/s | 30,801 | 8 |
| Sponza | 262,267 | 6.708 ms | 5.789 ms | 6.810 ms | 39.1 Mtri/s | 108,377 | 8 |
| San Miguel | 9,963,191 | 239.265 ms | 238.910 ms | 240.019 ms | 41.6 Mtri/s | 4,154,881 | 13 |

San Miguel uses 2,195,534,848 bytes of reusable workspace and 836,908,044 bytes of retained output. It completes within the GPU's 4 GB VRAM. A procedurally generated mesh with the exact same triangle count completed in 234.728 ms in a one-sample scale check.

## Legacy comparison

The preserved legacy builder measured the complete synchronous `BuildAABB` call under the same five-warmup/30-sample protocol:

| Scene | Legacy median (p5–p95) | v0.1 median | Observation |
| --- | ---: | ---: | --- |
| Sibenik | 1.747 ms (1.712–1.908) | 2.349 ms | v0.1 is 34% slower |
| Sponza | 4.776 ms (4.684–4.924) | 6.708 ms | v0.1 is 40% slower |
| San Miguel | fails at launch | 239.265 ms | v0.1 removes the grid-Y failure |

Legacy output changed across identical runs: Sibenik produced 30,378–30,382 total AABBs in this run; Sponza produced 108,202–108,211. v0.1 emitted the same topology and primitive permutation across all samples and the dedicated 100-run test.

These are related but not equivalent algorithms. Legacy construction carries normal indices and uses atomic scatter order; v0.1 emits a deterministic primitive permutation and gathers attributes separately. The comparison demonstrates the tradeoff, not a speedup.

The pinned `nolmoonen/cuda-lbvh` smoke reference at commit `605802671beb6473b74a43552168f61e63af46db` measured roughly 0.454–0.460 ms on Sibenik. LBVH uses Morton ordering and emits a binary hierarchy, so it is a reference point rather than a drop-in contract comparison.

## Profiler evidence and next experiments

NVTX brackets both public operations and the validation, centroid generation, reduction, partition, scan, packing, and bounds-propagation stages. Reproducible capture commands are in `scripts/profile_nsys.sh` and `scripts/profile_ncu.sh`.

One profiled Sibenik warmup plus one recorded build issued 502 kernels, or 251 per build. In aggregate GPU time, CUB radix-sort kernels accounted for 24.4%, segmented reduction for 20.4%, and the statistics-gather kernel for 16.6%. No `cudaMalloc` occurred in the recorded operation after warmup; the five allocations in the complete trace belong to input and first-use setup. The profile also exposed 33 stream synchronizations per build, driven by synchronous per-level control and bottom-up bounds propagation.

The current evidence points to repeated per-level global sort/reduction and host-visible level control as the hierarchy cost. The next controlled experiments are persistent or batched level scheduling, narrower carried state, dynamic output growth instead of worst-case node capacity, and high-valence normal grouping using segmented graph primitives. Results should be reported even when an experiment regresses.

No Nsight Compute metrics are claimed in v0.1.0 because `ncu` was not installed on the validation host. The capture script and target-kernel workflow are included for reproduction.
