# From nondeterministic atomics to stable CUDA mesh preprocessing

I rebuilt an old set of CUDA geometry kernels as `meshprep-cuda`, an MIT-licensed C++20 library for deterministic normal generation and eight-way AABB hierarchy construction.

The original was fast on small meshes, but identical runs changed leaf ordering and a 9,963,191-triangle scene exceeded CUDA's grid-Y launch limit. The redesign uses CUB radix sort and scans to make `(vertex, triangle, corner)` adjacency and `(parent, octant)` partitioning explicit. Flattened launches now complete San Miguel in a median 239.3 ms on my 4 GB RTX 3050 Ti, while 100-run tests verify bit-identical topology and indexing.

There is no hidden speedup claim: stable ordering costs 34–40% on Sibenik and Sponza versus the legacy path, and a pinned CUDA LBVH is faster still while producing a different binary hierarchy. The performance report includes that regression, memory use, profiler workflow, and the experiments I would run next.

Links to add at publication: source repository, design report, performance report, tests, and reproducible benchmark commands.
