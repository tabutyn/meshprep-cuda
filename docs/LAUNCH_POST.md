# ParallelMater: deterministic CUDA geometry and composable physics

ParallelMater began as a recovery of my old CUDA geometry kernels and grew into
an MIT-licensed C++20 library for deterministic normal generation, eight-way
AABB hierarchies, and composable GPU physics owners.

The original geometry code was fast on small meshes, but identical runs changed
leaf ordering and a 9,963,191-triangle scene exceeded CUDA's grid-Y launch
limit. CUB radix sort, scans, and flattened launches made ordering explicit and
removed that scale failure. The newer physics layer adds independent Fluid,
Cloth, Rope, SoftBody, RigidBody, and Smoke owners behind one frame/substep
protocol, plus generic colliders and deterministic constraint gathering.

There is no hidden speedup claim: stable geometry ordering costs 34–40% on
Sibenik and Sponza versus the legacy path, and a pinned CUDA LBVH is faster
while producing a different hierarchy. Measurements, memory use, constraints,
and limitations are recorded beside reproducible commands.

Links to add at publication: source repository, design report, performance report, tests, and reproducible benchmark commands.
