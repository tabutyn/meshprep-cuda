# Design

## Normal pipeline

1. Validate positions, triangle indices, and sharp edges on the device.
2. Compute a normalized face normal and retain the unnormalized cross product as the area vector.
3. Emit three records per triangle and radix-sort by `(vertex, corner)`; corner order encodes triangle order.
4. Canonicalize sharp edges, emit both directions, radix-sort, and construct per-vertex ranges with scans.
5. For each vertex, label connected incident triangle fans. Shared non-sharp edges connect faces; sharing only a point does not.
6. Scan group counts, sum area vectors in stable triangle order, normalize, and emit corner-normal indices.

The per-vertex fan labeling kernel favors simple, reviewable deterministic behavior over pathological high-valence performance. That hotspot is a deliberate roadmap item.

## Hierarchy pipeline

The hierarchy preserves the original eight-way, centroid-mean concept while changing its ordering mechanism.

At each breadth-first level, a segmented reduction computes each parent's centroid mean and extent. Every active primitive receives a 64-bit `(parent, octant)` key. CUB radix sort carries primitive IDs into stable child runs; run-length encoding and scans determine child ranges without atomic scatter. Leaves write their stable primitive range immediately. Branch ranges feed the next level.

Triangle bounds stay device-resident. Leaf bounds reduce their primitive bounds; after construction, branch bounds propagate in reverse level order. Normals are independent of hierarchy construction—consumers gather any attributes through the primitive permutation.

All work kernels use flattened one-dimensional launches. This removes the legacy grid-Y ceiling that rejected San Miguel.

## Determinism boundary

Topology, primitive order, and normal indexing are bit-identical across repeated calls on the same supported environment. Sorting and scans define order; atomics are used only for commutative validation/statistics counters. Cross-environment floating-point bit identity is not promised because compiler and architecture math may differ.

## Complexity and memory

Both pipelines are dominated by linear passes plus radix sorts. The current hierarchy implementation reserves worst-case scratch proportional to triangle count and node output capacity of `2 * triangle_count`; measured San Miguel scratch and output total about 2.84 GiB, excluding mesh input. Output capacities are intentionally reusable across calls.
