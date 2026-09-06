# API contract

The public API is declared in `<meshprep/meshprep.hpp>` under namespace `meshprep`.

Installed-package consumers must enable CUDA as a CMake project language so CUDA headers and runtime link flags are available.

## Views and ownership

`DeviceMeshView` borrows CUDA device pointers to `float3` positions and indexed `uint3` triangles. `SharpEdgeView` borrows device `uint2` pairs. The caller owns these buffers and must keep them valid until the call returns.

`Workspace`, `NormalOutput`, and `Hierarchy` own their device allocations. They are movable, not copyable. Capacity is retained and reused. A workspace or output must not participate in overlapping calls.

`NormalOutput` contains one normalized face normal per triangle, a compact array of area-weighted vertex normals, and one compact-normal index per triangle corner. Degenerate triangles emit a zero face normal, contribute no area vector, and increment `degenerate_triangle_count`.

`Hierarchy` contains breadth-first nodes and a stable permutation of input triangle IDs. A branch owns the contiguous range `[first_child, first_child + child_count)`. A leaf owns `[first_primitive, first_primitive + primitive_count)` in `primitive_indices()`.

## Ordering and sharp edges

Adjacency is sorted by `(vertex, triangle, corner)`. Smooth incident triangle fans are connected through shared edges. A canonical edge `(min(a,b), max(a,b))` in `SharpEdgeView` breaks that connection. Duplicate and reversed edge records have the same effect as one record. Stable vertex order and the smallest incident corner in each component define compact normal IDs.

Hierarchy construction recursively classifies triangle centroids against each parent segment's mean centroid. Octant order is numeric `[0, 7]`; CUB radix sorting preserves primitive-ID order within equal keys. Identical centroids use a stable rank fallback so subdivision terminates.

## Streams and errors

Both public operations enqueue work on the supplied stream, then synchronize that stream before returning. A null stream uses CUDA's default-stream semantics. Results are ready for use on successful return.

No public operation throws. `Status::code`, `Status::cuda_error`, and `Status::message` report validation, allocation, CUDA, or invariant failures. A failed call resets output statistics to zero; retained allocation capacity remains reusable.

## Validation

Calls reject null or empty mesh buffers, counts outside v0.1's 32-bit domain, non-finite position components, out-of-range triangle indices, null sharp-edge storage with a nonzero count, self edges, out-of-range edge endpoints, and a hierarchy leaf capacity outside `[1, 32]`.
