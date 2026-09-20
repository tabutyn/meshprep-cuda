// SPDX-License-Identifier: MIT
#include <parallel_mater/geometry.hpp>

#include <cub/cub.cuh>
#if MESHPREP_ENABLE_NVTX
#include <nvtx3/nvtx3.hpp>
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <utility>
#include <vector>

namespace parallel_mater {
namespace {

constexpr std::uint32_t block_size = 256;
constexpr float degenerate_epsilon = 1.0e-20F;

class StageRange {
public:
    explicit StageRange(const char* name)
#if MESHPREP_ENABLE_NVTX
        : range_(name)
#endif
    {
#if !MESHPREP_ENABLE_NVTX
        (void)name;
#endif
    }

private:
#if MESHPREP_ENABLE_NVTX
    nvtx3::scoped_range range_;
#endif
};

struct Vec3 {
    float x;
    float y;
    float z;
};

struct Bounds {
    Vec3 min;
    Vec3 max;
};

struct SegmentStats {
    Vec3 sum;
    Vec3 min;
    Vec3 max;
};

struct MergeSegmentStats {
    __host__ __device__ SegmentStats operator()(const SegmentStats& a, const SegmentStats& b) const
    {
        return {
            {a.sum.x + b.sum.x, a.sum.y + b.sum.y, a.sum.z + b.sum.z},
            {fminf(a.min.x, b.min.x), fminf(a.min.y, b.min.y), fminf(a.min.z, b.min.z)},
            {fmaxf(a.max.x, b.max.x), fmaxf(a.max.y, b.max.y), fmaxf(a.max.z, b.max.z)},
        };
    }
};

constexpr SegmentStats segment_identity{
    {0.0F, 0.0F, 0.0F},
    {INFINITY, INFINITY, INFINITY},
    {-INFINITY, -INFINITY, -INFINITY},
};

[[nodiscard]] Status success() noexcept { return {}; }

[[nodiscard]] Status invalid(const char* message) noexcept
{
    return {StatusCode::invalid_argument, cudaSuccess, message};
}

[[nodiscard]] Status invalid_mesh(const char* message) noexcept
{
    return {StatusCode::invalid_mesh, cudaSuccess, message};
}

[[nodiscard]] Status unsupported(const char* message) noexcept
{
    return {StatusCode::unsupported_size, cudaSuccess, message};
}

[[nodiscard]] Status cuda_status(cudaError_t error, const char* message) noexcept
{
    if (error == cudaSuccess) return success();
    return {
        error == cudaErrorMemoryAllocation ? StatusCode::allocation_failure
                                           : StatusCode::cuda_failure,
        error,
        message,
    };
}

template <typename T>
void release(T*& pointer) noexcept
{
    if (pointer != nullptr) cudaFree(pointer);
    pointer = nullptr;
}

template <typename T>
[[nodiscard]] Status ensure_allocation(T*& pointer, std::size_t& capacity, std::size_t count)
{
    if (capacity >= count) return success();
    T* replacement = nullptr;
    const cudaError_t allocation = cudaMalloc(&replacement, sizeof(T) * count);
    if (allocation != cudaSuccess) return cuda_status(allocation, "device allocation failed");
    release(pointer);
    pointer = replacement;
    capacity = count;
    return success();
}

[[nodiscard]] constexpr std::size_t align_up(std::size_t value, std::size_t alignment)
{
    return (value + alignment - 1) & ~(alignment - 1);
}

class ArenaLayout {
public:
    explicit ArenaLayout(void* base = nullptr) : base_(static_cast<std::byte*>(base)) {}

    template <typename T>
    T* take(std::size_t count)
    {
        offset_ = align_up(offset_, alignof(T));
        T* result = base_ == nullptr ? nullptr : reinterpret_cast<T*>(base_ + offset_);
        offset_ += sizeof(T) * count;
        return result;
    }

    void* take_bytes(std::size_t bytes, std::size_t alignment = 256)
    {
        offset_ = align_up(offset_, alignment);
        void* result = base_ == nullptr ? nullptr : base_ + offset_;
        offset_ += bytes;
        return result;
    }

    [[nodiscard]] std::size_t size() const { return align_up(offset_, 256); }

private:
    std::byte* base_{};
    std::size_t offset_{};
};

__host__ __device__ Vec3 to_vec3(float3 value) { return {value.x, value.y, value.z}; }
__host__ __device__ float3 to_float3(Vec3 value) { return make_float3(value.x, value.y, value.z); }

__device__ Vec3 subtract(Vec3 a, Vec3 b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

__device__ Vec3 cross(Vec3 a, Vec3 b)
{
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    };
}

__device__ float length(Vec3 value)
{
    return sqrtf(value.x * value.x + value.y * value.y + value.z * value.z);
}

__device__ Vec3 normalize_or_zero(Vec3 value)
{
    const float magnitude = length(value);
    if (!(magnitude > degenerate_epsilon)) return {0.0F, 0.0F, 0.0F};
    return {value.x / magnitude, value.y / magnitude, value.z / magnitude};
}

__global__ void validate_positions_kernel(
    const float3* positions, std::uint32_t count, std::uint32_t* flags)
{
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float3 p = positions[index];
    if (!isfinite(p.x) || !isfinite(p.y) || !isfinite(p.z)) atomicOr(flags, 1U);
}

__global__ void validate_triangles_kernel(
    const uint3* triangles,
    std::uint32_t count,
    std::uint32_t vertex_count,
    std::uint32_t* flags)
{
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const uint3 triangle = triangles[index];
    if (triangle.x >= vertex_count || triangle.y >= vertex_count || triangle.z >= vertex_count) {
        atomicOr(flags, 2U);
    }
}

__global__ void validate_edges_kernel(
    const uint2* edges, std::uint32_t count, std::uint32_t vertex_count, std::uint32_t* flags)
{
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const uint2 edge = edges[index];
    if (edge.x >= vertex_count || edge.y >= vertex_count || edge.x == edge.y) {
        atomicOr(flags, 4U);
    }
}

[[nodiscard]] Status validate_mesh(
    DeviceMeshView mesh,
    SharpEdgeView edges,
    std::uint32_t* flags,
    cudaStream_t stream)
{
    if (mesh.positions == nullptr || mesh.triangles == nullptr) {
        return invalid("mesh pointers must not be null");
    }
    if (mesh.vertex_count == 0 || mesh.triangle_count == 0) {
        return invalid("mesh must contain vertices and triangles");
    }
    if (mesh.vertex_count >= (std::uint64_t{1} << 32) ||
        mesh.triangle_count >= (std::uint64_t{1} << 32)) {
        return unsupported("v0.1 supports fewer than 2^32 vertices and triangles");
    }
    if (edges.edge_count > 0 && edges.edges == nullptr) {
        return invalid("sharp-edge pointer must not be null when edge_count is nonzero");
    }
    if (edges.edge_count >= (std::uint64_t{1} << 31)) {
        return unsupported("sharp-edge count is too large");
    }

    const auto vertex_count = static_cast<std::uint32_t>(mesh.vertex_count);
    const auto triangle_count = static_cast<std::uint32_t>(mesh.triangle_count);
    const auto edge_count = static_cast<std::uint32_t>(edges.edge_count);
    cudaError_t error = cudaMemsetAsync(flags, 0, sizeof(std::uint32_t), stream);
    if (error != cudaSuccess) return cuda_status(error, "failed to clear validation state");
    validate_positions_kernel<<<(vertex_count + block_size - 1) / block_size, block_size, 0, stream>>>(
        mesh.positions, vertex_count, flags);
    validate_triangles_kernel<<<(triangle_count + block_size - 1) / block_size, block_size, 0, stream>>>(
        mesh.triangles, triangle_count, vertex_count, flags);
    if (edge_count > 0) {
        validate_edges_kernel<<<(edge_count + block_size - 1) / block_size, block_size, 0, stream>>>(
            edges.edges, edge_count, vertex_count, flags);
    }
    error = cudaPeekAtLastError();
    if (error != cudaSuccess) return cuda_status(error, "mesh validation kernel launch failed");
    std::uint32_t host_flags = 0;
    error = cudaMemcpyAsync(&host_flags, flags, sizeof(host_flags), cudaMemcpyDeviceToHost, stream);
    if (error != cudaSuccess) return cuda_status(error, "failed to read validation state");
    error = cudaStreamSynchronize(stream);
    if (error != cudaSuccess) return cuda_status(error, "mesh validation failed");
    if ((host_flags & 1U) != 0) return invalid_mesh("positions must be finite");
    if ((host_flags & 2U) != 0) return invalid_mesh("triangle index is out of range");
    if ((host_flags & 4U) != 0) return invalid_mesh("sharp edge is invalid");
    return success();
}

[[nodiscard]] Status validate_arguments_host(DeviceMeshView mesh, SharpEdgeView edges)
{
    if (mesh.positions == nullptr || mesh.triangles == nullptr) {
        return invalid("mesh pointers must not be null");
    }
    if (mesh.vertex_count == 0 || mesh.triangle_count == 0) {
        return invalid("mesh must contain vertices and triangles");
    }
    if (mesh.vertex_count >= (std::uint64_t{1} << 32) ||
        mesh.triangle_count >= (std::uint64_t{1} << 32)) {
        return unsupported("v0.1 supports fewer than 2^32 vertices and triangles");
    }
    if (edges.edge_count > 0 && edges.edges == nullptr) {
        return invalid("sharp-edge pointer must not be null when edge_count is nonzero");
    }
    if (edges.edge_count >= (std::uint64_t{1} << 31)) {
        return unsupported("sharp-edge count is too large");
    }
    return success();
}

__global__ void face_normals_kernel(
    const float3* positions,
    const uint3* triangles,
    float3* face_normals,
    Vec3* area_normals,
    std::uint32_t count,
    std::uint32_t* degenerate_count)
{
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const uint3 triangle = triangles[index];
    const Vec3 a = to_vec3(positions[triangle.x]);
    const Vec3 b = to_vec3(positions[triangle.y]);
    const Vec3 c = to_vec3(positions[triangle.z]);
    const Vec3 area = cross(subtract(b, a), subtract(c, a));
    area_normals[index] = area;
    const Vec3 normal = normalize_or_zero(area);
    face_normals[index] = to_float3(normal);
    if (normal.x == 0.0F && normal.y == 0.0F && normal.z == 0.0F) {
        atomicAdd(degenerate_count, 1U);
    }
}

__global__ void make_corner_records_kernel(
    const uint3* triangles,
    std::uint64_t* keys,
    std::uint32_t* corners,
    std::uint32_t triangle_count)
{
    const std::uint32_t triangle_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (triangle_index >= triangle_count) return;
    const uint3 triangle = triangles[triangle_index];
    const std::uint32_t vertices[3] = {triangle.x, triangle.y, triangle.z};
    for (std::uint32_t corner = 0; corner < 3; ++corner) {
        const std::uint32_t global_corner = triangle_index * 3U + corner;
        keys[global_corner] =
            (static_cast<std::uint64_t>(vertices[corner]) << 32U) | global_corner;
        corners[global_corner] = global_corner;
    }
}

__global__ void make_edge_records_kernel(
    const uint2* edges, std::uint64_t* keys, std::uint32_t edge_count)
{
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= edge_count) return;
    const uint2 edge = edges[index];
    keys[index * 2U] = (static_cast<std::uint64_t>(edge.x) << 32U) | edge.y;
    keys[index * 2U + 1U] = (static_cast<std::uint64_t>(edge.y) << 32U) | edge.x;
}

__global__ void count_vertices_from_corner_keys_kernel(
    const std::uint64_t* keys, std::uint32_t count, std::uint32_t* vertex_counts)
{
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    atomicAdd(vertex_counts + static_cast<std::uint32_t>(keys[index] >> 32U), 1U);
}

__global__ void count_vertices_from_edge_keys_kernel(
    const std::uint64_t* keys, std::uint32_t count, std::uint32_t* vertex_counts)
{
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    atomicAdd(vertex_counts + static_cast<std::uint32_t>(keys[index] >> 32U), 1U);
}

__device__ bool contains_edge(
    const std::uint64_t* edge_keys,
    std::uint32_t begin,
    std::uint32_t end,
    std::uint64_t wanted)
{
    const std::uint32_t original_end = end;
    while (begin < end) {
        const std::uint32_t middle = begin + (end - begin) / 2U;
        if (edge_keys[middle] < wanted) begin = middle + 1U;
        else end = middle;
    }
    return begin < original_end && edge_keys[begin] == wanted;
}

__device__ std::uint32_t find_root(std::uint32_t* labels, std::uint32_t base, std::uint32_t value)
{
    std::uint32_t root = value;
    while (labels[base + root] != root) root = labels[base + root];
    while (labels[base + value] != value) {
        const std::uint32_t parent = labels[base + value];
        labels[base + value] = root;
        value = parent;
    }
    return root;
}

__device__ void other_vertices(
    uint3 triangle, std::uint32_t corner, std::uint32_t& first, std::uint32_t& second)
{
    if (corner == 0U) {
        first = triangle.y;
        second = triangle.z;
    } else if (corner == 1U) {
        first = triangle.x;
        second = triangle.z;
    } else {
        first = triangle.x;
        second = triangle.y;
    }
}

__global__ void group_vertex_fans_kernel(
    const uint3* triangles,
    const std::uint32_t* sorted_corners,
    const std::uint32_t* corner_offsets,
    const std::uint64_t* sorted_edge_keys,
    const std::uint32_t* edge_offsets,
    std::uint32_t* labels,
    std::uint32_t* group_counts,
    std::uint32_t vertex_count)
{
    const std::uint32_t vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex >= vertex_count) return;
    const std::uint32_t begin = corner_offsets[vertex];
    const std::uint32_t end = corner_offsets[vertex + 1U];
    const std::uint32_t count = end - begin;
    for (std::uint32_t i = 0; i < count; ++i) labels[begin + i] = i;

    const std::uint32_t edge_begin = edge_offsets[vertex];
    const std::uint32_t edge_end = edge_offsets[vertex + 1U];
    for (std::uint32_t i = 0; i < count; ++i) {
        const std::uint32_t corner_a = sorted_corners[begin + i];
        const std::uint32_t triangle_a_index = corner_a / 3U;
        const std::uint32_t local_a = corner_a % 3U;
        std::uint32_t a0 = 0;
        std::uint32_t a1 = 0;
        other_vertices(triangles[triangle_a_index], local_a, a0, a1);
        for (std::uint32_t j = i + 1U; j < count; ++j) {
            const std::uint32_t corner_b = sorted_corners[begin + j];
            const std::uint32_t triangle_b_index = corner_b / 3U;
            const std::uint32_t local_b = corner_b % 3U;
            std::uint32_t b0 = 0;
            std::uint32_t b1 = 0;
            other_vertices(triangles[triangle_b_index], local_b, b0, b1);
            std::uint32_t shared = UINT32_MAX;
            if (a0 == b0 || a0 == b1) shared = a0;
            else if (a1 == b0 || a1 == b1) shared = a1;
            if (shared == UINT32_MAX) continue;
            const std::uint64_t edge_key =
                (static_cast<std::uint64_t>(vertex) << 32U) | shared;
            const bool sharp = edge_begin < edge_end &&
                contains_edge(sorted_edge_keys, edge_begin, edge_end, edge_key);
            if (!sharp) {
                const std::uint32_t root_a = find_root(labels, begin, i);
                const std::uint32_t root_b = find_root(labels, begin, j);
                if (root_a != root_b) {
                    const std::uint32_t low = min(root_a, root_b);
                    const std::uint32_t high = max(root_a, root_b);
                    labels[begin + high] = low;
                }
            }
        }
    }
    std::uint32_t groups = 0;
    for (std::uint32_t i = 0; i < count; ++i) {
        labels[begin + i] = find_root(labels, begin, i);
        if (labels[begin + i] == i) ++groups;
    }
    group_counts[vertex] = groups;
}

__global__ void emit_vertex_normals_kernel(
    const std::uint32_t* sorted_corners,
    const std::uint32_t* corner_offsets,
    const std::uint32_t* labels,
    const std::uint32_t* group_offsets,
    const Vec3* area_normals,
    float3* vertex_normals,
    std::uint32_t* corner_normal_indices,
    std::uint32_t vertex_count)
{
    const std::uint32_t vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex >= vertex_count) return;
    const std::uint32_t begin = corner_offsets[vertex];
    const std::uint32_t count = corner_offsets[vertex + 1U] - begin;
    std::uint32_t ordinal = 0;
    for (std::uint32_t root = 0; root < count; ++root) {
        if (labels[begin + root] != root) continue;
        Vec3 sum{0.0F, 0.0F, 0.0F};
        const std::uint32_t normal_index = group_offsets[vertex] + ordinal++;
        for (std::uint32_t i = 0; i < count; ++i) {
            if (labels[begin + i] != root) continue;
            const std::uint32_t corner = sorted_corners[begin + i];
            const Vec3 area = area_normals[corner / 3U];
            sum.x += area.x;
            sum.y += area.y;
            sum.z += area.z;
            corner_normal_indices[corner] = normal_index;
        }
        vertex_normals[normal_index] = to_float3(normalize_or_zero(sum));
    }
}

[[nodiscard]] Status synchronize(cudaStream_t stream, const char* message)
{
    const cudaError_t launch = cudaPeekAtLastError();
    if (launch != cudaSuccess) return cuda_status(launch, message);
    return cuda_status(cudaStreamSynchronize(stream), message);
}

__global__ void hierarchy_geometry_kernel(
    const float3* positions,
    const uint3* triangles,
    Vec3* centroids,
    Bounds* triangle_bounds,
    std::uint32_t count)
{
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const uint3 triangle = triangles[index];
    const Vec3 a = to_vec3(positions[triangle.x]);
    const Vec3 b = to_vec3(positions[triangle.y]);
    const Vec3 c = to_vec3(positions[triangle.z]);
    centroids[index] = {
        (a.x + b.x + c.x) / 3.0F,
        (a.y + b.y + c.y) / 3.0F,
        (a.z + b.z + c.z) / 3.0F,
    };
    triangle_bounds[index] = {
        {fminf(a.x, fminf(b.x, c.x)), fminf(a.y, fminf(b.y, c.y)),
         fminf(a.z, fminf(b.z, c.z))},
        {fmaxf(a.x, fmaxf(b.x, c.x)), fmaxf(a.y, fmaxf(b.y, c.y)),
         fmaxf(a.z, fmaxf(b.z, c.z))},
    };
}

__global__ void aabb_proxy_kernel(
    const Aabb* bounds,
    float3* positions,
    uint3* triangles,
    std::uint32_t count,
    std::uint32_t* invalid)
{
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const Aabb box = bounds[index];
    const bool valid = isfinite(box.minimum.x) && isfinite(box.minimum.y) &&
        isfinite(box.minimum.z) && isfinite(box.maximum.x) &&
        isfinite(box.maximum.y) && isfinite(box.maximum.z) &&
        box.minimum.x <= box.maximum.x && box.minimum.y <= box.maximum.y &&
        box.minimum.z <= box.maximum.z;
    if (!valid) {
        atomicExch(invalid, 1U);
        return;
    }
    const std::uint32_t base = index * 3U;
    positions[base] = box.minimum;
    positions[base + 1U] = box.maximum;
    positions[base + 2U] = make_float3(
        0.5F * (box.minimum.x + box.maximum.x),
        0.5F * (box.minimum.y + box.maximum.y),
        0.5F * (box.minimum.z + box.maximum.z));
    triangles[index] = make_uint3(base, base + 1U, base + 2U);
}

__global__ void iota_kernel(std::uint32_t* values, std::uint32_t count)
{
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) values[index] = index;
}

__global__ void fill_kernel(std::uint32_t* values, std::uint32_t count, std::uint32_t value)
{
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) values[index] = value;
}

__global__ void gather_segment_stats_kernel(
    const Vec3* centroids,
    const std::uint32_t* primitive_indices,
    SegmentStats* ordered,
    std::uint32_t count)
{
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const Vec3 centroid = centroids[primitive_indices[index]];
    ordered[index] = {centroid, centroid, centroid};
}

__global__ void make_partition_keys_kernel(
    const Vec3* centroids,
    const std::uint32_t* primitive_indices,
    const std::uint32_t* segment_ids,
    const std::uint32_t* segment_offsets,
    const SegmentStats* stats,
    std::uint64_t* keys,
    std::uint32_t count,
    bool force_balanced)
{
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const std::uint32_t segment = segment_ids[index];
    const std::uint32_t begin = segment_offsets[segment];
    const std::uint32_t segment_count = segment_offsets[segment + 1U] - begin;
    const std::uint32_t local = index - begin;
    const std::uint32_t balanced_bucket = min(
        7U,
        static_cast<std::uint32_t>(
            (static_cast<std::uint64_t>(local) * 8U) / segment_count));
    const Vec3 centroid = centroids[primitive_indices[index]];
    const SegmentStats value = stats[segment];
    const Vec3 mean{
        value.sum.x / static_cast<float>(segment_count),
        value.sum.y / static_cast<float>(segment_count),
        value.sum.z / static_cast<float>(segment_count),
    };
    const bool flat_x = value.min.x == value.max.x;
    const bool flat_y = value.min.y == value.max.y;
    const bool flat_z = value.min.z == value.max.z;
    const std::uint32_t x = (force_balanced || flat_x) ? (balanced_bucket & 1U)
                                                       : static_cast<std::uint32_t>(centroid.x > mean.x);
    const std::uint32_t y = (force_balanced || flat_y) ? ((balanced_bucket >> 1U) & 1U)
                                                       : static_cast<std::uint32_t>(centroid.y > mean.y);
    const std::uint32_t z = (force_balanced || flat_z) ? ((balanced_bucket >> 2U) & 1U)
                                                       : static_cast<std::uint32_t>(centroid.z > mean.z);
    const std::uint32_t octant = x | (y << 1U) | (z << 2U);
    keys[index] = (static_cast<std::uint64_t>(segment) << 3U) | octant;
}

__global__ void classify_children_kernel(
    const std::uint32_t* child_counts,
    std::uint32_t* branch_flags,
    std::uint32_t* branch_weights,
    std::uint32_t* leaf_weights,
    std::uint32_t child_count,
    std::uint32_t max_leaf_size)
{
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= child_count) return;
    const std::uint32_t count = child_counts[index];
    const bool branch = count > max_leaf_size;
    branch_flags[index] = branch ? 1U : 0U;
    branch_weights[index] = branch ? count : 0U;
    leaf_weights[index] = branch ? 0U : count;
}

__global__ void collect_totals_kernel(
    const std::uint32_t* branch_flags,
    const std::uint32_t* branch_prefix,
    const std::uint32_t* branch_weights,
    const std::uint32_t* branch_weight_prefix,
    const std::uint32_t* leaf_weights,
    const std::uint32_t* leaf_weight_prefix,
    std::uint32_t count,
    std::uint32_t* totals)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const std::uint32_t last = count - 1U;
    totals[0] = branch_prefix[last] + branch_flags[last];
    totals[1] = branch_weight_prefix[last] + branch_weights[last];
    totals[2] = leaf_weight_prefix[last] + leaf_weights[last];
}

__global__ void count_parent_children_kernel(
    const std::uint64_t* unique_keys,
    std::uint32_t child_count,
    std::uint32_t* parent_counts)
{
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= child_count) return;
    atomicAdd(parent_counts + static_cast<std::uint32_t>(unique_keys[index] >> 3U), 1U);
}

__global__ void set_parent_nodes_kernel(
    const std::uint32_t* parent_node_ids,
    const std::uint32_t* parent_counts,
    const std::uint32_t* parent_prefix,
    HierarchyNode* nodes,
    std::uint32_t parent_count,
    std::uint32_t child_node_base)
{
    const std::uint32_t parent = blockIdx.x * blockDim.x + threadIdx.x;
    if (parent >= parent_count) return;
    HierarchyNode& node = nodes[parent_node_ids[parent]];
    node.first_child = child_node_base + parent_prefix[parent];
    node.child_count = parent_counts[parent];
    node.first_primitive = 0;
    node.primitive_count = 0;
}

__global__ void initialize_child_nodes_kernel(
    const std::uint32_t* child_counts,
    const std::uint32_t* branch_flags,
    const std::uint32_t* branch_prefix,
    const std::uint32_t* branch_weight_prefix,
    const std::uint32_t* leaf_weight_prefix,
    std::uint32_t* next_offsets,
    std::uint32_t* next_node_ids,
    std::uint32_t* all_branch_node_ids,
    HierarchyNode* nodes,
    std::uint32_t child_count,
    std::uint32_t child_node_base,
    std::uint32_t branch_list_base,
    std::uint32_t leaf_primitive_base)
{
    const std::uint32_t child = blockIdx.x * blockDim.x + threadIdx.x;
    if (child >= child_count) return;
    const std::uint32_t node_id = child_node_base + child;
    HierarchyNode node{};
    node.bounds_min = make_float3(INFINITY, INFINITY, INFINITY);
    node.bounds_max = make_float3(-INFINITY, -INFINITY, -INFINITY);
    if (branch_flags[child] != 0U) {
        const std::uint32_t branch = branch_prefix[child];
        next_offsets[branch] = branch_weight_prefix[child];
        next_node_ids[branch] = node_id;
        all_branch_node_ids[branch_list_base + branch] = node_id;
    } else {
        node.first_primitive = leaf_primitive_base + leaf_weight_prefix[child];
        node.primitive_count = child_counts[child];
    }
    nodes[node_id] = node;
}

__global__ void scatter_child_primitives_kernel(
    const std::uint32_t* child_offsets,
    const std::uint32_t* child_counts,
    const std::uint32_t* branch_flags,
    const std::uint32_t* branch_prefix,
    const std::uint32_t* branch_weight_prefix,
    const std::uint32_t* leaf_weight_prefix,
    const std::uint32_t* sorted_primitives,
    std::uint32_t* next_primitives,
    std::uint32_t* next_segment_ids,
    std::uint32_t* final_primitives,
    std::uint32_t child_count,
    std::uint32_t leaf_primitive_base)
{
    const std::uint32_t child = blockIdx.x;
    if (child >= child_count) return;
    const std::uint32_t count = child_counts[child];
    const std::uint32_t source = child_offsets[child];
    if (branch_flags[child] != 0U) {
        const std::uint32_t branch = branch_prefix[child];
        const std::uint32_t destination = branch_weight_prefix[child];
        for (std::uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
            next_primitives[destination + i] = sorted_primitives[source + i];
            next_segment_ids[destination + i] = branch;
        }
    } else {
        const std::uint32_t destination = leaf_primitive_base + leaf_weight_prefix[child];
        for (std::uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
            final_primitives[destination + i] = sorted_primitives[source + i];
        }
    }
}

__global__ void leaf_bounds_kernel(
    const std::uint32_t* child_offsets,
    const std::uint32_t* child_counts,
    const std::uint32_t* branch_flags,
    const std::uint32_t* sorted_primitives,
    const Bounds* triangle_bounds,
    HierarchyNode* nodes,
    std::uint32_t child_count,
    std::uint32_t child_node_base)
{
    const std::uint32_t child = blockIdx.x * blockDim.x + threadIdx.x;
    if (child >= child_count || branch_flags[child] != 0U) return;
    Vec3 minimum{INFINITY, INFINITY, INFINITY};
    Vec3 maximum{-INFINITY, -INFINITY, -INFINITY};
    const std::uint32_t begin = child_offsets[child];
    const std::uint32_t end = begin + child_counts[child];
    for (std::uint32_t i = begin; i < end; ++i) {
        const Bounds bounds = triangle_bounds[sorted_primitives[i]];
        minimum.x = fminf(minimum.x, bounds.min.x);
        minimum.y = fminf(minimum.y, bounds.min.y);
        minimum.z = fminf(minimum.z, bounds.min.z);
        maximum.x = fmaxf(maximum.x, bounds.max.x);
        maximum.y = fmaxf(maximum.y, bounds.max.y);
        maximum.z = fmaxf(maximum.z, bounds.max.z);
    }
    nodes[child_node_base + child].bounds_min = to_float3(minimum);
    nodes[child_node_base + child].bounds_max = to_float3(maximum);
}

__global__ void root_leaf_kernel(
    const Bounds* triangle_bounds,
    HierarchyNode* nodes,
    std::uint32_t count)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    Vec3 minimum{INFINITY, INFINITY, INFINITY};
    Vec3 maximum{-INFINITY, -INFINITY, -INFINITY};
    for (std::uint32_t i = 0; i < count; ++i) {
        const Bounds bounds = triangle_bounds[i];
        minimum.x = fminf(minimum.x, bounds.min.x);
        minimum.y = fminf(minimum.y, bounds.min.y);
        minimum.z = fminf(minimum.z, bounds.min.z);
        maximum.x = fmaxf(maximum.x, bounds.max.x);
        maximum.y = fmaxf(maximum.y, bounds.max.y);
        maximum.z = fmaxf(maximum.z, bounds.max.z);
    }
    HierarchyNode root{};
    root.bounds_min = to_float3(minimum);
    root.bounds_max = to_float3(maximum);
    root.first_primitive = 0;
    root.primitive_count = count;
    nodes[0] = root;
}

__global__ void branch_bounds_kernel(
    const std::uint32_t* branch_node_ids,
    HierarchyNode* nodes,
    std::uint32_t count)
{
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    HierarchyNode& node = nodes[branch_node_ids[index]];
    Vec3 minimum{INFINITY, INFINITY, INFINITY};
    Vec3 maximum{-INFINITY, -INFINITY, -INFINITY};
    for (std::uint32_t i = 0; i < node.child_count; ++i) {
        const HierarchyNode child = nodes[node.first_child + i];
        minimum.x = fminf(minimum.x, child.bounds_min.x);
        minimum.y = fminf(minimum.y, child.bounds_min.y);
        minimum.z = fminf(minimum.z, child.bounds_min.z);
        maximum.x = fmaxf(maximum.x, child.bounds_max.x);
        maximum.y = fmaxf(maximum.y, child.bounds_max.y);
        maximum.z = fmaxf(maximum.z, child.bounds_max.z);
    }
    node.bounds_min = to_float3(minimum);
    node.bounds_max = to_float3(maximum);
}

__global__ void validate_aabb_kernel(
    const Aabb* bounds,
    std::uint32_t count,
    std::uint32_t* invalid_count)
{
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const Aabb box = bounds[index];
    const bool finite = isfinite(box.minimum.x) && isfinite(box.minimum.y) &&
        isfinite(box.minimum.z) && isfinite(box.maximum.x) &&
        isfinite(box.maximum.y) && isfinite(box.maximum.z);
    const bool ordered = box.minimum.x <= box.maximum.x &&
        box.minimum.y <= box.maximum.y && box.minimum.z <= box.maximum.z;
    if (!finite || !ordered) atomicAdd(invalid_count, 1U);
}

__global__ void refit_leaf_bounds_kernel(
    const Aabb* primitive_bounds,
    const std::uint32_t* primitive_indices,
    HierarchyNode* nodes,
    std::uint32_t node_count)
{
    const std::uint32_t node_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (node_index >= node_count || !nodes[node_index].is_leaf()) return;
    HierarchyNode& node = nodes[node_index];
    float3 minimum = make_float3(INFINITY, INFINITY, INFINITY);
    float3 maximum = make_float3(-INFINITY, -INFINITY, -INFINITY);
    for (std::uint32_t item = 0U; item < node.primitive_count; ++item) {
        const Aabb box = primitive_bounds[primitive_indices[node.first_primitive + item]];
        minimum.x = fminf(minimum.x, box.minimum.x);
        minimum.y = fminf(minimum.y, box.minimum.y);
        minimum.z = fminf(minimum.z, box.minimum.z);
        maximum.x = fmaxf(maximum.x, box.maximum.x);
        maximum.y = fmaxf(maximum.y, box.maximum.y);
        maximum.z = fmaxf(maximum.z, box.maximum.z);
    }
    node.bounds_min = minimum;
    node.bounds_max = maximum;
}

template <typename Function>
[[nodiscard]] Status cub_query(std::size_t& maximum, Function&& function, const char* message)
{
    std::size_t bytes = 0;
    const cudaError_t error = function(nullptr, bytes);
    if (error != cudaSuccess) return cuda_status(error, message);
    maximum = std::max(maximum, bytes);
    return success();
}

} // namespace

Workspace::~Workspace()
{
    if (storage_ != nullptr) cudaFree(storage_);
}

Status Workspace::reserve(std::size_t bytes)
{
    if (capacity_bytes_ >= bytes) return success();
    void* replacement = nullptr;
    const cudaError_t allocation = cudaMalloc(&replacement, bytes);
    if (allocation != cudaSuccess) return cuda_status(allocation, "workspace allocation failed");
    if (storage_ != nullptr) cudaFree(storage_);
    storage_ = replacement;
    capacity_bytes_ = bytes;
    return success();
}

Workspace::Workspace(Workspace&& other) noexcept
    : storage_(std::exchange(other.storage_, nullptr)),
      capacity_bytes_(std::exchange(other.capacity_bytes_, 0))
{
}

Workspace& Workspace::operator=(Workspace&& other) noexcept
{
    if (this == &other) return *this;
    if (storage_ != nullptr) cudaFree(storage_);
    storage_ = std::exchange(other.storage_, nullptr);
    capacity_bytes_ = std::exchange(other.capacity_bytes_, 0);
    return *this;
}

NormalOutput::~NormalOutput()
{
    release(face_normals_);
    release(vertex_normals_);
    release(corner_normal_indices_);
}

NormalOutput::NormalOutput(NormalOutput&& other) noexcept
    : face_normals_(std::exchange(other.face_normals_, nullptr)),
      vertex_normals_(std::exchange(other.vertex_normals_, nullptr)),
      corner_normal_indices_(std::exchange(other.corner_normal_indices_, nullptr)),
      face_capacity_(std::exchange(other.face_capacity_, 0)),
      vertex_capacity_(std::exchange(other.vertex_capacity_, 0)),
      corner_capacity_(std::exchange(other.corner_capacity_, 0)),
      statistics_(std::exchange(other.statistics_, {}))
{
}

NormalOutput& NormalOutput::operator=(NormalOutput&& other) noexcept
{
    if (this == &other) return *this;
    release(face_normals_);
    release(vertex_normals_);
    release(corner_normal_indices_);
    face_normals_ = std::exchange(other.face_normals_, nullptr);
    vertex_normals_ = std::exchange(other.vertex_normals_, nullptr);
    corner_normal_indices_ = std::exchange(other.corner_normal_indices_, nullptr);
    face_capacity_ = std::exchange(other.face_capacity_, 0);
    vertex_capacity_ = std::exchange(other.vertex_capacity_, 0);
    corner_capacity_ = std::exchange(other.corner_capacity_, 0);
    statistics_ = std::exchange(other.statistics_, {});
    return *this;
}

Hierarchy::~Hierarchy()
{
    release(nodes_);
    release(primitive_indices_);
    release(branch_node_ids_);
    release(proxy_positions_);
    release(proxy_triangles_);
    release(proxy_validation_);
}

Hierarchy::Hierarchy(Hierarchy&& other) noexcept
    : nodes_(std::exchange(other.nodes_, nullptr)),
      primitive_indices_(std::exchange(other.primitive_indices_, nullptr)),
      branch_node_ids_(std::exchange(other.branch_node_ids_, nullptr)),
      node_capacity_(std::exchange(other.node_capacity_, 0)),
      primitive_capacity_(std::exchange(other.primitive_capacity_, 0)),
      branch_node_capacity_(std::exchange(other.branch_node_capacity_, 0)),
      primitive_count_(std::exchange(other.primitive_count_, 0)),
      proxy_positions_(std::exchange(other.proxy_positions_, nullptr)),
      proxy_triangles_(std::exchange(other.proxy_triangles_, nullptr)),
      proxy_validation_(std::exchange(other.proxy_validation_, nullptr)),
      proxy_position_capacity_(std::exchange(other.proxy_position_capacity_, 0)),
      proxy_triangle_capacity_(std::exchange(other.proxy_triangle_capacity_, 0)),
      statistics_(std::exchange(other.statistics_, {})),
      branch_levels_(std::move(other.branch_levels_))
{
}

Hierarchy& Hierarchy::operator=(Hierarchy&& other) noexcept
{
    if (this == &other) return *this;
    release(nodes_);
    release(primitive_indices_);
    release(branch_node_ids_);
    release(proxy_positions_);
    release(proxy_triangles_);
    release(proxy_validation_);
    nodes_ = std::exchange(other.nodes_, nullptr);
    primitive_indices_ = std::exchange(other.primitive_indices_, nullptr);
    branch_node_ids_ = std::exchange(other.branch_node_ids_, nullptr);
    node_capacity_ = std::exchange(other.node_capacity_, 0);
    primitive_capacity_ = std::exchange(other.primitive_capacity_, 0);
    branch_node_capacity_ = std::exchange(other.branch_node_capacity_, 0);
    primitive_count_ = std::exchange(other.primitive_count_, 0);
    proxy_positions_ = std::exchange(other.proxy_positions_, nullptr);
    proxy_triangles_ = std::exchange(other.proxy_triangles_, nullptr);
    proxy_validation_ = std::exchange(other.proxy_validation_, nullptr);
    proxy_position_capacity_ = std::exchange(other.proxy_position_capacity_, 0);
    proxy_triangle_capacity_ = std::exchange(other.proxy_triangle_capacity_, 0);
    statistics_ = std::exchange(other.statistics_, {});
    branch_levels_ = std::move(other.branch_levels_);
    return *this;
}

const char* status_code_name(StatusCode code) noexcept
{
    switch (code) {
    case StatusCode::success: return "success";
    case StatusCode::invalid_argument: return "invalid_argument";
    case StatusCode::invalid_mesh: return "invalid_mesh";
    case StatusCode::unsupported_size: return "unsupported_size";
    case StatusCode::allocation_failure: return "allocation_failure";
    case StatusCode::cuda_failure: return "cuda_failure";
    case StatusCode::internal_error: return "internal_error";
    }
    return "unknown";
}

Status compute_normals(
    DeviceMeshView mesh,
    SharpEdgeView sharp_edges,
    Workspace& workspace,
    NormalOutput& output,
    cudaStream_t stream)
{
#if MESHPREP_ENABLE_NVTX
    nvtx3::scoped_range function_range{"parallel_mater::compute_normals"};
#endif
    output.statistics_ = {};
    Status status = validate_arguments_host(mesh, sharp_edges);
    if (!status) return status;
    if (mesh.triangle_count > std::numeric_limits<std::uint32_t>::max() / 3U) {
        return unsupported("corner count exceeds 32-bit indexing");
    }
    const auto vertex_count = static_cast<std::uint32_t>(mesh.vertex_count);
    const auto triangle_count = static_cast<std::uint32_t>(mesh.triangle_count);
    const auto edge_count = static_cast<std::uint32_t>(sharp_edges.edge_count);
    const std::uint32_t corner_count = triangle_count * 3U;
    const std::uint32_t edge_record_count = edge_count * 2U;

    status = ensure_allocation(output.face_normals_, output.face_capacity_, triangle_count);
    if (!status) return status;
    status = ensure_allocation(output.vertex_normals_, output.vertex_capacity_, corner_count);
    if (!status) return status;
    status = ensure_allocation(
        output.corner_normal_indices_, output.corner_capacity_, corner_count);
    if (!status) return status;

    std::size_t cub_bytes = 0;
    status = cub_query(
        cub_bytes,
        [&](void* temporary, std::size_t& bytes) {
            return cub::DeviceRadixSort::SortPairs(
                temporary,
                bytes,
                static_cast<std::uint64_t*>(nullptr),
                static_cast<std::uint64_t*>(nullptr),
                static_cast<std::uint32_t*>(nullptr),
                static_cast<std::uint32_t*>(nullptr),
                corner_count,
                0,
                64,
                stream);
        },
        "failed to size corner sort");
    if (!status) return status;
    status = cub_query(
        cub_bytes,
        [&](void* temporary, std::size_t& bytes) {
            return cub::DeviceScan::ExclusiveSum(
                temporary,
                bytes,
                static_cast<std::uint32_t*>(nullptr),
                static_cast<std::uint32_t*>(nullptr),
                vertex_count,
                stream);
        },
        "failed to size prefix scan");
    if (!status) return status;
    if (edge_record_count > 0) {
        status = cub_query(
            cub_bytes,
            [&](void* temporary, std::size_t& bytes) {
                return cub::DeviceRadixSort::SortKeys(
                    temporary,
                    bytes,
                    static_cast<std::uint64_t*>(nullptr),
                    static_cast<std::uint64_t*>(nullptr),
                    edge_record_count,
                    0,
                    64,
                    stream);
            },
            "failed to size edge sort");
        if (!status) return status;
    }

    ArenaLayout sizing;
    sizing.take<std::uint32_t>(2); // validation flags and degenerate count
    sizing.take<Vec3>(triangle_count);
    sizing.take<std::uint64_t>(corner_count);
    sizing.take<std::uint64_t>(corner_count);
    sizing.take<std::uint32_t>(corner_count);
    sizing.take<std::uint32_t>(corner_count);
    sizing.take<std::uint64_t>(std::max(1U, edge_record_count));
    sizing.take<std::uint64_t>(std::max(1U, edge_record_count));
    sizing.take<std::uint32_t>(vertex_count);
    sizing.take<std::uint32_t>(vertex_count + 1U);
    sizing.take<std::uint32_t>(vertex_count);
    sizing.take<std::uint32_t>(vertex_count + 1U);
    sizing.take<std::uint32_t>(corner_count);
    sizing.take<std::uint32_t>(vertex_count);
    sizing.take<std::uint32_t>(vertex_count);
    sizing.take_bytes(cub_bytes);
    status = workspace.reserve(sizing.size());
    if (!status) return status;

    ArenaLayout arena(workspace.storage_);
    std::uint32_t* counters = arena.take<std::uint32_t>(2);
    Vec3* area_normals = arena.take<Vec3>(triangle_count);
    std::uint64_t* corner_keys_a = arena.take<std::uint64_t>(corner_count);
    std::uint64_t* corner_keys_b = arena.take<std::uint64_t>(corner_count);
    std::uint32_t* corner_values_a = arena.take<std::uint32_t>(corner_count);
    std::uint32_t* corner_values_b = arena.take<std::uint32_t>(corner_count);
    std::uint64_t* edge_keys_a = arena.take<std::uint64_t>(std::max(1U, edge_record_count));
    std::uint64_t* edge_keys_b = arena.take<std::uint64_t>(std::max(1U, edge_record_count));
    std::uint32_t* corner_counts = arena.take<std::uint32_t>(vertex_count);
    std::uint32_t* corner_offsets = arena.take<std::uint32_t>(vertex_count + 1U);
    std::uint32_t* edge_counts = arena.take<std::uint32_t>(vertex_count);
    std::uint32_t* edge_offsets = arena.take<std::uint32_t>(vertex_count + 1U);
    std::uint32_t* labels = arena.take<std::uint32_t>(corner_count);
    std::uint32_t* group_counts = arena.take<std::uint32_t>(vertex_count);
    std::uint32_t* group_offsets = arena.take<std::uint32_t>(vertex_count);
    void* cub_temporary = arena.take_bytes(cub_bytes);

    {
        StageRange range{"meshprep/validation"};
        status = validate_mesh(mesh, sharp_edges, counters, stream);
        if (!status) return status;
    }
    cudaError_t error = cudaMemsetAsync(counters + 1, 0, sizeof(std::uint32_t), stream);
    if (error != cudaSuccess) return cuda_status(error, "failed to clear normal statistics");
    face_normals_kernel<<<(triangle_count + block_size - 1) / block_size, block_size, 0, stream>>>(
        mesh.positions,
        mesh.triangles,
        output.face_normals_,
        area_normals,
        triangle_count,
        counters + 1);
    make_corner_records_kernel<<<
        (triangle_count + block_size - 1) / block_size, block_size, 0, stream>>>(
        mesh.triangles, corner_keys_a, corner_values_a, triangle_count);
    error = cudaPeekAtLastError();
    if (error != cudaSuccess) return cuda_status(error, "normal setup kernel launch failed");

    std::size_t bytes = cub_bytes;
    error = cub::DeviceRadixSort::SortPairs(
        cub_temporary,
        bytes,
        corner_keys_a,
        corner_keys_b,
        corner_values_a,
        corner_values_b,
        corner_count,
        0,
        64,
        stream);
    if (error != cudaSuccess) return cuda_status(error, "corner sort failed");
    error = cudaMemsetAsync(corner_counts, 0, sizeof(std::uint32_t) * vertex_count, stream);
    if (error != cudaSuccess) return cuda_status(error, "failed to clear corner counts");
    count_vertices_from_corner_keys_kernel<<<
        (corner_count + block_size - 1) / block_size, block_size, 0, stream>>>(
        corner_keys_b, corner_count, corner_counts);
    bytes = cub_bytes;
    error = cub::DeviceScan::ExclusiveSum(
        cub_temporary, bytes, corner_counts, corner_offsets, vertex_count, stream);
    if (error != cudaSuccess) return cuda_status(error, "corner prefix scan failed");
    error = cudaMemcpyAsync(
        corner_offsets + vertex_count,
        &corner_count,
        sizeof(corner_count),
        cudaMemcpyHostToDevice,
        stream);
    if (error != cudaSuccess) return cuda_status(error, "failed to terminate corner offsets");

    error = cudaMemsetAsync(edge_counts, 0, sizeof(std::uint32_t) * vertex_count, stream);
    if (error != cudaSuccess) return cuda_status(error, "failed to clear edge counts");
    const std::uint64_t* sorted_edge_keys = edge_keys_b;
    if (edge_record_count > 0) {
        make_edge_records_kernel<<<
            (edge_count + block_size - 1) / block_size, block_size, 0, stream>>>(
            sharp_edges.edges, edge_keys_a, edge_count);
        bytes = cub_bytes;
        error = cub::DeviceRadixSort::SortKeys(
            cub_temporary,
            bytes,
            edge_keys_a,
            edge_keys_b,
            edge_record_count,
            0,
            64,
            stream);
        if (error != cudaSuccess) return cuda_status(error, "sharp-edge sort failed");
        count_vertices_from_edge_keys_kernel<<<
            (edge_record_count + block_size - 1) / block_size, block_size, 0, stream>>>(
            edge_keys_b, edge_record_count, edge_counts);
    }
    bytes = cub_bytes;
    error = cub::DeviceScan::ExclusiveSum(
        cub_temporary, bytes, edge_counts, edge_offsets, vertex_count, stream);
    if (error != cudaSuccess) return cuda_status(error, "edge prefix scan failed");
    error = cudaMemcpyAsync(
        edge_offsets + vertex_count,
        &edge_record_count,
        sizeof(edge_record_count),
        cudaMemcpyHostToDevice,
        stream);
    if (error != cudaSuccess) return cuda_status(error, "failed to terminate edge offsets");

    group_vertex_fans_kernel<<<
        (vertex_count + block_size - 1) / block_size, block_size, 0, stream>>>(
        mesh.triangles,
        corner_values_b,
        corner_offsets,
        sorted_edge_keys,
        edge_offsets,
        labels,
        group_counts,
        vertex_count);
    bytes = cub_bytes;
    error = cub::DeviceScan::ExclusiveSum(
        cub_temporary, bytes, group_counts, group_offsets, vertex_count, stream);
    if (error != cudaSuccess) return cuda_status(error, "normal-group prefix scan failed");

    std::uint32_t host_last_offset = 0;
    std::uint32_t host_last_count = 0;
    std::uint32_t host_degenerate_count = 0;
    error = cudaMemcpyAsync(
        &host_last_offset,
        group_offsets + vertex_count - 1U,
        sizeof(host_last_offset),
        cudaMemcpyDeviceToHost,
        stream);
    if (error == cudaSuccess) {
        error = cudaMemcpyAsync(
            &host_last_count,
            group_counts + vertex_count - 1U,
            sizeof(host_last_count),
            cudaMemcpyDeviceToHost,
            stream);
    }
    if (error == cudaSuccess) {
        error = cudaMemcpyAsync(
            &host_degenerate_count,
            counters + 1,
            sizeof(host_degenerate_count),
            cudaMemcpyDeviceToHost,
            stream);
    }
    if (error != cudaSuccess) return cuda_status(error, "failed to read normal statistics");
    error = cudaStreamSynchronize(stream);
    if (error != cudaSuccess) return cuda_status(error, "normal grouping failed");
    const std::uint32_t normal_count = host_last_offset + host_last_count;
    if (normal_count > corner_count) {
        return {StatusCode::internal_error, cudaSuccess, "normal count exceeded output capacity"};
    }
    emit_vertex_normals_kernel<<<
        (vertex_count + block_size - 1) / block_size, block_size, 0, stream>>>(
        corner_values_b,
        corner_offsets,
        labels,
        group_offsets,
        area_normals,
        output.vertex_normals_,
        output.corner_normal_indices_,
        vertex_count);
    status = synchronize(stream, "normal emission failed");
    if (!status) return status;
    output.statistics_ = {normal_count, host_degenerate_count};
    return success();
}

Status build_hierarchy(
    DeviceMeshView mesh,
    HierarchyOptions options,
    Workspace& workspace,
    Hierarchy& output,
    cudaStream_t stream)
{
#if MESHPREP_ENABLE_NVTX
    nvtx3::scoped_range function_range{"parallel_mater::build_hierarchy"};
#endif
    output.statistics_ = {};
    Status status = validate_arguments_host(mesh, {});
    if (!status) return status;
    if (options.max_leaf_size == 0 || options.max_leaf_size > 32) {
        return invalid("max_leaf_size must be in [1, 32]");
    }
    if (mesh.triangle_count > std::numeric_limits<std::uint32_t>::max() / 2U) {
        return unsupported("hierarchy node capacity exceeds 32-bit indexing");
    }
    const auto triangle_count = static_cast<std::uint32_t>(mesh.triangle_count);
    const std::size_t maximum_nodes = static_cast<std::size_t>(triangle_count) * 2U;
    output.primitive_count_ = triangle_count;
    output.branch_levels_.clear();
    status = ensure_allocation(output.nodes_, output.node_capacity_, maximum_nodes);
    if (!status) return status;
    status = ensure_allocation(
        output.primitive_indices_, output.primitive_capacity_, triangle_count);
    if (!status) return status;

    std::size_t cub_bytes = 0;
    status = cub_query(
        cub_bytes,
        [&](void* temporary, std::size_t& bytes) {
            return cub::DeviceRadixSort::SortPairs(
                temporary,
                bytes,
                static_cast<std::uint64_t*>(nullptr),
                static_cast<std::uint64_t*>(nullptr),
                static_cast<std::uint32_t*>(nullptr),
                static_cast<std::uint32_t*>(nullptr),
                triangle_count,
                0,
                64,
                stream);
        },
        "failed to size hierarchy sort");
    if (!status) return status;
    status = cub_query(
        cub_bytes,
        [&](void* temporary, std::size_t& bytes) {
            return cub::DeviceRunLengthEncode::Encode(
                temporary,
                bytes,
                static_cast<std::uint64_t*>(nullptr),
                static_cast<std::uint64_t*>(nullptr),
                static_cast<std::uint32_t*>(nullptr),
                static_cast<std::uint32_t*>(nullptr),
                triangle_count,
                stream);
        },
        "failed to size hierarchy run-length encoding");
    if (!status) return status;
    status = cub_query(
        cub_bytes,
        [&](void* temporary, std::size_t& bytes) {
            return cub::DeviceScan::ExclusiveSum(
                temporary,
                bytes,
                static_cast<std::uint32_t*>(nullptr),
                static_cast<std::uint32_t*>(nullptr),
                triangle_count,
                stream);
        },
        "failed to size hierarchy scan");
    if (!status) return status;
    status = cub_query(
        cub_bytes,
        [&](void* temporary, std::size_t& bytes) {
            return cub::DeviceSegmentedReduce::Reduce(
                temporary,
                bytes,
                static_cast<SegmentStats*>(nullptr),
                static_cast<SegmentStats*>(nullptr),
                triangle_count,
                static_cast<std::uint32_t*>(nullptr),
                static_cast<std::uint32_t*>(nullptr),
                MergeSegmentStats{},
                segment_identity,
                stream);
        },
        "failed to size segmented reduction");
    if (!status) return status;

    ArenaLayout sizing;
    sizing.take<std::uint32_t>(4); // validation, run count, three totals (overlap intentional below)
    sizing.take<Vec3>(triangle_count);
    sizing.take<Bounds>(triangle_count);
    sizing.take<std::uint32_t>(triangle_count);
    sizing.take<std::uint32_t>(triangle_count);
    sizing.take<std::uint32_t>(triangle_count);
    sizing.take<std::uint32_t>(triangle_count);
    sizing.take<std::uint64_t>(triangle_count);
    sizing.take<std::uint64_t>(triangle_count);
    sizing.take<std::uint64_t>(triangle_count);
    sizing.take<std::uint32_t>(triangle_count);
    sizing.take<std::uint32_t>(triangle_count + 1U);
    sizing.take<std::uint32_t>(triangle_count + 1U);
    sizing.take<std::uint32_t>(triangle_count);
    sizing.take<std::uint32_t>(triangle_count);
    sizing.take<std::uint32_t>(triangle_count);
    sizing.take<std::uint32_t>(triangle_count);
    sizing.take<std::uint32_t>(triangle_count);
    sizing.take<std::uint32_t>(triangle_count);
    sizing.take<std::uint32_t>(triangle_count);
    sizing.take<std::uint32_t>(triangle_count);
    sizing.take<std::uint32_t>(triangle_count);
    sizing.take<std::uint32_t>(triangle_count);
    sizing.take<std::uint32_t>(triangle_count);
    sizing.take<SegmentStats>(triangle_count);
    sizing.take<SegmentStats>(triangle_count);
    sizing.take<std::uint32_t>(triangle_count);
    sizing.take_bytes(cub_bytes);
    status = workspace.reserve(sizing.size());
    if (!status) return status;

    ArenaLayout arena(workspace.storage_);
    std::uint32_t* scalar = arena.take<std::uint32_t>(4);
    std::uint32_t* validation_flags = scalar;
    std::uint32_t* device_run_count = scalar + 1;
    std::uint32_t* device_totals = scalar + 1; // run count no longer needed when totals are written
    Vec3* centroids = arena.take<Vec3>(triangle_count);
    Bounds* triangle_bounds = arena.take<Bounds>(triangle_count);
    std::uint32_t* primitives_a = arena.take<std::uint32_t>(triangle_count);
    std::uint32_t* primitives_b = arena.take<std::uint32_t>(triangle_count);
    std::uint32_t* segment_ids = arena.take<std::uint32_t>(triangle_count);
    std::uint64_t* keys_a = arena.take<std::uint64_t>(triangle_count);
    std::uint64_t* keys_b = arena.take<std::uint64_t>(triangle_count);
    std::uint64_t* unique_keys = arena.take<std::uint64_t>(triangle_count);
    std::uint32_t* child_counts = arena.take<std::uint32_t>(triangle_count);
    std::uint32_t* offsets_a = arena.take<std::uint32_t>(triangle_count + 1U);
    std::uint32_t* offsets_b = arena.take<std::uint32_t>(triangle_count + 1U);
    std::uint32_t* node_ids_a = arena.take<std::uint32_t>(triangle_count);
    std::uint32_t* node_ids_b = arena.take<std::uint32_t>(triangle_count);
    std::uint32_t* branch_flags = arena.take<std::uint32_t>(triangle_count);
    std::uint32_t* branch_prefix = arena.take<std::uint32_t>(triangle_count);
    std::uint32_t* branch_weights = arena.take<std::uint32_t>(triangle_count);
    std::uint32_t* branch_weight_prefix = arena.take<std::uint32_t>(triangle_count);
    std::uint32_t* leaf_weights = arena.take<std::uint32_t>(triangle_count);
    std::uint32_t* leaf_weight_prefix = arena.take<std::uint32_t>(triangle_count);
    std::uint32_t* parent_counts = arena.take<std::uint32_t>(triangle_count);
    std::uint32_t* parent_prefix = arena.take<std::uint32_t>(triangle_count);
    std::uint32_t* child_offsets = arena.take<std::uint32_t>(triangle_count);
    SegmentStats* ordered_stats = arena.take<SegmentStats>(triangle_count);
    SegmentStats* segment_stats = arena.take<SegmentStats>(triangle_count);
    std::uint32_t* all_branch_node_ids = arena.take<std::uint32_t>(triangle_count);
    void* cub_temporary = arena.take_bytes(cub_bytes);

    {
        StageRange range{"meshprep/validation"};
        status = validate_mesh(mesh, {}, validation_flags, stream);
        if (!status) return status;
    }
    {
        StageRange range{"meshprep/centroid_generation"};
        hierarchy_geometry_kernel<<<
            (triangle_count + block_size - 1) / block_size, block_size, 0, stream>>>(
            mesh.positions, mesh.triangles, centroids, triangle_bounds, triangle_count);
        iota_kernel<<<(triangle_count + block_size - 1) / block_size, block_size, 0, stream>>>(
            primitives_a, triangle_count);
        fill_kernel<<<(triangle_count + block_size - 1) / block_size, block_size, 0, stream>>>(
            segment_ids, triangle_count, 0U);
    }
    std::uint32_t zero = 0;
    cudaError_t error = cudaMemcpyAsync(offsets_a, &zero, sizeof(zero), cudaMemcpyHostToDevice, stream);
    if (error == cudaSuccess) {
        error = cudaMemcpyAsync(
            offsets_a + 1, &triangle_count, sizeof(triangle_count), cudaMemcpyHostToDevice, stream);
    }
    if (error == cudaSuccess) {
        error = cudaMemcpyAsync(node_ids_a, &zero, sizeof(zero), cudaMemcpyHostToDevice, stream);
    }
    if (error == cudaSuccess) {
        error = cudaMemcpyAsync(all_branch_node_ids, &zero, sizeof(zero), cudaMemcpyHostToDevice, stream);
    }
    if (error != cudaSuccess) return cuda_status(error, "failed to initialize hierarchy");

    if (triangle_count <= options.max_leaf_size) {
        error = cudaMemcpyAsync(
            output.primitive_indices_,
            primitives_a,
            sizeof(std::uint32_t) * triangle_count,
            cudaMemcpyDeviceToDevice,
            stream);
        if (error != cudaSuccess) return cuda_status(error, "failed to emit root primitives");
        root_leaf_kernel<<<1, 1, 0, stream>>>(triangle_bounds, output.nodes_, triangle_count);
        status = synchronize(stream, "root-leaf hierarchy build failed");
        if (!status) return status;
        output.statistics_ = {1, 1, 0, 0};
        return success();
    }

    std::uint32_t active_count = triangle_count;
    std::uint32_t segment_count = 1;
    std::uint32_t node_count = 1;
    std::uint32_t leaf_count = 0;
    std::uint32_t branch_count_total = 1;
    std::uint32_t leaf_primitive_cursor = 0;
    std::uint32_t depth = 0;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> branch_levels;
    branch_levels.emplace_back(0U, 1U);

    std::uint32_t* current_offsets = offsets_a;
    std::uint32_t* next_offsets = offsets_b;
    std::uint32_t* current_node_ids = node_ids_a;
    std::uint32_t* next_node_ids = node_ids_b;

    while (segment_count > 0) {
        std::size_t bytes = cub_bytes;
        {
            StageRange range{"meshprep/reductions"};
            gather_segment_stats_kernel<<<
                (active_count + block_size - 1) / block_size, block_size, 0, stream>>>(
                centroids, primitives_a, ordered_stats, active_count);
            error = cub::DeviceSegmentedReduce::Reduce(
                cub_temporary,
                bytes,
                ordered_stats,
                segment_stats,
                segment_count,
                current_offsets,
                current_offsets + 1,
                MergeSegmentStats{},
                segment_identity,
                stream);
            if (error != cudaSuccess) {
                return cuda_status(error, "segment statistics reduction failed");
            }
        }
        std::uint32_t child_count = 0;
        {
            StageRange range{"meshprep/partitioning"};
            make_partition_keys_kernel<<<
                (active_count + block_size - 1) / block_size, block_size, 0, stream>>>(
                centroids,
                primitives_a,
                segment_ids,
                current_offsets,
                segment_stats,
                keys_a,
                active_count,
                depth >= 32U);
            bytes = cub_bytes;
            error = cub::DeviceRadixSort::SortPairs(
                cub_temporary,
                bytes,
                keys_a,
                keys_b,
                primitives_a,
                primitives_b,
                active_count,
                0,
                64,
                stream);
            if (error != cudaSuccess) {
                return cuda_status(error, "hierarchy partition sort failed");
            }
            bytes = cub_bytes;
            error = cub::DeviceRunLengthEncode::Encode(
                cub_temporary,
                bytes,
                keys_b,
                unique_keys,
                child_counts,
                device_run_count,
                active_count,
                stream);
            if (error != cudaSuccess) {
                return cuda_status(error, "hierarchy child encoding failed");
            }
            error = cudaMemcpyAsync(
                &child_count,
                device_run_count,
                sizeof(child_count),
                cudaMemcpyDeviceToHost,
                stream);
            if (error != cudaSuccess) {
                return cuda_status(error, "failed to read hierarchy child count");
            }
            error = cudaStreamSynchronize(stream);
            if (error != cudaSuccess) return cuda_status(error, "hierarchy partition failed");
        }
        if (child_count == 0 || node_count + child_count > maximum_nodes) {
            return {StatusCode::internal_error, cudaSuccess, "hierarchy node capacity invariant failed"};
        }

        std::uint32_t host_totals[3]{};
        {
            StageRange range{"meshprep/scans"};
            classify_children_kernel<<<
                (child_count + block_size - 1) / block_size, block_size, 0, stream>>>(
                child_counts,
                branch_flags,
                branch_weights,
                leaf_weights,
                child_count,
                options.max_leaf_size);
            error = cudaMemsetAsync(parent_counts, 0, sizeof(std::uint32_t) * segment_count, stream);
            if (error != cudaSuccess) return cuda_status(error, "failed to clear parent counts");
            count_parent_children_kernel<<<
                (child_count + block_size - 1) / block_size, block_size, 0, stream>>>(
                unique_keys, child_count, parent_counts);

            auto scan = [&](const std::uint32_t* input,
                            std::uint32_t* destination,
                            std::uint32_t count) {
                std::size_t scan_bytes = cub_bytes;
                return cub::DeviceScan::ExclusiveSum(
                    cub_temporary, scan_bytes, input, destination, count, stream);
            };
            if ((error = scan(branch_flags, branch_prefix, child_count)) != cudaSuccess ||
                (error = scan(branch_weights, branch_weight_prefix, child_count)) != cudaSuccess ||
                (error = scan(leaf_weights, leaf_weight_prefix, child_count)) != cudaSuccess ||
                (error = scan(parent_counts, parent_prefix, segment_count)) != cudaSuccess ||
                (error = scan(child_counts, child_offsets, child_count)) != cudaSuccess) {
                return cuda_status(error, "hierarchy prefix scan failed");
            }
            collect_totals_kernel<<<1, 1, 0, stream>>>(
                branch_flags,
                branch_prefix,
                branch_weights,
                branch_weight_prefix,
                leaf_weights,
                leaf_weight_prefix,
                child_count,
                device_totals);
            error = cudaMemcpyAsync(
                host_totals, device_totals, sizeof(host_totals), cudaMemcpyDeviceToHost, stream);
            if (error != cudaSuccess) return cuda_status(error, "failed to read hierarchy totals");
            error = cudaStreamSynchronize(stream);
            if (error != cudaSuccess) return cuda_status(error, "hierarchy classification failed");
        }
        const std::uint32_t next_branch_count = host_totals[0];
        const std::uint32_t next_active_count = host_totals[1];
        const std::uint32_t level_leaf_primitives = host_totals[2];

        {
            StageRange range{"meshprep/packing"};
            set_parent_nodes_kernel<<<
                (segment_count + block_size - 1) / block_size, block_size, 0, stream>>>(
                current_node_ids,
                parent_counts,
                parent_prefix,
                output.nodes_,
                segment_count,
                node_count);
            initialize_child_nodes_kernel<<<
                (child_count + block_size - 1) / block_size, block_size, 0, stream>>>(
                child_counts,
                branch_flags,
                branch_prefix,
                branch_weight_prefix,
                leaf_weight_prefix,
                next_offsets,
                next_node_ids,
                all_branch_node_ids,
                output.nodes_,
                child_count,
                node_count,
                branch_count_total,
                leaf_primitive_cursor);
            scatter_child_primitives_kernel<<<child_count, 128, 0, stream>>>(
                child_offsets,
                child_counts,
                branch_flags,
                branch_prefix,
                branch_weight_prefix,
                leaf_weight_prefix,
                primitives_b,
                primitives_a,
                segment_ids,
                output.primitive_indices_,
                child_count,
                leaf_primitive_cursor);
            leaf_bounds_kernel<<<
                (child_count + block_size - 1) / block_size, block_size, 0, stream>>>(
                child_offsets,
                child_counts,
                branch_flags,
                primitives_b,
                triangle_bounds,
                output.nodes_,
                child_count,
                node_count);

            error = cudaMemcpyAsync(
                next_offsets + next_branch_count,
                &next_active_count,
                sizeof(next_active_count),
                cudaMemcpyHostToDevice,
                stream);
            if (error != cudaSuccess) {
                return cuda_status(error, "failed to terminate next segment offsets");
            }
            status = synchronize(stream, "hierarchy node emission failed");
            if (!status) return status;
        }

        node_count += child_count;
        leaf_count += child_count - next_branch_count;
        leaf_primitive_cursor += level_leaf_primitives;
        if (next_branch_count > 0) {
            branch_levels.emplace_back(branch_count_total, next_branch_count);
        }
        branch_count_total += next_branch_count;
        ++depth;
        active_count = next_active_count;
        segment_count = next_branch_count;
        std::swap(current_offsets, next_offsets);
        std::swap(current_node_ids, next_node_ids);
    }

    if (leaf_primitive_cursor != triangle_count) {
        return {StatusCode::internal_error, cudaSuccess, "hierarchy lost primitives"};
    }
    status = ensure_allocation(
        output.branch_node_ids_, output.branch_node_capacity_, branch_count_total);
    if (!status) return status;
    error = cudaMemcpyAsync(
        output.branch_node_ids_, all_branch_node_ids,
        static_cast<std::size_t>(branch_count_total) * sizeof(std::uint32_t),
        cudaMemcpyDeviceToDevice, stream);
    if (error != cudaSuccess) {
        return cuda_status(error, "failed to retain hierarchy branch levels");
    }
    {
        StageRange range{"meshprep/bounds_propagation"};
        for (auto level = branch_levels.rbegin(); level != branch_levels.rend(); ++level) {
            branch_bounds_kernel<<<
                (level->second + block_size - 1) / block_size, block_size, 0, stream>>>(
                all_branch_node_ids + level->first, output.nodes_, level->second);
            status = synchronize(stream, "bottom-up hierarchy bounds failed");
            if (!status) return status;
        }
    }
    output.statistics_ = {node_count, leaf_count, branch_count_total, depth};
    output.branch_levels_ = std::move(branch_levels);
    return success();
}

Status refit_hierarchy(
    DeviceAabbView primitives,
    Hierarchy& hierarchy,
    cudaStream_t stream)
{
#if MESHPREP_ENABLE_NVTX
    nvtx3::scoped_range function_range{"parallel_mater::refit_hierarchy"};
#endif
    if (primitives.bounds == nullptr || primitives.primitive_count == 0U) {
        return invalid("refit AABB view must contain bounds");
    }
    if (primitives.primitive_count != hierarchy.primitive_count_ ||
        hierarchy.nodes_ == nullptr || hierarchy.primitive_indices_ == nullptr ||
        hierarchy.statistics_.node_count == 0U) {
        return invalid("refit primitive count must match a built hierarchy");
    }
    if (hierarchy.proxy_validation_ == nullptr) {
        const cudaError_t allocation = cudaMalloc(
            &hierarchy.proxy_validation_, sizeof(std::uint32_t));
        if (allocation != cudaSuccess) {
            return cuda_status(allocation, "refit validation allocation failed");
        }
    }
    cudaError_t error = cudaMemsetAsync(
        hierarchy.proxy_validation_, 0, sizeof(std::uint32_t), stream);
    if (error != cudaSuccess) return cuda_status(error, "failed to clear refit validation");
    const auto primitive_count = static_cast<std::uint32_t>(primitives.primitive_count);
    validate_aabb_kernel<<<
        (primitive_count + block_size - 1U) / block_size, block_size, 0, stream>>>(
        primitives.bounds, primitive_count, hierarchy.proxy_validation_);
    std::uint32_t invalid_bounds = 0U;
    error = cudaMemcpyAsync(
        &invalid_bounds, hierarchy.proxy_validation_, sizeof(invalid_bounds),
        cudaMemcpyDeviceToHost, stream);
    if (error != cudaSuccess) return cuda_status(error, "failed to read refit validation");
    Status status = synchronize(stream, "hierarchy refit validation failed");
    if (!status) return status;
    if (invalid_bounds != 0U) {
        return invalid_mesh("AABB bounds must be finite and ordered");
    }
    refit_leaf_bounds_kernel<<<
        (hierarchy.statistics_.node_count + block_size - 1U) / block_size,
        block_size, 0, stream>>>(
        primitives.bounds, hierarchy.primitive_indices_, hierarchy.nodes_,
        hierarchy.statistics_.node_count);
    for (auto level = hierarchy.branch_levels_.rbegin();
         level != hierarchy.branch_levels_.rend(); ++level) {
        branch_bounds_kernel<<<
            (level->second + block_size - 1U) / block_size, block_size, 0, stream>>>(
            hierarchy.branch_node_ids_ + level->first,
            hierarchy.nodes_, level->second);
    }
    status = synchronize(stream, "hierarchy refit failed");
    if (!status) return status;
    return success();
}

Status refit_hierarchy_unchecked_async(
    DeviceAabbView primitives,
    Hierarchy& hierarchy,
    cudaStream_t stream)
{
#if MESHPREP_ENABLE_NVTX
    nvtx3::scoped_range function_range{"parallel_mater::refit_hierarchy_async"};
#endif
    if (primitives.bounds == nullptr || primitives.primitive_count == 0U) {
        return invalid("refit AABB view must contain bounds");
    }
    if (primitives.primitive_count != hierarchy.primitive_count_ ||
        hierarchy.nodes_ == nullptr || hierarchy.primitive_indices_ == nullptr ||
        hierarchy.statistics_.node_count == 0U) {
        return invalid("refit primitive count must match a built hierarchy");
    }
    refit_leaf_bounds_kernel<<<
        (hierarchy.statistics_.node_count + block_size - 1U) / block_size,
        block_size, 0, stream>>>(
        primitives.bounds, hierarchy.primitive_indices_, hierarchy.nodes_,
        hierarchy.statistics_.node_count);
    for (auto level = hierarchy.branch_levels_.rbegin();
         level != hierarchy.branch_levels_.rend(); ++level) {
        branch_bounds_kernel<<<
            (level->second + block_size - 1U) / block_size, block_size, 0, stream>>>(
            hierarchy.branch_node_ids_ + level->first,
            hierarchy.nodes_, level->second);
    }
    const cudaError_t launch_error = cudaPeekAtLastError();
    if (launch_error != cudaSuccess) {
        return cuda_status(launch_error, "failed to enqueue hierarchy refit");
    }
    return success();
}

Status build_hierarchy(
    DeviceAabbView primitives,
    HierarchyOptions options,
    Workspace& workspace,
    Hierarchy& output,
    cudaStream_t stream)
{
#if MESHPREP_ENABLE_NVTX
    nvtx3::scoped_range function_range{"parallel_mater::build_aabb_hierarchy"};
#endif
    if (primitives.bounds == nullptr || primitives.primitive_count == 0) {
        return invalid("AABB view must contain bounds");
    }
    if (primitives.primitive_count >= (std::uint64_t{1} << 32) ||
        primitives.primitive_count > std::numeric_limits<std::uint32_t>::max() / 3U) {
        return unsupported("AABB proxy count exceeds 32-bit indexing");
    }
    const auto primitive_count = static_cast<std::uint32_t>(primitives.primitive_count);
    Status status = ensure_allocation(
        output.proxy_positions_, output.proxy_position_capacity_,
        static_cast<std::size_t>(primitive_count) * 3U);
    if (!status) return status;
    status = ensure_allocation(
        output.proxy_triangles_, output.proxy_triangle_capacity_, primitive_count);
    if (!status) return status;
    if (output.proxy_validation_ == nullptr) {
        const cudaError_t allocation = cudaMalloc(&output.proxy_validation_, sizeof(std::uint32_t));
        if (allocation != cudaSuccess) {
            return cuda_status(allocation, "AABB validation allocation failed");
        }
    }
    cudaError_t error = cudaMemsetAsync(
        output.proxy_validation_, 0, sizeof(std::uint32_t), stream);
    if (error != cudaSuccess) return cuda_status(error, "failed to clear AABB validation");
    aabb_proxy_kernel<<<
        (primitive_count + block_size - 1U) / block_size, block_size, 0, stream>>>(
        primitives.bounds,
        output.proxy_positions_,
        output.proxy_triangles_,
        primitive_count,
        output.proxy_validation_);
    std::uint32_t invalid_bounds = 0;
    error = cudaMemcpyAsync(
        &invalid_bounds,
        output.proxy_validation_,
        sizeof(invalid_bounds),
        cudaMemcpyDeviceToHost,
        stream);
    if (error != cudaSuccess) return cuda_status(error, "failed to read AABB validation");
    status = synchronize(stream, "AABB proxy generation failed");
    if (!status) return status;
    if (invalid_bounds != 0U) {
        return invalid_mesh("AABB bounds must be finite and ordered");
    }
    return build_hierarchy(
        DeviceMeshView{
            output.proxy_positions_,
            static_cast<std::uint64_t>(primitive_count) * 3U,
            output.proxy_triangles_,
            primitive_count},
        options,
        workspace,
        output,
        stream);
}

} // namespace parallel_mater
