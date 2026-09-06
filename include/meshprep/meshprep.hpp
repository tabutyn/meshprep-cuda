// SPDX-License-Identifier: MIT
#pragma once

#include <cuda_runtime_api.h>
#include <vector_types.h>

#include <cstddef>
#include <cstdint>

namespace meshprep {

enum class StatusCode : std::uint8_t {
    success = 0,
    invalid_argument,
    invalid_mesh,
    unsupported_size,
    allocation_failure,
    cuda_failure,
    internal_error,
};

struct Status {
    StatusCode code{StatusCode::success};
    cudaError_t cuda_error{cudaSuccess};
    const char* message{"success"};

    [[nodiscard]] constexpr bool ok() const noexcept { return code == StatusCode::success; }
    [[nodiscard]] constexpr explicit operator bool() const noexcept { return ok(); }
};

struct DeviceMeshView {
    const float3* positions{};
    std::uint64_t vertex_count{};
    const uint3* triangles{};
    std::uint64_t triangle_count{};
};

struct SharpEdgeView {
    const uint2* edges{};
    std::uint64_t edge_count{};
};

struct NormalStatistics {
    std::uint32_t vertex_normal_count{};
    std::uint32_t degenerate_triangle_count{};
};

struct HierarchyOptions {
    std::uint32_t max_leaf_size{8};
};

struct HierarchyNode {
    float3 bounds_min{};
    float3 bounds_max{};
    std::uint32_t first_child{};
    std::uint32_t child_count{};
    std::uint32_t first_primitive{};
    std::uint32_t primitive_count{};

    [[nodiscard]] constexpr bool is_leaf() const noexcept { return child_count == 0; }
};

struct HierarchyStatistics {
    std::uint32_t node_count{};
    std::uint32_t leaf_count{};
    std::uint32_t branch_count{};
    std::uint32_t max_depth{};
};

class Workspace {
public:
    Workspace() noexcept = default;
    ~Workspace();
    Workspace(Workspace&& other) noexcept;
    Workspace& operator=(Workspace&& other) noexcept;
    Workspace(const Workspace&) = delete;
    Workspace& operator=(const Workspace&) = delete;

    [[nodiscard]] Status reserve(std::size_t bytes);
    [[nodiscard]] std::size_t capacity_bytes() const noexcept { return capacity_bytes_; }

private:
    void* storage_{};
    std::size_t capacity_bytes_{};

    friend Status compute_normals(
        DeviceMeshView, SharpEdgeView, Workspace&, class NormalOutput&, cudaStream_t);
    friend Status build_hierarchy(
        DeviceMeshView, HierarchyOptions, Workspace&, class Hierarchy&, cudaStream_t);
};

class NormalOutput {
public:
    NormalOutput() noexcept = default;
    ~NormalOutput();
    NormalOutput(NormalOutput&& other) noexcept;
    NormalOutput& operator=(NormalOutput&& other) noexcept;
    NormalOutput(const NormalOutput&) = delete;
    NormalOutput& operator=(const NormalOutput&) = delete;

    [[nodiscard]] const float3* face_normals() const noexcept { return face_normals_; }
    [[nodiscard]] const float3* vertex_normals() const noexcept { return vertex_normals_; }
    [[nodiscard]] const std::uint32_t* corner_normal_indices() const noexcept {
        return corner_normal_indices_;
    }
    [[nodiscard]] NormalStatistics statistics() const noexcept { return statistics_; }
    [[nodiscard]] std::size_t allocated_bytes() const noexcept
    {
        return face_capacity_ * sizeof(float3) + vertex_capacity_ * sizeof(float3) +
            corner_capacity_ * sizeof(std::uint32_t);
    }

private:
    float3* face_normals_{};
    float3* vertex_normals_{};
    std::uint32_t* corner_normal_indices_{};
    std::size_t face_capacity_{};
    std::size_t vertex_capacity_{};
    std::size_t corner_capacity_{};
    NormalStatistics statistics_{};

    friend Status compute_normals(
        DeviceMeshView, SharpEdgeView, Workspace&, NormalOutput&, cudaStream_t);
};

class Hierarchy {
public:
    Hierarchy() noexcept = default;
    ~Hierarchy();
    Hierarchy(Hierarchy&& other) noexcept;
    Hierarchy& operator=(Hierarchy&& other) noexcept;
    Hierarchy(const Hierarchy&) = delete;
    Hierarchy& operator=(const Hierarchy&) = delete;

    [[nodiscard]] const HierarchyNode* nodes() const noexcept { return nodes_; }
    [[nodiscard]] const std::uint32_t* primitive_indices() const noexcept {
        return primitive_indices_;
    }
    [[nodiscard]] HierarchyStatistics statistics() const noexcept { return statistics_; }
    [[nodiscard]] std::size_t allocated_bytes() const noexcept
    {
        return node_capacity_ * sizeof(HierarchyNode) +
            primitive_capacity_ * sizeof(std::uint32_t);
    }

private:
    HierarchyNode* nodes_{};
    std::uint32_t* primitive_indices_{};
    std::size_t node_capacity_{};
    std::size_t primitive_capacity_{};
    HierarchyStatistics statistics_{};

    friend Status build_hierarchy(
        DeviceMeshView, HierarchyOptions, Workspace&, Hierarchy&, cudaStream_t);
};

[[nodiscard]] Status compute_normals(
    DeviceMeshView mesh,
    SharpEdgeView sharp_edges,
    Workspace& workspace,
    NormalOutput& output,
    cudaStream_t stream = nullptr);

[[nodiscard]] Status build_hierarchy(
    DeviceMeshView mesh,
    HierarchyOptions options,
    Workspace& workspace,
    Hierarchy& output,
    cudaStream_t stream = nullptr);

[[nodiscard]] const char* status_code_name(StatusCode code) noexcept;

} // namespace meshprep
