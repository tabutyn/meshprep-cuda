// SPDX-License-Identifier: MIT
#pragma once

#include <parallel_mater/soft_body.hpp>

#include <cuda_runtime_api.h>
#include <vector_types.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace parallel_mater::physics::detail {

inline constexpr std::uint32_t asset_node_surface = 1U << 0U;
inline constexpr std::uint32_t asset_node_pinned = 1U << 1U;
inline constexpr std::uint32_t asset_y_up = 1U << 0U;
inline constexpr std::uint32_t asset_delta_skinning = 1U << 1U;
inline constexpr std::uint32_t asset_free_body = 1U << 2U;

struct FixedTopologyNeighbor {
    std::uint32_t node{};
    std::uint32_t bond{};
};

struct SurfaceBinding {
    uint4 nodes{};
    float4 weights{};
};

// Solver-neutral host representation of the versioned .msb payload. Authored
// scene layout, collision fixtures, and presentation policy deliberately live
// outside this private library type.
struct FixedTopologyAsset {
    float nominal_spacing{0.05F};
    float node_radius{0.025F};
    std::uint32_t relaxation_iterations{};
    std::uint32_t file_flags{asset_y_up | asset_delta_skinning};
    std::vector<float3> rest_nodes;
    std::vector<std::uint32_t> node_flags;
    std::vector<Bond> bonds;
    std::vector<std::uint32_t> neighbor_offsets;
    std::vector<FixedTopologyNeighbor> neighbors;
    std::vector<float3> surface_positions;
    std::vector<float2> surface_uvs;
    std::vector<SurfaceBinding> surface_bindings;
    std::vector<uint3> surface_triangles;
};

[[nodiscard]] FixedTopologyAsset load_fixed_topology_asset(const std::string &path);
[[nodiscard]] FixedTopologyAsset load_fixed_topology_asset(std::span<const std::byte> bytes);
void validate_fixed_topology_asset(const FixedTopologyAsset &asset);

class FixedTopology {
  public:
    FixedTopology(FixedTopologyAsset asset, SoftBodyOptions options);
    FixedTopology(const std::string &asset_path, SoftBodyOptions options);
    ~FixedTopology();
    FixedTopology(FixedTopology &&) noexcept;
    FixedTopology &operator=(FixedTopology &&) noexcept;
    FixedTopology(const FixedTopology &) = delete;
    FixedTopology &operator=(const FixedTopology &) = delete;

    void begin_frame(cudaStream_t stream);
    void prepare_substep(float timestep, float3 acceleration, cudaStream_t stream);
    void finish_substep(float timestep, float3 acceleration, cudaStream_t stream);
    void finish_frame_async(cudaStream_t stream);
    void collect_telemetry_async(cudaStream_t stream);
    [[nodiscard]] SoftBodyTimings resolve_telemetry();
    [[nodiscard]] SoftBodyTimings finish_frame(cudaStream_t stream);
    void reset(cudaStream_t stream);

    void set_material(SoftBodyMaterial material);
    void set_substeps(std::uint32_t substeps);
    void set_constraint_iterations(std::uint32_t iterations);
    void set_strength_multiplier(float multiplier);
    void set_node_mass(float mass);

    [[nodiscard]] SoftBodyMaterial material() const noexcept;
    [[nodiscard]] SoftBodyNodeView nodes() const noexcept;
    [[nodiscard]] SoftBodyBondView bonds() const noexcept;
    [[nodiscard]] SoftBodySurfaceView surface() const noexcept;
    [[nodiscard]] SoftBodyStatistics statistics() const noexcept;
    [[nodiscard]] std::size_t allocated_bytes() const noexcept;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace parallel_mater::physics::detail
