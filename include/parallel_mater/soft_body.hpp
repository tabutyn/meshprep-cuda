// SPDX-License-Identifier: MIT
#pragma once

#include <parallel_mater/frame.hpp>
#include <parallel_mater/geometry.hpp>

#include <vector_types.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>

namespace parallel_mater::physics {

inline constexpr std::uint32_t node_surface = 1U << 0U;
inline constexpr std::uint32_t node_pinned = 1U << 1U;

struct Bond {
    uint2 vertices{};
    float rest_length{};
};
struct SoftBodyAssetView {
    const std::byte *data{};
    std::size_t size{};
};

struct SoftBodyOptions {
    static constexpr std::uint32_t maximum_instances{256U};
    std::uint32_t instance_count{1U};
    std::uint32_t substeps{4U};
    std::uint32_t constraint_iterations{8U};
    float timestep{1.0F / 60.0F};
    float node_mass{0.02F};
    float spring_stiffness{40'000.0F};
    float velocity_damping{0.8F};
    float maximum_projection_fraction{0.20F};
    float constraint_velocity_response{0.70F};
    float break_strain{0.12F};
    std::uint32_t fracture_persistence_substeps{8U};
    float maximum_speed{12.0F};
    float strength_multiplier{1.0F};
    std::uint32_t hierarchy_leaf_size{8U};
    float4 initial_color{0.42F, 0.22F, 0.72F, 1.0F};
    std::array<float3, maximum_instances> instance_origins{};
};

struct SoftBodyMaterial {
    float spring_stiffness{};
    float velocity_damping{};
    float maximum_speed{};
};

struct SoftBodyNodeView {
    const float3 *positions{};
    const float3 *velocities{};
    const float3 *rest_positions{};
    const std::uint32_t *flags{};
    float3 *external_impulses{};
    float3 *position_corrections{};
    std::uint32_t node_count{};
    std::uint32_t nodes_per_instance{};
    std::uint32_t instance_count{};
    float node_radius{};
    float inverse_node_mass{};
    float4 *colors{};
};

struct SoftBodyBondView {
    const float3 *positions{};
    const std::uint32_t *flags{};
    const Bond *bonds{};
    const std::uint8_t *bond_active{};
    std::uint32_t node_count{};
    std::uint32_t nodes_per_instance{};
    std::uint32_t bonds_per_instance{};
    std::uint32_t instance_count{};
    float node_radius{};
};

struct SoftBodySurfaceView {
    DeviceMeshView mesh{};
    const float3 *vertex_normals{};
    const std::uint32_t *corner_normal_indices{};
    const float2 *texcoords{};
    const std::uint8_t *triangle_active{};
    const HierarchyNode *hierarchy_nodes{};
    const std::uint32_t *primitive_indices{};
    std::uint32_t hierarchy_node_count{};
    std::uint32_t hierarchy_max_depth{};
    const float4 *vertex_colors{};
};

struct SoftBodyTimings {
    float physics_ms{};
    float surface_deformation_ms{};
    float hierarchy_ms{};
    [[nodiscard]] constexpr float gpu_total_ms() const noexcept {
        return physics_ms + surface_deformation_ms + hierarchy_ms;
    }
};

struct SoftBodyStatistics {
    std::uint32_t instance_count{};
    std::uint32_t nodes_per_instance{};
    std::uint32_t node_count{};
    std::uint32_t surface_node_count{};
    std::uint32_t bonds_per_instance{};
    std::uint32_t bond_count{};
    std::uint32_t broken_bond_count{};
    std::uint32_t finite_failure_count{};
    std::uint64_t frame_index{};
};

struct SoftBodyTelemetry {
    SoftBodyTimings timings{};
    SoftBodyStatistics statistics{};
};

class SoftBody {
  public:
    SoftBody() noexcept;
    ~SoftBody();
    SoftBody(SoftBody &&) noexcept;
    SoftBody &operator=(SoftBody &&) noexcept;
    SoftBody(const SoftBody &) = delete;
    SoftBody &operator=(const SoftBody &) = delete;

    [[nodiscard]] static Status create(std::string_view asset_path, SoftBodyOptions options,
                                       SoftBody &output, cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] static Status create(SoftBodyAssetView asset, SoftBodyOptions options,
                                       SoftBody &output, cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status initialize(std::string_view asset_path, SoftBodyOptions options = {},
                                    cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status initialize(SoftBodyAssetView asset, SoftBodyOptions options = {},
                                    cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status begin_frame(FrameOptions frame, cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status prepare_substep(SubstepContext substep,
                                         cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status finish_substep(SubstepContext substep,
                                        cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status finish_frame(Completion &completion,
                                      cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status abandon_frame(cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status advance_async(FrameOptions frame, Completion &completion,
                                       cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status advance(FrameOptions frame, cudaStream_t stream = nullptr) noexcept;

    [[nodiscard]] Status collect_telemetry_async(Completion &completion,
                                                 cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status resolve_telemetry(SoftBodyTelemetry &output) noexcept;
    [[nodiscard]] Status collect_telemetry(cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] SoftBodyTelemetry telemetry() const noexcept;

    [[nodiscard]] Status step(float3 acceleration, cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status step(float3 acceleration, SoftBodyTimings &timings,
                              cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status begin_frame(cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status prepare_substep(float dt, float3 acceleration,
                                         cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status finish_substep(float dt, float3 acceleration,
                                        cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status finish_frame(SoftBodyTimings &timings,
                                      cudaStream_t stream = nullptr) noexcept;

    [[nodiscard]] Status reset(cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status set_material(SoftBodyMaterial material) noexcept;
    [[nodiscard]] Status set_substeps(std::uint32_t substeps) noexcept;
    [[nodiscard]] Status set_constraint_iterations(std::uint32_t iterations) noexcept;
    [[nodiscard]] Status set_strength_multiplier(float multiplier) noexcept;
    [[nodiscard]] Status set_node_mass(float mass) noexcept;
    [[nodiscard]] bool initialized() const noexcept;
    [[nodiscard]] SoftBodyOptions options() const noexcept;
    [[nodiscard]] SoftBodyMaterial material() const noexcept;
    [[nodiscard]] float node_mass() const noexcept;
    [[nodiscard]] SoftBodyNodeView nodes() const noexcept;
    [[nodiscard]] PointStateView point_state() const noexcept;
    [[nodiscard]] PointCouplingView coupling_points() const noexcept;
    [[nodiscard]] SoftBodyBondView bonds() const noexcept;
    [[nodiscard]] SoftBodySurfaceView surface() const noexcept;
    [[nodiscard]] SoftBodyStatistics statistics() const noexcept;
    [[nodiscard]] std::size_t allocated_bytes() const noexcept;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    friend class Cloth;
    friend class Rope;
};

} // namespace parallel_mater::physics
