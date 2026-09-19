// SPDX-License-Identifier: MIT
#pragma once

#include <meshprep/meshprep.hpp>

#include <cuda_runtime_api.h>
#include <vector_types.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>

namespace meshprep::physics {

inline constexpr std::uint32_t node_surface = 1U << 0U;
inline constexpr std::uint32_t node_pinned = 1U << 1U;

struct Bond {
    uint2 nodes{};
    float rest_length{};
};

// General fixed-topology lattice configuration. Scene layout, gravity,
// colliders, and rendering remain application policy rather than presets.
struct SoftBodyOptions {
    // Keep this limit aligned with the CUDA solver.  Dense gallery fixtures
    // (for example an adjustable N x M field) are valid API clients too.
    static constexpr std::uint32_t maximum_instances{256U};

    std::uint32_t instance_count{1U};
    std::uint32_t substeps{4U};
    std::uint32_t constraint_iterations{8U};
    float timestep{1.0F / 60.0F};
    float node_mass{0.02F};
    float spring_stiffness{40'000.0F};
    float spring_damping_ratio{0.85F};
    float velocity_damping{0.8F};
    float maximum_projection_fraction{0.20F};
    float constraint_velocity_response{0.70F};
    float break_strain{0.12F};
    std::uint32_t fracture_persistence_substeps{8U};
    float maximum_speed{12.0F};
    float strength_multiplier{1.0F};
    float ground_friction{1.0F};
    std::uint32_t hierarchy_leaf_size{8U};
    bool unbonded_node_collisions{false};
    bool preserve_fractured_triangle_shape{false};
    bool render_internal_members{false};
    std::array<float3, maximum_instances> instance_origins{};
};

struct SoftBodyMaterial {
    float spring_stiffness{};
    float spring_damping_ratio{};
    float velocity_damping{};
    float maximum_speed{};
    float ground_friction{};
};

// Borrowed device arrays for coupling custom collision or force kernels.
// prepare_substep() predicts positions and clears the two writable buffers.
// A caller may then add impulses (N*s) and positional corrections before
// finish_substep() consumes them. Reacquire this view after reset or move.
struct SoftBodyNodeView {
    const float3* positions{};
    const float3* velocities{};
    const float3* rest_positions{};
    const std::uint32_t* flags{};
    float3* external_impulses{};
    float3* position_corrections{};
    std::uint32_t node_count{};
    std::uint32_t nodes_per_instance{};
    std::uint32_t instance_count{};
    float node_radius{};
    float inverse_node_mass{};
};

struct SoftBodyBondView {
    const float3* positions{};
    const std::uint32_t* flags{};
    const Bond* bonds{};
    const std::uint8_t* bond_active{};
    std::uint32_t node_count{};
    std::uint32_t nodes_per_instance{};
    std::uint32_t bonds_per_instance{};
    std::uint32_t instance_count{};
    float node_radius{};
};

struct SoftBodySurfaceView {
    DeviceMeshView mesh{};
    const float3* vertex_normals{};
    const std::uint32_t* corner_normal_indices{};
    const float2* texcoords{};
    const std::uint8_t* triangle_active{};
    const HierarchyNode* hierarchy_nodes{};
    const std::uint32_t* primitive_indices{};
    std::uint32_t hierarchy_node_count{};
    std::uint32_t hierarchy_max_depth{};
};

struct SoftBodyTimings {
    float physics_ms{};
    float surface_deformation_ms{};
    float hierarchy_ms{};

    [[nodiscard]] constexpr float gpu_total_ms() const noexcept
    {
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

// An owning CUDA soft body loaded from the versioned .msb asset format.
// Calls are synchronous before return. The manual frame protocol permits
// application-owned CUDA contact kernels without coupling this library to a
// particular rigid-body engine, scene, renderer, or input system.
class SoftBody {
public:
    SoftBody() noexcept;
    ~SoftBody();
    SoftBody(SoftBody&&) noexcept;
    SoftBody& operator=(SoftBody&&) noexcept;
    SoftBody(const SoftBody&) = delete;
    SoftBody& operator=(const SoftBody&) = delete;

    [[nodiscard]] static Status create(
        std::string_view asset_path,
        SoftBodyOptions options,
        SoftBody& output,
        cudaStream_t stream = nullptr) noexcept;

    [[nodiscard]] Status initialize(
        std::string_view asset_path,
        SoftBodyOptions options = {},
        cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status step(
        float3 gravity, cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status step(
        float3 gravity,
        SoftBodyTimings& timings,
        cudaStream_t stream = nullptr) noexcept;

    [[nodiscard]] Status begin_frame(cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status prepare_substep(
        float dt, float3 gravity, cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status finish_substep(
        float dt, float3 gravity, cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status finish_frame(
        SoftBodyTimings& timings, cudaStream_t stream = nullptr) noexcept;

    [[nodiscard]] Status reset(cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status set_material(SoftBodyMaterial material) noexcept;
    [[nodiscard]] Status set_substeps(std::uint32_t substeps) noexcept;
    [[nodiscard]] Status set_constraint_iterations(
        std::uint32_t iterations) noexcept;
    [[nodiscard]] Status set_strength_multiplier(float multiplier) noexcept;
    [[nodiscard]] Status set_node_mass(float mass) noexcept;

    [[nodiscard]] bool initialized() const noexcept;
    [[nodiscard]] SoftBodyOptions options() const noexcept;
    [[nodiscard]] SoftBodyMaterial material() const noexcept;
    [[nodiscard]] float node_mass() const noexcept;
    [[nodiscard]] SoftBodyNodeView nodes() const noexcept;
    [[nodiscard]] SoftBodyBondView bonds() const noexcept;
    [[nodiscard]] SoftBodySurfaceView surface() const noexcept;
    [[nodiscard]] SoftBodyStatistics statistics() const noexcept;
    [[nodiscard]] std::size_t allocated_bytes() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace meshprep::physics
