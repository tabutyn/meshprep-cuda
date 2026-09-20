// SPDX-License-Identifier: MIT
#pragma once

#include <parallel_mater/frame.hpp>
#include <parallel_mater/geometry.hpp>

#include <cuda_runtime_api.h>
#include <vector_types.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>

namespace parallel_mater::physics {

inline constexpr std::uint32_t node_surface = 1U << 0U;
inline constexpr std::uint32_t node_pinned = 1U << 1U;

struct Bond {
    uint2 vertices{};
    float rest_length{};
};

// Borrowed bytes containing one complete versioned .msb asset. initialize()
// parses and copies the payload, so the caller may release the bytes when the
// call returns.
struct SoftBodyAssetView {
    const std::byte* data{};
    std::size_t size{};
};

// General fixed-topology lattice configuration. Scene layout, gravity,
// colliders, and rendering remain application policy rather than presets.
struct SoftBodyOptions {
    // Keep this limit aligned with the CUDA solver; dense multi-instance
    // fields are valid clients of the same topology owner.
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
    [[nodiscard]] static Status create(
        SoftBodyAssetView asset,
        SoftBodyOptions options,
        SoftBody& output,
        cudaStream_t stream = nullptr) noexcept;

    [[nodiscard]] Status initialize(
        std::string_view asset_path,
        SoftBodyOptions options = {},
        cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status initialize(
        SoftBodyAssetView asset,
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

    [[nodiscard]] Status begin_frame(
        FrameOptions frame, cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status prepare_substep(
        SubstepContext substep, cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status finish_substep(
        SubstepContext substep, cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status finish_frame(
        Completion& completion, cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status advance_async(
        FrameOptions frame, Completion& completion,
        cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status advance(
        FrameOptions frame, cudaStream_t stream = nullptr) noexcept;

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

struct FluidParticle {
    float3 position{};
    float3 velocity{};
};

struct FluidOptions {
    float particle_radius{0.0225F};
    float interaction_radius{0.09F};
    float particle_mass{1.0F};
    float repulsion{20.0F};
    float viscosity{0.02F};
    float velocity_damping{0.2F};
    float maximum_speed{8.0F};
};

// Borrowed device arrays. Couplers may add impulses between prepare_substep()
// and finish_substep(); positions and velocities remain solver-owned.
struct FluidView {
    const float3* positions{};
    const float3* velocities{};
    float3* external_impulses{};
    std::uint32_t particle_count{};
    float particle_radius{};
    float inverse_particle_mass{};
};

struct FluidStatistics {
    std::uint32_t particle_count{};
    std::uint32_t maximum_neighbors{};
    std::uint32_t finite_failure_count{};
    std::uint64_t frame_index{};
    std::size_t allocated_bytes{};
};

// Independent, owning CUDA particle fluid. It uses a deterministic sorted
// uniform grid and fixed-radius pressure/viscosity interactions; boundaries
// and cross-solver contacts are supplied through the coupling buffer.
class Fluid {
public:
    Fluid() noexcept;
    ~Fluid();
    Fluid(Fluid&&) noexcept;
    Fluid& operator=(Fluid&&) noexcept;
    Fluid(const Fluid&) = delete;
    Fluid& operator=(const Fluid&) = delete;

    [[nodiscard]] static Status create(
        std::span<const FluidParticle> particles, FluidOptions options,
        Fluid& output, cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status initialize(
        std::span<const FluidParticle> particles, FluidOptions options = {},
        cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status begin_frame(
        FrameOptions frame, cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status prepare_substep(
        SubstepContext substep, cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status finish_substep(
        SubstepContext substep, cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status finish_frame(
        Completion& completion, cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status advance_async(
        FrameOptions frame, Completion& completion,
        cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status advance(
        FrameOptions frame, cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status reset(cudaStream_t stream = nullptr) noexcept;

    [[nodiscard]] bool initialized() const noexcept;
    [[nodiscard]] FluidOptions options() const noexcept;
    [[nodiscard]] FluidView particles() const noexcept;
    [[nodiscard]] PointCouplingView coupling_points() const noexcept;
    [[nodiscard]] FluidStatistics statistics() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

struct ClothOptions {
    std::uint32_t columns{24U};
    std::uint32_t rows{18U};
    float spacing{0.05F};
    float3 top_center{};
    bool shear_springs{true};
    bool bend_springs{true};
    SoftBodyOptions solver{};
};

class Cloth {
public:
    Cloth() noexcept;
    ~Cloth();
    Cloth(Cloth&&) noexcept;
    Cloth& operator=(Cloth&&) noexcept;
    Cloth(const Cloth&) = delete;
    Cloth& operator=(const Cloth&) = delete;

    [[nodiscard]] static Status create(
        ClothOptions options, Cloth& output,
        cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status initialize(
        ClothOptions options = {}, cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status begin_frame(
        FrameOptions frame, cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status prepare_substep(
        SubstepContext substep, cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status finish_substep(
        SubstepContext substep, cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status finish_frame(
        Completion& completion, cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status advance_async(
        FrameOptions frame, Completion& completion,
        cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status advance(
        FrameOptions frame, cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status reset(cudaStream_t stream = nullptr) noexcept;

    [[nodiscard]] bool initialized() const noexcept;
    [[nodiscard]] ClothOptions options() const noexcept;
    [[nodiscard]] SoftBodyNodeView nodes() const noexcept;
    [[nodiscard]] PointCouplingView coupling_points() const noexcept;
    [[nodiscard]] SoftBodyBondView bonds() const noexcept;
    [[nodiscard]] SoftBodySurfaceView surface() const noexcept;
    [[nodiscard]] SoftBodyStatistics statistics() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

struct RopeOptions {
    std::uint32_t node_count{32U};
    float spacing{0.055F};
    float3 origin{};
    float3 direction{1.0F, 0.0F, 0.0F};
    SoftBodyOptions solver{};
};

class Rope {
public:
    Rope() noexcept;
    ~Rope();
    Rope(Rope&&) noexcept;
    Rope& operator=(Rope&&) noexcept;
    Rope(const Rope&) = delete;
    Rope& operator=(const Rope&) = delete;

    [[nodiscard]] static Status create(
        RopeOptions options, Rope& output,
        cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status initialize(
        RopeOptions options = {}, cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status begin_frame(
        FrameOptions frame, cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status prepare_substep(
        SubstepContext substep, cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status finish_substep(
        SubstepContext substep, cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status finish_frame(
        Completion& completion, cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status advance_async(
        FrameOptions frame, Completion& completion,
        cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status advance(
        FrameOptions frame, cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status reset(cudaStream_t stream = nullptr) noexcept;

    [[nodiscard]] bool initialized() const noexcept;
    [[nodiscard]] RopeOptions options() const noexcept;
    [[nodiscard]] SoftBodyNodeView nodes() const noexcept;
    [[nodiscard]] PointCouplingView coupling_points() const noexcept;
    [[nodiscard]] SoftBodyBondView bonds() const noexcept;
    [[nodiscard]] SoftBodySurfaceView surface() const noexcept;
    [[nodiscard]] SoftBodyStatistics statistics() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

enum class RigidShape : std::uint8_t {
    sphere,
    box,
};

struct RigidBodyOptions {
    RigidShape shape{RigidShape::sphere};
    float mass{1.0F};
    float radius{0.5F};
    float3 half_extents{0.5F, 0.5F, 0.5F};
    float3 inertia{0.1F, 0.1F, 0.1F};
    float linear_damping{0.1F};
    float angular_damping{0.1F};
    float maximum_linear_speed{100.0F};
    float maximum_angular_speed{100.0F};
};

struct RigidBodyState {
    float3 position{};
    float4 orientation{0.0F, 0.0F, 0.0F, 1.0F};
    float3 linear_velocity{};
    float3 angular_velocity{};
};

// Independent finite-mass rigid dynamics owner. Contact detection remains a
// separate coupling concern; apply_force/apply_torque accumulate reactions for
// the currently prepared substep.
class RigidBody {
public:
    RigidBody() noexcept;
    ~RigidBody();
    RigidBody(RigidBody&&) noexcept;
    RigidBody& operator=(RigidBody&&) noexcept;
    RigidBody(const RigidBody&) = delete;
    RigidBody& operator=(const RigidBody&) = delete;

    [[nodiscard]] static Status create(
        RigidBodyState state, RigidBodyOptions options, RigidBody& output) noexcept;
    [[nodiscard]] Status initialize(
        RigidBodyState state = {}, RigidBodyOptions options = {}) noexcept;
    [[nodiscard]] Status begin_frame(
        FrameOptions frame, cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status prepare_substep(
        SubstepContext substep, cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status apply_force(float3 force, float3 world_point) noexcept;
    [[nodiscard]] Status apply_torque(float3 torque) noexcept;
    [[nodiscard]] Status finish_substep(
        SubstepContext substep, cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status finish_frame(
        Completion& completion, cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status advance_async(
        FrameOptions frame, Completion& completion,
        cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status advance(
        FrameOptions frame, cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status reset() noexcept;

    [[nodiscard]] bool initialized() const noexcept;
    [[nodiscard]] RigidBodyOptions options() const noexcept;
    [[nodiscard]] RigidBodyState state() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace parallel_mater::physics
