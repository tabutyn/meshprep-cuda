// SPDX-License-Identifier: MIT
#pragma once

#include <meshprep/simulation.hpp>
#include "obstacle_course.hpp"

#include <cuda_runtime_api.h>
#include <vector_types.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace waterlab {

inline constexpr std::uint32_t soft_body_voxel_surface = 1U << 0U;
inline constexpr std::uint32_t soft_body_voxel_pinned = 1U << 1U;
inline constexpr std::uint32_t soft_body_voxel_rim_anchor = 1U << 2U;
inline constexpr std::uint32_t soft_body_asset_y_up = 1U << 0U;
inline constexpr std::uint32_t soft_body_asset_delta_skinning = 1U << 1U;
inline constexpr std::uint32_t soft_body_asset_free_body = 1U << 2U;

// An edge is stored once. The CSR adjacency below refers back to this stable
// edge ID so a connection can be broken without changing the graph topology.
using SoftBodyEdge = meshprep::sim::LatticeBond;

struct SoftBodyNeighbor {
    std::uint32_t vertex{};
    std::uint32_t edge{};
};

// The asset records four weighted surface candidates for each UV-mapped
// vertex. Runtime picks the strongest candidate as its material anchor and
// uses two bonded candidates as a local corotational frame. This avoids
// blending across separated fracture components while preserving the authored
// rest-pose offset.
struct SoftBodyBinding {
    uint4 voxels{};
    float4 weights{};
};

// Runtime coordinates are right-handed, Y-up. A converted asset is local to
// one post and centered on the Y axis. Course layout grounds its render-space
// minimum Y on the shared board; custom origins are applied without adjustment.
struct SoftBodyAsset {
    float nominal_spacing{0.05F};
    float voxel_radius{0.025F};
    std::uint32_t relaxation_iterations{};
    std::uint32_t file_flags{
        soft_body_asset_y_up | soft_body_asset_delta_skinning};
    std::vector<float3> rest_voxels;
    std::vector<std::uint32_t> voxel_flags;
    std::vector<SoftBodyEdge> edges;
    std::vector<std::uint32_t> neighbor_offsets;
    std::vector<SoftBodyNeighbor> neighbors;
    std::vector<float3> render_positions;
    std::vector<float2> render_uvs;
    std::vector<SoftBodyBinding> render_bindings;
    std::vector<uint3> render_triangles;
};

// The compact .msb format is deliberately independent of Blender and CUDA
// struct padding. It is versioned and validated before any GPU allocation.
[[nodiscard]] SoftBodyAsset load_soft_body_asset(const std::string& path);
void save_soft_body_asset(const SoftBodyAsset& asset, const std::string& path);
void validate_soft_body_asset(const SoftBodyAsset& asset);

struct SoftBodyOptions {
    static constexpr std::uint32_t maximum_instances{32U};

    std::uint32_t instance_count{8U};
    std::uint32_t solver_substeps{4U};
    std::uint32_t spring_solver_iterations{8U};
    float fixed_dt{1.0F / 60.0F};
    float voxel_mass{0.02F};
    float spring_stiffness{40'000.0F};
    float spring_damping_ratio{0.85F};
    float velocity_damping{0.8F};
    // Bounds one Jacobi graph correction to a fraction of nominal voxel
    // spacing and controls how much projection displacement becomes momentum.
    float maximum_projection_fraction{0.20F};
    float constraint_velocity_response{0.70F};
    float break_strain{0.12F};
    std::uint32_t fracture_persistence_substeps{8U};
    float maximum_speed{12.0F};
    float strength_multiplier{1.0F};
    float ground_friction{1.0F};
    std::uint32_t hierarchy_leaf_size{8U};
    bool use_course_layout{true};
    bool require_1000_voxels{true};
    // The graph solver is also used by cloth/soft-body examples that do not
    // declare a rigid arena. Keep the historical course behavior by default,
    // while allowing those compositions to avoid an invisible floor/rails.
    bool course_board_collisions{true};
    // Gallery bodies share the exact analytic arena used by fluid particles.
    GalleryArena arena{GalleryArena::none};
    // Optional fixed-topology cross-component contact: source physical nodes
    // [0, cross_source_nodes) collide against render triangles beginning at
    // cross_target_triangle_first. Target triangle bindings name physical nodes.
    std::uint32_t cross_source_nodes{};
    std::uint32_t cross_target_triangle_first{};
    // Optional presentation boundaries for merged procedural assets. They do
    // not affect physics; the renderer uses them to distinguish a source body,
    // a goal cloth, and any remaining support cloth.
    std::uint32_t surface_triangle_split{};
    std::uint32_t secondary_surface_triangle_split{};
    // Spatial-cell barrier between non-bonded voxels, including detached
    // fragments and separate post instances. Kept optional for cloth recipes.
    bool unbonded_voxel_collisions{};
    // Impact cloth can accumulate fracture damage from the unprojected contact
    // strain. Load-bearing volumes use converged residual strain instead.
    bool fracture_before_projection{};
    // Give every render triangle private vertices.  Once any of its lattice
    // edges fractures, preserve the authored triangle size while attaching
    // the loose piece to a surviving edge or vertex.
    bool preserve_fractured_triangle_shape{};
    // Render active structural bonds as opaque rectangular members through a
    // separate hierarchy.  This is presentation geometry only; the spring
    // graph remains the single source of topology and fracture state.
    bool render_internal_members{};
    std::array<float3, maximum_instances> instance_origins{};
};

struct SoftBodyMaterial {
    float spring_stiffness{};
    float spring_damping_ratio{};
    float velocity_damping{};
    float maximum_speed{};
    float ground_friction{};
};

struct SoftBodyVoxelView {
    const float3* positions{};
    const float3* velocities{};
    const float3* rest_positions{};
    const std::uint32_t* flags{};
    // Cleared before each prepared substep, then written by the coupled water
    // contact pass. Values are impulses in N*s, not forces.
    float3* external_impulses{};
    float3* position_corrections{};
    std::uint32_t voxel_count{};
    std::uint32_t voxels_per_instance{};
    std::uint32_t instance_count{};
    float voxel_radius{};
    float inverse_voxel_mass{};
};

// One combined mesh contains all post instances. UVs and triangle indices are
// immutable; positions and hierarchy bounds are refreshed after every step.
struct SoftBodyRenderView {
    const float3* positions{};
    const float3* vertex_normals{};
    const std::uint32_t* corner_normal_indices{};
    const float2* texcoords{};
    const uint3* triangles{};
    const SoftBodyBinding* bindings{};
    const std::uint8_t* triangle_active{};
    std::uint32_t vertex_count{};
    std::uint32_t triangle_count{};
    const meshprep::HierarchyNode* nodes{};
    const std::uint32_t* primitive_indices{};
    std::uint32_t node_count{};
    std::uint32_t max_depth{};
    const float3* member_positions{};
    const SoftBodyEdge* member_edges{};
    const std::uint8_t* member_active{};
    const meshprep::HierarchyNode* member_nodes{};
    const std::uint32_t* member_indices{};
    std::uint32_t member_count{};
    std::uint32_t member_voxels_per_instance{};
    std::uint32_t members_per_instance{};
    std::uint32_t member_node_count{};
    std::uint32_t member_max_depth{};
    float member_half_width{};
    std::uint32_t surface_triangle_split{};
    std::uint32_t secondary_surface_triangle_split{};
};

// Borrowed CUDA arrays for the complete volume and actual spring graph.
// Edges contain local voxel IDs shared by every instance. Activity is indexed
// by instance * edges_per_instance + edge; positions/flags by
// instance * voxels_per_instance + local voxel. Reacquire after each step.
struct SoftBodyLatticeView {
    const float3* positions{};
    const std::uint32_t* flags{};
    const SoftBodyEdge* edges{};
    const std::uint8_t* active_edges{};
    std::uint32_t voxel_count{};
    std::uint32_t voxels_per_instance{};
    std::uint32_t edges_per_instance{};
    std::uint32_t instance_count{};
    float voxel_radius{};
};

struct SoftBodyTimings {
    float physics_ms{};
    float render_deformation_ms{};
    float render_hierarchy_ms{};
    float rigid_contact_ms{};

    [[nodiscard]] float gpu_total_ms() const noexcept {
        return physics_ms + render_deformation_ms + render_hierarchy_ms +
            rigid_contact_ms;
    }
};

struct RigidSphereState {
    float3 center{};
    float3 velocity{};
    float3 angular_velocity{};
    // Unit quaternion (x,y,z,w). Rendering and persistent paint use this same
    // material frame so ground friction produces visible rolling.
    float4 orientation{0.0F, 0.0F, 0.0F, 1.0F};
    float radius{0.40F};
    float mass{4.0F};
};

struct SoftBodyStatistics {
    std::uint32_t instance_count{};
    std::uint32_t voxels_per_instance{};
    std::uint32_t total_voxel_count{};
    std::uint32_t surface_voxel_count{};
    std::uint32_t edges_per_instance{};
    std::uint32_t total_edge_count{};
    std::uint32_t broken_edge_count{};
    std::uint32_t finite_failure_count{};
    std::uint64_t frame_index{};
};

struct SoftBodyState {
    float strength_multiplier{1.0F};
    SoftBodyStatistics statistics{};
    std::vector<float3> positions;
    std::vector<float3> velocities;
    std::vector<std::uint8_t> active_edges;
    std::vector<std::uint8_t> edge_damage;
    std::vector<std::uint8_t> active_render_triangles;
};

// Applies finite ground friction and advances the sphere's material frame.
// Kept shared so stand-alone and fluid-coupled gallery scenes roll identically.
void advance_rigid_sphere_rotation(
    RigidSphereState& sphere, GalleryArena arena, float friction, float dt);

// A deterministic, fixed-topology spring lattice. Each voxel gathers its own
// sorted CSR neighbors; simulation forces never use floating-point atomics.
// The only mutable topology state is one byte per breakable edge instance.
class SoftBodyCourse {
public:
    explicit SoftBodyCourse(SoftBodyAsset asset, SoftBodyOptions options = {});
    explicit SoftBodyCourse(const std::string& asset_path, SoftBodyOptions options = {});
    ~SoftBodyCourse();
    SoftBodyCourse(SoftBodyCourse&&) noexcept;
    SoftBodyCourse& operator=(SoftBodyCourse&&) noexcept;
    SoftBodyCourse(const SoftBodyCourse&) = delete;
    SoftBodyCourse& operator=(const SoftBodyCourse&) = delete;

    // Frame protocol used by HybridDroplet: begin_frame(), then one
    // prepare/contact/finish sequence for each of its own substeps, followed by
    // finish_frame(). Contact writes impulses (N*s) and optional positional
    // corrections. The monolithic step() is a no-contact convenience wrapper.
    void begin_frame(cudaStream_t stream = nullptr);
    void prepare_substep(float dt, float3 gravity, cudaStream_t stream = nullptr);
    // Applies one finite-mass sphere contact while a prepared substep is open.
    // Reactions are gathered in node order before updating the host sphere, so
    // the result is deterministic on one supported GPU.
    void contact_rigid_sphere_substep(
        RigidSphereState& sphere, float dt, cudaStream_t stream = nullptr);
    void finish_substep(float dt, float3 gravity, cudaStream_t stream = nullptr);
    [[nodiscard]] SoftBodyTimings finish_frame(cudaStream_t stream = nullptr);
    void clear_external_impulses(cudaStream_t stream = nullptr);
    void clear_external_forces(cudaStream_t stream = nullptr);
    [[nodiscard]] SoftBodyTimings step(float3 gravity, cudaStream_t stream = nullptr);
    [[nodiscard]] SoftBodyTimings step_with_rigid_sphere(
        RigidSphereState& sphere, float3 gravity, cudaStream_t stream = nullptr);
    // Bilateral endpoint attachment plus a tension-only maximum-length
    // constraint against the rope's pinned first node. The latter transfers
    // the post reaction to the finite-mass sphere once the chain is taut.
    [[nodiscard]] SoftBodyTimings step_with_tethered_rigid_sphere(
        RigidSphereState& sphere, std::uint32_t endpoint_node,
        float attachment_distance, float3 gravity,
        cudaStream_t stream = nullptr);
    void reset(cudaStream_t stream = nullptr);
    void set_strength_multiplier(float multiplier);
    void set_solver_substeps(std::uint32_t substeps);
    void set_spring_solver_iterations(std::uint32_t iterations);
    void set_material(SoftBodyMaterial material);
    [[nodiscard]] SoftBodyMaterial material() const noexcept;
    [[nodiscard]] std::uint32_t spring_solver_iterations() const noexcept;
    void set_uniform_velocity(
        std::uint32_t first_node, std::uint32_t node_count, float3 velocity,
        cudaStream_t stream = nullptr);
    // Prescribes the water-wheel pose for central axle anchors and outer-rim
    // tip anchors from immutable authored positions; free nodes remain dynamic.
    void set_pinned_rotation_z(
        float3 center, float angle, cudaStream_t stream = nullptr);
    void set_wheel_anchor_rotations(
        float3 center, float axle_angle, float rim_angle,
        cudaStream_t stream = nullptr);
    [[nodiscard]] float wheel_rim_reaction_torque(
        float3 center, cudaStream_t stream = nullptr);
    void capture_state(SoftBodyState& output, cudaStream_t stream = nullptr) const;
    void restore_state(const SoftBodyState& state, cudaStream_t stream = nullptr);

    [[nodiscard]] float strength_multiplier() const noexcept;
    [[nodiscard]] SoftBodyVoxelView voxel_view() const noexcept;
    [[nodiscard]] SoftBodyRenderView render_view() const noexcept;
    [[nodiscard]] SoftBodyLatticeView lattice_view() const noexcept;
    [[nodiscard]] meshprep::DeviceMeshView render_mesh() const noexcept;
    [[nodiscard]] const meshprep::Hierarchy& render_hierarchy() const noexcept;
    [[nodiscard]] SoftBodyStatistics statistics() const noexcept;
    [[nodiscard]] std::size_t allocated_bytes() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace waterlab
