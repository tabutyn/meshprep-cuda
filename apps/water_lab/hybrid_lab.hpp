// SPDX-License-Identifier: MIT
#pragma once

#include "water_lab.hpp"
#include "particle_cells.hpp"
#include "obstacle_course.hpp"

#include <meshprep/meshprep.hpp>

#include <cuda_runtime_api.h>
#include <vector_types.h>

#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace waterlab {

inline constexpr float course_motion_scale = 4.0F;
inline constexpr float course_gravity_magnitude = 1.8F * course_motion_scale;
inline constexpr float course_material_scale = 4.0F;
// Four substeps are the measured stable floor for the course preset. Raising
// the motion multiplier raises this floor to preserve bounded displacement.
inline constexpr std::uint32_t course_minimum_iterations = 4U;

struct HybridOptions {
    std::uint32_t physics_iterations{1U};
    std::uint32_t particle_count{10'000U};
    std::uint32_t physical_skin_frequency{10U};
    std::uint32_t render_skin_frequency{45U};
    float fixed_dt{1.0F / 60.0F};
    float skin_radius{0.75F};
    float particle_radius{0.0225F};
    float particle_spacing{0.045F};
    float particle_support_radius{0.12F};
    float particle_repulsion{1.5F};
    float particle_damping{2.0F};
    float particle_velocity_damping{0.4F};
    float particle_skin_distance{0.05F};
    float particle_skin_interaction_radius{0.30F};
    float particle_skin_stiffness{220.0F};
    float particle_skin_damping{12.0F};
    float skin_spring_stiffness{140.0F};
    float skin_spring_damping{3.0F};
    float skin_velocity_damping{1.2F};
    float box_skin_stiffness{3'000.0F};
    float box_skin_damping{64.0F};
    float box_skin_contact_thickness{0.002F};
    bool collect_contact_diagnostics{false};
    float particle_mass{1.0F};
    float skin_vertex_mass{2.0F};
    float rectangle_mass{200.0F};
    float rectangle_inertia{80.0F};
    float rectangle_target_stiffness{900.0F};
    float rectangle_target_damping_ratio{1.0F};
    float maximum_rectangle_control_force{1200.0F};
    float maximum_particle_force{55.0F};
    float maximum_skin_force{240.0F};
    float maximum_box_ejection_force{200.0F};
    float maximum_particle_speed{3.0F};
    float maximum_skin_speed{2.0F};
    float maximum_rectangle_speed{1.25F};
    float maximum_rectangle_angular_speed{1.5F};
    // App-only tilt course. Defaults retain the original lab and its fixtures.
    bool obstacle_course{false};
    float3 gravity{};
    // Particle-only examples can use the same deterministic particle broad
    // phase without an invisible containing membrane. The skin remains
    // allocated by HybridDroplet until ParticleSystem is extracted.
    bool particle_skin_coupling{true};
    // Appended to preserve the prefix of recorded HybridOptions v1-v5.
    std::uint32_t particle_capacity{10'000U};
    float3 particle_initial_center{};
    GalleryArena arena{GalleryArena::none};
};

// Keep the gravity-loaded game's material preset separate from the lab and
// recordings. Its boundary must withstand the pressure of particles settling.
[[nodiscard]] inline HybridOptions course_options()
{
    HybridOptions options;
    options.obstacle_course = true;
    options.physics_iterations = course_minimum_iterations;
    options.gravity = make_float3(0.0F, -course_gravity_magnitude, 0.0F);
    options.maximum_particle_speed *= course_motion_scale;
    options.maximum_skin_speed *= course_motion_scale;
    // Match the existing material forces to the increased gravity load. Raising
    // acceleration alone compresses the fluid and overwhelms its soft boundary.
    options.particle_repulsion *= course_material_scale;
    options.skin_spring_stiffness *= course_material_scale;
    options.particle_skin_stiffness = 2'000.0F * course_material_scale;
    options.maximum_particle_force = 240.0F * course_material_scale;
    options.maximum_skin_force *= course_material_scale;
    return options;
}

// Derived from recorded speed caps: no extra simulation/capture state needed.
[[nodiscard]] inline float course_motion_multiplier(const HybridOptions& options)
{
    return options.maximum_particle_speed / HybridOptions{}.maximum_particle_speed;
}

[[nodiscard]] inline std::uint32_t course_motion_iterations(const HybridOptions& options)
{
    return std::max(course_minimum_iterations,
        static_cast<std::uint32_t>(std::ceil(course_motion_multiplier(options))));
}

[[nodiscard]] inline HybridOptions with_course_motion_multiplier(
    HybridOptions options, float requested)
{
    if (!std::isfinite(requested)) throw std::invalid_argument("non-finite course motion multiplier");
    const auto old_floor = course_motion_iterations(options);
    const float old_multiplier = course_motion_multiplier(options);
    const float multiplier = std::clamp(requested, 1.0F, 8.0F);
    const float gravity = 1.8F * multiplier;
    const float old_gravity = std::sqrt(options.gravity.x*options.gravity.x +
        options.gravity.y*options.gravity.y + options.gravity.z*options.gravity.z);
    options.gravity = old_gravity > 0.0F ? make_float3(
        options.gravity.x*gravity/old_gravity, options.gravity.y*gravity/old_gravity,
        options.gravity.z*gravity/old_gravity) : make_float3(0, -gravity, 0);
    options.maximum_particle_speed = HybridOptions{}.maximum_particle_speed * multiplier;
    options.maximum_skin_speed = HybridOptions{}.maximum_skin_speed * multiplier;
    // Gravity and the course material form one dimensionless gameplay preset.
    // Scaling acceleration alone changes the equilibrium density and makes the
    // same droplet visibly compress as the speed control is raised. Preserve
    // that balance while retaining any live parameter edits as relative tuning.
    const float material_ratio = multiplier / old_multiplier;
    options.particle_repulsion *= material_ratio;
    options.particle_skin_stiffness *= material_ratio;
    options.skin_spring_stiffness *= material_ratio;
    options.maximum_particle_force *= material_ratio;
    options.maximum_skin_force *= material_ratio;
    const auto new_floor = course_motion_iterations(options);
    // Track the automatic floor in both directions while retaining explicit
    // extra iterations selected above the previous floor.
    options.physics_iterations = options.physics_iterations <= old_floor ? new_floor :
        std::max(options.physics_iterations, new_floor);
    return options;
}

struct RectangleState {
    float3 center{1.45F, 0.0F, 0.0F};
    float3 velocity{};
    float3 half_extents{0.30F, 0.65F, 0.42F};
    float yaw{};
    float angular_velocity{};
};

struct DeviceRectangleState {
    float3 center{};
    float3 velocity{};
    float3 half_extents{};
    float yaw{};
    float angular_velocity{};
};

struct HybridTimings {
    float rebuild_fluid_hierarchy_ms{};
    float rebuild_skin_hierarchy_ms{};
    float update_fluid_physics_ms{};
    float update_skin_physics_ms{};
    float update_rectangle_physics_ms{};
    float update_surface_normals_ms{};
    float update_render_surface_ms{};
    float update_soft_body_physics_ms{};
    float rebuild_soft_body_hierarchy_ms{};
    float update_soft_body_contact_ms{};
    float update_soft_body_render_ms{};
    float update_rigid_body_contact_ms{};
    float particle_recycling_ms{};

    [[nodiscard]] float gpu_total_ms() const noexcept {
        return rebuild_fluid_hierarchy_ms + rebuild_skin_hierarchy_ms +
            update_fluid_physics_ms + update_skin_physics_ms +
            update_rectangle_physics_ms + update_surface_normals_ms +
            update_render_surface_ms + update_soft_body_physics_ms +
            rebuild_soft_body_hierarchy_ms + update_soft_body_contact_ms +
            update_soft_body_render_ms + update_rigid_body_contact_ms +
            particle_recycling_ms;
    }
};

struct HybridStatistics {
    std::uint32_t particle_count{};
    std::uint32_t physical_skin_vertices{};
    std::uint32_t physical_skin_triangles{};
    std::uint32_t render_skin_vertices{};
    std::uint32_t render_skin_triangles{};
    std::uint32_t particles_outside{};
    std::uint32_t skin_vertices_inside_rectangle{};
    std::uint32_t finite_failures{};
    std::uint32_t maximum_particle_neighbors{};
    float average_particle_neighbors{};
    float maximum_particle_force{};
    float maximum_skin_force{};
    float maximum_rectangle_reaction{};
    std::uint32_t particle_force_cap_hits{};
    std::uint32_t skin_force_cap_hits{};
    std::uint32_t box_force_cap_hits{};
    // Keep frame_index before newly appended statistics so v1-v3 captures
    // retain a readable prefix.
    std::uint64_t frame_index{};
    std::uint32_t soft_body_contact_count{};
    float maximum_soft_body_penetration{};
    std::uint32_t recycled_particles{};
};

// Complete persistent simulation state plus the force diagnostics needed to
// inspect a recorded frame. Hierarchies, normals, and render vertices are
// derived and rebuilt when a state is restored.
struct HybridState {
    HybridOptions options{};
    RectangleState rectangle{};
    HybridStatistics statistics{};
    std::vector<float3> particle_positions;
    std::vector<float3> particle_velocities;
    std::vector<float3> particle_forces;
    std::vector<std::uint32_t> particle_skin_owners;
    std::vector<float3> skin_positions;
    std::vector<float3> skin_velocities;
    std::vector<float3> skin_forces;
    std::vector<float3> skin_box_forces;
    std::vector<float3> skin_box_impulses;
    std::vector<float> skin_box_pair_work;
    std::vector<float3> skin_particle_forces;
    std::vector<float3> skin_spring_forces;
};

// One fixed 1/60-second force integration per call. The rectangle is a finite-
// mass dynamic body; callers provide force/torque, never a new transform.
class HybridDroplet {
public:
    static constexpr std::uint32_t maximum_physics_iterations{16U};

    explicit HybridDroplet(HybridOptions options = {});
    ~HybridDroplet();
    HybridDroplet(const HybridDroplet&) = delete;
    HybridDroplet& operator=(const HybridDroplet&) = delete;

    [[nodiscard]] HybridTimings step(
        float3 rectangle_control_force = {},
        float rectangle_control_torque = 0.0F,
        cudaStream_t stream = nullptr,
        SoftBodyCourse* soft_bodies = nullptr,
        bool enable_rectangle_collider = true,
        RigidSphereState* rigid_sphere = nullptr,
        WaterWheelState* water_wheel = nullptr,
        const float3* rigid_sphere_gravity_override = nullptr);
    // Updates coefficients used by subsequent ticks. Allocation- and topology-
    // defining fields must remain unchanged.
    void set_runtime_options(const HybridOptions& options);
    // Active IDs occupy [0, particle_count). Shrinking removes the tail;
    // growing restores new IDs at deterministic authored spawn positions.
    // Both operations rebuild the particle broad phase before returning.
    void resize_particles(std::uint32_t active_count, cudaStream_t stream = nullptr);
    void reset(cudaStream_t stream = nullptr);
    void capture_state(HybridState& output, cudaStream_t stream = nullptr) const;
    void restore_state(const HybridState& state, cudaStream_t stream = nullptr);

    [[nodiscard]] meshprep::DeviceMeshView skin_mesh() const noexcept;
    [[nodiscard]] const meshprep::NormalOutput& skin_normals() const noexcept {
        return render_normals_;
    }
    [[nodiscard]] const meshprep::Hierarchy& skin_hierarchy() const noexcept {
        return render_hierarchy_;
    }
    [[nodiscard]] const meshprep::Hierarchy& particle_hierarchy() const noexcept {
        return particle_hierarchy_;
    }
    [[nodiscard]] const meshprep::Hierarchy& physical_skin_hierarchy() const noexcept {
        return skin_vertex_hierarchy_;
    }
    [[nodiscard]] const float3* particle_positions() const noexcept {
        return particle_positions_;
    }
    [[nodiscard]] const float3* particle_velocities() const noexcept {
        return particle_velocities_;
    }
    [[nodiscard]] ParticleCellView particle_cells() const noexcept {
        return {particle_positions_, particle_cell_keys_b_, particle_cell_indices_b_,
            options_.particle_count, particle_cell_size_};
    }
    [[nodiscard]] const float3* physical_skin_positions() const noexcept {
        return skin_positions_;
    }
    [[nodiscard]] const float3* physical_skin_normals() const noexcept {
        return physics_normals_.vertex_normals();
    }
    [[nodiscard]] const float3* skin_box_forces() const noexcept {
        return skin_box_forces_;
    }
    [[nodiscard]] const float3* skin_box_impulses() const noexcept {
        return skin_box_impulses_;
    }
    [[nodiscard]] const float* skin_box_pair_work() const noexcept {
        return skin_box_pair_work_;
    }
    [[nodiscard]] const std::uint32_t* particle_skin_owners() const noexcept {
        return particle_skin_owners_;
    }
    [[nodiscard]] const float3* skin_particle_forces() const noexcept {
        return skin_particle_forces_;
    }
    [[nodiscard]] const float3* skin_spring_forces() const noexcept {
        return skin_spring_forces_;
    }
    [[nodiscard]] std::uint32_t physical_skin_vertex_count() const noexcept {
        return skin_vertex_count_;
    }
    [[nodiscard]] float particle_radius() const noexcept { return options_.particle_radius; }
    [[nodiscard]] RectangleState rectangle() const noexcept { return host_rectangle_; }
    [[nodiscard]] OrientedBox render_box() const noexcept;
    [[nodiscard]] HybridStatistics statistics() const noexcept { return statistics_; }
    [[nodiscard]] HybridOptions options() const noexcept { return options_; }
    [[nodiscard]] std::size_t allocated_bytes() const noexcept;

private:
    static constexpr std::uint32_t stage_count_{9U};
    static constexpr std::uint32_t event_count_{
        stage_count_ * maximum_physics_iterations};
    HybridOptions options_;
    RectangleState initial_rectangle_{};
    RectangleState host_rectangle_{};
    HybridStatistics statistics_{};

    float3* particle_positions_{};
    float3* particle_initial_positions_{};
    float3* particle_velocities_{};
    float3* particle_forces_{};
    std::uint32_t* particle_skin_owners_{};
    meshprep::Aabb* particle_bounds_{};
    std::uint64_t* particle_cell_keys_a_{};
    std::uint64_t* particle_cell_keys_b_{};
    std::uint32_t* particle_cell_indices_a_{};
    std::uint32_t* particle_cell_indices_b_{};

    float3* skin_positions_{};
    float3* skin_initial_positions_{};
    float3* skin_velocities_{};
    float3* skin_forces_{};
    float3* skin_box_forces_{};
    float3* skin_box_impulses_{};
    float* skin_box_pair_work_{};
    float3* skin_particle_forces_{};
    float3* skin_spring_forces_{};
    float3* skin_rest_positions_{};
    float3* course_center_sum_{};
    float4* course_rotation_{};
    uint3* skin_triangles_{};
    std::uint32_t* skin_incident_offsets_{};
    std::uint32_t* skin_incident_triangles_{};
    std::uint32_t* skin_neighbor_offsets_{};
    std::uint32_t* skin_neighbors_{};
    float* skin_rest_lengths_{};
    meshprep::Aabb* skin_vertex_bounds_{};

    float3* render_positions_{};
    float3* render_rest_positions_{};
    uint3* render_triangles_{};
    SurfaceVertexEmbedding* render_embedding_{};

    std::uint32_t* reaction_keys_a_{};
    std::uint32_t* reaction_keys_b_{};
    std::uint32_t* reaction_unique_vertices_{};
    float3* reaction_values_a_{};
    float3* reaction_values_b_{};
    float3* reaction_reduced_values_{};
    std::uint32_t* reaction_run_count_{};
    float3* rectangle_force_rows_{};
    float* rectangle_torque_rows_{};
    float3* rectangle_force_sum_{};
    float* rectangle_torque_sum_{};
    DeviceRectangleState* device_rectangle_{};
    std::uint32_t* device_statistics_{};
    void* cub_storage_{};
    std::size_t cub_storage_bytes_{};

    std::uint32_t skin_vertex_count_{};
    std::uint32_t skin_triangle_count_{};
    std::uint32_t render_vertex_count_{};
    std::uint32_t render_triangle_count_{};
    std::uint32_t skin_neighbor_count_{};
    std::uint32_t skin_incident_count_{};
    std::uint32_t reaction_record_capacity_{};
    float particle_cell_size_{};

    meshprep::Workspace particle_workspace_;
    meshprep::Workspace skin_workspace_;
    meshprep::Workspace physics_normal_workspace_;
    meshprep::Workspace render_workspace_;
    meshprep::Workspace render_normal_workspace_;
    meshprep::Hierarchy particle_hierarchy_;
    meshprep::Hierarchy skin_vertex_hierarchy_;
    meshprep::Hierarchy render_hierarchy_;
    meshprep::NormalOutput physics_normals_;
    meshprep::NormalOutput render_normals_;
    cudaEvent_t stage_begin_[event_count_]{};
    cudaEvent_t stage_end_[event_count_]{};

    void rebuild_particle_cells(cudaStream_t stream);
    void rebuild_particle_broadphase(cudaStream_t stream);
    void rebuild_derived_state(cudaStream_t stream);
};

} // namespace waterlab
