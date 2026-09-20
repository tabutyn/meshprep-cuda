// SPDX-License-Identifier: MIT
#include "hybrid_lab.hpp"
#include "campaign.hpp"
#include "cloth.hpp"
#include "fluid_visuals.hpp"
#include "obstacle_course.hpp"
#include "simulation_gallery.hpp"
#include "water_lab.hpp"

#include <parallel_mater/gallery.hpp>
#include <parallel_mater/physics.hpp>
#include <parallel_mater/smoke.hpp>

#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <cstdint>
#include <ctime>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

#ifndef MESHPREP_SOFT_BODY_ASSET_PATH
#define MESHPREP_SOFT_BODY_ASSET_PATH "assets/softbody/checker_cylinder.msb"
#endif

enum class Scene {
    Course,
    Lab,
};

struct Options {
    std::uint32_t width{960U};
    std::uint32_t height{720U};
    std::uint32_t profile_frames{};
    std::uint32_t warmups{10U};
    std::uint32_t physics_iterations{}; // zero selects the scene preset
    Scene scene{Scene::Course};
    parallel_mater::sim::SimulationRecipe context{
        parallel_mater::sim::SimulationRecipe::water};
    waterlab::FluidDisplay fluid_display{waterlab::FluidDisplay::Surface};
    bool view_explicit{};
    bool show_foam{true};
    bool drive_box{};
    std::filesystem::path replay_path;
};

struct Interaction {
    bool orbiting{};
    bool panning{};
    bool box_dragging{};
    bool paused{};
    waterlab::FluidDisplay fluid_display{waterlab::FluidDisplay::Surface};
    bool show_foam{true};
    bool show_physics{};
    bool show_quantities{};
    bool show_timings{true};
    bool show_context_browser{};
    bool show_normals{};
    bool show_box_forces{};
    bool show_particle_forces{};
    bool show_spring_forces{};
    bool reset{};
    bool capture_requested{};
    bool replay_mode{};
    bool course_mode{true};
    bool course_finished{};
    waterlab::GalleryProgression progression{};
    waterlab::ObjectiveProgress objective_progress{};
    float painted_fraction{};
    bool reset_bowl_paint{true};
    float rope_turns{};
    float previous_rope_angle{};
    bool have_previous_rope_angle{};
    float3 previous_rigid_center{};
    bool have_previous_rigid_center{};
    bool goal_cloth_damaged{};
    std::uint32_t inspected_broken_edges{};
    parallel_mater::sim::SimulationRecipe context{parallel_mater::sim::SimulationRecipe::water};
    parallel_mater::sim::SimulationRecipe pending_context{parallel_mater::sim::SimulationRecipe::water};
    bool have_follow_center{};
    bool rotate_left{};
    bool rotate_right{};
    int physics_parameter{};
    int physics_adjustment{};
    int quantity_parameter{};
    int quantity_adjustment{};
    int course_motion_adjustment{};
    int soft_body_strength_adjustment{};
    int replay_delta{};
    int context_browser_index{};
    double previous_x{};
    double previous_y{};
    double drag_x{};
    double drag_y{};
    double pan_x{};
    double pan_y{};
    float orbit_yaw{};
    float orbit_pitch{0.72F};
    float orbit_radius{5.2F};
    float3 camera_target{};
    float3 course_gravity{0.0F, -waterlab::course_gravity_magnitude, 0.0F};
    float3 droplet_center{};
    float course_progress{};
    float3 follow_center{};
    float3 rectangle_target{1.45F, 0.0F, 0.0F};
    std::uint32_t rope_node_count{waterlab::gallery::default_rope_nodes};
    std::uint32_t cloth_detail{waterlab::gallery::default_hanging_cloth_detail};
    std::uint32_t bridge_columns{waterlab::rope_bridge_columns};
    std::uint32_t bridge_rows{waterlab::rope_bridge_rows};
    std::uint32_t cylinder_columns{5U};
    std::uint32_t cylinder_rows{4U};
    std::uint32_t staged_particle_target{};
    std::uint32_t staged_spawn_frame{};
    float fishing_head_x{};
    float fishing_rope_scale{1.0F};
    bool fishing_latched{};
    float smoke_rotor_angle{};
    float smoke_rotor_angular_velocity{};
};

void begin_context_particle_spawn(
    Interaction& input, waterlab::HybridDroplet& droplet)
{
    input.staged_particle_target=droplet.options().particle_count;
    input.staged_spawn_frame=0U;
    if (input.context==parallel_mater::sim::SimulationRecipe::water_soft_body &&
        input.staged_particle_target>256U)
        droplet.resize_particles(256U);
}

void advance_context_particle_spawn(
    Interaction& input, waterlab::HybridDroplet& droplet)
{
    if (input.context!=parallel_mater::sim::SimulationRecipe::water_soft_body ||
        input.staged_particle_target<=droplet.options().particle_count) return;
    input.staged_spawn_frame=std::min(300U,input.staged_spawn_frame+1U);
    const std::uint32_t range=input.staged_particle_target-256U;
    const std::uint32_t desired=std::min(input.staged_particle_target,
        256U+(range*input.staged_spawn_frame+299U)/300U);
    if (desired>droplet.options().particle_count) droplet.resize_particles(desired);
}

constexpr int physics_parameter_count = 31;
bool recipe_has_particles(parallel_mater::sim::SimulationRecipe context);
std::vector<int> relevant_physics_parameters(const Interaction& input)
{
    std::vector<int> result{0};
    const bool particles = recipe_has_particles(input.context);
    const bool water_skin = waterlab::gallery::recipe_has(
        input.context, parallel_mater::sim::Component::water_skin);
    const bool deformable = waterlab::gallery::recipe_has(
        input.context, parallel_mater::sim::Component::soft_body) ||
        (input.context!=parallel_mater::sim::SimulationRecipe::water_cloth &&
         waterlab::gallery::recipe_has(
            input.context, parallel_mater::sim::Component::cloth)) ||
        waterlab::gallery::recipe_has(
            input.context, parallel_mater::sim::Component::rope);
    if (particles) {
        result.insert(result.end(), {1, 2, 3});
        result.insert(result.end(), {25, 26, 27});
    }
    if (water_skin) {
        result.insert(result.end(), {4, 5, 6, 7, 8, 12});
    }
    if (!input.course_mode) {
        result.insert(result.end(), {9, 10, 11, 13, 14});
    }
    if (deformable) result.insert(result.end(), {15, 19, 20, 21, 22, 28});
    if (input.context == parallel_mater::sim::SimulationRecipe::cloth ||
        input.context == parallel_mater::sim::SimulationRecipe::soft_body ||
        input.context == parallel_mater::sim::SimulationRecipe::water ||
        input.context == parallel_mater::sim::SimulationRecipe::water_rope ||
        input.context == parallel_mater::sim::SimulationRecipe::water_soft_body ||
        input.context == parallel_mater::sim::SimulationRecipe::rope ||
        input.context == parallel_mater::sim::SimulationRecipe::cloth_rope ||
        input.context == parallel_mater::sim::SimulationRecipe::soft_body_rope)
        result.push_back(23);
    if (input.context == parallel_mater::sim::SimulationRecipe::cloth ||
        input.context == parallel_mater::sim::SimulationRecipe::soft_body ||
        input.context == parallel_mater::sim::SimulationRecipe::water_soft_body ||
        input.context == parallel_mater::sim::SimulationRecipe::cloth_soft_body ||
        input.context == parallel_mater::sim::SimulationRecipe::rope ||
        input.context == parallel_mater::sim::SimulationRecipe::cloth_rope ||
        input.context == parallel_mater::sim::SimulationRecipe::soft_body_rope)
        result.push_back(24);
    if (input.context == parallel_mater::sim::SimulationRecipe::cloth_rope ||
        input.context == parallel_mater::sim::SimulationRecipe::soft_body_rope)
        result.insert(result.end(), {29,30});
    result.push_back(18);
    return result;
}

std::vector<int> relevant_quantity_parameters(const Interaction& input)
{
    std::vector<int> result;
    if (recipe_has_particles(input.context)) result.push_back(0);
    if (waterlab::gallery::recipe_has(
            input.context, parallel_mater::sim::Component::water_skin)) {
        result.push_back(1);
        result.push_back(2);
    }
    if (waterlab::gallery::recipe_has(
            input.context, parallel_mater::sim::Component::soft_body) ||
        waterlab::gallery::recipe_has(
            input.context, parallel_mater::sim::Component::cloth) ||
        waterlab::gallery::recipe_has(
            input.context, parallel_mater::sim::Component::rope))
        result.push_back(3);
    if (input.context == parallel_mater::sim::SimulationRecipe::rope ||
        input.context == parallel_mater::sim::SimulationRecipe::water_rope)
        result.push_back(4);
    if (input.context == parallel_mater::sim::SimulationRecipe::cloth)
        result.push_back(5);
    if (input.context == parallel_mater::sim::SimulationRecipe::soft_body)
        result.insert(result.end(),{6,7});
    return result;
}
constexpr std::size_t capture_frame_capacity = 360U;
static_assert(sizeof(waterlab::FoamParticle) == 48U,
    "changing the foam record layout requires a new capture format version");

bool needs_fluid_visual_update(const Interaction& input)
{
    return waterlab::gallery::recipe_has(
               input.context, parallel_mater::sim::Component::fluid_particles) &&
        input.fluid_display == waterlab::FluidDisplay::Surface;
}

bool recipe_has_particles(parallel_mater::sim::SimulationRecipe context)
{
    return waterlab::gallery::recipe_has(
               context, parallel_mater::sim::Component::fluid_particles) ||
        waterlab::gallery::recipe_has(
            context, parallel_mater::sim::Component::hand_particles);
}

bool recipe_has_smoke(parallel_mater::sim::SimulationRecipe context)
{
    return waterlab::gallery::recipe_has(
        context,parallel_mater::sim::Component::smoke);
}

parallel_mater::physics::SmokeOptions smoke_options_for(
    parallel_mater::sim::SimulationRecipe context)
{
    using parallel_mater::sim::SimulationRecipe;
    parallel_mater::physics::SmokeOptions options;
    options.particle_count=6'000U;
    options.capacity=12'000U;
    options.emitter_center=make_float3(-2.15F,-0.30F,-1.20F);
    options.emitter_half_extents=make_float3(0.05F,0.48F,0.52F);
    options.initial_velocity=make_float3(2.8F,0.12F,0.0F);
    options.turbulence_strength=1.15F;
    if (context==SimulationRecipe::fluid_smoke) {
        options.particle_count=7'500U;
        options.emitter_center=make_float3(0.0F,-0.78F,-1.20F);
        options.emitter_half_extents=make_float3(1.15F,0.04F,0.78F);
        options.initial_velocity=make_float3(0.0F,0.42F,0.0F);
        options.buoyancy=1.85F;
        options.turbulence_strength=0.72F;
        options.lifetime=3.6F;
    } else if (context==SimulationRecipe::cloth_smoke) {
        // Windmill is in XY with its axle along Z; smoke travels toward -Z.
        options.emitter_center=make_float3(0.0F,0.10F,1.10F);
        options.emitter_half_extents=make_float3(0.95F,0.95F,0.04F);
        options.initial_velocity=make_float3(0.0F,0.05F,-2.9F);
        options.buoyancy=0.18F;
    } else if (context==SimulationRecipe::soft_body_smoke ||
              context==SimulationRecipe::rope_smoke) {
        options.emitter_center=make_float3(-2.25F,-0.35F,-1.20F);
        options.emitter_half_extents=make_float3(0.05F,0.32F,1.05F);
        options.initial_velocity=make_float3(3.0F,0.12F,0.0F);
        options.buoyancy=0.22F;
    }
    return options;
}

std::optional<parallel_mater::physics::Smoke> create_smoke_system(
    parallel_mater::sim::SimulationRecipe context)
{
    if (!recipe_has_smoke(context)) return std::nullopt;
    parallel_mater::physics::Smoke smoke;
    const parallel_mater::Status status=smoke.initialize(smoke_options_for(context));
    if (!status.ok()) throw std::runtime_error(
        std::string("initialize smoke: ")+status.message);
    return smoke;
}

waterlab::FoamSettings context_foam_settings(
    parallel_mater::sim::SimulationRecipe context) noexcept
{
    if (context == parallel_mater::sim::SimulationRecipe::water)
        return {32.0F, 2.0F, 1.6F};
    return {};
}

void configure_context_camera(Interaction& input)
{
    input.orbit_pitch = 0.72F;
    input.orbit_yaw = 0.0F;
    input.camera_target = make_float3(0.0F, -0.10F, -0.75F);
    input.orbit_radius = 5.2F;
    if (input.context == parallel_mater::sim::SimulationRecipe::water) {
        input.camera_target = make_float3(0.0F, -0.35F, -1.0F);
        input.orbit_radius = 7.0F;
    } else if (input.context == parallel_mater::sim::SimulationRecipe::cloth) {
        input.camera_target = make_float3(0.0F, -0.15F, -0.62F);
        input.orbit_radius = 4.2F;
        input.orbit_pitch = 0.22F;
    } else if (input.context == parallel_mater::sim::SimulationRecipe::soft_body) {
        input.camera_target = make_float3(-0.55F, -0.18F, -1.8F);
        input.orbit_radius = 5.0F;
        input.orbit_pitch = 0.32F;
    } else if (input.context == parallel_mater::sim::SimulationRecipe::water_rope) {
        input.camera_target = waterlab::fishing_tank_center;
        input.orbit_radius = 6.2F;
        input.orbit_pitch = 0.08F;
    } else if (input.context == parallel_mater::sim::SimulationRecipe::water_soft_body) {
        input.camera_target = make_float3(
            0.5F * (waterlab::water_wheel_entry_x +
                waterlab::water_wheel_collector_start_x),
            0.5F * (waterlab::water_wheel_center.y +
                waterlab::water_wheel_ground_y),
            waterlab::water_wheel_center.z);
        input.orbit_radius = 8.4F;
        input.orbit_pitch = 0.78539816F;
    } else if (input.context == parallel_mater::sim::SimulationRecipe::cloth_soft_body) {
        input.camera_target = make_float3(0.0F, -0.35F, -1.05F);
        input.orbit_radius = 4.6F;
        input.orbit_pitch = 0.28F;
    } else if (input.context == parallel_mater::sim::SimulationRecipe::rope) {
        input.camera_target = make_float3(0.85F, -0.20F, waterlab::rope_anchor.z);
        input.orbit_radius = 4.8F;
        input.orbit_pitch = 0.30F;
    } else if (input.context == parallel_mater::sim::SimulationRecipe::cloth_rope ||
               input.context == parallel_mater::sim::SimulationRecipe::soft_body_rope ||
               input.context == parallel_mater::sim::SimulationRecipe::rope_smoke) {
        input.camera_target = make_float3(0.0F, -0.45F, 0.0F);
        input.orbit_radius = 7.2F;
        input.orbit_pitch = 0.42F;
    } else if (input.context == parallel_mater::sim::SimulationRecipe::smoke ||
               input.context == parallel_mater::sim::SimulationRecipe::soft_body_smoke) {
        input.camera_target=make_float3(0.0F,-0.35F,-1.2F);
        input.orbit_radius=5.4F;
        input.orbit_pitch=0.28F;
    } else if (input.context == parallel_mater::sim::SimulationRecipe::fluid_smoke) {
        input.camera_target=make_float3(0.0F,-0.15F,-1.2F);
        input.orbit_radius=5.6F;
        input.orbit_pitch=0.46F;
    } else if (input.context == parallel_mater::sim::SimulationRecipe::cloth_smoke) {
        input.camera_target=make_float3(0.0F,0.0F,-1.2F);
        input.orbit_radius=4.8F;
        input.orbit_pitch=0.12F;
    }
}

struct CaptureFrame {
    waterlab::HybridState state;
    bool has_soft_body{};
    waterlab::SoftBodyState soft_body_state;
    waterlab::HybridTimings timings{};
    float3 rectangle_target{};
    float3 rectangle_control_force{};
    float rectangle_control_torque{};
    float visual_ms{};
    std::vector<float4> normal_foam;
    std::vector<waterlab::FoamParticle> foam_particles;
    std::uint64_t foam_tick{};
};

struct CaptureFileHeader {
    char magic[8]{'M','P','H','C','A','P','1','\0'};
    std::uint32_t version{6U};
    std::uint32_t frame_count{};
    std::uint32_t particle_count{};
    std::uint32_t skin_vertex_count{};
    std::uint32_t options_size{sizeof(waterlab::HybridOptions)};
    std::uint32_t rectangle_size{sizeof(waterlab::RectangleState)};
    std::uint32_t statistics_size{sizeof(waterlab::HybridStatistics)};
    std::uint32_t timings_size{sizeof(waterlab::HybridTimings)};
    std::uint32_t float3_size{sizeof(float3)};
};

struct SoftBodyCaptureHeader {
    std::uint32_t present{};
    std::uint32_t statistics_size{sizeof(waterlab::SoftBodyStatistics)};
    std::uint32_t voxel_count{};
    std::uint32_t edge_count{};
    std::uint32_t render_triangle_count{};
};

template <typename T>
void write_value(std::ofstream& output, const T& value)
{
    output.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

template <typename T>
void write_values(std::ofstream& output, const std::vector<T>& values)
{
    output.write(reinterpret_cast<const char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(T)));
}

template <typename T>
void read_value(std::ifstream& input, T& value)
{
    input.read(reinterpret_cast<char*>(&value), sizeof(value));
}

template <typename T>
void read_sized_value(std::ifstream& input, T& value, std::uint32_t byte_count)
{
    if (byte_count > sizeof(value)) {
        input.setstate(std::ios::failbit);
        return;
    }
    value = {};
    input.read(reinterpret_cast<char*>(&value), byte_count);
}

template <typename T>
void read_values(std::ifstream& input, std::vector<T>& values, std::size_t count)
{
    values.resize(count);
    input.read(reinterpret_cast<char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(T)));
}

void write_capture_frame(std::ofstream& output, const CaptureFrame& frame)
{
    write_value(output, frame.state.options);
    write_value(output, frame.state.rectangle);
    write_value(output, frame.state.statistics);
    write_value(output, frame.timings);
    write_value(output, frame.rectangle_target);
    write_value(output, frame.rectangle_control_force);
    write_value(output, frame.rectangle_control_torque);
    write_values(output, frame.state.particle_positions);
    write_values(output, frame.state.particle_velocities);
    write_values(output, frame.state.particle_forces);
    write_values(output, frame.state.particle_skin_owners);
    write_values(output, frame.state.skin_positions);
    write_values(output, frame.state.skin_velocities);
    write_values(output, frame.state.skin_forces);
    write_values(output, frame.state.skin_box_forces);
    write_values(output, frame.state.skin_box_impulses);
    write_values(output, frame.state.skin_box_pair_work);
    write_values(output, frame.state.skin_particle_forces);
    write_values(output, frame.state.skin_spring_forces);
    write_value(output, frame.visual_ms);
    write_values(output, frame.normal_foam);
    write_value(output, frame.foam_tick);
    write_values(output, frame.foam_particles);
    SoftBodyCaptureHeader soft_header;
    soft_header.present = frame.has_soft_body ? 1U : 0U;
    if (frame.has_soft_body) {
        soft_header.voxel_count = static_cast<std::uint32_t>(
            frame.soft_body_state.positions.size());
        soft_header.edge_count = static_cast<std::uint32_t>(
            frame.soft_body_state.active_edges.size());
        soft_header.render_triangle_count = static_cast<std::uint32_t>(
            frame.soft_body_state.active_render_triangles.size());
    }
    write_value(output, soft_header);
    if (frame.has_soft_body) {
        const std::uint32_t damage_count = static_cast<std::uint32_t>(
            frame.soft_body_state.edge_damage.size());
        write_value(output, damage_count);
    } else {
        write_value(output, std::uint32_t{});
    }
    if (frame.has_soft_body) {
        write_value(output, frame.soft_body_state.strength_multiplier);
        write_value(output, frame.soft_body_state.statistics);
        write_values(output, frame.soft_body_state.positions);
        write_values(output, frame.soft_body_state.velocities);
        write_values(output, frame.soft_body_state.active_edges);
        write_values(output, frame.soft_body_state.edge_damage);
        write_values(output, frame.soft_body_state.active_render_triangles);
    }
}

CaptureFrame read_capture_frame(
    std::ifstream& input, std::uint32_t particle_count, std::uint32_t skin_count,
    std::uint32_t options_size, std::uint32_t statistics_size,
    std::uint32_t timings_size, std::uint32_t version)
{
    CaptureFrame frame;
    // Course fields were appended to HybridOptions. Reading the old prefix
    // into defaults preserves build-specific lab captures without changing
    // the offsets of the remaining frame payload.
    frame.state.options = {};
    input.read(reinterpret_cast<char*>(&frame.state.options), options_size);
    read_value(input, frame.state.rectangle);
    read_sized_value(input, frame.state.statistics, statistics_size);
    read_sized_value(input, frame.timings, timings_size);
    read_value(input, frame.rectangle_target);
    read_value(input, frame.rectangle_control_force);
    read_value(input, frame.rectangle_control_torque);
    read_values(input, frame.state.particle_positions, particle_count);
    read_values(input, frame.state.particle_velocities, particle_count);
    read_values(input, frame.state.particle_forces, particle_count);
    read_values(input, frame.state.particle_skin_owners, particle_count);
    read_values(input, frame.state.skin_positions, skin_count);
    read_values(input, frame.state.skin_velocities, skin_count);
    read_values(input, frame.state.skin_forces, skin_count);
    read_values(input, frame.state.skin_box_forces, skin_count);
    read_values(input, frame.state.skin_box_impulses, skin_count);
    read_values(input, frame.state.skin_box_pair_work, skin_count);
    read_values(input, frame.state.skin_particle_forces, skin_count);
    read_values(input, frame.state.skin_spring_forces, skin_count);
    if (version >= 2U) {
        read_value(input, frame.visual_ms);
        read_values(input, frame.normal_foam, particle_count);
    }
    if (version >= 3U) {
        read_value(input, frame.foam_tick);
        read_values(input, frame.foam_particles, waterlab::FluidVisuals::foam_capacity);
    }
    if (version >= 4U) {
        SoftBodyCaptureHeader soft_header;
        read_value(input, soft_header);
        std::uint32_t damage_count{};
        if (version >= 5U) read_value(input, damage_count);
        if (!input || soft_header.present > 1U ||
            soft_header.statistics_size != sizeof(waterlab::SoftBodyStatistics) ||
            soft_header.voxel_count > 1'000'000U ||
            soft_header.edge_count > 10'000'000U ||
            soft_header.render_triangle_count > 10'000'000U ||
            damage_count > 10'000'000U ||
            (damage_count != 0U && damage_count != soft_header.edge_count)) {
            input.setstate(std::ios::failbit);
            return frame;
        }
        frame.has_soft_body = soft_header.present != 0U;
        if (frame.has_soft_body) {
            read_value(input, frame.soft_body_state.strength_multiplier);
            read_value(input, frame.soft_body_state.statistics);
            read_values(input, frame.soft_body_state.positions, soft_header.voxel_count);
            read_values(input, frame.soft_body_state.velocities, soft_header.voxel_count);
            read_values(input, frame.soft_body_state.active_edges, soft_header.edge_count);
            if (version >= 5U) {
                read_values(input, frame.soft_body_state.edge_damage, damage_count);
            }
            read_values(input, frame.soft_body_state.active_render_triangles,
                soft_header.render_triangle_count);
        } else if (soft_header.voxel_count != 0U || soft_header.edge_count != 0U ||
                   soft_header.render_triangle_count != 0U || damage_count != 0U) {
            input.setstate(std::ios::failbit);
        }
    }
    return frame;
}

class CaptureRing {
public:
    CaptureRing(
        std::size_t capacity, std::uint32_t particle_count, std::uint32_t skin_count)
        : frames_(capacity)
    {
        if (capacity == 0U) throw std::invalid_argument("capture capacity is zero");
        for (CaptureFrame& frame : frames_) {
            frame.state.particle_positions.resize(particle_count);
            frame.normal_foam.resize(particle_count);
            frame.foam_particles.resize(waterlab::FluidVisuals::foam_capacity);
            frame.state.particle_velocities.resize(particle_count);
            frame.state.particle_forces.resize(particle_count);
            frame.state.particle_skin_owners.resize(particle_count);
            frame.state.skin_positions.resize(skin_count);
            frame.state.skin_velocities.resize(skin_count);
            frame.state.skin_forces.resize(skin_count);
            frame.state.skin_box_forces.resize(skin_count);
            frame.state.skin_box_impulses.resize(skin_count);
            frame.state.skin_box_pair_work.resize(skin_count);
            frame.state.skin_particle_forces.resize(skin_count);
            frame.state.skin_spring_forces.resize(skin_count);
        }
    }

    void clear() noexcept { next_ = 0U; count_ = 0U; }

    void record(const waterlab::HybridDroplet& droplet,
        waterlab::HybridTimings timings, float3 target, float3 control_force,
        float control_torque, const waterlab::FluidVisuals& visuals, float visual_ms,
        const waterlab::SoftBodyCourse* soft_bodies = nullptr)
    {
        CaptureFrame& frame = frames_[next_];
        droplet.capture_state(frame.state);
        frame.has_soft_body = soft_bodies != nullptr;
        if (soft_bodies != nullptr) {
            soft_bodies->capture_state(frame.soft_body_state);
        } else {
            frame.soft_body_state = {};
        }
        frame.timings = timings;
        frame.rectangle_target = target;
        frame.rectangle_control_force = control_force;
        frame.rectangle_control_torque = control_torque;
        visuals.capture(frame.normal_foam);
        visuals.capture_foam(frame.foam_particles, frame.foam_tick);
        frame.visual_ms = visual_ms;
        next_ = (next_ + 1U) % frames_.size();
        count_ = std::min(count_ + 1U, frames_.size());
    }

    [[nodiscard]] std::size_t size() const noexcept { return count_; }

    [[nodiscard]] const CaptureFrame* latest() const noexcept
    {
        return count_ == 0U ? nullptr : &chronological(count_ - 1U);
    }

    [[nodiscard]] const CaptureFrame& chronological(std::size_t index) const
    {
        const std::size_t begin = (next_ + frames_.size() - count_) % frames_.size();
        return frames_[(begin + index) % frames_.size()];
    }

private:
    std::vector<CaptureFrame> frames_;
    std::size_t next_{};
    std::size_t count_{};
};

std::filesystem::path capture_directory(std::uint64_t frame_index)
{
    const std::time_t now = std::time(nullptr);
    std::tm local{};
    localtime_r(&now, &local);
    std::ostringstream name;
    name << "capture-" << std::put_time(&local, "%Y%m%d-%H%M%S")
         << "-frame-" << frame_index;
    return std::filesystem::path("/tmp/meshprep-hybrid-captures") / name.str();
}

std::filesystem::path save_capture(const CaptureRing& capture)
{
    if (capture.size() == 0U) throw std::runtime_error("capture buffer is empty");
    const CaptureFrame& final_frame = capture.chronological(capture.size() - 1U);
    const std::filesystem::path directory =
        capture_directory(final_frame.state.statistics.frame_index);
    std::filesystem::create_directories(directory);
    const std::filesystem::path data_path = directory / "capture.bin";
    std::ofstream output(data_path, std::ios::binary);
    if (!output) throw std::runtime_error("cannot create " + data_path.string());
    CaptureFileHeader header;
    header.frame_count = static_cast<std::uint32_t>(capture.size());
    header.particle_count = final_frame.state.statistics.particle_count;
    header.skin_vertex_count = final_frame.state.statistics.physical_skin_vertices;
    write_value(output, header);
    for (std::size_t frame = 0U; frame < capture.size(); ++frame) {
        write_capture_frame(output, capture.chronological(frame));
    }
    output.close();
    if (!output) throw std::runtime_error("failed writing " + data_path.string());

    std::ofstream manifest(directory / "manifest.txt");
    if (!manifest) throw std::runtime_error("cannot create capture manifest");
    manifest << "format=meshprep-hybrid-capture-v" << header.version << '\n'
             << "visual_state=particle-normals-independent-foam-and-soft-bodies\n"
             << "soft_body_state=" << (final_frame.has_soft_body ? "present" : "absent") << '\n'
             << "foam_capacity=" << waterlab::FluidVisuals::foam_capacity << '\n'
             << "frames=" << capture.size() << '\n'
             << "first_physics_frame="
             << capture.chronological(0U).state.statistics.frame_index << '\n'
             << "last_physics_frame=" << final_frame.state.statistics.frame_index << '\n'
             << "particles=" << header.particle_count << '\n'
             << "physical_skin_vertices=" << header.skin_vertex_count << '\n'
             << "fixed_dt=" << final_frame.state.options.fixed_dt << '\n'
             << "replay_command=./build/parallel-mater-lab --replay "
             << directory.string() << '\n';
    manifest.close();
    std::filesystem::create_directories("/tmp/meshprep-hybrid-captures");
    std::ofstream latest("/tmp/meshprep-hybrid-captures/LAST_CAPTURE.txt");
    latest << directory.string() << '\n';
    return directory;
}

std::vector<CaptureFrame> load_capture(std::filesystem::path path)
{
    if (std::filesystem::is_directory(path)) path /= "capture.bin";
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open capture " + path.string());
    CaptureFileHeader header;
    read_value(input, header);
    const CaptureFileHeader expected;
    constexpr std::uint32_t legacy_options_size = static_cast<std::uint32_t>(
        offsetof(waterlab::HybridOptions, obstacle_course));
    constexpr std::uint32_t previous_options_size = static_cast<std::uint32_t>(
        offsetof(waterlab::HybridOptions, particle_skin_coupling));
    constexpr std::uint32_t recorded_options_size = static_cast<std::uint32_t>(
        offsetof(waterlab::HybridOptions, particle_capacity));
    constexpr std::uint32_t legacy_statistics_size = static_cast<std::uint32_t>(
        offsetof(waterlab::HybridStatistics, soft_body_contact_count));
    constexpr std::uint32_t recorded_statistics_size = static_cast<std::uint32_t>(
        offsetof(waterlab::HybridStatistics, recycled_particles));
    constexpr std::uint32_t legacy_timings_size = static_cast<std::uint32_t>(
        offsetof(waterlab::HybridTimings, update_soft_body_physics_ms));
    constexpr std::uint32_t recorded_timings_size = static_cast<std::uint32_t>(
        offsetof(waterlab::HybridTimings, update_rigid_body_contact_ms));
    const bool supported_options = header.options_size == expected.options_size ||
        header.options_size == recorded_options_size ||
        header.options_size == previous_options_size ||
        header.options_size == legacy_options_size;
    const bool supported_statistics =
        header.statistics_size == expected.statistics_size ||
        header.statistics_size == recorded_statistics_size ||
        header.statistics_size == legacy_statistics_size;
    const bool supported_timings = header.timings_size == expected.timings_size ||
        header.timings_size == recorded_timings_size ||
        header.timings_size == legacy_timings_size;
    if (!input || std::memcmp(header.magic, expected.magic, sizeof(header.magic)) != 0 ||
        (header.version < 1U || header.version > expected.version) || !supported_options ||
        header.rectangle_size != expected.rectangle_size ||
        !supported_statistics || !supported_timings ||
        header.float3_size != expected.float3_size) {
        throw std::runtime_error("capture format does not match this build");
    }
    if (header.frame_count == 0U || header.frame_count > 10'000U ||
        header.particle_count != 10'000U || header.skin_vertex_count != 1'002U) {
        throw std::runtime_error("capture dimensions are invalid");
    }
    std::vector<CaptureFrame> frames;
    frames.reserve(header.frame_count);
    for (std::uint32_t frame = 0U; frame < header.frame_count; ++frame) {
        frames.push_back(read_capture_frame(
            input, header.particle_count, header.skin_vertex_count,
            header.options_size, header.statistics_size, header.timings_size,
            header.version));
        if (!input) throw std::runtime_error("capture ended before its declared frame count");
    }
    return frames;
}

bool parse_u32(std::string_view text, std::uint32_t& value)
{
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    return result.ec == std::errc{} && result.ptr == text.data() + text.size();
}

bool parse_options(int argc, char** argv, Options& options)
{
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument = argv[index];
        if (argument == "--help") return false;
        if (argument == "--drive-box") {
            options.drive_box = true;
            continue;
        }
        if (index + 1 >= argc) return false;
        const std::string_view value = argv[++index];
        if (argument == "--width") {
            if (!parse_u32(value, options.width) || options.width < 64U) return false;
        } else if (argument == "--height") {
            if (!parse_u32(value, options.height) || options.height < 64U) return false;
        } else if (argument == "--profile") {
            if (!parse_u32(value, options.profile_frames) || options.profile_frames == 0U) return false;
        } else if (argument == "--warmups") {
            if (!parse_u32(value, options.warmups)) return false;
        } else if (argument == "--iterations") {
            if (!parse_u32(value, options.physics_iterations) ||
                options.physics_iterations < 1U ||
                options.physics_iterations >
                    waterlab::HybridDroplet::maximum_physics_iterations) return false;
        } else if (argument == "--scene") {
            if (value == "course") options.scene = Scene::Course;
            else if (value == "lab") options.scene = Scene::Lab;
            else return false;
        } else if (argument == "--context") {
            const auto* context = parallel_mater::sim::find_simulation_recipe(value);
            if (context == nullptr) return false;
            options.context = context->recipe;
            options.scene = Scene::Course;
        } else if (argument == "--view") {
            if (value == "surface") options.fluid_display = waterlab::FluidDisplay::Surface;
            else if (value == "particles") options.fluid_display = waterlab::FluidDisplay::Particles;
            else if (value == "billboards") options.fluid_display = waterlab::FluidDisplay::Billboards;
            else if (value == "skin" || value == "wire")
                options.fluid_display = waterlab::FluidDisplay::Wireframe;
            else return false;
            options.view_explicit = true;
        } else if (argument == "--foam") {
            if (value != "on" && value != "off") return false;
            options.show_foam = value == "on";
        } else if (argument == "--replay") {
            options.replay_path = value;
        } else {
            return false;
        }
    }
    if (options.view_explicit &&
        options.fluid_display == waterlab::FluidDisplay::Surface &&
        !waterlab::gallery::recipe_has(
            options.context, parallel_mater::sim::Component::fluid_particles)) {
        return false;
    }
    return true;
}

void usage(const char* executable)
{
    std::fprintf(stderr,
        "usage: %s [--width N] [--height N] [--profile FRAMES] [--warmups N] "
        "[--iterations 1..16] [--scene course|lab] [--context RECIPE] [--drive-box] "
        "[--view surface|particles|billboards|wire] [--foam on|off] [--replay CAPTURE_DIRECTORY]\n"
        "context recipes use catalog slugs such as water, fluid-smoke, or cloth-rope\n"
        "rigid contexts default to 4 substeps per fixed 60 Hz tick at 4x speed; "
        "non-rigid and lab contexts to 1\n"
        "brackets use 1x..8x and raise the course substep floor with speed\n",
        executable);
}

float3 add(float3 a, float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

float3 subtract(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

float3 multiply(float3 value, float scale)
{
    return make_float3(value.x * scale, value.y * scale, value.z * scale);
}

float3 cross(float3 a, float3 b)
{
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

float dot(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

float length(float3 value)
{
    return std::sqrt(value.x * value.x + value.y * value.y + value.z * value.z);
}

float3 normalize(float3 value)
{
    return multiply(value, 1.0F / std::max(length(value), 1.0e-20F));
}

float3 clamp_length(float3 value, float maximum)
{
    const float magnitude = length(value);
    return magnitude > maximum ? multiply(value, maximum / magnitude) : value;
}

waterlab::SoftBodyTimings step_smoke_coupled_body(
    waterlab::SoftBodyCourse& body,parallel_mater::physics::Smoke& smoke,
    waterlab::RigidSphereState* sphere,float3 body_gravity,float3 sphere_gravity,
    waterlab::GalleryArena arena,std::uint32_t substeps,float fixed_dt,
    parallel_mater::physics::SmokeTimings& smoke_timings)
{
    body.begin_frame();
    const float dt=fixed_dt/static_cast<float>(std::max(1U,substeps));
    for (std::uint32_t substep=0U;substep<std::max(1U,substeps);++substep) {
        if (sphere!=nullptr) {
            sphere->velocity=add(sphere->velocity,multiply(sphere_gravity,dt));
            sphere->velocity=multiply(sphere->velocity,1.0F/(1.0F+0.22F*dt));
            sphere->center=add(sphere->center,multiply(sphere->velocity,dt));
            waterlab::project_gallery_contact(
                sphere->center,sphere->velocity,sphere->radius,arena);
        }
        body.prepare_substep(dt,body_gravity);
        if (sphere!=nullptr) body.contact_rigid_sphere_substep(*sphere,dt);
        const waterlab::SoftBodyVoxelView nodes=body.voxel_view();
        const parallel_mater::Status coupled=smoke.couple({
            nodes.positions,nodes.velocities,nodes.external_impulses,
            nodes.voxel_count,nodes.inverse_voxel_mass,nodes.voxel_radius,dt},
            5.0F,smoke_timings);
        if (!coupled.ok()) throw std::runtime_error(
            std::string("couple smoke: ")+coupled.message);
        body.finish_substep(dt,body_gravity);
        if (sphere!=nullptr)
            waterlab::advance_rigid_sphere_rotation(*sphere,arena,
                body.material().ground_friction,dt);
    }
    return body.finish_frame();
}

void update_camera(const Interaction& input, waterlab::Camera& camera)
{
    const float horizontal = input.orbit_radius * std::cos(input.orbit_pitch);
    camera.target = input.camera_target;
    camera.eye = add(input.camera_target, make_float3(
        horizontal * std::sin(input.orbit_yaw),
        input.orbit_radius * std::sin(input.orbit_pitch),
        horizontal * std::cos(input.orbit_yaw)));
}

float3 skin_center(const CaptureFrame& frame)
{
    float3 center{};
    for (const float3 position : frame.state.skin_positions) {
        center = add(center, position);
    }
    return frame.state.skin_positions.empty()
        ? center
        : multiply(center, 1.0F / static_cast<float>(frame.state.skin_positions.size()));
}

void follow_course(Interaction& input, const CaptureFrame& frame)
{
    const float3 center = skin_center(frame);
    if (!input.have_follow_center) {
        input.camera_target = add(center, make_float3(0.0F, -0.10F, -0.75F));
        input.have_follow_center = true;
    } else {
        input.camera_target = add(input.camera_target, subtract(center, input.follow_center));
    }
    input.follow_center = center;
    input.droplet_center = center;
    input.course_progress = std::clamp(
        -center.z / -waterlab::course_goal_z,
        0.0F, 1.0F);
    const float goal_x = center.x;
    const float goal_z = center.z - waterlab::course_goal_z;
    if (goal_x * goal_x + goal_z * goal_z <=
        waterlab::course_goal_radius * waterlab::course_goal_radius) {
        input.course_finished = true;
    }
}

void reset_level_tracking(Interaction& input, const waterlab::RigidSphereState& sphere)
{
    input.progression.select(input.context);
    input.objective_progress = {input.context};
    input.painted_fraction = 0.0F;
    input.reset_bowl_paint = true;
    input.rope_turns = 0.0F;
    input.have_previous_rope_angle = false;
    input.previous_rigid_center = sphere.center;
    input.have_previous_rigid_center = true;
    input.goal_cloth_damaged = false;
    input.inspected_broken_edges = 0U;
    input.fishing_latched = false;
}

void update_level_progress(Interaction& input,
    const waterlab::HybridDroplet& droplet,
    const waterlab::SoftBodyCourse* soft_bodies,
    const waterlab::RigidSphereState& rigid_sphere)
{
    if (!input.course_mode || input.replay_mode || input.paused) return;
    using parallel_mater::sim::SimulationRecipe;
    waterlab::ObjectiveMetrics metrics;
    metrics.course_goal_reached = input.course_finished;

    if (input.context == SimulationRecipe::water ||
        input.context == SimulationRecipe::soft_body)
        metrics.painted_fraction = input.painted_fraction;
    if (soft_bodies != nullptr) {
        const std::uint32_t broken = soft_bodies->statistics().broken_edge_count;
        if (input.context == SimulationRecipe::cloth_soft_body) {
            // The merged asset stores sphere nodes first, then the 24x24 goal
            // curtain, then the ground cloth. Only a broken goal-curtain bond
            // completes this level; damage in the rolling body or pit cover
            // is deliberately ignored.
            if (!input.goal_cloth_damaged &&
                broken != input.inspected_broken_edges) {
                const auto lattice = soft_bodies->lattice_view();
                std::vector<waterlab::SoftBodyEdge> edges(
                    lattice.edges_per_instance);
                std::vector<std::uint8_t> active(lattice.edges_per_instance);
                if (cudaMemcpy(edges.data(), lattice.edges,
                        edges.size()*sizeof(edges[0]), cudaMemcpyDeviceToHost) ==
                        cudaSuccess &&
                    cudaMemcpy(active.data(), lattice.active_edges,
                        active.size()*sizeof(active[0]), cudaMemcpyDeviceToHost) ==
                        cudaSuccess) {
                    constexpr std::uint32_t first_goal_node = 1'000U;
                    constexpr std::uint32_t goal_node_count = 24U*24U;
                    const std::uint32_t goal_end = first_goal_node + goal_node_count;
                    for (std::size_t edge = 0U; edge < edges.size(); ++edge) {
                        if (active[edge] != 0U) continue;
                        const uint2 vertices = edges[edge].vertices;
                        if (vertices.x >= first_goal_node && vertices.x < goal_end &&
                            vertices.y >= first_goal_node && vertices.y < goal_end) {
                            input.goal_cloth_damaged = true;
                            break;
                        }
                    }
                    input.inspected_broken_edges = broken;
                }
            }
            metrics.broken_connections = input.goal_cloth_damaged ? 1U : 0U;
        } else {
            metrics.broken_connections = broken;
        }
    }
    metrics.exit_reached = input.context == SimulationRecipe::soft_body &&
        rigid_sphere.center.x > 1.85F;
    if (input.context == SimulationRecipe::cloth_rope ||
        input.context == SimulationRecipe::soft_body_rope)
        metrics.exit_reached = rigid_sphere.center.z <
            -waterlab::rope_bridge_land_inner_z - 0.45F;
    if (input.context == SimulationRecipe::water_soft_body) {
        const float start = waterlab::water_wheel_center.x +
            0.5F*(waterlab::water_wheel_top_platform_outer_x+
                waterlab::water_wheel_top_platform_gap_half_width);
        const float exit = waterlab::water_wheel_center.x-
            waterlab::water_wheel_top_platform_outer_x-rigid_sphere.radius;
        metrics.lift_progress = std::clamp(
            (start-rigid_sphere.center.x)/std::max(start-exit,1.0e-5F),
            0.0F, 1.0F);
    }
    metrics.treasure_caught = input.fishing_latched;
    if (input.context == SimulationRecipe::water_rope && input.fishing_latched) {
        const float start = waterlab::fishing_chest_start.y;
        const float recovered = waterlab::fishing_tank_center.y +
            waterlab::fishing_tank_half_extents.y - rigid_sphere.radius - 0.12F;
        metrics.treasure_lift_progress = std::clamp(
            (rigid_sphere.center.y - start) / std::max(recovered - start, 1.0e-5F),
            0.0F, 1.0F);
    }
    if (input.context == SimulationRecipe::rope) {
        const float angle = std::atan2(
            rigid_sphere.center.z - waterlab::rope_post_center.z,
            rigid_sphere.center.x - waterlab::rope_post_center.x);
        if (input.have_previous_rope_angle) {
            float delta = angle - input.previous_rope_angle;
            constexpr float pi = 3.14159265358979323846F;
            if (delta > pi) delta -= 2.0F * pi;
            if (delta < -pi) delta += 2.0F * pi;
            input.rope_turns += delta / (2.0F * pi);
        }
        input.previous_rope_angle = angle;
        input.have_previous_rope_angle = true;
        metrics.rope_turns = std::fabs(input.rope_turns);
    }

    input.previous_rigid_center = rigid_sphere.center;
    input.have_previous_rigid_center = true;
    input.objective_progress = input.progression.update(metrics);
    if (input.objective_progress.advanced)
        input.pending_context = input.progression.current();
    (void)droplet;
}

void update_course_gravity(
    GLFWwindow* window,
    Interaction& input,
    const waterlab::Camera& camera,
    waterlab::HybridDroplet& droplet)
{
    const auto pressed = [window](int key) {
        return glfwGetKey(window, key) == GLFW_PRESS;
    };
    float forward_input = static_cast<float>(pressed(GLFW_KEY_W)) -
        static_cast<float>(pressed(GLFW_KEY_S));
    float right_input = static_cast<float>(pressed(GLFW_KEY_D)) -
        static_cast<float>(pressed(GLFW_KEY_A));
    if (!input.show_physics && !input.show_quantities &&
        input.context!=parallel_mater::sim::SimulationRecipe::water_rope) {
        forward_input += static_cast<float>(pressed(GLFW_KEY_UP)) -
            static_cast<float>(pressed(GLFW_KEY_DOWN));
        right_input += static_cast<float>(pressed(GLFW_KEY_RIGHT)) -
            static_cast<float>(pressed(GLFW_KEY_LEFT));
    }
    float3 forward = subtract(camera.target, camera.eye);
    forward.y = 0.0F;
    forward = normalize(forward);
    const float3 right = normalize(cross(forward, make_float3(0.0F, 1.0F, 0.0F)));
    float3 steering = add(multiply(forward, forward_input), multiply(right, right_input));
    const float steering_length = length(steering);
    if (input.course_motion_adjustment != 0) {
        const auto options = waterlab::with_course_motion_multiplier(droplet.options(),
            waterlab::course_motion_multiplier(droplet.options()) + input.course_motion_adjustment);
        droplet.set_runtime_options(options);
        input.course_gravity = options.gravity;
        input.course_motion_adjustment = 0;
    }
    const bool sphere_only_gravity = input.context ==
        parallel_mater::sim::SimulationRecipe::water_soft_body;
    const float gravity_magnitude = sphere_only_gravity
        ? length(input.course_gravity) : length(droplet.options().gravity);
    const float maximum_tilt = waterlab::gallery::gravity_tilt_degrees(input.context) *
        3.14159265358979323846F / 180.0F;
    float3 desired = make_float3(0.0F, -gravity_magnitude, 0.0F);
    if (steering_length > 0.0F) {
        steering = multiply(steering, 1.0F / steering_length);
        desired = make_float3(
            gravity_magnitude * std::sin(maximum_tilt) * steering.x,
            -gravity_magnitude * std::cos(maximum_tilt),
            gravity_magnitude * std::sin(maximum_tilt) * steering.z);
    }
    constexpr float response_seconds = 0.16F;
    const float blend = 1.0F - std::exp(-droplet.options().fixed_dt / response_seconds);
    input.course_gravity = add(
        input.course_gravity, multiply(subtract(desired, input.course_gravity), blend));
    input.course_gravity = length(input.course_gravity) > 1.0e-8F
        ? multiply(normalize(input.course_gravity), gravity_magnitude) : float3{};
    waterlab::HybridOptions options = droplet.options();
    if (!sphere_only_gravity) {
        options.gravity = input.course_gravity;
        droplet.set_runtime_options(options);
    }
}

void update_fishing_controls(GLFWwindow* window,Interaction& input,
    waterlab::SoftBodyCourse* rope,float dt)
{
    if (input.context!=parallel_mater::sim::SimulationRecipe::water_rope ||
        rope==nullptr || input.show_physics || input.show_quantities) return;
    const auto pressed=[window](int key) {
        return glfwGetKey(window,key)==GLFW_PRESS;
    };
    const float horizontal=static_cast<float>(pressed(GLFW_KEY_RIGHT))-
        static_cast<float>(pressed(GLFW_KEY_LEFT));
    if (horizontal!=0.0F) {
        const float desired=std::clamp(
            input.fishing_head_x+horizontal*2.0F*dt,
            -waterlab::fishing_tank_half_extents.x+0.10F,
            waterlab::fishing_tank_half_extents.x-0.10F);
        const float delta=desired-input.fishing_head_x;
        rope->translate_pinned(make_float3(delta,0.0F,0.0F));
        input.fishing_head_x=desired;
    }
    const float reel=static_cast<float>(pressed(GLFW_KEY_DOWN))-
        static_cast<float>(pressed(GLFW_KEY_UP));
    if (reel!=0.0F) {
        const float desired=std::clamp(
            input.fishing_rope_scale+reel*0.36F*dt,0.12F,1.18F);
        if (desired!=input.fishing_rope_scale) {
            rope->scale_rest_lengths(desired/input.fishing_rope_scale);
            input.fishing_rope_scale=desired;
        }
    }
}

void update_fishing_latch(Interaction& input,waterlab::SoftBodyCourse* rope,
    waterlab::RigidSphereState& chest)
{
    if (input.context!=parallel_mater::sim::SimulationRecipe::water_rope || rope==nullptr)
        return;
    const auto lattice=rope->voxel_view();
    if (lattice.voxel_count==0U) return;
    const std::uint32_t hook=lattice.voxel_count-1U;
    float3 hook_position{};
    float3 hook_velocity{};
    if (cudaMemcpy(&hook_position,lattice.positions+hook,sizeof(float3),
            cudaMemcpyDeviceToHost)!=cudaSuccess ||
        cudaMemcpy(&hook_velocity,lattice.velocities+hook,sizeof(float3),
            cudaMemcpyDeviceToHost)!=cudaSuccess) return;
    const float3 separation=subtract(chest.center,hook_position);
    if (!input.fishing_latched && length(separation)<=chest.radius+0.14F)
        input.fishing_latched=true;
    if (!input.fishing_latched) return;
    const float3 target=add(hook_position,make_float3(0.0F,-chest.radius-0.08F,0.0F));
    const float3 correction=subtract(target,chest.center);
    chest.velocity=add(multiply(chest.velocity,0.20F),
        multiply(correction,0.80F/((1.0F/60.0F))));
    chest.center=target;
    rope->set_uniform_velocity(hook,1U,
        add(multiply(hook_velocity,0.55F),multiply(chest.velocity,0.45F)));
}

void apply_soft_body_strength(
    Interaction& input, waterlab::SoftBodyCourse* soft_bodies)
{
    if (soft_bodies == nullptr || input.soft_body_strength_adjustment == 0) return;
    const float multiplier = std::clamp(
        soft_bodies->strength_multiplier() *
            std::pow(1.25F, static_cast<float>(input.soft_body_strength_adjustment)),
        0.0625F, 64.0F);
    soft_bodies->set_strength_multiplier(multiplier);
    input.soft_body_strength_adjustment = 0;
}

void apply_camera_pan(Interaction& input, const waterlab::Camera& camera, int height)
{
    if (input.pan_x == 0.0 && input.pan_y == 0.0) return;
    const float3 forward = normalize(subtract(camera.target, camera.eye));
    const float3 right = normalize(cross(forward, camera.up));
    const float3 up = normalize(cross(right, forward));
    constexpr float pi = 3.14159265358979323846F;
    const float world_per_pixel = 2.0F * input.orbit_radius *
        std::tan(camera.vertical_fov_degrees * pi / 360.0F) /
        static_cast<float>(std::max(height, 1));
    input.camera_target = add(input.camera_target, add(
        multiply(right, static_cast<float>(-input.pan_x) * world_per_pixel),
        multiply(up, static_cast<float>(input.pan_y) * world_per_pixel)));
    input.pan_x = 0.0;
    input.pan_y = 0.0;
}

void apply_box_drag(Interaction& input, const waterlab::Camera& camera, int height)
{
    if (input.drag_x == 0.0 && input.drag_y == 0.0) return;
    const float3 forward = normalize(subtract(camera.target, camera.eye));
    const float3 right = normalize(cross(forward, camera.up));
    const float3 up = normalize(cross(right, forward));
    constexpr float pi = 3.14159265358979323846F;
    const float world_per_pixel = 2.0F * input.orbit_radius *
        std::tan(camera.vertical_fov_degrees * pi / 360.0F) /
        static_cast<float>(std::max(height, 1));
    input.rectangle_target = add(input.rectangle_target, add(
        multiply(right, static_cast<float>(input.drag_x) * world_per_pixel),
        multiply(up, static_cast<float>(-input.drag_y) * world_per_pixel)));
    input.drag_x = 0.0;
    input.drag_y = 0.0;
}

float3 rectangle_control_force(
    const Interaction& input,
    const waterlab::RectangleState& rectangle,
    const waterlab::HybridOptions& options)
{
    const float3 position_force = multiply(
        subtract(input.rectangle_target, rectangle.center),
        options.rectangle_target_stiffness);
    const float critical_damping = 2.0F * std::sqrt(
        options.rectangle_target_stiffness * options.rectangle_mass);
    const float3 damping_force = multiply(
        rectangle.velocity,
        -options.rectangle_target_damping_ratio * critical_damping);
    return clamp_length(
        add(position_force, damping_force), options.maximum_rectangle_control_force);
}

void mouse_button(GLFWwindow* window, int button, int action, int modifiers)
{
    auto& input = *static_cast<Interaction*>(glfwGetWindowUserPointer(window));
    if (button != GLFW_MOUSE_BUTTON_LEFT) return;
    if (action == GLFW_PRESS) {
        glfwGetCursorPos(window, &input.previous_x, &input.previous_y);
        input.panning = (modifiers & GLFW_MOD_CONTROL) != 0;
        input.box_dragging = !input.course_mode && !input.panning &&
            (modifiers & GLFW_MOD_SHIFT) != 0;
        input.orbiting = !input.panning && !input.box_dragging;
    } else if (action == GLFW_RELEASE) {
        input.panning = false;
        input.box_dragging = false;
        input.orbiting = false;
    }
}

void cursor_position(GLFWwindow* window, double x, double y)
{
    auto& input = *static_cast<Interaction*>(glfwGetWindowUserPointer(window));
    const double dx = x - input.previous_x;
    const double dy = y - input.previous_y;
    input.previous_x = x;
    input.previous_y = y;
    if (input.orbiting) {
        input.orbit_yaw -= static_cast<float>(dx) * 0.006F;
        input.orbit_pitch = std::clamp(
            input.orbit_pitch - static_cast<float>(dy) * 0.006F, -1.45F, 1.45F);
    } else if (input.panning) {
        input.pan_x += dx;
        input.pan_y += dy;
    } else if (input.box_dragging) {
        input.drag_x += dx;
        input.drag_y += dy;
    }
}

void scroll(GLFWwindow* window, double, double offset)
{
    auto& input = *static_cast<Interaction*>(glfwGetWindowUserPointer(window));
    input.orbit_radius = std::clamp(
        input.orbit_radius * std::exp(static_cast<float>(-offset) * 0.12F), 1.8F, 9.0F);
}

void key_callback(GLFWwindow* window, int key, int, int action, int modifiers)
{
    auto& input = *static_cast<Interaction*>(glfwGetWindowUserPointer(window));
    if (key == GLFW_KEY_Q) input.rotate_left = action != GLFW_RELEASE;
    if (key == GLFW_KEY_E) input.rotate_right = action != GLFW_RELEASE;
    if (action == GLFW_PRESS) {
        if (input.show_context_browser) {
            const int count=static_cast<int>(parallel_mater::sim::simulation_recipes.size());
            if (key==GLFW_KEY_ESCAPE || key==GLFW_KEY_TAB) {
                input.show_context_browser=false;
                return;
            }
            if (key==GLFW_KEY_UP || key==GLFW_KEY_DOWN) {
                input.context_browser_index=(input.context_browser_index+
                    (key==GLFW_KEY_DOWN ? 1 : -1)+count)%count;
                return;
            }
            if (key==GLFW_KEY_ENTER || key==GLFW_KEY_KP_ENTER) {
                input.pending_context=parallel_mater::sim::simulation_recipes[
                    static_cast<std::size_t>(input.context_browser_index)].recipe;
                input.show_context_browser=false;
                return;
            }
        }
        if (key == GLFW_KEY_ESCAPE) glfwSetWindowShouldClose(window, GLFW_TRUE);
        else if (key == GLFW_KEY_TAB) {
            input.show_context_browser=true;
            const auto found=std::find_if(parallel_mater::sim::simulation_recipes.begin(),
                parallel_mater::sim::simulation_recipes.end(),[&](const auto& item) {
                    return item.recipe==input.context;
                });
            input.context_browser_index=found==parallel_mater::sim::simulation_recipes.end()
                ? 0 : static_cast<int>(found-parallel_mater::sim::simulation_recipes.begin());
        }
        else if (key == GLFW_KEY_SPACE) input.paused = !input.paused;
        else if (key == GLFW_KEY_P) {
            input.show_physics = !input.show_physics;
            if (input.show_physics) input.show_quantities = false;
        }
        else if (key == GLFW_KEY_L) {
            input.show_quantities = !input.show_quantities;
            if (input.show_quantities) {
                input.show_physics = false;
                const std::vector<int> rows = relevant_quantity_parameters(input);
                if (!rows.empty() && std::find(rows.begin(), rows.end(),
                        input.quantity_parameter) == rows.end())
                    input.quantity_parameter = rows.front();
            }
        }
        else if (key == GLFW_KEY_V) {
            const auto default_view =
                waterlab::gallery::default_recipe_display(input.context);
            input.fluid_display = input.fluid_display == waterlab::FluidDisplay::Wireframe
                ? (default_view == waterlab::FluidDisplay::Wireframe
                    ? waterlab::FluidDisplay::Particles : default_view)
                : waterlab::FluidDisplay::Wireframe;
        }
        else if (key == GLFW_KEY_K) {
            input.fluid_display = input.fluid_display == waterlab::FluidDisplay::Billboards
                ? waterlab::gallery::default_recipe_display(input.context)
                : waterlab::FluidDisplay::Billboards;
        }
        else if (key == GLFW_KEY_F) input.show_foam = !input.show_foam;
        else if (key == GLFW_KEY_T) input.show_timings = !input.show_timings;
        else if (key == GLFW_KEY_Z) input.show_normals = !input.show_normals;
        else if (key == GLFW_KEY_X) input.show_box_forces = !input.show_box_forces;
        else if (key == GLFW_KEY_C) input.show_particle_forces = !input.show_particle_forces;
        else if (key == GLFW_KEY_B) input.show_spring_forces = !input.show_spring_forces;
        else if (key == GLFW_KEY_M && !input.replay_mode &&
                 input.context == parallel_mater::sim::SimulationRecipe::water_cloth) {
            input.capture_requested = true;
            input.paused = true;
        }
        else if (key == GLFW_KEY_R) input.reset = true;
    }
    if (input.replay_mode && (action == GLFW_PRESS || action == GLFW_REPEAT)) {
        const int amount = (modifiers & GLFW_MOD_SHIFT) != 0 ? 60 : 1;
        if (key == GLFW_KEY_LEFT) input.replay_delta -= amount;
        else if (key == GLFW_KEY_RIGHT) input.replay_delta += amount;
        return;
    }
    if (input.course_mode &&
        input.context == parallel_mater::sim::SimulationRecipe::water_cloth &&
        !input.replay_mode &&
        (action == GLFW_PRESS || action == GLFW_REPEAT) &&
        (key == GLFW_KEY_LEFT_BRACKET || key == GLFW_KEY_RIGHT_BRACKET)) {
        input.course_motion_adjustment += key == GLFW_KEY_RIGHT_BRACKET ? 1 : -1;
        return;
    }
    const bool deformable_context =
        waterlab::gallery::recipe_has(
            input.context, parallel_mater::sim::Component::soft_body) ||
        waterlab::gallery::recipe_has(
            input.context, parallel_mater::sim::Component::cloth) ||
        waterlab::gallery::recipe_has(
            input.context, parallel_mater::sim::Component::rope);
    if (input.course_mode && deformable_context && !input.replay_mode &&
        (action == GLFW_PRESS || action == GLFW_REPEAT) &&
        (key == GLFW_KEY_COMMA || key == GLFW_KEY_PERIOD)) {
        input.soft_body_strength_adjustment += key == GLFW_KEY_PERIOD ? 1 : -1;
        return;
    }
    if ((!input.show_physics && !input.show_quantities) ||
        (action != GLFW_PRESS && action != GLFW_REPEAT)) return;
    if (input.show_quantities) {
        const std::vector<int> rows = relevant_quantity_parameters(input);
        if (rows.empty()) return;
        if (key == GLFW_KEY_UP || key == GLFW_KEY_DOWN) {
            const int direction = key == GLFW_KEY_DOWN ? 1 : -1;
            const auto found = std::find(
                rows.begin(), rows.end(), input.quantity_parameter);
            const int current = found == rows.end() ? 0 :
                static_cast<int>(found - rows.begin());
            const int next = (current + direction + static_cast<int>(rows.size())) %
                static_cast<int>(rows.size());
            input.quantity_parameter = rows[static_cast<std::size_t>(next)];
        } else if (key == GLFW_KEY_LEFT || key == GLFW_KEY_RIGHT) {
            const int direction = key == GLFW_KEY_RIGHT ? 1 : -1;
            input.quantity_adjustment += direction *
                ((modifiers & GLFW_MOD_SHIFT) != 0 ? 10 : 1);
        }
        return;
    }
    if (key == GLFW_KEY_UP || key == GLFW_KEY_DOWN) {
        const int direction = key == GLFW_KEY_DOWN ? 1 : -1;
        const std::vector<int> rows = relevant_physics_parameters(input);
        const auto found = std::find(rows.begin(), rows.end(), input.physics_parameter);
        const int current = found == rows.end() ? 0 :
            static_cast<int>(found - rows.begin());
        const int next = (current + direction + static_cast<int>(rows.size())) %
            static_cast<int>(rows.size());
        input.physics_parameter = rows[static_cast<std::size_t>(next)];
    } else if (key == GLFW_KEY_LEFT || key == GLFW_KEY_RIGHT) {
        const int direction = key == GLFW_KEY_RIGHT ? 1 : -1;
        input.physics_adjustment += direction * ((modifiers & GLFW_MOD_SHIFT) != 0 ? 10 : 1);
    }
}

std::array<std::uint8_t, 7> glyph(char c)
{
    switch (c) {
    case 'A': return {14,17,17,31,17,17,17};
    case 'B': return {30,17,17,30,17,17,30};
    case 'C': return {14,17,16,16,16,17,14};
    case 'D': return {30,17,17,17,17,17,30};
    case 'E': return {31,16,16,30,16,16,31};
    case 'F': return {31,16,16,30,16,16,16};
    case 'G': return {14,17,16,23,17,17,14};
    case 'H': return {17,17,17,31,17,17,17};
    case 'I': return {31,4,4,4,4,4,31};
    case 'K': return {17,18,20,24,20,18,17};
    case 'L': return {16,16,16,16,16,16,31};
    case 'M': return {17,27,21,21,17,17,17};
    case 'N': return {17,25,21,19,17,17,17};
    case 'O': return {14,17,17,17,17,17,14};
    case 'P': return {30,17,17,30,16,16,16};
    case 'Q': return {14,17,17,17,21,18,13};
    case 'R': return {30,17,17,30,20,18,17};
    case 'S': return {15,16,16,14,1,1,30};
    case 'T': return {31,4,4,4,4,4,4};
    case 'U': return {17,17,17,17,17,17,14};
    case 'V': return {17,17,17,17,17,10,4};
    case 'W': return {17,17,17,21,21,21,10};
    case 'X': return {17,17,10,4,10,17,17};
    case 'Y': return {17,17,10,4,4,4,4};
    case 'Z': return {31,1,2,4,8,16,31};
    case '0': return {14,17,19,21,25,17,14};
    case '1': return {4,12,4,4,4,4,14};
    case '2': return {14,17,1,2,4,8,31};
    case '3': return {30,1,1,14,1,1,30};
    case '4': return {2,6,10,18,31,2,2};
    case '5': return {31,16,16,30,1,1,30};
    case '6': return {14,16,16,30,17,17,14};
    case '7': return {31,1,2,4,8,8,8};
    case '8': return {14,17,17,14,17,17,14};
    case '9': return {14,17,17,15,1,1,14};
    case '.': return {0,0,0,0,0,6,6};
    case ',': return {0,0,0,0,0,6,4};
    case '%': return {17,2,4,4,8,16,17};
    case '[': return {14,8,8,8,8,8,14};
    case ']': return {14,2,2,2,2,2,14};
    default: return {};
    }
}

void rectangle(float x, float y, float w, float h, int width, int height,
               float red, float green, float blue, float alpha)
{
    const auto cx = [width](float p) { return 2.0F * p / width - 1.0F; };
    const auto cy = [height](float p) { return 1.0F - 2.0F * p / height; };
    glColor4f(red, green, blue, alpha);
    glBegin(GL_QUADS);
    glVertex2f(cx(x), cy(y)); glVertex2f(cx(x + w), cy(y));
    glVertex2f(cx(x + w), cy(y + h)); glVertex2f(cx(x), cy(y + h));
    glEnd();
}

void text(std::string_view value, float x, float y, float scale, int width, int height)
{
    for (char c : value) {
        const auto rows = glyph(c);
        for (std::size_t row = 0; row < rows.size(); ++row) {
            for (int column = 0; column < 5; ++column) {
                if ((rows[row] & (1U << (4 - column))) != 0U) {
                    rectangle(x + column * scale, y + static_cast<float>(row) * scale,
                        scale, scale, width, height, 0.93F, 0.96F, 1.0F, 1.0F);
                }
            }
        }
        x += 6.0F * scale;
    }
}

std::string uppercase(std::string_view value)
{
    std::string result(value);
    std::transform(result.begin(),result.end(),result.begin(),[](unsigned char c) {
        return c>='a' && c<='z' ? static_cast<char>(c-'a'+'A') : static_cast<char>(c);
    });
    return result;
}

void draw_context_browser(int width,int height,const Interaction& input)
{
    if (!input.show_context_browser) return;
    constexpr float panel_width=500.0F;
    constexpr float row_height=23.0F;
    const float panel_height=92.0F+row_height*
        static_cast<float>(parallel_mater::sim::simulation_recipes.size());
    const float x=0.5F*(static_cast<float>(width)-panel_width);
    const float y=std::max(12.0F,0.5F*(static_cast<float>(height)-panel_height));
    glDisable(GL_TEXTURE_2D);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
    rectangle(x,y,panel_width,panel_height,width,height,0.01F,0.02F,0.04F,0.94F);
    text("CONTEXTS",x+14.0F,y+12.0F,2.0F,width,height);
    text("UP DOWN SELECT  ENTER OPEN  TAB CLOSE",x+14.0F,y+36.0F,1.25F,width,height);
    const struct Key { const char* label; float r,g,b; } keys[]{
        {"FLUID",0.05F,0.52F,1.0F},{"CLOTH",0.12F,0.88F,0.32F},
        {"SOFTBODY",0.68F,0.22F,0.95F},{"ROPE",1.0F,0.48F,0.08F},
        {"SMOKE",0.92F,0.94F,0.98F}};
    float key_x=x+14.0F;
    for (const auto& key:keys) {
        rectangle(key_x,y+57.0F,10.0F,10.0F,width,height,key.r,key.g,key.b,1.0F);
        text(key.label,key_x+15.0F,y+55.0F,1.0F,width,height);
        key_x+=static_cast<float>(std::strlen(key.label))*6.0F+30.0F;
    }
    float row_y=y+82.0F;
    for (std::size_t index=0;index<parallel_mater::sim::simulation_recipes.size();++index) {
        const auto& context=parallel_mater::sim::simulation_recipes[index];
        if (static_cast<int>(index)==input.context_browser_index)
            rectangle(x+8.0F,row_y-4.0F,panel_width-16.0F,row_height-1.0F,
                width,height,0.08F,0.22F,0.38F,0.95F);
        const std::string title=uppercase(context.title);
        text(title,x+16.0F,row_y,1.5F,width,height);
        float chip_x=x+365.0F;
        const auto chip=[&](parallel_mater::sim::Component component,
                            float r,float g,float b) {
            if (!parallel_mater::sim::has_component(context.components,component)) return;
            rectangle(chip_x,row_y-2.0F,18.0F,12.0F,width,height,r,g,b,0.95F);
            chip_x+=22.0F;
        };
        chip(parallel_mater::sim::Component::fluid_particles,0.05F,0.52F,1.0F);
        chip(parallel_mater::sim::Component::cloth,0.12F,0.88F,0.32F);
        chip(parallel_mater::sim::Component::soft_body,0.68F,0.22F,0.95F);
        chip(parallel_mater::sim::Component::rope,1.0F,0.48F,0.08F);
        chip(parallel_mater::sim::Component::smoke,0.92F,0.94F,0.98F);
        row_y+=row_height;
    }
    glDisable(GL_BLEND);
    glEnable(GL_TEXTURE_2D);
    glColor4f(1,1,1,1);
}

struct SkinDebugData {
    std::vector<float3> positions;
    std::vector<float3> normals;
    std::vector<float3> box_forces;
    std::vector<float3> particle_forces;
    std::vector<float3> spring_forces;
};

struct WireframeDebugData {
    std::vector<float3> fluid_positions;
    std::vector<float3> water_positions;
    std::vector<uint3> water_triangles;
    std::vector<float3> soft_body_positions;
    std::vector<uint3> soft_body_triangles;
    std::vector<std::uint8_t> soft_body_triangle_active;
    std::vector<float3> voxel_positions;
    std::vector<std::uint32_t> voxel_flags;
    std::vector<waterlab::SoftBodyEdge> voxel_edges;
    std::vector<std::uint8_t> voxel_edge_active;
    std::uint32_t voxels_per_instance{};
    float voxel_radius{};
};

struct SmokeDebugData {
    std::vector<float3> positions;
    std::vector<float> temperatures;
};

template <typename T>
void copy_device_vector(
    std::vector<T>& destination,
    const T* source,
    std::uint32_t count,
    const char* operation)
{
    destination.resize(count);
    if (count == 0U) return;
    const cudaError_t status = cudaMemcpy(
        destination.data(), source, static_cast<std::size_t>(count) * sizeof(T),
        cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

void download_skin_debug(
    const Interaction& input,
    const waterlab::HybridDroplet& droplet,
    SkinDebugData& data)
{
    if (!input.show_normals && !input.show_box_forces &&
        !input.show_particle_forces && !input.show_spring_forces) return;
    const std::uint32_t count = droplet.physical_skin_vertex_count();
    copy_device_vector(data.positions, droplet.physical_skin_positions(), count,
        "download debug skin positions");
    if (input.show_normals) {
        copy_device_vector(data.normals, droplet.physical_skin_normals(), count,
            "download debug skin normals");
    }
    if (input.show_box_forces) {
        copy_device_vector(data.box_forces, droplet.skin_box_forces(), count,
            "download box-to-skin forces");
    }
    if (input.show_particle_forces) {
        copy_device_vector(data.particle_forces, droplet.skin_particle_forces(), count,
            "download particle-to-skin forces");
    }
    if (input.show_spring_forces) {
        copy_device_vector(data.spring_forces, droplet.skin_spring_forces(), count,
            "download skin spring forces");
    }
}

void download_wireframe_debug(const Interaction& input,
    const waterlab::HybridDroplet& droplet,
    const waterlab::SoftBodyCourse* soft_bodies,
    WireframeDebugData& data)
{
    const bool wireframe =
        input.fluid_display == waterlab::FluidDisplay::Wireframe;
    const bool particle_points = recipe_has_particles(input.context) &&
        (wireframe || input.fluid_display == waterlab::FluidDisplay::Particles ||
            input.fluid_display == waterlab::FluidDisplay::Billboards);
    if (!wireframe && !particle_points) return;
    if (particle_points) {
        copy_device_vector(data.fluid_positions, droplet.particle_positions(),
            droplet.statistics().particle_count, "download diagnostic fluid particles");
    } else {
        data.fluid_positions.clear();
    }
    if (wireframe && waterlab::gallery::recipe_has(
            input.context, parallel_mater::sim::Component::water_skin)) {
        const parallel_mater::DeviceMeshView water = droplet.skin_mesh();
        copy_device_vector(data.water_positions, water.positions,
            static_cast<std::uint32_t>(water.vertex_count), "download wire water vertices");
        if (data.water_triangles.size() != water.triangle_count) {
            copy_device_vector(data.water_triangles, water.triangles,
                static_cast<std::uint32_t>(water.triangle_count),
                "download wire water triangles");
        }
    } else {
        data.water_positions.clear();
        data.water_triangles.clear();
    }
    if (soft_bodies == nullptr) {
        data.soft_body_positions.clear();
        data.soft_body_triangles.clear();
        data.soft_body_triangle_active.clear();
        data.voxel_positions.clear();
        data.voxel_flags.clear();
        data.voxel_edges.clear();
        data.voxel_edge_active.clear();
        data.voxels_per_instance = 0U;
        data.voxel_radius = 0.0F;
        return;
    }
    const waterlab::SoftBodyRenderView soft = soft_bodies->render_view();
    if (wireframe) {
        copy_device_vector(data.soft_body_positions, soft.positions, soft.vertex_count,
            "download wire soft-body vertices");
        if (data.soft_body_triangles.size() != soft.triangle_count) {
            copy_device_vector(data.soft_body_triangles, soft.triangles, soft.triangle_count,
                "download wire soft-body triangles");
        }
        copy_device_vector(data.soft_body_triangle_active, soft.triangle_active,
            soft.triangle_count, "download wire soft-body triangle activity");
    } else {
        data.soft_body_positions.clear();
        data.soft_body_triangles.clear();
        data.soft_body_triangle_active.clear();
    }
    const auto lattice = soft_bodies->lattice_view();
    copy_device_vector(data.voxel_positions, lattice.positions, lattice.voxel_count,
        "download soft-body lattice positions");
    copy_device_vector(data.voxel_flags, lattice.flags, lattice.voxel_count,
        "download soft-body lattice flags");
    copy_device_vector(data.voxel_edges, lattice.edges, lattice.edges_per_instance,
        "download soft-body lattice bonds");
    copy_device_vector(data.voxel_edge_active, lattice.active_edges,
        lattice.edges_per_instance * lattice.instance_count,
        "download soft-body lattice bond activity");
    data.voxels_per_instance = lattice.voxels_per_instance;
    data.voxel_radius = lattice.voxel_radius;
}

void download_smoke_debug(const parallel_mater::physics::Smoke* smoke,
    SmokeDebugData& data)
{
    if (smoke==nullptr || !smoke->initialized()) {
        data.positions.clear();
        data.temperatures.clear();
        return;
    }
    const auto view=smoke->particles();
    copy_device_vector(data.positions,view.positions,view.count,
        "download smoke positions");
    copy_device_vector(data.temperatures,view.temperatures,view.count,
        "download smoke temperatures");
}

struct ScreenPoint {
    float x{};
    float y{};
    float depth{};
    bool visible{};
};

ScreenPoint project_point(
    float3 point,
    const waterlab::Camera& camera,
    int width,
    int height)
{
    const float3 forward = normalize(subtract(camera.target, camera.eye));
    const float3 right = normalize(cross(forward, camera.up));
    const float3 up = normalize(cross(right, forward));
    const float3 relative = subtract(point, camera.eye);
    const float depth = dot(relative, forward);
    if (depth <= 1.0e-4F) return {};
    constexpr float pi = 3.14159265358979323846F;
    const float tangent = std::tan(camera.vertical_fov_degrees * pi / 360.0F);
    const float aspect = static_cast<float>(width) / static_cast<float>(height);
    const float ndc_x = dot(relative, right) / (depth * tangent * aspect);
    const float ndc_y = dot(relative, up) / (depth * tangent);
    return {
        0.5F * static_cast<float>(width) * (ndc_x + 1.0F),
        0.5F * static_cast<float>(height) * (1.0F - ndc_y),
        depth,
        std::abs(ndc_x) <= 1.1F && std::abs(ndc_y) <= 1.1F};
}

void draw_smoke_particles(const SmokeDebugData& data,
    parallel_mater::sim::SimulationRecipe context,const waterlab::Camera& camera,
    int width,int height)
{
    if (data.positions.empty()) return;
    struct Projected { ScreenPoint point; float temperature; };
    std::vector<Projected> points;
    points.reserve(data.positions.size());
    for (std::size_t index=0;index<data.positions.size();++index) {
        const ScreenPoint point=project_point(data.positions[index],camera,width,height);
        if (point.visible) points.push_back({point,index<data.temperatures.size()
            ? data.temperatures[index] : 0.0F});
    }
    std::stable_sort(points.begin(),points.end(),[](const auto& a,const auto& b) {
        return a.point.depth>b.point.depth;
    });
    const bool steam=context==parallel_mater::sim::SimulationRecipe::fluid_smoke;
    glDisable(GL_TEXTURE_2D);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_POINT_SMOOTH);
    glPointSize(steam ? 14.0F : 16.0F);
    glBegin(GL_POINTS);
    for (const auto& particle:points) {
        const float fade=std::clamp(particle.temperature,0.0F,1.0F);
        const float grey=steam ? 0.88F+0.10F*fade : 0.22F+0.38F*fade;
        glColor4f(grey,grey,steam ? std::min(1.0F,grey+0.05F) : grey,
            0.025F+0.055F*fade);
        glVertex2f(2.0F*particle.point.x/static_cast<float>(width)-1.0F,
            1.0F-2.0F*particle.point.y/static_cast<float>(height));
    }
    glEnd();
    glPointSize(steam ? 6.0F : 7.0F);
    glBegin(GL_POINTS);
    for (const auto& particle:points) {
        const float fade=std::clamp(particle.temperature,0.0F,1.0F);
        const float grey=steam ? 0.96F : 0.42F+0.42F*fade;
        glColor4f(grey,grey,steam ? 1.0F : grey,0.10F+0.22F*fade);
        glVertex2f(2.0F*particle.point.x/static_cast<float>(width)-1.0F,
            1.0F-2.0F*particle.point.y/static_cast<float>(height));
    }
    glEnd();
    glPointSize(1.0F);
    glDisable(GL_POINT_SMOOTH);
    glDisable(GL_BLEND);
    glEnable(GL_TEXTURE_2D);
    glColor4f(1,1,1,1);
}

void emit_screen_line(ScreenPoint a, ScreenPoint b, int width, int height)
{
    glVertex2f(2.0F * a.x / static_cast<float>(width) - 1.0F,
        1.0F - 2.0F * a.y / static_cast<float>(height));
    glVertex2f(2.0F * b.x / static_cast<float>(width) - 1.0F,
        1.0F - 2.0F * b.y / static_cast<float>(height));
}

void draw_arrow_set(
    const std::vector<float3>& positions,
    const std::vector<float3>& vectors,
    const waterlab::Camera& camera,
    int width,
    int height,
    float fixed_length,
    float red,
    float green,
    float blue)
{
    if (positions.size() != vectors.size()) return;
    glColor4f(red, green, blue, 0.95F);
    glBegin(GL_LINES);
    for (std::size_t index = 0; index < positions.size(); ++index) {
        const float magnitude = length(vectors[index]);
        if (!(magnitude > 1.0e-5F) || !std::isfinite(magnitude)) continue;
        const float arrow_length = fixed_length > 0.0F
            ? fixed_length
            : std::clamp(0.003F * magnitude, 0.012F, 0.14F);
        const float3 endpoint = add(
            positions[index], multiply(vectors[index], arrow_length / magnitude));
        const ScreenPoint start = project_point(positions[index], camera, width, height);
        const ScreenPoint end = project_point(endpoint, camera, width, height);
        if (!start.visible || !end.visible) continue;
        const float dx = end.x - start.x;
        const float dy = end.y - start.y;
        const float screen_length = std::sqrt(dx * dx + dy * dy);
        if (!(screen_length > 0.5F)) continue;
        const float ux = dx / screen_length;
        const float uy = dy / screen_length;
        const float head_length = std::min(6.0F, 0.45F * screen_length);
        const float head_width = 0.55F * head_length;
        const ScreenPoint left{
            end.x - ux * head_length - uy * head_width,
            end.y - uy * head_length + ux * head_width, true};
        const ScreenPoint right{
            end.x - ux * head_length + uy * head_width,
            end.y - uy * head_length - ux * head_width, true};
        emit_screen_line(start, end, width, height);
        emit_screen_line(end, left, width, height);
        emit_screen_line(end, right, width, height);
    }
    glEnd();
}

void draw_skin_debug(
    const Interaction& input,
    const SkinDebugData& data,
    const waterlab::Camera& camera,
    int width,
    int height)
{
    if (!input.show_normals && !input.show_box_forces &&
        !input.show_particle_forces && !input.show_spring_forces) return;
    glDisable(GL_TEXTURE_2D);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glLineWidth(1.5F);
    if (input.show_normals) {
        draw_arrow_set(data.positions, data.normals, camera, width, height,
            0.065F, 0.12F, 1.0F, 0.22F);
    }
    if (input.show_box_forces) {
        draw_arrow_set(data.positions, data.box_forces, camera, width, height,
            0.0F, 0.12F, 0.48F, 1.0F);
    }
    if (input.show_particle_forces) {
        draw_arrow_set(data.positions, data.particle_forces, camera, width, height,
            0.0F, 1.0F, 0.86F, 0.08F);
    }
    if (input.show_spring_forces) {
        draw_arrow_set(data.positions, data.spring_forces, camera, width, height,
            0.0F, 0.72F, 0.20F, 1.0F);
    }
    glLineWidth(1.0F);
    glDisable(GL_BLEND);
    glEnable(GL_TEXTURE_2D);
    glColor4f(1,1,1,1);
}

void draw_wire_mesh(const std::vector<float3>& positions,
    const std::vector<uint3>& triangles, const std::vector<std::uint8_t>* active,
    const waterlab::Camera& camera,
    int width, int height, float red, float green, float blue, float alpha = 0.80F)
{
    glBegin(GL_LINES);
    for (std::size_t triangle_index = 0U;
         triangle_index < triangles.size(); ++triangle_index) {
        // Draw only material that is still connected. The capture audit keeps
        // measuring detached bindings, but they must not tether pieces visually.
        if (active != nullptr &&
            (triangle_index >= active->size() || (*active)[triangle_index] == 0U)) continue;
        glColor4f(red, green, blue, alpha);
        const uint3 triangle = triangles[triangle_index];
        if (triangle.x >= positions.size() || triangle.y >= positions.size() ||
            triangle.z >= positions.size()) continue;
        const float3 points[3]{positions[triangle.x], positions[triangle.y], positions[triangle.z]};
        const float3 normal = cross(subtract(points[1], points[0]), subtract(points[2], points[0]));
        // This is a topology diagnostic, not an opaque surface pass. Drawing
        // only camera-facing triangles made a fractured or locally folded
        // column appear to lose most of its mesh precisely when both sides
        // were useful for diagnosis.
        if (!(length(normal) > 1.0e-10F)) continue;
        const ScreenPoint screen[3]{project_point(points[0], camera, width, height),
            project_point(points[1], camera, width, height),
            project_point(points[2], camera, width, height)};
        for (unsigned edge = 0U; edge < 3U; ++edge) {
            const ScreenPoint a = screen[edge];
            const ScreenPoint b = screen[(edge + 1U) % 3U];
            if (a.visible && b.visible) emit_screen_line(a, b, width, height);
        }
    }
    glEnd();
}

void draw_soft_body_vertices(const std::vector<float3>& positions,
    const waterlab::Camera& camera, int width, int height)
{
    glPointSize(3.0F);
    glColor4f(1.0F, 0.72F, 0.18F, 1.0F);
    glBegin(GL_POINTS);
    for (std::size_t vertex = 0U; vertex < positions.size(); ++vertex) {
        const float3 position = positions[vertex];
        const ScreenPoint point = project_point(position, camera, width, height);
        if (!point.visible) continue;
        glVertex2f(2.0F * point.x / static_cast<float>(width) - 1.0F,
            1.0F - 2.0F * point.y / static_cast<float>(height));
    }
    glEnd();
    glPointSize(1.0F);
}

void draw_soft_body_lattice(const WireframeDebugData& data,
    const waterlab::Camera& camera, int width, int height)
{
    if (data.voxel_edges.empty() || data.voxels_per_instance == 0U) return;
    std::vector<ScreenPoint> points;
    points.reserve(data.voxel_positions.size());
    for (const float3 position : data.voxel_positions) {
        points.push_back(project_point(position, camera, width, height));
    }
    // Render every live structural member as a thin world-space box. Its
    // square cross-section follows camera depth and remains readable as an
    // actual beam rather than an abstract line or a round wire.
    glBegin(GL_QUADS);
    constexpr float ring_cos[4]{1.0F, -1.0F, -1.0F, 1.0F};
    constexpr float ring_sin[4]{1.0F, 1.0F, -1.0F, -1.0F};
    const float3 light = normalize(make_float3(-0.3F, 0.8F, 0.5F));
    const float radius = 0.18F * data.voxel_radius;
    const auto emit = [&](const ScreenPoint& point) {
        glVertex2f(2.0F * point.x / static_cast<float>(width) - 1.0F,
            1.0F - 2.0F * point.y / static_cast<float>(height));
    };
    for (std::size_t bond = 0U; bond < data.voxel_edge_active.size(); ++bond) {
        if (data.voxel_edge_active[bond] == 0U) continue;
        const auto edge = data.voxel_edges[bond % data.voxel_edges.size()].vertices;
        const std::size_t base = (bond / data.voxel_edges.size()) * data.voxels_per_instance;
        if (base + edge.x >= points.size() || base + edge.y >= points.size()) continue;
        const bool internal_member = base + edge.x < data.voxel_flags.size() &&
            base + edge.y < data.voxel_flags.size() &&
            (((data.voxel_flags[base + edge.x] & waterlab::soft_body_voxel_surface) == 0U) ||
             ((data.voxel_flags[base + edge.y] & waterlab::soft_body_voxel_surface) == 0U));
        const ScreenPoint a = points[base + edge.x];
        const ScreenPoint b = points[base + edge.y];
        if (!a.visible || !b.visible) continue;
        const float3 start = data.voxel_positions[base + edge.x];
        const float3 end = data.voxel_positions[base + edge.y];
        const float3 axis = normalize(subtract(end, start));
        const float3 view = normalize(subtract(camera.eye,
            multiply(add(start, end), 0.5F)));
        float3 radial = cross(axis, view);
        if (length(radial) < 1.0e-5F)
            radial = cross(axis, make_float3(0.0F, 1.0F, 0.0F));
        if (length(radial) < 1.0e-5F) continue;
        radial = normalize(radial);
        const float3 tangent = normalize(cross(axis, radial));
        for (int side = 0; side < 4; ++side) {
            const int next = (side + 1) % 4;
            const float3 offset_a = multiply(add(
                multiply(radial, ring_cos[side]),
                multiply(tangent, ring_sin[side])), radius);
            const float3 offset_b = multiply(add(
                multiply(radial, ring_cos[next]),
                multiply(tangent, ring_sin[next])), radius);
            const ScreenPoint p0 = project_point(add(start, offset_a), camera, width, height);
            const ScreenPoint p1 = project_point(add(end, offset_a), camera, width, height);
            const ScreenPoint p2 = project_point(add(end, offset_b), camera, width, height);
            const ScreenPoint p3 = project_point(add(start, offset_b), camera, width, height);
            if (!p0.visible || !p1.visible || !p2.visible || !p3.visible) continue;
            const float3 side_normal = normalize(add(offset_a, offset_b));
            const float shade = std::clamp(0.45F + 0.55F * dot(side_normal, light),
                0.22F, 1.0F);
            if (internal_member)
                glColor4f(0.16F * shade, 1.0F * shade, 0.72F * shade, 0.96F);
            else
                glColor4f(0.92F * shade, 0.48F * shade, 0.16F * shade, 0.72F);
            emit(p0); emit(p1); emit(p2); emit(p3);
        }
    }
    glEnd();
    glPointSize(3.0F);
    glBegin(GL_POINTS);
    for (std::size_t voxel = 0U; voxel < points.size(); ++voxel) {
        const auto point = points[voxel];
        if (!point.visible || voxel >= data.voxel_flags.size()) continue;
        const auto flags = data.voxel_flags[voxel];
        if ((flags & waterlab::soft_body_voxel_pinned) != 0U)
            glColor4f(0.25F, 0.55F, 1.0F, 0.95F);
        else if ((flags & waterlab::soft_body_voxel_surface) != 0U)
            glColor4f(1.0F, 0.72F, 0.18F, 0.80F);
        else
            glColor4f(0.18F, 0.92F, 0.56F, 0.95F);
        glVertex2f(2.0F * point.x / static_cast<float>(width) - 1.0F,
            1.0F - 2.0F * point.y / static_cast<float>(height));
    }
    glEnd();
    glPointSize(1.0F);
}

void draw_wireframe_debug(const Interaction& input, const WireframeDebugData& data,
    const waterlab::Camera& camera, int width, int height)
{
    const bool wireframe =
        input.fluid_display == waterlab::FluidDisplay::Wireframe;
    const bool particle_points = recipe_has_particles(input.context) &&
        (wireframe || input.fluid_display == waterlab::FluidDisplay::Particles ||
            input.fluid_display == waterlab::FluidDisplay::Billboards);
    if (!wireframe && !particle_points) return;
    glDisable(GL_TEXTURE_2D);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glLineWidth(1.0F);
    if (particle_points) {
        // Preserve the inexpensive diagnostic path, but draw actual round
        // water beads instead of driver-dependent square one-pixel dots.
        glEnable(GL_POINT_SMOOTH);
        glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);
        const bool billboard =
            input.fluid_display == waterlab::FluidDisplay::Billboards;
        std::vector<ScreenPoint> points;
        points.reserve(data.fluid_positions.size());
        for (const float3 position : data.fluid_positions) {
            const ScreenPoint point=project_point(position,camera,width,height);
            if (point.visible) points.push_back(point);
        }
        if (billboard) {
            std::stable_sort(points.begin(), points.end(),
                [](const ScreenPoint& a,const ScreenPoint& b) {
                    return a.depth>b.depth;
                });
        }
        glPointSize(billboard ? 10.0F : 7.0F);
        glColor4f(0.01F,0.16F,0.30F,billboard ? 0.30F : 0.90F);
        glBegin(GL_POINTS);
        for (const ScreenPoint point : points) {
            glVertex2f(2.0F*point.x/static_cast<float>(width)-1.0F,
                1.0F-2.0F*point.y/static_cast<float>(height));
        }
        glEnd();
        glPointSize(billboard ? 7.0F : 4.5F);
        glColor4f(0.10F,0.72F,1.0F,billboard ? 0.42F : 0.96F);
        glBegin(GL_POINTS);
        for (const ScreenPoint point : points) {
            glVertex2f(2.0F*point.x/static_cast<float>(width)-1.0F,
                1.0F-2.0F*point.y/static_cast<float>(height));
        }
        glEnd();
        glPointSize(1.0F);
        glDisable(GL_POINT_SMOOTH);
    }
    if (wireframe) {
        draw_wire_mesh(data.water_positions, data.water_triangles, nullptr, camera,
            width, height, 0.10F, 0.90F, 1.0F);
        // In the particle/cloth diagnostic the dense spring lattice obscures
        // the particle and green-foam layers. Keep only a faint triangle wire
        // plus the node dots so both systems remain legible.
        if (input.context != parallel_mater::sim::SimulationRecipe::water_rope) {
            draw_wire_mesh(data.soft_body_positions, data.soft_body_triangles,
                &data.soft_body_triangle_active, camera,
                width, height, 1.0F, 0.48F, 0.10F);
        } else {
            draw_wire_mesh(data.soft_body_positions, data.soft_body_triangles,
                &data.soft_body_triangle_active, camera,
                width, height, 1.0F, 0.62F, 0.15F, 0.22F);
        }
    }
    // Draw the volume graph last so interior box members are not hidden by
    // the diagnostic surface wireframe selected with V.
    draw_soft_body_lattice(data, camera, width, height);
    if (wireframe)
        draw_soft_body_vertices(data.soft_body_positions, camera, width, height);
    glDisable(GL_BLEND);
    glEnable(GL_TEXTURE_2D);
    glColor4f(1,1,1,1);
}

void draw_timings(int width, int height, bool visible, const Interaction& input,
                  const waterlab::HybridTimings& t, float visual, float raytrace,
                  const parallel_mater::physics::SmokeTimings& smoke,
                  float debug_draw, float wall)
{
    if (!visible) return;
    const bool particles = recipe_has_particles(input.context);
    const bool water_skin = !input.course_mode || waterlab::gallery::recipe_has(
        input.context, parallel_mater::sim::Component::water_skin);
    const bool deformable = input.course_mode &&
        input.context != parallel_mater::sim::SimulationRecipe::water_cloth && (
        waterlab::gallery::recipe_has(input.context, parallel_mater::sim::Component::soft_body) ||
        waterlab::gallery::recipe_has(input.context, parallel_mater::sim::Component::cloth) ||
        waterlab::gallery::recipe_has(input.context, parallel_mater::sim::Component::rope));
    const bool cloth = input.course_mode && waterlab::gallery::recipe_has(
        input.context, parallel_mater::sim::Component::cloth);
    std::vector<std::pair<const char*, float>> rows;
    if (particles) {
        if (input.context == parallel_mater::sim::SimulationRecipe::water_soft_body)
            rows.emplace_back("PARTICLE RECYCLE", t.particle_recycling_ms);
        rows.emplace_back("FLUID HIERARCHY", t.rebuild_fluid_hierarchy_ms);
        rows.emplace_back(
            input.context == parallel_mater::sim::SimulationRecipe::water_soft_body
                ? "FLUID + LIFT" :
            (input.context == parallel_mater::sim::SimulationRecipe::water ||
             input.context == parallel_mater::sim::SimulationRecipe::water_rope)
                ? "FLUID + RIGID" : "FLUID PHYSICS",
            t.update_fluid_physics_ms);
    }
    if (water_skin) {
        rows.emplace_back("SKIN HIERARCHY", t.rebuild_skin_hierarchy_ms);
        rows.emplace_back("SKIN PHYSICS", t.update_skin_physics_ms);
        rows.emplace_back("NORMALS", t.update_surface_normals_ms);
        rows.emplace_back("SURFACE PREP", t.update_render_surface_ms);
    } else if (particles) {
        // HybridDroplet still runs its hidden adapter until the particle
        // broad phase becomes a stand-alone solver. Keep this cost visible.
        rows.emplace_back("ADAPTER SKIN", t.rebuild_skin_hierarchy_ms +
            t.update_skin_physics_ms + t.update_surface_normals_ms +
            t.update_render_surface_ms);
    }
    if (!input.course_mode) rows.emplace_back("RECT PHYSICS", t.update_rectangle_physics_ms);
    if (deformable) {
        rows.emplace_back(input.context ==
                parallel_mater::sim::SimulationRecipe::cloth_soft_body
                ? "SOFT + CLOTH" : (cloth ? "CLOTH PHYSICS" : "SOFT PHYSICS"),
            t.update_soft_body_physics_ms);
        rows.emplace_back("MESH HIERARCHY", t.rebuild_soft_body_hierarchy_ms);
        if (particles) rows.emplace_back("MESH CONTACT", t.update_soft_body_contact_ms);
        rows.emplace_back("MESH RENDER", t.update_soft_body_render_ms);
    }
    if (input.course_mode && (
        input.context == parallel_mater::sim::SimulationRecipe::cloth ||
        input.context == parallel_mater::sim::SimulationRecipe::soft_body ||
        input.context == parallel_mater::sim::SimulationRecipe::water ||
        input.context == parallel_mater::sim::SimulationRecipe::water_rope ||
        input.context == parallel_mater::sim::SimulationRecipe::water_soft_body ||
        input.context == parallel_mater::sim::SimulationRecipe::rope ||
        input.context == parallel_mater::sim::SimulationRecipe::cloth_rope ||
        input.context == parallel_mater::sim::SimulationRecipe::soft_body_rope)) {
        rows.emplace_back("RIGID CONTACT", t.update_rigid_body_contact_ms);
    }
    if (visual > 0.0F) rows.emplace_back("WATER FOAM", visual);
    if (recipe_has_smoke(input.context)) {
        rows.emplace_back("SMOKE ADVECT",smoke.integrate_ms);
        if (smoke.couple_ms>0.0F) rows.emplace_back("SMOKE COUPLE",smoke.couple_ms);
    }
    rows.emplace_back("RAYTRACE", raytrace);
    if (debug_draw > 0.0F) rows.emplace_back("DEBUG DRAW", debug_draw);
    rows.emplace_back("GPU TOTAL", t.gpu_total_ms() + visual + raytrace +
        smoke.gpu_total_ms());
    rows.emplace_back("FRAME WALL", wall);
    glDisable(GL_TEXTURE_2D);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    rectangle(12, 12, 342, 16.0F + 24.0F * static_cast<float>(rows.size()),
        width, height, 0.015F, 0.025F, 0.045F, 0.82F);
    char line[64];
    const auto row = [&](const char* label, float value, float y) {
        std::snprintf(line, sizeof(line), "%-17s %.3f MS", label, value);
        text(line, 22, y, 2, width, height);
    };
    float y = 24.0F;
    for (const auto& [label, milliseconds] : rows) {
        row(label, milliseconds, y);
        y += 24.0F;
    }
    glDisable(GL_BLEND);
    glEnable(GL_TEXTURE_2D);
    glColor4f(1,1,1,1);
}

void draw_course_hud(int width, int height, const Interaction& input,
    const waterlab::HybridOptions& options,
    const waterlab::HybridStatistics& physics_statistics,
    const waterlab::SoftBodyCourse* soft_bodies)
{
    if (!input.course_mode || input.replay_mode) return;
    (void)physics_statistics;
    (void)soft_bodies;
    const float x = 12.0F;
    const float y = std::max(350.0F, static_cast<float>(height) - 158.0F);
    glDisable(GL_TEXTURE_2D);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    rectangle(x, y, 560.0F, 146.0F, width, height,
        0.015F, 0.025F, 0.045F, 0.86F);
    char status[96];
    const auto& objective = waterlab::gallery_objective(input.context);
    const auto& recipe = waterlab::gallery::recipe_info(input.context);
    std::snprintf(status, sizeof(status), "%.*s%s",
        static_cast<int>(recipe.title.size()), recipe.title.data(),
        input.objective_progress.completed ? "  COMPLETE" : "");
    text(status,
        x + 10.0F, y + 10.0F, 1.5F, width, height);
    text(objective.description, x + 10.0F, y + 31.0F, 1.25F, width, height);
    rectangle(x + 10.0F, y + 51.0F, 530.0F, 8.0F, width, height,
        0.08F, 0.11F, 0.15F, 0.95F);
    rectangle(x + 10.0F, y + 51.0F,
        530.0F * input.objective_progress.normalized, 8.0F, width, height,
        0.10F, 0.78F, 0.90F, 0.95F);
    text("Z NORMAL  X OBSTACLE  C PARTICLE  V VIEW  K BILLBOARD  B SPRING",
        x + 10.0F, y + 68.0F, 1.5F, width, height);
    text("R RESET  T TIMING  P PHYSICS  L LOAD  M CAPTURE",
        x + 10.0F, y + 91.0F, 1.5F, width, height);
    if (input.context == parallel_mater::sim::SimulationRecipe::water_cloth) {
        std::snprintf(status, sizeof(status), "[ ] SPEED %.0FX   N %u/16",
            waterlab::course_motion_multiplier(options), options.physics_iterations);
    } else {
        std::snprintf(status, sizeof(status), "[ ] COURSE SPEED   N %u   PARTICLES %u",
            options.physics_iterations,
            recipe_has_particles(input.context) ? options.particle_count : 0U);
    }
    text(status, x + 10.0F, y + 114.0F, 1.5F, width, height);
    glDisable(GL_BLEND);
    glEnable(GL_TEXTURE_2D);
    glColor4f(1,1,1,1);
}

void apply_physics_adjustment(Interaction& input, waterlab::HybridDroplet& droplet,
    waterlab::SoftBodyCourse* soft_bodies,
    waterlab::RigidSphereState* rigid_sphere,
    waterlab::FluidVisuals* visuals)
{
    if (input.physics_adjustment == 0) return;
    waterlab::HybridOptions options = droplet.options();
    const float amount = static_cast<float>(input.physics_adjustment);
    if (input.physics_parameter >= 25 && input.physics_parameter <= 27 && visuals) {
        auto settings = visuals->foam_settings();
        switch (input.physics_parameter) {
        case 25: settings.emission_rate = std::clamp(
            settings.emission_rate + amount, 0.0F, 64.0F); break;
        case 26: settings.radius_scale = std::clamp(
            settings.radius_scale + 0.1F * amount, 0.25F, 4.0F); break;
        case 27: settings.lifetime_scale = std::clamp(
            settings.lifetime_scale + 0.1F * amount, 0.25F, 4.0F); break;
        }
        visuals->set_foam_settings(settings);
        input.physics_adjustment = 0;
        return;
    }
    if (input.physics_parameter == 15 && soft_bodies != nullptr) {
        soft_bodies->set_strength_multiplier(std::clamp(
            soft_bodies->strength_multiplier() * std::pow(1.1F, amount),
            0.0625F, 64.0F));
        input.physics_adjustment = 0;
        return;
    }
    if (((input.physics_parameter >= 19 && input.physics_parameter <= 22) ||
         input.physics_parameter == 24) &&
        soft_bodies != nullptr) {
        auto material = soft_bodies->material();
        switch (input.physics_parameter) {
        case 19: material.spring_stiffness = std::clamp(
            material.spring_stiffness + 2'000.0F * amount, 100.0F, 160'000.0F); break;
        case 20: material.spring_damping_ratio = std::clamp(
            material.spring_damping_ratio + 0.1F * amount, 0.0F, 4.0F); break;
        case 21: material.velocity_damping = std::clamp(
            material.velocity_damping + 0.1F * amount, 0.0F, 30.0F); break;
        case 22: material.maximum_speed = std::clamp(
            material.maximum_speed + 0.5F * amount, 0.5F, 30.0F); break;
        case 24: material.ground_friction = std::clamp(
            material.ground_friction + 0.5F * amount, 0.0F, 50.0F); break;
        }
        soft_bodies->set_material(material);
        input.physics_adjustment = 0;
        return;
    }
    if (input.physics_parameter == 23 && rigid_sphere != nullptr) {
        rigid_sphere->mass = std::clamp(
            rigid_sphere->mass * std::pow(1.25F, amount), 1.0F, 20'000.0F);
        input.physics_adjustment = 0;
        return;
    }
    if (input.physics_parameter == 28 && soft_bodies != nullptr) {
        if (input.context==parallel_mater::sim::SimulationRecipe::cloth_soft_body)
            soft_bodies->set_primary_body_mass(std::clamp(
                soft_bodies->primary_body_mass()*std::pow(1.25F,amount),
                soft_bodies->voxel_mass(),20'000.0F));
        else
            soft_bodies->set_voxel_mass(std::clamp(
                soft_bodies->voxel_mass() * std::pow(1.25F, amount),
                0.001F, 100.0F));
        input.physics_adjustment = 0;
        return;
    }
    if (input.physics_parameter == 16 && input.course_mode) {
        droplet.resize_particles(static_cast<std::uint32_t>(std::clamp(
            static_cast<int>(options.particle_count) + 250 * input.physics_adjustment,
            256, static_cast<int>(options.particle_capacity))));
        input.physics_adjustment = 0;
        return;
    }
    if (input.physics_parameter == 17 && input.course_mode) return;
    switch (input.physics_parameter) {
    case 0:
        options.physics_iterations = static_cast<std::uint32_t>(std::clamp(
            static_cast<int>(options.physics_iterations) + input.physics_adjustment,
            options.obstacle_course ? static_cast<int>(waterlab::course_motion_iterations(options)) : 1,
            static_cast<int>(waterlab::HybridDroplet::maximum_physics_iterations)));
        break;
    case 1: options.particle_repulsion = std::clamp(
        options.particle_repulsion + 1.0F * amount, 0.0F, 120.0F); break;
    case 2: options.particle_damping = std::clamp(options.particle_damping + 0.25F * amount, 0.0F, 30.0F); break;
    case 3: options.particle_velocity_damping = std::clamp(options.particle_velocity_damping + 0.1F * amount, 0.0F, 20.0F); break;
    case 4: options.particle_skin_stiffness = std::clamp(
        options.particle_skin_stiffness + 250.0F * amount, 20.0F,
        options.obstacle_course ? 64'000.0F : 2'000.0F); break;
    case 5: options.particle_skin_damping = std::clamp(options.particle_skin_damping + amount, 0.0F, 50.0F); break;
    case 6: options.skin_spring_stiffness = std::clamp(options.skin_spring_stiffness + 10.0F * amount, 10.0F, 2'000.0F); break;
    case 7: options.skin_spring_damping = std::clamp(options.skin_spring_damping + 0.5F * amount, 0.0F, 50.0F); break;
    case 8: options.skin_velocity_damping = std::clamp(options.skin_velocity_damping + 0.2F * amount, 0.0F, 30.0F); break;
    case 9: options.box_skin_stiffness = std::clamp(options.box_skin_stiffness + 250.0F * amount, 250.0F, 20'000.0F); break;
    case 10: options.box_skin_damping = std::clamp(
        options.box_skin_damping + amount, 0.0F, 100.0F); break;
    case 11: options.maximum_box_ejection_force = std::clamp(options.maximum_box_ejection_force + 10.0F * amount, 10.0F, 1'000.0F); break;
    case 12: options.maximum_skin_force = std::clamp(options.maximum_skin_force + 10.0F * amount, 10.0F, 1'000.0F); break;
    case 13: options.rectangle_target_stiffness = std::clamp(options.rectangle_target_stiffness + 100.0F * amount, 100.0F, 10'000.0F); break;
    case 14: options.rectangle_target_damping_ratio = std::clamp(options.rectangle_target_damping_ratio + 0.1F * amount, 0.0F, 4.0F); break;
    case 18: {
        const float magnitude = std::clamp(length(options.gravity) + 0.5F * amount,
            0.0F, 20.0F);
        const float3 direction = length(options.gravity) > 1.0e-8F
            ? normalize(options.gravity) : make_float3(0.0F, -1.0F, 0.0F);
        input.course_gravity = multiply(direction, magnitude);
        if (input.context != parallel_mater::sim::SimulationRecipe::water_soft_body)
            options.gravity = input.course_gravity;
        break;
    }
    default: break;
    }
    input.physics_adjustment = 0;
    if (input.physics_parameter == 0 && soft_bodies != nullptr)
        soft_bodies->set_solver_substeps(options.physics_iterations);
    droplet.set_runtime_options(options);
}

void draw_physics_panel(
    int width,
    int height,
    bool visible,
    const Interaction& input,
    const waterlab::HybridOptions& options,
    const waterlab::SoftBodyCourse* soft_bodies,
    const waterlab::RigidSphereState* rigid_sphere,
    const waterlab::FluidVisuals* visuals)
{
    if (!visible) return;
    constexpr const char* labels[physics_parameter_count]{
        "ITERATIONS", "PARTICLE REPEL", "PARTICLE DAMP", "PARTICLE DRAG",
        "BOUNDARY FORCE", "BOUNDARY DAMP", "SKIN SPRING", "SKIN DAMP", "SKIN DRAG",
        "BOX FORCE", "BOX DAMP", "BOX FORCE CAP", "SKIN FORCE CAP",
        "RECT TARGET", "RECT DAMP", "SOFT BOND", "PARTICLE COUNT",
        "SKIN DETAIL", "GRAVITY", "SOFT SPRING", "SOFT DAMP",
        "SOFT DRAG", "SOFT SPEED", "RIGID MASS", "GROUND FRICTION",
        "FOAM RATE", "FOAM SIZE", "FOAM LIFE", "SOFT BODY MASS",
        "BRIDGE COLUMNS", "BRIDGE ROWS"};
    const auto soft_material = soft_bodies
        ? soft_bodies->material() : waterlab::SoftBodyMaterial{};
    const auto foam = visuals ? visuals->foam_settings() : waterlab::FoamSettings{};
    const float values[physics_parameter_count]{
        static_cast<float>(options.physics_iterations), options.particle_repulsion,
        options.particle_damping, options.particle_velocity_damping,
        options.particle_skin_stiffness, options.particle_skin_damping,
        options.skin_spring_stiffness,
        options.skin_spring_damping, options.skin_velocity_damping,
        options.box_skin_stiffness, options.box_skin_damping,
        options.maximum_box_ejection_force, options.maximum_skin_force,
        options.rectangle_target_stiffness, options.rectangle_target_damping_ratio,
        soft_bodies ? soft_bodies->strength_multiplier() : 0.0F,
        static_cast<float>(options.particle_count),
        static_cast<float>(options.physical_skin_frequency), length(options.gravity),
        soft_material.spring_stiffness, soft_material.spring_damping_ratio,
        soft_material.velocity_damping, soft_material.maximum_speed,
        rigid_sphere ? rigid_sphere->mass : 0.0F,
        soft_material.ground_friction, foam.emission_rate,
        foam.radius_scale, foam.lifetime_scale,
        soft_bodies ? (input.context==
            parallel_mater::sim::SimulationRecipe::cloth_soft_body
                ? soft_bodies->primary_body_mass()
                : soft_bodies->voxel_mass()) : 0.0F,
        static_cast<float>(input.bridge_columns),
        static_cast<float>(input.bridge_rows)};
    const float panel_width = 374.0F;
    const float panel_x = std::max(12.0F, static_cast<float>(width) - panel_width - 12.0F);
    const std::vector<int> rows = relevant_physics_parameters(input);
    const int visible_rows = static_cast<int>(rows.size());
    const float panel_height = 44.0F + visible_rows * 22.0F;
    glDisable(GL_TEXTURE_2D);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    rectangle(panel_x, 12, panel_width, panel_height, width, height,
        0.015F, 0.025F, 0.045F, 0.88F);
    text("PHYSICS  ARROWS CHANGE", panel_x + 10.0F, 22.0F, 2.0F, width, height);
    char line[64];
    int display_row = 0;
    for (const int parameter : rows) {
        const float y = 48.0F + static_cast<float>(display_row++) * 22.0F;
        if (parameter == input.physics_parameter) {
            rectangle(panel_x + 6.0F, y - 4.0F, panel_width - 12.0F, 20.0F,
                width, height, 0.08F, 0.25F, 0.42F, 0.90F);
        }
        if (parameter == 0 || parameter == 16 || parameter == 17) {
            std::snprintf(line, sizeof(line), "%-18s %u", labels[parameter],
                static_cast<unsigned>(std::lround(values[parameter])));
        } else {
            std::snprintf(line, sizeof(line), "%-18s %.2f",
                labels[parameter], values[parameter]);
        }
        text(line, panel_x + 12.0F, y, 2.0F, width, height);
    }
    glDisable(GL_BLEND);
    glEnable(GL_TEXTURE_2D);
    glColor4f(1,1,1,1);
}

void draw_quantity_panel(int width, int height, bool visible,
    const Interaction& input, const waterlab::HybridOptions& options,
    const waterlab::SoftBodyCourse* soft_bodies)
{
    if (!visible) return;
    constexpr const char* labels[8]{
        "FLUID PARTICLES", "PHYS SKIN FREQ", "RENDER SKIN FREQ", "SOFT SOLVES",
        "ROPE NODES", "CLOTH DETAIL", "CYLINDER COLUMNS", "CYLINDER ROWS"};
    const float values[8]{static_cast<float>(input.context==
            parallel_mater::sim::SimulationRecipe::water_soft_body &&
            input.staged_particle_target!=0U
        ? input.staged_particle_target : options.particle_count),
        static_cast<float>(options.physical_skin_frequency),
        static_cast<float>(options.render_skin_frequency),
        static_cast<float>(soft_bodies ? soft_bodies->spring_solver_iterations() : 0U),
        static_cast<float>(input.rope_node_count),
        static_cast<float>(input.cloth_detail),
        static_cast<float>(input.cylinder_columns),
        static_cast<float>(input.cylinder_rows)};
    const std::vector<int> rows = relevant_quantity_parameters(input);
    const float panel_width = 374.0F;
    const float panel_x = std::max(12.0F,
        static_cast<float>(width) - panel_width - 12.0F);
    const float panel_height = 66.0F + static_cast<float>(rows.size()) * 22.0F;
    glDisable(GL_TEXTURE_2D);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    rectangle(panel_x, 12, panel_width, panel_height, width, height,
        0.025F, 0.045F, 0.025F, 0.90F);
    text("QUANTITY  ARROWS CHANGE", panel_x + 10.0F, 22.0F, 2.0F, width, height);
    text("CHANGES RESET THE SCENE", panel_x + 10.0F, 43.0F, 1.5F, width, height);
    char line[64];
    int display_row = 0;
    for (const int parameter : rows) {
        const float y = 68.0F + static_cast<float>(display_row++) * 22.0F;
        if (parameter == input.quantity_parameter) {
            rectangle(panel_x + 6.0F, y - 4.0F, panel_width - 12.0F, 20.0F,
                width, height, 0.10F, 0.30F, 0.16F, 0.92F);
        }
        std::snprintf(line, sizeof(line), "%-20s %u", labels[parameter],
            static_cast<unsigned>(std::lround(values[parameter])));
        text(line, panel_x + 12.0F, y, 2.0F, width, height);
    }
    glDisable(GL_BLEND);
    glEnable(GL_TEXTURE_2D);
    glColor4f(1,1,1,1);
}

float percentile(std::vector<float> values, float fraction)
{
    std::sort(values.begin(), values.end());
    const float position = fraction * static_cast<float>(values.size() - 1U);
    const std::size_t lower = static_cast<std::size_t>(position);
    const std::size_t upper = std::min(lower + 1U, values.size() - 1U);
    const float blend = position - static_cast<float>(lower);
    return values[lower] * (1.0F - blend) + values[upper] * blend;
}

void print_profile(const char* label, const std::vector<float>& values)
{
    std::printf("PROFILE,%s,median_ms=%.4f,p5_ms=%.4f,p95_ms=%.4f,max_ms=%.4f\n",
        label, percentile(values, 0.5F), percentile(values, 0.05F),
        percentile(values, 0.95F), *std::max_element(values.begin(), values.end()));
}

int run_profile(const Options& options)
{
    waterlab::HybridOptions physics_options;
    if (options.scene == Scene::Course) {
        waterlab::gallery::RecipePhysicsOverrides overrides;
        if (options.physics_iterations != 0U) {
            overrides.physics_iterations = options.physics_iterations;
        }
        physics_options = waterlab::gallery::make_recipe_physics(
            options.context, overrides);
    } else {
        physics_options = waterlab::HybridOptions{};
        if (options.physics_iterations != 0U) {
            physics_options.physics_iterations = options.physics_iterations;
        }
    }
    waterlab::HybridDroplet droplet(physics_options);
    std::unique_ptr<waterlab::SoftBodyCourse> soft_bodies;
    if (options.scene == Scene::Course)
        soft_bodies = waterlab::gallery::make_recipe_deformable(
            options.context, physics_options, MESHPREP_SOFT_BODY_ASSET_PATH);
    waterlab::RigidSphereState rigid_sphere =
        waterlab::gallery::initial_rigid_sphere(options.context);
    waterlab::RigidSphereState caged_rigid_sphere =
        waterlab::gallery::initial_caged_rigid_sphere();
    waterlab::WaterWheelState water_wheel{};
    waterlab::FluidVisuals visuals(100'000U);
    visuals.set_active_count(physics_options.particle_count);
    visuals.set_foam_settings(context_foam_settings(options.context));
    waterlab::RayTracer raytracer;
    parallel_mater::Hierarchy empty_hierarchy;
    waterlab::Camera camera;
    Interaction control;
    control.context = options.context;
    control.course_mode = options.scene == Scene::Course;
    control.fluid_display = options.view_explicit ? options.fluid_display :
        waterlab::gallery::default_recipe_display(options.context);
    configure_context_camera(control);
    update_camera(control, camera);
    if (options.drive_box) control.rectangle_target.x = 0.0F;
    float previous_target_error = droplet.rectangle().center.x - control.rectangle_target.x;
    std::uint32_t target_crossings = 0U;
    float maximum_box_speed = 0.0F;
    std::array<std::vector<float>, 15> samples;
    const std::uint32_t frames = options.warmups + options.profile_frames;
    for (std::uint32_t frame = 0U; frame < frames; ++frame) {
        const auto begin = std::chrono::steady_clock::now();
        const waterlab::RectangleState box = droplet.rectangle();
        const float3 force = options.scene == Scene::Lab && options.drive_box
            ? rectangle_control_force(control, box, droplet.options())
            : options.scene == Scene::Lab ? multiply(box.velocity, -20.0F) : float3{};
        waterlab::HybridTimings timing{};
        const bool particle_context = recipe_has_particles(options.context);
        if (options.scene == Scene::Lab || particle_context) {
            const bool dynamic_sphere = options.scene == Scene::Course &&
                (options.context == parallel_mater::sim::SimulationRecipe::water ||
                 options.context == parallel_mater::sim::SimulationRecipe::water_rope ||
                 options.context == parallel_mater::sim::SimulationRecipe::water_soft_body);
            timing = droplet.step(force, 0.0F, nullptr, soft_bodies.get(),
                options.scene == Scene::Lab,
                dynamic_sphere ? &rigid_sphere : nullptr,
                options.context == parallel_mater::sim::SimulationRecipe::water_soft_body
                    ? &water_wheel : nullptr);
        } else if (soft_bodies) {
            const bool rolling_rigid =
                options.context == parallel_mater::sim::SimulationRecipe::cloth ||
                options.context == parallel_mater::sim::SimulationRecipe::soft_body ||
                options.context == parallel_mater::sim::SimulationRecipe::rope ||
                options.context == parallel_mater::sim::SimulationRecipe::cloth_rope ||
                options.context == parallel_mater::sim::SimulationRecipe::soft_body_rope;
            waterlab::SoftBodyTimings soft{};
            if (options.context == parallel_mater::sim::SimulationRecipe::rope) {
                const auto lattice = soft_bodies->lattice_view();
                soft = soft_bodies->step_with_tethered_rigid_spheres(
                    rigid_sphere, lattice.voxels_per_instance - 1U,
                    rigid_sphere.radius + lattice.voxel_radius,caged_rigid_sphere,
                    physics_options.gravity);
            } else if (options.context ==
                    parallel_mater::sim::SimulationRecipe::cloth_rope ||
                options.context ==
                    parallel_mater::sim::SimulationRecipe::soft_body_rope) {
                soft = soft_bodies->step_with_rigid_sphere(
                    rigid_sphere,make_float3(0.0F,0.0F,0.0F),
                    physics_options.gravity);
            } else if (rolling_rigid) {
                soft = soft_bodies->step_with_rigid_sphere(
                    rigid_sphere, physics_options.gravity);
            } else {
                soft = soft_bodies->step(physics_options.gravity);
            }
            timing.update_soft_body_physics_ms = soft.physics_ms;
            timing.rebuild_soft_body_hierarchy_ms = soft.render_hierarchy_ms;
            timing.update_soft_body_render_ms = soft.render_deformation_ms;
            timing.update_rigid_body_contact_ms = soft.rigid_contact_ms;
        }
        const waterlab::RectangleState updated_box = droplet.rectangle();
        const float target_error = updated_box.center.x - control.rectangle_target.x;
        if (options.drive_box && previous_target_error * target_error < 0.0F) {
            ++target_crossings;
        }
        previous_target_error = target_error;
        maximum_box_speed = std::max(maximum_box_speed, length(updated_box.velocity));
        const float visual_ms = particle_context &&
            control.fluid_display == waterlab::FluidDisplay::Surface
            ? visuals.update(droplet.particle_positions(), droplet.particle_velocities(),
                droplet.particle_hierarchy(), physics_options.particle_support_radius,
                physics_options.gravity, physics_options.fixed_dt, nullptr,
                physics_options.obstacle_course, droplet.particle_cells())
            : 0.0F;
        waterlab::OrientedBox render_box = droplet.render_box();
        if (options.scene == Scene::Course) {
            render_box.center = make_float3(1.0e6F, 1.0e6F, 1.0e6F);
            if (physics_options.arena == waterlab::GalleryArena::enclosed_box ||
                physics_options.arena == waterlab::GalleryArena::low_ceiling_box ||
                physics_options.arena == waterlab::GalleryArena::cloth_basin ||
                physics_options.arena == waterlab::GalleryArena::bowl ||
                physics_options.arena == waterlab::GalleryArena::rope_post ||
                physics_options.arena == waterlab::GalleryArena::rope_bridge ||
                physics_options.arena == waterlab::GalleryArena::grass ||
                physics_options.arena == waterlab::GalleryArena::fishing_tank) {
                render_box.center = rigid_sphere.center;
                render_box.half_extents = make_float3(
                    rigid_sphere.radius, rigid_sphere.radius, rigid_sphere.radius);
                render_box.sphere_orientation = rigid_sphere.orientation;
            } else if (physics_options.arena == waterlab::GalleryArena::water_wheel) {
                render_box.center = rigid_sphere.center;
                render_box.half_extents = make_float3(
                    rigid_sphere.radius, rigid_sphere.radius, rigid_sphere.radius);
                render_box.yaw = water_wheel.rim_angle;
                render_box.sphere_orientation = rigid_sphere.orientation;
            }
            if (options.context==parallel_mater::sim::SimulationRecipe::rope) {
                render_box.secondary_sphere_center=caged_rigid_sphere.center;
                render_box.secondary_sphere_radius=caged_rigid_sphere.radius;
            }
        }
        const bool render_water_skin = options.scene == Scene::Lab ||
            waterlab::gallery::recipe_has(
                options.context, parallel_mater::sim::Component::water_skin);
        const float raytrace = raytracer.render_hybrid(
            droplet.skin_mesh(), droplet.skin_normals(),
            render_water_skin ? droplet.skin_hierarchy() : empty_hierarchy,
            particle_context ? droplet.particle_positions() : nullptr,
            droplet.particle_radius(),
            particle_context ? droplet.particle_hierarchy() : empty_hierarchy,
            false, render_box, camera,
            options.width, options.height, nullptr, physics_options.obstacle_course,
            visuals.view(control.fluid_display,
                options.show_foam && control.fluid_display == waterlab::FluidDisplay::Surface,
                particle_context && control.fluid_display == waterlab::FluidDisplay::Surface),
            control.fluid_display == waterlab::FluidDisplay::Wireframe
                ? waterlab::SoftBodyRenderView{}
                : soft_bodies ? soft_bodies->render_view() : waterlab::SoftBodyRenderView{},
            options.scene == Scene::Course ? physics_options.arena
                                           : waterlab::GalleryArena::none);
        const float wall = std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - begin).count();
        if (frame < options.warmups) continue;
        const float values[15]{
            timing.rebuild_fluid_hierarchy_ms, timing.rebuild_skin_hierarchy_ms,
            timing.update_fluid_physics_ms, timing.update_skin_physics_ms,
            timing.update_rectangle_physics_ms, timing.update_surface_normals_ms,
            timing.update_render_surface_ms, timing.gpu_total_ms() + visual_ms + raytrace,
            raytrace, wall, visual_ms, timing.update_soft_body_physics_ms,
            timing.rebuild_soft_body_hierarchy_ms, timing.update_soft_body_contact_ms,
            timing.update_soft_body_render_ms};
        for (std::size_t item = 0; item < samples.size(); ++item) samples[item].push_back(values[item]);
    }
    const char* labels[15]{"rebuild_fluid_hierarchy", "rebuild_skin_hierarchy",
        "update_fluid_physics", "update_skin_physics", "update_rectangle_physics",
        "update_surface_normals", "update_render_surface", "gpu_total", "raytrace",
        "frame_wall", "water_foam", "soft_body_physics", "soft_body_hierarchy",
        "soft_body_contact", "soft_body_render"};
    for (std::size_t item = 0; item < samples.size(); ++item) print_profile(labels[item], samples[item]);
    std::vector<waterlab::FoamParticle> foam_state;
    std::uint64_t foam_tick = 0U;
    visuals.capture_foam(foam_state, foam_tick);
    const auto active_foam = std::count_if(foam_state.begin(), foam_state.end(),
        [](const waterlab::FoamParticle& particle) { return particle.active(); });
    std::printf("WATER_VISUALS,view=%u,foam_visible=%u,active_foam=%zu,foam_tick=%llu,resident_bytes=%zu\n",
        static_cast<unsigned>(control.fluid_display), options.show_foam ? 1U : 0U,
        static_cast<std::size_t>(active_foam), static_cast<unsigned long long>(foam_tick),
        visuals.allocated_bytes());
    const waterlab::HybridStatistics s = droplet.statistics();
    const waterlab::RectangleState box = droplet.rectangle();
    const auto fluid_hierarchy = droplet.particle_hierarchy().statistics();
    const auto physical_skin_hierarchy = droplet.physical_skin_hierarchy().statistics();
    const auto render_hierarchy = droplet.skin_hierarchy().statistics();
    std::vector<float3> physical_skin(droplet.physical_skin_vertex_count());
    const cudaError_t download_status = cudaMemcpy(
        physical_skin.data(), droplet.physical_skin_positions(),
        physical_skin.size() * sizeof(float3), cudaMemcpyDeviceToHost);
    if (download_status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(download_status));
    }
    double center_x = 0.0;
    double center_y = 0.0;
    double center_z = 0.0;
    for (const float3 point : physical_skin) {
        center_x += point.x;
        center_y += point.y;
        center_z += point.z;
    }
    const double inverse_count = 1.0 / static_cast<double>(physical_skin.size());
    center_x *= inverse_count;
    center_y *= inverse_count;
    center_z *= inverse_count;
    double mean_radius = 0.0;
    for (const float3 point : physical_skin) {
        const double dx = point.x - center_x;
        const double dy = point.y - center_y;
        const double dz = point.z - center_z;
        mean_radius += std::sqrt(dx * dx + dy * dy + dz * dz);
    }
    mean_radius *= inverse_count;
    double radius_variance = 0.0;
    for (const float3 point : physical_skin) {
        const double dx = point.x - center_x;
        const double dy = point.y - center_y;
        const double dz = point.z - center_z;
        const double radius = std::sqrt(dx * dx + dy * dy + dz * dz);
        const double error = radius - mean_radius;
        radius_variance += error * error;
    }
    const double radial_deviation_percent = mean_radius > 0.0
        ? 100.0 * std::sqrt(radius_variance * inverse_count) / mean_radius
        : std::numeric_limits<double>::infinity();
    const bool soft_body_pass = soft_bodies == nullptr ||
        soft_bodies->statistics().finite_failure_count == 0U;
    const bool profile_pass = soft_body_pass && s.finite_failures == 0U &&
        s.particles_outside == 0U &&
        (physics_options.obstacle_course || s.skin_vertices_inside_rectangle == 0U) &&
        std::isfinite(mean_radius) && std::isfinite(radial_deviation_percent) &&
        mean_radius > 0.0 && radial_deviation_percent <= 20.0;
    if (soft_bodies != nullptr) {
        const auto soft = soft_bodies->statistics();
        std::printf(
            "SOFT_BODY,instances=%u,voxels_per_instance=%u,total_voxels=%u,"
            "surface_voxels=%u,edges_per_instance=%u,total_edges=%u,broken=%u,"
            "finite_failures=%u,strength=%.4f,resident_bytes=%zu\n",
            soft.instance_count, soft.voxels_per_instance, soft.total_voxel_count,
            soft.surface_voxel_count, soft.edges_per_instance, soft.total_edge_count,
            soft.broken_edge_count, soft.finite_failure_count,
            soft_bodies->strength_multiplier(), soft_bodies->allocated_bytes());
    }
    std::printf(
        "BOUNDED_FORCE,mode=%s,frames=%llu,iterations=%u,particles=%u,physical_skin=%u/%u,render_skin=%u/%u,"
        "outside=%u,skin_in_box=%u,finite_failures=%u,neighbors_avg=%.2f,neighbors_max=%u,"
        "particle_force_max=%.3f,skin_force_max=%.3f,box_reaction_max=%.3f,"
        "soft_contacts=%u,soft_penetration_max=%.6f,"
        "box_center=(%.4f,%.4f,%.4f),box_speed=%.4f,box_max_speed=%.4f,"
        "target_crossings=%u,skin_mean_radius=%.5f,skin_radial_stddev_pct=%.4f,"
        "fluid_nodes=%u,fluid_depth=%u,skin_nodes=%u,skin_depth=%u,"
        "render_nodes=%u,render_depth=%u,"
        "resident_bytes=%zu,status=%s\n",
        physics_options.obstacle_course ? "course" :
            (options.drive_box ? "driven_box" : "idle"),
        static_cast<unsigned long long>(s.frame_index), physics_options.physics_iterations,
        s.particle_count,
        s.physical_skin_vertices, s.physical_skin_triangles,
        s.render_skin_vertices, s.render_skin_triangles,
        s.particles_outside, s.skin_vertices_inside_rectangle, s.finite_failures,
        s.average_particle_neighbors, s.maximum_particle_neighbors,
        s.maximum_particle_force, s.maximum_skin_force, s.maximum_rectangle_reaction,
        s.soft_body_contact_count, s.maximum_soft_body_penetration,
        box.center.x, box.center.y, box.center.z, length(box.velocity), maximum_box_speed,
        target_crossings, mean_radius, radial_deviation_percent,
        fluid_hierarchy.node_count, fluid_hierarchy.max_depth,
        physical_skin_hierarchy.node_count, physical_skin_hierarchy.max_depth,
        render_hierarchy.node_count, render_hierarchy.max_depth,
        droplet.allocated_bytes(), profile_pass ? "PASS" : "FAIL");
    return profile_pass ? 0 : 2;
}

int run_interactive(const Options& options)
{
    const std::vector<CaptureFrame> replay_frames = options.replay_path.empty()
        ? std::vector<CaptureFrame>{}
        : load_capture(options.replay_path);
    if (glfwInit() != GLFW_TRUE) throw std::runtime_error("GLFW initialization failed");
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    GLFWwindow* window = glfwCreateWindow(
        static_cast<int>(options.width), static_cast<int>(options.height),
        "ParallelMater simulation gallery", nullptr, nullptr);
    if (window == nullptr) {
        glfwTerminate();
        throw std::runtime_error("window creation failed");
    }
    glfwMakeContextCurrent(window);
    // One fixed simulation tick per displayed frame. When rendering misses the
    // display cadence, simulated time intentionally lags instead of catching up.
    glfwSwapInterval(1);
    Interaction input;
    input.context = options.context;
    input.pending_context = options.context;
    input.cloth_detail = waterlab::gallery::default_cloth_detail(input.context);
    input.fluid_display = options.view_explicit
        ? options.fluid_display
        : waterlab::gallery::default_recipe_display(input.context);
    input.show_foam = options.show_foam && recipe_has_particles(input.context);
    configure_context_camera(input);
    input.replay_mode = !replay_frames.empty();
    input.course_mode = input.replay_mode
        ? replay_frames.front().state.options.obstacle_course
        : options.scene == Scene::Course;
    if (!input.course_mode) {
        input.orbit_pitch = 0.0F;
        input.orbit_radius = 4.2F;
    }
    glfwSetWindowUserPointer(window, &input);
    glfwSetMouseButtonCallback(window, mouse_button);
    glfwSetCursorPosCallback(window, cursor_position);
    glfwSetScrollCallback(window, scroll);
    glfwSetKeyCallback(window, key_callback);

    GLuint texture = 0;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glEnable(GL_TEXTURE_2D);

    waterlab::HybridOptions physics_options;
    if (input.replay_mode) {
        physics_options = replay_frames.front().state.options;
    } else if (input.course_mode) {
        waterlab::gallery::RecipePhysicsOverrides overrides;
        if (options.physics_iterations != 0U) {
            overrides.physics_iterations = options.physics_iterations;
        }
        physics_options = waterlab::gallery::make_recipe_physics(
            input.context, overrides);
    } else {
        physics_options = waterlab::HybridOptions{};
        if (options.physics_iterations != 0U) {
            physics_options.physics_iterations = options.physics_iterations;
        }
    }
    input.course_gravity = physics_options.gravity;
    std::optional<waterlab::HybridDroplet> droplet_storage;
    droplet_storage.emplace(physics_options);
    waterlab::HybridDroplet& droplet = *droplet_storage;
    std::optional<parallel_mater::physics::Smoke> smoke_storage=
        create_smoke_system(input.context);
    begin_context_particle_spawn(input,droplet);
    std::unique_ptr<waterlab::SoftBodyCourse> soft_bodies;
    if (input.course_mode) {
        soft_bodies = waterlab::gallery::make_recipe_deformable(
            input.context, physics_options, MESHPREP_SOFT_BODY_ASSET_PATH,
            input.rope_node_count, input.cloth_detail,
            input.bridge_columns,input.bridge_rows,
            input.cylinder_columns,input.cylinder_rows);
    }
    waterlab::RigidSphereState rigid_sphere =
        waterlab::gallery::initial_rigid_sphere(input.context);
    waterlab::RigidSphereState caged_rigid_sphere =
        waterlab::gallery::initial_caged_rigid_sphere(input.rope_node_count);
    reset_level_tracking(input, rigid_sphere);
    waterlab::WaterWheelState water_wheel{};
    waterlab::FluidVisuals visuals(100'000U);
    visuals.set_active_count(droplet.options().particle_count);
    visuals.set_foam_settings(context_foam_settings(input.context));
    float visual_ms = needs_fluid_visual_update(input)
        ? visuals.update(droplet.particle_positions(), droplet.particle_velocities(),
            droplet.particle_hierarchy(), physics_options.particle_support_radius,
            physics_options.gravity, 0.0F, nullptr, physics_options.obstacle_course,
            droplet.particle_cells())
        : 0.0F;
    CaptureRing capture(
        capture_frame_capacity, physics_options.particle_count,
        droplet.physical_skin_vertex_count());
    waterlab::RayTracer raytracer;
    parallel_mater::Hierarchy empty_hierarchy;
    waterlab::Camera camera;
    waterlab::HybridTimings timings;
    SkinDebugData skin_debug;
    WireframeDebugData wireframe_debug;
    SmokeDebugData smoke_debug;
    parallel_mater::physics::SmokeTimings smoke_timings{};
    float raytrace_ms = 0.0F;
    float debug_draw_ms = 0.0F;
    float wall_ms = 0.0F;
    int texture_width = 0;
    int texture_height = 0;
    std::size_t replay_index = 0U;
    std::size_t restored_replay_index = std::numeric_limits<std::size_t>::max();
    std::string capture_status;
    if (input.replay_mode) {
        std::printf(
            "Loaded %zu captured frames. Space plays/pauses; Left/Right steps one "
            "frame; Shift+Left/Right steps 60; R returns to frame one. Camera and "
            "debug-view controls remain available.\n",
            replay_frames.size());
    } else if (input.course_mode) {
        std::printf(
            "Simulation gallery ready. In-app help lists the active shortcuts; "
            "keys 1-9 and 0 select component examples.\n");
    } else {
        std::printf(
            "Bounded-force droplet ready. Left drag orbits; Ctrl+left drag pans; "
            "Shift+left drag applies force to the box; Q/E torque; wheel zoom; "
            "P physics; V toggles smooth water / combined particle-and-mesh wireframe; "
            "F foam; Z normals; X box force; C particle force; "
            "B spring force; T timings; M saves the preceding six seconds; "
            "Space pause; R reset; Esc quit.\n");
    }

    while (glfwWindowShouldClose(window) == GLFW_FALSE) {
        const auto display_frame_begin = std::chrono::steady_clock::now();
        glfwPollEvents();
        int width = 0;
        int height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        if (width <= 0 || height <= 0) continue;
        width = std::max(width, 64);
        height = std::max(height, 64);
        if (!input.replay_mode && input.pending_context != input.context) {
            input.context = input.pending_context;
            input.cloth_detail = waterlab::gallery::default_cloth_detail(input.context);
            input.course_mode = true;
            waterlab::gallery::RecipePhysicsOverrides overrides;
            if (options.physics_iterations != 0U) {
                overrides.physics_iterations = options.physics_iterations;
            }
            physics_options = waterlab::gallery::make_recipe_physics(
                input.context, overrides);
            droplet_storage.emplace(physics_options);
            smoke_storage=create_smoke_system(input.context);
            begin_context_particle_spawn(input,droplet);
            visuals.set_active_count(droplet.options().particle_count);
            visuals.set_foam_settings(context_foam_settings(input.context));
            soft_bodies = waterlab::gallery::make_recipe_deformable(
                input.context, physics_options, MESHPREP_SOFT_BODY_ASSET_PATH,
                input.rope_node_count, input.cloth_detail,
                input.bridge_columns,input.bridge_rows,
                input.cylinder_columns,input.cylinder_rows);
            rigid_sphere = waterlab::gallery::initial_rigid_sphere(
                input.context, input.rope_node_count);
            caged_rigid_sphere = waterlab::gallery::initial_caged_rigid_sphere(
                input.rope_node_count);
            reset_level_tracking(input, rigid_sphere);
            water_wheel = {};
            input.fluid_display =
                waterlab::gallery::default_recipe_display(input.context);
            input.show_foam = options.show_foam && recipe_has_particles(input.context);
            input.course_gravity = physics_options.gravity;
            input.course_finished = false;
            input.course_progress = 0.0F;
            input.course_motion_adjustment = 0;
            input.fishing_head_x=0.0F;
            input.fishing_rope_scale=1.0F;
            input.smoke_rotor_angle=0.0F;
            input.smoke_rotor_angular_velocity=0.0F;
            input.have_follow_center = false;
            configure_context_camera(input);
            capture.clear();
            visuals.reset();
            if (needs_fluid_visual_update(input)) {
                visual_ms = visuals.update(droplet.particle_positions(),
                    droplet.particle_velocities(), droplet.particle_hierarchy(),
                    droplet.options().particle_support_radius, droplet.options().gravity,
                    0.0F, nullptr, droplet.options().obstacle_course,
                    droplet.particle_cells());
            } else {
                visual_ms = 0.0F;
            }
            const auto& selected = waterlab::gallery::recipe_info(input.context);
            std::printf("Context: %.*s\n",
                static_cast<int>(selected.title.size()), selected.title.data());
        }
        update_camera(input, camera);
        apply_camera_pan(input, camera, height);
        update_camera(input, camera);
        if (!input.replay_mode) {
            if (!input.course_mode) apply_box_drag(input, camera, height);
            if (input.quantity_adjustment != 0) {
                waterlab::HybridOptions updated = droplet.options();
                const int adjustment = input.quantity_adjustment;
                std::uint32_t soft_solves = soft_bodies
                    ? soft_bodies->spring_solver_iterations() : 0U;
                bool changed = false;
                switch (input.quantity_parameter) {
                case 0: {
                    const int step = 1'000;
                    const std::uint32_t current = input.context==
                            parallel_mater::sim::SimulationRecipe::water_soft_body &&
                            input.staged_particle_target!=0U
                        ? input.staged_particle_target : updated.particle_count;
                    const std::uint32_t count = static_cast<std::uint32_t>(std::clamp(
                        static_cast<int>(current) + step * adjustment,
                        256, 100'000));
                    changed = count != current;
                    updated.particle_count = count;
                    updated.particle_capacity = std::max(updated.particle_capacity, count);
                    break;
                }
                case 1: {
                    const std::uint32_t frequency = static_cast<std::uint32_t>(std::clamp(
                        static_cast<int>(updated.physical_skin_frequency) + adjustment,
                        2, 90));
                    changed = frequency != updated.physical_skin_frequency;
                    updated.physical_skin_frequency = frequency;
                    updated.render_skin_frequency = std::max(
                        updated.render_skin_frequency, frequency);
                    break;
                }
                case 2: {
                    const std::uint32_t frequency = static_cast<std::uint32_t>(std::clamp(
                        static_cast<int>(updated.render_skin_frequency) + 5 * adjustment,
                        static_cast<int>(updated.physical_skin_frequency), 180));
                    changed = frequency != updated.render_skin_frequency;
                    updated.render_skin_frequency = frequency;
                    break;
                }
                case 3:
                    if (soft_bodies) {
                        const std::uint32_t value = static_cast<std::uint32_t>(std::clamp(
                            static_cast<int>(soft_solves) + 4 * adjustment, 1, 256));
                        changed = value != soft_solves;
                        soft_solves = value;
                    }
                    break;
                case 4: {
                    const std::uint32_t count = static_cast<std::uint32_t>(std::clamp(
                        static_cast<int>(input.rope_node_count) + 8 * adjustment,
                        16, 512));
                    changed = count != input.rope_node_count;
                    input.rope_node_count = count;
                    break;
                }
                case 5: {
                    const std::uint32_t detail = static_cast<std::uint32_t>(std::clamp(
                        static_cast<int>(input.cloth_detail) + adjustment, 1, 8));
                    changed = detail != input.cloth_detail;
                    input.cloth_detail = detail;
                    break;
                }
                case 6: {
                    const auto value=static_cast<std::uint32_t>(std::clamp(
                        static_cast<int>(input.cylinder_columns)+adjustment,1,16));
                    changed=value!=input.cylinder_columns;
                    input.cylinder_columns=value;
                    break;
                }
                case 7: {
                    const auto value=static_cast<std::uint32_t>(std::clamp(
                        static_cast<int>(input.cylinder_rows)+adjustment,1,16));
                    changed=value!=input.cylinder_rows;
                    input.cylinder_rows=value;
                    break;
                }
                default: break;
                }
                input.quantity_adjustment = 0;
                if (changed) {
                    physics_options = updated;
                    droplet_storage.emplace(updated);
                    begin_context_particle_spawn(input,droplet);
                    soft_bodies = waterlab::gallery::make_recipe_deformable(
                        input.context, updated, MESHPREP_SOFT_BODY_ASSET_PATH,
                        input.rope_node_count, input.cloth_detail,
                        input.bridge_columns,input.bridge_rows,
                        input.cylinder_columns,input.cylinder_rows);
                    if (soft_bodies && soft_solves != 0U)
                        soft_bodies->set_spring_solver_iterations(soft_solves);
                    rigid_sphere = waterlab::gallery::initial_rigid_sphere(
                        input.context, input.rope_node_count);
                    caged_rigid_sphere =
                        waterlab::gallery::initial_caged_rigid_sphere(
                            input.rope_node_count);
                    reset_level_tracking(input, rigid_sphere);
                    water_wheel = {};
                    visuals.set_active_count(droplet.options().particle_count);
                    visuals.reset();
                    capture.clear();
                    input.reset = false;
                }
            }
            if (input.course_mode && input.physics_parameter == 17 &&
                input.physics_adjustment != 0) {
                waterlab::HybridOptions updated = droplet.options();
                updated.physical_skin_frequency = static_cast<std::uint32_t>(
                    std::clamp(static_cast<int>(updated.physical_skin_frequency) +
                        input.physics_adjustment, 2,
                        static_cast<int>(updated.render_skin_frequency)));
                input.physics_adjustment = 0;
                droplet_storage.emplace(updated);
                if (soft_bodies) soft_bodies->reset();
                capture.clear();
                visuals.reset();
            }
            if (input.course_mode &&
                (input.context==parallel_mater::sim::SimulationRecipe::cloth_rope ||
                 input.context==parallel_mater::sim::SimulationRecipe::soft_body_rope) &&
                (input.physics_parameter==29 || input.physics_parameter==30) &&
                input.physics_adjustment!=0) {
                if (input.physics_parameter==29)
                    input.bridge_columns=static_cast<std::uint32_t>(std::clamp(
                        static_cast<int>(input.bridge_columns)+input.physics_adjustment,
                        2,16));
                else
                    input.bridge_rows=static_cast<std::uint32_t>(std::clamp(
                        static_cast<int>(input.bridge_rows)+input.physics_adjustment,
                        2,64));
                input.physics_adjustment=0;
                soft_bodies=waterlab::gallery::make_recipe_deformable(
                    input.context,droplet.options(),MESHPREP_SOFT_BODY_ASSET_PATH,
                    input.rope_node_count,input.cloth_detail,
                    input.bridge_columns,input.bridge_rows,
                    input.cylinder_columns,input.cylinder_rows);
                rigid_sphere=waterlab::gallery::initial_rigid_sphere(
                    input.context,input.rope_node_count);
                caged_rigid_sphere=waterlab::gallery::initial_caged_rigid_sphere(
                    input.rope_node_count);
                reset_level_tracking(input,rigid_sphere);
            }
            apply_physics_adjustment(input, droplet, soft_bodies.get(),
                input.context == parallel_mater::sim::SimulationRecipe::cloth ||
                input.context == parallel_mater::sim::SimulationRecipe::soft_body ||
                input.context == parallel_mater::sim::SimulationRecipe::water ||
                input.context == parallel_mater::sim::SimulationRecipe::water_rope ||
                input.context == parallel_mater::sim::SimulationRecipe::water_soft_body ||
                input.context == parallel_mater::sim::SimulationRecipe::rope ||
                input.context == parallel_mater::sim::SimulationRecipe::cloth_rope ||
                input.context == parallel_mater::sim::SimulationRecipe::soft_body_rope
                    ? &rigid_sphere : nullptr, &visuals);
            if (visuals.view().particle_count != droplet.options().particle_count) {
                visuals.set_active_count(droplet.options().particle_count);
                capture.clear();
            }
            if (input.course_mode) {
                update_course_gravity(window, input, camera, droplet);
                update_fishing_controls(window,input,soft_bodies.get(),
                    droplet.options().fixed_dt);
            }
            if (input.course_mode &&
                input.context == parallel_mater::sim::SimulationRecipe::water_cloth) {
                apply_soft_body_strength(input, soft_bodies.get());
            } else if (input.course_mode) {
                apply_soft_body_strength(input, soft_bodies.get());
            }
        }
        if (input.reset) {
            if (input.replay_mode) {
                replay_index = 0U;
                restored_replay_index = std::numeric_limits<std::size_t>::max();
                input.paused = true;
            } else {
                droplet.reset();
                if (smoke_storage) {
                    const parallel_mater::Status reset=smoke_storage->reset();
                    if (!reset.ok()) throw std::runtime_error(
                        std::string("reset smoke: ")+reset.message);
                }
                begin_context_particle_spawn(input,droplet);
                visuals.set_active_count(droplet.options().particle_count);
                if (soft_bodies != nullptr) {
                    if (input.context==
                        parallel_mater::sim::SimulationRecipe::water_rope) {
                        soft_bodies->translate_pinned(
                            make_float3(-input.fishing_head_x,0.0F,0.0F));
                        soft_bodies->scale_rest_lengths(1.0F/input.fishing_rope_scale);
                        input.fishing_head_x=0.0F;
                        input.fishing_rope_scale=1.0F;
                    }
                    if (input.context ==
                        parallel_mater::sim::SimulationRecipe::water_soft_body) {
                        soft_bodies->set_pinned_rotation_z(
                            waterlab::water_wheel_center, 0.0F);
                    }
                    soft_bodies->reset();
                    waterlab::gallery::initialize_recipe_motion(
                        input.context, *soft_bodies);
                }
                rigid_sphere = waterlab::gallery::initial_rigid_sphere(
                    input.context, input.rope_node_count);
                caged_rigid_sphere = waterlab::gallery::initial_caged_rigid_sphere(
                    input.rope_node_count);
                reset_level_tracking(input, rigid_sphere);
                water_wheel = {};
                visuals.reset();
                visual_ms = needs_fluid_visual_update(input)
                    ? visuals.update(droplet.particle_positions(), droplet.particle_velocities(),
                        droplet.particle_hierarchy(), droplet.options().particle_support_radius,
                        droplet.options().gravity, 0.0F, nullptr,
                        droplet.options().obstacle_course, droplet.particle_cells())
                    : 0.0F;
                capture.clear();
                input.rectangle_target = droplet.rectangle().center;
                input.course_finished = false;
                input.have_follow_center = input.course_mode;
                input.course_progress = 0.0F;
                if (input.course_mode) {
                    input.droplet_center = {};
                    input.follow_center = {};
                    configure_context_camera(input);
                    input.course_gravity = waterlab::gallery::make_recipe_physics(
                        input.context).gravity;
                    waterlab::HybridOptions reset_options = droplet.options();
                    reset_options.gravity = input.course_gravity;
                    droplet.set_runtime_options(reset_options);
                }
            }
            input.reset = false;
        }
        if (input.replay_mode && input.replay_delta != 0) {
            const std::int64_t requested = static_cast<std::int64_t>(replay_index) +
                static_cast<std::int64_t>(input.replay_delta);
            replay_index = static_cast<std::size_t>(std::clamp<std::int64_t>(
                requested, 0, static_cast<std::int64_t>(replay_frames.size() - 1U)));
            input.replay_delta = 0;
            input.paused = true;
        }

        const auto begin = std::chrono::steady_clock::now();
        float3 control_force{};
        float control_torque = 0.0F;
        if (input.replay_mode) {
            if (restored_replay_index != replay_index) {
                const CaptureFrame& frame = replay_frames[replay_index];
                droplet.restore_state(frame.state);
                if (frame.has_soft_body) {
                    if (soft_bodies == nullptr &&
                        input.context != parallel_mater::sim::SimulationRecipe::water_cloth) {
                        throw std::runtime_error(
                            "capture contains soft-body state outside course mode");
                    }
                    // Course captures made during the deformable-post experiment
                    // remain useful for water playback. Their retired post state
                    // is intentionally ignored now that Water-Cloth uses rigid posts.
                    if (soft_bodies != nullptr)
                        soft_bodies->restore_state(frame.soft_body_state);
                } else if (soft_bodies != nullptr &&
                           restored_replay_index == std::numeric_limits<std::size_t>::max()) {
                    soft_bodies->reset();
                }
                timings = frame.timings;
                // The field is derived from this frame's restored positions. Old
                // captures have no tracer history; never invent it during replay.
                visuals.reset();
                visual_ms = needs_fluid_visual_update(input)
                    ? visuals.update(droplet.particle_positions(), droplet.particle_velocities(),
                        droplet.particle_hierarchy(), droplet.options().particle_support_radius,
                        droplet.options().gravity, 0.0F, nullptr,
                        droplet.options().obstacle_course, droplet.particle_cells())
                    : 0.0F;
                if (!frame.foam_particles.empty()) {
                    visuals.restore(frame.normal_foam);
                    visuals.restore_foam(frame.foam_particles, frame.foam_tick);
                    visual_ms = frame.visual_ms;
                }
                input.rectangle_target = frame.rectangle_target;
                restored_replay_index = replay_index;
            }
        } else if (!input.paused) {
            smoke_timings={};
            advance_context_particle_spawn(input,droplet);
            if (visuals.view().particle_count!=droplet.options().particle_count)
                visuals.set_active_count(droplet.options().particle_count);
            if (soft_bodies != nullptr)
                soft_bodies->set_solver_substeps(droplet.options().physics_iterations);
            const waterlab::RectangleState box = droplet.rectangle();
            if (!input.course_mode) {
                control_force = rectangle_control_force(input, box, droplet.options());
                control_torque = (input.rotate_right ? 260.0F : 0.0F) -
                    (input.rotate_left ? 260.0F : 0.0F) - 25.0F * box.angular_velocity;
            }
            // Exactly one fixed 1/60-second tick. A slow render delays simulated time;
            // it never launches catch-up ticks or drops a partial collider trajectory.
            if (smoke_storage) {
                parallel_mater::physics::SmokeSphereCollider collider{
                    rigid_sphere.center,rigid_sphere.velocity,rigid_sphere.radius,0.20F};
                const bool collide_sphere=
                    input.context==parallel_mater::sim::SimulationRecipe::smoke ||
                    input.context==parallel_mater::sim::SimulationRecipe::soft_body_smoke ||
                    input.context==parallel_mater::sim::SimulationRecipe::rope_smoke;
                const parallel_mater::Status status=smoke_storage->step({{},
                    collide_sphere ? &collider : nullptr,collide_sphere ? 1U : 0U},
                    smoke_timings);
                if (!status.ok()) throw std::runtime_error(
                    std::string("step smoke: ")+status.message);
            }
            const bool particle_context = recipe_has_particles(input.context);
            if (particle_context) {
                const bool dynamic_sphere =
                    input.context == parallel_mater::sim::SimulationRecipe::water ||
                    input.context == parallel_mater::sim::SimulationRecipe::water_rope ||
                    input.context == parallel_mater::sim::SimulationRecipe::water_soft_body;
                timings = droplet.step(
                    control_force, control_torque, nullptr, soft_bodies.get(),
                    !input.course_mode, dynamic_sphere ? &rigid_sphere : nullptr,
                    input.context == parallel_mater::sim::SimulationRecipe::water_soft_body
                        ? &water_wheel : nullptr,
                    input.context == parallel_mater::sim::SimulationRecipe::water_soft_body
                        ? &input.course_gravity : nullptr);
                if (input.context==parallel_mater::sim::SimulationRecipe::fluid_smoke &&
                    droplet.statistics().particle_count>256U &&
                    (droplet.statistics().frame_index&1U)==0U)
                    droplet.resize_particles(std::max(256U,
                        droplet.statistics().particle_count-16U));
                visual_ms = needs_fluid_visual_update(input)
                    ? visuals.update(droplet.particle_positions(),
                        droplet.particle_velocities(), droplet.particle_hierarchy(),
                        droplet.options().particle_support_radius,
                        droplet.options().gravity, droplet.options().fixed_dt,
                        nullptr, droplet.options().obstacle_course, droplet.particle_cells())
                    : 0.0F;
                if (input.context == parallel_mater::sim::SimulationRecipe::water_cloth) {
                    capture.record(droplet, timings, input.rectangle_target,
                        control_force, control_torque, visuals, visual_ms,
                        soft_bodies.get());
                    follow_course(input, *capture.latest());
                }
            } else if (soft_bodies != nullptr) {
                const bool rolling_rigid =
                    input.context == parallel_mater::sim::SimulationRecipe::cloth ||
                    input.context == parallel_mater::sim::SimulationRecipe::soft_body ||
                    input.context == parallel_mater::sim::SimulationRecipe::rope ||
                    input.context == parallel_mater::sim::SimulationRecipe::cloth_rope ||
                    input.context == parallel_mater::sim::SimulationRecipe::soft_body_rope;
                waterlab::SoftBodyTimings soft{};
                if (smoke_storage && recipe_has_smoke(input.context)) {
                    if (input.context==parallel_mater::sim::SimulationRecipe::cloth_smoke) {
                        soft_bodies->set_pinned_rotation_z(
                            make_float3(0.0F,0.15F,-1.20F),input.smoke_rotor_angle);
                    }
                    const bool rolling=
                        input.context==parallel_mater::sim::SimulationRecipe::soft_body_smoke ||
                        input.context==parallel_mater::sim::SimulationRecipe::rope_smoke;
                    const float3 body_gravity=
                        input.context==parallel_mater::sim::SimulationRecipe::soft_body_smoke
                        ? input.course_gravity : make_float3(0.0F,0.0F,0.0F);
                    soft=step_smoke_coupled_body(*soft_bodies,*smoke_storage,
                        rolling ? &rigid_sphere : nullptr,body_gravity,
                        input.course_gravity,droplet.options().arena,
                        droplet.options().physics_iterations,droplet.options().fixed_dt,
                        smoke_timings);
                    if (input.context==parallel_mater::sim::SimulationRecipe::cloth_smoke) {
                        const float torque=soft_bodies->wheel_rim_reaction_torque(
                            make_float3(0.0F,0.15F,-1.20F));
                        const float dt=droplet.options().fixed_dt;
                        const float acceleration=std::clamp(0.00025F*torque,-8.0F,8.0F);
                        input.smoke_rotor_angular_velocity=std::clamp(
                            (input.smoke_rotor_angular_velocity+acceleration*dt)/
                                (1.0F+0.8F*dt),-3.0F,3.0F);
                        input.smoke_rotor_angle+=
                            input.smoke_rotor_angular_velocity*dt;
                    }
                } else if (input.context == parallel_mater::sim::SimulationRecipe::rope) {
                    const auto lattice = soft_bodies->lattice_view();
                    soft = soft_bodies->step_with_tethered_rigid_spheres(
                        rigid_sphere, lattice.voxels_per_instance - 1U,
                        rigid_sphere.radius + lattice.voxel_radius,
                        caged_rigid_sphere,
                        input.course_gravity);
                } else if (input.context ==
                        parallel_mater::sim::SimulationRecipe::cloth_rope ||
                    input.context ==
                        parallel_mater::sim::SimulationRecipe::soft_body_rope) {
                    soft = soft_bodies->step_with_rigid_sphere(
                        rigid_sphere,make_float3(0.0F,0.0F,0.0F),
                        input.course_gravity);
                } else if (rolling_rigid) {
                    soft = soft_bodies->step_with_rigid_sphere(
                        rigid_sphere, input.course_gravity);
                } else {
                    soft = soft_bodies->step(input.course_gravity);
                }
                timings = {};
                timings.update_soft_body_physics_ms = soft.physics_ms;
                timings.rebuild_soft_body_hierarchy_ms = soft.render_hierarchy_ms;
                timings.update_soft_body_render_ms = soft.render_deformation_ms;
                timings.update_rigid_body_contact_ms = soft.rigid_contact_ms;
                visual_ms = 0.0F;
            } else if (input.context==parallel_mater::sim::SimulationRecipe::smoke) {
                const float dt=droplet.options().fixed_dt;
                rigid_sphere.velocity=add(rigid_sphere.velocity,
                    multiply(input.course_gravity,dt));
                rigid_sphere.velocity=multiply(rigid_sphere.velocity,
                    1.0F/(1.0F+0.12F*dt));
                rigid_sphere.center=add(rigid_sphere.center,
                    multiply(rigid_sphere.velocity,dt));
                waterlab::project_gallery_contact(rigid_sphere.center,
                    rigid_sphere.velocity,rigid_sphere.radius,droplet.options().arena);
                waterlab::advance_rigid_sphere_rotation(rigid_sphere,
                    droplet.options().arena,5.0F,dt);
                timings={};
                visual_ms=0.0F;
            }
            if (input.context == parallel_mater::sim::SimulationRecipe::water) {
                input.painted_fraction = raytracer.update_bowl_paint(
                    droplet.particle_positions(), droplet.statistics().particle_count,
                    droplet.particle_radius(), input.reset_bowl_paint);
                input.reset_bowl_paint = false;
            }
            if (input.context == parallel_mater::sim::SimulationRecipe::soft_body &&
                soft_bodies != nullptr) {
                const auto lattice = soft_bodies->lattice_view();
                input.painted_fraction = raytracer.update_sphere_paint(
                    lattice.positions, lattice.voxel_count, lattice.voxel_radius,
                    rigid_sphere, input.reset_bowl_paint);
                input.reset_bowl_paint = false;
            }
            if (input.context == parallel_mater::sim::SimulationRecipe::cloth_soft_body &&
                soft_bodies != nullptr) {
                const auto lattice = soft_bodies->lattice_view();
                raytracer.update_goal_cloth_paint(lattice.positions,
                    std::min(1'000U, lattice.voxel_count), lattice.voxel_radius,
                    input.reset_bowl_paint);
                input.reset_bowl_paint = false;
            }
            if (input.context == parallel_mater::sim::SimulationRecipe::soft_body_rope &&
                soft_bodies != nullptr) {
                const auto lattice=soft_bodies->lattice_view();
                raytracer.update_rope_bridge_paint(lattice.positions,
                    lattice.voxel_count,rigid_sphere,
                    input.bridge_columns,input.bridge_rows,input.reset_bowl_paint);
                input.reset_bowl_paint=false;
            }
            update_fishing_latch(input,soft_bodies.get(),rigid_sphere);
            update_level_progress(input, droplet, soft_bodies.get(), rigid_sphere);
        }
        if (input.capture_requested) {
            try {
                const std::filesystem::path saved = save_capture(capture);
                capture_status = "CAPTURE SAVED";
                std::printf("Capture saved: %s\nReplay with: ./build/parallel-mater-lab --replay %s\n",
                    saved.c_str(), saved.c_str());
            } catch (const std::exception& error) {
                capture_status = "CAPTURE FAILED";
                std::fprintf(stderr, "capture failed: %s\n", error.what());
            }
            input.capture_requested = false;
        }
        if (input.replay_mode && input.course_mode) {
            follow_course(input, replay_frames[replay_index]);
        }
        update_camera(input, camera);
        const bool render_particles = recipe_has_particles(input.context);
        const waterlab::SoftBodyRenderView raytraced_soft_bodies =
            input.fluid_display == waterlab::FluidDisplay::Wireframe
            ? waterlab::SoftBodyRenderView{}
            : soft_bodies ? soft_bodies->render_view() : waterlab::SoftBodyRenderView{};
        const bool render_course = input.course_mode &&
            input.context == parallel_mater::sim::SimulationRecipe::water_cloth;
        waterlab::OrientedBox render_box = droplet.render_box();
        if (input.course_mode && !render_course) {
            // Non-rigid gallery recipes use the checkerboard room without
            // inheriting the legacy lab rectangle as an undeclared collider.
            render_box.center = make_float3(1.0e6F, 1.0e6F, 1.0e6F);
        }
        if (input.course_mode && (
                droplet.options().arena == waterlab::GalleryArena::enclosed_box ||
                droplet.options().arena == waterlab::GalleryArena::low_ceiling_box ||
                droplet.options().arena == waterlab::GalleryArena::cloth_basin ||
                droplet.options().arena == waterlab::GalleryArena::bowl ||
                droplet.options().arena == waterlab::GalleryArena::rope_post ||
                droplet.options().arena == waterlab::GalleryArena::rope_bridge ||
                droplet.options().arena == waterlab::GalleryArena::ground ||
                droplet.options().arena == waterlab::GalleryArena::grass ||
                droplet.options().arena == waterlab::GalleryArena::fishing_tank)) {
            render_box.center = rigid_sphere.center;
            render_box.half_extents = make_float3(
                rigid_sphere.radius, rigid_sphere.radius, rigid_sphere.radius);
            render_box.sphere_orientation = rigid_sphere.orientation;
        }
        if (input.course_mode && input.context ==
                parallel_mater::sim::SimulationRecipe::rope) {
            render_box.secondary_sphere_center=caged_rigid_sphere.center;
            render_box.secondary_sphere_radius=caged_rigid_sphere.radius;
        }
        if (input.course_mode && droplet.options().arena ==
                waterlab::GalleryArena::water_wheel) {
            render_box.center = rigid_sphere.center;
            render_box.half_extents = make_float3(
                rigid_sphere.radius, rigid_sphere.radius, rigid_sphere.radius);
            render_box.yaw = water_wheel.rim_angle;
            render_box.sphere_orientation = rigid_sphere.orientation;
        }
        const bool render_water_skin = !input.course_mode ||
            waterlab::gallery::recipe_has(
                input.context, parallel_mater::sim::Component::water_skin);
        raytrace_ms = raytracer.render_hybrid(
            droplet.skin_mesh(), droplet.skin_normals(),
            render_water_skin ? droplet.skin_hierarchy() : empty_hierarchy,
            render_particles ? droplet.particle_positions() : nullptr,
            droplet.particle_radius(),
            render_particles ? droplet.particle_hierarchy() : empty_hierarchy, false,
            render_box, camera,
            static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height),
            nullptr, render_course, visuals.view(
                input.fluid_display,
                input.show_foam && input.fluid_display == waterlab::FluidDisplay::Surface,
                render_particles && input.fluid_display == waterlab::FluidDisplay::Surface),
            raytraced_soft_bodies,
            input.course_mode ? droplet.options().arena : waterlab::GalleryArena::none);
        const auto debug_download_begin=std::chrono::steady_clock::now();
        download_skin_debug(input, droplet, skin_debug);
        download_wireframe_debug(input, droplet, soft_bodies.get(), wireframe_debug);
        download_smoke_debug(smoke_storage ? &*smoke_storage : nullptr,smoke_debug);
        debug_draw_ms=std::chrono::duration<float,std::milli>(
            std::chrono::steady_clock::now()-debug_download_begin).count();

        glViewport(0, 0, width, height);
        glBindTexture(GL_TEXTURE_2D, texture);
        if (texture_width != width || texture_height != height) {
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0,
                GL_RGBA, GL_UNSIGNED_BYTE, raytracer.pixels());
            texture_width = width;
            texture_height = height;
        } else {
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height,
                GL_RGBA, GL_UNSIGNED_BYTE, raytracer.pixels());
        }
        glClear(GL_COLOR_BUFFER_BIT);
        glColor4f(1,1,1,1);
        glBegin(GL_QUADS);
        glTexCoord2f(0,1); glVertex2f(-1,-1);
        glTexCoord2f(1,1); glVertex2f(1,-1);
        glTexCoord2f(1,0); glVertex2f(1,1);
        glTexCoord2f(0,0); glVertex2f(-1,1);
        glEnd();
        const auto debug_render_begin=std::chrono::steady_clock::now();
        draw_smoke_particles(smoke_debug,input.context,camera,width,height);
        draw_wireframe_debug(input, wireframe_debug, camera, width, height);
        draw_skin_debug(input, skin_debug, camera, width, height);
        debug_draw_ms+=std::chrono::duration<float,std::milli>(
            std::chrono::steady_clock::now()-debug_render_begin).count();
        wall_ms = std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - begin).count();
        draw_timings(width, height, input.show_timings, input,
            timings, visual_ms, raytrace_ms, smoke_timings,debug_draw_ms, wall_ms);
        draw_course_hud(width, height, input, droplet.options(),
            droplet.statistics(), soft_bodies.get());
        draw_physics_panel(width, height, input.show_physics,
            input, droplet.options(), soft_bodies.get(),
            input.context == parallel_mater::sim::SimulationRecipe::cloth ||
            input.context == parallel_mater::sim::SimulationRecipe::soft_body ||
            input.context == parallel_mater::sim::SimulationRecipe::water ||
            input.context == parallel_mater::sim::SimulationRecipe::water_rope ||
            input.context == parallel_mater::sim::SimulationRecipe::water_soft_body ||
            input.context == parallel_mater::sim::SimulationRecipe::rope ||
            input.context == parallel_mater::sim::SimulationRecipe::cloth_rope ||
            input.context == parallel_mater::sim::SimulationRecipe::soft_body_rope
                ? &rigid_sphere : nullptr, &visuals);
        draw_quantity_panel(width, height, input.show_quantities,
            input, droplet.options(), soft_bodies.get());
        draw_context_browser(width,height,input);
        glfwSwapBuffers(window);

        const waterlab::HybridStatistics stats = droplet.statistics();
        char title[384];
        if (input.replay_mode) {
            std::snprintf(title, sizeof(title),
                "Hybrid replay | %zu/%zu | physics frame %llu | outside %u | in box %u",
                replay_index + 1U, replay_frames.size(),
                static_cast<unsigned long long>(stats.frame_index),
                stats.particles_outside, stats.skin_vertices_inside_rectangle);
        } else if (input.course_mode &&
                   input.context == parallel_mater::sim::SimulationRecipe::water_cloth) {
            const auto& recipe = waterlab::gallery::recipe_info(input.context);
            std::snprintf(title, sizeof(title),
                "%.*s | %.0fx motion | N %u | %.2f ms%s%s",
                static_cast<int>(recipe.title.size()), recipe.title.data(),
                waterlab::course_motion_multiplier(droplet.options()),
                droplet.options().physics_iterations,
                wall_ms, input.course_finished ? " | FINISH" : "",
                capture_status.empty() ? "" :
                    (capture_status == "CAPTURE SAVED" ? " | CAPTURE SAVED" :
                        " | CAPTURE FAILED"));
        } else if (input.course_mode) {
            const auto& recipe = waterlab::gallery::recipe_info(input.context);
            const auto soft_stats = soft_bodies
                ? soft_bodies->statistics() : waterlab::SoftBodyStatistics{};
            std::snprintf(title, sizeof(title),
                "%.*s | goal %.0f%%%s | gravity %.1f | N %u | particles %u | "
                "bonds %u broken | %.2f ms",
                static_cast<int>(recipe.title.size()), recipe.title.data(),
                100.0F * input.objective_progress.normalized,
                input.objective_progress.completed ? " COMPLETE" : "",
                length(input.course_gravity), droplet.options().physics_iterations,
                recipe_has_particles(input.context) ? stats.particle_count : 0U,
                soft_stats.broken_edge_count, wall_ms);
        } else {
            std::snprintf(title, sizeof(title),
                "Bounded-force water | N %u | %.2f ms | fluid %.2f | skin %.2f | "
                "box %.3f | outside %u | in box %u%s%s",
                droplet.options().physics_iterations,
                wall_ms, timings.update_fluid_physics_ms, timings.update_skin_physics_ms,
                timings.update_rectangle_physics_ms, stats.particles_outside,
                stats.skin_vertices_inside_rectangle,
                capture_status.empty() ? "" : " | ", capture_status.c_str());
        }
        glfwSetWindowTitle(window, title);

        if (input.replay_mode && !input.paused) {
            if (replay_index + 1U < replay_frames.size()) {
                ++replay_index;
            } else {
                input.paused = true;
            }
        }

        // Cap presentation at 60 Hz. A frame that takes longer still advances
        // only one tick, so physical time lags instead of being caught up.
        std::this_thread::sleep_until(
            display_frame_begin + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                std::chrono::duration<double>(1.0 / 60.0)));
    }
    glDeleteTextures(1, &texture);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    Options options;
    if (!parse_options(argc, argv, options)) {
        usage(argv[0]);
        return argc > 1 && std::string_view(argv[1]) == "--help" ? 0 : 1;
    }
    try {
        return options.profile_frames != 0U ? run_profile(options) : run_interactive(options);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "water lab failed: %s\n", error.what());
        return 1;
    }
}
