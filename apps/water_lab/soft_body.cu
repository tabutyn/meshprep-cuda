// SPDX-License-Identifier: MIT
#include "soft_body.hpp"

#include "obstacle_course.hpp"
#include "particle_cells.cuh"
#include "status_exception.hpp"
#include "triangle_contact.cuh"

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace waterlab {
namespace {

constexpr std::uint32_t block_size = 256U;
constexpr std::uint32_t invalid_edge = std::numeric_limits<std::uint32_t>::max();
constexpr std::array<char, 8> file_magic{'M', 'S', 'B', 'O', 'D', 'Y', '1', '\0'};
constexpr std::uint32_t file_version = 1U;
constexpr std::uint32_t file_endian = 0x01020304U;

struct FileHeader {
    char magic[8];
    std::uint32_t version;
    std::uint32_t endian;
    std::uint32_t header_bytes;
    std::uint32_t voxel_count;
    std::uint32_t edge_count;
    std::uint32_t neighbor_count;
    std::uint32_t render_vertex_count;
    std::uint32_t render_triangle_count;
    std::uint32_t surface_count;
    std::uint32_t pinned_count;
    std::uint32_t relaxation_iterations;
    std::uint32_t flags;
    float nominal_spacing;
    std::uint32_t reserved;
};
static_assert(sizeof(FileHeader) == 64U);

struct VoxelRecord {
    float x, y, z;
    std::uint8_t flags;
    std::uint8_t padding[3];
};
struct EdgeRecord {
    std::uint32_t a, b;
    float rest_length;
};
struct NeighborRecord {
    std::uint32_t vertex, edge;
};
struct RenderVertexRecord {
    float x, y, z, u, v;
    std::uint32_t voxels[4];
    float weights[4];
};
struct TriangleRecord {
    std::uint32_t a, b, c;
};
static_assert(sizeof(VoxelRecord) == 16U);
static_assert(sizeof(EdgeRecord) == 12U);
static_assert(sizeof(NeighborRecord) == 8U);
static_assert(sizeof(RenderVertexRecord) == 52U);
static_assert(sizeof(TriangleRecord) == 12U);

[[nodiscard]] bool finite(float value) { return std::isfinite(value); }
[[nodiscard]] bool finite(float2 value) { return finite(value.x) && finite(value.y); }
[[nodiscard]] bool finite(float3 value) {
    return finite(value.x) && finite(value.y) && finite(value.z);
}
[[nodiscard]] bool finite(float4 value) {
    return finite(value.x) && finite(value.y) && finite(value.z) && finite(value.w);
}

void check(cudaError_t status, const char* operation)
{
    detail::throw_if_failed(status, operation);
}

void check(meshprep::Status status, const char* operation)
{
    detail::throw_if_failed(status, operation);
}

template <typename T>
void allocate(T*& pointer, std::size_t count)
{
    if (count != 0U) check(cudaMalloc(&pointer, count * sizeof(T)), "soft-body cudaMalloc");
}

template <typename T>
void upload(T* destination, const std::vector<T>& source)
{
    if (!source.empty()) {
        check(cudaMemcpy(destination, source.data(), source.size() * sizeof(T),
            cudaMemcpyHostToDevice), "soft-body upload");
    }
}

template <typename T>
void read_records(std::ifstream& input, std::vector<T>& records, const char* label)
{
    if (records.empty()) return;
    input.read(reinterpret_cast<char*>(records.data()),
        static_cast<std::streamsize>(records.size() * sizeof(T)));
    if (!input) throw std::runtime_error(std::string("truncated soft-body ") + label);
}

template <typename T>
void write_records(std::ofstream& output, const std::vector<T>& records, const char* label)
{
    if (records.empty()) return;
    output.write(reinterpret_cast<const char*>(records.data()),
        static_cast<std::streamsize>(records.size() * sizeof(T)));
    if (!output) throw std::runtime_error(std::string("failed to write soft-body ") + label);
}

[[nodiscard]] std::uint32_t checked_count(std::size_t value, const char* label)
{
    if (value > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument(std::string("soft-body ") + label + " exceeds uint32");
    }
    return static_cast<std::uint32_t>(value);
}

[[nodiscard]] std::size_t checked_product(
    std::size_t first, std::size_t second, const char* label)
{
    if (first != 0U && second > std::numeric_limits<std::size_t>::max() / first) {
        throw std::invalid_argument(std::string("soft-body ") + label + " overflows size_t");
    }
    const std::size_t product = first * second;
    if (product > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument(std::string("soft-body ") + label + " exceeds uint32");
    }
    return product;
}

[[nodiscard]] float4 binding_weights(const SoftBodyBinding& binding)
{
    return binding.weights;
}

[[nodiscard]] std::array<std::uint32_t, 4> binding_voxels(const SoftBodyBinding& binding)
{
    return {binding.voxels.x, binding.voxels.y, binding.voxels.z, binding.voxels.w};
}

[[nodiscard]] std::array<float, 4> binding_weight_array(const SoftBodyBinding& binding)
{
    return {binding.weights.x, binding.weights.y, binding.weights.z, binding.weights.w};
}

[[nodiscard]] std::uint32_t find_structural_edge(
    const std::vector<SoftBodyEdge>& edges, std::uint32_t first, std::uint32_t second)
{
    if (first == second) return invalid_edge;
    const uint2 key = make_uint2(std::min(first, second), std::max(first, second));
    const auto found = std::lower_bound(edges.begin(), edges.end(), key,
        [](const SoftBodyEdge& edge, uint2 value) {
            return edge.vertices.x < value.x ||
                (edge.vertices.x == value.x && edge.vertices.y < value.y);
        });
    return found != edges.end() && found->vertices.x == key.x &&
            found->vertices.y == key.y
        ? static_cast<std::uint32_t>(found - edges.begin()) : invalid_edge;
}

void expand_render_triangles(SoftBodyAsset& asset)
{
    std::vector<float3> positions;
    std::vector<float2> uvs;
    std::vector<SoftBodyBinding> bindings;
    std::vector<uint3> triangles;
    positions.reserve(3U * asset.render_triangles.size());
    uvs.reserve(3U * asset.render_triangles.size());
    bindings.reserve(3U * asset.render_triangles.size());
    triangles.reserve(asset.render_triangles.size());
    for (const uint3 triangle : asset.render_triangles) {
        const std::uint32_t source[3]{triangle.x, triangle.y, triangle.z};
        const std::uint32_t first = static_cast<std::uint32_t>(positions.size());
        for (const std::uint32_t vertex : source) {
            positions.push_back(asset.render_positions[vertex]);
            uvs.push_back(asset.render_uvs[vertex]);
            bindings.push_back(asset.render_bindings[vertex]);
        }
        triangles.push_back(make_uint3(first, first + 1U, first + 2U));
    }
    asset.render_positions = std::move(positions);
    asset.render_uvs = std::move(uvs);
    asset.render_bindings = std::move(bindings);
    asset.render_triangles = std::move(triangles);
}

void validate_options(const SoftBodyOptions& options)
{
    if (options.instance_count == 0U ||
        options.instance_count > SoftBodyOptions::maximum_instances) {
        throw std::invalid_argument("soft-body instance count must be in [1, 256]");
    }
    if (options.solver_substeps == 0U || options.solver_substeps > 32U) {
        throw std::invalid_argument("soft-body solver substeps must be in [1, 32]");
    }
    if (options.spring_solver_iterations == 0U || options.spring_solver_iterations > 256U) {
        throw std::invalid_argument("soft-body spring iterations must be in [1, 256]");
    }
    if (!finite(options.fixed_dt) || options.fixed_dt <= 0.0F ||
        !finite(options.voxel_mass) || options.voxel_mass <= 0.0F ||
        !finite(options.spring_stiffness) || options.spring_stiffness <= 0.0F ||
        !finite(options.spring_damping_ratio) || options.spring_damping_ratio < 0.0F ||
        !finite(options.velocity_damping) || options.velocity_damping < 0.0F ||
        !finite(options.maximum_projection_fraction) ||
            options.maximum_projection_fraction <= 0.0F ||
            options.maximum_projection_fraction > 1.0F ||
        !finite(options.constraint_velocity_response) ||
            options.constraint_velocity_response < 0.0F ||
            options.constraint_velocity_response > 1.0F ||
        !finite(options.break_strain) || options.break_strain <= 0.0F ||
        options.fracture_persistence_substeps == 0U ||
            options.fracture_persistence_substeps > 64U ||
        !finite(options.maximum_speed) || options.maximum_speed <= 0.0F ||
        !finite(options.strength_multiplier) || options.strength_multiplier <= 0.0F ||
        !finite(options.cross_source_mass_multiplier) ||
            options.cross_source_mass_multiplier < 1.0F ||
            options.cross_source_mass_multiplier > 1'000'000.0F ||
        !finite(options.ground_friction) || options.ground_friction < 0.0F ||
        options.rope_bridge_columns < 2U || options.rope_bridge_columns > 16U ||
        options.rope_bridge_rows < 2U || options.rope_bridge_rows > 64U ||
        (options.rope_bridge_nodes_per_tile != 4U &&
            options.rope_bridge_nodes_per_tile != 36U &&
            options.rope_bridge_nodes_per_tile != 256U) ||
        options.hierarchy_leaf_size == 0U) {
        throw std::invalid_argument("invalid soft-body solver option");
    }
    if (!options.use_course_layout) {
        for (std::uint32_t i = 0; i < options.instance_count; ++i) {
            if (!finite(options.instance_origins[i])) {
                throw std::invalid_argument("non-finite soft-body instance origin");
            }
        }
    }
}

__host__ __device__ float3 add(float3 a, float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ float3 subtract(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ float3 multiply(float3 value, float scale)
{
    return make_float3(value.x * scale, value.y * scale, value.z * scale);
}

__host__ __device__ float dot(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ float3 cross(float3 a, float3 b)
{
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

__host__ __device__ float length(float3 value)
{
    return sqrtf(dot(value, value));
}

float4 quaternion_product(float4 a, float4 b)
{
    return make_float4(
        a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
        a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
        a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w,
        a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z);
}

bool sphere_has_support(const RigidSphereState& sphere, GalleryArena arena)
{
    constexpr float tolerance = 0.035F;
    if (arena == GalleryArena::enclosed_box || arena == GalleryArena::hot_pan ||
        arena == GalleryArena::ground_box ||
        arena == GalleryArena::cloth_basin || arena == GalleryArena::low_ceiling_box) {
        const float floor = gallery_box_center.y - gallery_box_half_extents.y;
        return sphere.center.y <= floor + sphere.radius + tolerance;
    }
    if (arena == GalleryArena::ground || arena == GalleryArena::grass ||
        arena == GalleryArena::rope_post)
        return sphere.center.y <= course_floor_y + sphere.radius + tolerance;
    if (arena == GalleryArena::rope_bridge) {
        const bool above_land = fabsf(sphere.center.z) >=
            rope_bridge_land_inner_z - sphere.radius;
        const float support = above_land ? rope_bridge_land_y : rope_bridge_deck_y;
        return sphere.center.y <= support + sphere.radius + 0.12F;
    }
    if (arena == GalleryArena::water_wheel) {
        if (water_wheel_top_platform_contact(sphere.center, sphere.radius)) return true;
        const float dx=sphere.center.x-water_wheel_center.x;
        const float dy=sphere.center.y-water_wheel_center.y;
        const float radial=sqrtf(dx*dx+dy*dy);
        const float radial_excess=fabsf(radial-water_wheel_outer_disk_radius)-0.065F;
        const float axial_excess=fabsf(sphere.center.z-water_wheel_stage_z)-
            water_wheel_outer_disk_half_thickness;
        const float rim_distance=sqrtf(fmaxf(radial_excess,0.0F)*
            fmaxf(radial_excess,0.0F)+fmaxf(axial_excess,0.0F)*
            fmaxf(axial_excess,0.0F))+fminf(fmaxf(radial_excess,axial_excess),0.0F);
        if (rim_distance<=sphere.radius+0.035F) return true;
        float support{};
        return water_wheel_support_height(sphere.center.x, support) &&
            sphere.center.y <= support + sphere.radius + tolerance;
    }
    return false;
}

void advance_rigid_sphere_rotation_impl(
    RigidSphereState& sphere, GalleryArena arena, float friction, float dt)
{
    if (sphere_has_support(sphere, arena) && friction > 0.0F) {
        const float3 normal = make_float3(0.0F, 1.0F, 0.0F);
        const float normal_speed = dot(sphere.velocity, normal);
        const float3 tangent = subtract(sphere.velocity,
            multiply(normal, normal_speed));
        const float3 rolling = multiply(cross(normal, tangent),
            1.0F / fmaxf(sphere.radius, 1.0e-5F));
        const float response = 1.0F - expf(-friction * dt);
        sphere.angular_velocity = add(
            multiply(sphere.angular_velocity, 1.0F - response),
            multiply(rolling, response));
        // Coulomb-like loss is deliberately small: friction mainly transfers
        // translation into spin rather than acting as an invisible brake.
        const float drag = fmaxf(0.0F, 1.0F - 0.015F * friction * dt);
        sphere.velocity.x *= drag;
        sphere.velocity.z *= drag;
    } else {
        sphere.angular_velocity = multiply(sphere.angular_velocity,
            expf(-0.05F * dt));
    }
    const float angular_speed=length(sphere.angular_velocity);
    if (angular_speed>12.0F)
        sphere.angular_velocity=multiply(sphere.angular_velocity,12.0F/angular_speed);
    const float4 omega = make_float4(sphere.angular_velocity.x,
        sphere.angular_velocity.y, sphere.angular_velocity.z, 0.0F);
    const float4 derivative = quaternion_product(omega, sphere.orientation);
    sphere.orientation.x += 0.5F * dt * derivative.x;
    sphere.orientation.y += 0.5F * dt * derivative.y;
    sphere.orientation.z += 0.5F * dt * derivative.z;
    sphere.orientation.w += 0.5F * dt * derivative.w;
    const float magnitude = sqrtf(sphere.orientation.x*sphere.orientation.x +
        sphere.orientation.y*sphere.orientation.y +
        sphere.orientation.z*sphere.orientation.z +
        sphere.orientation.w*sphere.orientation.w);
    if (magnitude > 1.0e-8F && std::isfinite(magnitude)) {
        sphere.orientation.x /= magnitude;
        sphere.orientation.y /= magnitude;
        sphere.orientation.z /= magnitude;
        sphere.orientation.w /= magnitude;
    } else {
        sphere.orientation = make_float4(0.0F,0.0F,0.0F,1.0F);
    }
}

void apply_rigid_contact_friction(
    RigidSphereState& sphere, float3 inward_normal, float friction, float dt)
{
    if (!(friction > 0.0F) || !(dt > 0.0F)) return;
    const float normal_length=length(inward_normal);
    if (!(normal_length>1.0e-8F)) return;
    const float3 normal=multiply(inward_normal,1.0F/normal_length);
    const float3 arm=multiply(normal,-sphere.radius);
    const float3 contact_velocity=add(sphere.velocity,
        cross(sphere.angular_velocity,arm));
    const float3 tangent=subtract(contact_velocity,
        multiply(normal,dot(contact_velocity,normal)));
    const float tangent_speed=length(tangent);
    if (!(tangent_speed>1.0e-7F)) return;
    const float inertia=0.4F*sphere.mass*sphere.radius*sphere.radius;
    const float3 direction=multiply(tangent,1.0F/tangent_speed);
    const float3 arm_cross=cross(arm,direction);
    const float inverse_effective_mass=1.0F/sphere.mass+
        dot(arm_cross,arm_cross)/fmaxf(inertia,1.0e-8F);
    const float response=1.0F-expf(-friction*dt);
    const float impulse_magnitude=fminf(
        tangent_speed*response/inverse_effective_mass,
        0.5F*sphere.mass*fmaxf(sphere.radius/dt,0.1F));
    const float3 impulse=multiply(direction,-impulse_magnitude);
    sphere.velocity=add(sphere.velocity,multiply(impulse,1.0F/sphere.mass));
    sphere.angular_velocity=add(sphere.angular_velocity,
        multiply(cross(arm,impulse),1.0F/fmaxf(inertia,1.0e-8F)));
}

__device__ bool finite3(float3 value)
{
    return isfinite(value.x) && isfinite(value.y) && isfinite(value.z);
}

__device__ float3 clamp_length(float3 value, float maximum)
{
    const float magnitude = length(value);
    return magnitude > maximum && magnitude > 0.0F
        ? multiply(value, maximum / magnitude)
        : value;
}

__device__ void project_course_board(float3& position, float3& velocity, float radius)
{
    // The first five shared contacts are floor and containment rails. Existing
    // analytic pegs are intentionally excluded: these voxels replace them.
    for (unsigned contact_index = 0U; contact_index < 5U; ++contact_index) {
        const CourseContact contact = course_contact(position, contact_index);
        if (contact.distance >= radius) continue;
        const float depth = radius - contact.distance;
        position = add(position, multiply(contact.normal, depth));
        const float normal_velocity = dot(velocity, contact.normal);
        if (normal_velocity >= 0.0F) continue;
        velocity = subtract(velocity, multiply(contact.normal, normal_velocity));
        const float speed = length(velocity);
        const float friction = fmaxf(0.0F,
            1.0F - 0.35F * (-normal_velocity) / fmaxf(speed, 1.0e-8F));
        velocity = multiply(velocity, friction);
    }
}

__global__ void find_broken_edges(const float3* positions, const SoftBodyEdge* edges,
    std::uint8_t* active_edges, std::uint8_t* edge_damage,
    std::uint32_t voxels_per_instance,
    std::uint32_t edges_per_instance, std::uint32_t total_edges,
    std::uint32_t fracture_node_first,float maximum_strain,
    std::uint32_t persistence, std::uint32_t* counters)
{
    const std::uint32_t global_edge = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_edge >= total_edges || active_edges[global_edge] == 0U) return;
    const std::uint32_t instance = global_edge / edges_per_instance;
    const std::uint32_t edge_id = global_edge - instance * edges_per_instance;
    const SoftBodyEdge edge = edges[edge_id];
    if (edge.vertices.x<fracture_node_first ||
        edge.vertices.y<fracture_node_first) return;
    const std::uint32_t base = instance * voxels_per_instance;
    const float distance = length(subtract(
        positions[base + edge.vertices.y], positions[base + edge.vertices.x]));
    if (!isfinite(distance)) return;
    if (distance <= edge.rest_length * (1.0F + maximum_strain)) {
        edge_damage[global_edge] = 0U;
        return;
    }
    const std::uint8_t next_damage = static_cast<std::uint8_t>(min(
        persistence, static_cast<std::uint32_t>(edge_damage[global_edge]) + 1U));
    edge_damage[global_edge] = next_damage;
    if (next_damage >= persistence) {
        active_edges[global_edge] = 0U;
        edge_damage[global_edge] = 0U;
        atomicAdd(counters, 1U);
    }
}

__global__ void predict_voxels(const float3* positions, const float3* velocities,
    float3* next_positions, float3* next_velocities, float3* substep_start_positions,
    const float3* rest_positions, const std::uint32_t* flags,
    std::uint32_t total_voxels, float3 gravity, float dt,
    float velocity_damping, float maximum_speed, std::uint32_t* counters)
{
    const std::uint32_t global_voxel = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_voxel >= total_voxels) return;
    substep_start_positions[global_voxel] = positions[global_voxel];
    if ((flags[global_voxel] & soft_body_voxel_pinned) != 0U) {
        next_positions[global_voxel] = rest_positions[global_voxel];
        next_velocities[global_voxel] = make_float3(0.0F, 0.0F, 0.0F);
        return;
    }

    const float3 position = positions[global_voxel];
    const float3 velocity = velocities[global_voxel];
    float3 next_velocity = add(velocity, multiply(gravity, dt));
    next_velocity = multiply(next_velocity, 1.0F / (1.0F + velocity_damping * dt));
    next_velocity = clamp_length(next_velocity, maximum_speed);
    float3 next_position = add(position, multiply(next_velocity, dt));
    if (!finite3(next_position) || !finite3(next_velocity)) {
        next_position = rest_positions[global_voxel];
        next_velocity = make_float3(0.0F, 0.0F, 0.0F);
        atomicAdd(counters + 1U, 1U);
    }
    next_positions[global_voxel] = next_position;
    next_velocities[global_voxel] = next_velocity;
}

__global__ void contact_rigid_sphere_kernel(
    float3* positions, float3* velocities, const std::uint32_t* flags,
    std::uint32_t count, RigidSphereState sphere, float node_radius,
    float node_mass, float dt, float maximum_speed, float friction,
    bool expose_impact_strain, float3* sphere_impulses)
{
    const std::uint32_t node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= count) return;
    sphere_impulses[node] = {};
    const float3 delta = subtract(positions[node], sphere.center);
    const float distance = length(delta);
    const float target = sphere.radius + node_radius;
    if (distance >= target) return;
    const float3 normal = distance > 1.0e-8F
        ? multiply(delta, 1.0F / distance) : make_float3(1.0F, 0.0F, 0.0F);
    const float depth = target - distance;
    const bool pinned = (flags[node] & soft_body_voxel_pinned) != 0U;
    if (pinned) {
        float3 impulse=multiply(normal,
            -fminf(sphere.mass * depth / dt, sphere.mass * maximum_speed));
        const float3 arm=multiply(normal,sphere.radius);
        const float3 surface_velocity=add(sphere.velocity,
            cross(sphere.angular_velocity,arm));
        const float3 tangent=subtract(surface_velocity,
            multiply(normal,dot(surface_velocity,normal)));
        const float response=1.0F-expf(-friction*dt);
        impulse=add(impulse,multiply(tangent,-sphere.mass*response));
        sphere_impulses[node]=clamp_length(impulse,sphere.mass*maximum_speed);
        return;
    }
    const float share = sphere.mass / (sphere.mass + node_mass);
    // A tearable impact cloth must observe enough of the incoming displacement
    // for its pre-projection fracture sampler to distinguish an impact from
    // rest loading. Other deformables retain the smaller stability cap.
    const float contact_limit =
        (expose_impact_strain ? 0.75F : 0.25F) * node_radius;
    const float correction_distance = fminf(depth * share, contact_limit);
    const float3 correction = multiply(normal, correction_distance);
    positions[node] = add(positions[node], correction);
    const float3 old_velocity = velocities[node];
    const float relative_normal = dot(subtract(old_velocity, sphere.velocity), normal);
    const float approach = fmaxf(0.0F, -relative_normal);
    const float3 arm=multiply(normal,sphere.radius);
    const float3 surface_velocity=add(sphere.velocity,
        cross(sphere.angular_velocity,arm));
    const float3 relative=subtract(old_velocity,surface_velocity);
    const float3 tangent=subtract(relative,multiply(normal,dot(relative,normal)));
    const float friction_response=1.0F-expf(-friction*dt);
    velocities[node] = clamp_length(add(add(old_velocity,
        multiply(normal, approach * share + correction_distance / dt)),
        multiply(tangent,-friction_response*share)), maximum_speed);
    sphere_impulses[node] = multiply(
        subtract(velocities[node], old_velocity), -node_mass);
}

__global__ void fracture_edges_from_rigid_impact(
    const SoftBodyEdge* edges,const float3* node_impulses,
    std::uint8_t* active_edges,std::uint32_t voxels_per_instance,
    std::uint32_t edges_per_instance,std::uint32_t total_edges,
    float impulse_threshold,std::uint32_t* counters)
{
    const std::uint32_t global_edge=blockIdx.x*blockDim.x+threadIdx.x;
    if (global_edge>=total_edges || active_edges[global_edge]==0U) return;
    const std::uint32_t instance=global_edge/edges_per_instance;
    const SoftBodyEdge edge=edges[global_edge-instance*edges_per_instance];
    const std::uint32_t base=instance*voxels_per_instance;
    const float load=length(node_impulses[base+edge.vertices.x])+
        length(node_impulses[base+edge.vertices.y]);
    if (!(load>impulse_threshold)) return;
    active_edges[global_edge]=0U;
    atomicAdd(counters,1U);
}

__global__ void tether_rigid_sphere_kernel(
    float3* positions, float3* velocities, const std::uint32_t* flags,
    std::uint32_t endpoint, RigidSphereState sphere,
    float attachment_distance, float node_mass,
    float maximum_rope_reach, float maximum_correction, float dt,
    float maximum_speed, float3* result)
{
    if (blockIdx.x != 0U || threadIdx.x != 0U) return;
    result[0] = {};
    result[1] = {};
    if (endpoint == 0U || (flags[endpoint] & soft_body_voxel_pinned) != 0U) return;
    const float3 node_position = positions[endpoint];
    const float3 separation = subtract(node_position, sphere.center);
    const float distance = length(separation);
    // A rope may pull but never push. The former bilateral material-point
    // constraint kept applying corrections while the ball rested on the
    // floor, injecting the observed perpetual spin.
    if (!(distance > attachment_distance) || !isfinite(distance)) return;
    const float3 direction = multiply(separation, 1.0F/distance);
    const float excess = distance - attachment_distance;
    const float sphere_inverse_mass = 1.0F/sphere.mass;
    const float node_inverse_mass = 1.0F/node_mass;
    const float inverse_sum = sphere_inverse_mass + node_inverse_mass;
    const float sphere_share = sphere_inverse_mass/inverse_sum;
    const float node_share = node_inverse_mass/inverse_sum;
    float3 sphere_correction = clamp_length(
        multiply(direction, excess*sphere_share), maximum_correction);
    const float3 node_correction = clamp_length(
        multiply(direction, -excess*node_share), maximum_correction);
    positions[endpoint] = add(node_position, node_correction);
    const float separating_speed = dot(
        subtract(velocities[endpoint], sphere.velocity), direction);
    const float damped_speed = fmaxf(0.0F, separating_speed)*0.85F;
    float3 sphere_velocity_change = multiply(
        direction, damped_speed*sphere_share);
    const float3 node_velocity_change = multiply(
        direction, -damped_speed*node_share);
    velocities[endpoint] = clamp_length(add(velocities[endpoint],
        add(node_velocity_change, multiply(node_correction, 1.0F / dt))),
        maximum_speed);
    const float3 corrected_center=add(sphere.center,sphere_correction);
    const float3 anchor_to_sphere=subtract(corrected_center,positions[0U]);
    const float reach=length(anchor_to_sphere);
    if (reach>maximum_rope_reach && reach>1.0e-8F) {
        const float3 reach_direction=multiply(anchor_to_sphere,1.0F/reach);
        sphere_correction=add(sphere_correction,multiply(
            reach_direction,maximum_rope_reach-reach));
        const float outward=dot(add(sphere.velocity,sphere_velocity_change),
            reach_direction);
        if (outward>0.0F) sphere_velocity_change=add(sphere_velocity_change,
            multiply(reach_direction,-outward));
    }
    result[0] = sphere_correction;
    result[1] = sphere_velocity_change;
}

__global__ void set_voxel_velocity_kernel(
    float3* velocities, const std::uint32_t* flags,
    std::uint32_t first, std::uint32_t last, float3 velocity)
{
    const std::uint32_t node = first + blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= last) return;
    if ((flags[node] & soft_body_voxel_pinned) == 0U) velocities[node] = velocity;
}

__global__ void translate_pinned_kernel(float3* rest_positions,
    float3* positions, const std::uint32_t* flags, std::uint32_t count,
    float3 delta)
{
    const std::uint32_t node=blockIdx.x*blockDim.x+threadIdx.x;
    if (node>=count || (flags[node]&soft_body_voxel_pinned)==0U) return;
    rest_positions[node]=add(rest_positions[node],delta);
    positions[node]=add(positions[node],delta);
}

__global__ void scale_edge_rest_lengths_kernel(
    SoftBodyEdge* edges,std::uint32_t count,float factor)
{
    const std::uint32_t edge=blockIdx.x*blockDim.x+threadIdx.x;
    if (edge<count) edges[edge].rest_length*=factor;
}

__global__ void rotate_pinned_rest_positions_kernel(
    float3* rest_positions, const float3* authored_positions,
    const std::uint32_t* flags, std::uint32_t count,
    float3 center, float axle_cosine, float axle_sine,
    float rim_cosine, float rim_sine)
{
    const std::uint32_t node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= count ||
        (flags[node] & soft_body_voxel_pinned) == 0U) return;
    const float3 relative = subtract(authored_positions[node], center);
    const bool rim = (flags[node] & soft_body_voxel_rim_anchor) != 0U;
    const float cosine = rim ? rim_cosine : axle_cosine;
    const float sine = rim ? rim_sine : axle_sine;
    rest_positions[node] = make_float3(
        center.x + cosine * relative.x - sine * relative.y,
        center.y + sine * relative.x + cosine * relative.y,
        authored_positions[node].z);
}

__global__ void wheel_rim_reaction_kernel(
    const float3* positions, const std::uint32_t* flags,
    const std::uint32_t* neighbor_offsets, const SoftBodyNeighbor* neighbors,
    const SoftBodyEdge* edges, const std::uint8_t* active_edges,
    std::uint32_t count, float stiffness, float3 center, float3* torque_rows)
{
    const std::uint32_t node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= count) return;
    torque_rows[node] = {};
    if ((flags[node] & soft_body_voxel_rim_anchor) == 0U) return;
    float3 force{};
    for (std::uint32_t row = neighbor_offsets[node];
         row < neighbor_offsets[node + 1U]; ++row) {
        const SoftBodyNeighbor neighbor = neighbors[row];
        if (active_edges[neighbor.edge] == 0U) continue;
        const float3 delta = subtract(positions[neighbor.vertex], positions[node]);
        const float distance = length(delta);
        if (!(distance > 1.0e-8F) || !isfinite(distance)) continue;
        const float extension = distance - edges[neighbor.edge].rest_length;
        force = add(force, multiply(delta, stiffness * extension / distance));
    }
    const float3 arm = subtract(positions[node], center);
    torque_rows[node].z = arm.x * force.y - arm.y * force.x;
}

__global__ void contact_soft_nodes_against_mesh_kernel(
    float3* positions, float3* velocities, const float3* start_positions,
    const std::uint32_t* flags, std::uint32_t source_count,
    const float3* render_positions, const uint3* triangles,
    const SoftBodyBinding* bindings, std::uint32_t first_target_triangle,
    std::uint32_t target_triangle_count, float radius, float maximum_correction,
    float dt,float source_inverse_mass,float target_inverse_mass,
    float maximum_speed,
    std::uint32_t* reaction_keys, float3* reaction_values)
{
    const std::uint32_t source = blockIdx.x * blockDim.x + threadIdx.x;
    if (source >= source_count) return;
    const std::uint32_t base = 3U * source;
    for (std::uint32_t corner = 0U; corner < 3U; ++corner) {
        reaction_keys[base + corner] = UINT32_MAX;
        reaction_values[base + corner] = {};
    }
    if ((flags[source] & soft_body_voxel_pinned) != 0U) return;
    const float3 point = positions[source];
    const float3 start = start_positions[source];
    float best_squared = INFINITY;
    std::uint32_t best_triangle = UINT32_MAX;
    float3 best_weights{};
    float3 best_closest{};
    float3 best_normal{};
    for (std::uint32_t local = 0U; local < target_triangle_count; ++local) {
        const std::uint32_t triangle_id = first_target_triangle + local;
        const uint3 triangle = triangles[triangle_id];
        const float3 a = render_positions[triangle.x];
        const float3 b = render_positions[triangle.y];
        const float3 c = render_positions[triangle.z];
        // Broad reject both segment endpoints before the barycentric query.
        const float min_x = fminf(a.x, fminf(b.x, c.x));
        const float max_x = fmaxf(a.x, fmaxf(b.x, c.x));
        const float min_y = fminf(a.y, fminf(b.y, c.y));
        const float max_y = fmaxf(a.y, fmaxf(b.y, c.y));
        const float min_z = fminf(a.z, fminf(b.z, c.z));
        const float max_z = fmaxf(a.z, fmaxf(b.z, c.z));
        if ((point.x < min_x - radius && start.x < min_x - radius) ||
            (point.x > max_x + radius && start.x > max_x + radius) ||
            (point.y < min_y - radius && start.y < min_y - radius) ||
            (point.y > max_y + radius && start.y > max_y + radius) ||
            (point.z < min_z - radius && start.z < min_z - radius) ||
            (point.z > max_z + radius && start.z > max_z + radius)) continue;
        const float3 weights = detail::closest_triangle_barycentric(point, a, b, c);
        const float3 closest = add(add(multiply(a, weights.x),
            multiply(b, weights.y)), multiply(c, weights.z));
        const float3 delta = subtract(point, closest);
        const float squared = dot(delta, delta);
        if (squared >= best_squared) continue;
        float3 normal = cross(subtract(b, a), subtract(c, a));
        const float normal_length = length(normal);
        if (normal_length <= 1.0e-10F) continue;
        normal = multiply(normal, 1.0F / normal_length);
        const float signed_distance = dot(delta, normal);
        const float signed_start = dot(subtract(start, closest), normal);
        const float travel = length(subtract(point, start));
        const bool shell_contact = squared <= radius * radius * 4.0F;
        const bool swept_contact = signed_start * signed_distance <= 0.0F &&
            fabsf(signed_start) <= travel + radius &&
            squared <= (travel + radius) * (travel + radius);
        if (!shell_contact && !swept_contact) continue;
        best_squared = squared;
        best_triangle = triangle_id;
        best_weights = weights;
        best_closest = closest;
        const float distance = sqrtf(squared);
        if (swept_contact && signed_start * signed_distance <= 0.0F) {
            // Preserve the side occupied at the beginning of the substep.
            // This makes the sheet a two-sided barrier and prevents a fast
            // node from being accepted on the opposite side after crossing.
            best_normal = signed_start >= 0.0F ? normal : multiply(normal, -1.0F);
        } else if (distance > 1.0e-8F) {
            best_normal = multiply(delta, 1.0F / distance);
        } else {
            best_normal = signed_start >= 0.0F ? normal : multiply(normal, -1.0F);
        }
    }
    if (best_triangle == UINT32_MAX) return;
    const float signed_distance = dot(subtract(point, best_closest), best_normal);
    const float penetration = fminf(radius - signed_distance, maximum_correction);
    if (penetration <= 0.0F) return;
    const uint3 triangle = triangles[best_triangle];
    const std::uint32_t nodes[3]{bindings[triangle.x].voxels.x,
        bindings[triangle.y].voxels.x, bindings[triangle.z].voxels.x};
    const float weights[3]{best_weights.x, best_weights.y, best_weights.z};
    float inverse_mass_sum = source_inverse_mass;
    for (std::uint32_t corner = 0U; corner < 3U; ++corner)
        if ((flags[nodes[corner]] & soft_body_voxel_pinned) == 0U)
            inverse_mass_sum += weights[corner] * weights[corner] *
                target_inverse_mass;
    const float multiplier = penetration / inverse_mass_sum;
    const float3 source_correction = multiply(
        best_normal,source_inverse_mass*multiplier);
    positions[source] = add(point, source_correction);
    velocities[source] = clamp_length(add(velocities[source],
        multiply(source_correction, 1.0F / dt)), maximum_speed);
    for (std::uint32_t corner = 0U; corner < 3U; ++corner) {
        if ((flags[nodes[corner]] & soft_body_voxel_pinned) != 0U) continue;
        reaction_keys[base + corner] = nodes[corner];
        reaction_values[base + corner] = multiply(best_normal,
            -weights[corner]*target_inverse_mass*multiplier);
    }
}

__global__ void gather_mesh_contact_kernel(
    float3* positions, float3* velocities, const std::uint32_t* flags,
    std::uint32_t first_target_node, std::uint32_t target_count,
    const std::uint32_t* keys, const float3* values,
    std::uint32_t record_count, float maximum_correction,
    float dt, float maximum_speed)
{
    const std::uint32_t local = blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= target_count) return;
    const std::uint32_t node = first_target_node + local;
    if ((flags[node] & soft_body_voxel_pinned) != 0U) return;
    float3 sum{};
    std::uint32_t contributors = 0U;
    for (std::uint32_t record = 0U; record < record_count; ++record) {
        if (keys[record] != node) continue;
        sum = add(sum, values[record]);
        ++contributors;
    }
    if (contributors == 0U) return;
    const float3 correction = clamp_length(
        multiply(sum, 1.0F / static_cast<float>(contributors)), maximum_correction);
    positions[node] = add(positions[node], correction);
    velocities[node] = clamp_length(add(velocities[node],
        multiply(correction, 1.0F / dt)), maximum_speed);
}

__global__ void project_spring_constraints(const float3* positions, float3* next_positions,
    const float3* rest_positions, const std::uint32_t* flags,
    const std::uint32_t* neighbor_offsets, const SoftBodyNeighbor* neighbors,
    const SoftBodyEdge* edges, const std::uint8_t* active_edges,
    std::uint32_t voxels_per_instance, std::uint32_t edges_per_instance,
    std::uint32_t total_voxels, float relaxation, float maximum_correction)
{
    const std::uint32_t global_voxel = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_voxel >= total_voxels) return;
    if ((flags[global_voxel] & soft_body_voxel_pinned) != 0U) {
        next_positions[global_voxel] = rest_positions[global_voxel];
        return;
    }
    const std::uint32_t instance = global_voxel / voxels_per_instance;
    const std::uint32_t local_voxel = global_voxel - instance * voxels_per_instance;
    const std::uint32_t instance_base = instance * voxels_per_instance;
    const float3 position = positions[global_voxel];
    float3 correction{};
    std::uint32_t active_count{};
    for (std::uint32_t row = neighbor_offsets[local_voxel];
         row < neighbor_offsets[local_voxel + 1U]; ++row) {
        const SoftBodyNeighbor neighbor = neighbors[row];
        const bool active =
            active_edges[instance * edges_per_instance + neighbor.edge] != 0U;
        const std::uint32_t neighbor_global = instance_base + neighbor.vertex;
        const float3 delta = subtract(positions[neighbor_global], position);
        const float distance = length(delta);
        if (!(distance > 1.0e-8F) || !isfinite(distance)) continue;
        const float rest_length = edges[neighbor.edge].rest_length;
        // A fractured edge loses tensile stiffness but retains a short-range
        // unilateral barrier. Adjacent fragments may separate; they may not
        // collapse through one another immediately after the cut.
        const float target_length = active ? rest_length : 0.55F * rest_length;
        if (!active && distance >= target_length) continue;
        const float movable_share =
            (flags[neighbor_global] & soft_body_voxel_pinned) != 0U ? 1.0F : 0.5F;
        correction = add(correction, multiply(delta,
            movable_share * (distance - target_length) / distance));
        ++active_count;
    }
    // Normalize by the rest-graph degree. Dividing by only surviving edges
    // makes every remaining spring abruptly stiffer after fracture and drives
    // the positive-feedback cascade seen in captured post failures.
    const std::uint32_t rest_degree =
        neighbor_offsets[local_voxel + 1U] - neighbor_offsets[local_voxel];
    const float3 proposal = active_count != 0U && rest_degree != 0U
        ? multiply(correction, relaxation / static_cast<float>(rest_degree))
        : float3{};
    next_positions[global_voxel] = add(
        position, clamp_length(proposal, maximum_correction));
}

__global__ void emit_voxel_cells(const float3* positions,
    std::uint64_t* keys, std::uint32_t* indices,
    std::uint32_t count, float inverse_cell_size)
{
    const std::uint32_t node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= count) return;
    const float3 p = positions[node];
    keys[node] = detail::particle_cell_key(
        static_cast<int>(floorf(p.x * inverse_cell_size)),
        static_cast<int>(floorf(p.y * inverse_cell_size)),
        static_cast<int>(floorf(p.z * inverse_cell_size)));
    indices[node] = node;
}

__device__ bool active_voxel_bond(std::uint32_t node, std::uint32_t other,
    const std::uint32_t* offsets, const SoftBodyNeighbor* neighbors,
    const std::uint8_t* active_edges, std::uint32_t voxels_per_instance,
    std::uint32_t edges_per_instance)
{
    const std::uint32_t instance = node / voxels_per_instance;
    if (instance != other / voxels_per_instance) return false;
    const std::uint32_t local = node - instance * voxels_per_instance;
    const std::uint32_t target = other - instance * voxels_per_instance;
    std::uint32_t first = offsets[local];
    std::uint32_t last = offsets[local + 1U];
    while (first < last) {
        const std::uint32_t middle = first + (last - first) / 2U;
        if (neighbors[middle].vertex < target) first = middle + 1U;
        else last = middle;
    }
    return first < offsets[local + 1U] &&
        neighbors[first].vertex == target &&
        active_edges[instance * edges_per_instance + neighbors[first].edge] != 0U;
}

__global__ void project_unbonded_voxel_contacts(
    const float3* positions, float3* next_positions,
    const std::uint32_t* flags, const std::uint32_t* neighbor_offsets,
    const SoftBodyNeighbor* neighbors, const std::uint8_t* active_edges,
    const std::uint64_t* sorted_keys, const std::uint32_t* sorted_indices,
    std::uint32_t voxels_per_instance, std::uint32_t edges_per_instance,
    std::uint32_t count, float radius, float maximum_correction)
{
    const std::uint32_t node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= count) return;
    const float3 point = positions[node];
    if ((flags[node] & soft_body_voxel_pinned) != 0U) {
        next_positions[node] = point;
        return;
    }
    const float cell_size = 2.0F * radius;
    const int cx = static_cast<int>(floorf(point.x / cell_size));
    const int cy = static_cast<int>(floorf(point.y / cell_size));
    const int cz = static_cast<int>(floorf(point.z / cell_size));
    const ParticleCellView cells{positions, sorted_keys, sorted_indices, count, cell_size};
    float3 correction{};
    const float minimum_distance = 2.0F * radius;
    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                const std::uint64_t key = detail::particle_cell_key(
                    cx + dx, cy + dy, cz + dz);
                const std::uint32_t first = detail::particle_cell_lower_bound(cells, key);
                for (std::uint32_t item = first;
                     item < count && sorted_keys[item] == key; ++item) {
                    const std::uint32_t other = sorted_indices[item];
                    if (other == node || active_voxel_bond(node, other,
                        neighbor_offsets, neighbors, active_edges,
                        voxels_per_instance, edges_per_instance)) continue;
                    const float3 delta = subtract(point, positions[other]);
                    const float distance = length(delta);
                    if (!(distance < minimum_distance)) continue;
                    const float3 normal = distance > 1.0e-8F
                        ? multiply(delta, 1.0F / distance)
                        : make_float3(node < other ? -1.0F : 1.0F, 0.0F, 0.0F);
                    const float share = (flags[other] & soft_body_voxel_pinned) != 0U
                        ? 1.0F : 0.5F;
                    correction = add(correction, multiply(normal,
                        share * (minimum_distance - distance)));
                }
            }
        }
    }
    next_positions[node] = add(point,
        clamp_length(correction, maximum_correction));
}

__global__ void finalize_constraint_velocities(const float3* positions,
    const float3* substep_start_positions, const float3* predicted_velocities,
    const std::uint32_t* flags, float3* velocities, std::uint32_t total_voxels,
    float inverse_dt, float projection_response, float maximum_speed,
    std::uint32_t* counters)
{
    const std::uint32_t voxel = blockIdx.x * blockDim.x + threadIdx.x;
    if (voxel >= total_voxels) return;
    if ((flags[voxel] & soft_body_voxel_pinned) != 0U) {
        velocities[voxel] = make_float3(0.0F, 0.0F, 0.0F);
        return;
    }
    const float3 projected_velocity = multiply(subtract(
        positions[voxel], substep_start_positions[voxel]), inverse_dt);
    float3 velocity = add(predicted_velocities[voxel], multiply(
        subtract(projected_velocity, predicted_velocities[voxel]), projection_response));
    velocity = clamp_length(velocity, maximum_speed);
    if (!finite3(velocity)) {
        velocity = make_float3(0.0F, 0.0F, 0.0F);
        atomicAdd(counters + 1U, 1U);
    }
    velocities[voxel] = velocity;
}

__global__ void damp_spring_velocities(const float3* positions,
    const float3* velocities, float3* next_velocities, const std::uint32_t* flags,
    const std::uint32_t* neighbor_offsets, const SoftBodyNeighbor* neighbors,
    const std::uint8_t* active_edges, std::uint32_t voxels_per_instance,
    std::uint32_t edges_per_instance, std::uint32_t total_voxels, float response,
    float maximum_speed)
{
    const std::uint32_t global_voxel = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_voxel >= total_voxels) return;
    if ((flags[global_voxel] & soft_body_voxel_pinned) != 0U) {
        next_velocities[global_voxel] = make_float3(0.0F, 0.0F, 0.0F);
        return;
    }
    const std::uint32_t instance = global_voxel / voxels_per_instance;
    const std::uint32_t local_voxel = global_voxel - instance * voxels_per_instance;
    const std::uint32_t instance_base = instance * voxels_per_instance;
    const float3 velocity = velocities[global_voxel];
    float3 correction{};
    std::uint32_t active_count{};
    for (std::uint32_t row = neighbor_offsets[local_voxel];
         row < neighbor_offsets[local_voxel + 1U]; ++row) {
        const SoftBodyNeighbor neighbor = neighbors[row];
        if (active_edges[instance * edges_per_instance + neighbor.edge] == 0U) continue;
        const std::uint32_t neighbor_global = instance_base + neighbor.vertex;
        const float3 delta = subtract(positions[neighbor_global], positions[global_voxel]);
        const float distance = length(delta);
        if (!(distance > 1.0e-8F)) continue;
        const float3 direction = multiply(delta, 1.0F / distance);
        const float relative_speed = dot(
            subtract(velocities[neighbor_global], velocity), direction);
        const float movable_share =
            (flags[neighbor_global] & soft_body_voxel_pinned) != 0U ? 1.0F : 0.5F;
        correction = add(correction, multiply(direction, movable_share * relative_speed));
        ++active_count;
    }
    next_velocities[global_voxel] = clamp_length(active_count != 0U
        ? add(velocity, multiply(correction, response / static_cast<float>(active_count)))
        : velocity, maximum_speed);
}

__global__ void apply_contact_impulses(float3* positions, float3* velocities,
    const float3* rest_positions,const float3* substep_start_positions,
    const std::uint32_t* flags,
    const float3* external_impulses, const float3* position_corrections,
    std::uint32_t total_voxels, float dt, float inverse_mass,
    float maximum_speed, float voxel_radius, bool course_board_collisions,
    GalleryArena arena, float ground_friction,
    std::uint32_t* counters)
{
    const std::uint32_t voxel = blockIdx.x * blockDim.x + threadIdx.x;
    if (voxel >= total_voxels) return;
    if ((flags[voxel] & soft_body_voxel_pinned) != 0U) {
        positions[voxel] = rest_positions[voxel];
        velocities[voxel] = make_float3(0.0F, 0.0F, 0.0F);
        return;
    }
    const float3 impulse = external_impulses[voxel];
    const float3 correction = position_corrections[voxel];
    float3 position = add(positions[voxel], finite3(correction)
        ? correction : make_float3(0.0F, 0.0F, 0.0F));
    if (finite3(impulse)) {
        // Contact arrives after free prediction. Advance by the impulse-driven
        // velocity change so the subsequent constraint projection and velocity
        // reconstruction both observe it.
        position = add(position, multiply(impulse, inverse_mass * dt));
    }
    float3 velocity = add(velocities[voxel], finite3(impulse)
        ? multiply(impulse, inverse_mass) : make_float3(0.0F, 0.0F, 0.0F));
    // Position-level contact is not an impulse. Converting the full projection
    // into velocity here and again during constraint reconstruction injects
    // energy proportional to penetration depth. The bounded reconstruction
    // below transfers only the configured fraction into momentum.
    velocity = clamp_length(velocity, maximum_speed);
    if (arena != GalleryArena::none) {
        if (arena==GalleryArena::rope_post)
            project_swept_rope_post(substep_start_positions[voxel],position,
                velocity,2.0F*voxel_radius);
        project_gallery_contact(position, velocity, voxel_radius, arena);
        // Friction is deliberately applied after constraint velocity
        // reconstruction below. Applying it here let the spring solve rebuild
        // the pre-friction tangential velocity, which made traction appear to
        // work only after steering stopped.
        (void)ground_friction;
    } else if (course_board_collisions) {
        project_course_board(position, velocity, voxel_radius);
    }
    if (!finite3(position) || !finite3(velocity)) {
        position = rest_positions[voxel];
        velocity = make_float3(0.0F, 0.0F, 0.0F);
        atomicAdd(counters + 1U, 1U);
    }
    positions[voxel] = position;
    velocities[voxel] = velocity;
}

__global__ void apply_post_constraint_ground_friction(
    const float3* positions, float3* velocities, const std::uint32_t* flags,
    std::uint32_t count, GalleryArena arena, float radius,
    float coefficient, float normal_acceleration, float dt)
{
    const std::uint32_t node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= count || (flags[node] & soft_body_voxel_pinned) != 0U ||
        (arena != GalleryArena::ground && arena != GalleryArena::grass &&
         arena != GalleryArena::ground_box &&
         arena != GalleryArena::rope_post) ||
        positions[node].y > (arena == GalleryArena::ground_box &&
                inside_ground_pit(positions[node], radius)
            ? ground_pit_bottom_y : course_floor_y) + radius + 1.0e-4F) return;
    float3 velocity = velocities[node];
    const float tangential_speed = hypotf(velocity.x, velocity.z);
    const float reduced = fmaxf(0.0F, tangential_speed -
        coefficient * fmaxf(normal_acceleration, 0.0F) * dt);
    const float scale = tangential_speed > 1.0e-8F
        ? reduced / tangential_speed : 0.0F;
    velocity.x *= scale;
    velocity.z *= scale;
    velocities[node] = velocity;
}

__device__ bool make_local_frame(float3 first, float3 second, float3 third,
    float3& tangent, float3& bitangent, float3& normal)
{
    const float3 first_edge = subtract(second, first);
    const float first_length = length(first_edge);
    if (!(first_length > 1.0e-8F) || !isfinite(first_length)) return false;
    tangent = multiply(first_edge, 1.0F / first_length);
    const float3 second_edge = subtract(third, first);
    const float3 perpendicular = subtract(
        second_edge, multiply(tangent, dot(second_edge, tangent)));
    const float perpendicular_length = length(perpendicular);
    if (!(perpendicular_length > 1.0e-8F) || !isfinite(perpendicular_length)) return false;
    bitangent = multiply(perpendicular, 1.0F / perpendicular_length);
    normal = cross(tangent, bitangent);
    return finite3(tangent) && finite3(bitangent) && finite3(normal);
}

__global__ void deform_render_vertices(const float3* voxel_positions,
    const float3* local_rest_voxels, const float3* local_render_positions,
    const SoftBodyBinding* combined_bindings, std::uint32_t voxels_per_instance,
    std::uint32_t render_vertices_per_instance, std::uint32_t total_vertices,
    const uint2* local_frame_edges, const std::uint8_t* active_edges,
    std::uint32_t edges_per_instance, float3* render_positions)
{
    const std::uint32_t global_vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_vertex >= total_vertices) return;
    const std::uint32_t instance = global_vertex / render_vertices_per_instance;
    const std::uint32_t local_vertex = global_vertex - instance * render_vertices_per_instance;
    const SoftBodyBinding binding = combined_bindings[global_vertex];
    const std::uint32_t voxel_base = instance * voxels_per_instance;
    const std::uint32_t anchor = binding.voxels.x;
    const float3 current_anchor = voxel_positions[anchor];
    const float3 rest_anchor = local_rest_voxels[anchor - voxel_base];
    const float3 rest_offset = subtract(local_render_positions[local_vertex], rest_anchor);
    float3 deformed = add(current_anchor, rest_offset);
    const uint2 frame_edges = local_frame_edges[local_vertex];
    const std::uint32_t edge_base = instance * edges_per_instance;
    if (frame_edges.x != invalid_edge && frame_edges.y != invalid_edge &&
        active_edges[edge_base + frame_edges.x] != 0U &&
        active_edges[edge_base + frame_edges.y] != 0U) {
        const std::uint32_t second = binding.voxels.y;
        const std::uint32_t third = binding.voxels.z;
        float3 rest_tangent{}, rest_bitangent{}, rest_normal{};
        float3 current_tangent{}, current_bitangent{}, current_normal{};
        if (make_local_frame(rest_anchor,
                local_rest_voxels[second - voxel_base],
                local_rest_voxels[third - voxel_base],
                rest_tangent, rest_bitangent, rest_normal) &&
            make_local_frame(current_anchor, voxel_positions[second],
                voxel_positions[third], current_tangent, current_bitangent,
                current_normal)) {
            deformed = add(current_anchor, add(
                multiply(current_tangent, dot(rest_offset, rest_tangent)),
                add(multiply(current_bitangent, dot(rest_offset, rest_bitangent)),
                    multiply(current_normal, dot(rest_offset, rest_normal)))));
        }
    }
    render_positions[global_vertex] = deformed;
}

__global__ void preserve_fractured_triangles(
    float3* render_positions, const float3* local_render_positions,
    const uint3* triangles, const uint3* local_triangle_edges,
    const std::uint8_t* active_edges, std::uint32_t triangles_per_instance,
    std::uint32_t vertices_per_instance, std::uint32_t edges_per_instance,
    std::uint32_t triangle_count)
{
    const std::uint32_t global_triangle = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_triangle >= triangle_count) return;
    const std::uint32_t instance = global_triangle / triangles_per_instance;
    const std::uint32_t local_triangle = global_triangle - instance * triangles_per_instance;
    const uint3 edge = local_triangle_edges[local_triangle];
    const std::uint32_t edge_base = instance * edges_per_instance;
    const bool active[3]{
        edge.x != invalid_edge && active_edges[edge_base + edge.x] != 0U,
        edge.y != invalid_edge && active_edges[edge_base + edge.y] != 0U,
        edge.z != invalid_edge && active_edges[edge_base + edge.z] != 0U};
    if (active[0] && active[1] && active[2]) return;

    const uint3 triangle = triangles[global_triangle];
    const std::uint32_t vertex_base = instance * vertices_per_instance;
    const std::uint32_t local_vertex[3]{
        triangle.x - vertex_base, triangle.y - vertex_base, triangle.z - vertex_base};
    const float3 rest[3]{local_render_positions[local_vertex[0]],
        local_render_positions[local_vertex[1]], local_render_positions[local_vertex[2]]};
    const float3 current[3]{render_positions[triangle.x],
        render_positions[triangle.y], render_positions[triangle.z]};

    // Prefer an intact side as the hinge; otherwise retain corner zero as a
    // material-point attachment.  Triangle-local vertices make this update
    // race free even when adjacent faces choose different hinges.
    int first = 0;
    int second = 1;
    int third = 2;
    if (!active[0] && active[1]) { first = 1; second = 2; third = 0; }
    else if (!active[0] && !active[1] && active[2]) {
        first = 2; second = 0; third = 1;
    }
    float3 rest_tangent{}, rest_bitangent{}, rest_normal{};
    float3 current_tangent{}, current_bitangent{}, current_normal{};
    const bool rest_frame = make_local_frame(rest[first], rest[second], rest[third],
        rest_tangent, rest_bitangent, rest_normal);
    const bool current_frame = make_local_frame(current[first], current[second], current[third],
        current_tangent, current_bitangent, current_normal);
    if (!rest_frame) return;
    if (!current_frame) {
        current_tangent = rest_tangent;
        current_bitangent = rest_bitangent;
        current_normal = rest_normal;
    }
    const bool hinge = active[0] || active[1] || active[2];
    const float3 rest_origin = hinge
        ? multiply(add(rest[first], rest[second]), 0.5F) : rest[first];
    const float3 current_origin = hinge
        ? multiply(add(current[first], current[second]), 0.5F) : current[first];
    const std::uint32_t output[3]{triangle.x, triangle.y, triangle.z};
    for (int corner = 0; corner < 3; ++corner) {
        const float3 offset = subtract(rest[corner], rest_origin);
        render_positions[output[corner]] = add(current_origin, add(
            multiply(current_tangent, dot(offset, rest_tangent)),
            add(multiply(current_bitangent, dot(offset, rest_bitangent)),
                multiply(current_normal, dot(offset, rest_normal)))));
    }
}

__global__ void emit_triangle_bounds(const float3* positions, const uint3* triangles,
    std::uint32_t count, meshprep::Aabb* bounds)
{
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const uint3 triangle = triangles[index];
    const float3 a = positions[triangle.x];
    const float3 b = positions[triangle.y];
    const float3 c = positions[triangle.z];
    constexpr float padding = 1.0e-5F;
    bounds[index] = {
        make_float3(fminf(a.x, fminf(b.x, c.x)) - padding,
            fminf(a.y, fminf(b.y, c.y)) - padding,
            fminf(a.z, fminf(b.z, c.z)) - padding),
        make_float3(fmaxf(a.x, fmaxf(b.x, c.x)) + padding,
            fmaxf(a.y, fmaxf(b.y, c.y)) + padding,
            fmaxf(a.z, fmaxf(b.z, c.z)) + padding)};
}

__global__ void emit_member_bounds(const float3* positions,
    const SoftBodyEdge* edges, const std::uint8_t* active,
    std::uint32_t voxels_per_instance, std::uint32_t edges_per_instance,
    std::uint32_t count, float half_width, meshprep::Aabb* bounds)
{
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const std::uint32_t instance = index / edges_per_instance;
    const SoftBodyEdge edge = edges[index - instance * edges_per_instance];
    const std::uint32_t base = instance * voxels_per_instance;
    const float3 a = positions[base + edge.vertices.x];
    const float3 b = positions[base + edge.vertices.y];
    // Broken members retain a tiny finite bound at an endpoint.  Traversal
    // rejects them through member_active, while the stable primitive count
    // lets hierarchy refitting stay allocation-free.
    const float padding = active[index] != 0U ? half_width : 1.0e-6F;
    bounds[index] = {
        make_float3(fminf(a.x, b.x) - padding, fminf(a.y, b.y) - padding,
            fminf(a.z, b.z) - padding),
        make_float3(fmaxf(a.x, b.x) + padding, fmaxf(a.y, b.y) + padding,
            fmaxf(a.z, b.z) + padding)};
}

[[nodiscard]] float elapsed(cudaEvent_t begin, cudaEvent_t end)
{
    float milliseconds{};
    check(cudaEventElapsedTime(&milliseconds, begin, end), "measure soft-body stage");
    return milliseconds;
}

void contain_sphere_in_rope_cage(RigidSphereState& sphere,
    const std::vector<float3>& positions,std::uint32_t first,float clearance)
{
    constexpr std::uint32_t faces[6][4]{
        {0,1,3,2},{4,6,7,5},{0,4,5,1},
        {2,3,7,6},{0,2,6,4},{1,5,7,3}};
    float3 center{};
    for (std::uint32_t node=0U;node<8U;++node)
        center=add(center,positions[first+node]);
    center=multiply(center,1.0F/8.0F);
    // Sequential half-space projection closes the six quadrilateral faces.
    // Node spheres still handle ordinary force transfer; this is the hard
    // geometric invariant that prevents the glass sphere escaping a gap.
    for (std::uint32_t pass=0U;pass<8U;++pass) {
        for (const auto& face:faces) {
            float3 face_center{};
            for (std::uint32_t corner:face)
                face_center=add(face_center,positions[first+corner]);
            face_center=multiply(face_center,0.25F);
            const float3 a=positions[first+face[0]];
            const float3 b=positions[first+face[1]];
            const float3 c=positions[first+face[2]];
            float3 normal=cross(subtract(b,a),subtract(c,a));
            const float normal_length=length(normal);
            if (!(normal_length>1.0e-8F)) continue;
            normal=multiply(normal,1.0F/normal_length);
            if (dot(normal,subtract(face_center,center))<0.0F)
                normal=multiply(normal,-1.0F);
            const float limit=-sphere.radius-clearance;
            const float signed_distance=dot(
                subtract(sphere.center,face_center),normal);
            if (signed_distance<=limit) continue;
            sphere.center=add(sphere.center,
                multiply(normal,limit-signed_distance));
            const float outward=dot(sphere.velocity,normal);
            if (outward>0.0F)
                sphere.velocity=add(sphere.velocity,multiply(normal,-outward));
        }
    }
}

} // namespace

void advance_rigid_sphere_rotation(
    RigidSphereState& sphere, GalleryArena arena, float friction, float dt)
{
    advance_rigid_sphere_rotation_impl(sphere,arena,friction,dt);
}

void validate_soft_body_asset(const SoftBodyAsset& asset)
{
    const std::uint32_t voxel_count = checked_count(asset.rest_voxels.size(), "voxel count");
    const std::uint32_t edge_count = checked_count(asset.edges.size(), "edge count");
    if (voxel_count == 0U || edge_count == 0U || !finite(asset.nominal_spacing) ||
        asset.nominal_spacing <= 0.0F || !finite(asset.voxel_radius) ||
        asset.voxel_radius <= 0.0F ||
        (asset.file_flags & (soft_body_asset_y_up | soft_body_asset_delta_skinning)) !=
            (soft_body_asset_y_up | soft_body_asset_delta_skinning)) {
        throw std::invalid_argument("soft-body asset requires voxels, edges, and positive radius");
    }
    if (asset.voxel_flags.size() != voxel_count ||
        asset.neighbor_offsets.size() != static_cast<std::size_t>(voxel_count) + 1U) {
        throw std::invalid_argument("soft-body voxel flags or CSR offsets have the wrong size");
    }
    if (asset.neighbor_offsets.front() != 0U ||
        asset.neighbor_offsets.back() != asset.neighbors.size()) {
        throw std::invalid_argument("soft-body CSR endpoints are invalid");
    }
    bool has_pinned = false;
    bool has_surface = false;
    for (std::uint32_t vertex = 0U; vertex < voxel_count; ++vertex) {
        if (!finite(asset.rest_voxels[vertex])) {
            throw std::invalid_argument("soft-body voxel contains a non-finite coordinate");
        }
        const std::uint32_t flags = asset.voxel_flags[vertex];
        if ((flags & ~(soft_body_voxel_pinned | soft_body_voxel_surface |
                soft_body_voxel_rim_anchor)) != 0U) {
            throw std::invalid_argument("soft-body voxel uses an unknown flag");
        }
        if ((flags & soft_body_voxel_rim_anchor) != 0U &&
            (flags & soft_body_voxel_pinned) == 0U)
            throw std::invalid_argument("soft-body rim anchor must also be pinned");
        has_pinned |= (flags & soft_body_voxel_pinned) != 0U;
        has_surface |= (flags & soft_body_voxel_surface) != 0U;
        if (asset.neighbor_offsets[vertex] > asset.neighbor_offsets[vertex + 1U]) {
            throw std::invalid_argument("soft-body CSR offsets are not monotonic");
        }
    }
    if (!has_surface ||
        (!has_pinned && (asset.file_flags & soft_body_asset_free_body) == 0U)) {
        throw std::invalid_argument("soft-body asset requires surface voxels and anchors unless free");
    }

    std::vector<std::uint32_t> edge_occurrences(edge_count, 0U);
    uint2 previous_edge{};
    for (std::uint32_t edge_id = 0U; edge_id < edge_count; ++edge_id) {
        const SoftBodyEdge edge = asset.edges[edge_id];
        if (edge.vertices.x >= voxel_count || edge.vertices.y >= voxel_count ||
            edge.vertices.x >= edge.vertices.y || !finite(edge.rest_length) ||
            edge.rest_length <= 0.0F) {
            throw std::invalid_argument("invalid or non-canonical soft-body edge");
        }
        if (edge_id != 0U && (edge.vertices.x < previous_edge.x ||
            (edge.vertices.x == previous_edge.x && edge.vertices.y <= previous_edge.y))) {
            throw std::invalid_argument("soft-body edges must be uniquely lexicographically sorted");
        }
        previous_edge = edge.vertices;
    }
    for (std::uint32_t vertex = 0U; vertex < voxel_count; ++vertex) {
        std::uint32_t previous_neighbor = 0U;
        bool first = true;
        for (std::uint32_t row = asset.neighbor_offsets[vertex];
             row < asset.neighbor_offsets[vertex + 1U]; ++row) {
            const SoftBodyNeighbor neighbor = asset.neighbors[row];
            if (neighbor.vertex >= voxel_count || neighbor.vertex == vertex ||
                neighbor.edge >= edge_count || (!first && neighbor.vertex <= previous_neighbor)) {
                throw std::invalid_argument("invalid, duplicate, or unsorted soft-body neighbor");
            }
            const uint2 endpoints = asset.edges[neighbor.edge].vertices;
            if (!((endpoints.x == vertex && endpoints.y == neighbor.vertex) ||
                  (endpoints.y == vertex && endpoints.x == neighbor.vertex))) {
                throw std::invalid_argument("soft-body neighbor refers to an unrelated edge");
            }
            ++edge_occurrences[neighbor.edge];
            previous_neighbor = neighbor.vertex;
            first = false;
        }
    }
    if (std::any_of(edge_occurrences.begin(), edge_occurrences.end(),
            [](std::uint32_t count) { return count != 2U; })) {
        throw std::invalid_argument("every soft-body edge must occur twice in CSR adjacency");
    }

    const std::size_t render_vertex_count = asset.render_positions.size();
    if (render_vertex_count == 0U || asset.render_triangles.empty() ||
        asset.render_uvs.size() != render_vertex_count ||
        asset.render_bindings.size() != render_vertex_count) {
        throw std::invalid_argument("soft-body render arrays have inconsistent sizes");
    }
    (void)checked_count(render_vertex_count, "render vertex count");
    (void)checked_count(asset.render_triangles.size(), "render triangle count");
    for (std::size_t vertex = 0U; vertex < render_vertex_count; ++vertex) {
        if (!finite(asset.render_positions[vertex]) || !finite(asset.render_uvs[vertex]) ||
            !finite(binding_weights(asset.render_bindings[vertex]))) {
            throw std::invalid_argument("soft-body render vertex is non-finite");
        }
        const auto voxel_ids = binding_voxels(asset.render_bindings[vertex]);
        const auto weights = binding_weight_array(asset.render_bindings[vertex]);
        float weight_sum{};
        for (unsigned slot = 0U; slot < 4U; ++slot) {
            if (voxel_ids[slot] >= voxel_count || weights[slot] < 0.0F) {
                throw std::invalid_argument("invalid soft-body render binding");
            }
            if (weights[slot] > 0.0F &&
                (asset.voxel_flags[voxel_ids[slot]] & soft_body_voxel_surface) == 0U) {
                throw std::invalid_argument("render binding refers to an interior voxel");
            }
            weight_sum += weights[slot];
        }
        if (std::abs(weight_sum - 1.0F) > 1.0e-4F) {
            throw std::invalid_argument("soft-body render weights do not sum to one");
        }
    }
    for (const uint3 triangle : asset.render_triangles) {
        if (triangle.x >= render_vertex_count || triangle.y >= render_vertex_count ||
            triangle.z >= render_vertex_count || triangle.x == triangle.y ||
            triangle.y == triangle.z || triangle.z == triangle.x) {
            throw std::invalid_argument("invalid soft-body render triangle");
        }
    }
    for (const SoftBodyEdge edge : asset.render_member_edges) {
        if (edge.vertices.x >= voxel_count || edge.vertices.y >= voxel_count ||
            edge.vertices.x >= edge.vertices.y || !finite(edge.rest_length) ||
            edge.rest_length <= 0.0F) {
            throw std::invalid_argument("invalid presentation-only soft-body member");
        }
    }
}

SoftBodyAsset load_soft_body_asset(const std::string& path)
{
    if constexpr (std::endian::native != std::endian::little) {
        throw std::runtime_error("soft-body .msb loading currently requires little-endian host");
    }
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open soft-body asset: " + path);
    FileHeader header{};
    input.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!input || !std::equal(file_magic.begin(), file_magic.end(), header.magic) ||
        header.version != file_version || header.endian != file_endian ||
        header.header_bytes != sizeof(FileHeader)) {
        throw std::runtime_error("unsupported or corrupt soft-body asset header: " + path);
    }
    constexpr std::uint32_t maximum_records = 50'000'000U;
    if (header.voxel_count > maximum_records || header.edge_count > maximum_records ||
        header.neighbor_count > maximum_records ||
        header.render_vertex_count > maximum_records ||
        header.render_triangle_count > maximum_records) {
        throw std::runtime_error("soft-body asset count exceeds safety limit");
    }

    std::vector<VoxelRecord> voxels(header.voxel_count);
    std::vector<EdgeRecord> edges(header.edge_count);
    std::vector<std::uint32_t> offsets(static_cast<std::size_t>(header.voxel_count) + 1U);
    std::vector<NeighborRecord> neighbors(header.neighbor_count);
    std::vector<RenderVertexRecord> render_vertices(header.render_vertex_count);
    std::vector<TriangleRecord> triangles(header.render_triangle_count);
    read_records(input, voxels, "voxels");
    read_records(input, edges, "edges");
    read_records(input, offsets, "neighbor offsets");
    read_records(input, neighbors, "neighbors");
    read_records(input, render_vertices, "render vertices");
    read_records(input, triangles, "render triangles");
    if (input.peek() != std::char_traits<char>::eof()) {
        throw std::runtime_error("soft-body asset has unexpected trailing bytes: " + path);
    }

    SoftBodyAsset asset;
    asset.nominal_spacing = header.nominal_spacing;
    asset.voxel_radius = 0.45F * header.nominal_spacing;
    asset.relaxation_iterations = header.relaxation_iterations;
    asset.file_flags = header.flags;
    asset.rest_voxels.reserve(voxels.size());
    asset.voxel_flags.reserve(voxels.size());
    for (const auto& voxel : voxels) {
        asset.rest_voxels.push_back(make_float3(voxel.x, voxel.y, voxel.z));
        asset.voxel_flags.push_back(voxel.flags);
    }
    asset.edges.reserve(edges.size());
    for (const auto& edge : edges) {
        asset.edges.push_back({make_uint2(edge.a, edge.b), edge.rest_length});
    }
    asset.neighbor_offsets = std::move(offsets);
    asset.neighbors.reserve(neighbors.size());
    for (const auto& neighbor : neighbors) {
        asset.neighbors.push_back({neighbor.vertex, neighbor.edge});
    }
    asset.render_positions.reserve(render_vertices.size());
    asset.render_uvs.reserve(render_vertices.size());
    for (const auto& vertex : render_vertices) {
        asset.render_positions.push_back(make_float3(vertex.x, vertex.y, vertex.z));
        asset.render_uvs.push_back(make_float2(vertex.u, vertex.v));
    }
    asset.render_bindings.reserve(render_vertices.size());
    for (const auto& binding : render_vertices) {
        asset.render_bindings.push_back({
            make_uint4(binding.voxels[0], binding.voxels[1],
                binding.voxels[2], binding.voxels[3]),
            make_float4(binding.weights[0], binding.weights[1],
                binding.weights[2], binding.weights[3])});
    }
    asset.render_triangles.reserve(triangles.size());
    for (const auto& triangle : triangles) {
        asset.render_triangles.push_back(make_uint3(triangle.a, triangle.b, triangle.c));
    }
    validate_soft_body_asset(asset);
    const auto surface_count = static_cast<std::uint32_t>(std::count_if(
        asset.voxel_flags.begin(), asset.voxel_flags.end(), [](std::uint32_t flags) {
            return (flags & soft_body_voxel_surface) != 0U;
        }));
    const auto pinned_count = static_cast<std::uint32_t>(std::count_if(
        asset.voxel_flags.begin(), asset.voxel_flags.end(), [](std::uint32_t flags) {
            return (flags & soft_body_voxel_pinned) != 0U;
        }));
    if (surface_count != header.surface_count || pinned_count != header.pinned_count) {
        throw std::runtime_error("soft-body header flag counts do not match payload: " + path);
    }
    return asset;
}

void save_soft_body_asset(const SoftBodyAsset& asset, const std::string& path)
{
    validate_soft_body_asset(asset);
    if constexpr (std::endian::native != std::endian::little) {
        throw std::runtime_error("soft-body .msb writing currently requires little-endian host");
    }
    FileHeader header{};
    std::copy(file_magic.begin(), file_magic.end(), header.magic);
    header.version = file_version;
    header.endian = file_endian;
    header.header_bytes = sizeof(FileHeader);
    header.voxel_count = checked_count(asset.rest_voxels.size(), "voxel count");
    header.edge_count = checked_count(asset.edges.size(), "edge count");
    header.neighbor_count = checked_count(asset.neighbors.size(), "neighbor count");
    header.render_vertex_count = checked_count(asset.render_positions.size(), "render vertices");
    header.render_triangle_count = checked_count(asset.render_triangles.size(), "render triangles");
    header.surface_count = static_cast<std::uint32_t>(std::count_if(
        asset.voxel_flags.begin(), asset.voxel_flags.end(), [](std::uint32_t flags) {
            return (flags & soft_body_voxel_surface) != 0U;
        }));
    header.pinned_count = static_cast<std::uint32_t>(std::count_if(
        asset.voxel_flags.begin(), asset.voxel_flags.end(), [](std::uint32_t flags) {
            return (flags & soft_body_voxel_pinned) != 0U;
        }));
    header.relaxation_iterations = asset.relaxation_iterations;
    header.flags = asset.file_flags;
    header.nominal_spacing = asset.nominal_spacing;

    std::vector<VoxelRecord> voxels;
    voxels.reserve(asset.rest_voxels.size());
    for (std::size_t i = 0U; i < asset.rest_voxels.size(); ++i) {
        const float3 position = asset.rest_voxels[i];
        voxels.push_back({position.x, position.y, position.z,
            static_cast<std::uint8_t>(asset.voxel_flags[i]), {0U, 0U, 0U}});
    }
    std::vector<EdgeRecord> edges;
    edges.reserve(asset.edges.size());
    for (const auto& edge : asset.edges) {
        edges.push_back({edge.vertices.x, edge.vertices.y, edge.rest_length});
    }
    std::vector<NeighborRecord> neighbors;
    neighbors.reserve(asset.neighbors.size());
    for (const auto& neighbor : asset.neighbors) {
        neighbors.push_back({neighbor.vertex, neighbor.edge});
    }
    std::vector<RenderVertexRecord> render_vertices;
    render_vertices.reserve(asset.render_positions.size());
    for (std::size_t i = 0U; i < asset.render_positions.size(); ++i) {
        const float3 position = asset.render_positions[i];
        const float2 uv = asset.render_uvs[i];
        const auto& binding = asset.render_bindings[i];
        render_vertices.push_back({position.x, position.y, position.z, uv.x, uv.y,
            {binding.voxels.x, binding.voxels.y, binding.voxels.z, binding.voxels.w},
            {binding.weights.x, binding.weights.y, binding.weights.z, binding.weights.w}});
    }
    std::vector<TriangleRecord> triangles;
    triangles.reserve(asset.render_triangles.size());
    for (const auto& triangle : asset.render_triangles) {
        triangles.push_back({triangle.x, triangle.y, triangle.z});
    }

    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot create soft-body asset: " + path);
    output.write(reinterpret_cast<const char*>(&header), sizeof(header));
    write_records(output, voxels, "voxels");
    write_records(output, edges, "edges");
    write_records(output, asset.neighbor_offsets, "neighbor offsets");
    write_records(output, neighbors, "neighbors");
    write_records(output, render_vertices, "render vertices");
    write_records(output, triangles, "render triangles");
}

struct SoftBodyCourse::Impl {
    SoftBodyAsset asset;
    SoftBodyOptions options;
    SoftBodyStatistics statistics;

    float3* position_a{};
    float3* position_b{};
    float3* velocity_a{};
    float3* velocity_b{};
    float3* positions{};
    float3* velocities{};
    float3* next_positions{};
    float3* next_velocities{};
    float3* substep_start_positions{};
    float3* rest_positions{};
    float3* external_impulses{};
    float3* position_corrections{};
    float3* rigid_sphere_impulses{};
    std::uint32_t* cross_contact_keys{};
    float3* cross_contact_values{};
    std::vector<float3> host_rigid_sphere_impulses;
    std::vector<float3> host_contact_positions;
    std::uint32_t* flags{};
    SoftBodyEdge* edges{};
    SoftBodyEdge* presentation_member_edges{};
    std::uint32_t* neighbor_offsets{};
    SoftBodyNeighbor* neighbors{};
    std::uint8_t* active_edges{};
    std::uint8_t* presentation_member_active{};
    std::uint8_t* edge_damage{};
    std::uint64_t* voxel_cell_keys_a{};
    std::uint64_t* voxel_cell_keys_b{};
    std::uint32_t* voxel_cell_indices_a{};
    std::uint32_t* voxel_cell_indices_b{};
    void* voxel_cell_sort_storage{};
    std::size_t voxel_cell_sort_storage_bytes{};
    std::uint32_t* counters{};

    float3* local_rest_voxels{};
    float3* local_render_positions{};
    SoftBodyBinding* render_bindings{};
    float3* render_positions{};
    float2* render_uvs{};
    uint3* render_triangles{};
    uint2* local_render_frame_edges{};
    uint3* local_render_triangle_edges{};
    std::uint8_t* render_triangle_active{};
    meshprep::Aabb* render_bounds{};
    meshprep::Aabb* member_bounds{};

    meshprep::Workspace render_workspace;
    meshprep::Hierarchy render_hierarchy;
    meshprep::Workspace render_normal_workspace;
    meshprep::NormalOutput render_normals;
    meshprep::Workspace member_workspace;
    meshprep::Hierarchy member_hierarchy;
    static constexpr std::uint32_t maximum_substeps = 32U;
    static constexpr std::uint32_t markers_per_substep = 6U;
    cudaEvent_t substep_markers[maximum_substeps * markers_per_substep]{};
    cudaEvent_t frame_markers[3]{};
    cudaEvent_t rigid_contact_begin[maximum_substeps]{};
    cudaEvent_t rigid_contact_end[maximum_substeps]{};
    std::uint32_t completed_substeps{};
    std::uint32_t rigid_contact_substeps{};
    bool frame_open{};
    bool substep_prepared{};

    std::uint32_t total_voxels{};
    std::uint32_t total_edges{};
    std::uint32_t members_per_instance{};
    std::uint32_t total_members{};
    std::uint32_t total_render_vertices{};
    std::uint32_t total_render_triangles{};

    Impl(SoftBodyAsset input_asset, SoftBodyOptions input_options)
        : asset(std::move(input_asset)), options(input_options)
    {
        if (options.preserve_fractured_triangle_shape)
            expand_render_triangles(asset);
        validate_soft_body_asset(asset);
        validate_options(options);
        if (options.require_1000_voxels && asset.rest_voxels.size() != 1'000U) {
            throw std::invalid_argument("course soft-body assets require exactly 1,000 voxels");
        }
        // Four-way translation blending can straddle separated lattice pieces.
        // Each render vertex therefore follows its strongest material voxel,
        // while two bonded supports provide a deterministic local rotation.
        std::vector<uint2> host_frame_edges;
        host_frame_edges.reserve(asset.render_bindings.size());
        for (SoftBodyBinding& binding : asset.render_bindings) {
            const auto original_voxels = binding_voxels(binding);
            const auto original_weights = binding_weight_array(binding);
            unsigned winner = 0U;
            for (unsigned slot = 1U; slot < 4U; ++slot) {
                if (original_weights[slot] > original_weights[winner]) winner = slot;
            }
            const std::uint32_t anchor = original_voxels[winner];
            unsigned best_first = 4U;
            unsigned best_second = 4U;
            float best_area_squared = 0.0F;
            for (unsigned first = 0U; first < 4U; ++first) {
                if (first == winner || find_structural_edge(
                    asset.edges, anchor, original_voxels[first]) == invalid_edge) continue;
                for (unsigned second = first + 1U; second < 4U; ++second) {
                    if (second == winner || find_structural_edge(
                        asset.edges, anchor, original_voxels[second]) == invalid_edge) continue;
                    const float3 a = subtract(
                        asset.rest_voxels[original_voxels[first]], asset.rest_voxels[anchor]);
                    const float3 b = subtract(
                        asset.rest_voxels[original_voxels[second]], asset.rest_voxels[anchor]);
                    const float3 normal = cross(a, b);
                    const float area_squared = dot(normal, normal);
                    if (area_squared > best_area_squared) {
                        best_area_squared = area_squared;
                        best_first = first;
                        best_second = second;
                    }
                }
            }
            if (best_first == 4U || best_second == 4U ||
                !(best_area_squared > 1.0e-12F)) {
                binding.voxels = make_uint4(anchor, anchor, anchor, anchor);
                host_frame_edges.push_back(make_uint2(invalid_edge, invalid_edge));
            } else {
                unsigned remaining = winner;
                for (unsigned slot = 0U; slot < 4U; ++slot) {
                    if (slot != winner && slot != best_first && slot != best_second) {
                        remaining = slot;
                        break;
                    }
                }
                binding.voxels = make_uint4(anchor, original_voxels[best_first],
                    original_voxels[best_second], original_voxels[remaining]);
                host_frame_edges.push_back(make_uint2(
                    find_structural_edge(asset.edges, anchor, original_voxels[best_first]),
                    find_structural_edge(asset.edges, anchor, original_voxels[best_second])));
            }
            binding.weights = make_float4(1.0F, 0.0F, 0.0F, 0.0F);
        }
        std::vector<uint3> host_triangle_edges;
        host_triangle_edges.reserve(asset.render_triangles.size());
        for (const uint3 triangle : asset.render_triangles) {
            const std::uint32_t a = asset.render_bindings[triangle.x].voxels.x;
            const std::uint32_t b = asset.render_bindings[triangle.y].voxels.x;
            const std::uint32_t c = asset.render_bindings[triangle.z].voxels.x;
            host_triangle_edges.push_back(make_uint3(
                find_structural_edge(asset.edges, a, b),
                find_structural_edge(asset.edges, b, c),
                find_structural_edge(asset.edges, c, a)));
        }
        const std::size_t voxel_total = checked_product(
            asset.rest_voxels.size(), options.instance_count, "instance voxels");
        const std::size_t edge_total = checked_product(
            asset.edges.size(), options.instance_count, "instance edges");
        const std::size_t render_vertex_total = checked_product(
            asset.render_positions.size(), options.instance_count, "instance render vertices");
        const std::size_t render_triangle_total = checked_product(
            asset.render_triangles.size(), options.instance_count, "instance render triangles");
        total_voxels = static_cast<std::uint32_t>(voxel_total);
        host_rigid_sphere_impulses.resize(total_voxels);
        host_contact_positions.resize(total_voxels);
        total_edges = static_cast<std::uint32_t>(edge_total);
        members_per_instance = static_cast<std::uint32_t>(
            asset.render_member_edges.empty()
                ? asset.edges.size() : asset.render_member_edges.size());
        total_members = static_cast<std::uint32_t>(checked_product(
            members_per_instance, options.instance_count, "presentation members"));
        total_render_vertices = static_cast<std::uint32_t>(render_vertex_total);
        total_render_triangles = static_cast<std::uint32_t>(render_triangle_total);
        if (options.cross_source_nodes != 0U &&
            (options.cross_source_nodes >= total_voxels ||
             options.cross_target_triangle_first >= total_render_triangles)) {
            throw std::invalid_argument("invalid soft/cloth cross-component ranges");
        }
        if (options.fracture_node_first>asset.rest_voxels.size())
            throw std::invalid_argument("invalid soft-body fracture range");
        if (options.cage_node_count != 0U &&
            (options.instance_count != 1U || options.cage_node_count != 8U ||
             options.cage_first_node + options.cage_node_count > total_voxels)) {
            throw std::invalid_argument("invalid closed rope-cage range");
        }

        std::vector<float3> host_rest_positions;
        std::vector<std::uint32_t> host_flags;
        std::vector<float2> host_uvs;
        std::vector<uint3> host_triangles;
        std::vector<SoftBodyBinding> host_bindings;
        host_rest_positions.reserve(total_voxels);
        host_flags.reserve(total_voxels);
        host_uvs.reserve(total_render_vertices);
        host_triangles.reserve(total_render_triangles);
        host_bindings.reserve(total_render_vertices);
        const float local_render_min_y = std::min_element(asset.render_positions.begin(),
            asset.render_positions.end(), [](float3 a, float3 b) { return a.y < b.y; })->y;
        for (std::uint32_t instance = 0U; instance < options.instance_count; ++instance) {
            float3 origin = options.use_course_layout
                ? course_peg(instance) : options.instance_origins[instance];
            if (options.use_course_layout) origin.y -= local_render_min_y;
            for (std::size_t voxel = 0U; voxel < asset.rest_voxels.size(); ++voxel) {
                host_rest_positions.push_back(add(asset.rest_voxels[voxel], origin));
                host_flags.push_back(asset.voxel_flags[voxel]);
            }
            host_uvs.insert(host_uvs.end(), asset.render_uvs.begin(), asset.render_uvs.end());
            const std::uint32_t voxel_base =
                instance * static_cast<std::uint32_t>(asset.rest_voxels.size());
            for (const SoftBodyBinding binding : asset.render_bindings) {
                host_bindings.push_back({make_uint4(
                    voxel_base + binding.voxels.x, voxel_base + binding.voxels.y,
                    voxel_base + binding.voxels.z, voxel_base + binding.voxels.w),
                    binding.weights});
            }
            const std::uint32_t vertex_base =
                instance * static_cast<std::uint32_t>(asset.render_positions.size());
            for (const uint3 triangle : asset.render_triangles) {
                host_triangles.push_back(make_uint3(
                    vertex_base + triangle.x, vertex_base + triangle.y, vertex_base + triangle.z));
            }
        }
        const std::uint32_t surface_voxels_per_instance = static_cast<std::uint32_t>(
            std::count_if(asset.voxel_flags.begin(), asset.voxel_flags.end(),
                [](std::uint32_t flags) {
                    return (flags & soft_body_voxel_surface) != 0U;
                }));
        const std::uint32_t total_surface_voxels = static_cast<std::uint32_t>(checked_product(
            surface_voxels_per_instance, options.instance_count, "surface voxels"));

        statistics = {options.instance_count,
            static_cast<std::uint32_t>(asset.rest_voxels.size()), total_voxels,
            total_surface_voxels, static_cast<std::uint32_t>(asset.edges.size()),
            total_edges, 0U, 0U, 0U};

        try {
            allocate(position_a, total_voxels);
            allocate(position_b, total_voxels);
            allocate(velocity_a, total_voxels);
            allocate(velocity_b, total_voxels);
            allocate(substep_start_positions, total_voxels);
            allocate(rest_positions, total_voxels);
            allocate(external_impulses, total_voxels);
            allocate(position_corrections, total_voxels);
            allocate(rigid_sphere_impulses, total_voxels);
            if (options.cross_source_nodes != 0U) {
                allocate(cross_contact_keys, 3U * options.cross_source_nodes);
                allocate(cross_contact_values, 3U * options.cross_source_nodes);
            }
            allocate(flags, total_voxels);
            allocate(edges, asset.edges.size());
            allocate(neighbor_offsets, asset.neighbor_offsets.size());
            allocate(neighbors, asset.neighbors.size());
            allocate(active_edges, total_edges);
            if (!asset.render_member_edges.empty()) {
                allocate(presentation_member_edges, asset.render_member_edges.size());
                allocate(presentation_member_active, total_members);
            }
            allocate(edge_damage, total_edges);
            if (options.unbonded_voxel_collisions) {
                allocate(voxel_cell_keys_a, total_voxels);
                allocate(voxel_cell_keys_b, total_voxels);
                allocate(voxel_cell_indices_a, total_voxels);
                allocate(voxel_cell_indices_b, total_voxels);
                check(cub::DeviceRadixSort::SortPairs(nullptr,
                    voxel_cell_sort_storage_bytes,
                    voxel_cell_keys_a, voxel_cell_keys_b,
                    voxel_cell_indices_a, voxel_cell_indices_b,
                    static_cast<int>(total_voxels)),
                    "size voxel-cell sort workspace");
                check(cudaMalloc(&voxel_cell_sort_storage,
                    voxel_cell_sort_storage_bytes),
                    "allocate voxel-cell sort workspace");
            }
            allocate(counters, 2U);
            allocate(local_rest_voxels, asset.rest_voxels.size());
            allocate(local_render_positions, asset.render_positions.size());
            allocate(render_bindings, total_render_vertices);
            allocate(render_positions, total_render_vertices);
            allocate(render_uvs, total_render_vertices);
            allocate(render_triangles, total_render_triangles);
            allocate(local_render_frame_edges, asset.render_positions.size());
            if (options.preserve_fractured_triangle_shape)
                allocate(local_render_triangle_edges, asset.render_triangles.size());
            allocate(render_triangle_active, total_render_triangles);
            allocate(render_bounds, total_render_triangles);
            if (options.render_internal_members) allocate(member_bounds, total_members);
            upload(rest_positions, host_rest_positions);
            upload(flags, host_flags);
            upload(edges, asset.edges);
            if (!asset.render_member_edges.empty()) {
                upload(presentation_member_edges, asset.render_member_edges);
                check(cudaMemset(presentation_member_active, 1,
                    total_members * sizeof(std::uint8_t)),
                    "initialize presentation member activity");
            }
            upload(neighbor_offsets, asset.neighbor_offsets);
            upload(neighbors, asset.neighbors);
            upload(local_rest_voxels, asset.rest_voxels);
            upload(local_render_positions, asset.render_positions);
            upload(render_bindings, host_bindings);
            upload(render_uvs, host_uvs);
            upload(render_triangles, host_triangles);
            upload(local_render_frame_edges, host_frame_edges);
            if (options.preserve_fractured_triangle_shape)
                upload(local_render_triangle_edges, host_triangle_edges);
            for (auto& marker : substep_markers) {
                check(cudaEventCreate(&marker), "create soft-body substep event");
            }
            for (auto& marker : rigid_contact_begin)
                check(cudaEventCreate(&marker), "create rigid contact begin event");
            for (auto& marker : rigid_contact_end)
                check(cudaEventCreate(&marker), "create rigid contact end event");
            for (auto& marker : frame_markers) {
                check(cudaEventCreate(&marker), "create soft-body frame event");
            }
            positions = position_a;
            next_positions = position_b;
            velocities = velocity_a;
            next_velocities = velocity_b;
            initialize(host_rest_positions);
        } catch (...) {
            release();
            throw;
        }
    }

    ~Impl() { release(); }

    void release() noexcept
    {
        for (auto& marker : rigid_contact_begin) {
            if (marker) cudaEventDestroy(marker);
            marker = nullptr;
        }
        for (auto& marker : rigid_contact_end) {
            if (marker) cudaEventDestroy(marker);
            marker = nullptr;
        }
        for (auto& marker : frame_markers) {
            if (marker) cudaEventDestroy(marker);
            marker = nullptr;
        }
        for (auto& marker : substep_markers) {
            if (marker) cudaEventDestroy(marker);
            marker = nullptr;
        }
        cudaFree(member_bounds);
        cudaFree(presentation_member_active);
        cudaFree(presentation_member_edges);
        cudaFree(render_bounds);
        cudaFree(render_triangle_active);
        cudaFree(local_render_triangle_edges);
        cudaFree(local_render_frame_edges);
        cudaFree(render_triangles);
        cudaFree(render_uvs);
        cudaFree(render_positions);
        cudaFree(render_bindings);
        cudaFree(local_render_positions);
        cudaFree(local_rest_voxels);
        cudaFree(counters);
        cudaFree(voxel_cell_sort_storage);
        cudaFree(voxel_cell_indices_b);
        cudaFree(voxel_cell_indices_a);
        cudaFree(voxel_cell_keys_b);
        cudaFree(voxel_cell_keys_a);
        cudaFree(edge_damage);
        cudaFree(active_edges);
        cudaFree(neighbors);
        cudaFree(neighbor_offsets);
        cudaFree(edges);
        cudaFree(flags);
        cudaFree(position_corrections);
        cudaFree(rigid_sphere_impulses);
        cudaFree(cross_contact_keys);
        cudaFree(cross_contact_values);
        cudaFree(external_impulses);
        cudaFree(rest_positions);
        cudaFree(substep_start_positions);
        cudaFree(velocity_b);
        cudaFree(velocity_a);
        cudaFree(position_b);
        cudaFree(position_a);
        render_bounds = nullptr;
        member_bounds = nullptr;
        presentation_member_active = nullptr;
        presentation_member_edges = nullptr;
        render_triangle_active = nullptr;
        local_render_frame_edges = nullptr;
        local_render_triangle_edges = nullptr;
        render_triangles = nullptr;
        render_uvs = nullptr;
        render_positions = nullptr;
        render_bindings = nullptr;
        local_render_positions = nullptr;
        local_rest_voxels = nullptr;
        counters = nullptr;
        edge_damage = nullptr;
        active_edges = nullptr;
        voxel_cell_sort_storage = nullptr;
        voxel_cell_indices_b = nullptr;
        voxel_cell_indices_a = nullptr;
        voxel_cell_keys_b = nullptr;
        voxel_cell_keys_a = nullptr;
        neighbors = nullptr;
        neighbor_offsets = nullptr;
        edges = nullptr;
        flags = nullptr;
        position_corrections = nullptr;
        external_impulses = nullptr;
        rest_positions = nullptr;
        substep_start_positions = nullptr;
        velocity_b = nullptr;
        velocity_a = nullptr;
        position_b = nullptr;
        position_a = nullptr;
    }

    void initialize(const std::vector<float3>& host_rest_positions)
    {
        upload(position_a, host_rest_positions);
        upload(position_b, host_rest_positions);
        check(cudaMemset(velocity_a, 0, static_cast<std::size_t>(total_voxels) * sizeof(float3)),
            "clear soft-body velocities A");
        check(cudaMemset(velocity_b, 0, static_cast<std::size_t>(total_voxels) * sizeof(float3)),
            "clear soft-body velocities B");
        check(cudaMemset(external_impulses, 0,
            static_cast<std::size_t>(total_voxels) * sizeof(float3)),
            "clear soft-body external impulses");
        check(cudaMemset(position_corrections, 0,
            static_cast<std::size_t>(total_voxels) * sizeof(float3)),
            "clear soft-body position corrections");
        check(cudaMemset(active_edges, 1, total_edges * sizeof(std::uint8_t)),
            "restore soft-body edges");
        check(cudaMemset(edge_damage, 0, total_edges * sizeof(std::uint8_t)),
            "clear soft-body fracture damage");
        check(cudaMemset(render_triangle_active, 1,
            static_cast<std::size_t>(total_render_triangles) * sizeof(std::uint8_t)),
            "restore soft-body render triangles");
        check(cudaMemset(counters, 0, 2U * sizeof(std::uint32_t)),
            "clear soft-body counters");
        positions = position_a;
        next_positions = position_b;
        velocities = velocity_a;
        next_velocities = velocity_b;
        enqueue_render(nullptr);
        check(cudaDeviceSynchronize(), "initialize soft-body derived data");
        check(meshprep::build_hierarchy({render_bounds, total_render_triangles},
            {options.hierarchy_leaf_size}, render_workspace, render_hierarchy),
            "build soft-body render hierarchy");
        update_render_normals(nullptr);
        if (options.render_internal_members) {
            check(meshprep::build_hierarchy({member_bounds, total_members},
                {options.hierarchy_leaf_size}, member_workspace, member_hierarchy),
                "build soft-body member hierarchy");
        }
        statistics.broken_edge_count = 0U;
        statistics.finite_failure_count = 0U;
        statistics.frame_index = 0U;
        frame_open = false;
        substep_prepared = false;
        completed_substeps = 0U;
    }

    void enqueue_render(cudaStream_t stream)
    {
        deform_render_vertices<<<(total_render_vertices + block_size - 1U) / block_size,
            block_size, 0, stream>>>(positions, local_rest_voxels, local_render_positions,
            render_bindings, static_cast<std::uint32_t>(asset.rest_voxels.size()),
            static_cast<std::uint32_t>(asset.render_positions.size()),
            total_render_vertices, local_render_frame_edges, active_edges,
            static_cast<std::uint32_t>(asset.edges.size()), render_positions);
        check(cudaGetLastError(), "launch soft-body render deformation");
        if (options.preserve_fractured_triangle_shape) {
            preserve_fractured_triangles<<<
                (total_render_triangles + block_size - 1U) / block_size,
                block_size, 0, stream>>>(render_positions, local_render_positions,
                render_triangles, local_render_triangle_edges, active_edges,
                static_cast<std::uint32_t>(asset.render_triangles.size()),
                static_cast<std::uint32_t>(asset.render_positions.size()),
                static_cast<std::uint32_t>(asset.edges.size()),
                total_render_triangles);
            check(cudaGetLastError(), "preserve fractured render triangle shape");
        }
        // Fracture changes the structural graph, not material coverage. Each
        // render vertex remains bound to its dominant surviving voxel, so a
        // torn face stays attached by an edge or vertex instead of vanishing
        // and exposing large artificial holes.
        check(cudaMemsetAsync(render_triangle_active, 1,
            static_cast<std::size_t>(total_render_triangles) * sizeof(std::uint8_t),
            stream), "preserve soft-body render triangles");
        emit_triangle_bounds<<<(total_render_triangles + block_size - 1U) / block_size,
            block_size, 0, stream>>>(render_positions, render_triangles,
            total_render_triangles, render_bounds);
        check(cudaGetLastError(), "launch soft-body render bounds");
        if (options.render_internal_members) {
            const SoftBodyEdge* member_edges = asset.render_member_edges.empty()
                ? edges : presentation_member_edges;
            const std::uint8_t* member_active = asset.render_member_edges.empty()
                ? active_edges : presentation_member_active;
            emit_member_bounds<<<(total_members + block_size - 1U) / block_size,
                block_size, 0, stream>>>(positions, member_edges, member_active,
                static_cast<std::uint32_t>(asset.rest_voxels.size()),
                members_per_instance, total_members,
                0.11F * asset.voxel_radius, member_bounds);
            check(cudaGetLastError(), "launch soft-body member bounds");
        }
    }

    void update_render_normals(cudaStream_t stream)
    {
        check(meshprep::compute_normals(
            {render_positions, total_render_vertices,
             render_triangles, total_render_triangles}, {},
            render_normal_workspace, render_normals, stream),
            "compute soft-body render normals");
    }

    void refit_members(cudaStream_t stream)
    {
        if (!options.render_internal_members) return;
        check(meshprep::refit_hierarchy_unchecked_async(
            {member_bounds, total_members}, member_hierarchy, stream),
            "refit soft-body member hierarchy");
    }
};

SoftBodyCourse::SoftBodyCourse(SoftBodyAsset asset, SoftBodyOptions options)
    : impl_(std::make_unique<Impl>(std::move(asset), options))
{
}

SoftBodyCourse::SoftBodyCourse(const std::string& asset_path, SoftBodyOptions options)
    : SoftBodyCourse(load_soft_body_asset(asset_path), options)
{
}

SoftBodyCourse::~SoftBodyCourse() = default;
SoftBodyCourse::SoftBodyCourse(SoftBodyCourse&&) noexcept = default;
SoftBodyCourse& SoftBodyCourse::operator=(SoftBodyCourse&&) noexcept = default;

void SoftBodyCourse::clear_external_impulses(cudaStream_t stream)
{
    if (!impl_) throw std::logic_error("moved-from soft-body course");
    check(cudaMemsetAsync(impl_->external_impulses, 0,
        static_cast<std::size_t>(impl_->total_voxels) * sizeof(float3), stream),
        "clear soft-body external impulses");
    check(cudaMemsetAsync(impl_->position_corrections, 0,
        static_cast<std::size_t>(impl_->total_voxels) * sizeof(float3), stream),
        "clear soft-body position corrections");
}

void SoftBodyCourse::clear_external_forces(cudaStream_t stream)
{
    // Compatibility spelling for early integration code. The buffer semantics
    // are impulses; new callers should use clear_external_impulses().
    clear_external_impulses(stream);
}

void SoftBodyCourse::begin_frame(cudaStream_t stream)
{
    if (!impl_) throw std::logic_error("moved-from soft-body course");
    auto& state = *impl_;
    if (state.frame_open) throw std::logic_error("soft-body frame is already open");
    check(cudaMemsetAsync(state.counters, 0, 2U * sizeof(std::uint32_t), stream),
        "clear soft-body frame counters");
    state.completed_substeps = 0U;
    state.rigid_contact_substeps = 0U;
    state.frame_open = true;
    state.substep_prepared = false;
}

void SoftBodyCourse::prepare_substep(float dt, float3 gravity, cudaStream_t stream)
{
    if (!impl_) throw std::logic_error("moved-from soft-body course");
    if (!finite(dt) || dt <= 0.0F || !finite(gravity)) {
        throw std::invalid_argument("invalid soft-body substep time or gravity");
    }
    auto& state = *impl_;
    if (!state.frame_open || state.substep_prepared ||
        state.completed_substeps >= Impl::maximum_substeps) {
        throw std::logic_error("invalid soft-body prepare_substep sequence");
    }
    const std::uint32_t marker =
        state.completed_substeps * Impl::markers_per_substep;
    check(cudaEventRecord(state.substep_markers[marker], stream),
        "begin soft-body substep prediction");
    predict_voxels<<<(state.total_voxels + block_size - 1U) / block_size,
        block_size, 0, stream>>>(state.positions, state.velocities,
        state.next_positions, state.next_velocities, state.substep_start_positions,
        state.rest_positions, state.flags, state.total_voxels, gravity, dt,
        state.options.velocity_damping, state.options.maximum_speed, state.counters);
    check(cudaGetLastError(), "launch soft-body prediction");
    std::swap(state.positions, state.next_positions);
    std::swap(state.velocities, state.next_velocities);
    clear_external_impulses(stream);
    check(cudaEventRecord(state.substep_markers[marker + 1U], stream),
        "end soft-body substep prediction");

    state.enqueue_render(stream);
    check(cudaEventRecord(state.substep_markers[marker + 2U], stream),
        "end soft-body substep render deformation");
    check(meshprep::refit_hierarchy_unchecked_async(
        {state.render_bounds, state.total_render_triangles}, state.render_hierarchy, stream),
        "refit soft-body substep render hierarchy");
    state.refit_members(stream);
    check(cudaEventRecord(state.substep_markers[marker + 3U], stream),
        "end soft-body substep render hierarchy");
    state.substep_prepared = true;
}

void SoftBodyCourse::finish_substep(float dt, float3 gravity, cudaStream_t stream)
{
    if (!impl_) throw std::logic_error("moved-from soft-body course");
    if (!finite(dt) || dt <= 0.0F || !finite(gravity)) {
        throw std::invalid_argument("invalid soft-body substep time or gravity");
    }
    auto& state = *impl_;
    if (!state.frame_open || !state.substep_prepared) {
        throw std::logic_error("soft-body finish_substep has no prepared substep");
    }
    const std::uint32_t marker =
        state.completed_substeps * Impl::markers_per_substep;
    check(cudaEventRecord(state.substep_markers[marker + 4U], stream),
        "begin soft-body contact application");
    apply_contact_impulses<<<(state.total_voxels + block_size - 1U) / block_size,
        block_size, 0, stream>>>(state.positions, state.velocities,
        state.rest_positions,state.substep_start_positions,state.flags,
        state.external_impulses,
        state.position_corrections, state.total_voxels, dt,
        1.0F / state.options.voxel_mass, state.options.maximum_speed,
        state.asset.voxel_radius, state.options.course_board_collisions,
        state.options.arena, state.options.ground_friction,
        state.counters);
    check(cudaGetLastError(), "launch soft-body contact application");
    if (state.options.cross_source_nodes != 0U) {
        const std::uint32_t source_count = state.options.cross_source_nodes;
        const std::uint32_t target_first =
            state.options.cross_target_triangle_first;
        contact_soft_nodes_against_mesh_kernel<<<
            (source_count + block_size - 1U) / block_size,
            block_size, 0, stream>>>(state.positions, state.velocities,
            state.substep_start_positions, state.flags, source_count,
            state.render_positions, state.render_triangles, state.render_bindings,
            target_first, state.total_render_triangles - target_first,
            1.5F * state.asset.voxel_radius,
            2.0F * state.asset.voxel_radius,
            dt,1.0F/(state.options.voxel_mass*
                state.options.cross_source_mass_multiplier),
            1.0F/state.options.voxel_mass,
            state.options.maximum_speed, state.cross_contact_keys,
            state.cross_contact_values);
        check(cudaGetLastError(), "generate soft sphere/cloth contacts");
        gather_mesh_contact_kernel<<<
            (state.total_voxels - source_count + block_size - 1U) / block_size,
            block_size, 0, stream>>>(state.positions, state.velocities,
            state.flags, source_count, state.total_voxels - source_count,
            state.cross_contact_keys, state.cross_contact_values,
            3U * source_count, state.asset.voxel_radius,
            dt, state.options.maximum_speed);
        check(cudaGetLastError(), "gather soft sphere/cloth reactions");
    }

    // Observe impact strain before the spring projection. Sampling only the
    // converged positions made extra solve iterations erase the strain that
    // should tear cloth, so a heavy sphere could phase through an apparently
    // indestructible sheet.
    if (state.options.fracture_before_projection) {
        find_broken_edges<<<(state.total_edges + block_size - 1U) / block_size,
            block_size, 0, stream>>>(state.positions, state.edges, state.active_edges,
            state.edge_damage,
            static_cast<std::uint32_t>(state.asset.rest_voxels.size()),
            static_cast<std::uint32_t>(state.asset.edges.size()), state.total_edges,
            state.options.fracture_node_first,
            state.options.break_strain * state.options.strength_multiplier,
            state.options.fracture_persistence_substeps, state.counters);
        check(cudaGetLastError(), "sample pre-projection soft-body fracture");
    }

    // Contact and springs form one position solve. Spreading the gathered
    // contact displacement through the fixed graph before velocity recovery
    // avoids leaving a large one-voxel kick for the next substep.
    const float stiffness = state.options.spring_stiffness;
    // Map the complete live range onto a complete Jacobi response. The old
    // min(1, k*dt^2/m) plateaued near the bottom of the UI; an asymptotic map
    // never reached the known stable full-projection endpoint. Every value is
    // distinct below 40k, and 40k is intentionally exact rather than 0.91.
    constexpr float full_response_stiffness = 40'000.0F;
    const float relaxation = fminf(1.0F,
        sqrtf(stiffness / full_response_stiffness));
    // Values above the one-pass full-response point buy additional Jacobi
    // convergence instead of disappearing into a clamp. This makes the live
    // stiffness control materially strengthen load-bearing volumes while the
    // historical <=40k range retains its calibrated behavior.
    const float solve_scale = fmaxf(1.0F, stiffness / full_response_stiffness);
    const std::uint32_t solve_iterations = static_cast<std::uint32_t>(ceilf(
        solve_scale * static_cast<float>(state.options.spring_solver_iterations)));
    const float maximum_projection =
        state.options.maximum_projection_fraction * state.asset.nominal_spacing;
    for (std::uint32_t iteration = 0U;
         iteration < solve_iterations; ++iteration) {
        project_spring_constraints<<<
            (state.total_voxels + block_size - 1U) / block_size,
            block_size, 0, stream>>>(state.positions, state.next_positions,
            state.rest_positions, state.flags, state.neighbor_offsets,
            state.neighbors, state.edges, state.active_edges,
            static_cast<std::uint32_t>(state.asset.rest_voxels.size()),
            static_cast<std::uint32_t>(state.asset.edges.size()),
            state.total_voxels, relaxation, maximum_projection);
        check(cudaGetLastError(), "launch post-contact soft-body spring projection");
        std::swap(state.positions, state.next_positions);
    }
    finalize_constraint_velocities<<<
        (state.total_voxels + block_size - 1U) / block_size,
        block_size, 0, stream>>>(state.positions, state.substep_start_positions,
        state.velocities, state.flags, state.next_velocities,
        state.total_voxels, 1.0F / dt,
        state.options.constraint_velocity_response, state.options.maximum_speed,
        state.counters);
    check(cudaGetLastError(), "launch post-contact soft-body velocity reconstruction");
    std::swap(state.velocities, state.next_velocities);
    const float damping_response = 1.0F - expf(
        -state.options.spring_damping_ratio *
        sqrtf(stiffness / state.options.voxel_mass) * dt);
    damp_spring_velocities<<<
        (state.total_voxels + block_size - 1U) / block_size,
        block_size, 0, stream>>>(state.positions, state.velocities,
        state.next_velocities, state.flags, state.neighbor_offsets,
        state.neighbors, state.active_edges,
        static_cast<std::uint32_t>(state.asset.rest_voxels.size()),
        static_cast<std::uint32_t>(state.asset.edges.size()),
        state.total_voxels, damping_response, state.options.maximum_speed);
    check(cudaGetLastError(), "launch post-contact soft-body velocity damping");
    std::swap(state.velocities, state.next_velocities);

    if (state.options.unbonded_voxel_collisions) {
        // Keep this a unilateral position barrier. Feeding its projection back
        // through velocity reconstruction produced an artificial rebound in
        // the detached-mesh fixture (0.040 -> 0.376 separation in six ticks).
        emit_voxel_cells<<<
            (state.total_voxels + block_size - 1U) / block_size,
            block_size, 0, stream>>>(state.positions,
            state.voxel_cell_keys_a, state.voxel_cell_indices_a,
            state.total_voxels, 1.0F / (2.0F * state.asset.voxel_radius));
        check(cudaGetLastError(), "emit voxel collision cells");
        check(cub::DeviceRadixSort::SortPairs(
            state.voxel_cell_sort_storage, state.voxel_cell_sort_storage_bytes,
            state.voxel_cell_keys_a, state.voxel_cell_keys_b,
            state.voxel_cell_indices_a, state.voxel_cell_indices_b,
            static_cast<int>(state.total_voxels), 0, 64, stream),
            "sort voxel collision cells");
        project_unbonded_voxel_contacts<<<
            (state.total_voxels + block_size - 1U) / block_size,
            block_size, 0, stream>>>(state.positions, state.next_positions,
            state.flags, state.neighbor_offsets, state.neighbors,
            state.active_edges, state.voxel_cell_keys_b,
            state.voxel_cell_indices_b,
            static_cast<std::uint32_t>(state.asset.rest_voxels.size()),
            static_cast<std::uint32_t>(state.asset.edges.size()),
            state.total_voxels, state.asset.voxel_radius,
            maximum_projection);
        check(cudaGetLastError(), "project detached voxel contacts");
        std::swap(state.positions, state.next_positions);
    }

    apply_post_constraint_ground_friction<<<
        (state.total_voxels + block_size - 1U) / block_size,
        block_size, 0, stream>>>(state.positions, state.velocities, state.flags,
        state.total_voxels, state.options.arena, state.asset.voxel_radius,
        state.options.ground_friction, fabsf(gravity.y), dt);
    check(cudaGetLastError(), "apply post-constraint ground friction");

    if (!state.options.fracture_before_projection) {
        find_broken_edges<<<(state.total_edges + block_size - 1U) / block_size,
            block_size, 0, stream>>>(state.positions, state.edges, state.active_edges,
            state.edge_damage,
            static_cast<std::uint32_t>(state.asset.rest_voxels.size()),
            static_cast<std::uint32_t>(state.asset.edges.size()), state.total_edges,
            state.options.fracture_node_first,
            state.options.break_strain * state.options.strength_multiplier,
            state.options.fracture_persistence_substeps, state.counters);
        check(cudaGetLastError(), "sample residual soft-body fracture");
    }

    check(cudaEventRecord(state.substep_markers[marker + 5U], stream),
        "end soft-body contact application");
    state.substep_prepared = false;
    ++state.completed_substeps;
}

SoftBodyTimings SoftBodyCourse::finish_frame(cudaStream_t stream)
{
    if (!impl_) throw std::logic_error("moved-from soft-body course");
    auto& state = *impl_;
    if (!state.frame_open || state.substep_prepared || state.completed_substeps == 0U) {
        throw std::logic_error("soft-body finish_frame requires completed substeps");
    }
    check(cudaEventRecord(state.frame_markers[0], stream),
        "begin final soft-body render deformation");
    state.enqueue_render(stream);
    state.update_render_normals(stream);
    check(cudaEventRecord(state.frame_markers[1], stream),
        "end final soft-body render deformation");
    check(meshprep::refit_hierarchy_unchecked_async(
        {state.render_bounds, state.total_render_triangles}, state.render_hierarchy, stream),
        "refit final soft-body render hierarchy");
    state.refit_members(stream);
    check(cudaEventRecord(state.frame_markers[2], stream),
        "end final soft-body render hierarchy");

    std::array<std::uint32_t, 2> counters{};
    check(cudaMemcpyAsync(counters.data(), state.counters, sizeof(counters),
        cudaMemcpyDeviceToHost, stream), "read soft-body counters");
    check(cudaEventSynchronize(state.frame_markers[2]), "complete soft-body frame");
    SoftBodyTimings timings{};
    for (std::uint32_t substep = 0U; substep < state.completed_substeps; ++substep) {
        const std::uint32_t marker = substep * Impl::markers_per_substep;
        timings.physics_ms += elapsed(state.substep_markers[marker],
            state.substep_markers[marker + 1U]);
        timings.render_deformation_ms += elapsed(state.substep_markers[marker + 1U],
            state.substep_markers[marker + 2U]);
        timings.render_hierarchy_ms += elapsed(state.substep_markers[marker + 2U],
            state.substep_markers[marker + 3U]);
        timings.physics_ms += elapsed(state.substep_markers[marker + 4U],
            state.substep_markers[marker + 5U]);
    }
    timings.render_deformation_ms += elapsed(
        state.frame_markers[0], state.frame_markers[1]);
    timings.render_hierarchy_ms += elapsed(
        state.frame_markers[1], state.frame_markers[2]);
    for (std::uint32_t substep = 0U;
         substep < state.rigid_contact_substeps; ++substep) {
        timings.rigid_contact_ms += elapsed(
            state.rigid_contact_begin[substep], state.rigid_contact_end[substep]);
    }
    state.statistics.broken_edge_count += counters[0];
    state.statistics.finite_failure_count += counters[1];
    ++state.statistics.frame_index;
    state.frame_open = false;
    return timings;
}

SoftBodyTimings SoftBodyCourse::step(float3 gravity, cudaStream_t stream)
{
    if (!impl_) throw std::logic_error("moved-from soft-body course");
    begin_frame(stream);
    const float dt = impl_->options.fixed_dt /
        static_cast<float>(impl_->options.solver_substeps);
    for (std::uint32_t substep = 0U; substep < impl_->options.solver_substeps; ++substep) {
        prepare_substep(dt, gravity, stream);
        finish_substep(dt, gravity, stream);
    }
    return finish_frame(stream);
}

void SoftBodyCourse::contact_rigid_sphere_substep(
    RigidSphereState& sphere, float dt, cudaStream_t stream)
{
    if (!impl_) throw std::logic_error("moved-from soft-body course");
    if (!finite(sphere.center) || !finite(sphere.velocity) ||
        !finite(sphere.angular_velocity) || !finite(sphere.orientation) ||
        !finite(sphere.radius) || !finite(sphere.mass) ||
        sphere.radius <= 0.0F || sphere.mass <= 0.0F ||
        !finite(dt) || dt <= 0.0F) {
        throw std::invalid_argument("invalid rolling rigid sphere contact");
    }
    auto& state = *impl_;
    if (!state.frame_open || !state.substep_prepared ||
        state.completed_substeps >= Impl::maximum_substeps) {
        throw std::logic_error("rigid sphere contact requires a prepared substep");
    }
    const std::uint32_t substep = state.completed_substeps;
    check(cudaEventRecord(state.rigid_contact_begin[substep], stream),
        "record rigid sphere contact begin");
    contact_rigid_sphere_kernel<<<
        (state.total_voxels + block_size - 1U) / block_size,
        block_size, 0, stream>>>(state.positions, state.velocities,
        state.flags, state.total_voxels, sphere, state.asset.voxel_radius,
        state.options.voxel_mass, dt, state.options.maximum_speed,
        state.options.ground_friction,
        state.options.arena == GalleryArena::enclosed_box,
        state.rigid_sphere_impulses);
    check(cudaGetLastError(), "launch rigid sphere/mesh contact");
    if (state.options.fracture_before_projection) {
        fracture_edges_from_rigid_impact<<<
            (state.total_edges+block_size-1U)/block_size,block_size,0,stream>>>(
            state.edges,state.rigid_sphere_impulses,state.active_edges,
            static_cast<std::uint32_t>(state.asset.rest_voxels.size()),
            static_cast<std::uint32_t>(state.asset.edges.size()),state.total_edges,
            0.10F*state.options.voxel_mass*state.options.strength_multiplier,
            state.counters);
        check(cudaGetLastError(),"fracture cloth edges from rigid impact");
    }
    check(cudaEventRecord(state.rigid_contact_end[substep], stream),
        "record rigid sphere contact end");
    state.rigid_contact_substeps = std::max(
        state.rigid_contact_substeps, substep + 1U);
    check(cudaMemcpyAsync(state.host_rigid_sphere_impulses.data(),
        state.rigid_sphere_impulses,
        static_cast<std::size_t>(state.total_voxels) * sizeof(float3),
        cudaMemcpyDeviceToHost, stream), "download deterministic sphere reactions");
    check(cudaMemcpyAsync(state.host_contact_positions.data(),state.positions,
        static_cast<std::size_t>(state.total_voxels)*sizeof(float3),
        cudaMemcpyDeviceToHost,stream),"download sphere contact positions");
    check(cudaStreamSynchronize(stream), "finish sphere contact reactions");
    float3 total_impulse{};
    float3 total_torque{};
    for (std::size_t node=0U;node<state.host_rigid_sphere_impulses.size();++node) {
        const float3 impulse=state.host_rigid_sphere_impulses[node];
        total_impulse = add(total_impulse, impulse);
        total_torque=add(total_torque,cross(
            subtract(state.host_contact_positions[node],sphere.center),impulse));
    }
    // The node kernel resolves penetration at position level. Apply the same
    // deterministic reaction to the finite-mass sphere position as well as
    // its velocity; velocity-only response allowed a heavy sphere to remain
    // geometrically past the cloth and then fall through on the next substep.
    // A fixed rope-bridge diagnostic is resolved by the one-sided tile
    // support below. Summing one full pinned-node impulse per nearby tile
    // corner multiplied the reaction and was itself able to launch the ball.
    // Dynamic scenes retain their equal-and-opposite node reaction.
    const auto clamp_host=[](float3 value,float maximum) {
        const float magnitude=length(value);
        return magnitude>maximum && magnitude>1.0e-8F
            ? multiply(value,maximum/magnitude) : value;
    };
    total_impulse=clamp_host(total_impulse,sphere.mass*3.0F);
    sphere.center = add(sphere.center,
        multiply(total_impulse, dt / sphere.mass));
    sphere.velocity = add(sphere.velocity,
        multiply(total_impulse, 1.0F / sphere.mass));
    const float inertia=0.4F*sphere.mass*sphere.radius*sphere.radius;
    sphere.angular_velocity=add(sphere.angular_velocity,
        multiply(clamp_host(total_torque,inertia*20.0F),
            1.0F/fmaxf(inertia,1.0e-8F)));
    sphere.angular_velocity=clamp_host(sphere.angular_velocity,12.0F);
    if (state.options.arena == GalleryArena::cloth_basin) {
        // The boat uses a compact spherical physics proxy. Node contacts alone
        // leave triangle-sized holes through which that proxy can tunnel, so
        // close the live cloth surface with the same deformed anchor triangles
        // used by rendering. This remains one-sided: the boat is supported
        // from above and never teleported onto cloth approached from below.
        float support = -std::numeric_limits<float>::infinity();
        for (const uint3 triangle : state.asset.render_triangles) {
            const std::uint32_t ia =
                state.asset.render_bindings[triangle.x].voxels.x;
            const std::uint32_t ib =
                state.asset.render_bindings[triangle.y].voxels.x;
            const std::uint32_t ic =
                state.asset.render_bindings[triangle.z].voxels.x;
            const float3 a = state.host_contact_positions[ia];
            const float3 b = state.host_contact_positions[ib];
            const float3 c = state.host_contact_positions[ic];
            const float determinant =
                (b.z-c.z)*(a.x-c.x)+(c.x-b.x)*(a.z-c.z);
            if (fabsf(determinant) <= 1.0e-8F) continue;
            const float wa = ((b.z-c.z)*(sphere.center.x-c.x)+
                (c.x-b.x)*(sphere.center.z-c.z))/determinant;
            const float wb = ((c.z-a.z)*(sphere.center.x-c.x)+
                (a.x-c.x)*(sphere.center.z-c.z))/determinant;
            const float wc = 1.0F-wa-wb;
            if (fminf(wa,fminf(wb,wc)) < -1.0e-4F) continue;
            support = fmaxf(support,wa*a.y+wb*b.y+wc*c.y);
        }
        if (std::isfinite(support) && sphere.center.y < support+sphere.radius) {
            sphere.center.y = support+sphere.radius;
            if (sphere.velocity.y < 0.0F) sphere.velocity.y = 0.0F;
        }
    }
    if (state.options.arena == GalleryArena::rope_bridge) {
        // Node spheres transfer force to the ropes; this one-sided tile pass
        // closes the square interiors so a fast sphere cannot tunnel between
        // four corner samples. It follows the deformed tile heights.
        float support = -std::numeric_limits<float>::infinity();
        const std::uint32_t corners=state.options.rope_bridge_nodes_per_tile;
        const float margin = 0.70F*sphere.radius;
        for (std::uint32_t tile=0U;
             tile<state.options.rope_bridge_columns*
                 state.options.rope_bridge_rows;++tile) {
            float minimum_x=std::numeric_limits<float>::infinity();
            float maximum_x=-minimum_x;
            float minimum_z=minimum_x;
            float maximum_z=-minimum_x;
            float average_y{};
            for (std::uint32_t corner=0U;corner<corners;++corner) {
                const float3 point=state.host_contact_positions[tile*corners+corner];
                minimum_x=std::min(minimum_x,point.x);
                maximum_x=std::max(maximum_x,point.x);
                minimum_z=std::min(minimum_z,point.z);
                maximum_z=std::max(maximum_z,point.z);
                average_y+=point.y/static_cast<float>(corners);
            }
            if (sphere.center.x>=minimum_x-margin && sphere.center.x<=maximum_x+margin &&
                sphere.center.z>=minimum_z-margin && sphere.center.z<=maximum_z+margin)
                support=std::max(support,average_y);
        }
        if (std::isfinite(support) && sphere.center.y<support+sphere.radius) {
            sphere.center.y=support+sphere.radius;
            if (sphere.velocity.y<0.0F) sphere.velocity.y=0.0F;
        }
    }
    const float speed = length(sphere.velocity);
    if (speed > 3.0F) sphere.velocity = multiply(sphere.velocity, 3.0F / speed);
}

SoftBodyTimings SoftBodyCourse::step_with_rigid_sphere(
    RigidSphereState& sphere, float3 gravity, cudaStream_t stream)
{
    return step_with_rigid_sphere(sphere, gravity, gravity, stream);
}

SoftBodyTimings SoftBodyCourse::step_with_rigid_sphere(
    RigidSphereState& sphere, float3 body_gravity, float3 sphere_gravity,
    cudaStream_t stream)
{
    if (!impl_) throw std::logic_error("moved-from soft-body course");
    if (!finite(sphere.center) || !finite(sphere.velocity) ||
        !finite(sphere.angular_velocity) || !finite(sphere.orientation) ||
        !finite(sphere.radius) || !finite(sphere.mass) ||
        sphere.radius <= 0.0F || sphere.mass <= 0.0F ||
        !finite(body_gravity) || !finite(sphere_gravity)) {
        throw std::invalid_argument("invalid rolling rigid sphere");
    }
    auto& state = *impl_;
    begin_frame(stream);
    const float dt = state.options.fixed_dt /
        static_cast<float>(state.options.solver_substeps);
    for (std::uint32_t substep = 0U;
         substep < state.options.solver_substeps; ++substep) {
        sphere.velocity = add(sphere.velocity, multiply(sphere_gravity, dt));
        sphere.velocity = multiply(sphere.velocity, 1.0F / (1.0F + 0.25F * dt));
        sphere.center = add(sphere.center, multiply(sphere.velocity, dt));
        const float3 unprojected=sphere.center;
        project_gallery_contact(sphere.center, sphere.velocity,
            sphere.radius, state.options.arena == GalleryArena::none
                ? GalleryArena::ground : state.options.arena);
        apply_rigid_contact_friction(sphere,
            subtract(sphere.center,unprojected),state.options.ground_friction,dt);
        prepare_substep(dt, body_gravity, stream);
        contact_rigid_sphere_substep(sphere, dt, stream);
        advance_rigid_sphere_rotation(sphere,
            state.options.arena == GalleryArena::none
                ? GalleryArena::ground : state.options.arena,
            state.options.ground_friction, dt);
        finish_substep(dt, body_gravity, stream);
    }
    return finish_frame(stream);
}

SoftBodyTimings SoftBodyCourse::step_with_tethered_rigid_sphere(
    RigidSphereState& sphere, std::uint32_t endpoint_node,
    float attachment_distance, float3 gravity, cudaStream_t stream)
{
    if (!impl_) throw std::logic_error("moved-from soft-body course");
    if (!finite(sphere.center) || !finite(sphere.velocity) ||
        !finite(sphere.angular_velocity) || !finite(sphere.orientation) ||
        !finite(sphere.radius) || !finite(sphere.mass) ||
        sphere.radius <= 0.0F || sphere.mass <= 0.0F || !finite(gravity) ||
        !finite(attachment_distance) || attachment_distance <= 0.0F ||
        endpoint_node >= impl_->total_voxels) {
        throw std::invalid_argument("invalid tethered rigid sphere");
    }
    auto& state = *impl_;
    if (endpoint_node >= state.asset.rest_voxels.size()) {
        throw std::invalid_argument("tether endpoint is outside its authored rope");
    }
    begin_frame(stream);
    const float dt = state.options.fixed_dt /
        static_cast<float>(state.options.solver_substeps);
    const float3 rest_reach=subtract(state.asset.rest_voxels[endpoint_node],
        state.asset.rest_voxels[0U]);
    const float maximum_rope_reach=length(rest_reach)+attachment_distance;
    for (std::uint32_t substep = 0U;
         substep < state.options.solver_substeps; ++substep) {
        sphere.velocity = add(sphere.velocity, multiply(gravity, dt));
        sphere.velocity = multiply(sphere.velocity, 1.0F / (1.0F + 0.15F * dt));
        sphere.center = add(sphere.center, multiply(sphere.velocity, dt));
        const float3 unprojected=sphere.center;
        project_gallery_contact(sphere.center, sphere.velocity,
            sphere.radius, state.options.arena);
        apply_rigid_contact_friction(sphere,
            subtract(sphere.center,unprojected),state.options.ground_friction,dt);
        prepare_substep(dt, gravity, stream);
        contact_rigid_sphere_substep(sphere, dt, stream);
        finish_substep(dt, gravity, stream);
        // Attach after graph projection so the committed rope endpoint and
        // sphere agree at the end of every substep.
        tether_rigid_sphere_kernel<<<1, 1, 0, stream>>>(
            state.positions, state.velocities, state.flags, endpoint_node,
            sphere, attachment_distance, state.options.voxel_mass,
            maximum_rope_reach,4.0F * state.asset.voxel_radius,
            dt, state.options.maximum_speed,
            state.rigid_sphere_impulses);
        check(cudaGetLastError(), "launch rope/sphere tether constraint");
        check(cudaMemcpyAsync(state.host_rigid_sphere_impulses.data(),
            state.rigid_sphere_impulses, 2U * sizeof(float3),
            cudaMemcpyDeviceToHost, stream), "download rope/sphere tether reaction");
        check(cudaStreamSynchronize(stream), "finish rope/sphere tether reaction");
        sphere.center = add(sphere.center, state.host_rigid_sphere_impulses[0]);
        sphere.velocity = add(sphere.velocity, state.host_rigid_sphere_impulses[1]);
        project_gallery_contact(sphere.center, sphere.velocity,
            sphere.radius, state.options.arena);
        advance_rigid_sphere_rotation(sphere, state.options.arena,
            state.options.ground_friction, dt);
        const float sphere_speed = length(sphere.velocity);
        if (sphere_speed > 3.0F)
            sphere.velocity = multiply(sphere.velocity, 3.0F / sphere_speed);
    }
    return finish_frame(stream);
}

SoftBodyTimings SoftBodyCourse::step_with_tethered_rigid_spheres(
    RigidSphereState& sphere, std::uint32_t endpoint_node,
    float attachment_distance, RigidSphereState& caged_sphere,
    float3 gravity, cudaStream_t stream)
{
    if (!impl_) throw std::logic_error("moved-from soft-body course");
    const auto valid_sphere=[](const RigidSphereState& value) {
        return finite(value.center) && finite(value.velocity) &&
            finite(value.angular_velocity) && finite(value.orientation) &&
            finite(value.radius) && finite(value.mass) && value.radius>0.0F &&
            value.mass>0.0F;
    };
    if (!valid_sphere(sphere) || !valid_sphere(caged_sphere) || !finite(gravity) ||
        !finite(attachment_distance) || attachment_distance<=0.0F ||
        endpoint_node>=impl_->total_voxels ||
        endpoint_node>=impl_->asset.rest_voxels.size())
        throw std::invalid_argument("invalid paired rope rigid spheres");
    auto& state=*impl_;
    begin_frame(stream);
    const float dt=state.options.fixed_dt/
        static_cast<float>(state.options.solver_substeps);
    const float maximum_rope_reach=length(subtract(
        state.asset.rest_voxels[endpoint_node],state.asset.rest_voxels[0U]))+
        attachment_distance;
    for (std::uint32_t substep=0U;substep<state.options.solver_substeps;++substep) {
        auto integrate=[&](RigidSphereState& value) {
            value.velocity=add(value.velocity,multiply(gravity,dt));
            value.velocity=multiply(value.velocity,1.0F/(1.0F+0.15F*dt));
            value.center=add(value.center,multiply(value.velocity,dt));
            const float3 unprojected=value.center;
            project_gallery_contact(value.center,value.velocity,value.radius,
                state.options.arena);
            apply_rigid_contact_friction(value,subtract(value.center,unprojected),
                state.options.ground_friction,dt);
        };
        integrate(sphere);
        integrate(caged_sphere);
        prepare_substep(dt,gravity,stream);
        contact_rigid_sphere_substep(sphere,dt,stream);
        contact_rigid_sphere_substep(caged_sphere,dt,stream);
        if (state.options.cage_node_count==8U)
            contain_sphere_in_rope_cage(caged_sphere,state.host_contact_positions,
                state.options.cage_first_node,0.10F*state.asset.voxel_radius);
        finish_substep(dt,gravity,stream);
        tether_rigid_sphere_kernel<<<1,1,0,stream>>>(
            state.positions,state.velocities,state.flags,endpoint_node,sphere,
            attachment_distance,state.options.voxel_mass,maximum_rope_reach,
            4.0F*state.asset.voxel_radius,dt,state.options.maximum_speed,
            state.rigid_sphere_impulses);
        check(cudaGetLastError(),"launch paired rope/sphere tether constraint");
        check(cudaMemcpyAsync(state.host_rigid_sphere_impulses.data(),
            state.rigid_sphere_impulses,2U*sizeof(float3),cudaMemcpyDeviceToHost,
            stream),"download paired rope/sphere tether reaction");
        if (state.options.cage_node_count==8U)
            check(cudaMemcpyAsync(state.host_contact_positions.data(),state.positions,
                static_cast<std::size_t>(state.total_voxels)*sizeof(float3),
                cudaMemcpyDeviceToHost,stream),
                "download committed rope cage positions");
        check(cudaStreamSynchronize(stream),
            "finish paired rope/sphere tether reaction");
        if (state.options.cage_node_count==8U)
            contain_sphere_in_rope_cage(caged_sphere,state.host_contact_positions,
                state.options.cage_first_node,0.10F*state.asset.voxel_radius);
        sphere.center=add(sphere.center,state.host_rigid_sphere_impulses[0]);
        sphere.velocity=add(sphere.velocity,state.host_rigid_sphere_impulses[1]);
        project_gallery_contact(sphere.center,sphere.velocity,sphere.radius,
            state.options.arena);
        advance_rigid_sphere_rotation(sphere,state.options.arena,
            state.options.ground_friction,dt);
        advance_rigid_sphere_rotation(caged_sphere,state.options.arena,
            state.options.ground_friction,dt);
        const auto limit_speed=[](RigidSphereState& value) {
            const float speed=length(value.velocity);
            if (speed>3.0F) value.velocity=multiply(value.velocity,3.0F/speed);
        };
        limit_speed(sphere);
        limit_speed(caged_sphere);
    }
    return finish_frame(stream);
}

void SoftBodyCourse::reset(cudaStream_t stream)
{
    if (!impl_) throw std::logic_error("moved-from soft-body course");
    auto& state = *impl_;
    check(cudaMemcpyAsync(state.position_a, state.rest_positions,
        static_cast<std::size_t>(state.total_voxels) * sizeof(float3),
        cudaMemcpyDeviceToDevice, stream), "reset soft-body positions A");
    check(cudaMemcpyAsync(state.position_b, state.rest_positions,
        static_cast<std::size_t>(state.total_voxels) * sizeof(float3),
        cudaMemcpyDeviceToDevice, stream), "reset soft-body positions B");
    check(cudaMemsetAsync(state.velocity_a, 0,
        static_cast<std::size_t>(state.total_voxels) * sizeof(float3), stream),
        "reset soft-body velocities A");
    check(cudaMemsetAsync(state.velocity_b, 0,
        static_cast<std::size_t>(state.total_voxels) * sizeof(float3), stream),
        "reset soft-body velocities B");
    check(cudaMemsetAsync(state.external_impulses, 0,
        static_cast<std::size_t>(state.total_voxels) * sizeof(float3), stream),
        "reset soft-body impulses");
    check(cudaMemsetAsync(state.position_corrections, 0,
        static_cast<std::size_t>(state.total_voxels) * sizeof(float3), stream),
        "reset soft-body position corrections");
    check(cudaMemsetAsync(state.active_edges, 1,
        static_cast<std::size_t>(state.total_edges) * sizeof(std::uint8_t), stream),
        "reset soft-body edges");
    check(cudaMemsetAsync(state.edge_damage, 0,
        static_cast<std::size_t>(state.total_edges) * sizeof(std::uint8_t), stream),
        "reset soft-body fracture damage");
    check(cudaMemsetAsync(state.render_triangle_active, 1,
        static_cast<std::size_t>(state.total_render_triangles) * sizeof(std::uint8_t), stream),
        "reset soft-body render triangles");
    state.positions = state.position_a;
    state.next_positions = state.position_b;
    state.velocities = state.velocity_a;
    state.next_velocities = state.velocity_b;
    state.enqueue_render(stream);
    state.update_render_normals(stream);
    check(meshprep::refit_hierarchy_unchecked_async(
        {state.render_bounds, state.total_render_triangles}, state.render_hierarchy, stream),
        "reset soft-body render hierarchy");
    state.refit_members(stream);
    check(cudaStreamSynchronize(stream), "complete soft-body reset");
    state.statistics.broken_edge_count = 0U;
    state.statistics.finite_failure_count = 0U;
    state.statistics.frame_index = 0U;
    state.frame_open = false;
    state.substep_prepared = false;
    state.completed_substeps = 0U;
}

void SoftBodyCourse::set_pinned_rotation_z(
    float3 center, float angle, cudaStream_t stream)
{
    set_wheel_anchor_rotations(center, angle, angle, stream);
}

void SoftBodyCourse::set_wheel_anchor_rotations(
    float3 center, float axle_angle, float rim_angle, cudaStream_t stream)
{
    if (!impl_) throw std::logic_error("moved-from soft-body course");
    if (!finite(center) || !finite(axle_angle) || !finite(rim_angle) ||
        impl_->options.instance_count != 1U) {
        throw std::invalid_argument(
            "wheel anchor rotation requires one instance and finite input");
    }
    auto& state = *impl_;
    rotate_pinned_rest_positions_kernel<<<
        (state.total_voxels + block_size - 1U) / block_size,
        block_size, 0, stream>>>(state.rest_positions, state.local_rest_voxels,
        state.flags, state.total_voxels, center,
        std::cos(axle_angle), std::sin(axle_angle),
        std::cos(rim_angle), std::sin(rim_angle));
    check(cudaGetLastError(), "rotate pinned soft-body wheel anchors");
}

float SoftBodyCourse::wheel_rim_reaction_torque(
    float3 center, cudaStream_t stream)
{
    if (!impl_) throw std::logic_error("moved-from soft-body course");
    if (!finite(center) || impl_->options.instance_count != 1U)
        throw std::invalid_argument("wheel torque requires one instance and finite center");
    auto& state = *impl_;
    wheel_rim_reaction_kernel<<<
        (state.total_voxels + block_size - 1U) / block_size,
        block_size, 0, stream>>>(state.positions, state.flags,
        state.neighbor_offsets, state.neighbors, state.edges, state.active_edges,
        state.total_voxels,
        state.options.spring_stiffness * state.options.strength_multiplier,
        center, state.rigid_sphere_impulses);
    check(cudaGetLastError(), "measure soft-cross rim reaction torque");
    check(cudaMemcpyAsync(state.host_rigid_sphere_impulses.data(),
        state.rigid_sphere_impulses,
        static_cast<std::size_t>(state.total_voxels) * sizeof(float3),
        cudaMemcpyDeviceToHost, stream), "download soft-cross rim torque");
    check(cudaStreamSynchronize(stream), "finish soft-cross rim torque");
    float torque{};
    for (const float3 row : state.host_rigid_sphere_impulses) torque += row.z;
    return torque;
}

void SoftBodyCourse::set_strength_multiplier(float multiplier)
{
    if (!impl_) throw std::logic_error("moved-from soft-body course");
    if (!finite(multiplier) || multiplier < 0.0625F || multiplier > 64.0F) {
        throw std::invalid_argument("soft-body strength multiplier must be in [0.0625, 64]");
    }
    impl_->options.strength_multiplier = multiplier;
}

void SoftBodyCourse::set_solver_substeps(std::uint32_t substeps)
{
    if (!impl_) throw std::logic_error("moved-from soft-body course");
    if (impl_->frame_open || substeps == 0U ||
        substeps > Impl::maximum_substeps) {
        throw std::invalid_argument("soft-body substeps must be in [1, 32] between frames");
    }
    impl_->options.solver_substeps = substeps;
}

void SoftBodyCourse::set_spring_solver_iterations(std::uint32_t iterations)
{
    if (!impl_) throw std::logic_error("moved-from soft-body course");
    if (impl_->frame_open || iterations == 0U || iterations > 256U)
        throw std::invalid_argument(
            "soft-body spring iterations must be in [1, 256] between frames");
    impl_->options.spring_solver_iterations = iterations;
}

void SoftBodyCourse::set_material(SoftBodyMaterial material)
{
    if (!impl_) throw std::logic_error("moved-from soft-body course");
    if (impl_->frame_open || !finite(material.spring_stiffness) ||
        !finite(material.spring_damping_ratio) ||
        !finite(material.velocity_damping) || !finite(material.maximum_speed) ||
        !finite(material.ground_friction) ||
        material.spring_stiffness < 100.0F || material.spring_stiffness > 160'000.0F ||
        material.spring_damping_ratio < 0.0F ||
        material.spring_damping_ratio > 4.0F ||
        material.velocity_damping < 0.0F || material.velocity_damping > 30.0F ||
        material.maximum_speed < 0.5F || material.maximum_speed > 30.0F ||
        material.ground_friction < 0.0F || material.ground_friction > 50.0F) {
        throw std::invalid_argument("invalid live soft-body material");
    }
    impl_->options.spring_stiffness = material.spring_stiffness;
    impl_->options.spring_damping_ratio = material.spring_damping_ratio;
    impl_->options.velocity_damping = material.velocity_damping;
    impl_->options.maximum_speed = material.maximum_speed;
    impl_->options.ground_friction = material.ground_friction;
}

void SoftBodyCourse::set_voxel_mass(float mass)
{
    if (!impl_) throw std::logic_error("moved-from soft-body course");
    if (impl_->frame_open || !finite(mass) || mass < 0.001F || mass > 100.0F)
        throw std::invalid_argument(
            "soft-body voxel mass must be in [0.001, 100] between frames");
    impl_->options.voxel_mass = mass;
}

float SoftBodyCourse::voxel_mass() const noexcept
{
    return impl_ ? impl_->options.voxel_mass : 0.0F;
}

void SoftBodyCourse::set_primary_body_mass(float mass)
{
    if (!impl_) throw std::logic_error("moved-from soft-body course");
    if (impl_->frame_open || impl_->options.cross_source_nodes==0U ||
        !finite(mass) || mass<impl_->options.voxel_mass || mass>20'000.0F)
        throw std::invalid_argument(
            "primary soft-body node mass must be between target mass and 20000");
    impl_->options.cross_source_mass_multiplier=mass/impl_->options.voxel_mass;
}

float SoftBodyCourse::primary_body_mass() const noexcept
{
    return impl_ ? impl_->options.voxel_mass*
        impl_->options.cross_source_mass_multiplier : 0.0F;
}

SoftBodyMaterial SoftBodyCourse::material() const noexcept
{
    if (!impl_) return {};
    return {impl_->options.spring_stiffness,
        impl_->options.spring_damping_ratio,
        impl_->options.velocity_damping, impl_->options.maximum_speed,
        impl_->options.ground_friction};
}

std::uint32_t SoftBodyCourse::spring_solver_iterations() const noexcept
{
    return impl_ ? impl_->options.spring_solver_iterations : 0U;
}

void SoftBodyCourse::set_uniform_velocity(
    std::uint32_t first_node, std::uint32_t node_count, float3 velocity,
    cudaStream_t stream)
{
    if (!impl_) throw std::logic_error("moved-from soft-body course");
    if (!finite(velocity) || first_node > impl_->total_voxels ||
        node_count > impl_->total_voxels - first_node) {
        throw std::invalid_argument("invalid soft-body velocity range");
    }
    if (node_count == 0U) return;
    set_voxel_velocity_kernel<<<
        (node_count + block_size - 1U) / block_size,
        block_size, 0, stream>>>(impl_->velocities, impl_->flags,
        first_node, first_node + node_count, velocity);
    check(cudaGetLastError(), "launch soft-body initial velocity");
    check(cudaStreamSynchronize(stream), "commit soft-body initial velocity");
}

void SoftBodyCourse::translate_pinned(float3 delta,cudaStream_t stream)
{
    if (!impl_) throw std::logic_error("moved-from soft-body course");
    if (impl_->frame_open || !finite(delta))
        throw std::invalid_argument("pinned translation requires finite input between frames");
    translate_pinned_kernel<<<
        (impl_->total_voxels+block_size-1U)/block_size,block_size,0,stream>>>(
        impl_->rest_positions,impl_->positions,impl_->flags,
        impl_->total_voxels,delta);
    check(cudaGetLastError(),"translate pinned soft-body anchors");
}

void SoftBodyCourse::scale_rest_lengths(float factor,cudaStream_t stream)
{
    if (!impl_) throw std::logic_error("moved-from soft-body course");
    if (impl_->frame_open || !finite(factor) || factor<0.25F || factor>4.0F)
        throw std::invalid_argument("rope rest scale factor must be in [0.25, 4]");
    const std::uint32_t count=static_cast<std::uint32_t>(impl_->asset.edges.size());
    scale_edge_rest_lengths_kernel<<<
        (count+block_size-1U)/block_size,block_size,0,stream>>>(
        impl_->edges,count,factor);
    check(cudaGetLastError(),"scale rope rest lengths");
}

void SoftBodyCourse::capture_state(SoftBodyState& output, cudaStream_t stream) const
{
    if (!impl_) throw std::logic_error("moved-from soft-body course");
    const auto& state = *impl_;
    if (state.frame_open) {
        throw std::logic_error("cannot capture soft-body state during an open frame");
    }
    output.strength_multiplier = state.options.strength_multiplier;
    output.statistics = state.statistics;
    output.positions.resize(state.total_voxels);
    output.velocities.resize(state.total_voxels);
    output.active_edges.resize(state.total_edges);
    output.edge_damage.resize(state.total_edges);
    output.active_render_triangles.resize(state.total_render_triangles);
    check(cudaMemcpyAsync(output.positions.data(), state.positions,
        output.positions.size() * sizeof(float3), cudaMemcpyDeviceToHost, stream),
        "capture soft-body positions");
    check(cudaMemcpyAsync(output.velocities.data(), state.velocities,
        output.velocities.size() * sizeof(float3), cudaMemcpyDeviceToHost, stream),
        "capture soft-body velocities");
    check(cudaMemcpyAsync(output.active_edges.data(), state.active_edges,
        output.active_edges.size() * sizeof(std::uint8_t), cudaMemcpyDeviceToHost, stream),
        "capture soft-body edges");
    check(cudaMemcpyAsync(output.edge_damage.data(), state.edge_damage,
        output.edge_damage.size() * sizeof(std::uint8_t), cudaMemcpyDeviceToHost, stream),
        "capture soft-body fracture damage");
    check(cudaMemcpyAsync(output.active_render_triangles.data(), state.render_triangle_active,
        output.active_render_triangles.size() * sizeof(std::uint8_t),
        cudaMemcpyDeviceToHost, stream), "capture soft-body render triangles");
    check(cudaStreamSynchronize(stream), "complete soft-body capture");
}

void SoftBodyCourse::restore_state(const SoftBodyState& input, cudaStream_t stream)
{
    if (!impl_) throw std::logic_error("moved-from soft-body course");
    auto& state = *impl_;
    if (state.frame_open) {
        throw std::logic_error("cannot restore soft-body state during an open frame");
    }
    if (!finite(input.strength_multiplier) || input.strength_multiplier < 0.0625F ||
        input.strength_multiplier > 64.0F || input.positions.size() != state.total_voxels ||
        input.velocities.size() != state.total_voxels ||
        input.active_edges.size() != state.total_edges ||
        (!input.edge_damage.empty() && input.edge_damage.size() != state.total_edges) ||
        input.active_render_triangles.size() != state.total_render_triangles ||
        input.statistics.instance_count != state.statistics.instance_count ||
        input.statistics.voxels_per_instance != state.statistics.voxels_per_instance ||
        input.statistics.total_voxel_count != state.statistics.total_voxel_count ||
        input.statistics.total_edge_count != state.statistics.total_edge_count) {
        throw std::invalid_argument("soft-body restore state does not match this asset");
    }
    if (std::any_of(input.positions.begin(), input.positions.end(),
            [](float3 value) { return !finite(value); }) ||
        std::any_of(input.velocities.begin(), input.velocities.end(),
            [](float3 value) { return !finite(value); }) ||
        std::any_of(input.active_edges.begin(), input.active_edges.end(),
            [](std::uint8_t value) { return value > 1U; }) ||
        std::any_of(input.edge_damage.begin(), input.edge_damage.end(),
            [&state](std::uint8_t value) {
                return value > state.options.fracture_persistence_substeps;
            }) ||
        std::any_of(input.active_render_triangles.begin(),
            input.active_render_triangles.end(),
            [](std::uint8_t value) { return value > 1U; })) {
        throw std::invalid_argument("soft-body restore state contains invalid values");
    }
    const std::uint32_t broken = static_cast<std::uint32_t>(std::count(
        input.active_edges.begin(), input.active_edges.end(), std::uint8_t{0U}));
    if (broken != input.statistics.broken_edge_count) {
        throw std::invalid_argument("soft-body restore broken-edge statistic is inconsistent");
    }
    check(cudaMemcpyAsync(state.position_a, input.positions.data(),
        input.positions.size() * sizeof(float3), cudaMemcpyHostToDevice, stream),
        "restore soft-body positions A");
    check(cudaMemcpyAsync(state.position_b, input.positions.data(),
        input.positions.size() * sizeof(float3), cudaMemcpyHostToDevice, stream),
        "restore soft-body positions B");
    check(cudaMemcpyAsync(state.velocity_a, input.velocities.data(),
        input.velocities.size() * sizeof(float3), cudaMemcpyHostToDevice, stream),
        "restore soft-body velocities A");
    check(cudaMemcpyAsync(state.velocity_b, input.velocities.data(),
        input.velocities.size() * sizeof(float3), cudaMemcpyHostToDevice, stream),
        "restore soft-body velocities B");
    check(cudaMemcpyAsync(state.active_edges, input.active_edges.data(),
        input.active_edges.size() * sizeof(std::uint8_t), cudaMemcpyHostToDevice, stream),
        "restore soft-body edges");
    if (input.edge_damage.empty()) {
        check(cudaMemsetAsync(state.edge_damage, 0,
            static_cast<std::size_t>(state.total_edges) * sizeof(std::uint8_t), stream),
            "clear legacy restored soft-body fracture damage");
    } else {
        check(cudaMemcpyAsync(state.edge_damage, input.edge_damage.data(),
            input.edge_damage.size() * sizeof(std::uint8_t), cudaMemcpyHostToDevice, stream),
            "restore soft-body fracture damage");
    }
    check(cudaMemcpyAsync(state.render_triangle_active,
        input.active_render_triangles.data(),
        input.active_render_triangles.size() * sizeof(std::uint8_t),
        cudaMemcpyHostToDevice, stream), "restore soft-body render triangles");
    state.positions = state.position_a;
    state.next_positions = state.position_b;
    state.velocities = state.velocity_a;
    state.next_velocities = state.velocity_b;
    state.options.strength_multiplier = input.strength_multiplier;
    check(cudaMemsetAsync(state.external_impulses, 0,
        static_cast<std::size_t>(state.total_voxels) * sizeof(float3), stream),
        "clear restored soft-body impulses");
    check(cudaMemsetAsync(state.position_corrections, 0,
        static_cast<std::size_t>(state.total_voxels) * sizeof(float3), stream),
        "clear restored soft-body corrections");
    state.enqueue_render(stream);
    state.update_render_normals(stream);
    check(meshprep::refit_hierarchy_unchecked_async(
        {state.render_bounds, state.total_render_triangles}, state.render_hierarchy, stream),
        "restore soft-body render hierarchy");
    state.refit_members(stream);
    check(cudaStreamSynchronize(stream), "complete soft-body restore");
    state.statistics = input.statistics;
    state.completed_substeps = 0U;
    state.substep_prepared = false;
}

float SoftBodyCourse::strength_multiplier() const noexcept
{
    return impl_ ? impl_->options.strength_multiplier : 0.0F;
}

SoftBodyVoxelView SoftBodyCourse::voxel_view() const noexcept
{
    if (!impl_) return {};
    return {impl_->positions, impl_->velocities, impl_->rest_positions, impl_->flags,
        impl_->external_impulses, impl_->position_corrections, impl_->total_voxels,
        static_cast<std::uint32_t>(impl_->asset.rest_voxels.size()),
        impl_->options.instance_count, impl_->asset.voxel_radius,
        1.0F / impl_->options.voxel_mass};
}

SoftBodyLatticeView SoftBodyCourse::lattice_view() const noexcept
{
    if (!impl_) return {};
    return {impl_->positions, impl_->flags, impl_->edges, impl_->active_edges,
        impl_->total_voxels,
        static_cast<std::uint32_t>(impl_->asset.rest_voxels.size()),
        static_cast<std::uint32_t>(impl_->asset.edges.size()),
        impl_->options.instance_count, impl_->asset.voxel_radius};
}

SoftBodyRenderView SoftBodyCourse::render_view() const noexcept
{
    if (!impl_) return {};
    const auto hierarchy = impl_->render_hierarchy.statistics();
    const auto members = impl_->member_hierarchy.statistics();
    return {impl_->render_positions, impl_->render_normals.vertex_normals(),
        impl_->render_normals.corner_normal_indices(),
        impl_->render_uvs, impl_->render_triangles,
        impl_->render_bindings, impl_->render_triangle_active,
        impl_->total_render_vertices, impl_->total_render_triangles,
        impl_->render_hierarchy.nodes(), impl_->render_hierarchy.primitive_indices(),
        hierarchy.node_count, hierarchy.max_depth,
        impl_->options.render_internal_members ? impl_->positions : nullptr,
        impl_->options.render_internal_members
            ? (impl_->asset.render_member_edges.empty()
                ? impl_->edges : impl_->presentation_member_edges) : nullptr,
        impl_->options.render_internal_members
            ? (impl_->asset.render_member_edges.empty()
                ? impl_->active_edges : impl_->presentation_member_active) : nullptr,
        impl_->member_hierarchy.nodes(), impl_->member_hierarchy.primitive_indices(),
        impl_->options.render_internal_members ? impl_->total_members : 0U,
        impl_->options.render_internal_members
            ? static_cast<std::uint32_t>(impl_->asset.rest_voxels.size()) : 0U,
        impl_->options.render_internal_members
            ? impl_->members_per_instance : 0U,
        members.node_count, members.max_depth, 0.11F * impl_->asset.voxel_radius,
        impl_->options.surface_triangle_split,
        impl_->options.secondary_surface_triangle_split,
        impl_->options.rope_bridge_columns,
        impl_->options.rope_bridge_rows,
        impl_->options.rope_bridge_nodes_per_tile};
}

meshprep::DeviceMeshView SoftBodyCourse::render_mesh() const noexcept
{
    if (!impl_) return {};
    return {impl_->render_positions, impl_->total_render_vertices,
        impl_->render_triangles, impl_->total_render_triangles};
}

const meshprep::Hierarchy& SoftBodyCourse::render_hierarchy() const noexcept
{
    return impl_->render_hierarchy;
}

SoftBodyStatistics SoftBodyCourse::statistics() const noexcept
{
    return impl_ ? impl_->statistics : SoftBodyStatistics{};
}

std::size_t SoftBodyCourse::allocated_bytes() const noexcept
{
    if (!impl_) return 0U;
    const auto& state = *impl_;
    return static_cast<std::size_t>(state.total_voxels) *
            (9U * sizeof(float3) + sizeof(std::uint32_t)) +
        state.asset.edges.size() * sizeof(SoftBodyEdge) +
        state.asset.neighbor_offsets.size() * sizeof(std::uint32_t) +
        state.asset.neighbors.size() * sizeof(SoftBodyNeighbor) +
        static_cast<std::size_t>(state.total_edges) * sizeof(std::uint8_t) +
        static_cast<std::size_t>(state.total_edges) * sizeof(std::uint8_t) +
        state.asset.rest_voxels.size() * sizeof(float3) +
        state.asset.render_positions.size() * sizeof(float3) +
        state.asset.render_positions.size() * sizeof(uint2) +
        state.asset.render_triangles.size() * sizeof(uint3) +
        static_cast<std::size_t>(state.total_render_vertices) *
            (sizeof(float3) + sizeof(float2) + sizeof(SoftBodyBinding)) +
        static_cast<std::size_t>(state.total_render_triangles) *
            (sizeof(uint3) + sizeof(meshprep::Aabb) + sizeof(std::uint8_t)) +
        (state.options.render_internal_members
            ? static_cast<std::size_t>(state.total_edges) * sizeof(meshprep::Aabb) : 0U) +
        static_cast<std::size_t>(state.options.cross_source_nodes) * 3U *
            (sizeof(std::uint32_t) + sizeof(float3)) +
        state.render_workspace.capacity_bytes() + state.render_hierarchy.allocated_bytes() +
        state.render_normal_workspace.capacity_bytes() +
        state.render_normals.allocated_bytes() +
        state.member_workspace.capacity_bytes() + state.member_hierarchy.allocated_bytes();
}

} // namespace waterlab
