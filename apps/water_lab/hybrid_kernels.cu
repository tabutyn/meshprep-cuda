// SPDX-License-Identifier: MIT
#include "hybrid_lab.hpp"
#include "triangle_contact.cuh"
#include "obstacle_course.hpp"
#include "particle_cells.cuh"
#include "course_rotation.cuh"
#include "status_exception.hpp"

#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <nvtx3/nvtx3.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace waterlab {
namespace {

constexpr std::uint32_t block_size = 256U;

enum Stage : std::uint32_t {
    stage_fluid_hierarchy,
    stage_skin_hierarchy,
    stage_fluid_physics,
    stage_skin_physics,
    stage_rectangle_physics,
    stage_surface_normals,
    stage_render_surface,
    stage_soft_body_contact,
    stage_particle_recycling,
};

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
    if (count != 0U) check(cudaMalloc(&pointer, count * sizeof(T)), "hybrid cudaMalloc");
}

template <typename T>
void upload(T* destination, const std::vector<T>& source)
{
    if (!source.empty()) {
        check(cudaMemcpy(destination, source.data(), source.size() * sizeof(T),
            cudaMemcpyHostToDevice), "hybrid upload");
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

__host__ __device__ bool finite3(float3 value)
{
    return isfinite(value.x) && isfinite(value.y) && isfinite(value.z);
}

__host__ __device__ float3 clamp_length(float3 value, float maximum)
{
    const float magnitude = length(value);
    return magnitude > maximum && magnitude > 0.0F
        ? multiply(value, maximum / magnitude)
        : value;
}

__host__ __device__ float3 rotate_y(float3 value, float angle)
{
    const float cosine = cosf(angle);
    const float sine = sinf(angle);
    return make_float3(
        cosine * value.x + sine * value.z,
        value.y,
        -sine * value.x + cosine * value.z);
}

__device__ float smoothstep01(float value)
{
    const float x = fminf(1.0F, fmaxf(0.0F, value));
    return x * x * (3.0F - 2.0F * x);
}

__device__ float bounds_distance_squared(
    float3 point, const meshprep::HierarchyNode& node)
{
    const float dx = fmaxf(fmaxf(node.bounds_min.x - point.x, 0.0F),
        point.x - node.bounds_max.x);
    const float dy = fmaxf(fmaxf(node.bounds_min.y - point.y, 0.0F),
        point.y - node.bounds_max.y);
    const float dz = fmaxf(fmaxf(node.bounds_min.z - point.z, 0.0F),
        point.z - node.bounds_max.z);
    return dx * dx + dy * dy + dz * dz;
}

__device__ float bounds_xz_distance_squared(
    float3 point, const meshprep::HierarchyNode& node)
{
    const float dx = fmaxf(fmaxf(node.bounds_min.x - point.x, 0.0F),
        point.x - node.bounds_max.x);
    const float dz = fmaxf(fmaxf(node.bounds_min.z - point.z, 0.0F),
        point.z - node.bounds_max.z);
    return dx * dx + dz * dz;
}

__device__ bool cloth_xz_weights(float3 point, float3 a, float3 b, float3 c,
    float3& weights)
{
    const float determinant = (b.z - c.z) * (a.x - c.x) +
        (c.x - b.x) * (a.z - c.z);
    if (fabsf(determinant) < 1.0e-8F) return false;
    const float w0 = ((b.z - c.z) * (point.x - c.x) +
        (c.x - b.x) * (point.z - c.z)) / determinant;
    const float w1 = ((c.z - a.z) * (point.x - c.x) +
        (a.x - c.x) * (point.z - c.z)) / determinant;
    const float w2 = 1.0F - w0 - w1;
    if (fminf(w0, fminf(w1, w2)) < -1.0e-5F) return false;
    weights = make_float3(w0, w1, w2);
    return true;
}

using detail::closest_triangle_barycentric;

std::vector<float3> make_hcp_particles(const HybridOptions& options)
{
    struct Candidate { float radius_squared; float3 point; };
    std::vector<Candidate> candidates;
    constexpr int extent = 26;
    const float row_height = options.particle_spacing * std::sqrt(3.0F) * 0.5F;
    const float layer_height = options.particle_spacing * std::sqrt(2.0F / 3.0F);
    for (int layer = -extent; layer <= extent; ++layer) {
        for (int row = -extent; row <= extent; ++row) {
            for (int column = -extent; column <= extent; ++column) {
                const bool shifted_layer = (layer & 1) != 0;
                const bool shifted_row = (row & 1) != 0;
                const float3 point = make_float3(
                    options.particle_spacing *
                        (static_cast<float>(column) + (shifted_row ? 0.5F : 0.0F) +
                         (shifted_layer ? 0.5F : 0.0F)),
                    static_cast<float>(layer) * layer_height,
                    static_cast<float>(row) * row_height +
                        (shifted_layer ? row_height / 3.0F : 0.0F));
                candidates.push_back({dot(point, point), point});
            }
        }
    }
    std::sort(candidates.begin(), candidates.end(),
        [](const Candidate& a, const Candidate& b) {
            if (a.radius_squared != b.radius_squared) return a.radius_squared < b.radius_squared;
            if (a.point.y != b.point.y) return a.point.y < b.point.y;
            if (a.point.z != b.point.z) return a.point.z < b.point.z;
            return a.point.x < b.point.x;
        });
    if (candidates.size() < options.particle_capacity) {
        throw std::runtime_error("HCP generator produced too few particles");
    }
    std::vector<float3> result;
    result.reserve(options.particle_capacity);
    for (std::uint32_t index = 0U; index < options.particle_capacity; ++index) {
        float3 point = candidates[index].point;
        point.x *= options.particle_initial_scale.x;
        point.y *= options.particle_initial_scale.y;
        point.z *= options.particle_initial_scale.z;
        result.push_back(add(point, options.particle_initial_center));
    }
    return result;
}

struct AddFloat3 {
    __host__ __device__ float3 operator()(float3 a, float3 b) const { return add(a, b); }
};

__global__ void emit_bounds_kernel(
    const float3* positions,
    meshprep::Aabb* bounds,
    std::uint32_t count,
    float radius)
{
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float3 extent = make_float3(radius, radius, radius);
    bounds[index] = {subtract(positions[index], extent), add(positions[index], extent)};
}

__global__ void initialize_added_particles_kernel(
    float3* positions, const float3* initial_positions, float3* velocities,
    float3* forces, std::uint32_t* skin_owners,
    std::uint32_t first, std::uint32_t last)
{
    const std::uint32_t index = first + blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= last) return;
    positions[index] = initial_positions[index];
    velocities[index] = {};
    forces[index] = {};
    skin_owners[index] = 0U;
}

// The slope is an emitter/sink, not an ever-growing pool. Recycling stable
// particle IDs avoids capacity churn and keeps neighbor sorting reproducible.
__global__ void recycle_slope_particles_kernel(
    float3* positions, const float3* spawn_positions, float3* velocities,
    float3* forces, std::uint32_t* skin_owners, std::uint32_t count,
    GalleryArena arena, std::uint32_t* recycled_count)
{
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float3 point = positions[index];
    const bool inside_cycle = arena == GalleryArena::water_wheel
        ? point.x >= water_wheel_inlet_start_x - 0.8F &&
            point.x <= water_wheel_collector_end_x + 0.35F &&
            point.y >= water_wheel_ground_y - 1.0F
        : point.x <= 2.45F && point.y >= -1.65F;
    if (inside_cycle) return;
    positions[index] = spawn_positions[index];
    velocities[index] = {};
    forces[index] = {};
    skin_owners[index] = 0U;
    atomicAdd(recycled_count, 1U);
}

__global__ void emit_particle_cells_kernel(
    const float3* positions,
    std::uint64_t* keys,
    std::uint32_t* indices,
    std::uint32_t count,
    float inverse_cell_size)
{
    const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle >= count) return;
    const float3 point = positions[particle];
    const int x = __float2int_rd(point.x * inverse_cell_size);
    const int y = __float2int_rd(point.y * inverse_cell_size);
    const int z = __float2int_rd(point.z * inverse_cell_size);
    keys[particle] = detail::particle_cell_key(x, y, z);
    indices[particle] = particle;
}

__device__ void consider_closest_vertex(
    float3 point, const float3* positions, std::uint32_t vertex_count,
    std::uint32_t vertex, float& closest_squared, std::uint32_t& closest)
{
    if (vertex >= vertex_count) return;
    const float3 delta = subtract(point, positions[vertex]);
    const float distance_squared = dot(delta, delta);
    if (distance_squared < closest_squared ||
        (distance_squared == closest_squared && vertex < closest)) {
        closest_squared = distance_squared;
        closest = vertex;
    }
}

__global__ void particle_forces_kernel(
    const float3* positions,
    const float3* velocities,
    std::uint32_t particle_count,
    const std::uint64_t* particle_cell_keys,
    const std::uint32_t* particle_cell_indices,
    const float3* skin_positions,
    const float3* skin_velocities,
    const float3* skin_normals,
    std::uint32_t skin_vertex_count,
    std::uint32_t skin_triangle_count,
    const uint3* skin_triangles,
    const std::uint32_t* skin_incident_offsets,
    const std::uint32_t* skin_incident_triangles,
    const meshprep::HierarchyNode* skin_nodes,
    const std::uint32_t* skin_indices,
    std::uint32_t skin_node_count,
    HybridOptions options,
    float3* particle_forces,
    std::uint32_t* reaction_keys,
    float3* reaction_values,
    std::uint32_t* particle_skin_owners,
    std::uint32_t* statistics)
{
    const std::uint32_t work_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (work_index >= particle_count) return;
    // Schedule in deterministic cell order while externally visible arrays
    // remain indexed by stable particle ID.
    const std::uint32_t particle = particle_cell_indices[work_index];
    if (particle >= particle_count) {
        atomicAdd(statistics + 2U, 1U);
        return;
    }
    const float3 point = positions[particle];
    const float3 velocity = velocities[particle];
    float3 force{};
    std::uint32_t neighbors = 0U;
    const float inverse_cell_size = 1.0F / options.particle_support_radius;
    const int cell_x = __float2int_rd(point.x * inverse_cell_size);
    const int cell_y = __float2int_rd(point.y * inverse_cell_size);
    const int cell_z = __float2int_rd(point.z * inverse_cell_size);
    const ParticleCellView cells{positions, particle_cell_keys, particle_cell_indices,
        particle_count, options.particle_support_radius};
    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                const std::uint64_t key = detail::particle_cell_key(
                    cell_x + dx, cell_y + dy, cell_z + dz);
                const std::uint32_t first = detail::particle_cell_lower_bound(cells, key);
                for (std::uint32_t item = first;
                     item < particle_count && particle_cell_keys[item] == key; ++item) {
                    const std::uint32_t other = particle_cell_indices[item];
                    if (other == particle || other >= particle_count) continue;
                    const float3 delta = subtract(point, positions[other]);
                    const float distance = length(delta);
                    if (!(distance > 1.0e-6F) ||
                        distance >= options.particle_support_radius) continue;
                    ++neighbors;
                    const float3 direction = multiply(delta, 1.0F / distance);
                    const float normalized_distance =
                        1.0F - distance / options.particle_support_radius;
                    float magnitude = options.particle_repulsion *
                        normalized_distance * normalized_distance;
                    magnitude -= options.particle_damping *
                        dot(subtract(velocity, velocities[other]), direction);
                    force = add(force, multiply(direction, magnitude));
                }
            }
        }
    }

    const std::uint32_t previous_owner = particle_skin_owners[particle];
    const std::uint32_t reaction_base = 3U * particle;
    for (std::uint32_t record = 0U; record < 3U; ++record) {
        reaction_keys[reaction_base + record] = UINT32_MAX;
        reaction_values[reaction_base + record] = {};
    }
    particle_skin_owners[particle] = UINT32_MAX;

    if (options.particle_skin_coupling) {
    std::uint32_t closest = UINT32_MAX;
    float closest_squared = INFINITY;
    if (previous_owner < skin_triangle_count) {
        const uint3 triangle = skin_triangles[previous_owner];
        consider_closest_vertex(point, skin_positions, skin_vertex_count,
            triangle.x, closest_squared, closest);
        consider_closest_vertex(point, skin_positions, skin_vertex_count,
            triangle.y, closest_squared, closest);
        consider_closest_vertex(point, skin_positions, skin_vertex_count,
            triangle.z, closest_squared, closest);
    }
    std::uint32_t stack[128];
    std::uint32_t stack_size = 1U;
    stack[0] = 0U;
    while (stack_size != 0U && skin_node_count != 0U) {
        const meshprep::HierarchyNode node = skin_nodes[stack[--stack_size]];
        if (bounds_distance_squared(point, node) > closest_squared) continue;
        if (node.is_leaf()) {
            for (std::uint32_t item = 0U; item < node.primitive_count; ++item) {
                const std::uint32_t vertex = skin_indices[node.first_primitive + item];
                if (vertex >= skin_vertex_count) continue;
                const float distance_squared = dot(
                    subtract(point, skin_positions[vertex]),
                    subtract(point, skin_positions[vertex]));
                if (distance_squared < closest_squared ||
                    (distance_squared == closest_squared && vertex < closest)) {
                    closest_squared = distance_squared;
                    closest = vertex;
                }
            }
        } else {
            std::uint32_t nearest_child = UINT32_MAX;
            float nearest_child_squared = INFINITY;
            for (std::uint32_t child = 0U; child < node.child_count; ++child) {
                const std::uint32_t child_index = node.first_child + child;
                const float child_squared = bounds_distance_squared(
                    point, skin_nodes[child_index]);
                if (child_squared < nearest_child_squared ||
                    (child_squared == nearest_child_squared && child_index < nearest_child)) {
                    nearest_child_squared = child_squared;
                    nearest_child = child_index;
                }
            }
            // Push the nearest child last so the LIFO walk visits it first and
            // tightens the exact pruning bound before considering its siblings.
            for (std::uint32_t child = 0U; child < node.child_count; ++child) {
                const std::uint32_t child_index = node.first_child + child;
                if (child_index == nearest_child) continue;
                if (stack_size < 128U) stack[stack_size++] = child_index;
                else atomicAdd(statistics + 2U, 1U);
            }
            if (nearest_child != UINT32_MAX) {
                if (stack_size < 128U) stack[stack_size++] = nearest_child;
                else atomicAdd(statistics + 2U, 1U);
            }
        }
    }

    if (closest != UINT32_MAX) {
        float3 surface_point = skin_positions[closest];
        float3 normal = skin_normals[closest];
        uint3 reaction_vertices{closest, closest, closest};
        float3 reaction_weights = make_float3(1.0F, 0.0F, 0.0F);
        std::uint32_t owner = closest;

        float triangle_distance_squared = INFINITY;
        std::uint32_t best_triangle = UINT32_MAX;
        for (std::uint32_t incident = skin_incident_offsets[closest];
             incident < skin_incident_offsets[closest + 1U]; ++incident) {
            const std::uint32_t triangle_id = skin_incident_triangles[incident];
            const uint3 triangle = skin_triangles[triangle_id];
            const float3 barycentric = closest_triangle_barycentric(
                point, skin_positions[triangle.x], skin_positions[triangle.y],
                skin_positions[triangle.z]);
            const float3 candidate = add(
                multiply(skin_positions[triangle.x], barycentric.x),
                add(multiply(skin_positions[triangle.y], barycentric.y),
                    multiply(skin_positions[triangle.z], barycentric.z)));
            const float distance_squared = dot(
                subtract(point, candidate), subtract(point, candidate));
            if (distance_squared < triangle_distance_squared ||
                (distance_squared == triangle_distance_squared &&
                 triangle_id < best_triangle)) {
                triangle_distance_squared = distance_squared;
                best_triangle = triangle_id;
                reaction_vertices = triangle;
                reaction_weights = barycentric;
                surface_point = candidate;
            }
        }
        if (best_triangle != UINT32_MAX) {
            owner = best_triangle;
            closest_squared = triangle_distance_squared;
            normal = add(
                multiply(skin_normals[reaction_vertices.x], reaction_weights.x),
                add(multiply(skin_normals[reaction_vertices.y], reaction_weights.y),
                    multiply(skin_normals[reaction_vertices.z], reaction_weights.z)));
            const float normal_length = length(normal);
            if (normal_length > 1.0e-10F) normal = multiply(normal, 1.0F / normal_length);
        }
        if (dot(normal, normal) < 1.0e-10F) {
            const uint3 triangle = reaction_vertices;
            normal = cross(
                subtract(skin_positions[triangle.y], skin_positions[triangle.x]),
                subtract(skin_positions[triangle.z], skin_positions[triangle.x]));
            const float normal_length = length(normal);
            if (normal_length > 1.0e-10F) {
                normal = multiply(normal, 1.0F / normal_length);
            }
        }
        const float3 surface_offset = subtract(point, surface_point);
        const float signed_distance = dot(surface_offset, normal);
        const float violation = options.particle_skin_distance + signed_distance;
        const float interaction_squared = options.particle_skin_interaction_radius *
            options.particle_skin_interaction_radius;
        const bool outside = signed_distance > 0.0F;
        if ((outside || closest_squared <= interaction_squared) && violation > 0.0F) {
            const float3 surface_velocity = add(
                multiply(skin_velocities[reaction_vertices.x], reaction_weights.x),
                add(multiply(skin_velocities[reaction_vertices.y], reaction_weights.y),
                    multiply(skin_velocities[reaction_vertices.z], reaction_weights.z)));
            const float3 relative_velocity = subtract(velocity, surface_velocity);
            const float relative_normal_speed = dot(relative_velocity, normal);
            const float magnitude = fminf(options.maximum_particle_force,
                fmaxf(0.0F, options.particle_skin_stiffness * violation +
                    options.particle_skin_damping * relative_normal_speed));
            // Normal pressure contains the fluid. A deliberately small
            // tangential term transfers moving-wall velocity into nearby
            // particles, so a rolling skin does not slide around an
            // effectively frictionless fluid interior.
            const float3 tangent_velocity = subtract(relative_velocity,
                multiply(normal, relative_normal_speed));
            const float tangential_drag = 0.12F * options.particle_skin_damping;
            const float3 surface_force = clamp_length(
                add(multiply(normal, -magnitude),
                    multiply(tangent_velocity, -tangential_drag)),
                options.maximum_particle_force);
            force = add(force, surface_force);
            const float3 reaction = multiply(surface_force, -1.0F);
            particle_skin_owners[particle] = owner;
            reaction_keys[reaction_base] = reaction_vertices.x;
            reaction_keys[reaction_base + 1U] = reaction_vertices.y;
            reaction_keys[reaction_base + 2U] = reaction_vertices.z;
            reaction_values[reaction_base] = multiply(reaction, reaction_weights.x);
            reaction_values[reaction_base + 1U] = multiply(reaction, reaction_weights.y);
            reaction_values[reaction_base + 2U] = multiply(reaction, reaction_weights.z);
        }
        if (outside) atomicAdd(statistics, 1U);
    }
    }
    if (options.collect_contact_diagnostics &&
        length(force) > options.maximum_particle_force) atomicAdd(statistics + 8U, 1U);
    force = clamp_length(force, options.maximum_particle_force);
    if (!finite3(force) || !finite3(point) || !finite3(velocity)) {
        force = {};
        atomicAdd(statistics + 2U, 1U);
    }
    particle_forces[particle] = force;
    atomicAdd(statistics + 3U, neighbors);
    atomicMax(statistics + 4U, neighbors);
    atomicMax(statistics + 5U, __float_as_uint(length(force)));
}

__global__ void integrate_particles_kernel(
    float3* positions,
    float3* velocities,
    const float3* forces,
    std::uint32_t count,
    float inverse_mass,
    float velocity_damping,
    float maximum_speed,
    float dt,
    float3 gravity,
    GalleryArena arena,
    float radius,
    std::uint32_t* finite_failures)
{
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float3 old_position = positions[index];
    float3 velocity = add(velocities[index], multiply(forces[index], inverse_mass * dt));
    velocity = add(velocity, multiply(gravity, dt));
    if (arena == GalleryArena::bowl) {
        // The pairwise repulsion model is intentionally cheaper than an
        // incompressible pressure solve. In a curved vessel its outer annulus
        // otherwise finds a stable, visibly raised equilibrium. Apply a
        // bounded hydrostatic equalization only to high, outer particles: it
        // drives excess head inward while leaving the settled bulk and falling
        // stream under ordinary gravity.
        const float dx = old_position.x - bowl_center.x;
        const float dz = old_position.z - bowl_center.z;
        const float radial = sqrtf(dx * dx + dz * dz);
        const float high = fmaxf(0.0F,
            old_position.y - (bowl_center.y - 1.24F));
        const float outer = fmaxf(0.0F, (radial - 0.82F) / 0.70F);
        if (radial > 1.0e-6F && high > 0.0F && outer > 0.0F) {
            const float acceleration = fminf(2.0F, 40.0F * high * outer);
            velocity.x -= acceleration * dt * dx / radial;
            velocity.z -= acceleration * dt * dz / radial;
        }
    }
    velocity = multiply(velocity, expf(-velocity_damping * dt));
    velocity = clamp_length(velocity, maximum_speed);
    float3 position = add(old_position, multiply(velocity, dt));
    project_gallery_contact(position, velocity, radius, arena);
    if (!finite3(position) || !finite3(velocity)) {
        velocities[index] = {};
        positions[index] = old_position;
        atomicAdd(finite_failures, 1U);
        return;
    }
    velocities[index] = velocity;
    positions[index] = position;
}

__global__ void contact_particles_rigid_sphere_kernel(
    float3* positions, float3* velocities, std::uint32_t count,
    RigidSphereState sphere, float particle_radius, float particle_mass,
    float dt, float maximum_speed, float3* sphere_impulses)
{
    const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle >= count) return;
    sphere_impulses[particle] = {};
    const float3 delta = subtract(positions[particle], sphere.center);
    const float distance = length(delta);
    const float target = sphere.radius + particle_radius;
    if (!(distance < target)) return;
    const float3 normal = distance > 1.0e-8F
        ? multiply(delta, 1.0F / distance)
        : make_float3((particle & 1U) == 0U ? 1.0F : -1.0F, 0.0F, 0.0F);
    const float share = sphere.mass / (sphere.mass + particle_mass);
    const float correction_distance = fminf(
        (target - distance) * share, 0.5F * particle_radius);
    positions[particle] = add(
        positions[particle], multiply(normal, correction_distance));
    const float3 old_velocity = velocities[particle];
    const float approach = fmaxf(0.0F,
        -dot(subtract(old_velocity, sphere.velocity), normal));
    velocities[particle] = clamp_length(add(old_velocity, multiply(normal,
        approach * share + correction_distance / dt)), maximum_speed);
    sphere_impulses[particle] = multiply(
        subtract(velocities[particle], old_velocity), -particle_mass);
}

__global__ void contact_particles_water_wheel_kernel(
    float3* positions, float3* velocities, std::uint32_t count,
    WaterWheelState wheel, float particle_radius, float particle_mass,
    float maximum_speed, float* wheel_torque_impulses)
{
    const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle >= count) return;
    wheel_torque_impulses[particle] = 0.0F;
    float3 position = positions[particle];
    float3 velocity = velocities[particle];
    float3 relative = subtract(position, water_wheel_center);
    const float channel_limit = water_wheel_half_depth - particle_radius;
    if (fabsf(relative.z) > channel_limit) {
        const float sign = relative.z >= 0.0F ? 1.0F : -1.0F;
        position.z = water_wheel_center.z + sign * channel_limit;
        if (velocity.z * sign > 0.0F) velocity.z = 0.0F;
        relative = subtract(position, water_wheel_center);
    }

    float3 total_impulse{};
    float radial_length = sqrtf(relative.x * relative.x + relative.y * relative.y);
    const float core_limit = water_wheel_hub_radius + particle_radius;
    if (radial_length < core_limit && radial_length > 1.0e-8F) {
        const float3 normal = make_float3(
            relative.x / radial_length, relative.y / radial_length, 0.0F);
        const float correction = fminf(core_limit - radial_length, particle_radius);
        position = add(position, multiply(normal, correction));
        const float3 wheel_velocity = make_float3(
            -wheel.angular_velocity * relative.y,
             wheel.angular_velocity * relative.x, 0.0F);
        const float3 old_velocity = velocity;
        const float inward_speed = dot(subtract(velocity, wheel_velocity), normal);
        if (inward_speed < 0.0F)
            velocity = subtract(velocity, multiply(normal, inward_speed));
        total_impulse = add(total_impulse,
            multiply(subtract(velocity, old_velocity), -particle_mass));
        relative = subtract(position, water_wheel_center);
    }
    constexpr float two_pi = 6.28318530717958647692F;
    for (std::uint32_t fin = 0U; fin < water_wheel_fin_count; ++fin) {
        const float angle = wheel.angle + two_pi * static_cast<float>(fin) /
            static_cast<float>(water_wheel_fin_count);
        const float cosine = cosf(angle);
        const float sine = sinf(angle);
        const float radial = relative.x * cosine + relative.y * sine;
        const float tangent = -relative.x * sine + relative.y * cosine;
        const float inner = water_wheel_radius;
        const float outer = water_wheel_fin_outer_radius;
        // A water-wheel paddle is a one-way contact plane. Water on its
        // trailing (-tangent) face can push it and receive an equal reaction;
        // the descending leading face is permeable, so it cannot sweep an
        // incoming wedge back uphill and cancel the useful torque.
        if (radial < inner - particle_radius || radial > outer + particle_radius ||
            !water_wheel_back_face_overlap(tangent, particle_radius)) continue;
        const float3 normal = make_float3(sine, -cosine, 0.0F);
        const float3 arm = subtract(position, water_wheel_center);
        const float3 wheel_velocity = make_float3(
            -wheel.angular_velocity * arm.y,
             wheel.angular_velocity * arm.x, 0.0F);
        const float inward_speed = dot(subtract(velocity, wheel_velocity), normal);
        const float separation = -(tangent + water_wheel_fin_thickness);
        const float hard_depth = fmaxf(0.0F, particle_radius - separation);
        const bool approaching = water_wheel_back_face_contact(
            tangent, inward_speed, particle_radius);
        // The leading face remains permeable, but once a particle overlaps the
        // physical paddle volume it must be cleared even if relative normal
        // speed has already reached zero. Otherwise water can live inside a
        // rising fin and no empty slot appears behind it.
        if (!approaching && hard_depth <= 0.0F) continue;
        const float correction = fminf(hard_depth, 0.5F * particle_radius);
        position = add(position, multiply(normal, correction));
        const float3 old_velocity = velocity;
        // This is an inelastic velocity projection, not a spring. The former
        // spring term plus correction/dt bias created separation velocity from
        // resting overlap; the following fin then amplified that stored energy
        // into the visible upward eruption.
        const float barrier_delta_speed = fminf(maximum_speed,
            water_wheel_fin_response_delta(
                tangent, inward_speed, particle_radius));
        velocity = clamp_length(add(velocity,
            multiply(normal, barrier_delta_speed)), maximum_speed);
        total_impulse = add(total_impulse,
            multiply(subtract(velocity, old_velocity), -particle_mass));
        break;
    }

    // The circular housing is open at the uphill inlet and downhill outlet.
    // Once a particle is in the wheel chamber, the remaining arc is a single
    // smooth volume constraint rather than a polygon of edge contacts.
    const float3 chamber = subtract(position, water_wheel_center);
    radial_length = sqrtf(chamber.x * chamber.x + chamber.y * chamber.y);
    const float limit = water_wheel_shell_radius - particle_radius;
    const bool opening = water_wheel_shell_opening(chamber);
    if (radial_length > limit && radial_length < water_wheel_shell_radius + 0.45F &&
        !opening) {
        const float3 outward = make_float3(
            chamber.x / radial_length, chamber.y / radial_length, 0.0F);
        position = subtract(position, multiply(outward, radial_length - limit));
        const float outward_speed = dot(velocity, outward);
        if (outward_speed > 0.0F)
            velocity = subtract(velocity, multiply(outward, outward_speed));
    }
    positions[particle] = position;
    velocities[particle] = velocity;
    const float3 final_arm = subtract(position, water_wheel_center);
    wheel_torque_impulses[particle] =
        final_arm.x * total_impulse.y - final_arm.y * total_impulse.x;
}

__global__ void skin_spring_forces_kernel(
    const float3* positions,
    const float3* velocities,
    const std::uint32_t* neighbor_offsets,
    const std::uint32_t* neighbors,
    const float* rest_lengths,
    std::uint32_t vertex_count,
    float stiffness,
    float damping,
    float3* forces,
    float3* spring_forces)
{
    const std::uint32_t vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex >= vertex_count) return;
    float3 force{};
    for (std::uint32_t edge = neighbor_offsets[vertex];
         edge < neighbor_offsets[vertex + 1U]; ++edge) {
        const std::uint32_t other = neighbors[edge];
        const float3 delta = subtract(positions[other], positions[vertex]);
        const float distance = length(delta);
        if (!(distance > 1.0e-7F)) continue;
        const float3 direction = multiply(delta, 1.0F / distance);
        const float relative_speed = dot(subtract(velocities[other], velocities[vertex]), direction);
        const float magnitude = stiffness * (distance - rest_lengths[edge]) +
            damping * relative_speed;
        force = add(force, multiply(direction, magnitude));
    }
    forces[vertex] = force;
    spring_forces[vertex] = force;
}

__global__ void apply_skin_reactions_kernel(
    float3* skin_forces,
    const std::uint32_t* vertices,
    const float3* values,
    const std::uint32_t* run_count,
    std::uint32_t maximum_runs,
    std::uint32_t vertex_count,
    float3* particle_forces)
{
    const std::uint32_t run = blockIdx.x * blockDim.x + threadIdx.x;
    if (run >= maximum_runs || run >= *run_count) return;
    const std::uint32_t vertex = vertices[run];
    if (vertex < vertex_count) {
        skin_forces[vertex] = add(skin_forces[vertex], values[run]);
        particle_forces[vertex] = values[run];
    }
}

__global__ void skin_rectangle_forces_kernel(
    const float3* positions,
    const float3* velocities,
    std::uint32_t vertex_count,
    const DeviceRectangleState* rectangle,
    HybridOptions options,
    float3* skin_forces,
    float3* box_forces,
    float3* box_impulses,
    float* box_pair_work,
    float3* rectangle_force_rows,
    float* rectangle_torque_rows,
    std::uint32_t* statistics)
{
    const std::uint32_t vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex >= vertex_count) return;
    const DeviceRectangleState box = *rectangle;
    const float3 relative = subtract(positions[vertex], box.center);
    const float3 local = rotate_y(relative, -box.yaw);
    box_forces[vertex] = {};
    rectangle_force_rows[vertex] = {};
    rectangle_torque_rows[vertex] = 0.0F;

    const float3 q = make_float3(
        fabsf(local.x) - box.half_extents.x,
        fabsf(local.y) - box.half_extents.y,
        fabsf(local.z) - box.half_extents.z);
    const float3 outside = make_float3(
        fmaxf(q.x, 0.0F), fmaxf(q.y, 0.0F), fmaxf(q.z, 0.0F));
    const float outside_distance = length(outside);
    const float signed_distance = outside_distance +
        fminf(fmaxf(q.x, fmaxf(q.y, q.z)), 0.0F);
    if (signed_distance >= options.box_skin_contact_thickness) return;
    if (signed_distance <= 0.0F) atomicAdd(statistics + 1U, 1U);

    float3 local_normal{};
    if (outside_distance > 1.0e-8F) {
        local_normal = multiply(make_float3(
            copysignf(outside.x, local.x),
            copysignf(outside.y, local.y),
            copysignf(outside.z, local.z)), 1.0F / outside_distance);
    } else {
        local_normal = make_float3(copysignf(1.0F, local.x), 0.0F, 0.0F);
        float nearest_face = q.x;
        if (q.y > nearest_face) {
            nearest_face = q.y;
            local_normal = make_float3(0.0F, copysignf(1.0F, local.y), 0.0F);
        }
        if (q.z > nearest_face) {
            local_normal = make_float3(0.0F, 0.0F, copysignf(1.0F, local.z));
        }
    }
    const float penetration = options.box_skin_contact_thickness - signed_distance;
    const float activation = smoothstep01(
        penetration / options.box_skin_contact_thickness);
    const float stiffness = options.box_skin_stiffness * activation;
    if (stiffness <= 0.0F) return;

    const float3 normal = rotate_y(local_normal, box.yaw);
    const float3 angular_velocity = make_float3(
        box.angular_velocity * relative.z, 0.0F,
        -box.angular_velocity * relative.x);
    const float3 box_point_velocity = add(box.velocity, angular_velocity);
    const float separation_speed = dot(subtract(velocities[vertex], box_point_velocity), normal);
    float magnitude = stiffness * penetration -
        options.box_skin_damping * activation * separation_speed;
    if (options.collect_contact_diagnostics &&
        magnitude > options.maximum_box_ejection_force) atomicAdd(statistics + 10U, 1U);
    magnitude = fminf(options.maximum_box_ejection_force, fmaxf(0.0F, magnitude));
    const float3 skin_force = multiply(normal, magnitude);
    skin_forces[vertex] = add(skin_forces[vertex], skin_force);
    box_forces[vertex] = skin_force;
    if (options.collect_contact_diagnostics) {
        box_impulses[vertex] = add(
            box_impulses[vertex], multiply(skin_force, options.fixed_dt));
        box_pair_work[vertex] += magnitude * separation_speed * options.fixed_dt;
    }
    const float3 reaction = multiply(skin_force, -1.0F);
    rectangle_force_rows[vertex] = reaction;
    rectangle_torque_rows[vertex] = relative.z * reaction.x - relative.x * reaction.z;
    atomicMax(statistics + 7U, __float_as_uint(length(reaction)));
}

__global__ void integrate_skin_kernel(
    float3* positions,
    float3* velocities,
    const float3* forces,
    std::uint32_t count,
    float inverse_mass,
    float damping,
    float maximum_force,
    float maximum_speed,
    float dt,
    bool collect_diagnostics,
    float3 gravity,
    GalleryArena arena,
    float3* course_forces,
    std::uint32_t* statistics)
{
    const std::uint32_t vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex >= count) return;
    if (collect_diagnostics &&
        length(forces[vertex]) > maximum_force) atomicAdd(statistics + 9U, 1U);
    const float3 force = clamp_length(forces[vertex], maximum_force);
    const float3 old_position = positions[vertex];
    float3 velocity = add(velocities[vertex], multiply(force, inverse_mass * dt));
    velocity = add(velocity, multiply(gravity, dt));
    velocity = multiply(velocity, expf(-damping * dt));
    velocity = clamp_length(velocity, maximum_speed);
    float3 position = add(old_position, multiply(velocity, dt));
    if (arena != GalleryArena::none) {
        const float3 free_velocity = velocity;
        project_gallery_contact(position, velocity, 0.006F, arena);
        course_forces[vertex] = multiply(subtract(velocity, free_velocity),
            1.0F / (inverse_mass * dt));
    }
    if (!finite3(position) || !finite3(velocity)) {
        positions[vertex] = old_position;
        velocities[vertex] = {};
        atomicAdd(statistics + 2U, 1U);
        return;
    }
    positions[vertex] = position;
    velocities[vertex] = velocity;
    atomicMax(statistics + 6U, __float_as_uint(length(force)));
}

__global__ void integrate_rectangle_kernel(
    DeviceRectangleState* rectangle,
    const float3* reaction_force,
    const float* reaction_torque,
    float3 control_force,
    float control_torque,
    HybridOptions options)
{
    if (blockIdx.x != 0U || threadIdx.x != 0U) return;
    DeviceRectangleState box = *rectangle;
    float3 total_force = add(*reaction_force, control_force);
    float total_torque = *reaction_torque + control_torque;
    float3 acceleration = multiply(total_force, 1.0F / options.rectangle_mass);
    acceleration = clamp_length(acceleration, 20.0F);
    const float angular_acceleration = fminf(20.0F, fmaxf(-20.0F,
        total_torque / options.rectangle_inertia));
    box.velocity = add(box.velocity, multiply(acceleration, options.fixed_dt));
    box.velocity = multiply(box.velocity, expf(-0.35F * options.fixed_dt));
    box.velocity = clamp_length(box.velocity, options.maximum_rectangle_speed);
    box.angular_velocity += angular_acceleration * options.fixed_dt;
    box.angular_velocity *= expf(-0.5F * options.fixed_dt);
    box.angular_velocity = fminf(options.maximum_rectangle_angular_speed,
        fmaxf(-options.maximum_rectangle_angular_speed, box.angular_velocity));
    box.center = add(box.center, multiply(box.velocity, options.fixed_dt));
    box.yaw += box.angular_velocity * options.fixed_dt;
    if (!finite3(box.center) || !finite3(box.velocity) ||
        !isfinite(box.yaw) || !isfinite(box.angular_velocity)) {
        box.center = make_float3(1.45F, 0.0F, 0.0F);
        box.velocity = {};
        box.yaw = 0.0F;
        box.angular_velocity = 0.0F;
    }
    *rectangle = box;
}

// Rotation-aware soft shape matching is a course gameplay constraint, not a
// CFD or incompressibility solve. A radius-only constraint permits the whole
// mesh to collapse into an overlapping ring; retaining angular structure is
// necessary for a rollable shell. The lab does not use this constraint.
__global__ void course_rotation_kernel(
    const float3* positions, const float3* rest, const float3* sum,
    std::uint32_t count, float4* rotation)
{
    __shared__ float3 rows[3][128];
    const unsigned lane = threadIdx.x;
    const float3 center = multiply(*sum, 1.0F / static_cast<float>(count));
    float3 a{}, b{}, c{};
    for (unsigned i = lane; i < count; i += blockDim.x) {
        const float3 d = subtract(positions[i], center);
        a = add(a, multiply(rest[i], d.x));
        b = add(b, multiply(rest[i], d.y));
        c = add(c, multiply(rest[i], d.z));
    }
    rows[0][lane] = a; rows[1][lane] = b; rows[2][lane] = c;
    __syncthreads();
    for (unsigned stride = 64; stride != 0; stride >>= 1U) {
        if (lane < stride) {
            for (unsigned i = 0; i < 3; ++i)
                rows[i][lane] = add(rows[i][lane], rows[i][lane + stride]);
        }
        __syncthreads();
    }
    if (lane == 0) *rotation = best_fit_rotation(rows[0][0], rows[1][0], rows[2][0]);
}

__global__ void course_shape_kernel(
    float3* positions, float3* velocities, const float3* rest,
    const float3* sum, const float4* rotation,
    std::uint32_t count, float inverse_mass, float dt,
    float maximum_speed, float3* contact_forces, std::uint32_t* statistics)
{
    const std::uint32_t vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex >= count) return;
    const float3 center = multiply(*sum, 1.0F / static_cast<float>(count));
    const float3 goal = add(center, rotate_course(*rotation, rest[vertex]));
    const float strength = 1.0F - expf(-30.0F * dt);
    const float3 correction = clamp_length(
        multiply(subtract(goal, positions[vertex]), strength), maximum_speed * dt);
    const float3 velocity = clamp_length(add(velocities[vertex],
        multiply(correction, 1.0F / dt)), maximum_speed);
    // Limit the combined integration/shape velocity before deriving the added
    // displacement. Clipping only the saved velocity would permit a second
    // full-speed translation that was absent from the reported state.
    float3 position = add(positions[vertex],
        multiply(subtract(velocity, velocities[vertex]), dt));
    float3 contact_velocity = velocity;
    const float3 free_velocity = velocity;
    project_course_contact(position, contact_velocity, 0.006F);
    if (!finite3(position) || !finite3(contact_velocity)) {
        atomicAdd(statistics + 2U, 1U);
        return;
    }
    contact_forces[vertex] = add(contact_forces[vertex], multiply(
        subtract(contact_velocity, free_velocity), 1.0F / (inverse_mass * dt)));
    positions[vertex] = position;
    velocities[vertex] = contact_velocity;
}

// The authored post surface is the collision surface. A contact reaction is
// distributed through the same four-voxel bindings that deform each render
// vertex, so the visible post and the simulated lattice cannot drift apart.
__global__ void skin_soft_body_contact_kernel(
    float3* skin_positions,
    float3* skin_velocities,
    std::uint32_t skin_count,
    SoftBodyVoxelView voxels,
    SoftBodyRenderView render,
    float skin_inverse_mass,
    float maximum_contact_distance,
    float maximum_position_correction,
    float dt,
    GalleryArena arena,
    float source_radius,
    bool one_sided_cloth,
    std::uint32_t* reaction_keys,
    float3* reaction_corrections,
    std::uint32_t* statistics)
{
    const std::uint32_t vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex >= skin_count) return;
    constexpr std::uint32_t records_per_contact = 12U;
    const std::uint32_t record_base = records_per_contact * vertex;
    for (std::uint32_t record = 0U; record < records_per_contact; ++record) {
        reaction_keys[record_base + record] = UINT32_MAX;
        reaction_corrections[record_base + record] = {};
    }
    if (render.node_count == 0U || render.positions == nullptr ||
        render.triangles == nullptr || render.bindings == nullptr ||
        voxels.positions == nullptr || voxels.flags == nullptr ||
        voxels.position_corrections == nullptr) return;

    const float3 point = skin_positions[vertex];
    float closest_squared = one_sided_cloth
        ? source_radius * source_radius : INFINITY;
    std::uint32_t closest_triangle = UINT32_MAX;
    float3 closest_weights{};
    float3 closest_point{};
    float highest_cloth_y = -INFINITY;
    std::uint32_t stack[128];
    std::uint32_t stack_size = 1U;
    stack[0] = 0U;
    while (stack_size != 0U) {
        const std::uint32_t node_index = stack[--stack_size];
        if (node_index >= render.node_count) continue;
        const meshprep::HierarchyNode node = render.nodes[node_index];
        const float node_distance = one_sided_cloth
            ? bounds_xz_distance_squared(point, node)
            : bounds_distance_squared(point, node);
        if (node_distance > closest_squared) continue;
        if (node.is_leaf()) {
            for (std::uint32_t item = 0U; item < node.primitive_count; ++item) {
                const std::uint32_t triangle_id =
                    render.primitive_indices[node.first_primitive + item];
                if (triangle_id >= render.triangle_count ||
                    (render.triangle_active != nullptr &&
                     render.triangle_active[triangle_id] == 0U)) continue;
                const uint3 triangle = render.triangles[triangle_id];
                if (triangle.x >= render.vertex_count || triangle.y >= render.vertex_count ||
                    triangle.z >= render.vertex_count) continue;
                float3 barycentric{};
                if (one_sided_cloth) {
                    if (!cloth_xz_weights(point, render.positions[triangle.x],
                            render.positions[triangle.y],
                            render.positions[triangle.z], barycentric)) continue;
                } else {
                    barycentric = closest_triangle_barycentric(
                        point, render.positions[triangle.x], render.positions[triangle.y],
                        render.positions[triangle.z]);
                }
                const float3 candidate = add(multiply(render.positions[triangle.x], barycentric.x),
                    add(multiply(render.positions[triangle.y], barycentric.y),
                        multiply(render.positions[triangle.z], barycentric.z)));
                const float3 delta = subtract(point, candidate);
                const float distance_squared = dot(delta, delta);
                if (one_sided_cloth) {
                    if (candidate.y > highest_cloth_y ||
                        (candidate.y == highest_cloth_y &&
                         triangle_id < closest_triangle)) {
                        highest_cloth_y = candidate.y;
                        closest_triangle = triangle_id;
                        closest_weights = barycentric;
                        closest_point = candidate;
                    }
                } else if (distance_squared < closest_squared ||
                    (distance_squared == closest_squared && triangle_id < closest_triangle)) {
                    closest_squared = distance_squared;
                    closest_triangle = triangle_id;
                    closest_weights = barycentric;
                    closest_point = candidate;
                }
            }
            continue;
        }
        std::uint32_t nearest_child = UINT32_MAX;
        float nearest_distance = INFINITY;
        for (std::uint32_t child = 0U; child < node.child_count; ++child) {
            const std::uint32_t child_index = node.first_child + child;
            const float distance = one_sided_cloth
                ? bounds_xz_distance_squared(point, render.nodes[child_index])
                : bounds_distance_squared(point, render.nodes[child_index]);
            if (distance < nearest_distance ||
                (distance == nearest_distance && child_index < nearest_child)) {
                nearest_distance = distance;
                nearest_child = child_index;
            }
        }
        for (std::uint32_t child = 0U; child < node.child_count; ++child) {
            const std::uint32_t child_index = node.first_child + child;
            if (child_index == nearest_child) continue;
            if (stack_size < 128U) stack[stack_size++] = child_index;
            else atomicAdd(statistics + 2U, 1U);
        }
        if (nearest_child != UINT32_MAX) {
            if (stack_size < 128U) stack[stack_size++] = nearest_child;
            else atomicAdd(statistics + 2U, 1U);
        }
    }
    if (closest_triangle == UINT32_MAX) return;

    if (one_sided_cloth) {
        // A catch cloth is a unilateral surface. A particle that already lies
        // below it must not be teleported to the visible/front side merely
        // because its XZ projection overlaps a triangle. Only resolve a
        // current contact or a crossing whose previous point was above.
        const float previous_y = point.y - skin_velocities[vertex].y * dt;
        if (point.y < highest_cloth_y - source_radius &&
            previous_y < highest_cloth_y - source_radius) return;
    }

    // A closest triangle is only a collision candidate within one bounded
    // relative-motion step. Without this finite support, the signed plane of a
    // distant post triangle acts as an infinite half-space and can teleport the
    // entire droplet onto the obstacle course.
    if (!one_sided_cloth &&
        closest_squared > maximum_contact_distance * maximum_contact_distance) return;

    const uint3 triangle = render.triangles[closest_triangle];
    float3 normal = cross(subtract(render.positions[triangle.y], render.positions[triangle.x]),
        subtract(render.positions[triangle.z], render.positions[triangle.x]));
    const float normal_length = length(normal);
    if (!(normal_length > 1.0e-10F) || !finite3(normal)) return;
    normal = multiply(normal, 1.0F / normal_length);
    if (one_sided_cloth && normal.y < 0.0F)
        normal = multiply(normal, -1.0F);
    // Winding alone is not a reliable outward direction after a post bends.
    // The bound voxel blend lies beneath the authored render surface and gives
    // each contact a deformation-following outward hint.
    float3 support_point{};
    float support_weight = 0.0F;
    const std::uint32_t support_vertices[3]{triangle.x, triangle.y, triangle.z};
    const float support_triangle_weights[3]{
        closest_weights.x, closest_weights.y, closest_weights.z};
    for (std::uint32_t corner = 0U; corner < 3U; ++corner) {
        const SoftBodyBinding binding = render.bindings[support_vertices[corner]];
        const std::uint32_t ids[4]{
            binding.voxels.x, binding.voxels.y, binding.voxels.z, binding.voxels.w};
        const float weights[4]{
            binding.weights.x, binding.weights.y, binding.weights.z, binding.weights.w};
        for (std::uint32_t slot = 0U; slot < 4U; ++slot) {
            if (ids[slot] >= voxels.voxel_count) continue;
            const float weight = support_triangle_weights[corner] * weights[slot];
            support_point = add(support_point, multiply(voxels.positions[ids[slot]], weight));
            support_weight += weight;
        }
    }
    if (support_weight > 1.0e-8F && !one_sided_cloth) {
        support_point = multiply(support_point, 1.0F / support_weight);
        if (dot(normal, subtract(closest_point, support_point)) < 0.0F) {
            normal = multiply(normal, -1.0F);
        }
    }
    const float contact_thickness = source_radius;
    const float signed_distance = dot(subtract(point, closest_point), normal);
    if (signed_distance >= contact_thickness) return;
    const float penetration = contact_thickness - signed_distance;

    const std::uint32_t render_vertices[3]{triangle.x, triangle.y, triangle.z};
    const float triangle_weights[3]{closest_weights.x, closest_weights.y, closest_weights.z};
    std::uint32_t voxel_ids[records_per_contact];
    float voxel_weights[records_per_contact];
    std::uint32_t unique_count = 0U;
    for (std::uint32_t corner = 0U; corner < 3U; ++corner) {
        const SoftBodyBinding binding = render.bindings[render_vertices[corner]];
        const std::uint32_t ids[4]{
            binding.voxels.x, binding.voxels.y, binding.voxels.z, binding.voxels.w};
        const float weights[4]{
            binding.weights.x, binding.weights.y, binding.weights.z, binding.weights.w};
        for (std::uint32_t slot = 0U; slot < 4U; ++slot) {
            if (ids[slot] >= voxels.voxel_count) continue;
            const float weight = triangle_weights[corner] * weights[slot];
            if (!(weight > 0.0F)) continue;
            std::uint32_t unique = 0U;
            while (unique < unique_count && voxel_ids[unique] != ids[slot]) ++unique;
            if (unique == unique_count) {
                voxel_ids[unique_count] = ids[slot];
                voxel_weights[unique_count] = 0.0F;
                ++unique_count;
            }
            voxel_weights[unique] += weight;
        }
    }
    if (unique_count == 0U) return;

    // The supported cloth has many fluid particles above a much lighter
    // lattice. At contact it should yield, but not become an effectively
    // massless trapdoor that lets a heavy particle pass through in one tick.
    const float mesh_inverse_mass = voxels.inverse_voxel_mass *
        (one_sided_cloth ? 0.01F : 1.0F);
    float surface_inverse_mass = 0.0F;
    for (std::uint32_t slot = 0U; slot < unique_count; ++slot) {
        const std::uint32_t voxel = voxel_ids[slot];
        const float weight = voxel_weights[slot];
        if ((voxels.flags[voxel] & soft_body_voxel_pinned) == 0U) {
            surface_inverse_mass += weight * weight * mesh_inverse_mass;
        }
    }
    const float inverse_mass_sum = skin_inverse_mass + surface_inverse_mass;
    if (!(inverse_mass_sum > 0.0F)) return;

    // Solve one inverse-mass weighted unilateral position constraint. The
    // connected water skin and the deformable post share the same multiplier;
    // the lighter post therefore yields instead of forcing one coarse water
    // vertex through nearly a whole incident edge.
    // The capture first contacted 7.86 cm deep and previously transferred a
    // five-centimetre correction in one substep. Limit one contact transfer to
    // a small fraction of a voxel; persistent overlap resolves over iterations
    // without launching the receiving graph at its speed cap.
    const float contact_transfer_limit = one_sided_cloth
        ? penetration
        : fminf(maximum_position_correction,
            fmaxf(0.001F, 0.20F * voxels.voxel_radius));
    const float constrained_distance = fminf(penetration, contact_transfer_limit);
    const float multiplier = constrained_distance / inverse_mass_sum;
    float3 corrected_position = add(
        point, multiply(normal, skin_inverse_mass * multiplier));
    float3 corrected_velocity = skin_velocities[vertex];
    // Post contact is sequenced after ordinary course projection. Re-apply the
    // immutable board/rail constraints so an oblique normal near a mounted
    // post base cannot push the water skin through the floor in the same tick.
    project_gallery_contact(corrected_position, corrected_velocity,
        source_radius, arena);
    const float3 skin_correction = subtract(corrected_position, point);
    float3 surface_velocity{};
    for (std::uint32_t slot = 0U; slot < unique_count; ++slot) {
        surface_velocity = add(surface_velocity, multiply(
            voxels.velocities[voxel_ids[slot]], voxel_weights[slot]));
    }
    const float inward_relative_speed = dot(
        subtract(corrected_velocity, surface_velocity), normal);
    if (inward_relative_speed < 0.0F) {
        corrected_velocity = subtract(
            corrected_velocity, multiply(normal, inward_relative_speed));
    }
    corrected_velocity = clamp_length(corrected_velocity,
        maximum_position_correction / dt);
    skin_positions[vertex] = corrected_position;
    skin_velocities[vertex] = corrected_velocity;
    for (std::uint32_t slot = 0U; slot < unique_count; ++slot) {
        const std::uint32_t voxel = voxel_ids[slot];
        if ((voxels.flags[voxel] & soft_body_voxel_pinned) != 0U) continue;
        reaction_keys[record_base + slot] = voxel_ids[slot];
        reaction_corrections[record_base + slot] = multiply(normal,
            -mesh_inverse_mass * voxel_weights[slot] * multiplier);
    }
    atomicAdd(statistics + 11U, 1U);
    atomicMax(statistics + 12U, __float_as_uint(penetration));
}

__global__ void apply_soft_body_corrections_kernel(
    SoftBodyVoxelView voxels,
    const std::uint32_t* vertices,
    const float3* corrections,
    const std::uint32_t* run_count,
    std::uint32_t maximum_runs,
    const std::uint32_t* sorted_vertices,
    std::uint32_t record_count,
    float maximum_correction)
{
    const std::uint32_t run = blockIdx.x * blockDim.x + threadIdx.x;
    if (run >= maximum_runs || run >= *run_count) return;
    const std::uint32_t voxel = vertices[run];
    if (voxel < voxels.voxel_count) {
        // All constraints were generated from the same pre-contact state.
        // Jacobi therefore averages incident proposals at a shared voxel;
        // summing them would apply N times the intended correction in dense
        // water/post contact patches. The sorted-key run length is recovered
        // deterministically without another allocation or atomic counter.
        std::uint32_t first = 0U;
        std::uint32_t last = record_count;
        while (first < last) {
            const std::uint32_t middle = first + (last - first) / 2U;
            if (sorted_vertices[middle] < voxel) first = middle + 1U;
            else last = middle;
        }
        const std::uint32_t begin = first;
        last = record_count;
        while (first < last) {
            const std::uint32_t middle = first + (last - first) / 2U;
            if (sorted_vertices[middle] <= voxel) first = middle + 1U;
            else last = middle;
        }
        const std::uint32_t contribution_count = first - begin;
        if (contribution_count == 0U) return;
        const float3 correction = clamp_length(
            multiply(corrections[run], 1.0F / static_cast<float>(contribution_count)),
            maximum_correction);
        voxels.position_corrections[voxel] = add(
            voxels.position_corrections[voxel], correction);
    }
}

__global__ void embed_render_surface_kernel(
    const float3* physical_positions,
    const float3* physical_rest_positions,
    const float3* render_rest_positions,
    const SurfaceVertexEmbedding* embedding,
    float3* render_positions,
    std::uint32_t count)
{
    const std::uint32_t vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex >= count) return;
    const SurfaceVertexEmbedding item = embedding[vertex];
    const float3 da = subtract(
        physical_positions[item.source_vertices.x],
        physical_rest_positions[item.source_vertices.x]);
    const float3 db = subtract(
        physical_positions[item.source_vertices.y],
        physical_rest_positions[item.source_vertices.y]);
    const float3 dc = subtract(
        physical_positions[item.source_vertices.z],
        physical_rest_positions[item.source_vertices.z]);
    render_positions[vertex] = add(render_rest_positions[vertex], add(
        multiply(da, item.barycentric.x), add(
        multiply(db, item.barycentric.y), multiply(dc, item.barycentric.z))));
}

float event_elapsed(cudaEvent_t begin, cudaEvent_t end)
{
    float milliseconds = 0.0F;
    check(cudaEventElapsedTime(&milliseconds, begin, end), "measure hybrid stage");
    return milliseconds;
}

constexpr std::uint32_t event_slot(Stage stage, std::uint32_t iteration)
{
    return static_cast<std::uint32_t>(stage) *
        HybridDroplet::maximum_physics_iterations + iteration;
}

bool valid_options(const HybridOptions& options)
{
    return options.physics_iterations >= 1U &&
        options.physics_iterations <= HybridDroplet::maximum_physics_iterations &&
        options.particle_count >= 256U &&
        options.particle_count <= options.particle_capacity &&
        options.particle_capacity <= 100'000U &&
        options.physical_skin_frequency != 0U &&
        options.render_skin_frequency >= options.physical_skin_frequency &&
        options.fixed_dt > 0.0F && options.particle_support_radius > 0.0F &&
        options.particle_repulsion > 0.0F && options.particle_damping >= 0.0F &&
        options.particle_velocity_damping >= 0.0F &&
        options.particle_skin_interaction_radius > options.particle_skin_distance &&
        options.particle_skin_stiffness > 0.0F && options.particle_skin_damping >= 0.0F &&
        options.skin_spring_stiffness > 0.0F && options.skin_spring_damping >= 0.0F &&
        options.skin_velocity_damping >= 0.0F && options.box_skin_stiffness > 0.0F &&
        options.box_skin_damping >= 0.0F &&
        options.box_skin_contact_thickness > 0.0F &&
        options.particle_mass > 0.0F &&
        options.skin_vertex_mass > 0.0F && options.rectangle_mass > 0.0F &&
        options.rectangle_inertia > 0.0F && options.rectangle_target_stiffness > 0.0F &&
        options.rectangle_target_damping_ratio >= 0.0F &&
        options.maximum_rectangle_control_force > 0.0F &&
        options.maximum_particle_force > 0.0F && options.maximum_skin_force > 0.0F &&
        options.maximum_box_ejection_force > 0.0F &&
        std::isfinite(options.maximum_particle_speed) &&
        std::isfinite(options.maximum_skin_speed) &&
        options.maximum_particle_speed > 0.0F && options.maximum_skin_speed > 0.0F &&
        options.maximum_rectangle_speed > 0.0F &&
        options.maximum_rectangle_angular_speed > 0.0F &&
        finite3(options.gravity) && finite3(options.particle_initial_center) &&
        finite3(options.particle_initial_scale) &&
        options.particle_initial_scale.x > 0.0F &&
        options.particle_initial_scale.y > 0.0F &&
        options.particle_initial_scale.z > 0.0F &&
        length(options.gravity) <= 20.0F &&
        (!options.obstacle_course || options.fixed_dt <=
            (0.1F * options.physics_iterations) /
                fmaxf(options.maximum_particle_speed, options.maximum_skin_speed));
}

} // namespace

HybridDroplet::HybridDroplet(HybridOptions options) : options_(options)
{
    if (!valid_options(options_)) {
        throw std::invalid_argument("invalid bounded-force hybrid options");
    }
    const HostSurfaceMesh physical = make_geodesic_sphere(
        options_.physical_skin_frequency, options_.skin_radius);
    const HostSurfaceMesh render = make_geodesic_sphere(
        options_.render_skin_frequency, options_.skin_radius);
    const std::vector<SurfaceVertexEmbedding> embedding =
        make_surface_embedding(physical, render.positions);
    const std::vector<float3> particles = make_hcp_particles(options_);
    skin_vertex_count_ = static_cast<std::uint32_t>(physical.positions.size());
    skin_triangle_count_ = static_cast<std::uint32_t>(physical.triangles.size());
    skin_neighbor_count_ = static_cast<std::uint32_t>(physical.neighbors.size());
    std::vector<std::vector<std::uint32_t>> incident(skin_vertex_count_);
    for (std::uint32_t triangle = 0U; triangle < skin_triangle_count_; ++triangle) {
        const uint3 vertices = physical.triangles[triangle];
        incident[vertices.x].push_back(triangle);
        incident[vertices.y].push_back(triangle);
        incident[vertices.z].push_back(triangle);
    }
    std::vector<std::uint32_t> incident_offsets(skin_vertex_count_ + 1U);
    std::vector<std::uint32_t> incident_triangles;
    incident_triangles.reserve(3U * skin_triangle_count_);
    for (std::uint32_t vertex = 0U; vertex < skin_vertex_count_; ++vertex) {
        incident_offsets[vertex] = static_cast<std::uint32_t>(incident_triangles.size());
        incident_triangles.insert(
            incident_triangles.end(), incident[vertex].begin(), incident[vertex].end());
    }
    incident_offsets[skin_vertex_count_] =
        static_cast<std::uint32_t>(incident_triangles.size());
    skin_incident_count_ = static_cast<std::uint32_t>(incident_triangles.size());
    reaction_record_capacity_ = std::max(
        12U * options_.particle_capacity, 12U * skin_vertex_count_);
    render_vertex_count_ = static_cast<std::uint32_t>(render.positions.size());
    render_triangle_count_ = static_cast<std::uint32_t>(render.triangles.size());

    allocate(particle_positions_, options_.particle_capacity);
    allocate(particle_initial_positions_, options_.particle_capacity);
    allocate(particle_velocities_, options_.particle_capacity);
    allocate(particle_forces_, options_.particle_capacity);
    allocate(particle_skin_owners_, options_.particle_capacity);
    allocate(particle_bounds_, options_.particle_capacity);
    allocate(particle_cell_keys_a_, options_.particle_capacity);
    allocate(particle_cell_keys_b_, options_.particle_capacity);
    allocate(particle_cell_indices_a_, options_.particle_capacity);
    allocate(particle_cell_indices_b_, options_.particle_capacity);
    allocate(skin_positions_, skin_vertex_count_);
    allocate(skin_initial_positions_, skin_vertex_count_);
    allocate(skin_velocities_, skin_vertex_count_);
    allocate(skin_forces_, skin_vertex_count_);
    allocate(skin_box_forces_, skin_vertex_count_);
    allocate(skin_box_impulses_, skin_vertex_count_);
    allocate(skin_box_pair_work_, skin_vertex_count_);
    allocate(skin_particle_forces_, skin_vertex_count_);
    allocate(skin_spring_forces_, skin_vertex_count_);
    allocate(skin_rest_positions_, skin_vertex_count_);
    allocate(course_center_sum_, 1U);
    allocate(course_rotation_, 1U);
    allocate(skin_triangles_, skin_triangle_count_);
    allocate(skin_incident_offsets_, skin_vertex_count_ + 1U);
    allocate(skin_incident_triangles_, skin_incident_count_);
    allocate(skin_neighbor_offsets_, skin_vertex_count_ + 1U);
    allocate(skin_neighbors_, skin_neighbor_count_);
    allocate(skin_rest_lengths_, skin_neighbor_count_);
    allocate(skin_vertex_bounds_, skin_vertex_count_);
    allocate(render_positions_, render_vertex_count_);
    allocate(render_rest_positions_, render_vertex_count_);
    allocate(render_triangles_, render_triangle_count_);
    allocate(render_embedding_, render_vertex_count_);
    allocate(reaction_keys_a_, reaction_record_capacity_);
    allocate(reaction_keys_b_, reaction_record_capacity_);
    allocate(reaction_unique_vertices_, reaction_record_capacity_);
    allocate(reaction_values_a_, reaction_record_capacity_);
    allocate(reaction_values_b_, reaction_record_capacity_);
    allocate(reaction_reduced_values_, reaction_record_capacity_);
    allocate(reaction_run_count_, 1U);
    const std::uint32_t rigid_reaction_capacity = std::max(
        skin_vertex_count_, options_.particle_capacity);
    allocate(rectangle_force_rows_, rigid_reaction_capacity);
    allocate(rectangle_torque_rows_, rigid_reaction_capacity);
    allocate(rectangle_force_sum_, 1U);
    allocate(rectangle_torque_sum_, 1U);
    allocate(device_rectangle_, 1U);
    allocate(device_statistics_, 14U);

    upload(particle_positions_, particles);
    upload(particle_initial_positions_, particles);
    upload(skin_positions_, physical.positions);
    upload(skin_initial_positions_, physical.positions);
    upload(skin_rest_positions_, physical.positions);
    upload(skin_triangles_, physical.triangles);
    upload(skin_incident_offsets_, incident_offsets);
    upload(skin_incident_triangles_, incident_triangles);
    upload(skin_neighbor_offsets_, physical.neighbor_offsets);
    upload(skin_neighbors_, physical.neighbors);
    upload(skin_rest_lengths_, physical.rest_lengths);
    upload(render_positions_, render.positions);
    upload(render_rest_positions_, render.positions);
    upload(render_triangles_, render.triangles);
    upload(render_embedding_, embedding);

    std::size_t sort_bytes = 0U;
    check(cub::DeviceRadixSort::SortPairs(
        nullptr, sort_bytes, reaction_keys_a_, reaction_keys_b_,
        reaction_values_a_, reaction_values_b_, static_cast<int>(reaction_record_capacity_)),
        "query reaction sort storage");
    std::size_t particle_cell_sort_bytes = 0U;
    check(cub::DeviceRadixSort::SortPairs(
        nullptr, particle_cell_sort_bytes, particle_cell_keys_a_, particle_cell_keys_b_,
        particle_cell_indices_a_, particle_cell_indices_b_,
        static_cast<int>(options_.particle_capacity)), "query particle cell sort storage");
    std::size_t reduce_by_key_bytes = 0U;
    check(cub::DeviceReduce::ReduceByKey(
        nullptr, reduce_by_key_bytes, reaction_keys_b_, reaction_unique_vertices_,
        reaction_values_b_, reaction_reduced_values_, reaction_run_count_, AddFloat3{},
        static_cast<int>(reaction_record_capacity_)), "query reaction reduction storage");
    std::size_t force_reduce_bytes = 0U;
    check(cub::DeviceReduce::Reduce(
        nullptr, force_reduce_bytes, rectangle_force_rows_, rectangle_force_sum_,
        static_cast<int>(rigid_reaction_capacity), AddFloat3{},
        make_float3(0.0F, 0.0F, 0.0F)),
        "query rectangle force reduction storage");
    std::size_t torque_reduce_bytes = 0U;
    check(cub::DeviceReduce::Sum(
        nullptr, torque_reduce_bytes, rectangle_torque_rows_, rectangle_torque_sum_,
        static_cast<int>(rigid_reaction_capacity)),
        "query rectangle torque reduction storage");
    cub_storage_bytes_ = std::max(
        {sort_bytes, particle_cell_sort_bytes, reduce_by_key_bytes,
         force_reduce_bytes, torque_reduce_bytes});
    check(cudaMalloc(&cub_storage_, cub_storage_bytes_), "allocate shared CUB storage");
    for (std::uint32_t event = 0U; event < event_count_; ++event) {
        check(cudaEventCreate(stage_begin_ + event), "create stage begin event");
        check(cudaEventCreate(stage_end_ + event), "create stage end event");
    }

    statistics_.particle_count = options_.particle_count;
    statistics_.physical_skin_vertices = skin_vertex_count_;
    statistics_.physical_skin_triangles = skin_triangle_count_;
    statistics_.render_skin_vertices = render_vertex_count_;
    statistics_.render_skin_triangles = render_triangle_count_;
    reset();
}

HybridDroplet::~HybridDroplet()
{
    for (std::uint32_t event = 0U; event < event_count_; ++event) {
        cudaEventDestroy(stage_end_[event]);
        cudaEventDestroy(stage_begin_[event]);
    }
    cudaFree(cub_storage_);
    cudaFree(device_statistics_);
    cudaFree(device_rectangle_);
    cudaFree(rectangle_torque_sum_);
    cudaFree(rectangle_force_sum_);
    cudaFree(rectangle_torque_rows_);
    cudaFree(rectangle_force_rows_);
    cudaFree(reaction_run_count_);
    cudaFree(reaction_reduced_values_);
    cudaFree(reaction_values_b_);
    cudaFree(reaction_values_a_);
    cudaFree(reaction_unique_vertices_);
    cudaFree(reaction_keys_b_);
    cudaFree(reaction_keys_a_);
    cudaFree(render_embedding_);
    cudaFree(render_triangles_);
    cudaFree(render_rest_positions_);
    cudaFree(render_positions_);
    cudaFree(skin_vertex_bounds_);
    cudaFree(skin_rest_lengths_);
    cudaFree(skin_neighbors_);
    cudaFree(skin_neighbor_offsets_);
    cudaFree(skin_incident_triangles_);
    cudaFree(skin_incident_offsets_);
    cudaFree(skin_triangles_);
    cudaFree(skin_rest_positions_);
    cudaFree(course_center_sum_);
    cudaFree(course_rotation_);
    cudaFree(skin_spring_forces_);
    cudaFree(skin_particle_forces_);
    cudaFree(skin_box_pair_work_);
    cudaFree(skin_box_impulses_);
    cudaFree(skin_box_forces_);
    cudaFree(skin_forces_);
    cudaFree(skin_velocities_);
    cudaFree(skin_initial_positions_);
    cudaFree(skin_positions_);
    cudaFree(particle_bounds_);
    cudaFree(particle_cell_indices_b_);
    cudaFree(particle_cell_indices_a_);
    cudaFree(particle_cell_keys_b_);
    cudaFree(particle_cell_keys_a_);
    cudaFree(particle_skin_owners_);
    cudaFree(particle_forces_);
    cudaFree(particle_velocities_);
    cudaFree(particle_initial_positions_);
    cudaFree(particle_positions_);
}

meshprep::DeviceMeshView HybridDroplet::skin_mesh() const noexcept
{
    return {render_positions_, render_vertex_count_, render_triangles_, render_triangle_count_};
}

OrientedBox HybridDroplet::render_box() const noexcept
{
    OrientedBox result;
    result.center = host_rectangle_.center;
    result.half_extents = host_rectangle_.half_extents;
    result.yaw = host_rectangle_.yaw;
    return result;
}

void HybridDroplet::set_runtime_options(const HybridOptions& options)
{
    const bool allocation_shape_changed =
        options.particle_count != options_.particle_count ||
        options.particle_capacity != options_.particle_capacity ||
        options.particle_initial_center.x != options_.particle_initial_center.x ||
        options.particle_initial_center.y != options_.particle_initial_center.y ||
        options.particle_initial_center.z != options_.particle_initial_center.z ||
        options.particle_initial_scale.x != options_.particle_initial_scale.x ||
        options.particle_initial_scale.y != options_.particle_initial_scale.y ||
        options.particle_initial_scale.z != options_.particle_initial_scale.z ||
        options.physical_skin_frequency != options_.physical_skin_frequency ||
        options.render_skin_frequency != options_.render_skin_frequency ||
        options.skin_radius != options_.skin_radius ||
        options.particle_spacing != options_.particle_spacing;
    if (allocation_shape_changed) {
        throw std::invalid_argument(
            "runtime options cannot change initialized geometry");
    }
    if (!valid_options(options)) {
        throw std::invalid_argument("invalid bounded-force runtime options");
    }
    options_ = options;
}

void HybridDroplet::resize_particles(std::uint32_t active_count, cudaStream_t stream)
{
    if (active_count < 256U || active_count > options_.particle_capacity) {
        throw std::invalid_argument("active particle count exceeds initialized capacity");
    }
    if (active_count == options_.particle_count) return;
    const std::uint32_t previous = options_.particle_count;
    if (active_count > previous) {
        initialize_added_particles_kernel<<<
            (active_count - previous + block_size - 1U) / block_size,
            block_size, 0, stream>>>(particle_positions_, particle_initial_positions_,
            particle_velocities_, particle_forces_, particle_skin_owners_,
            previous, active_count);
        check(cudaGetLastError(), "initialize added particles");
    }
    options_.particle_count = active_count;
    statistics_.particle_count = active_count;
    rebuild_particle_broadphase(stream);
    check(cudaStreamSynchronize(stream), "finish particle count change");
}

HybridTimings HybridDroplet::step(
    float3 rectangle_control_force,
    float rectangle_control_torque,
    cudaStream_t stream,
    SoftBodyCourse* soft_bodies,
    bool enable_rectangle_collider,
    RigidSphereState* rigid_sphere,
    WaterWheelState* water_wheel,
    const float3* rigid_sphere_gravity_override)
{
    nvtx3::scoped_range frame{"bounded_force/frame"};
    if (!finite3(rectangle_control_force) || !std::isfinite(rectangle_control_torque)) {
        throw std::invalid_argument("non-finite rectangle control force");
    }
    if (rigid_sphere_gravity_override != nullptr &&
        !finite3(*rigid_sphere_gravity_override)) {
        throw std::invalid_argument("non-finite rigid sphere gravity override");
    }
    if (rigid_sphere != nullptr && (!finite3(rigid_sphere->center) ||
        !finite3(rigid_sphere->velocity) || !finite3(rigid_sphere->angular_velocity) ||
        !std::isfinite(rigid_sphere->orientation.x) ||
        !std::isfinite(rigid_sphere->orientation.y) ||
        !std::isfinite(rigid_sphere->orientation.z) ||
        !std::isfinite(rigid_sphere->orientation.w) ||
        !std::isfinite(rigid_sphere->radius) ||
        !std::isfinite(rigid_sphere->mass) || rigid_sphere->radius <= 0.0F ||
        rigid_sphere->mass <= 0.0F)) {
        throw std::invalid_argument("invalid gallery rigid sphere");
    }
    if (water_wheel != nullptr && (!std::isfinite(water_wheel->angle) ||
        !std::isfinite(water_wheel->angular_velocity) ||
        !std::isfinite(water_wheel->inertia) || water_wheel->inertia <= 0.0F)) {
        throw std::invalid_argument("invalid water wheel state");
    }
    const bool rectangle_collider =
        enable_rectangle_collider && !options_.obstacle_course;
    const std::uint32_t particle_blocks =
        (options_.particle_count + block_size - 1U) / block_size;
    const std::uint32_t skin_blocks =
        (skin_vertex_count_ + block_size - 1U) / block_size;
    const std::uint32_t iterations = options_.physics_iterations;
    const std::uint32_t reaction_record_count = 3U * options_.particle_count;
    HybridOptions substep_options = options_;
    substep_options.fixed_dt = options_.fixed_dt / static_cast<float>(iterations);
    DeviceRectangleState host_device_rectangle{
        host_rectangle_.center, host_rectangle_.velocity, host_rectangle_.half_extents,
        host_rectangle_.yaw, host_rectangle_.angular_velocity};
    if (particle_cell_size_ != options_.particle_support_radius) {
        rebuild_particle_cells(stream);
    }
    check(cudaMemsetAsync(device_statistics_, 0, 14U * sizeof(std::uint32_t), stream),
        "clear frame statistics");
    if (options_.arena == GalleryArena::slope ||
        options_.arena == GalleryArena::water_wheel) {
        const std::uint32_t recycle_event = event_slot(stage_particle_recycling, 0U);
        check(cudaEventRecord(stage_begin_[recycle_event], stream),
            "record slope recycling begin");
        recycle_slope_particles_kernel<<<particle_blocks, block_size, 0, stream>>>(
            particle_positions_, particle_initial_positions_, particle_velocities_,
            particle_forces_, particle_skin_owners_, options_.particle_count,
            options_.arena, device_statistics_ + 13U);
        check(cudaGetLastError(), "recycle slope particles");
        rebuild_particle_cells(stream);
        check(cudaEventRecord(stage_end_[recycle_event], stream),
            "record slope recycling end");
    }
    if (options_.collect_contact_diagnostics) {
        check(cudaMemsetAsync(skin_box_impulses_, 0,
            static_cast<std::size_t>(skin_vertex_count_) * sizeof(float3), stream),
            "clear box contact impulses");
        check(cudaMemsetAsync(skin_box_pair_work_, 0,
            static_cast<std::size_t>(skin_vertex_count_) * sizeof(float), stream),
            "clear box contact pair work");
    }
    if (soft_bodies != nullptr) soft_bodies->begin_frame(stream);
    // Pinned nodes stay fixed, while nodes detached from those anchors fall
    // under the same world gravity as the fluid and water skin.
    const float3 soft_body_gravity = options_.gravity;
    const GalleryArena arena = options_.obstacle_course
        ? GalleryArena::course : options_.arena;

    for (std::uint32_t iteration = 0U; iteration < iterations; ++iteration) {
        if (iteration != 0U) {
            // Keep non-finite failures cumulative, but expose contact and force
            // statistics from the final substep rather than summing samples.
            check(cudaMemsetAsync(device_statistics_, 0, 2U * sizeof(std::uint32_t), stream),
                "clear substep contact statistics");
            check(cudaMemsetAsync(device_statistics_ + 3U, 0, 5U * sizeof(std::uint32_t), stream),
                "clear substep force statistics");
        }
        if (soft_bodies != nullptr) {
            soft_bodies->prepare_substep(
                substep_options.fixed_dt, soft_body_gravity, stream);
        }
        if (rigid_sphere != nullptr) {
            const float3 sphere_gravity = rigid_sphere_gravity_override != nullptr
                ? *rigid_sphere_gravity_override : options_.gravity;
            rigid_sphere->velocity = add(rigid_sphere->velocity,
                multiply(sphere_gravity, substep_options.fixed_dt));
            rigid_sphere->velocity = multiply(rigid_sphere->velocity,
                expf(-0.25F * substep_options.fixed_dt));
            rigid_sphere->center = add(rigid_sphere->center,
                multiply(rigid_sphere->velocity, substep_options.fixed_dt));
            project_gallery_contact(rigid_sphere->center, rigid_sphere->velocity,
                rigid_sphere->radius, arena);
        }
        if (water_wheel != nullptr) {
            const float cross_torque = water_wheel->cross_reaction_torque;
            water_wheel->angular_velocity +=
                -cross_torque * substep_options.fixed_dt / water_wheel->inertia;
            water_wheel->rim_angular_velocity +=
                cross_torque * substep_options.fixed_dt / water_wheel->rim_inertia;
            contact_particles_water_wheel_kernel<<<particle_blocks, block_size, 0, stream>>>(
                particle_positions_, particle_velocities_, options_.particle_count,
                *water_wheel, options_.particle_radius, options_.particle_mass,
                options_.maximum_particle_speed,
                rectangle_torque_rows_);
            check(cudaGetLastError(), "launch fluid/water-wheel contact");
            check(cub::DeviceReduce::Sum(
                cub_storage_, cub_storage_bytes_, rectangle_torque_rows_,
                rectangle_torque_sum_, static_cast<int>(options_.particle_count), stream),
                "reduce fluid/water-wheel torque");
            float torque_impulse{};
            check(cudaMemcpyAsync(&torque_impulse, rectangle_torque_sum_, sizeof(float),
                cudaMemcpyDeviceToHost, stream), "download water-wheel torque");
            check(cudaStreamSynchronize(stream), "finish water-wheel reaction");
            water_wheel->angular_velocity += torque_impulse / water_wheel->inertia;
            water_wheel->angular_velocity *= expf(-0.35F * substep_options.fixed_dt);
            water_wheel->rim_angular_velocity *=
                expf(-water_wheel->rim_drag * substep_options.fixed_dt);
            water_wheel->angular_velocity = fminf(4.0F,
                fmaxf(-4.0F, water_wheel->angular_velocity));
            water_wheel->rim_angular_velocity = fminf(4.0F,
                fmaxf(-4.0F, water_wheel->rim_angular_velocity));
            water_wheel->angle +=
                water_wheel->angular_velocity * substep_options.fixed_dt;
            water_wheel->rim_angle +=
                water_wheel->rim_angular_velocity * substep_options.fixed_dt;
            if (soft_bodies != nullptr) {
                soft_bodies->set_wheel_anchor_rotations(
                    water_wheel_center, water_wheel->angle,
                    water_wheel->rim_angle, stream);
            }
        }

        const std::uint32_t fluid_physics_event = event_slot(stage_fluid_physics, iteration);
        check(cudaEventRecord(stage_begin_[fluid_physics_event], stream),
            "record fluid physics begin");
        particle_forces_kernel<<<particle_blocks, block_size, 0, stream>>>(
            particle_positions_, particle_velocities_, options_.particle_count,
            particle_cell_keys_b_, particle_cell_indices_b_,
            skin_positions_, skin_velocities_, physics_normals_.vertex_normals(),
            skin_vertex_count_, skin_triangle_count_, skin_triangles_, skin_incident_offsets_,
            skin_incident_triangles_,
            skin_vertex_hierarchy_.nodes(), skin_vertex_hierarchy_.primitive_indices(),
            skin_vertex_hierarchy_.statistics().node_count,
            substep_options, particle_forces_,
            reaction_keys_a_, reaction_values_a_, particle_skin_owners_,
            device_statistics_);
        integrate_particles_kernel<<<particle_blocks, block_size, 0, stream>>>(
            particle_positions_, particle_velocities_, particle_forces_, options_.particle_count,
            1.0F / options_.particle_mass, options_.particle_velocity_damping,
            options_.maximum_particle_speed,
            substep_options.fixed_dt, options_.gravity, arena,
            options_.particle_radius, device_statistics_ + 2U);
        if (rigid_sphere != nullptr) {
            contact_particles_rigid_sphere_kernel<<<particle_blocks, block_size, 0, stream>>>(
                particle_positions_, particle_velocities_, options_.particle_count,
                *rigid_sphere, options_.particle_radius, options_.particle_mass,
                substep_options.fixed_dt, options_.maximum_particle_speed,
                rectangle_force_rows_);
            check(cudaGetLastError(), "launch fluid/rigid-sphere contact");
            check(cub::DeviceReduce::Reduce(
                cub_storage_, cub_storage_bytes_, rectangle_force_rows_,
                rectangle_force_sum_, static_cast<int>(options_.particle_count),
                AddFloat3{}, make_float3(0.0F, 0.0F, 0.0F), stream),
                "reduce fluid/rigid-sphere reactions");
            float3 impulse{};
            check(cudaMemcpyAsync(&impulse, rectangle_force_sum_, sizeof(float3),
                cudaMemcpyDeviceToHost, stream), "download fluid/rigid-sphere reaction");
            check(cudaStreamSynchronize(stream), "finish fluid/rigid-sphere reaction");
            rigid_sphere->velocity = add(rigid_sphere->velocity,
                multiply(impulse, 1.0F / rigid_sphere->mass));
            rigid_sphere->center = add(rigid_sphere->center,
                multiply(impulse, substep_options.fixed_dt / rigid_sphere->mass));
            rigid_sphere->velocity = clamp_length(
                rigid_sphere->velocity, options_.maximum_particle_speed);
            project_gallery_contact(rigid_sphere->center, rigid_sphere->velocity,
                rigid_sphere->radius, arena);
        }
        check(cudaEventRecord(stage_end_[fluid_physics_event], stream),
            "record fluid physics end");

        const std::uint32_t skin_physics_event = event_slot(stage_skin_physics, iteration);
        check(cudaEventRecord(stage_begin_[skin_physics_event], stream),
            "record skin physics begin");
        if (options_.particle_skin_coupling) {
        check(cudaMemsetAsync(skin_particle_forces_, 0,
            static_cast<std::size_t>(skin_vertex_count_) * sizeof(float3), stream),
            "clear particle-to-skin debug forces");
        skin_spring_forces_kernel<<<skin_blocks, block_size, 0, stream>>>(
            skin_positions_, skin_velocities_, skin_neighbor_offsets_, skin_neighbors_,
            skin_rest_lengths_, skin_vertex_count_, options_.skin_spring_stiffness,
            options_.skin_spring_damping, skin_forces_, skin_spring_forces_);
        check(cub::DeviceRadixSort::SortPairs(
            cub_storage_, cub_storage_bytes_, reaction_keys_a_, reaction_keys_b_,
            reaction_values_a_, reaction_values_b_, static_cast<int>(reaction_record_count),
            0, 32, stream), "sort particle-to-skin reactions");
        check(cub::DeviceReduce::ReduceByKey(
            cub_storage_, cub_storage_bytes_, reaction_keys_b_, reaction_unique_vertices_,
            reaction_values_b_, reaction_reduced_values_, reaction_run_count_, AddFloat3{},
            static_cast<int>(reaction_record_count), stream),
            "reduce particle-to-skin reactions");
        apply_skin_reactions_kernel<<<
            (reaction_record_count + block_size - 1U) / block_size,
            block_size, 0, stream>>>(
            skin_forces_, reaction_unique_vertices_, reaction_reduced_values_,
            reaction_run_count_, reaction_record_count, skin_vertex_count_,
            skin_particle_forces_);
        if (rectangle_collider) {
            skin_rectangle_forces_kernel<<<skin_blocks, block_size, 0, stream>>>(
                skin_positions_, skin_velocities_, skin_vertex_count_, device_rectangle_,
                substep_options, skin_forces_, skin_box_forces_, skin_box_impulses_,
                skin_box_pair_work_, rectangle_force_rows_, rectangle_torque_rows_,
                device_statistics_);
        }
        integrate_skin_kernel<<<skin_blocks, block_size, 0, stream>>>(
            skin_positions_, skin_velocities_, skin_forces_, skin_vertex_count_,
            1.0F / options_.skin_vertex_mass, options_.skin_velocity_damping,
            options_.maximum_skin_force, options_.maximum_skin_speed,
            substep_options.fixed_dt, options_.collect_contact_diagnostics,
            options_.gravity, arena, skin_box_forces_,
            device_statistics_);
        if (options_.obstacle_course) {
            check(cub::DeviceReduce::Reduce(
                cub_storage_, cub_storage_bytes_, skin_positions_, course_center_sum_,
                static_cast<int>(skin_vertex_count_), AddFloat3{}, make_float3(0, 0, 0),
                stream), "reduce course skin center");
            course_rotation_kernel<<<1, 128, 0, stream>>>(skin_positions_,
                skin_rest_positions_, course_center_sum_, skin_vertex_count_, course_rotation_);
            course_shape_kernel<<<skin_blocks, block_size, 0, stream>>>(
                skin_positions_, skin_velocities_, skin_rest_positions_, course_center_sum_,
                course_rotation_, skin_vertex_count_, 1.0F / options_.skin_vertex_mass,
                substep_options.fixed_dt, options_.maximum_skin_speed, skin_box_forces_,
                device_statistics_);
        }
        }
        check(cudaEventRecord(stage_end_[skin_physics_event], stream),
            "record skin physics end");

        if (soft_bodies != nullptr) {
            const std::uint32_t soft_contact_event =
                event_slot(stage_soft_body_contact, iteration);
            check(cudaEventRecord(stage_begin_[soft_contact_event], stream),
                "record soft-body contact begin");
            const SoftBodyVoxelView voxels = soft_bodies->voxel_view();
            const SoftBodyRenderView render = soft_bodies->render_view();
            if (render.max_depth > 18U) {
                throw std::runtime_error("soft-body contact hierarchy exceeds stack contract");
            }
            const bool direct_particle_contact = !options_.particle_skin_coupling;
            const std::uint32_t source_count = direct_particle_contact
                ? options_.particle_count : skin_vertex_count_;
            const std::uint32_t contact_record_count = 12U * source_count;
            const std::uint32_t source_blocks = direct_particle_contact
                ? particle_blocks : skin_blocks;
            const float source_radius = direct_particle_contact
                ? options_.particle_radius : 0.006F;
            const float source_speed = direct_particle_contact
                ? options_.maximum_particle_speed : options_.maximum_skin_speed;
            const bool one_sided_cloth = direct_particle_contact &&
                (arena == GalleryArena::ground || arena == GalleryArena::ground_box ||
                 arena == GalleryArena::enclosed_box ||
                 arena == GalleryArena::cloth_basin);
            skin_soft_body_contact_kernel<<<source_blocks, block_size, 0, stream>>>(
                direct_particle_contact ? particle_positions_ : skin_positions_,
                direct_particle_contact ? particle_velocities_ : skin_velocities_,
                source_count, voxels, render,
                1.0F / (direct_particle_contact
                    ? options_.particle_mass : options_.skin_vertex_mass),
                source_radius + 2.0F * source_speed * substep_options.fixed_dt,
                source_speed * substep_options.fixed_dt,
                substep_options.fixed_dt, arena, source_radius,
                one_sided_cloth,
                reaction_keys_a_, reaction_values_a_, device_statistics_);
            check(cudaGetLastError(), "launch water-to-soft-body contacts");
            check(cub::DeviceRadixSort::SortPairs(
                cub_storage_, cub_storage_bytes_, reaction_keys_a_, reaction_keys_b_,
                reaction_values_a_, reaction_values_b_, static_cast<int>(contact_record_count),
                0, 32, stream), "sort water-to-soft-body reactions");
            check(cub::DeviceReduce::ReduceByKey(
                cub_storage_, cub_storage_bytes_, reaction_keys_b_, reaction_unique_vertices_,
                reaction_values_b_, reaction_reduced_values_, reaction_run_count_, AddFloat3{},
                static_cast<int>(contact_record_count), stream),
                "reduce water-to-soft-body reactions");
            apply_soft_body_corrections_kernel<<<
                (contact_record_count + block_size - 1U) / block_size,
                block_size, 0, stream>>>(
                voxels, reaction_unique_vertices_, reaction_reduced_values_,
                reaction_run_count_, contact_record_count, reaction_keys_b_,
                contact_record_count,
                options_.maximum_skin_speed * substep_options.fixed_dt);
            check(cudaGetLastError(), "apply water-to-soft-body corrections");
            if (rigid_sphere != nullptr) {
                soft_bodies->contact_rigid_sphere_substep(
                    *rigid_sphere, substep_options.fixed_dt, stream);
                project_gallery_contact(rigid_sphere->center, rigid_sphere->velocity,
                    rigid_sphere->radius, arena);
                advance_rigid_sphere_rotation(*rigid_sphere, arena,
                    soft_bodies->material().ground_friction,
                    substep_options.fixed_dt);
            }
            check(cudaEventRecord(stage_end_[soft_contact_event], stream),
                "record soft-body contact end");
            soft_bodies->finish_substep(
                substep_options.fixed_dt, soft_body_gravity, stream);
            if (water_wheel != nullptr) {
                water_wheel->cross_reaction_torque =
                    soft_bodies->wheel_rim_reaction_torque(
                        water_wheel_center, stream);
            }
        }

        if (rectangle_collider) {
        const std::uint32_t rectangle_event = event_slot(stage_rectangle_physics, iteration);
        check(cudaEventRecord(stage_begin_[rectangle_event], stream),
            "record rectangle physics begin");
        check(cub::DeviceReduce::Reduce(
            cub_storage_, cub_storage_bytes_, rectangle_force_rows_, rectangle_force_sum_,
            static_cast<int>(skin_vertex_count_), AddFloat3{}, make_float3(0.0F, 0.0F, 0.0F),
            stream), "reduce rectangle reaction force");
        check(cub::DeviceReduce::Sum(
            cub_storage_, cub_storage_bytes_, rectangle_torque_rows_, rectangle_torque_sum_,
            static_cast<int>(skin_vertex_count_), stream), "reduce rectangle reaction torque");
        integrate_rectangle_kernel<<<1U, 1U, 0, stream>>>(
            device_rectangle_, rectangle_force_sum_, rectangle_torque_sum_,
            rectangle_control_force, rectangle_control_torque, substep_options);
        if (iteration + 1U == iterations) {
            check(cudaMemcpyAsync(&host_device_rectangle, device_rectangle_,
                sizeof(DeviceRectangleState), cudaMemcpyDeviceToHost, stream),
                "download rectangle state");
        }
        check(cudaEventRecord(stage_end_[rectangle_event], stream),
            "record rectangle physics end");
        }

        const std::uint32_t fluid_hierarchy_event = event_slot(stage_fluid_hierarchy, iteration);
        check(cudaEventRecord(stage_begin_[fluid_hierarchy_event], stream),
            "record fluid hierarchy begin");
        emit_bounds_kernel<<<particle_blocks, block_size, 0, stream>>>(
            particle_positions_, particle_bounds_, options_.particle_count,
            options_.particle_radius);
        rebuild_particle_cells(stream);
        if (iteration + 1U == iterations) {
            check(meshprep::build_hierarchy(
                meshprep::DeviceAabbView{particle_bounds_, options_.particle_count}, {},
                particle_workspace_, particle_hierarchy_, stream), "rebuild fluid hierarchy");
        } else {
            check(meshprep::refit_hierarchy_unchecked_async(
                meshprep::DeviceAabbView{particle_bounds_, options_.particle_count},
                particle_hierarchy_, stream), "refit fluid hierarchy");
        }
        check(cudaEventRecord(stage_end_[fluid_hierarchy_event], stream),
            "record fluid hierarchy end");

        const std::uint32_t skin_hierarchy_event = event_slot(stage_skin_hierarchy, iteration);
        check(cudaEventRecord(stage_begin_[skin_hierarchy_event], stream),
            "record skin hierarchy begin");
        if (options_.particle_skin_coupling) {
        emit_bounds_kernel<<<skin_blocks, block_size, 0, stream>>>(
            skin_positions_, skin_vertex_bounds_, skin_vertex_count_, 1.0e-6F);
        if (iteration + 1U == iterations) {
            check(meshprep::build_hierarchy(
                meshprep::DeviceAabbView{skin_vertex_bounds_, skin_vertex_count_}, {},
                skin_workspace_, skin_vertex_hierarchy_, stream),
                "rebuild skin vertex hierarchy");
        } else {
            check(meshprep::refit_hierarchy_unchecked_async(
                meshprep::DeviceAabbView{skin_vertex_bounds_, skin_vertex_count_},
                skin_vertex_hierarchy_, stream), "refit skin vertex hierarchy");
        }
        }
        check(cudaEventRecord(stage_end_[skin_hierarchy_event], stream),
            "record skin hierarchy end");

        const std::uint32_t normal_event = event_slot(stage_surface_normals, iteration);
        check(cudaEventRecord(stage_begin_[normal_event], stream),
            "record physical surface normals begin");
        if (options_.particle_skin_coupling) {
        const meshprep::DeviceMeshView physical_mesh{
            skin_positions_, skin_vertex_count_, skin_triangles_, skin_triangle_count_};
        check(meshprep::compute_normals(
            physical_mesh, {}, physics_normal_workspace_, physics_normals_, stream),
            "update physical surface normals");
        }
        check(cudaEventRecord(stage_end_[normal_event], stream),
            "record physical surface normals end");
    }

    SoftBodyTimings soft_body_timings{};
    if (soft_bodies != nullptr) soft_body_timings = soft_bodies->finish_frame(stream);

    const std::uint32_t render_event = event_slot(stage_render_surface, 0U);
    check(cudaEventRecord(stage_begin_[render_event], stream),
        "record render surface begin");
    if (options_.particle_skin_coupling) {
    embed_render_surface_kernel<<<
        (render_vertex_count_ + block_size - 1U) / block_size,
        block_size, 0, stream>>>(
        skin_positions_, skin_rest_positions_, render_rest_positions_, render_embedding_,
        render_positions_, render_vertex_count_);
    check(meshprep::compute_normals(
        skin_mesh(), {}, render_normal_workspace_, render_normals_, stream),
        "update render normals");
    check(meshprep::build_hierarchy(
        skin_mesh(), {}, render_workspace_, render_hierarchy_, stream),
        "update render hierarchy");
    }
    check(cudaEventRecord(stage_end_[render_event], stream), "record render surface end");

    std::uint32_t host_statistics[14]{};
    check(cudaMemcpyAsync(host_statistics, device_statistics_, sizeof(host_statistics),
        cudaMemcpyDeviceToHost, stream), "download frame statistics");
    check(cudaStreamSynchronize(stream), "finish bounded-force frame");

    host_rectangle_.center = host_device_rectangle.center;
    host_rectangle_.velocity = host_device_rectangle.velocity;
    host_rectangle_.half_extents = host_device_rectangle.half_extents;
    host_rectangle_.yaw = host_device_rectangle.yaw;
    host_rectangle_.angular_velocity = host_device_rectangle.angular_velocity;

    ++statistics_.frame_index;
    statistics_.particles_outside = host_statistics[0];
    statistics_.skin_vertices_inside_rectangle = host_statistics[1];
    statistics_.finite_failures = host_statistics[2];
    statistics_.average_particle_neighbors =
        static_cast<float>(host_statistics[3]) / static_cast<float>(options_.particle_count);
    statistics_.maximum_particle_neighbors = host_statistics[4];
    static_assert(sizeof(float) == sizeof(std::uint32_t));
    std::memcpy(&statistics_.maximum_particle_force, host_statistics + 5U, sizeof(float));
    std::memcpy(&statistics_.maximum_skin_force, host_statistics + 6U, sizeof(float));
    std::memcpy(&statistics_.maximum_rectangle_reaction, host_statistics + 7U, sizeof(float));
    statistics_.particle_force_cap_hits = host_statistics[8];
    statistics_.skin_force_cap_hits = host_statistics[9];
    statistics_.box_force_cap_hits = host_statistics[10];
    statistics_.soft_body_contact_count = host_statistics[11];
    std::memcpy(&statistics_.maximum_soft_body_penetration,
        host_statistics + 12U, sizeof(float));
    statistics_.recycled_particles = host_statistics[13];

    HybridTimings timings;
    if (options_.arena == GalleryArena::slope ||
        options_.arena == GalleryArena::water_wheel) {
        const std::uint32_t recycle_event = event_slot(stage_particle_recycling, 0U);
        timings.particle_recycling_ms = event_elapsed(
            stage_begin_[recycle_event], stage_end_[recycle_event]);
    }
    for (std::uint32_t iteration = 0U; iteration < iterations; ++iteration) {
        timings.rebuild_fluid_hierarchy_ms += event_elapsed(
            stage_begin_[event_slot(stage_fluid_hierarchy, iteration)],
            stage_end_[event_slot(stage_fluid_hierarchy, iteration)]);
        timings.rebuild_skin_hierarchy_ms += event_elapsed(
            stage_begin_[event_slot(stage_skin_hierarchy, iteration)],
            stage_end_[event_slot(stage_skin_hierarchy, iteration)]);
        timings.update_fluid_physics_ms += event_elapsed(
            stage_begin_[event_slot(stage_fluid_physics, iteration)],
            stage_end_[event_slot(stage_fluid_physics, iteration)]);
        timings.update_skin_physics_ms += event_elapsed(
            stage_begin_[event_slot(stage_skin_physics, iteration)],
            stage_end_[event_slot(stage_skin_physics, iteration)]);
        if (rectangle_collider) {
            timings.update_rectangle_physics_ms += event_elapsed(
                stage_begin_[event_slot(stage_rectangle_physics, iteration)],
                stage_end_[event_slot(stage_rectangle_physics, iteration)]);
        }
        if (soft_bodies != nullptr) {
            timings.update_soft_body_contact_ms += event_elapsed(
                stage_begin_[event_slot(stage_soft_body_contact, iteration)],
                stage_end_[event_slot(stage_soft_body_contact, iteration)]);
        }
        timings.update_surface_normals_ms += event_elapsed(
            stage_begin_[event_slot(stage_surface_normals, iteration)],
            stage_end_[event_slot(stage_surface_normals, iteration)]);
    }
    timings.update_render_surface_ms = event_elapsed(
        stage_begin_[render_event], stage_end_[render_event]);
    timings.update_soft_body_physics_ms = soft_body_timings.physics_ms;
    timings.rebuild_soft_body_hierarchy_ms = soft_body_timings.render_hierarchy_ms;
    timings.update_soft_body_render_ms = soft_body_timings.render_deformation_ms;
    timings.update_rigid_body_contact_ms = soft_body_timings.rigid_contact_ms;
    return timings;
}

void HybridDroplet::capture_state(HybridState& output, cudaStream_t stream) const
{
    output.options = options_;
    output.rectangle = host_rectangle_;
    output.statistics = statistics_;
    output.particle_positions.resize(options_.particle_count);
    output.particle_velocities.resize(options_.particle_count);
    output.particle_forces.resize(options_.particle_count);
    output.particle_skin_owners.resize(options_.particle_count);
    output.skin_positions.resize(skin_vertex_count_);
    output.skin_velocities.resize(skin_vertex_count_);
    output.skin_forces.resize(skin_vertex_count_);
    output.skin_box_forces.resize(skin_vertex_count_);
    output.skin_box_impulses.resize(skin_vertex_count_);
    output.skin_box_pair_work.resize(skin_vertex_count_);
    output.skin_particle_forces.resize(skin_vertex_count_);
    output.skin_spring_forces.resize(skin_vertex_count_);

    const auto download = [stream](auto& destination, const auto* source, const char* label) {
        check(cudaMemcpyAsync(destination.data(), source,
            destination.size() * sizeof(typename std::decay_t<decltype(destination)>::value_type),
            cudaMemcpyDeviceToHost, stream), label);
    };
    download(output.particle_positions, particle_positions_, "capture particle positions");
    download(output.particle_velocities, particle_velocities_, "capture particle velocities");
    download(output.particle_forces, particle_forces_, "capture particle forces");
    download(output.particle_skin_owners, particle_skin_owners_, "capture particle owners");
    download(output.skin_positions, skin_positions_, "capture skin positions");
    download(output.skin_velocities, skin_velocities_, "capture skin velocities");
    download(output.skin_forces, skin_forces_, "capture skin forces");
    download(output.skin_box_forces, skin_box_forces_, "capture box forces");
    download(output.skin_box_impulses, skin_box_impulses_, "capture box impulses");
    download(output.skin_box_pair_work, skin_box_pair_work_, "capture box pair work");
    download(output.skin_particle_forces, skin_particle_forces_, "capture particle reactions");
    download(output.skin_spring_forces, skin_spring_forces_, "capture spring forces");
    check(cudaStreamSynchronize(stream), "finish hybrid state capture");
}

void HybridDroplet::rebuild_particle_cells(cudaStream_t stream)
{
    const std::uint32_t blocks =
        (options_.particle_count + block_size - 1U) / block_size;
    emit_particle_cells_kernel<<<blocks, block_size, 0, stream>>>(
        particle_positions_, particle_cell_keys_a_, particle_cell_indices_a_,
        options_.particle_count, 1.0F / options_.particle_support_radius);
    check(cub::DeviceRadixSort::SortPairs(
        cub_storage_, cub_storage_bytes_, particle_cell_keys_a_, particle_cell_keys_b_,
        particle_cell_indices_a_, particle_cell_indices_b_,
        static_cast<int>(options_.particle_count), 0, 64, stream),
        "sort particle cells");
    particle_cell_size_ = options_.particle_support_radius;
}

void HybridDroplet::rebuild_particle_broadphase(cudaStream_t stream)
{
    const std::uint32_t blocks =
        (options_.particle_count + block_size - 1U) / block_size;
    emit_bounds_kernel<<<blocks, block_size, 0, stream>>>(
        particle_positions_, particle_bounds_, options_.particle_count,
        options_.particle_radius);
    check(cudaGetLastError(), "emit active particle bounds");
    check(meshprep::build_hierarchy(
        meshprep::DeviceAabbView{particle_bounds_, options_.particle_count}, {},
        particle_workspace_, particle_hierarchy_, stream), "rebuild active fluid hierarchy");
    rebuild_particle_cells(stream);
}

void HybridDroplet::rebuild_derived_state(cudaStream_t stream)
{
    const std::uint32_t skin_blocks =
        (skin_vertex_count_ + block_size - 1U) / block_size;
    rebuild_particle_broadphase(stream);
    emit_bounds_kernel<<<skin_blocks, block_size, 0, stream>>>(
        skin_positions_, skin_vertex_bounds_, skin_vertex_count_, 1.0e-6F);
    check(meshprep::build_hierarchy(
        meshprep::DeviceAabbView{skin_vertex_bounds_, skin_vertex_count_}, {},
        skin_workspace_, skin_vertex_hierarchy_, stream), "rebuild captured skin hierarchy");
    const meshprep::DeviceMeshView physical_mesh{
        skin_positions_, skin_vertex_count_, skin_triangles_, skin_triangle_count_};
    check(meshprep::compute_normals(
        physical_mesh, {}, physics_normal_workspace_, physics_normals_, stream),
        "rebuild captured physical normals");
    embed_render_surface_kernel<<<
        (render_vertex_count_ + block_size - 1U) / block_size,
        block_size, 0, stream>>>(
        skin_positions_, skin_rest_positions_, render_rest_positions_, render_embedding_,
        render_positions_, render_vertex_count_);
    check(meshprep::compute_normals(
        skin_mesh(), {}, render_normal_workspace_, render_normals_, stream),
        "rebuild captured render normals");
    check(meshprep::build_hierarchy(
        skin_mesh(), {}, render_workspace_, render_hierarchy_, stream),
        "rebuild captured render hierarchy");
    check(cudaStreamSynchronize(stream), "finish rebuilding captured state");
}

void HybridDroplet::restore_state(const HybridState& state, cudaStream_t stream)
{
    if (state.particle_positions.size() != options_.particle_count ||
        state.particle_velocities.size() != options_.particle_count ||
        state.particle_forces.size() != options_.particle_count ||
        state.particle_skin_owners.size() != options_.particle_count ||
        state.skin_positions.size() != skin_vertex_count_ ||
        state.skin_velocities.size() != skin_vertex_count_ ||
        state.skin_forces.size() != skin_vertex_count_ ||
        state.skin_box_forces.size() != skin_vertex_count_ ||
        state.skin_box_impulses.size() != skin_vertex_count_ ||
        state.skin_box_pair_work.size() != skin_vertex_count_ ||
        state.skin_particle_forces.size() != skin_vertex_count_ ||
        state.skin_spring_forces.size() != skin_vertex_count_) {
        throw std::invalid_argument("captured hybrid state has incompatible dimensions");
    }
    set_runtime_options(state.options);
    const auto restore = [stream](auto* destination, const auto& source, const char* label) {
        check(cudaMemcpyAsync(destination, source.data(),
            source.size() * sizeof(typename std::decay_t<decltype(source)>::value_type),
            cudaMemcpyHostToDevice, stream), label);
    };
    restore(particle_positions_, state.particle_positions, "restore particle positions");
    restore(particle_velocities_, state.particle_velocities, "restore particle velocities");
    restore(particle_forces_, state.particle_forces, "restore particle forces");
    restore(particle_skin_owners_, state.particle_skin_owners, "restore particle owners");
    restore(skin_positions_, state.skin_positions, "restore skin positions");
    restore(skin_velocities_, state.skin_velocities, "restore skin velocities");
    restore(skin_forces_, state.skin_forces, "restore skin forces");
    restore(skin_box_forces_, state.skin_box_forces, "restore box forces");
    restore(skin_box_impulses_, state.skin_box_impulses, "restore box impulses");
    restore(skin_box_pair_work_, state.skin_box_pair_work, "restore box pair work");
    restore(skin_particle_forces_, state.skin_particle_forces, "restore particle reactions");
    restore(skin_spring_forces_, state.skin_spring_forces, "restore spring forces");
    host_rectangle_ = state.rectangle;
    const DeviceRectangleState device_box{
        host_rectangle_.center, host_rectangle_.velocity, host_rectangle_.half_extents,
        host_rectangle_.yaw, host_rectangle_.angular_velocity};
    check(cudaMemcpyAsync(device_rectangle_, &device_box, sizeof(device_box),
        cudaMemcpyHostToDevice, stream), "restore rectangle state");
    statistics_ = state.statistics;
    rebuild_derived_state(stream);
}

void HybridDroplet::reset(cudaStream_t stream)
{
    check(cudaMemcpyAsync(particle_positions_, particle_initial_positions_,
        static_cast<std::size_t>(options_.particle_count) * sizeof(float3),
        cudaMemcpyDeviceToDevice, stream), "reset particle positions");
    check(cudaMemsetAsync(particle_velocities_, 0,
        static_cast<std::size_t>(options_.particle_count) * sizeof(float3), stream),
        "reset particle velocities");
    check(cudaMemsetAsync(particle_skin_owners_, 0xff,
        static_cast<std::size_t>(options_.particle_count) * sizeof(std::uint32_t), stream),
        "reset particle skin owners");
    check(cudaMemcpyAsync(skin_positions_, skin_initial_positions_,
        static_cast<std::size_t>(skin_vertex_count_) * sizeof(float3),
        cudaMemcpyDeviceToDevice, stream), "reset skin positions");
    check(cudaMemsetAsync(skin_velocities_, 0,
        static_cast<std::size_t>(skin_vertex_count_) * sizeof(float3), stream),
        "reset skin velocities");
    check(cudaMemsetAsync(skin_box_forces_, 0,
        static_cast<std::size_t>(skin_vertex_count_) * sizeof(float3), stream),
        "reset box-to-skin debug forces");
    check(cudaMemsetAsync(skin_box_impulses_, 0,
        static_cast<std::size_t>(skin_vertex_count_) * sizeof(float3), stream),
        "reset box-to-skin debug impulses");
    check(cudaMemsetAsync(skin_box_pair_work_, 0,
        static_cast<std::size_t>(skin_vertex_count_) * sizeof(float), stream),
        "reset box-to-skin pair work");
    check(cudaMemsetAsync(skin_particle_forces_, 0,
        static_cast<std::size_t>(skin_vertex_count_) * sizeof(float3), stream),
        "reset particle-to-skin debug forces");
    check(cudaMemsetAsync(skin_spring_forces_, 0,
        static_cast<std::size_t>(skin_vertex_count_) * sizeof(float3), stream),
        "reset spring debug forces");
    host_rectangle_ = initial_rectangle_;
    DeviceRectangleState device_box{
        host_rectangle_.center, host_rectangle_.velocity, host_rectangle_.half_extents,
        host_rectangle_.yaw, host_rectangle_.angular_velocity};
    check(cudaMemcpyAsync(device_rectangle_, &device_box, sizeof(device_box),
        cudaMemcpyHostToDevice, stream), "reset rectangle state");

    rebuild_derived_state(stream);
    statistics_.frame_index = 0U;
    statistics_.particles_outside = 0U;
    statistics_.skin_vertices_inside_rectangle = 0U;
    statistics_.finite_failures = 0U;
    statistics_.particle_force_cap_hits = 0U;
    statistics_.skin_force_cap_hits = 0U;
    statistics_.box_force_cap_hits = 0U;
    statistics_.soft_body_contact_count = 0U;
    statistics_.maximum_soft_body_penetration = 0.0F;
    statistics_.recycled_particles = 0U;
}

std::size_t HybridDroplet::allocated_bytes() const noexcept
{
    const std::size_t particles = options_.particle_capacity;
    const std::size_t skin = skin_vertex_count_;
    const std::size_t render = render_vertex_count_;
    std::size_t bytes =
        particles * (4U * sizeof(float3) + sizeof(meshprep::Aabb) +
            3U * sizeof(std::uint32_t) + 2U * sizeof(std::uint64_t)) +
        skin * (9U * sizeof(float3) + sizeof(float) + sizeof(meshprep::Aabb)) +
        sizeof(float3) + sizeof(float4) +
        skin_triangle_count_ * sizeof(uint3) +
        (skin_vertex_count_ + 1U) * sizeof(std::uint32_t) +
        skin_incident_count_ * sizeof(std::uint32_t) +
        (skin_vertex_count_ + 1U) * sizeof(std::uint32_t) +
        skin_neighbor_count_ * (sizeof(std::uint32_t) + sizeof(float)) +
        render * (2U * sizeof(float3) + sizeof(SurfaceVertexEmbedding)) +
        render_triangle_count_ * sizeof(uint3) +
        reaction_record_capacity_ *
            (3U * sizeof(std::uint32_t) + 3U * sizeof(float3)) +
        std::max(skin, particles) * (sizeof(float3) + sizeof(float)) +
        cub_storage_bytes_;
    bytes += particle_workspace_.capacity_bytes() + skin_workspace_.capacity_bytes() +
        physics_normal_workspace_.capacity_bytes() + render_workspace_.capacity_bytes() +
        render_normal_workspace_.capacity_bytes();
    bytes += particle_hierarchy_.allocated_bytes() + skin_vertex_hierarchy_.allocated_bytes() +
        render_hierarchy_.allocated_bytes() + physics_normals_.allocated_bytes() +
        render_normals_.allocated_bytes();
    return bytes;
}

} // namespace waterlab
