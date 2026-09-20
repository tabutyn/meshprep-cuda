// SPDX-License-Identifier: MIT
#include "fluid_visuals.hpp"

#include "fluid_surface.cuh"
#include "obstacle_course.hpp"
#include "particle_cells.cuh"

#include <cuda_runtime.h>
#include <nvtx3/nvtx3.hpp>

#include <cmath>
#include <stdexcept>
#include <string>

namespace waterlab {
namespace {

constexpr std::uint32_t block_size = 256U;
constexpr float minimum_foam_radius = 0.006F;
constexpr float maximum_foam_radius = 0.014F;

void check(cudaError_t status, const char* operation)
{
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

void check(parallel_mater::Status status, const char* operation)
{
    if (!status.ok()) {
        throw std::runtime_error(std::string(operation) + ": " + status.message);
    }
}

void check_foam_depth(const parallel_mater::Hierarchy& hierarchy)
{
    if (hierarchy.statistics().max_depth > 18U) {
        throw std::runtime_error("foam hierarchy depth exceeds traversal stack contract");
    }
}

__device__ bool finite3(float3 value)
{
    return isfinite(value.x) && isfinite(value.y) && isfinite(value.z);
}

__device__ bool finite4(float4 value)
{
    return isfinite(value.x) && isfinite(value.y) && isfinite(value.z) && isfinite(value.w);
}

__device__ float3 add(float3 a, float3 b)
{
    return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
}

__device__ float3 subtract(float3 a, float3 b)
{
    return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
}

__device__ float3 multiply(float3 value, float scale)
{
    return make_float3(value.x*scale, value.y*scale, value.z*scale);
}

__device__ float dot(float3 a, float3 b)
{
    return a.x*b.x+a.y*b.y+a.z*b.z;
}

__device__ float smooth(float low, float high, float value)
{
    const float t = fminf(1.0F, fmaxf(0.0F, (value-low)/(high-low)));
    return t*t*(3.0F-2.0F*t);
}

__device__ float bounds_distance_squared(float3 p, const parallel_mater::HierarchyNode& node)
{
    const float x = fmaxf(fmaxf(node.bounds_min.x-p.x, 0.0F), p.x-node.bounds_max.x);
    const float y = fmaxf(fmaxf(node.bounds_min.y-p.y, 0.0F), p.y-node.bounds_max.y);
    const float z = fmaxf(fmaxf(node.bounds_min.z-p.z, 0.0F), p.z-node.bounds_max.z);
    return x*x+y*y+z*z;
}

__device__ std::uint32_t hash32(std::uint32_t value)
{
    value ^= value >> 16U;
    value *= 0x7feb352dU;
    value ^= value >> 15U;
    value *= 0x846ca68bU;
    return value ^ (value >> 16U);
}

__device__ float hash_unit(std::uint32_t value)
{
    return static_cast<float>(hash32(value) >> 8U) * (1.0F/16777216.0F);
}

__device__ bool valid_live_foam(const FoamParticle& foam)
{
    return foam.position_age.w >= 0.0F && foam.velocity_life.w > 0.0F &&
        foam.position_age.w < foam.velocity_life.w && foam.normal_radius.w > 0.0F &&
        finite4(foam.position_age) && finite4(foam.velocity_life) &&
        finite4(foam.normal_radius);
}

__device__ void clear_foam(FoamParticle& foam)
{
    foam.position_age = make_float4(0.0F, 0.0F, 0.0F, -1.0F);
    foam.velocity_life = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
    foam.normal_radius = make_float4(0.0F, 1.0F, 0.0F, 0.0F);
}

__device__ void emit_foam_bounds(const FoamParticle& foam, float3 anchor,
    parallel_mater::Aabb& bounds)
{
    if (!valid_live_foam(foam)) {
        bounds.minimum = anchor;
        bounds.maximum = anchor;
        return;
    }
    const float radius = foam.normal_radius.w;
    const float3 p = make_float3(foam.position_age.x, foam.position_age.y,
        foam.position_age.z);
    bounds.minimum = make_float3(p.x-radius, p.y-radius, p.z-radius);
    bounds.maximum = make_float3(p.x+radius, p.y+radius, p.z+radius);
}

__device__ bool push_children(const parallel_mater::HierarchyNode& node,
    std::uint32_t node_count, std::uint32_t* stack, std::uint32_t& pending)
{
    // Eight-way DFS needs at most 1+7*depth entries (127 at depth 18).
    if (node.child_count > 8U || node.first_child > node_count ||
        node.child_count > node_count-node.first_child || pending+node.child_count > 128U) {
        return false;
    }
    for (std::uint32_t child = 0; child < node.child_count; ++child) {
        stack[pending++] = node.first_child+child;
    }
    return true;
}

__device__ bool accumulate_fluid_velocity(float3 point, const float3* positions,
    const float3* velocities, std::uint32_t particle, std::uint32_t count,
    float support_radius, float3& weighted_velocity, float& weight_sum,
    std::uint32_t* errors)
{
    if (particle >= count) { atomicOr(errors, 2U); return false; }
    const float3 p = positions[particle], v = velocities[particle];
    if (!finite3(p) || !finite3(v)) { atomicOr(errors, 1U); return false; }
    const float3 d = subtract(point, p);
    const float distance = sqrtf(dot(d, d));
    if (distance >= support_radius) return true;
    const float q = 1.0F-distance/support_radius;
    const float weight = q*q;
    weighted_velocity = add(weighted_velocity, multiply(v, weight));
    weight_sum += weight;
    return true;
}

__device__ bool sample_fluid_velocity(float3 point, const float3* positions,
    const float3* velocities, const parallel_mater::HierarchyNode* nodes,
    const std::uint32_t* indices, std::uint32_t node_count, std::uint32_t count,
    ParticleCellView cells, float support_radius, float3& velocity,
    std::uint32_t* errors)
{
    float3 weighted_velocity{};
    float weight_sum = 0.0F;
    if (cells.keys != nullptr) {
        const float inverse_cell_size = 1.0F/cells.cell_size;
        const int cell_x = __float2int_rd(point.x*inverse_cell_size);
        const int cell_y = __float2int_rd(point.y*inverse_cell_size);
        const int cell_z = __float2int_rd(point.z*inverse_cell_size);
        for (int dz=-1; dz<=1; ++dz) {
            for (int dy=-1; dy<=1; ++dy) {
                for (int dx=-1; dx<=1; ++dx) {
                    const std::uint64_t key = detail::particle_cell_key(
                        cell_x+dx, cell_y+dy, cell_z+dz);
                    const std::uint32_t first = detail::particle_cell_lower_bound(cells, key);
                    for (std::uint32_t item=first;
                         item<cells.particle_count && cells.keys[item]==key; ++item) {
                        if (!accumulate_fluid_velocity(point, positions, velocities,
                                cells.indices[item], count, support_radius,
                                weighted_velocity, weight_sum, errors)) return false;
                    }
                }
            }
        }
    } else {
        std::uint32_t stack[128], pending = 1U;
        stack[0] = 0U;
        while (pending != 0U) {
            const std::uint32_t node_id = stack[--pending];
            if (node_id >= node_count) { atomicOr(errors, 2U); return false; }
            const auto node = nodes[node_id];
            if (bounds_distance_squared(point, node) > support_radius*support_radius) continue;
            if (!node.is_leaf()) {
                if (!push_children(node, node_count, stack, pending)) {
                    atomicOr(errors, 2U); return false;
                }
                continue;
            }
            if (node.first_primitive > count || node.primitive_count > count-node.first_primitive) {
                atomicOr(errors, 2U); return false;
            }
            for (std::uint32_t item = 0; item < node.primitive_count; ++item) {
                if (!accumulate_fluid_velocity(point, positions, velocities,
                        indices[node.first_primitive+item], count, support_radius,
                        weighted_velocity, weight_sum, errors)) return false;
            }
        }
    }
    if (!(weight_sum > 1.0e-8F) || !isfinite(weight_sum)) return false;
    velocity = multiply(weighted_velocity, 1.0F/weight_sum);
    return finite3(velocity);
}

__device__ bool project_to_surface(FluidSurfaceView surface, float support_radius,
    float3& position, float3& normal)
{
    const float maximum_step = 0.5F*support_radius;
    for (unsigned iteration = 0; iteration < 3U; ++iteration) {
        const float4 sample = detail::surface_value_gradient(surface, position);
        const float3 gradient = make_float3(sample.x, sample.y, sample.z);
        const float gradient_squared = dot(gradient, gradient);
        if (!finite4(sample) || !(gradient_squared > 1.0e-10F)) return false;
        float distance = sample.w/sqrtf(gradient_squared);
        distance = fminf(maximum_step, fmaxf(-maximum_step, distance));
        position = subtract(position, multiply(gradient, distance/sqrtf(gradient_squared)));
    }
    const float4 sample = detail::surface_value_gradient(surface, position);
    const float3 gradient = make_float3(sample.x, sample.y, sample.z);
    const float gradient_length = sqrtf(dot(gradient, gradient));
    if (!finite4(sample) || !(gradient_length > 1.0e-5F) ||
        fabsf(sample.w)/gradient_length > 0.02F*support_radius) return false;
    normal = multiply(gradient, 1.0F/gradient_length);
    return finite3(position) && finite3(normal);
}

__device__ bool accumulate_distribution_sample(std::uint32_t particle,
    std::uint32_t other, float3 point, float3 velocity, const float3* positions,
    const float3* velocities, std::uint32_t count, float radius,
    float3& imbalance, float& weights, float& relative_speed_squared,
    std::uint32_t* errors)
{
    if (other >= count) { atomicOr(errors, 2U); return false; }
    if (other == particle) return true;
    const float3 q = positions[other], other_velocity = velocities[other];
    if (!finite3(q) || !finite3(other_velocity)) { atomicOr(errors, 1U); return false; }
    const float3 d = subtract(point, q);
    const float distance = sqrtf(dot(d, d));
    if (distance <= 1.0e-8F || distance >= radius) return true;
    const float fraction = 1.0F-distance/radius;
    const float weight = fraction*fraction;
    imbalance = add(imbalance, multiply(d, weight/distance));
    const float3 dv = subtract(velocity, other_velocity);
    relative_speed_squared += weight*dot(dv, dv);
    weights += weight;
    return true;
}

__global__ void update_distribution_normals(const float3* positions,
    const float3* velocities, const parallel_mater::HierarchyNode* nodes,
    const std::uint32_t* indices, std::uint32_t node_count, std::uint32_t count,
    ParticleCellView cells, float radius, float3 up, float4* normal_source,
    std::uint32_t* errors)
{
    const std::uint32_t work_index = blockIdx.x*blockDim.x+threadIdx.x;
    if (work_index >= count) return;
    const std::uint32_t i = cells.keys ? cells.indices[work_index] : work_index;
    if (i >= count) { atomicOr(errors, 2U); return; }
    const float3 p = positions[i], v = velocities[i];
    if (!finite3(p) || !finite3(v)) { atomicOr(errors, 1U); return; }
    float3 imbalance{};
    float weights = 0.0F, relative_speed_squared = 0.0F;
    if (cells.keys != nullptr) {
        const float inverse_cell_size = 1.0F/cells.cell_size;
        const int cell_x = __float2int_rd(p.x*inverse_cell_size);
        const int cell_y = __float2int_rd(p.y*inverse_cell_size);
        const int cell_z = __float2int_rd(p.z*inverse_cell_size);
        for (int dz=-1; dz<=1; ++dz) {
            for (int dy=-1; dy<=1; ++dy) {
                for (int dx=-1; dx<=1; ++dx) {
                    const std::uint64_t key = detail::particle_cell_key(
                        cell_x+dx, cell_y+dy, cell_z+dz);
                    const std::uint32_t first = detail::particle_cell_lower_bound(cells, key);
                    for (std::uint32_t item=first;
                         item<cells.particle_count && cells.keys[item]==key; ++item) {
                        if (!accumulate_distribution_sample(i, cells.indices[item], p, v,
                                positions, velocities, count, radius, imbalance, weights,
                                relative_speed_squared, errors)) return;
                    }
                }
            }
        }
    } else {
        std::uint32_t stack[128], pending = 1U;
        stack[0] = 0U;
        while (pending != 0U) {
            const std::uint32_t node_id = stack[--pending];
            if (node_id >= node_count) { atomicOr(errors, 2U); return; }
            const auto node = nodes[node_id];
            if (bounds_distance_squared(p, node) > radius*radius) continue;
            if (!node.is_leaf()) {
                if (!push_children(node, node_count, stack, pending)) {
                    atomicOr(errors, 2U); return;
                }
                continue;
            }
            if (node.first_primitive > count || node.primitive_count > count-node.first_primitive) {
                atomicOr(errors, 2U); return;
            }
            for (std::uint32_t item = 0; item < node.primitive_count; ++item) {
                if (!accumulate_distribution_sample(i,
                        indices[node.first_primitive+item], p, v, positions, velocities,
                        count, radius, imbalance, weights, relative_speed_squared,
                        errors)) return;
            }
        }
    }
    const float magnitude = sqrtf(dot(imbalance, imbalance));
    float3 normal{};
    if (magnitude > 1.0e-6F) normal = multiply(imbalance, 1.0F/magnitude);
    const float exposure = smooth(0.08F, 0.28F, magnitude/fmaxf(weights, 1.0e-8F));
    const float upper = smooth(0.0F, 0.45F, dot(normal, up));
    const float agitation = smooth(0.015F, 0.20F,
        sqrtf(relative_speed_squared/fmaxf(weights, 1.0e-8F)));
    // Foam is an event diagnostic, not a permanent surface decoration.  An
    // exposed but motionless surface must therefore have exactly zero source.
    // The lower agitation ramp still makes impacts in the bowl easy to see.
    const float source = exposure*upper*agitation;
    if (!finite3(normal) || !isfinite(source) || !isfinite(relative_speed_squared)) {
        atomicOr(errors, 1U); return;
    }
    normal_source[i] = make_float4(normal.x, normal.y, normal.z, source);
}

__global__ void update_foam_particles(const float3* positions, const float3* velocities,
    const float4* normal_source, const parallel_mater::HierarchyNode* fluid_nodes,
    const std::uint32_t* fluid_indices, std::uint32_t fluid_node_count,
    std::uint32_t particle_count, ParticleCellView cells, FluidSurfaceView surface,
    float support_radius, float dt, std::uint64_t tick, bool use_course,
    FoamSettings settings, FoamParticle* particles, parallel_mater::Aabb* bounds,
    float3* anchor_output, std::uint32_t* errors)
{
    const std::uint32_t slot = blockIdx.x*blockDim.x+threadIdx.x;
    if (slot >= FluidVisuals::foam_capacity) return;
    const float3 anchor = fluid_nodes[0].bounds_min;
    if (slot == 0U) *anchor_output = anchor;

    FoamParticle foam = particles[slot];
    const bool was_alive = valid_live_foam(foam);
    if (dt == 0.0F) {
        emit_foam_bounds(foam, anchor, bounds[slot]);
        return;
    }
    if (was_alive) {
        foam.position_age.w += dt;
        if (foam.position_age.w >= foam.velocity_life.w) {
            clear_foam(foam);
        } else {
            float3 position = make_float3(foam.position_age.x, foam.position_age.y,
                foam.position_age.z);
            float3 velocity{};
            if (!sample_fluid_velocity(position, positions, velocities, fluid_nodes,
                    fluid_indices, fluid_node_count, particle_count, cells, support_radius,
                    velocity, errors)) {
                clear_foam(foam);
            } else {
                position = add(position, multiply(velocity, dt));
                float3 normal{};
                if (!project_to_surface(surface, support_radius, position, normal)) {
                    clear_foam(foam);
                } else {
                    const float radius = foam.normal_radius.w;
                    position = add(position, multiply(normal, 0.30F*radius));
                    if (use_course) project_course_contact(position, velocity, radius);
                    const float4 final_sample = detail::surface_value_gradient(surface, position);
                    const float3 final_gradient = make_float3(final_sample.x, final_sample.y,
                        final_sample.z);
                    const float gradient_length = sqrtf(dot(final_gradient, final_gradient));
                    if (!finite3(position) || !finite3(velocity) || !finite4(final_sample) ||
                        !(gradient_length > 1.0e-5F) ||
                        fabsf(final_sample.w)/gradient_length >
                            0.30F*radius+0.02F*support_radius) {
                        clear_foam(foam);
                    } else {
                        normal = multiply(final_gradient, 1.0F/gradient_length);
                        foam.position_age.x = position.x;
                        foam.position_age.y = position.y;
                        foam.position_age.z = position.z;
                        foam.velocity_life.x = velocity.x;
                        foam.velocity_life.y = velocity.y;
                        foam.velocity_life.z = velocity.z;
                        foam.normal_radius = make_float4(normal.x, normal.y, normal.z, radius);
                    }
                }
            }
        }
    } else if (!was_alive && slot < particle_count) {
        const std::uint64_t candidate_key = static_cast<std::uint64_t>(slot)*7919ULL+
            tick*104729ULL;
        const std::uint32_t candidate = static_cast<std::uint32_t>(candidate_key%particle_count);
        const float4 source = normal_source[candidate];
        const std::uint32_t seed = hash32(slot ^ static_cast<std::uint32_t>(tick) ^
            static_cast<std::uint32_t>(tick >> 32U));
        const float probability = fminf(1.0F, settings.emission_rate*source.w*dt);
        if (source.w > 0.0F && hash_unit(seed) < probability) {
            const float3 normal = make_float3(source.x, source.y, source.z);
            const float normal_length = sqrtf(dot(normal, normal));
            const float3 p = positions[candidate], v = velocities[candidate];
            if (!finite3(p) || !finite3(v)) {
                atomicOr(errors, 1U);
            } else if (normal_length > 0.5F) {
                const float radius = settings.radius_scale*(minimum_foam_radius+
                    (maximum_foam_radius-minimum_foam_radius)*hash_unit(seed+0x9e3779b9U));
                const float life = settings.lifetime_scale*(
                    1.5F+2.0F*hash_unit(seed+0x85ebca6bU));
                float3 position = add(p, multiply(normal, 0.06F));
                float3 surface_normal{};
                if (project_to_surface(surface, support_radius, position, surface_normal)) {
                    float3 velocity = v;
                    position = add(position, multiply(surface_normal, 0.30F*radius));
                    if (use_course) project_course_contact(position, velocity, radius);
                    const float4 final_sample = detail::surface_value_gradient(surface, position);
                    const float3 final_gradient = make_float3(final_sample.x, final_sample.y,
                        final_sample.z);
                    const float gradient_length = sqrtf(dot(final_gradient, final_gradient));
                    if (finite3(position) && finite3(velocity) && finite4(final_sample) &&
                        gradient_length > 1.0e-5F && fabsf(final_sample.w)/gradient_length <=
                            0.30F*radius+0.02F*support_radius) {
                        surface_normal = multiply(final_gradient, 1.0F/gradient_length);
                        foam.position_age = make_float4(position.x, position.y, position.z, 0.0F);
                        foam.velocity_life = make_float4(velocity.x, velocity.y, velocity.z, life);
                        foam.normal_radius = make_float4(surface_normal.x, surface_normal.y,
                            surface_normal.z, radius);
                    }
                }
            }
        }
    }
    if (!valid_live_foam(foam)) clear_foam(foam);
    particles[slot] = foam;
    emit_foam_bounds(foam, anchor, bounds[slot]);
}

__global__ void restore_foam_bounds(const FoamParticle* particles, float3 anchor,
    parallel_mater::Aabb* bounds)
{
    const std::uint32_t slot = blockIdx.x*blockDim.x+threadIdx.x;
    if (slot < FluidVisuals::foam_capacity) emit_foam_bounds(particles[slot], anchor, bounds[slot]);
}

} // namespace

FluidVisuals::FluidVisuals(std::uint32_t particle_count, std::uint32_t surface_resolution)
    : surface_(surface_resolution), count_(particle_count), capacity_(particle_count)
{
    if (count_ == 0U) throw std::invalid_argument("fluid visuals require particles");
    try {
        check(cudaMalloc(&normal_foam_, static_cast<std::size_t>(capacity_)*sizeof(float4)),
            "allocate fluid visual normals");
        check(cudaMalloc(&foam_particles_, foam_capacity*sizeof(FoamParticle)),
            "allocate foam particles");
        check(cudaMalloc(&foam_bounds_, foam_capacity*sizeof(parallel_mater::Aabb)),
            "allocate foam bounds");
        check(cudaMalloc(&foam_anchor_, sizeof(float3)), "allocate foam anchor");
        check(cudaMalloc(&errors_, sizeof(std::uint32_t)), "allocate fluid visual audit");
        check(cudaEventCreate(&begin_), "create fluid visual begin event");
        check(cudaEventCreate(&end_), "create fluid visual end event");
        reset();
    } catch (...) {
        if (begin_) cudaEventDestroy(begin_);
        if (end_) cudaEventDestroy(end_);
        cudaFree(errors_);
        cudaFree(foam_anchor_);
        cudaFree(foam_bounds_);
        cudaFree(foam_particles_);
        cudaFree(normal_foam_);
        throw;
    }
}

void FluidVisuals::set_active_count(std::uint32_t count)
{
    if (count == 0U || count > capacity_) {
        throw std::invalid_argument("fluid visuals count exceeds reserved capacity");
    }
    count_ = count;
}

void FluidVisuals::set_foam_settings(FoamSettings settings)
{
    if (!std::isfinite(settings.emission_rate) ||
        !std::isfinite(settings.radius_scale) ||
        !std::isfinite(settings.lifetime_scale) ||
        settings.emission_rate < 0.0F || settings.emission_rate > 64.0F ||
        settings.radius_scale < 0.25F || settings.radius_scale > 4.0F ||
        settings.lifetime_scale < 0.25F || settings.lifetime_scale > 4.0F) {
        throw std::invalid_argument("invalid foam settings");
    }
    foam_settings_ = settings;
}

FluidVisuals::~FluidVisuals()
{
    if (begin_) cudaEventDestroy(begin_);
    if (end_) cudaEventDestroy(end_);
    cudaFree(errors_);
    cudaFree(foam_anchor_);
    cudaFree(foam_bounds_);
    cudaFree(foam_particles_);
    cudaFree(normal_foam_);
}

float FluidVisuals::update(const float3* positions, const float3* velocities,
    const parallel_mater::Hierarchy& hierarchy, float support_radius, float3 gravity,
    float dt, cudaStream_t stream, bool obstacle_course, ParticleCellView cells)
{
    nvtx3::scoped_range range{"waterlab/fluid_visuals"};
    const auto statistics = hierarchy.statistics();
    if (!positions || !velocities || !hierarchy.nodes() || !hierarchy.primitive_indices() ||
        statistics.node_count == 0U || statistics.max_depth > 18U ||
        !std::isfinite(support_radius) || support_radius <= 0.0F ||
        !std::isfinite(dt) || dt < 0.0F || !std::isfinite(gravity.x) ||
        !std::isfinite(gravity.y) || !std::isfinite(gravity.z)) {
        throw std::invalid_argument("invalid fluid visual inputs or hierarchy depth above 18");
    }
    if (!cells.empty()) {
        if (cells.indexed_positions != positions || !cells.keys || !cells.indices ||
            cells.particle_count == 0U || cells.particle_count > capacity_ ||
            !std::isfinite(cells.cell_size) || cells.cell_size != support_radius) {
            throw std::invalid_argument("fluid visual particle cell view mismatch");
        }
        // The borrowed broad-phase view is authoritative for this update.
        // Keeping a second manually synchronized active count made a valid
        // runtime particle resize crash the next rendering pass.
        count_ = cells.particle_count;
    }
    const double gravity_length = std::hypot(static_cast<double>(gravity.x),
        static_cast<double>(gravity.y), static_cast<double>(gravity.z));
    const float3 up = gravity_length > 1.0e-12 ? make_float3(
        static_cast<float>(-gravity.x/gravity_length),
        static_cast<float>(-gravity.y/gravity_length),
        static_cast<float>(-gravity.z/gravity_length)) : make_float3(0.0F, 1.0F, 0.0F);

    check(cudaEventRecord(begin_, stream), "begin fluid visuals");
    (void)surface_.update(positions, count_, hierarchy, support_radius, stream, cells);
    check(cudaMemsetAsync(errors_, 0, sizeof(std::uint32_t), stream),
        "clear fluid visual audit");
    update_distribution_normals<<<
        (count_+block_size-1U)/block_size, block_size, 0, stream>>>(
        positions, velocities, hierarchy.nodes(), hierarchy.primitive_indices(),
        statistics.node_count, count_, cells, support_radius, up, normal_foam_, errors_);
    check(cudaGetLastError(), "launch fluid distribution normals");
    update_foam_particles<<<(foam_capacity+block_size-1U)/block_size, block_size, 0, stream>>>(
        positions, velocities, normal_foam_, hierarchy.nodes(), hierarchy.primitive_indices(),
        statistics.node_count, count_, cells, surface_.view(), support_radius, dt, tick_,
        obstacle_course, foam_settings_, foam_particles_, foam_bounds_, foam_anchor_, errors_);
    check(cudaGetLastError(), "launch foam particles");
    check(parallel_mater::build_hierarchy({foam_bounds_, foam_capacity}, {}, foam_workspace_,
        foam_hierarchy_, stream), "build foam hierarchy");
    check_foam_depth(foam_hierarchy_);
    check(cudaEventRecord(end_, stream), "end fluid visuals");
    std::uint32_t errors{};
    check(cudaMemcpyAsync(&errors, errors_, sizeof(errors), cudaMemcpyDeviceToHost, stream),
        "read fluid visual audit");
    check(cudaStreamSynchronize(stream), "complete fluid visuals");
    if (errors != 0U) {
        throw std::runtime_error("fluid visuals: nonfinite sample or invalid hierarchy traversal");
    }
    if (dt > 0.0F) ++tick_;
    float milliseconds{};
    check(cudaEventElapsedTime(&milliseconds, begin_, end_), "measure fluid visuals");
    return milliseconds;
}

void FluidVisuals::reset(cudaStream_t stream)
{
    tick_ = 0U;
    check(cudaMemsetAsync(normal_foam_, 0, static_cast<std::size_t>(count_)*sizeof(float4),
        stream), "reset fluid visual normals");
    check(cudaMemsetAsync(foam_particles_, 0, foam_capacity*sizeof(FoamParticle), stream),
        "reset foam particles");
    check(cudaMemsetAsync(foam_bounds_, 0, foam_capacity*sizeof(parallel_mater::Aabb), stream),
        "reset foam bounds");
    check(cudaMemsetAsync(foam_anchor_, 0, sizeof(float3), stream), "reset foam anchor");
    check(cudaMemsetAsync(errors_, 0, sizeof(std::uint32_t), stream),
        "reset fluid visual audit");
    check(parallel_mater::build_hierarchy({foam_bounds_, foam_capacity}, {}, foam_workspace_,
        foam_hierarchy_, stream), "build reset foam hierarchy");
    check_foam_depth(foam_hierarchy_);
    check(cudaStreamSynchronize(stream), "complete fluid visual reset");
}

void FluidVisuals::capture(std::vector<float4>& output, cudaStream_t stream) const
{
    output.resize(count_);
    check(cudaMemcpyAsync(output.data(), normal_foam_, output.size()*sizeof(float4),
        cudaMemcpyDeviceToHost, stream), "capture fluid visuals");
    check(cudaStreamSynchronize(stream), "complete fluid visual capture");
}

void FluidVisuals::restore(const std::vector<float4>& input, cudaStream_t stream)
{
    if (input.size() != count_) throw std::invalid_argument("fluid visual snapshot count mismatch");
    for (const auto value : input) {
        if (!std::isfinite(value.x) || !std::isfinite(value.y) || !std::isfinite(value.z) ||
            !std::isfinite(value.w) || value.w < 0.0F || value.w > 1.0F) {
            throw std::invalid_argument("invalid fluid visual snapshot value");
        }
    }
    check(cudaMemcpyAsync(normal_foam_, input.data(), input.size()*sizeof(float4),
        cudaMemcpyHostToDevice, stream), "restore fluid visuals");
    check(cudaStreamSynchronize(stream), "complete fluid visual restore");
}

void FluidVisuals::capture_foam(std::vector<FoamParticle>& output, std::uint64_t& tick,
    cudaStream_t stream) const
{
    output.resize(foam_capacity);
    check(cudaMemcpyAsync(output.data(), foam_particles_, output.size()*sizeof(FoamParticle),
        cudaMemcpyDeviceToHost, stream), "capture foam particles");
    check(cudaStreamSynchronize(stream), "complete foam capture");
    tick = tick_;
}

void FluidVisuals::restore_foam(const std::vector<FoamParticle>& input, std::uint64_t tick,
    cudaStream_t stream)
{
    if (input.size() != foam_capacity) throw std::invalid_argument("foam snapshot count mismatch");
    for (const FoamParticle& foam : input) {
        const bool finite = std::isfinite(foam.position_age.x) &&
            std::isfinite(foam.position_age.y) && std::isfinite(foam.position_age.z) &&
            std::isfinite(foam.position_age.w) && std::isfinite(foam.velocity_life.x) &&
            std::isfinite(foam.velocity_life.y) && std::isfinite(foam.velocity_life.z) &&
            std::isfinite(foam.velocity_life.w) && std::isfinite(foam.normal_radius.x) &&
            std::isfinite(foam.normal_radius.y) && std::isfinite(foam.normal_radius.z) &&
            std::isfinite(foam.normal_radius.w);
        if (!finite || (foam.active() &&
                (!(foam.position_age.w < foam.velocity_life.w) ||
                    !(foam.normal_radius.w > 0.0F)))) {
            throw std::invalid_argument("invalid foam snapshot value");
        }
    }
    check(cudaMemcpyAsync(foam_particles_, input.data(), input.size()*sizeof(FoamParticle),
        cudaMemcpyHostToDevice, stream), "restore foam particles");
    tick_ = tick;
    float3 anchor{};
    check(cudaMemcpyAsync(&anchor, foam_anchor_, sizeof(anchor), cudaMemcpyDeviceToHost, stream),
        "read foam anchor");
    check(cudaStreamSynchronize(stream), "complete foam anchor read");
    restore_foam_bounds<<<(foam_capacity+block_size-1U)/block_size, block_size, 0, stream>>>(
        foam_particles_, anchor, foam_bounds_);
    check(cudaGetLastError(), "launch restored foam bounds");
    check(parallel_mater::build_hierarchy({foam_bounds_, foam_capacity}, {}, foam_workspace_,
        foam_hierarchy_, stream), "build restored foam hierarchy");
    check_foam_depth(foam_hierarchy_);
    check(cudaStreamSynchronize(stream), "complete foam restore");
}

} // namespace waterlab
