// SPDX-License-Identifier: MIT
#include "water_lab.hpp"

#include <cuda_runtime.h>
#include <nvtx3/nvtx3.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace waterlab {
namespace {

constexpr std::uint32_t block_size = 256;
constexpr float pi = 3.14159265358979323846F;

void check(cudaError_t error, const char* operation)
{
    if (error == cudaSuccess) return;
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}

template <typename T>
void allocate(T*& pointer, std::size_t count)
{
    if (count == 0) return;
    check(cudaMalloc(&pointer, count * sizeof(T)), "cudaMalloc");
}

template <typename T>
void upload(T* destination, const std::vector<T>& source)
{
    if (source.empty()) return;
    check(
        cudaMemcpy(destination, source.data(), source.size() * sizeof(T), cudaMemcpyHostToDevice),
        "cudaMemcpy host to device");
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

__host__ __device__ float3 normalize(float3 value)
{
    const float inverse = 1.0F / fmaxf(length(value), 1.0e-20F);
    return multiply(value, inverse);
}

__global__ void accumulate_center_kernel(
    const float3* positions,
    std::uint32_t vertex_count,
    float4* state)
{
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= vertex_count) return;
    const float3 point = positions[index];
    atomicAdd(&state->x, point.x);
    atomicAdd(&state->y, point.y);
    atomicAdd(&state->z, point.z);
}

__global__ void accumulate_volume_kernel(
    const float3* positions,
    const uint3* triangles,
    std::uint32_t triangle_count,
    float4* state)
{
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= triangle_count) return;
    const uint3 triangle = triangles[index];
    const float3 a = positions[triangle.x];
    const float3 b = positions[triangle.y];
    const float3 c = positions[triangle.z];
    atomicAdd(&state->w, dot(a, cross(b, c)) / 6.0F);
}

__global__ void finalize_constraints_kernel(float4* state, std::uint32_t vertex_count)
{
    state->x /= static_cast<float>(vertex_count);
    state->y /= static_cast<float>(vertex_count);
    state->z /= static_cast<float>(vertex_count);
    state->w = fabsf(state->w);
}

__global__ void surface_step_kernel(
    const float3* positions,
    const float3* velocities,
    float3* next_positions,
    float3* next_velocities,
    const std::uint32_t* neighbor_offsets,
    const std::uint32_t* neighbors,
    const float* rest_lengths,
    const float4* constraint_state,
    std::uint32_t vertex_count,
    float rest_volume,
    float dt,
    PhysicsOptions options)
{
    const std::uint32_t vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex >= vertex_count) return;
    const float3 position = positions[vertex];
    const float3 velocity = velocities[vertex];
    float3 acceleration = make_float3(0.0F, 0.0F, 0.0F);

    for (std::uint32_t edge = neighbor_offsets[vertex];
         edge < neighbor_offsets[vertex + 1U];
         ++edge) {
        const std::uint32_t neighbor = neighbors[edge];
        const float3 delta = subtract(positions[neighbor], position);
        const float distance = fmaxf(length(delta), 1.0e-7F);
        const float3 direction = multiply(delta, 1.0F / distance);
        const float stretch = distance - rest_lengths[edge];
        const float relative_speed = dot(subtract(velocities[neighbor], velocity), direction);
        acceleration = add(
            acceleration,
            multiply(direction, options.stiffness * stretch + options.edge_damping * relative_speed));
    }

    const float3 center = make_float3(
        constraint_state->x, constraint_state->y, constraint_state->z);
    const float relative_volume_error =
        (rest_volume - constraint_state->w) / fmaxf(rest_volume, 1.0e-10F);
    acceleration = add(
        acceleration,
        multiply(normalize(subtract(position, center)), options.pressure * relative_volume_error));

    float3 next_velocity = add(velocity, multiply(acceleration, dt));
    next_velocity = multiply(next_velocity, expf(-options.velocity_damping * dt));
    const float speed = length(next_velocity);
    if (speed > options.maximum_speed) {
        next_velocity = multiply(next_velocity, options.maximum_speed / speed);
    }
    next_velocities[vertex] = next_velocity;
    next_positions[vertex] = add(position, multiply(next_velocity, dt));
}

__global__ void impulse_kernel(
    const float3* positions,
    float3* velocities,
    std::uint32_t vertex_count,
    float3 point,
    float3 direction,
    float radius,
    float strength)
{
    const std::uint32_t vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex >= vertex_count) return;
    const float distance_squared = dot(subtract(positions[vertex], point), subtract(positions[vertex], point));
    const float weight = expf(-distance_squared / fmaxf(radius * radius, 1.0e-8F));
    velocities[vertex] = add(velocities[vertex], multiply(direction, strength * weight));
}

struct Ray {
    float3 origin;
    float3 direction;
};

struct Hit {
    float distance;
    std::uint32_t triangle;
    float3 normal;
};

__device__ bool intersect_bounds(
    const Ray& ray,
    const meshprep::HierarchyNode& node,
    float maximum_distance,
    float& near_distance)
{
    float near_value = 0.0F;
    float far_value = maximum_distance;
    const float origin[3]{ray.origin.x, ray.origin.y, ray.origin.z};
    const float direction[3]{ray.direction.x, ray.direction.y, ray.direction.z};
    const float minimum[3]{node.bounds_min.x, node.bounds_min.y, node.bounds_min.z};
    const float maximum[3]{node.bounds_max.x, node.bounds_max.y, node.bounds_max.z};
    for (int axis = 0; axis < 3; ++axis) {
        if (fabsf(direction[axis]) < 1.0e-12F) {
            if (origin[axis] < minimum[axis] || origin[axis] > maximum[axis]) return false;
            continue;
        }
        const float inverse = 1.0F / direction[axis];
        float first = (minimum[axis] - origin[axis]) * inverse;
        float second = (maximum[axis] - origin[axis]) * inverse;
        if (first > second) {
            const float temporary = first;
            first = second;
            second = temporary;
        }
        near_value = fmaxf(near_value, first);
        far_value = fminf(far_value, second);
        if (near_value > far_value) return false;
    }
    near_distance = near_value;
    return true;
}

__device__ bool intersect_triangle(
    const Ray& ray,
    float3 a,
    float3 b,
    float3 c,
    float maximum_distance,
    float& distance,
    float3& normal)
{
    const float3 edge_ab = subtract(b, a);
    const float3 edge_ac = subtract(c, a);
    const float3 p = cross(ray.direction, edge_ac);
    const float determinant = dot(edge_ab, p);
    if (fabsf(determinant) < 1.0e-9F) return false;
    const float inverse = 1.0F / determinant;
    const float3 offset = subtract(ray.origin, a);
    const float u = dot(offset, p) * inverse;
    if (u < 0.0F || u > 1.0F) return false;
    const float3 q = cross(offset, edge_ab);
    const float v = dot(ray.direction, q) * inverse;
    if (v < 0.0F || u + v > 1.0F) return false;
    const float hit_distance = dot(edge_ac, q) * inverse;
    if (hit_distance <= 1.0e-5F || hit_distance >= maximum_distance) return false;
    distance = hit_distance;
    normal = normalize(cross(edge_ab, edge_ac));
    return true;
}

__device__ Hit trace_closest(
    const Ray& ray,
    const float3* positions,
    const uint3* triangles,
    const meshprep::HierarchyNode* nodes,
    const std::uint32_t* primitive_indices,
    std::uint32_t node_count)
{
    Hit hit{1.0e30F, UINT32_MAX, make_float3(0.0F, 0.0F, 0.0F)};
    if (node_count == 0) return hit;
    std::uint32_t stack[128];
    int stack_size = 1;
    stack[0] = 0;
    while (stack_size > 0) {
        const std::uint32_t node_index = stack[--stack_size];
        if (node_index >= node_count) continue;
        const meshprep::HierarchyNode node = nodes[node_index];
        float node_near = 0.0F;
        if (!intersect_bounds(ray, node, hit.distance, node_near)) continue;
        if (node.is_leaf()) {
            for (std::uint32_t i = 0; i < node.primitive_count; ++i) {
                const std::uint32_t triangle_index = primitive_indices[node.first_primitive + i];
                const uint3 triangle = triangles[triangle_index];
                float distance = 0.0F;
                float3 normal{};
                if (intersect_triangle(
                        ray,
                        positions[triangle.x],
                        positions[triangle.y],
                        positions[triangle.z],
                        hit.distance,
                        distance,
                        normal)) {
                    hit = {distance, triangle_index, normal};
                }
            }
            continue;
        }
        std::uint32_t child_indices[8];
        float child_near[8];
        int child_hits = 0;
        for (std::uint32_t i = 0; i < node.child_count; ++i) {
            const std::uint32_t child = node.first_child + i;
            float near_value = 0.0F;
            if (!intersect_bounds(ray, nodes[child], hit.distance, near_value)) continue;
            int insertion = child_hits;
            while (insertion > 0 && child_near[insertion - 1] > near_value) {
                child_indices[insertion] = child_indices[insertion - 1];
                child_near[insertion] = child_near[insertion - 1];
                --insertion;
            }
            child_indices[insertion] = child;
            child_near[insertion] = near_value;
            ++child_hits;
        }
        for (int i = child_hits - 1; i >= 0 && stack_size < 128; --i) {
            stack[stack_size++] = child_indices[i];
        }
    }
    return hit;
}

__host__ __device__ Ray camera_ray(
    Camera camera,
    std::uint32_t pixel_x,
    std::uint32_t pixel_y,
    std::uint32_t width,
    std::uint32_t height)
{
    const float3 forward = normalize(subtract(camera.target, camera.eye));
    const float3 right = normalize(cross(forward, camera.up));
    const float3 up = cross(right, forward);
    const float aspect = static_cast<float>(width) / static_cast<float>(height);
    const float tangent = tanf(camera.vertical_fov_degrees * pi / 360.0F);
    const float x =
        (2.0F * (static_cast<float>(pixel_x) + 0.5F) / static_cast<float>(width) - 1.0F) *
        aspect * tangent;
    const float y =
        (1.0F - 2.0F * (static_cast<float>(pixel_y) + 0.5F) / static_cast<float>(height)) * tangent;
    return {camera.eye, normalize(add(forward, add(multiply(right, x), multiply(up, y))))};
}

__device__ float hash(float value)
{
    const float hashed = sinf(value) * 43758.5453F;
    return hashed - floorf(hashed);
}

__device__ float3 background(float3 direction, float time_seconds)
{
    const float haze = 0.5F + 0.5F * sinf(
        direction.x * 5.3F + direction.y * 8.1F + direction.z * 3.7F + time_seconds * 0.025F);
    float3 color = make_float3(
        0.004F + 0.010F * haze,
        0.006F + 0.018F * haze,
        0.018F + 0.045F * haze);
    const float longitude = atan2f(direction.z, direction.x);
    const float latitude = asinf(fmaxf(-1.0F, fminf(1.0F, direction.y)));
    const float cell_x = floorf((longitude + pi) * 310.0F);
    const float cell_y = floorf((latitude + 0.5F * pi) * 310.0F);
    const float star = hash(cell_x * 17.0F + cell_y * 131.0F);
    if (star > 0.9945F) {
        const float brightness = 0.25F + 0.75F * hash(cell_x * 73.0F + cell_y * 19.0F);
        color = add(color, make_float3(brightness, brightness * 0.94F, brightness * 0.82F));
    }
    return color;
}

__device__ bool refract_direction(float3 incident, float3 normal, float ratio, float3& result)
{
    const float cosine = fminf(-dot(incident, normal), 1.0F);
    const float3 perpendicular = multiply(add(incident, multiply(normal, cosine)), ratio);
    const float parallel_squared = 1.0F - dot(perpendicular, perpendicular);
    if (parallel_squared < 0.0F) return false;
    result = add(perpendicular, multiply(normal, -sqrtf(parallel_squared)));
    return true;
}

__device__ float3 shade_water(
    const Ray& primary,
    const Hit& entry,
    const float3* positions,
    const uint3* triangles,
    const meshprep::HierarchyNode* nodes,
    const std::uint32_t* primitive_indices,
    std::uint32_t node_count,
    float time_seconds)
{
    float3 entry_normal = entry.normal;
    if (dot(primary.direction, entry_normal) > 0.0F) entry_normal = multiply(entry_normal, -1.0F);
    const float cosine = fmaxf(0.0F, -dot(primary.direction, entry_normal));
    constexpr float base_reflectance = 0.02037F;
    const float fresnel = base_reflectance +
        (1.0F - base_reflectance) * powf(1.0F - cosine, 5.0F);
    const float3 reflected = normalize(subtract(
        primary.direction, multiply(entry_normal, 2.0F * dot(primary.direction, entry_normal))));
    const float3 reflection_color = background(reflected, time_seconds);

    float3 inside_direction{};
    if (!refract_direction(primary.direction, entry_normal, 1.0F / 1.333F, inside_direction)) {
        return reflection_color;
    }
    const float3 entry_point = add(primary.origin, multiply(primary.direction, entry.distance));
    const Ray inside_ray{add(entry_point, multiply(inside_direction, 2.0e-4F)), inside_direction};
    const Hit exit = trace_closest(
        inside_ray, positions, triangles, nodes, primitive_indices, node_count);
    if (exit.triangle == UINT32_MAX) {
        return add(multiply(reflection_color, fresnel), make_float3(0.0F, 0.04F, 0.07F));
    }
    float3 opposing_exit_normal = exit.normal;
    if (dot(inside_direction, opposing_exit_normal) > 0.0F) {
        opposing_exit_normal = multiply(opposing_exit_normal, -1.0F);
    }
    float3 outside_direction{};
    if (!refract_direction(inside_direction, opposing_exit_normal, 1.333F, outside_direction)) {
        outside_direction = normalize(subtract(
            inside_direction,
            multiply(opposing_exit_normal, 2.0F * dot(inside_direction, opposing_exit_normal))));
    }
    const float3 transmitted = background(outside_direction, time_seconds);
    const float path_length = exit.distance;
    const float3 absorption = make_float3(
        expf(-0.22F * path_length), expf(-0.075F * path_length), expf(-0.035F * path_length));
    const float3 water_tint = make_float3(0.02F, 0.22F, 0.30F);
    const float3 transmission = add(
        make_float3(
            transmitted.x * absorption.x,
            transmitted.y * absorption.y,
            transmitted.z * absorption.z),
        multiply(water_tint, 1.0F - absorption.x));
    const float rim = powf(1.0F - cosine, 2.0F);
    return add(
        add(multiply(reflection_color, fresnel), multiply(transmission, 1.0F - fresnel)),
        multiply(make_float3(0.22F, 0.55F, 0.72F), 0.12F * rim));
}

__device__ unsigned char to_byte(float value)
{
    return static_cast<unsigned char>(255.0F * sqrtf(fmaxf(0.0F, fminf(1.0F, value))));
}

__global__ void render_kernel(
    uchar4* pixels,
    std::uint32_t width,
    std::uint32_t height,
    Camera camera,
    const float3* positions,
    const uint3* triangles,
    const meshprep::HierarchyNode* nodes,
    const std::uint32_t* primitive_indices,
    std::uint32_t node_count,
    float time_seconds)
{
    const std::uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const Ray ray = camera_ray(camera, x, y, width, height);
    const Hit hit = trace_closest(ray, positions, triangles, nodes, primitive_indices, node_count);
    const float3 color = hit.triangle == UINT32_MAX
        ? background(ray.direction, time_seconds)
        : shade_water(
              ray,
              hit,
              positions,
              triangles,
              nodes,
              primitive_indices,
              node_count,
              time_seconds);
    pixels[static_cast<std::size_t>(y) * width + x] =
        make_uchar4(to_byte(color.x), to_byte(color.y), to_byte(color.z), 255);
}

__global__ void pick_kernel(
    PickResult* result,
    std::uint32_t pixel_x,
    std::uint32_t pixel_y,
    std::uint32_t width,
    std::uint32_t height,
    Camera camera,
    const float3* positions,
    const uint3* triangles,
    const meshprep::HierarchyNode* nodes,
    const std::uint32_t* primitive_indices,
    std::uint32_t node_count)
{
    const Ray ray = camera_ray(camera, pixel_x, pixel_y, width, height);
    const Hit hit = trace_closest(ray, positions, triangles, nodes, primitive_indices, node_count);
    if (hit.triangle == UINT32_MAX) {
        *result = {};
        return;
    }
    *result = {true, add(ray.origin, multiply(ray.direction, hit.distance)), hit.normal};
}

} // namespace

WaterSurface::WaterSurface(const HostSurfaceMesh& mesh)
    : vertex_count_(static_cast<std::uint32_t>(mesh.positions.size())),
      triangle_count_(static_cast<std::uint32_t>(mesh.triangles.size())),
      directed_edge_count_(static_cast<std::uint32_t>(mesh.neighbors.size())),
      rest_volume_(mesh.rest_volume)
{
    if (mesh.neighbor_offsets.size() != mesh.positions.size() + 1U ||
        mesh.neighbors.size() != mesh.rest_lengths.size()) {
        throw std::invalid_argument("invalid surface adjacency");
    }
    allocate(positions_a_, vertex_count_);
    allocate(positions_b_, vertex_count_);
    allocate(velocities_a_, vertex_count_);
    allocate(velocities_b_, vertex_count_);
    allocate(rest_positions_, vertex_count_);
    allocate(triangles_, triangle_count_);
    allocate(neighbor_offsets_, mesh.neighbor_offsets.size());
    allocate(neighbors_, directed_edge_count_);
    allocate(rest_lengths_, directed_edge_count_);
    allocate(constraint_state_, 1);
    upload(positions_a_, mesh.positions);
    upload(positions_b_, mesh.positions);
    upload(rest_positions_, mesh.positions);
    upload(triangles_, mesh.triangles);
    upload(neighbor_offsets_, mesh.neighbor_offsets);
    upload(neighbors_, mesh.neighbors);
    upload(rest_lengths_, mesh.rest_lengths);
    check(cudaMemset(velocities_a_, 0, sizeof(float3) * vertex_count_), "clear velocities");
    check(cudaMemset(velocities_b_, 0, sizeof(float3) * vertex_count_), "clear velocities");
    check(cudaEventCreate(&step_begin_), "create physics event");
    check(cudaEventCreate(&step_end_), "create physics event");
}

WaterSurface::~WaterSurface()
{
    cudaEventDestroy(step_end_);
    cudaEventDestroy(step_begin_);
    cudaFree(constraint_state_);
    cudaFree(rest_lengths_);
    cudaFree(neighbors_);
    cudaFree(neighbor_offsets_);
    cudaFree(triangles_);
    cudaFree(rest_positions_);
    cudaFree(velocities_b_);
    cudaFree(velocities_a_);
    cudaFree(positions_b_);
    cudaFree(positions_a_);
}

meshprep::DeviceMeshView WaterSurface::mesh_view() const noexcept
{
    return {positions_a_, vertex_count_, triangles_, triangle_count_};
}

float WaterSurface::step(
    float frame_dt,
    std::uint32_t substeps,
    const PhysicsOptions& options,
    cudaStream_t stream)
{
    if (substeps == 0) return 0.0F;
    nvtx3::scoped_range operation_range{"waterlab/physics"};
    check(cudaEventRecord(step_begin_, stream), "record physics begin");
    const float dt = frame_dt / static_cast<float>(substeps);
    for (std::uint32_t iteration = 0; iteration < substeps; ++iteration) {
        check(cudaMemsetAsync(constraint_state_, 0, sizeof(float4), stream), "clear constraints");
        accumulate_center_kernel<<<
            (vertex_count_ + block_size - 1U) / block_size, block_size, 0, stream>>>(
            positions_a_, vertex_count_, constraint_state_);
        accumulate_volume_kernel<<<
            (triangle_count_ + block_size - 1U) / block_size, block_size, 0, stream>>>(
            positions_a_, triangles_, triangle_count_, constraint_state_);
        finalize_constraints_kernel<<<1, 1, 0, stream>>>(constraint_state_, vertex_count_);
        surface_step_kernel<<<
            (vertex_count_ + block_size - 1U) / block_size, block_size, 0, stream>>>(
            positions_a_,
            velocities_a_,
            positions_b_,
            velocities_b_,
            neighbor_offsets_,
            neighbors_,
            rest_lengths_,
            constraint_state_,
            vertex_count_,
            rest_volume_,
            dt,
            options);
        check(cudaPeekAtLastError(), "surface physics launch");
        std::swap(positions_a_, positions_b_);
        std::swap(velocities_a_, velocities_b_);
    }
    check(cudaEventRecord(step_end_, stream), "record physics end");
    check(cudaEventSynchronize(step_end_), "synchronize physics");
    float milliseconds = 0.0F;
    check(cudaEventElapsedTime(&milliseconds, step_begin_, step_end_), "measure physics");
    return milliseconds;
}

void WaterSurface::apply_impulse(
    float3 point,
    float3 direction,
    float radius,
    float strength,
    cudaStream_t stream)
{
    nvtx3::scoped_range operation_range{"waterlab/impulse"};
    impulse_kernel<<<
        (vertex_count_ + block_size - 1U) / block_size, block_size, 0, stream>>>(
        positions_a_,
        velocities_a_,
        vertex_count_,
        point,
        normalize(direction),
        radius,
        strength);
    check(cudaStreamSynchronize(stream), "apply impulse");
}

void WaterSurface::reset(cudaStream_t stream)
{
    check(
        cudaMemcpyAsync(
            positions_a_,
            rest_positions_,
            sizeof(float3) * vertex_count_,
            cudaMemcpyDeviceToDevice,
            stream),
        "reset positions");
    check(cudaMemsetAsync(velocities_a_, 0, sizeof(float3) * vertex_count_, stream), "reset velocities");
    check(cudaStreamSynchronize(stream), "synchronize reset");
}

RayTracer::RayTracer()
{
    allocate(device_pick_, 1);
    check(cudaEventCreate(&render_begin_), "create render event");
    check(cudaEventCreate(&render_end_), "create render event");
}

RayTracer::~RayTracer()
{
    cudaEventDestroy(render_end_);
    cudaEventDestroy(render_begin_);
    cudaFree(device_pick_);
    cudaFreeHost(host_pixels_);
    cudaFree(device_pixels_);
}

void RayTracer::reserve(std::uint32_t width, std::uint32_t height)
{
    const std::size_t required = static_cast<std::size_t>(width) * height;
    if (required <= pixel_capacity_) return;
    cudaFreeHost(host_pixels_);
    cudaFree(device_pixels_);
    host_pixels_ = nullptr;
    device_pixels_ = nullptr;
    check(cudaMalloc(&device_pixels_, required * sizeof(uchar4)), "allocate render target");
    check(cudaMallocHost(&host_pixels_, required * sizeof(uchar4)), "allocate pinned pixels");
    pixel_capacity_ = required;
}

float RayTracer::render(
    meshprep::DeviceMeshView mesh,
    const meshprep::Hierarchy& hierarchy,
    const Camera& camera,
    std::uint32_t width,
    std::uint32_t height,
    float time_seconds,
    cudaStream_t stream)
{
    if (hierarchy.statistics().max_depth > 18U) {
        throw std::runtime_error("hierarchy exceeds ray traversal stack contract");
    }
    reserve(width, height);
    nvtx3::scoped_range operation_range{"waterlab/raytrace"};
    check(cudaEventRecord(render_begin_, stream), "record render begin");
    const dim3 threads(16, 8);
    const dim3 blocks(
        (width + threads.x - 1U) / threads.x,
        (height + threads.y - 1U) / threads.y);
    render_kernel<<<blocks, threads, 0, stream>>>(
        device_pixels_,
        width,
        height,
        camera,
        mesh.positions,
        mesh.triangles,
        hierarchy.nodes(),
        hierarchy.primitive_indices(),
        hierarchy.statistics().node_count,
        time_seconds);
    check(cudaPeekAtLastError(), "raytrace launch");
    check(cudaEventRecord(render_end_, stream), "record render end");
    check(
        cudaMemcpyAsync(
            host_pixels_,
            device_pixels_,
            static_cast<std::size_t>(width) * height * sizeof(uchar4),
            cudaMemcpyDeviceToHost,
            stream),
        "download render target");
    check(cudaStreamSynchronize(stream), "synchronize raytrace");
    float milliseconds = 0.0F;
    check(cudaEventElapsedTime(&milliseconds, render_begin_, render_end_), "measure raytrace");
    return milliseconds;
}

PickResult RayTracer::pick(
    meshprep::DeviceMeshView mesh,
    const meshprep::Hierarchy& hierarchy,
    const Camera& camera,
    std::uint32_t pixel_x,
    std::uint32_t pixel_y,
    std::uint32_t width,
    std::uint32_t height,
    cudaStream_t stream)
{
    if (hierarchy.statistics().max_depth > 18U) {
        throw std::runtime_error("hierarchy exceeds ray traversal stack contract");
    }
    pick_kernel<<<1, 1, 0, stream>>>(
        device_pick_,
        pixel_x,
        pixel_y,
        width,
        height,
        camera,
        mesh.positions,
        mesh.triangles,
        hierarchy.nodes(),
        hierarchy.primitive_indices(),
        hierarchy.statistics().node_count);
    PickResult result{};
    check(
        cudaMemcpyAsync(&result, device_pick_, sizeof(result), cudaMemcpyDeviceToHost, stream),
        "download pick result");
    check(cudaStreamSynchronize(stream), "synchronize pick");
    return result;
}

} // namespace waterlab
