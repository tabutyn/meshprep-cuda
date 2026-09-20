// SPDX-License-Identifier: MIT
#include "water_lab.hpp"
#include "obstacle_course.hpp"
#include "fluid_surface.cuh"

#include <cuda_runtime.h>
#include <nvtx3/nvtx3.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace waterlab {

namespace {
constexpr std::uint32_t bowl_paint_width = 256U;
constexpr std::uint32_t bowl_paint_height = 128U;
constexpr std::uint32_t bowl_paint_pixel_count =
    bowl_paint_width * bowl_paint_height;
__device__ std::uint32_t* device_bowl_paint_pixels;
// Intentionally coarse: contact paint is a graphic gameplay mask rather than
// a per-pixel simulation. Cubic reconstruction below removes block edges.
constexpr std::uint32_t sphere_paint_width = 64U;
constexpr std::uint32_t sphere_paint_height = 32U;
constexpr std::uint32_t sphere_paint_pixel_count =
    sphere_paint_width * sphere_paint_height;
// Paint is deliberately coarser than the physical sheets. Bicubic sampling
// turns sparse contacts into a readable, stable painted patch instead of a
// high-resolution constellation of individual contact texels.
constexpr std::uint32_t cloth_paint_width = 64U;
constexpr std::uint32_t cloth_paint_height = 64U;
constexpr std::uint32_t cloth_paint_pixel_count =
    cloth_paint_width * cloth_paint_height;
__device__ std::uint32_t* device_sphere_paint_pixels;
__device__ std::uint32_t* device_cloth_paint_pixels;
__device__ std::uint32_t* device_ground_cloth_paint_pixels;
}
namespace {

constexpr float pi = 3.14159265358979323846F;

void check(cudaError_t error, const char* operation)
{
    if (error == cudaSuccess) return;
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
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

__host__ __device__ float3 world_to_box_vector(float3 value, float yaw)
{
    const float cosine = cosf(yaw);
    const float sine = sinf(yaw);
    return make_float3(
        cosine * value.x - sine * value.z,
        value.y,
        sine * value.x + cosine * value.z);
}

__host__ __device__ float3 box_to_world_vector(float3 value, float yaw)
{
    const float cosine = cosf(yaw);
    const float sine = sinf(yaw);
    return make_float3(
        cosine * value.x + sine * value.z,
        value.y,
        -sine * value.x + cosine * value.z);
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
    const parallel_mater::HierarchyNode& node,
    float maximum_distance,
    float& near_distance,
    float padding = 0.0F)
{
    float near_value = 0.0F;
    float far_value = maximum_distance;
    const float origin[3]{ray.origin.x, ray.origin.y, ray.origin.z};
    const float direction[3]{ray.direction.x, ray.direction.y, ray.direction.z};
    const float minimum[3]{node.bounds_min.x - padding,
        node.bounds_min.y - padding, node.bounds_min.z - padding};
    const float maximum[3]{node.bounds_max.x + padding,
        node.bounds_max.y + padding, node.bounds_max.z + padding};
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
    float& barycentric_u,
    float& barycentric_v)
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
    barycentric_u = u;
    barycentric_v = v;
    return true;
}

__device__ Hit trace_closest(
    const Ray& ray,
    const float3* positions,
    const float3* vertex_normals,
    const std::uint32_t* corner_normal_indices,
    const uint3* triangles,
    const parallel_mater::HierarchyNode* nodes,
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
        const parallel_mater::HierarchyNode node = nodes[node_index];
        float node_near = 0.0F;
        if (!intersect_bounds(ray, node, hit.distance, node_near)) continue;
        if (node.is_leaf()) {
            for (std::uint32_t i = 0; i < node.primitive_count; ++i) {
                const std::uint32_t triangle_index = primitive_indices[node.first_primitive + i];
                const uint3 triangle = triangles[triangle_index];
                float distance = 0.0F;
                float barycentric_u = 0.0F;
                float barycentric_v = 0.0F;
                if (intersect_triangle(
                        ray,
                        positions[triangle.x],
                        positions[triangle.y],
                        positions[triangle.z],
                        hit.distance,
                        distance,
                        barycentric_u,
                        barycentric_v)) {
                    const float barycentric_w = 1.0F - barycentric_u - barycentric_v;
                    const std::size_t corner = static_cast<std::size_t>(triangle_index) * 3U;
                    const float3 normal = normalize(add(
                        multiply(vertex_normals[corner_normal_indices[corner]], barycentric_w),
                        add(
                            multiply(vertex_normals[corner_normal_indices[corner + 1U]], barycentric_u),
                            multiply(vertex_normals[corner_normal_indices[corner + 2U]], barycentric_v))));
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

__device__ bool intersect_particle_sphere(
    const Ray& ray,
    float3 center,
    float radius,
    float maximum_distance,
    float& distance,
    float3& normal)
{
    const float3 offset = subtract(ray.origin, center);
    const float b = dot(offset, ray.direction);
    const float c = dot(offset, offset) - radius * radius;
    const float discriminant = b * b - c;
    if (discriminant < 0.0F) return false;
    const float root = sqrtf(discriminant);
    float hit_distance = -b - root;
    if (hit_distance <= 1.0e-5F) hit_distance = -b + root;
    if (hit_distance <= 1.0e-5F || hit_distance >= maximum_distance) return false;
    distance = hit_distance;
    normal = normalize(subtract(add(ray.origin, multiply(ray.direction, distance)), center));
    return true;
}

__device__ bool intersect_vertical_cylinder(const Ray& ray, float3 center,
    float radius, float half_height, float maximum_distance,
    float& distance, float3& normal)
{
    const float3 origin = subtract(ray.origin, center);
    float best = maximum_distance;
    float3 best_normal{};
    const float a = ray.direction.x * ray.direction.x +
        ray.direction.z * ray.direction.z;
    if (a > 1.0e-10F) {
        const float b = origin.x * ray.direction.x + origin.z * ray.direction.z;
        const float c = origin.x * origin.x + origin.z * origin.z - radius * radius;
        const float discriminant = b * b - a * c;
        if (discriminant >= 0.0F) {
            const float root = sqrtf(discriminant);
            const float candidates[2]{(-b - root) / a, (-b + root) / a};
            for (int candidate = 0; candidate < 2; ++candidate) {
                const float hit = candidates[candidate];
                if (hit <= 1.0e-5F || hit >= best) continue;
                const float y = origin.y + hit * ray.direction.y;
                if (fabsf(y) > half_height) continue;
                const float3 point = add(origin, multiply(ray.direction, hit));
                best = hit;
                best_normal = normalize(make_float3(point.x, 0.0F, point.z));
            }
        }
    }
    if (fabsf(ray.direction.y) > 1.0e-10F) {
        for (int cap = -1; cap <= 1; cap += 2) {
            const float hit = (static_cast<float>(cap) * half_height - origin.y) /
                ray.direction.y;
            if (hit <= 1.0e-5F || hit >= best) continue;
            const float x = origin.x + hit * ray.direction.x;
            const float z = origin.z + hit * ray.direction.z;
            if (x * x + z * z > radius * radius) continue;
            best = hit;
            best_normal = make_float3(0.0F, static_cast<float>(cap), 0.0F);
        }
    }
    if (best >= maximum_distance) return false;
    distance = best;
    normal = best_normal;
    return true;
}

__device__ Hit trace_particle_spheres(
    const Ray& ray,
    const float3* positions,
    float radius,
    const parallel_mater::HierarchyNode* nodes,
    const std::uint32_t* primitive_indices,
    std::uint32_t node_count)
{
    Hit hit{1.0e30F, UINT32_MAX, make_float3(0.0F, 0.0F, 0.0F)};
    if (positions == nullptr || node_count == 0U) return hit;
    std::uint32_t stack[128];
    int stack_size = 1;
    stack[0] = 0U;
    while (stack_size > 0) {
        const std::uint32_t node_index = stack[--stack_size];
        if (node_index >= node_count) continue;
        const parallel_mater::HierarchyNode node = nodes[node_index];
        float node_near = 0.0F;
        if (!intersect_bounds(ray, node, hit.distance, node_near)) continue;
        if (node.is_leaf()) {
            for (std::uint32_t item = 0U; item < node.primitive_count; ++item) {
                const std::uint32_t particle = primitive_indices[node.first_primitive + item];
                float distance = 0.0F;
                float3 normal{};
                if (intersect_particle_sphere(
                        ray, positions[particle], radius, hit.distance, distance, normal)) {
                    hit = {distance, particle, normal};
                }
            }
            continue;
        }
        for (std::uint32_t child = 0U;
             child < node.child_count && stack_size < 128; ++child) {
            stack[stack_size++] = node.first_child + child;
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

__device__ float3 checker_box(float3 origin, float3 direction)
{
    constexpr float half_extent = 6.0F;
    constexpr float maximum_distance = 1.0e20F;
    float distance = maximum_distance;
    int wall = -1;
    if (fabsf(direction.x) > 1.0e-7F) {
        const float candidate =
            ((direction.x > 0.0F ? half_extent : -half_extent) - origin.x) / direction.x;
        if (candidate > 0.0F && candidate < distance) {
            distance = candidate;
            wall = 0;
        }
    }
    if (fabsf(direction.y) > 1.0e-7F) {
        const float candidate =
            ((direction.y > 0.0F ? half_extent : -half_extent) - origin.y) / direction.y;
        if (candidate > 0.0F && candidate < distance) {
            distance = candidate;
            wall = 1;
        }
    }
    if (fabsf(direction.z) > 1.0e-7F) {
        const float candidate =
            ((direction.z > 0.0F ? half_extent : -half_extent) - origin.z) / direction.z;
        if (candidate > 0.0F && candidate < distance) {
            distance = candidate;
            wall = 2;
        }
    }
    if (wall < 0) return make_float3(0.015F, 0.018F, 0.024F);

    const float3 hit = add(origin, multiply(direction, distance));
    float u = hit.x;
    float v = hit.y;
    float wall_light = 0.78F;
    if (wall == 0) {
        u = hit.z;
        v = hit.y;
        wall_light = 0.72F;
    } else if (wall == 1) {
        u = hit.x;
        v = hit.z;
        wall_light = direction.y > 0.0F ? 0.92F : 0.58F;
    }

    constexpr float cell_size = 0.72F;
    const int checker_u = static_cast<int>(floorf((u + half_extent) / cell_size));
    const int checker_v = static_cast<int>(floorf((v + half_extent) / cell_size));
    const bool light_square = ((checker_u + checker_v) & 1) == 0;
    const float3 light = make_float3(0.72F, 0.74F, 0.76F);
    const float3 dark = make_float3(0.055F, 0.065F, 0.080F);
    return multiply(light_square ? light : dark, wall_light);
}

enum CourseMaterial : std::uint32_t {
    course_material_none,
    course_material_floor,
    course_material_rail,
    course_material_post,
};

struct CourseHit {
    float distance{1.0e30F};
    float3 normal{};
    CourseMaterial material{course_material_none};

};

__device__ bool intersect_box(
    const Ray& ray,
    float3 center,
    float3 half_extents,
    float maximum_distance,
    float& distance,
    float3& normal)
{
    const float3 local_origin = subtract(ray.origin, center);
    const float origins[3]{local_origin.x, local_origin.y, local_origin.z};
    const float directions[3]{ray.direction.x, ray.direction.y, ray.direction.z};
    const float extents[3]{half_extents.x, half_extents.y, half_extents.z};
    float near_value = -1.0e30F;
    float far_value = maximum_distance;
    float3 near_normal{};
    float3 far_normal{};
    for (int axis = 0; axis < 3; ++axis) {
        if (fabsf(directions[axis]) < 1.0e-10F) {
            if (origins[axis] < -extents[axis] || origins[axis] > extents[axis]) return false;
            continue;
        }
        float first = (-extents[axis] - origins[axis]) / directions[axis];
        float second = (extents[axis] - origins[axis]) / directions[axis];
        float first_sign = -1.0F;
        float second_sign = 1.0F;
        if (first > second) {
            const float temporary = first;
            first = second;
            second = temporary;
            const float sign_temporary = first_sign;
            first_sign = second_sign;
            second_sign = sign_temporary;
        }
        if (first > near_value) {
            near_value = first;
            near_normal = make_float3(0.0F, 0.0F, 0.0F);
            if (axis == 0) near_normal.x = first_sign;
            else if (axis == 1) near_normal.y = first_sign;
            else near_normal.z = first_sign;
        }
        if (second < far_value) {
            far_value = second;
            far_normal = make_float3(0.0F, 0.0F, 0.0F);
            if (axis == 0) far_normal.x = second_sign;
            else if (axis == 1) far_normal.y = second_sign;
            else far_normal.z = second_sign;
        }
        if (near_value > far_value) return false;
    }
    const bool inside = near_value <= 1.0e-5F;
    const float candidate = inside ? far_value : near_value;
    if (candidate <= 1.0e-5F || candidate >= maximum_distance) return false;
    distance = candidate;
    normal = inside ? far_normal : near_normal;
    return true;
}

__device__ void consider_course_box(
    const Ray& ray,
    float3 center,
    float3 half_extents,
    CourseMaterial material,
    CourseHit& hit)
{
    float distance = 0.0F;
    float3 normal{};
    if (intersect_box(ray, center, half_extents, hit.distance, distance, normal)) {
        hit = {distance, normal, material};
    }
}

__device__ void consider_course_post(
    const Ray& ray, unsigned peg_index, CourseHit& hit)
{
    const float3 base = course_peg(peg_index);
    const float ox = ray.origin.x - base.x;
    const float oz = ray.origin.z - base.z;
    const float dx = ray.direction.x;
    const float dz = ray.direction.z;
    const float minimum_y = course_floor_y;
    const float maximum_y = course_floor_y + course_peg_height;
    float best = hit.distance;
    float3 best_normal{};

    const float a = dx * dx + dz * dz;
    if (a > 1.0e-12F) {
        const float b = 2.0F * (ox * dx + oz * dz);
        const float c = ox * ox + oz * oz - course_peg_radius * course_peg_radius;
        const float discriminant = b * b - 4.0F * a * c;
        if (discriminant >= 0.0F) {
            const float root = sqrtf(discriminant);
            const float candidates[2]{(-b - root) / (2.0F * a),
                (-b + root) / (2.0F * a)};
            for (float candidate : candidates) {
                if (candidate <= 1.0e-5F || candidate >= best) continue;
                const float y = ray.origin.y + candidate * ray.direction.y;
                if (y < minimum_y || y > maximum_y) continue;
                const float x = ox + candidate * dx;
                const float z = oz + candidate * dz;
                const float inverse = rsqrtf(fmaxf(x * x + z * z, 1.0e-20F));
                best = candidate;
                best_normal = make_float3(x * inverse, 0.0F, z * inverse);
            }
        }
    }
    if (fabsf(ray.direction.y) > 1.0e-12F) {
        const float cap_y[2]{minimum_y, maximum_y};
        const float cap_normal_y[2]{-1.0F, 1.0F};
        for (unsigned cap = 0U; cap < 2U; ++cap) {
            const float candidate = (cap_y[cap] - ray.origin.y) / ray.direction.y;
            if (candidate <= 1.0e-5F || candidate >= best) continue;
            const float x = ox + candidate * dx;
            const float z = oz + candidate * dz;
            if (x * x + z * z > course_peg_radius * course_peg_radius) continue;
            best = candidate;
            best_normal = make_float3(0.0F, cap_normal_y[cap], 0.0F);
        }
    }
    if (best < hit.distance) hit = {best, best_normal, course_material_post};
}

__device__ CourseHit trace_course(const Ray& ray, float maximum_distance = 1.0e30F)
{
    CourseHit hit{maximum_distance, {}, course_material_none};
    const float course_mid_z = 0.5F * (course_near_z + course_far_z);
    const float course_half_length = 0.5F * (course_near_z - course_far_z);

    consider_course_box(
        ray,
        make_float3(0.0F, course_floor_y - 0.5F * course_floor_thickness, course_mid_z),
        make_float3(course_half_width + course_rail_thickness,
            0.5F * course_floor_thickness, course_half_length + course_rail_thickness),
        course_material_floor,
        hit);

    const float rail_y = course_floor_y + 0.5F * course_rail_height;
    const float side_x = course_half_width + 0.5F * course_rail_thickness;
    const float end_half_width = course_half_width + course_rail_thickness;
    consider_course_box(ray, make_float3(-side_x, rail_y, course_mid_z),
        make_float3(0.5F * course_rail_thickness, 0.5F * course_rail_height,
            course_half_length + course_rail_thickness), course_material_rail, hit);
    consider_course_box(ray, make_float3(side_x, rail_y, course_mid_z),
        make_float3(0.5F * course_rail_thickness, 0.5F * course_rail_height,
            course_half_length + course_rail_thickness), course_material_rail, hit);
    consider_course_box(ray,
        make_float3(0.0F, rail_y, course_near_z + 0.5F * course_rail_thickness),
        make_float3(end_half_width, 0.5F * course_rail_height,
            0.5F * course_rail_thickness), course_material_rail, hit);
    consider_course_box(ray,
        make_float3(0.0F, rail_y, course_far_z - 0.5F * course_rail_thickness),
        make_float3(end_half_width, 0.5F * course_rail_height,
            0.5F * course_rail_thickness), course_material_rail, hit);
    for (unsigned peg = 0U; peg < course_peg_count; ++peg)
        consider_course_post(ray, peg, hit);

    return hit;
}

__device__ float3 course_background(float3 direction)
{
    const float horizon = fminf(1.0F, fmaxf(0.0F, 0.5F * direction.y + 0.5F));
    return add(
        multiply(make_float3(0.020F, 0.027F, 0.045F), horizon),
        multiply(make_float3(0.075F, 0.095F, 0.135F), 1.0F - horizon));
}

__device__ float3 shade_course(const Ray& ray, const CourseHit& hit)
{
    const float3 point = add(ray.origin, multiply(ray.direction, hit.distance));
    const float3 light_direction = normalize(make_float3(-0.48F, 0.84F, 0.34F));
    const float diffuse = fmaxf(0.0F, dot(hit.normal, light_direction));
    float3 base{};
    float gloss = 0.0F;
    if (hit.material == course_material_floor) {
        if (hit.normal.y < 0.75F) {
            base = make_float3(0.045F, 0.055F, 0.075F);
        } else {
            constexpr float tile_size = 0.56F;
            const int tile_x = static_cast<int>(floorf(
                (point.x + course_half_width) / tile_size));
            const int tile_z = static_cast<int>(floorf(
                (point.z - course_far_z) / tile_size));
            const bool light_tile = ((tile_x + tile_z) & 1) == 0;
            base = light_tile
                ? make_float3(0.29F, 0.34F, 0.42F)
                : make_float3(0.065F, 0.082F, 0.115F);

            const float start_z = course_near_z - 0.72F;
            const bool start_line = fabsf(point.z - start_z) < 0.055F &&
                fabsf(point.x) < course_half_width - 0.22F;
            const float goal_dx = point.x;
            const float goal_dz = point.z - course_goal_z;
            const float goal_distance = sqrtf(goal_dx * goal_dx + goal_dz * goal_dz);
            const bool finish_ring = fabsf(goal_distance - course_goal_radius) < 0.075F;
            const bool finish_center = goal_distance < course_goal_radius - 0.075F;
            if (start_line) base = make_float3(0.92F, 0.76F, 0.28F);
            if (finish_center) {
                const bool finish_light = ((tile_x + tile_z) & 1) == 0;
                base = finish_light
                    ? make_float3(0.25F, 0.90F, 0.49F)
                    : make_float3(0.035F, 0.31F, 0.16F);
            }
            if (finish_ring) base = make_float3(0.52F, 1.0F, 0.60F);
        }
        gloss = 0.10F;
    } else if (hit.material == course_material_rail) {
        base = make_float3(0.12F, 0.22F, 0.34F);
        gloss = 0.42F;
    } else {
        base = make_float3(0.94F, 0.45F, 0.075F);
        gloss = 0.62F;
    }

    const float3 reflected_light = subtract(
        multiply(hit.normal, 2.0F * dot(hit.normal, light_direction)), light_direction);
    const float specular = powf(fmaxf(0.0F,
        dot(normalize(multiply(ray.direction, -1.0F)), reflected_light)), 36.0F);
    const float lighting = 0.24F + 0.76F * diffuse;
    float3 color = add(multiply(base, lighting),
        multiply(make_float3(1.0F, 0.86F, 0.62F), gloss * specular));
    const float fog = fminf(0.48F, hit.distance * 0.018F);
    return add(multiply(color, 1.0F - fog),
        multiply(make_float3(0.055F, 0.073F, 0.105F), fog));
}

struct SoftBodyHit {
    float distance{1.0e30F};
    float3 normal{};
    float2 uv{};
    std::uint32_t triangle{UINT32_MAX};
};

__device__ SoftBodyHit trace_soft_body(
    const Ray& ray, SoftBodyRenderView soft_body, float maximum_distance = 1.0e30F)
{
    SoftBodyHit hit{maximum_distance, {}, {}, UINT32_MAX};
    if (soft_body.positions == nullptr || soft_body.texcoords == nullptr ||
        soft_body.triangles == nullptr || soft_body.nodes == nullptr ||
        soft_body.primitive_indices == nullptr || soft_body.node_count == 0U) return hit;
    std::uint32_t stack[128];
    int stack_size = 1;
    stack[0] = 0U;
    while (stack_size > 0) {
        const std::uint32_t node_index = stack[--stack_size];
        if (node_index >= soft_body.node_count) continue;
        const parallel_mater::HierarchyNode node = soft_body.nodes[node_index];
        float node_near = 0.0F;
        if (!intersect_bounds(ray, node, hit.distance, node_near)) continue;
        if (node.is_leaf()) {
            for (std::uint32_t item = 0U; item < node.primitive_count; ++item) {
                const std::uint32_t triangle_id =
                    soft_body.primitive_indices[node.first_primitive + item];
                if (triangle_id >= soft_body.triangle_count) continue;
                if (soft_body.triangle_active != nullptr &&
                    soft_body.triangle_active[triangle_id] == 0U) continue;
                const uint3 triangle = soft_body.triangles[triangle_id];
                if (triangle.x >= soft_body.vertex_count || triangle.y >= soft_body.vertex_count ||
                    triangle.z >= soft_body.vertex_count) continue;
                float distance = 0.0F;
                float u = 0.0F;
                float v = 0.0F;
                if (!intersect_triangle(ray, soft_body.positions[triangle.x],
                        soft_body.positions[triangle.y], soft_body.positions[triangle.z],
                        hit.distance, distance, u, v)) continue;
                const float w = 1.0F - u - v;
                float3 normal = normalize(cross(
                    subtract(soft_body.positions[triangle.y], soft_body.positions[triangle.x]),
                    subtract(soft_body.positions[triangle.z], soft_body.positions[triangle.x])));
                if (soft_body.vertex_normals != nullptr &&
                    soft_body.corner_normal_indices != nullptr) {
                    const std::uint32_t corner = 3U * triangle_id;
                    const float3 na = soft_body.vertex_normals[
                        soft_body.corner_normal_indices[corner]];
                    const float3 nb = soft_body.vertex_normals[
                        soft_body.corner_normal_indices[corner + 1U]];
                    const float3 nc = soft_body.vertex_normals[
                        soft_body.corner_normal_indices[corner + 2U]];
                    const float3 smooth = add(add(multiply(na, w), multiply(nb, u)),
                        multiply(nc, v));
                    if (dot(smooth, smooth) > 1.0e-12F) normal = normalize(smooth);
                }
                const float2 ta = soft_body.texcoords[triangle.x];
                const float2 tb = soft_body.texcoords[triangle.y];
                const float2 tc = soft_body.texcoords[triangle.z];
                hit = {distance, normal,
                    make_float2(w * ta.x + u * tb.x + v * tc.x,
                        w * ta.y + u * tb.y + v * tc.y), triangle_id};
            }
            continue;
        }
        std::uint32_t child_indices[8];
        float child_near[8];
        int child_hits = 0;
        for (std::uint32_t child = 0U; child < node.child_count; ++child) {
            const std::uint32_t child_index = node.first_child + child;
            float near_value = 0.0F;
            if (!intersect_bounds(ray, soft_body.nodes[child_index], hit.distance, near_value)) {
                continue;
            }
            int insertion = child_hits;
            while (insertion > 0 && (child_near[insertion - 1] > near_value ||
                (child_near[insertion - 1] == near_value &&
                 child_indices[insertion - 1] > child_index))) {
                child_indices[insertion] = child_indices[insertion - 1];
                child_near[insertion] = child_near[insertion - 1];
                --insertion;
            }
            child_indices[insertion] = child_index;
            child_near[insertion] = near_value;
            ++child_hits;
        }
        for (int child = child_hits - 1; child >= 0 && stack_size < 128; --child) {
            stack[stack_size++] = child_indices[child];
        }
    }
    return hit;
}

__device__ bool intersect_soft_member(const Ray& ray, float3 a, float3 b,
    float half_width, float maximum_distance, float& distance, float3& normal)
{
    const float3 delta = subtract(b, a);
    const float member_length = length(delta);
    if (!(member_length > 1.0e-6F)) return false;
    const float3 axis = multiply(delta, 1.0F / member_length);
    const float3 reference = fabsf(axis.y) < 0.85F
        ? make_float3(0.0F, 1.0F, 0.0F) : make_float3(1.0F, 0.0F, 0.0F);
    const float3 side = normalize(cross(axis, reference));
    const float3 up = cross(side, axis);
    const float3 center = multiply(add(a, b), 0.5F);
    const float3 offset = subtract(ray.origin, center);
    const float origins[3]{dot(offset, axis), dot(offset, side), dot(offset, up)};
    const float directions[3]{dot(ray.direction, axis), dot(ray.direction, side),
        dot(ray.direction, up)};
    const float extents[3]{0.5F * member_length, half_width, half_width};
    const float3 axes[3]{axis, side, up};
    float near_value = 1.0e-5F;
    float far_value = maximum_distance;
    float3 near_normal{};
    for (int dimension = 0; dimension < 3; ++dimension) {
        if (fabsf(directions[dimension]) < 1.0e-9F) {
            if (fabsf(origins[dimension]) > extents[dimension]) return false;
            continue;
        }
        float first = (-extents[dimension] - origins[dimension]) /
            directions[dimension];
        float second = ( extents[dimension] - origins[dimension]) /
            directions[dimension];
        float first_sign = -1.0F;
        if (first > second) {
            const float temporary = first; first = second; second = temporary;
            first_sign = 1.0F;
        }
        if (first > near_value) {
            near_value = first;
            near_normal = multiply(axes[dimension], first_sign);
        }
        far_value = fminf(far_value, second);
        if (near_value > far_value) return false;
    }
    if (near_value <= 1.0e-5F || near_value >= maximum_distance) return false;
    distance = near_value;
    normal = near_normal;
    return true;
}

__device__ SoftBodyHit trace_soft_members(
    const Ray& ray, SoftBodyRenderView body, float maximum_distance)
{
    SoftBodyHit hit{maximum_distance, {}, {}};
    if (body.member_positions == nullptr || body.member_edges == nullptr ||
        body.member_active == nullptr || body.member_nodes == nullptr ||
        body.member_indices == nullptr || body.member_node_count == 0U ||
        body.members_per_instance == 0U || body.member_voxels_per_instance == 0U)
        return hit;
    std::uint32_t stack[128];
    int stack_size = 1;
    stack[0] = 0U;
    while (stack_size > 0) {
        const std::uint32_t node_index = stack[--stack_size];
        if (node_index >= body.member_node_count) continue;
        const parallel_mater::HierarchyNode node = body.member_nodes[node_index];
        float near_value{};
        if (!intersect_bounds(ray, node, hit.distance, near_value)) continue;
        if (node.is_leaf()) {
            for (std::uint32_t item = 0U; item < node.primitive_count; ++item) {
                const std::uint32_t member =
                    body.member_indices[node.first_primitive + item];
                if (member >= body.member_count || body.member_active[member] == 0U)
                    continue;
                const std::uint32_t instance = member / body.members_per_instance;
                const SoftBodyEdge edge =
                    body.member_edges[member - instance * body.members_per_instance];
                const std::uint32_t base = instance * body.member_voxels_per_instance;
                float member_distance{};
                float3 member_normal{};
                if (!intersect_soft_member(ray,
                        body.member_positions[base + edge.vertices.x],
                        body.member_positions[base + edge.vertices.y],
                        body.member_half_width, hit.distance,
                        member_distance, member_normal)) continue;
                hit.distance = member_distance;
                hit.normal = member_normal;
            }
            continue;
        }
        for (std::uint32_t child = 0U;
             child < node.child_count && stack_size < 128; ++child)
            stack[stack_size++] = node.first_child + child;
    }
    return hit;
}

__device__ bool goal_texel(float2 uv)
{
    // Four 5x7 bitmap glyphs embedded in the cloth UVs. This remains attached
    // to the deforming triangles and therefore tears with the target.
    const float u = (uv.x - 0.10F) / 0.80F;
    const float v = (uv.y - 0.34F) / 0.32F;
    if (u < 0.0F || u >= 1.0F || v < 0.0F || v >= 1.0F) return false;
    const int glyph = min(3, static_cast<int>(u * 4.0F));
    const float local_u = u * 4.0F - static_cast<float>(glyph);
    const int x = min(4, static_cast<int>(local_u * 5.0F));
    const int y = min(6, static_cast<int>((1.0F - v) * 7.0F));
    // Rows are encoded most-significant pixel first: G, O, A, L.
    const unsigned g[7]{14U,17U,16U,23U,17U,17U,14U};
    const unsigned o[7]{14U,17U,17U,17U,17U,17U,14U};
    const unsigned a[7]{14U,17U,17U,31U,17U,17U,17U};
    const unsigned l[7]{16U,16U,16U,16U,16U,16U,31U};
    const unsigned row = glyph == 0 ? g[y] : glyph == 1 ? o[y] :
        glyph == 2 ? a[y] : l[y];
    return (row & (1U << (4 - x))) != 0U;
}

__host__ __device__ float3 inverse_rotate_quaternion(float4 q, float3 value)
{
    const float3 inverse_axis = make_float3(-q.x,-q.y,-q.z);
    const float3 first = multiply(cross(inverse_axis,value),2.0F);
    return add(value,add(multiply(first,q.w),cross(inverse_axis,first)));
}

__device__ float cubic_weight(float value)
{
    value = fabsf(value);
    if (value <= 1.0F)
        return (1.5F*value-2.5F)*value*value+1.0F;
    if (value < 2.0F)
        return ((-0.5F*value+2.5F)*value-4.0F)*value+2.0F;
    return 0.0F;
}

__device__ bool painted_sphere_texel(float3 world_normal, float4 orientation)
{
    if (device_sphere_paint_pixels == nullptr) return false;
    const float3 normal = inverse_rotate_quaternion(orientation,world_normal);
    const float longitude = (atan2f(normal.z, normal.x) + pi) / (2.0F*pi);
    const float latitude = acosf(fminf(1.0F, fmaxf(-1.0F, normal.y))) / pi;
    const float sample_x = longitude*static_cast<float>(sphere_paint_width)-0.5F;
    const float sample_y = latitude*static_cast<float>(sphere_paint_height)-0.5F;
    const int base_x = static_cast<int>(floorf(sample_x));
    const int base_y = static_cast<int>(floorf(sample_y));
    float filtered = 0.0F;
    float total_weight = 0.0F;
    for (int oy=-1; oy<=2; ++oy) {
        const int y = max(0,min(static_cast<int>(sphere_paint_height)-1,base_y+oy));
        const float wy = cubic_weight(sample_y-static_cast<float>(base_y+oy));
        for (int ox=-1; ox<=2; ++ox) {
            int x = (base_x+ox)%static_cast<int>(sphere_paint_width);
            if (x < 0) x += static_cast<int>(sphere_paint_width);
            const float weight = wy*cubic_weight(
                sample_x-static_cast<float>(base_x+ox));
            filtered += weight*(device_sphere_paint_pixels[
                y*sphere_paint_width+x] != 0U ? 1.0F : 0.0F);
            total_weight += weight;
        }
    }
    return total_weight > 0.0F && filtered/total_weight >= 0.38F;
}

__device__ bool painted_cloth_texel(float2 uv, bool ground)
{
    const std::uint32_t* pixels = ground
        ? device_ground_cloth_paint_pixels : device_cloth_paint_pixels;
    if (pixels == nullptr) return false;
    const float u = fminf(1.0F, fmaxf(0.0F, uv.x));
    const float v = fminf(1.0F, fmaxf(0.0F, uv.y));
    const float sample_x = u*static_cast<float>(cloth_paint_width)-0.5F;
    const float sample_y = v*static_cast<float>(cloth_paint_height)-0.5F;
    const int base_x = static_cast<int>(floorf(sample_x));
    const int base_y = static_cast<int>(floorf(sample_y));
    float filtered = 0.0F;
    float total_weight = 0.0F;
    for (int oy=-1; oy<=2; ++oy) {
        const int y = max(0,min(static_cast<int>(cloth_paint_height)-1,base_y+oy));
        const float wy = cubic_weight(sample_y-static_cast<float>(base_y+oy));
        for (int ox=-1; ox<=2; ++ox) {
            const int x = max(0,min(static_cast<int>(cloth_paint_width)-1,base_x+ox));
            const float weight = wy*cubic_weight(
                sample_x-static_cast<float>(base_x+ox));
            filtered += weight*(pixels[y*cloth_paint_width+x] != 0U ? 1.0F : 0.0F);
            total_weight += weight;
        }
    }
    return total_weight > 0.0F && filtered/total_weight >= 0.38F;
}

__device__ float3 shade_colored_surface(
    const Ray& ray, SoftBodyHit hit, float3 base, bool goal, int paint_layer)
{
    if (dot(hit.normal, ray.direction) > 0.0F)
        hit.normal = multiply(hit.normal, -1.0F);
    if (goal && goal_texel(hit.uv)) base = make_float3(1.0F,0.82F,0.08F);
    if (paint_layer != 0 && painted_cloth_texel(hit.uv,paint_layer == 2))
        base = make_float3(0.04F,0.34F,1.0F);
    const float3 light = normalize(make_float3(-0.48F,0.84F,0.34F));
    const float diffuse = 0.24F+0.76F*fmaxf(0.0F,dot(hit.normal,light));
    const float3 reflected = subtract(multiply(hit.normal,
        2.0F*dot(hit.normal,light)),light);
    const float specular = 0.24F*powf(fmaxf(0.0F,
        dot(normalize(multiply(ray.direction,-1.0F)),reflected)),28.0F);
    return add(multiply(base,diffuse),multiply(make_float3(1,1,1),specular));
}

__device__ float3 shade_soft_body(const Ray& ray, SoftBodyHit hit,
    bool cloth_palette = false, bool crust_palette = false,
    bool goal_palette = false)
{
    if (dot(hit.normal, ray.direction) > 0.0F) hit.normal = multiply(hit.normal, -1.0F);
    const int checker_u = static_cast<int>(floorf(hit.uv.x));
    const int checker_v = static_cast<int>(floorf(hit.uv.y));
    const bool light_square = ((checker_u + checker_v) & 1) == 0;
    const float3 base = goal_palette && goal_texel(hit.uv)
        ? make_float3(1.0F, 0.78F, 0.08F)
        : crust_palette
        ? (light_square ? make_float3(0.93F, 0.47F, 0.13F)
                        : make_float3(0.46F, 0.16F, 0.035F))
        : cloth_palette
            ? (light_square ? make_float3(0.45F, 0.88F, 0.82F)
                            : make_float3(0.045F, 0.22F, 0.24F))
            : (light_square ? make_float3(0.88F, 0.91F, 0.95F)
                            : make_float3(0.055F, 0.075F, 0.11F));
    const float3 light_direction = normalize(make_float3(-0.48F, 0.84F, 0.34F));
    const float diffuse = 0.22F + 0.78F * fmaxf(0.0F, dot(hit.normal, light_direction));
    const float3 reflected_light = subtract(
        multiply(hit.normal, 2.0F * dot(hit.normal, light_direction)), light_direction);
    const float specular = (crust_palette ? 0.48F : 0.28F) * powf(fmaxf(0.0F,
        dot(normalize(multiply(ray.direction, -1.0F)), reflected_light)), 28.0F);
    return add(multiply(base, diffuse), multiply(make_float3(1.0F, 0.82F, 0.45F), specular));
}

__device__ float3 shade_soft_member(const Ray& ray, SoftBodyHit hit)
{
    if (dot(hit.normal, ray.direction) > 0.0F)
        hit.normal = multiply(hit.normal, -1.0F);
    const float3 light = normalize(make_float3(-0.48F, 0.84F, 0.34F));
    const float diffuse = 0.30F + 0.70F * fmaxf(0.0F, dot(hit.normal, light));
    // Warm ivory makes the exposed load-bearing interior read as mozzarella;
    // the outside remains an opaque browned crust.
    return multiply(make_float3(1.0F, 0.86F, 0.55F), diffuse);
}

__device__ bool trace_course_opaque(
    const Ray& ray, SoftBodyRenderView soft_body, float maximum_distance,
    float& distance, float3& color)
{
    const CourseHit course = trace_course(ray, maximum_distance);
    const SoftBodyHit body = trace_soft_body(ray, soft_body, maximum_distance);
    const SoftBodyHit member = trace_soft_members(ray, soft_body, maximum_distance);
    const bool course_valid = course.distance < maximum_distance;
    const bool body_valid = body.distance < maximum_distance;
    const bool member_valid = member.distance < maximum_distance;
    if (!course_valid && !body_valid && !member_valid) return false;
    if (member_valid && (!body_valid || member.distance < body.distance) &&
        (!course_valid || member.distance < course.distance)) {
        distance = member.distance;
        color = shade_soft_member(ray, member);
    } else if (body_valid && (!course_valid || body.distance < course.distance)) {
        distance = body.distance;
        color = shade_soft_body(ray, body);
    } else {
        distance = course.distance;
        color = shade_course(ray, course);
    }
    return true;
}

__device__ bool on_wire_edge(float3 point, float3 extent)
{
    constexpr float thickness = 0.018F;
    const int boundaries =
        (extent.x - fabsf(point.x) <= thickness ? 1 : 0) +
        (extent.y - fabsf(point.y) <= thickness ? 1 : 0) +
        (extent.z - fabsf(point.z) <= thickness ? 1 : 0);
    return boundaries >= 2;
}

__device__ bool intersect_wire_box(
    const Ray& ray,
    const OrientedBox& collider,
    float maximum_distance,
    float& distance,
    float3& world_normal)
{
    const float3 local_origin =
        world_to_box_vector(subtract(ray.origin, collider.center), collider.yaw);
    const float3 local_direction = world_to_box_vector(ray.direction, collider.yaw);
    const float origins[3]{local_origin.x, local_origin.y, local_origin.z};
    const float directions[3]{local_direction.x, local_direction.y, local_direction.z};
    const float extents[3]{
        collider.half_extents.x, collider.half_extents.y, collider.half_extents.z};
    float near_value = 0.0F;
    float far_value = maximum_distance;
    float3 near_normal{};
    float3 far_normal{};
    for (int axis = 0; axis < 3; ++axis) {
        if (fabsf(directions[axis]) < 1.0e-10F) {
            if (origins[axis] < -extents[axis] || origins[axis] > extents[axis]) return false;
            continue;
        }
        float first = (-extents[axis] - origins[axis]) / directions[axis];
        float second = (extents[axis] - origins[axis]) / directions[axis];
        float first_sign = -1.0F;
        float second_sign = 1.0F;
        if (first > second) {
            const float temporary = first;
            first = second;
            second = temporary;
            const float sign_temporary = first_sign;
            first_sign = second_sign;
            second_sign = sign_temporary;
        }
        if (first > near_value) {
            near_value = first;
            near_normal = make_float3(0.0F, 0.0F, 0.0F);
            if (axis == 0) near_normal.x = first_sign;
            else if (axis == 1) near_normal.y = first_sign;
            else near_normal.z = first_sign;
        }
        if (second < far_value) {
            far_value = second;
            far_normal = make_float3(0.0F, 0.0F, 0.0F);
            if (axis == 0) far_normal.x = second_sign;
            else if (axis == 1) far_normal.y = second_sign;
            else far_normal.z = second_sign;
        }
        if (near_value > far_value) return false;
    }
    const bool started_inside =
        fabsf(local_origin.x) < collider.half_extents.x &&
        fabsf(local_origin.y) < collider.half_extents.y &&
        fabsf(local_origin.z) < collider.half_extents.z;
    if (!started_inside && near_value > 1.0e-5F && near_value < maximum_distance) {
        const float3 near_point = add(local_origin, multiply(local_direction, near_value));
        if (on_wire_edge(near_point, collider.half_extents)) {
            distance = near_value;
            world_normal = normalize(box_to_world_vector(near_normal, collider.yaw));
            return true;
        }
    }
    if (far_value > 1.0e-5F && far_value < maximum_distance) {
        const float3 far_point = add(local_origin, multiply(local_direction, far_value));
        if (on_wire_edge(far_point, collider.half_extents)) {
            distance = far_value;
            world_normal = normalize(box_to_world_vector(far_normal, collider.yaw));
            return true;
        }
    }
    return false;
}

__device__ float3 shade_wire_box(float3 normal)
{
    const float light = 0.38F + 0.62F * fmaxf(0.0F, dot(normal, normalize(make_float3(-0.4F, 0.8F, 0.55F))));
    return multiply(make_float3(0.92F, 0.30F, 0.075F), light);
}

__device__ bool wheel_slab(float origin, float direction,
    float minimum, float maximum, float3 minimum_normal, float3 maximum_normal,
    float& near_distance, float& far_distance, float3& near_normal)
{
    if (fabsf(direction) < 1.0e-8F)
        return origin >= minimum && origin <= maximum;
    float first = (minimum - origin) / direction;
    float second = (maximum - origin) / direction;
    float3 first_normal = minimum_normal;
    float3 second_normal = maximum_normal;
    if (first > second) {
        const float swap_distance = first;
        first = second;
        second = swap_distance;
        const float3 swap_normal = first_normal;
        first_normal = second_normal;
        second_normal = swap_normal;
    }
    if (first > near_distance) {
        near_distance = first;
        near_normal = first_normal;
    }
    far_distance = fminf(far_distance, second);
    return near_distance <= far_distance;
}

__device__ bool intersect_water_wheel_fin(const Ray& ray, float angle,
    float maximum_distance, float& distance, float3& normal)
{
    const float cosine = cosf(angle);
    const float sine = sinf(angle);
    const float3 relative = subtract(ray.origin, water_wheel_center);
    const float3 local_origin = make_float3(
        relative.x * cosine + relative.y * sine,
        -relative.x * sine + relative.y * cosine, relative.z);
    const float3 local_direction = make_float3(
        ray.direction.x * cosine + ray.direction.y * sine,
        -ray.direction.x * sine + ray.direction.y * cosine, ray.direction.z);
    float near_distance = -1.0e30F;
    float far_distance = maximum_distance;
    float3 local_normal{};
    if (!wheel_slab(local_origin.x, local_direction.x, water_wheel_radius,
            water_wheel_fin_outer_radius, make_float3(-1,0,0), make_float3(1,0,0),
            near_distance, far_distance, local_normal) ||
        !wheel_slab(local_origin.y, local_direction.y, -water_wheel_fin_thickness,
            water_wheel_fin_thickness, make_float3(0,-1,0), make_float3(0,1,0),
            near_distance, far_distance, local_normal) ||
        !wheel_slab(local_origin.z, local_direction.z, -water_wheel_half_depth,
            water_wheel_half_depth, make_float3(0,0,-1), make_float3(0,0,1),
            near_distance, far_distance, local_normal)) return false;
    const float hit = near_distance > 1.0e-5F ? near_distance : far_distance;
    if (hit <= 1.0e-5F || hit >= maximum_distance) return false;
    distance = hit;
    normal = make_float3(
        local_normal.x * cosine - local_normal.y * sine,
        local_normal.x * sine + local_normal.y * cosine,
        local_normal.z);
    return true;
}

__device__ bool intersect_water_wheel_outer_rung(const Ray& ray,float angle,
    float maximum_distance,float& distance,float3& normal)
{
    const float cosine=cosf(angle),sine=sinf(angle);
    const float radial_center=water_wheel_outer_disk_radius+
        0.5F*water_wheel_outer_rung_radial_half_length;
    const float3 center=make_float3(
        water_wheel_center.x+radial_center*cosine,
        water_wheel_center.y+radial_center*sine,water_wheel_stage_z);
    const float3 relative=subtract(ray.origin,center);
    const float3 local_origin=make_float3(
        relative.x*cosine+relative.y*sine,
        -relative.x*sine+relative.y*cosine,relative.z);
    const float3 local_direction=make_float3(
        ray.direction.x*cosine+ray.direction.y*sine,
        -ray.direction.x*sine+ray.direction.y*cosine,ray.direction.z);
    float near_distance=-1.0e30F,far_distance=maximum_distance;
    float3 local_normal{};
    if (!wheel_slab(local_origin.x,local_direction.x,
            -water_wheel_outer_rung_radial_half_length,
            water_wheel_outer_rung_radial_half_length,
            make_float3(-1,0,0),make_float3(1,0,0),near_distance,far_distance,
            local_normal) ||
        !wheel_slab(local_origin.y,local_direction.y,
            -water_wheel_outer_rung_tangent_half_width,
            water_wheel_outer_rung_tangent_half_width,
            make_float3(0,-1,0),make_float3(0,1,0),near_distance,far_distance,
            local_normal) ||
        !wheel_slab(local_origin.z,local_direction.z,
            -water_wheel_outer_rung_axial_half_depth,
            water_wheel_outer_rung_axial_half_depth,
            make_float3(0,0,-1),make_float3(0,0,1),near_distance,far_distance,
            local_normal)) return false;
    const float hit=near_distance>1.0e-5F ? near_distance : far_distance;
    if (hit<=1.0e-5F || hit>=maximum_distance) return false;
    distance=hit;
    normal=make_float3(local_normal.x*cosine-local_normal.y*sine,
        local_normal.x*sine+local_normal.y*cosine,local_normal.z);
    return true;
}

__device__ bool intersect_gallery_opaque(
    const Ray& ray, GalleryArena arena, const OrientedBox& collider,
    float maximum_distance, float& distance, float3& color)
{
    float3 normal{};
    if (arena == GalleryArena::none) {
        if (!intersect_wire_box(ray, collider, maximum_distance, distance, normal))
            return false;
        color = shade_wire_box(normal);
        return true;
    }
    if (arena == GalleryArena::bowl) {
        const float3 offset = subtract(ray.origin, bowl_center);
        const float b = dot(offset, ray.direction);
        const float outer = bowl_inner_radius + bowl_wall_thickness;
        const float discriminant = b*b - (dot(offset, offset) - outer*outer);
        bool bowl = false;
        if (discriminant >= 0.0F) {
            const float root = sqrtf(discriminant);
            for (int side = 0; side < 2; ++side) {
                const float hit = side == 0 ? -b - root : -b + root;
                if (hit <= 1.0e-5F || hit >= maximum_distance) continue;
                const float3 point = add(ray.origin, multiply(ray.direction, hit));
                if (point.y > bowl_center.y + 0.015F) continue;
                distance = hit;
                normal = normalize(subtract(point, bowl_center));
                const int tiles = static_cast<int>(floorf(point.x * 6.0F)) +
                    static_cast<int>(floorf(point.y * 6.0F)) +
                    static_cast<int>(floorf(point.z * 6.0F));
                constexpr float pi = 3.14159265358979323846F;
                const float3 direction = normalize(subtract(point, bowl_center));
                const float longitude = (atan2f(direction.z, direction.x) + pi) /
                    (2.0F * pi);
                const float latitude = acosf(fminf(1.0F,
                    fmaxf(0.0F, -direction.y))) / (0.5F * pi);
                const std::uint32_t paint_x = min(bowl_paint_width - 1U,
                    static_cast<std::uint32_t>(longitude * bowl_paint_width));
                const std::uint32_t paint_y = min(bowl_paint_height - 1U,
                    static_cast<std::uint32_t>(latitude * bowl_paint_height));
                const bool painted = device_bowl_paint_pixels != nullptr &&
                    device_bowl_paint_pixels[paint_y * bowl_paint_width + paint_x] != 0U;
                const float3 ceramic = (tiles & 1) == 0
                    ? make_float3(0.78F, 0.10F, 0.08F)
                    : make_float3(0.36F, 0.025F, 0.02F);
                const float3 wet_blue = (tiles & 1) == 0
                    ? make_float3(0.08F, 0.45F, 1.0F)
                    : make_float3(0.01F, 0.15F, 0.52F);
                color = multiply(painted ? wet_blue : ceramic,
                    0.35F + 0.65F * fabsf(normal.y));
                bowl = true;
                break;
            }
        }
        for (std::uint32_t peg = 0U; peg < bowl_peg_count; ++peg) {
            float peg_distance{};
            float3 peg_normal{};
            if (!intersect_vertical_cylinder(ray, bowl_peg(peg), bowl_peg_radius,
                    0.5F * bowl_peg_height,
                    bowl ? distance : maximum_distance, peg_distance, peg_normal))
                continue;
            bowl = true;
            distance = peg_distance;
            normal = peg_normal;
            color = multiply(make_float3(0.78F, 0.34F, 0.08F),
                0.28F + 0.72F * fmaxf(0.0F,
                    dot(normal, normalize(make_float3(-0.48F, 0.84F, 0.34F)))));
        }
        float sphere_distance{};
        if (intersect_particle_sphere(ray, collider.center, collider.half_extents.x,
                bowl ? distance : maximum_distance, sphere_distance, normal)) {
            distance = sphere_distance;
            color = multiply(make_float3(0.94F, 0.47F, 0.12F),
                0.24F + 0.76F * fmaxf(0.0F,
                    dot(normal, normalize(make_float3(-0.48F, 0.84F, 0.34F)))));
            return true;
        }
        return bowl;
    }
    if (arena == GalleryArena::water_wheel) {
        bool hit_any = false;
        const float ramp_denominator =
            ray.direction.y - water_wheel_ramp_gradient * ray.direction.x;
        if (fabsf(ramp_denominator) > 1.0e-7F) {
            // Two disjoint 10-degree ramps: a high inlet ending at the wheel's
            // nine-o'clock point and a low collector beginning below its
            // vertical outlet. There is intentionally no invisible plane in
            // the open space between them.
            for (int support = 0; support < 2; ++support) {
                const float minimum_x = support == 0
                    ? water_wheel_inlet_start_x : water_wheel_collector_start_x;
                const float maximum_x = support == 0
                    ? water_wheel_entry_x : water_wheel_collector_end_x;
                const float reference_x = support == 0
                    ? water_wheel_entry_x : water_wheel_exit_x;
                const float reference_y = support == 0
                    ? water_wheel_center.y : water_wheel_ground_y;
                const float ramp_hit = (reference_y + water_wheel_ramp_gradient *
                    (ray.origin.x - reference_x) - ray.origin.y) /
                    ramp_denominator;
                if (ramp_hit <= 1.0e-5F || ramp_hit >= maximum_distance ||
                    (hit_any && ramp_hit >= distance)) continue;
                const float3 point = add(ray.origin, multiply(ray.direction, ramp_hit));
                if (point.x < minimum_x || point.x > maximum_x ||
                    fabsf(point.z - water_wheel_center.z) >
                        water_wheel_ground_half_depth) continue;
                distance = ramp_hit;
                const int tiles = static_cast<int>(floorf(point.x * 3.0F)) +
                    static_cast<int>(floorf(point.z * 3.0F));
                color = (tiles & 1) == 0 ? make_float3(0.49F, 0.54F, 0.60F)
                    : make_float3(0.08F, 0.10F, 0.14F);
                hit_any = true;
            }
        }
        float best = hit_any ? distance : maximum_distance;
        const float3 offset = subtract(ray.origin, water_wheel_center);
        const float a = ray.direction.x * ray.direction.x +
            ray.direction.y * ray.direction.y;
        const float b = offset.x * ray.direction.x + offset.y * ray.direction.y;
        const float c = offset.x * offset.x + offset.y * offset.y -
            water_wheel_shell_radius * water_wheel_shell_radius;
        const float discriminant = b*b - a*c;
        if (a > 1.0e-8F && discriminant >= 0.0F) {
            const float root = sqrtf(discriminant);
            const float candidates[2]{(-b - root) / a, (-b + root) / a};
            for (int candidate = 0; candidate < 2; ++candidate) {
                const float value = candidates[candidate];
                if (value <= 1.0e-5F || value >= best) continue;
                const float3 point = add(ray.origin, multiply(ray.direction, value));
                if (fabsf(point.z - water_wheel_center.z) >
                    water_wheel_half_depth + 0.08F) continue;
                if (water_wheel_shell_opening(
                    subtract(point, water_wheel_center))) continue;
                const float3 radial = normalize(make_float3(
                    point.x - water_wheel_center.x,
                    point.y - water_wheel_center.y, 0.0F));
                best = value;
                distance = value;
                color = multiply(make_float3(0.28F, 0.34F, 0.42F),
                    0.35F + 0.65F * fabsf(dot(radial,
                        normalize(make_float3(-0.48F, 0.84F, 0.34F)))));
                hit_any = true;
            }
        }
        if (fabsf(ray.direction.z) > 1.0e-8F) {
            for (int cap = -1; cap <= 1; cap += 2) {
                const float cap_z = water_wheel_center.z +
                    static_cast<float>(cap) * water_wheel_half_depth;
                const float value = (cap_z - ray.origin.z) / ray.direction.z;
                if (value <= 1.0e-5F || value >= best) continue;
                const float3 point = add(ray.origin, multiply(ray.direction, value));
                const float dx = point.x - water_wheel_center.x;
                const float dy = point.y - water_wheel_center.y;
                const float radial = sqrtf(dx*dx + dy*dy);
                if (fabsf(radial - water_wheel_shell_radius) > 0.08F) continue;
                if (water_wheel_shell_opening(
                    subtract(point, water_wheel_center))) continue;
                best = value;
                distance = value;
                color = make_float3(0.24F, 0.29F, 0.36F);
                hit_any = true;
            }
        }
        // Only the axle hub is solid. The old opaque 2 m disk hid the soft
        // cross and falsely occupied the space in which water should sit.
        const float core_c = offset.x * offset.x + offset.y * offset.y -
            water_wheel_hub_radius * water_wheel_hub_radius;
        const float core_discriminant = b*b - a*core_c;
        if (a > 1.0e-8F && core_discriminant >= 0.0F) {
            const float root = sqrtf(core_discriminant);
            const float candidates[2]{(-b - root) / a, (-b + root) / a};
            for (int candidate = 0; candidate < 2; ++candidate) {
                const float value = candidates[candidate];
                if (value <= 1.0e-5F || value >= best) continue;
                const float3 point = add(ray.origin, multiply(ray.direction, value));
                if (fabsf(point.z - water_wheel_center.z) > water_wheel_half_depth)
                    continue;
                best = value;
                distance = value;
                const float3 core_normal = normalize(make_float3(
                    point.x - water_wheel_center.x,
                    point.y - water_wheel_center.y, 0.0F));
                color = multiply(make_float3(0.16F, 0.20F, 0.25F),
                    0.35F + 0.65F * fabsf(dot(core_normal,
                        normalize(make_float3(-0.48F, 0.84F, 0.34F)))));
                hit_any = true;
            }
        }
        if (fabsf(ray.direction.z) > 1.0e-8F) {
            for (int cap = -1; cap <= 1; cap += 2) {
                const float cap_z = water_wheel_center.z +
                    static_cast<float>(cap) * water_wheel_half_depth;
                const float value = (cap_z - ray.origin.z) / ray.direction.z;
                if (value <= 1.0e-5F || value >= best) continue;
                const float3 point = add(ray.origin, multiply(ray.direction, value));
                const float dx = point.x - water_wheel_center.x;
                const float dy = point.y - water_wheel_center.y;
                if (dx*dx + dy*dy >
                    water_wheel_hub_radius * water_wheel_hub_radius) continue;
                best = value;
                distance = value;
                const int tiles = static_cast<int>(floorf(point.x * 2.0F)) +
                    static_cast<int>(floorf(point.y * 2.0F));
                color = (tiles & 1) == 0 ? make_float3(0.20F, 0.24F, 0.29F)
                    : make_float3(0.10F, 0.13F, 0.17F);
                hit_any = true;
            }
        }
        // A common axle joins the two soft crosses. Outside each cross is a
        // thin rigid rim attached at the cross's four pinned tips; using an
        // annulus keeps the compliant axle-to-rim load path visible.
        const float axle_radius = 0.11F;
        const float axle_c = offset.x * offset.x + offset.y * offset.y -
            axle_radius * axle_radius;
        const float axle_discriminant = b*b - a*axle_c;
        if (a > 1.0e-8F && axle_discriminant >= 0.0F) {
            const float root = sqrtf(axle_discriminant);
            const float candidates[2]{(-b - root) / a, (-b + root) / a};
            for (int candidate = 0; candidate < 2; ++candidate) {
                const float value = candidates[candidate];
                if (value <= 1.0e-5F || value >= best) continue;
                const float3 point = add(ray.origin, multiply(ray.direction, value));
                if (fabsf(point.z - water_wheel_center.z) >
                    water_wheel_axle_half_length) continue;
                best = value;
                distance = value;
                color = make_float3(0.34F, 0.38F, 0.43F);
                hit_any = true;
            }
        }
        constexpr float rim_half_width = 0.065F;
        const float rim_inner = water_wheel_outer_disk_radius - rim_half_width;
        const float rim_outer = water_wheel_outer_disk_radius + rim_half_width;
        for (int side = -1; side <= 1; side += 2) {
            const float rim_z = water_wheel_center.z +
                static_cast<float>(side) * water_wheel_outer_disk_offset;
            const float radii[2]{rim_inner, rim_outer};
            for (int surface = 0; surface < 2; ++surface) {
                const float rim_c = offset.x*offset.x + offset.y*offset.y -
                    radii[surface] * radii[surface];
                const float rim_discriminant = b*b - a*rim_c;
                if (!(a > 1.0e-8F) || rim_discriminant < 0.0F) continue;
                const float root = sqrtf(rim_discriminant);
                const float candidates[2]{(-b - root) / a, (-b + root) / a};
                for (int candidate = 0; candidate < 2; ++candidate) {
                    const float value = candidates[candidate];
                    if (value <= 1.0e-5F || value >= best) continue;
                    const float3 point = add(ray.origin, multiply(ray.direction, value));
                    if (fabsf(point.z - rim_z) >
                        water_wheel_outer_disk_half_thickness) continue;
                    best = value;
                    distance = value;
                    color = make_float3(0.24F, 0.28F, 0.33F);
                    hit_any = true;
                }
            }
            if (fabsf(ray.direction.z) > 1.0e-8F) {
                for (int cap = -1; cap <= 1; cap += 2) {
                    const float cap_z = rim_z + static_cast<float>(cap) *
                        water_wheel_outer_disk_half_thickness;
                    const float value = (cap_z - ray.origin.z) / ray.direction.z;
                    if (value <= 1.0e-5F || value >= best) continue;
                    const float3 point = add(ray.origin, multiply(ray.direction, value));
                    const float dx = point.x - water_wheel_center.x;
                    const float dy = point.y - water_wheel_center.y;
                    const float radius = sqrtf(dx*dx + dy*dy);
                    if (radius < rim_inner || radius > rim_outer) continue;
                    best = value;
                    distance = value;
                    color = make_float3(0.30F, 0.34F, 0.39F);
                    hit_any = true;
                }
            }
        }
        // The rigid-looking radial spokes were removed: the visible and
        // simulated load path from axle to rim is now the soft cross itself.
        constexpr float two_pi = 6.28318530717958647692F;
        for (std::uint32_t fin = 0U; fin < water_wheel_fin_count; ++fin) {
            float fin_distance{};
            float3 fin_normal{};
            const float angle = collider.yaw + two_pi * static_cast<float>(fin) /
                static_cast<float>(water_wheel_fin_count);
            if (!intersect_water_wheel_fin(
                    ray, angle, best, fin_distance, fin_normal)) continue;
            best = fin_distance;
            distance = fin_distance;
            color = multiply(make_float3(0.76F, 0.31F, 0.08F),
                0.30F + 0.70F * fmaxf(0.0F, dot(fin_normal,
                    normalize(make_float3(-0.48F, 0.84F, 0.34F)))));
            hit_any = true;
        }
        for (std::uint32_t rung=0U;rung<water_wheel_outer_rung_count;++rung) {
            float rung_distance{};
            float3 rung_normal{};
            const float angle=collider.yaw+two_pi*static_cast<float>(rung)/
                static_cast<float>(water_wheel_outer_rung_count);
            if (!intersect_water_wheel_outer_rung(
                    ray,angle,best,rung_distance,rung_normal)) continue;
            best=rung_distance;
            distance=rung_distance;
            color=multiply(make_float3(0.92F,0.66F,0.10F),
                0.30F+0.70F*fmaxf(0.0F,dot(rung_normal,
                    normalize(make_float3(-0.48F,0.84F,0.34F)))));
            hit_any=true;
        }
        // The abandoned ladder/tow overlay is replaced by two simple exit
        // platforms just below the wheel crown. Their gap exposes only the
        // top of the wheel; the soft crosses and outer rims sit farther along
        // the axle, clear of the central fins and level geometry.
        for (int side = -1; side <= 1; side += 2) {
            const float inner = water_wheel_center.x +
                static_cast<float>(side) * water_wheel_top_platform_gap_half_width;
            const float outer = water_wheel_center.x +
                static_cast<float>(side) * water_wheel_top_platform_outer_x;
            float platform_distance{};
            float3 platform_normal{};
            if (!intersect_box(ray,
                    make_float3(0.5F * (inner + outer),
                        water_wheel_top_platform_y - 0.035F,
                        water_wheel_stage_z),
                    make_float3(0.5F * fabsf(outer - inner), 0.035F,
                        water_wheel_top_platform_half_depth),
                    best, platform_distance, platform_normal)) continue;
            best = platform_distance;
            distance = platform_distance;
            color = multiply(make_float3(0.34F, 0.39F, 0.46F),
                0.32F + 0.68F * fabsf(platform_normal.y));
            hit_any = true;
        }
        // Low bumper rails keep the player sphere on the front stage while
        // preserving a clear view of the wheel and soft crosses.
        for (int side = -1; side <= 1; side += 2) {
            float bumper_distance{};
            float3 bumper_normal{};
            if (!intersect_box(ray,
                    make_float3(water_wheel_center.x,
                        water_wheel_top_platform_y+water_wheel_top_bumper_height,
                        water_wheel_stage_z+static_cast<float>(side)*
                            (water_wheel_top_platform_half_depth+
                                water_wheel_top_bumper_thickness)),
                    make_float3(water_wheel_top_platform_outer_x,
                        water_wheel_top_bumper_height,
                        water_wheel_top_bumper_thickness),
                    best,bumper_distance,bumper_normal)) continue;
            best=bumper_distance;
            distance=bumper_distance;
            color=multiply(make_float3(0.92F,0.66F,0.10F),
                0.34F+0.66F*fabsf(bumper_normal.y));
            hit_any=true;
        }
        // Close the starting end; the opposite (left) end remains open and
        // is the authored win direction.
        {
            float bumper_distance{};
            float3 bumper_normal{};
            if (intersect_box(ray,
                    make_float3(water_wheel_center.x+
                            water_wheel_top_platform_outer_x+
                            water_wheel_top_bumper_thickness,
                        water_wheel_top_platform_y+water_wheel_top_bumper_height,
                        water_wheel_stage_z),
                    make_float3(water_wheel_top_bumper_thickness,
                        water_wheel_top_bumper_height,
                        water_wheel_top_platform_half_depth+
                            2.0F*water_wheel_top_bumper_thickness),
                    best,bumper_distance,bumper_normal)) {
                best=bumper_distance;
                distance=bumper_distance;
                color=multiply(make_float3(0.92F,0.66F,0.10F),
                    0.34F+0.66F*fabsf(bumper_normal.y));
                hit_any=true;
            }
        }
        float sphere_distance{};
        float3 sphere_normal{};
        if (intersect_particle_sphere(ray, collider.center,
                collider.half_extents.x, hit_any ? distance : maximum_distance,
                sphere_distance, sphere_normal)) {
            distance = sphere_distance;
            const float3 material_normal=inverse_rotate_quaternion(
                collider.sphere_orientation,sphere_normal);
            const int longitude=static_cast<int>(floorf(
                (atan2f(material_normal.z,material_normal.x)+pi)/(2.0F*pi)*16.0F));
            const int latitude=static_cast<int>(floorf(
                acosf(fminf(1.0F,fmaxf(-1.0F,material_normal.y)))/pi*8.0F));
            const float3 base=((longitude+latitude)&1)==0
                ? make_float3(0.94F,0.72F,0.18F)
                : make_float3(0.07F,0.09F,0.13F);
            color = multiply(base,
                0.24F + 0.76F * fmaxf(0.0F,
                    dot(sphere_normal,
                        normalize(make_float3(-0.48F, 0.84F, 0.34F)))));
            return true;
        }
        return hit_any;
    }
    if (arena == GalleryArena::rope_post) {
        bool hit_any = false;
        float best = maximum_distance;
        if (ray.direction.y < -1.0e-7F) {
            const float hit = (course_floor_y - ray.origin.y) / ray.direction.y;
            if (hit > 1.0e-5F && hit < best) {
                const float3 point = add(ray.origin, multiply(ray.direction, hit));
                if (fabsf(point.x) <= 3.1F && fabsf(point.z) <= 2.7F) {
                    best = hit;
                    distance = hit;
                    const int tiles = static_cast<int>(floorf(point.x * 3.0F)) +
                        static_cast<int>(floorf(point.z * 3.0F));
                    color = (tiles & 1) == 0 ? make_float3(0.49F, 0.54F, 0.60F)
                        : make_float3(0.08F, 0.10F, 0.14F);
                    hit_any = true;
                }
            }
        }
        float hit{};
        float3 hit_normal{};
        if (intersect_vertical_cylinder(ray, rope_post_center, rope_post_radius,
                0.5F * rope_post_height, best, hit, hit_normal)) {
            best = hit;
            distance = hit;
            color = multiply(make_float3(0.32F, 0.37F, 0.44F),
                0.30F + 0.70F * fmaxf(0.0F,
                    dot(hit_normal, normalize(make_float3(-0.48F, 0.84F, 0.34F)))));
            hit_any = true;
        }
        // The compact rope cage contains a second real rigid sphere. It is analytic in
        // the renderer just like the primary sphere; the rope graph is no
        // longer abused as a deformable glass surface.
        if (collider.secondary_sphere_radius > 0.0F &&
            intersect_particle_sphere(ray,collider.secondary_sphere_center,
                collider.secondary_sphere_radius,best,hit,hit_normal)) {
            best=hit;
            distance=hit;
            const float facing=fabsf(dot(hit_normal,ray.direction));
            const float fresnel=0.04F+0.96F*powf(1.0F-facing,5.0F);
            const float light=0.20F+0.80F*fmaxf(0.0F,dot(hit_normal,
                normalize(make_float3(-0.48F,0.84F,0.34F))));
            color=add(multiply(make_float3(0.08F,0.28F,0.38F),0.55F*light),
                multiply(make_float3(0.72F,0.94F,1.0F),0.70F*fresnel));
            hit_any=true;
        }
        if (intersect_particle_sphere(ray, collider.center, collider.half_extents.x,
                best, hit, hit_normal)) {
            distance = hit;
            const float3 material_normal=inverse_rotate_quaternion(
                collider.sphere_orientation,hit_normal);
            const int longitude=static_cast<int>(floorf(
                (atan2f(material_normal.z,material_normal.x)+pi)/(2.0F*pi)*16.0F));
            const int latitude=static_cast<int>(floorf(
                acosf(fminf(1.0F,fmaxf(-1.0F,material_normal.y)))/pi*8.0F));
            const float3 base=((longitude+latitude)&1)==0
                ? make_float3(0.92F,0.92F,0.92F)
                : make_float3(0.045F,0.055F,0.075F);
            color = multiply(base,
                0.24F + 0.76F * fmaxf(0.0F,
                    dot(hit_normal, normalize(make_float3(-0.48F, 0.84F, 0.34F)))));
            return true;
        }
        return hit_any;
    }
    if (arena == GalleryArena::rope_bridge) {
        bool hit_any = false;
        float best = maximum_distance;
        for (int side = -1; side <= 1; side += 2) {
            const float inner = static_cast<float>(side)*rope_bridge_land_inner_z;
            const float outer = static_cast<float>(side)*rope_bridge_land_outer_z;
            float hit{};
            float3 hit_normal{};
            if (!intersect_box(ray,
                    make_float3(0.0F,rope_bridge_land_y-0.06F,
                        0.5F*(inner+outer)),
                    make_float3(rope_bridge_land_half_width,0.06F,
                        0.5F*fabsf(outer-inner)),
                    best,hit,hit_normal)) continue;
            best = hit;
            distance = hit;
            const float3 point = add(ray.origin,multiply(ray.direction,hit));
            const int tiles = static_cast<int>(floorf(point.x*3.0F))+
                static_cast<int>(floorf(point.z*3.0F));
            color = multiply((tiles&1)==0 ? make_float3(0.42F,0.48F,0.55F)
                                           : make_float3(0.07F,0.09F,0.13F),
                0.35F+0.65F*fabsf(hit_normal.y));
            hit_any = true;
        }
        float sphere_distance{};
        float3 sphere_normal{};
        if (intersect_particle_sphere(ray,collider.center,collider.half_extents.x,
                best,sphere_distance,sphere_normal)) {
            distance = sphere_distance;
            color = multiply(make_float3(0.08F,0.36F,0.95F),
                0.25F+0.75F*fmaxf(0.0F,dot(sphere_normal,
                    normalize(make_float3(-0.48F,0.84F,0.34F)))));
            return true;
        }
        return hit_any;
    }
    if (arena == GalleryArena::slope || arena == GalleryArena::ground ||
        arena == GalleryArena::grass) {
        const float gradient = arena == GalleryArena::slope ? slope_gradient : 0.0F;
        const float height = arena == GalleryArena::slope
            ? slope_start_y : course_floor_y;
        const float denominator = ray.direction.y - gradient * ray.direction.x;
        if (fabsf(denominator) < 1.0e-7F) return false;
        const float hit = (height + gradient * ray.origin.x - ray.origin.y) /
            denominator;
        if (hit <= 1.0e-5F || hit >= maximum_distance) return false;
        const float3 point = add(ray.origin, multiply(ray.direction, hit));
        if (fabsf(point.x) > 3.1F || fabsf(point.z) > 2.7F) return false;
        distance = hit;
        const int tiles = static_cast<int>(floorf(point.x * 3.0F)) +
            static_cast<int>(floorf(point.z * 3.0F));
        color = (tiles & 1) == 0 ? make_float3(0.49F, 0.54F, 0.60F)
            : make_float3(0.08F, 0.10F, 0.14F);
        return true;
    }
    if (arena == GalleryArena::enclosed_box || arena == GalleryArena::hot_pan ||
        arena == GalleryArena::cloth_basin ||
        arena == GalleryArena::ground_box || arena == GalleryArena::low_ceiling_box ||
        arena == GalleryArena::fishing_tank) {
        // The camera normally lives inside this room. Intersect the first of
        // its six inward-facing planes, then compose the dynamic sphere in
        // front of that wall. The same constants drive collision projection.
        float wall_distance = maximum_distance;
        float3 wall_normal{};
        const float3 room_center = arena == GalleryArena::fishing_tank
            ? fishing_tank_center : (arena == GalleryArena::low_ceiling_box
                ? low_gallery_box_center : gallery_box_center);
        const float3 room_half_extents = arena == GalleryArena::fishing_tank
            ? fishing_tank_half_extents : (arena == GalleryArena::low_ceiling_box
                ? low_gallery_box_half_extents : gallery_box_half_extents);
        const float3 minimum = subtract(room_center, room_half_extents);
        const float3 maximum = add(room_center, room_half_extents);
        const auto plane = [&](float coordinate, float direction, float boundary,
                               float3 candidate_normal) {
            if (fabsf(direction) < 1.0e-7F) return;
            if (arena == GalleryArena::low_ceiling_box &&
                candidate_normal.y < -0.5F) return;
            // Draw inward-facing exits only. Cameras outside the containment
            // box can therefore look through the near wall while still seeing
            // the fixed floor, ceiling, and far walls that bound the physics.
            if (dot(ray.direction, candidate_normal) >= 0.0F) return;
            const float hit = (boundary - coordinate) / direction;
            if (hit <= 1.0e-5F || hit >= wall_distance) return;
            const float3 point = add(ray.origin, multiply(ray.direction, hit));
            if (arena == GalleryArena::ground_box && candidate_normal.y > 0.5F &&
                inside_ground_pit(point)) return;
            if (point.x < minimum.x - 1.0e-4F || point.x > maximum.x + 1.0e-4F ||
                point.y < minimum.y - 1.0e-4F || point.y > maximum.y + 1.0e-4F ||
                point.z < minimum.z - 1.0e-4F || point.z > maximum.z + 1.0e-4F)
                return;
            wall_distance = hit;
            wall_normal = candidate_normal;
        };
        plane(ray.origin.x, ray.direction.x, minimum.x,
            make_float3(1.0F, 0.0F, 0.0F));
        plane(ray.origin.x, ray.direction.x, maximum.x,
            make_float3(-1.0F, 0.0F, 0.0F));
        plane(ray.origin.y, ray.direction.y, minimum.y,
            make_float3(0.0F, 1.0F, 0.0F));
        plane(ray.origin.y, ray.direction.y, maximum.y,
            make_float3(0.0F, -1.0F, 0.0F));
        plane(ray.origin.z, ray.direction.z, minimum.z,
            make_float3(0.0F, 0.0F, 1.0F));
        plane(ray.origin.z, ray.direction.z, maximum.z,
            make_float3(0.0F, 0.0F, -1.0F));
        if (arena == GalleryArena::ground_box && ray.direction.y < -1.0e-7F) {
            const float hit = (ground_pit_bottom_y - ray.origin.y) / ray.direction.y;
            if (hit > 1.0e-5F && hit < wall_distance) {
                const float3 point = add(ray.origin, multiply(ray.direction, hit));
                if (inside_ground_pit(point)) {
                    wall_distance = hit;
                    wall_normal = make_float3(0.0F, 1.0F, 0.0F);
                }
            }
        }
        bool wall = wall_distance < maximum_distance;
        if (wall) {
            const float3 point = add(ray.origin, multiply(ray.direction, wall_distance));
            const int tiles = static_cast<int>(floorf(point.x * 2.0F)) +
                static_cast<int>(floorf(point.y * 2.0F)) +
                static_cast<int>(floorf(point.z * 2.0F));
            distance = wall_distance;
            color = multiply((tiles & 1) == 0
                ? make_float3(0.48F, 0.54F, 0.61F)
                : make_float3(0.075F, 0.095F, 0.13F),
                0.38F + 0.62F * fabsf(wall_normal.y));
        }
        if (arena == GalleryArena::hot_pan) {
            float pan_distance{};
            float3 pan_normal{};
            if (intersect_box(ray,
                    make_float3(gallery_box_center.x,-1.04F,gallery_box_center.z),
                    make_float3(1.65F,0.05F,1.20F),
                    wall ? distance : maximum_distance,
                    pan_distance,pan_normal)) {
                const float3 point=add(ray.origin,multiply(ray.direction,pan_distance));
                const float dx=point.x-gallery_box_center.x;
                const float dz=point.z-gallery_box_center.z;
                const float radial=sqrtf(dx*dx+dz*dz);
                const bool burner=fabsf(radial-0.82F)<0.08F || radial<0.22F;
                wall=true;
                distance=pan_distance;
                color=multiply(burner ? make_float3(0.92F,0.12F,0.025F)
                                      : make_float3(0.075F,0.085F,0.095F),
                    0.32F+0.68F*fabsf(pan_normal.y));
            }
        }
        if (arena == GalleryArena::low_ceiling_box) {
            constexpr int grate_lines = 9;
            constexpr float grate_half_width = 0.025F;
            for (int line = 0; line < grate_lines; ++line) {
                const float alpha = static_cast<float>(line) /
                    static_cast<float>(grate_lines - 1);
                const float x = minimum.x + alpha * (maximum.x - minimum.x);
                const float z = minimum.z + alpha * (maximum.z - minimum.z);
                float grate_distance{};
                float3 grate_normal{};
                if (intersect_box(ray, make_float3(x, hanging_ceiling_y, room_center.z),
                        make_float3(grate_half_width, 0.025F, room_half_extents.z),
                        wall ? distance : maximum_distance,
                        grate_distance, grate_normal)) {
                    wall = true;
                    distance = grate_distance;
                    color = multiply(make_float3(0.34F, 0.38F, 0.43F),
                        0.32F + 0.68F * fabsf(grate_normal.y));
                }
                if (intersect_box(ray, make_float3(room_center.x, hanging_ceiling_y, z),
                        make_float3(room_half_extents.x, 0.025F, grate_half_width),
                        wall ? distance : maximum_distance,
                        grate_distance, grate_normal)) {
                    wall = true;
                    distance = grate_distance;
                    color = multiply(make_float3(0.34F, 0.38F, 0.43F),
                        0.32F + 0.68F * fabsf(grate_normal.y));
                }
            }
        }
        // Context 5 exposes the four-by-four square support lattice just below
        // the horizontal cloth. The tall containment walls remain invisible.
        if (arena == GalleryArena::cloth_basin) {
            constexpr float support_half_width = 0.018F;
            constexpr float support_half_height = 0.025F;
            const float support_y = cloth_basin_center.y - 0.040F;
            for (int line = 0; line < 4; ++line) {
                const float alpha = static_cast<float>(line) / 3.0F;
                const float x = cloth_basin_center.x +
                    (2.0F * alpha - 1.0F) * cloth_basin_inner_half_extents.x;
                const float z = cloth_basin_center.z +
                    (2.0F * alpha - 1.0F) * cloth_basin_inner_half_extents.y;
                float support_distance{};
                float3 support_normal{};
                if (intersect_box(ray, make_float3(x, support_y, cloth_basin_center.z),
                        make_float3(support_half_width, support_half_height,
                            cloth_basin_inner_half_extents.y + support_half_width),
                        wall ? distance : maximum_distance,
                        support_distance, support_normal)) {
                    wall = true;
                    distance = support_distance;
                    color = multiply(make_float3(0.24F, 0.28F, 0.34F),
                        0.35F + 0.65F * fabsf(support_normal.y));
                }
                if (intersect_box(ray, make_float3(cloth_basin_center.x, support_y, z),
                        make_float3(cloth_basin_inner_half_extents.x + support_half_width,
                            support_half_height, support_half_width),
                        wall ? distance : maximum_distance,
                        support_distance, support_normal)) {
                    wall = true;
                    distance = support_distance;
                    color = multiply(make_float3(0.24F, 0.28F, 0.34F),
                        0.35F + 0.65F * fabsf(support_normal.y));
                }
            }
            const float minimum_x = cloth_basin_center.x -
                cloth_basin_inner_half_extents.x;
            const float maximum_x = cloth_basin_center.x +
                cloth_basin_inner_half_extents.x;
            const float wall_bottom = cloth_basin_center.y + 0.035F;
            for (std::uint32_t divider = 0U;
                 divider < cloth_snake_wall_count; ++divider) {
                const bool gap_right = (divider & 1U) == 0U;
                const float start_x = minimum_x +
                    (gap_right ? 0.0F : cloth_snake_gap_width);
                const float end_x = maximum_x -
                    (gap_right ? cloth_snake_gap_width : 0.0F);
                float divider_distance{};
                float3 divider_normal{};
                if (!intersect_box(ray,
                        make_float3(0.5F * (start_x + end_x),
                            0.5F * (wall_bottom + cloth_snake_wall_top),
                            cloth_snake_wall_z(divider)),
                        make_float3(0.5F * (end_x - start_x),
                            0.5F * (cloth_snake_wall_top - wall_bottom),
                            cloth_snake_wall_half_thickness),
                        wall ? distance : maximum_distance,
                        divider_distance, divider_normal)) continue;
                wall = true;
                distance = divider_distance;
                color = multiply(make_float3(0.16F, 0.34F, 0.50F),
                    0.32F + 0.68F * fabsf(divider_normal.y));
            }

            // Context 5 uses a compact boat hull while retaining the stable
            // finite-mass sphere proxy for collision.  Three overlapping
            // boxes give it an unmistakable bow, deck, and cabin silhouette.
            float boat_distance = wall ? distance : maximum_distance;
            float3 boat_normal{};
            bool boat = false;
            const float radius = collider.half_extents.x;
            const auto boat_part = [&](float3 center, float3 half) {
                float candidate{};
                float3 candidate_normal{};
                if (intersect_box(ray, center, half, boat_distance,
                        candidate, candidate_normal)) {
                    boat = true;
                    boat_distance = candidate;
                    boat_normal = candidate_normal;
                }
            };
            boat_part(add(collider.center,make_float3(0.0F,-0.15F*radius,0.0F)),
                make_float3(1.55F*radius,0.42F*radius,0.78F*radius));
            boat_part(add(collider.center,make_float3(-0.20F*radius,0.32F*radius,0.0F)),
                make_float3(0.72F*radius,0.30F*radius,0.58F*radius));
            boat_part(add(collider.center,make_float3(1.15F*radius,-0.04F*radius,0.0F)),
                make_float3(0.40F*radius,0.25F*radius,0.60F*radius));
            if (boat) {
                distance=boat_distance;
                color=multiply(make_float3(0.92F,0.38F,0.055F),
                    0.28F+0.72F*fmaxf(0.0F,dot(boat_normal,
                        normalize(make_float3(-0.48F,0.84F,0.34F)))));
                wall=true;
            }
        }
        if (arena == GalleryArena::fishing_tank) {
            // The finite sphere used by the fluid/contact solver is presented
            // as a brass-banded treasure chest.
            float chest_distance = wall ? distance : maximum_distance;
            float3 chest_normal{};
            bool chest{};
            const float radius = collider.half_extents.x;
            const auto chest_part = [&](float3 offset, float3 half) {
                float candidate{};
                float3 candidate_normal{};
                if (intersect_box(ray, add(collider.center, offset), half,
                        chest_distance, candidate, candidate_normal)) {
                    chest = true;
                    chest_distance = candidate;
                    chest_normal = candidate_normal;
                }
            };
            chest_part(make_float3(0.0F,-0.12F*radius,0.0F),
                make_float3(1.30F*radius,0.66F*radius,0.72F*radius));
            chest_part(make_float3(0.0F,0.48F*radius,0.0F),
                make_float3(1.30F*radius,0.22F*radius,0.72F*radius));
            if (chest) {
                distance = chest_distance;
                const float band = fabsf(chest_normal.y) < 0.4F ? 0.70F : 0.30F;
                color = multiply(add(
                    multiply(make_float3(0.42F,0.16F,0.035F),1.0F-band),
                    multiply(make_float3(0.95F,0.62F,0.10F),band)),
                    0.35F+0.65F*fmaxf(0.0F,dot(chest_normal,
                        normalize(make_float3(-0.48F,0.84F,0.34F)))));
                wall = true;
            }
        }
        float sphere_distance{};
        if (arena != GalleryArena::ground_box && arena != GalleryArena::cloth_basin &&
            arena != GalleryArena::fishing_tank &&
            intersect_particle_sphere(ray, collider.center, collider.half_extents.x,
                wall ? distance : maximum_distance, sphere_distance, normal)) {
            distance = sphere_distance;
            const float3 material_normal = inverse_rotate_quaternion(
                collider.sphere_orientation,normal);
            const bool sphere_painted = arena == GalleryArena::low_ceiling_box &&
                painted_sphere_texel(normal,collider.sphere_orientation);
            const int longitude_tile = static_cast<int>(floorf(
                (atan2f(material_normal.z,material_normal.x)+pi)/(2.0F*pi)*16.0F));
            const int latitude_tile = static_cast<int>(floorf(
                acosf(fminf(1.0F,fmaxf(-1.0F,material_normal.y)))/pi*8.0F));
            const float3 sphere_base = sphere_painted
                ? make_float3(0.04F,0.34F,1.0F)
                : (((longitude_tile+latitude_tile)&1)==0
                    ? make_float3(0.94F,0.72F,0.18F)
                    : make_float3(0.26F,0.07F,0.025F));
            color = multiply(sphere_base,
                0.24F + 0.76F * fmaxf(0.0F,
                    dot(normal, normalize(make_float3(-0.48F, 0.84F, 0.34F)))));
            return true;
        }
        return wall;
    }
    return false;
}

template <bool obstacle_course, GalleryArena arena = GalleryArena::none>
__device__ bool trace_opaque_scene(const Ray& ray, const OrientedBox& collider,
    SoftBodyRenderView soft_body, float maximum_distance,
    float& distance, float3& color)
{
    if constexpr (obstacle_course) {
        return trace_course_opaque(ray, soft_body, maximum_distance, distance, color);
    }
    float3 opaque_color{};
    const bool box = intersect_gallery_opaque(
        ray, arena, collider, maximum_distance, distance, opaque_color);
    SoftBodyHit body = trace_soft_body(ray, soft_body, maximum_distance);
    if constexpr (arena == GalleryArena::rope_post) {
        // Historical builds skinned a fake sphere onto cage vertices. Keep the
        // asset topology for compatibility but never display that deforming
        // proxy now that the caged object has its own rigid state.
        body.distance=1.0e30F;
    }
    const SoftBodyHit member = trace_soft_members(ray, soft_body, maximum_distance);
    float hook_distance = maximum_distance;
    float3 hook_normal{};
    const bool hook = arena == GalleryArena::fishing_tank &&
        soft_body.member_positions != nullptr &&
        soft_body.member_voxels_per_instance != 0U &&
        intersect_particle_sphere(ray,
            soft_body.member_positions[soft_body.member_voxels_per_instance - 1U],
            0.075F, maximum_distance, hook_distance, hook_normal);
    const bool wheel_overlay = arena == GalleryArena::water_wheel && box &&
        body.distance < distance + 0.012F;
    if (hook && hook_distance < body.distance &&
        hook_distance < member.distance && (!box || hook_distance < distance)) {
        distance = hook_distance;
        const float light = 0.30F + 0.70F * fmaxf(0.0F,
            dot(hook_normal, normalize(make_float3(-0.48F, 0.84F, 0.34F))));
        color = multiply(make_float3(1.0F, 0.68F, 0.08F), light);
        return true;
    }
    if (member.distance < maximum_distance &&
        member.distance < body.distance && (!box || member.distance < distance)) {
        distance = member.distance;
        color = shade_soft_member(ray, member);
        return true;
    }
    if (body.distance < maximum_distance && (!box || body.distance < distance ||
            wheel_overlay)) {
        distance = body.distance;
        const bool crust = soft_body.member_count != 0U;
        if (arena == GalleryArena::low_ceiling_box) {
            // Context 4 uses the deformable graph for physics but presents the
            // twenty fixtures as clean blue cylinders without a crust/member
            // material overlay.
            color = shade_colored_surface(ray, body,
                make_float3(0.035F,0.30F,0.92F), false, false);
        } else if (arena == GalleryArena::ground_box &&
                   soft_body.surface_triangle_split != 0U) {
            const bool sphere_surface =
                body.triangle < soft_body.surface_triangle_split;
            const bool goal_surface = !sphere_surface &&
                (soft_body.secondary_surface_triangle_split == 0U ||
                 body.triangle < soft_body.secondary_surface_triangle_split);
            const bool ground_surface = !sphere_surface && !goal_surface;
            color = shade_colored_surface(ray, body,
                sphere_surface ? make_float3(0.035F,0.30F,0.92F)
                               : make_float3(0.06F,0.64F,0.20F),
                goal_surface, goal_surface ? 1 : (ground_surface ? 2 : 0));
        } else if (arena == GalleryArena::cloth_basin) {
            color = shade_colored_surface(ray, body,
                make_float3(0.08F,0.58F,0.20F), false, false);
        } else if (arena == GalleryArena::rope_post) {
            const float3 glass_surface=shade_colored_surface(ray,body,
                make_float3(0.30F,0.82F,1.0F),false,false);
            const float facing=fabsf(dot(body.normal,ray.direction));
            const float fresnel=0.04F+0.96F*powf(1.0F-facing,5.0F);
            color=add(add(multiply(make_float3(0.045F,0.075F,0.105F),0.68F),
                multiply(glass_surface,0.22F)),
                multiply(make_float3(0.72F,0.94F,1.0F),0.45F*fresnel));
        } else if (arena == GalleryArena::rope_bridge) {
            const int columns=static_cast<int>(soft_body.rope_bridge_columns);
            const int rows=static_cast<int>(soft_body.rope_bridge_rows);
            const int column=max(0,min(columns-1,
                static_cast<int>(floorf(body.uv.x))));
            const int row=max(0,min(rows-1,
                static_cast<int>(floorf(body.uv.y))));
            const bool outside=column==0 || column==columns-1;
            const bool painted=device_ground_cloth_paint_pixels != nullptr &&
                device_ground_cloth_paint_pixels[
                    row*columns+column] != 0U;
            color=shade_colored_surface(ray,body,
                outside || painted ? make_float3(0.035F,0.30F,0.92F)
                                   : make_float3(0.86F,0.06F,0.045F),
                false,0);
        } else if (arena == GalleryArena::grass) {
            color=shade_colored_surface(ray,body,
                make_float3(0.08F,0.58F,0.16F),false,false);
        } else {
            color = shade_soft_body(ray, body,
                arena == GalleryArena::ground || arena == GalleryArena::grass ||
                    arena == GalleryArena::ground_box,
                crust, arena == GalleryArena::enclosed_box);
        }
        // Context 4 uses a cheap single-hit translucency cue: the polished
        // crust is alpha-composited with the nearest structural member behind
        // it. There are no reflection/refraction continuation rays.
        if (crust && arena != GalleryArena::low_ceiling_box &&
            arena != GalleryArena::ground_box &&
            member.distance < maximum_distance && member.distance > body.distance)
            color = add(multiply(color, 0.58F),
                multiply(shade_soft_member(ray, member), 0.42F));
        return true;
    }
    if (box) color = opaque_color;
    return box;
}

template <bool obstacle_course, GalleryArena arena = GalleryArena::none>
__device__ float3 environment(
    float3 origin, float3 direction, const OrientedBox& collider,
    SoftBodyRenderView soft_body)
{
    const Ray ray{origin, direction};
    float distance = 0.0F;
    float3 color{};
    if (trace_opaque_scene<obstacle_course, arena>(
            ray, collider, soft_body, 1.0e30F, distance, color)) return color;
    if constexpr (obstacle_course) return course_background(direction);
    return checker_box(origin, direction);
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

template <bool obstacle_course, GalleryArena arena = GalleryArena::none>
__device__ float3 shade_water(
    const Ray& primary,
    const Hit& entry,
    const float3* positions,
    const float3* vertex_normals,
    const std::uint32_t* corner_normal_indices,
    const uint3* triangles,
    const parallel_mater::HierarchyNode* nodes,
    const std::uint32_t* primitive_indices,
    std::uint32_t node_count,
    OrientedBox collider,
    SoftBodyRenderView soft_body)
{
    float3 entry_normal = entry.normal;
    if (dot(primary.direction, entry_normal) > 0.0F) entry_normal = multiply(entry_normal, -1.0F);
    const float cosine = fmaxf(0.0F, -dot(primary.direction, entry_normal));
    constexpr float base_reflectance = 0.02037F;
    const float fresnel = base_reflectance +
        (1.0F - base_reflectance) * powf(1.0F - cosine, 5.0F);
    const float3 reflected = normalize(subtract(
        primary.direction, multiply(entry_normal, 2.0F * dot(primary.direction, entry_normal))));
    const float3 entry_point = add(primary.origin, multiply(primary.direction, entry.distance));
    const float3 reflection_color = environment<obstacle_course, arena>(
        add(entry_point, multiply(reflected, 2.0e-4F)), reflected, collider, soft_body);

    float3 inside_direction{};
    if (!refract_direction(primary.direction, entry_normal, 1.0F / 1.333F, inside_direction)) {
        return reflection_color;
    }
    const Ray inside_ray{add(entry_point, multiply(inside_direction, 2.0e-4F)), inside_direction};
    const Hit exit = trace_closest(
        inside_ray,
        positions,
        vertex_normals,
        corner_normal_indices,
        triangles,
        nodes,
        primitive_indices,
        node_count);
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
    const float3 exit_point = add(inside_ray.origin, multiply(inside_direction, exit.distance));
    float path_length = exit.distance;
    float3 transmitted{};
    float opaque_distance = 0.0F;
    if (trace_opaque_scene<obstacle_course, arena>(inside_ray, collider,
            soft_body, exit.distance, opaque_distance, transmitted)) {
        path_length = opaque_distance;
    } else {
        transmitted = environment<obstacle_course, arena>(
            add(exit_point, multiply(outside_direction, 2.0e-4F)),
            outside_direction, collider, soft_body);
    }
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

__device__ float unit_hash(std::uint32_t value)
{
    value ^= value >> 16U;
    value *= 0x7feb352dU;
    value ^= value >> 15U;
    value *= 0x846ca68bU;
    value ^= value >> 16U;
    return static_cast<float>(value & 0x00ffffffU) * (1.0F / 16777215.0F);
}

__device__ float smooth_unit(float value)
{
    const float t = fminf(1.0F, fmaxf(0.0F, value));
    return t * t * (3.0F - 2.0F * t);
}

struct FieldHit {
    float distance{1.0e30F};
    float3 normal{};
    __device__ bool valid() const { return distance < 1.0e29F; }
};

struct CellValues { float v[8]; };

__device__ CellValues load_cell(
    FluidSurfaceView surface, const FluidSurfaceGrid& grid, int x, int y, int z)
{
    const std::size_t row = grid.dimensions.x;
    const std::size_t layer = row * grid.dimensions.y;
    const std::size_t base = static_cast<std::size_t>(z) * layer +
        static_cast<std::size_t>(y) * row + static_cast<std::size_t>(x);
    return {{surface.values[base], surface.values[base + 1U],
        surface.values[base + row], surface.values[base + row + 1U],
        surface.values[base + layer], surface.values[base + layer + 1U],
        surface.values[base + layer + row], surface.values[base + layer + row + 1U]}};
}

__device__ float cell_value(const CellValues& cell, float3 p)
{
    const float x0 = cell.v[0] + (cell.v[1] - cell.v[0]) * p.x;
    const float x1 = cell.v[2] + (cell.v[3] - cell.v[2]) * p.x;
    const float x2 = cell.v[4] + (cell.v[5] - cell.v[4]) * p.x;
    const float x3 = cell.v[6] + (cell.v[7] - cell.v[6]) * p.x;
    const float y0 = x0 + (x1 - x0) * p.y;
    const float y1 = x2 + (x3 - x2) * p.y;
    return y0 + (y1 - y0) * p.z;
}

// A trilinear cell restricted to a ray is cubic. Split at both derivative
// extrema, then bisect the first bracket, so equal-sign segment endpoints
// cannot hide two interior crossings.
__device__ bool cell_first_root(
    const CellValues& cell, float3 p0, float3 p1, float& root)
{
    float low = cell.v[0], high = cell.v[0];
    for (int i = 1; i < 8; ++i) {
        low = fminf(low, cell.v[i]);
        high = fmaxf(high, cell.v[i]);
    }
    if (!(low <= 0.0F && high >= 0.0F)) return false;
    const float3 delta = subtract(p1, p0);
    const float y0 = cell_value(cell, p0);
    const float y1 = cell_value(cell, add(p0, multiply(delta, 1.0F / 3.0F)));
    const float y2 = cell_value(cell, add(p0, multiply(delta, 2.0F / 3.0F)));
    const float y3 = cell_value(cell, p1);
    return surface_cubic_first_root(y0, y1, y2, y3, root);
}

__device__ float3 smooth_surface_normal(
    FluidSurfaceView surface, const FluidSurfaceGrid& grid, float3 point, float3 fallback)
{
    // The exact trilinear gradient changes abruptly at cell boundaries. A
    // centered difference spanning neighboring cells keeps the exact zero for
    // intersection while presenting one visually coherent normal field.
    const float3 step = multiply(grid.cell_size, 0.80F);
    const float3 gradient = make_float3(
        (surface_sample(surface, add(point, make_float3(step.x, 0.0F, 0.0F))) -
            surface_sample(surface, subtract(point, make_float3(step.x, 0.0F, 0.0F)))) /
            (2.0F * step.x),
        (surface_sample(surface, add(point, make_float3(0.0F, step.y, 0.0F))) -
            surface_sample(surface, subtract(point, make_float3(0.0F, step.y, 0.0F)))) /
            (2.0F * step.y),
        (surface_sample(surface, add(point, make_float3(0.0F, 0.0F, step.z))) -
            surface_sample(surface, subtract(point, make_float3(0.0F, 0.0F, step.z)))) /
            (2.0F * step.z));
    return isfinite(gradient.x) && isfinite(gradient.y) && isfinite(gradient.z) &&
            dot(gradient, gradient) > 1.0e-12F
        ? normalize(gradient) : fallback;
}

__device__ FieldHit trace_fluid_surface(const Ray& ray, FluidSurfaceView surface)
{
    FieldHit hit;
    if (surface.grid == nullptr || surface.values == nullptr) return hit;
    const FluidSurfaceGrid grid = *surface.grid;
    if (grid.dimensions.x < 2U || grid.dimensions.y < 2U || grid.dimensions.z < 2U ||
        !(grid.cell_size.x > 0.0F && grid.cell_size.y > 0.0F && grid.cell_size.z > 0.0F)) {
        return hit;
    }
    const float origin[3]{ray.origin.x, ray.origin.y, ray.origin.z};
    const float direction[3]{ray.direction.x, ray.direction.y, ray.direction.z};
    const float minimum[3]{grid.minimum.x, grid.minimum.y, grid.minimum.z};
    const float cell[3]{grid.cell_size.x, grid.cell_size.y, grid.cell_size.z};
    const unsigned dimensions[3]{grid.dimensions.x, grid.dimensions.y, grid.dimensions.z};
    float enter = 0.0F, leave = 1.0e30F;
    for (int axis = 0; axis < 3; ++axis) {
        const float maximum = minimum[axis] + cell[axis] * (dimensions[axis] - 1U);
        if (fabsf(direction[axis]) < 1.0e-12F) {
            if (origin[axis] < minimum[axis] || origin[axis] > maximum) return hit;
            continue;
        }
        float first = (minimum[axis] - origin[axis]) / direction[axis];
        float second = (maximum - origin[axis]) / direction[axis];
        if (first > second) {
            const float temporary = first; first = second; second = temporary;
        }
        enter = fmaxf(enter, first);
        leave = fminf(leave, second);
        if (enter > leave) return hit;
    }
    float distance = fmaxf(1.0e-5F, enter) + 1.0e-6F;
    if (distance >= leave) return hit;
    const float3 start = add(ray.origin, multiply(ray.direction, distance));
    const float point[3]{start.x, start.y, start.z};
    int index[3], step[3];
    float next[3], stride[3];
    for (int axis = 0; axis < 3; ++axis) {
        const float q = (point[axis] - minimum[axis]) / cell[axis];
        int candidate = static_cast<int>(floorf(q));
        candidate = candidate < 0 ? 0 : candidate;
        const int last = static_cast<int>(dimensions[axis]) - 2;
        index[axis] = candidate > last ? last : candidate;
        step[axis] = direction[axis] > 0.0F ? 1 : direction[axis] < 0.0F ? -1 : 0;
        if (step[axis] == 0) {
            next[axis] = stride[axis] = 1.0e30F;
        } else {
            const int boundary = index[axis] + (step[axis] > 0 ? 1 : 0);
            next[axis] = (minimum[axis] + boundary * cell[axis] - origin[axis]) /
                direction[axis];
            stride[axis] = cell[axis] / fabsf(direction[axis]);
        }
    }
    const unsigned maximum_cells = dimensions[0] + dimensions[1] + dimensions[2] + 3U;
    for (unsigned visited = 0; visited < maximum_cells && distance < leave; ++visited) {
        const float end = fminf(leave, fminf(next[0], fminf(next[1], next[2])));
        if (end > distance + 1.0e-7F) {
            const CellValues values = load_cell(surface, grid, index[0], index[1], index[2]);
            const float3 world0 = add(ray.origin, multiply(ray.direction, distance));
            const float3 world1 = add(ray.origin, multiply(ray.direction, end));
            const float3 cell_min = make_float3(minimum[0] + index[0] * cell[0],
                minimum[1] + index[1] * cell[1], minimum[2] + index[2] * cell[2]);
            const float3 local0 = make_float3(
                fminf(1.0F, fmaxf(0.0F, (world0.x - cell_min.x) / cell[0])),
                fminf(1.0F, fmaxf(0.0F, (world0.y - cell_min.y) / cell[1])),
                fminf(1.0F, fmaxf(0.0F, (world0.z - cell_min.z) / cell[2])));
            const float3 local1 = make_float3(
                fminf(1.0F, fmaxf(0.0F, (world1.x - cell_min.x) / cell[0])),
                fminf(1.0F, fmaxf(0.0F, (world1.y - cell_min.y) / cell[1])),
                fminf(1.0F, fmaxf(0.0F, (world1.z - cell_min.z) / cell[2])));
            float root = 0.0F;
            if (cell_first_root(values, local0, local1, root)) {
                hit.distance = distance + root * (end - distance);
                const float3 position = add(ray.origin, multiply(ray.direction, hit.distance));
                const float3 gradient = surface_gradient(surface, position);
                const float3 exact_normal = dot(gradient, gradient) > 1.0e-12F
                    ? normalize(gradient) : multiply(ray.direction, -1.0F);
                hit.normal = smooth_surface_normal(surface, grid, position, exact_normal);
                return hit;
            }
        }
        if (end >= leave) break;
        bool valid = true;
        for (int axis = 0; axis < 3; ++axis) {
            if (next[axis] > end) continue;
            index[axis] += step[axis];
            next[axis] += stride[axis];
            valid = valid && index[axis] >= 0 &&
                index[axis] < static_cast<int>(dimensions[axis]) - 1;
        }
        if (!valid) break;
        distance = end;
    }
    return hit;
}

// One coherent interface from the particle-derived trilinear field. This is an
// interpolated scalar surface, not an exact density isosurface or distance field.
template <bool obstacle_course, GalleryArena arena = GalleryArena::none>
__device__ float3 shade_continuous_water(
    const Ray& ray, const FieldHit& hit, const OrientedBox& collider,
    SoftBodyRenderView soft_body)
{
    float3 normal = hit.normal;
    if (dot(ray.direction, normal) > 0.0F) normal = multiply(normal, -1.0F);
    const float facing = fmaxf(0.0F, -dot(ray.direction, normal));
    const float fresnel = 0.02037F + 0.97963F * powf(1.0F - facing, 5.0F);
    const float3 point = add(ray.origin, multiply(ray.direction, hit.distance));
    const float3 reflected = normalize(subtract(
        ray.direction, multiply(normal, 2.0F * dot(ray.direction, normal))));
    const float3 reflection = environment<obstacle_course, arena>(
        add(point, multiply(reflected, 2.0e-4F)), reflected, collider, soft_body);
    float3 underwater_direction{};
    if (!refract_direction(ray.direction, normal, 1.0F / 1.333F, underwater_direction)) {
        return reflection;
    }
    const Ray underwater{add(point, multiply(underwater_direction, 2.0e-4F)),
        underwater_direction};
    float path = 0.55F;
    float3 transmitted{};
    float opaque_distance = 0.0F;
    if (trace_opaque_scene<obstacle_course, arena>(
            underwater, collider, soft_body, 1.5F, opaque_distance, transmitted)) {
        path = fminf(1.5F, fmaxf(0.03F, opaque_distance));
    } else if constexpr (obstacle_course) {
        transmitted = course_background(underwater_direction);
    } else {
        transmitted = checker_box(underwater.origin, underwater_direction);
    }
    const float3 absorption = make_float3(
        expf(-1.35F * path), expf(-0.34F * path), expf(-0.16F * path));
    const float haze = 0.12F + 0.34F * (1.0F - absorption.x);
    const float3 tint = make_float3(0.012F, 0.40F, 0.46F);
    const float3 transmission = add(multiply(make_float3(transmitted.x * absorption.x,
        transmitted.y * absorption.y, transmitted.z * absorption.z), 1.0F - haze),
        multiply(tint, haze));
    float3 color = add(multiply(reflection, fresnel), multiply(transmission, 1.0F - fresnel));
    const float glint = powf(fmaxf(0.0F,
        dot(reflected, normalize(make_float3(-0.48F, 0.84F, 0.34F)))), 80.0F);
    return add(color, multiply(make_float3(0.62F, 0.91F, 1.0F),
        0.38F * glint + 0.055F * powf(1.0F - facing, 2.0F)));
}

struct FoamHit {
    float distance{1.0e30F};
    float3 normal{};
    std::uint32_t particle{UINT32_MAX};
    std::uint32_t lobe{};
    float fade{};
    float radius{};
    __device__ bool valid() const { return particle != UINT32_MAX; }
};

__device__ void intersect_foam_lobe(
    const Ray& ray, float3 center, float radius, float3 cap_normal,
    std::uint32_t particle, std::uint32_t lobe, float fade,
    float cluster_radius, FoamHit& hit)
{
    const float3 offset = subtract(ray.origin, center);
    const float b = dot(offset, ray.direction);
    const float c = dot(offset, offset) - radius * radius;
    const float discriminant = b * b - c;
    if (discriminant < 0.0F) return;
    const float root = sqrtf(discriminant);
    const float candidates[2]{-b - root, -b + root};
    for (int candidate = 0; candidate < 2; ++candidate) {
        const float distance = candidates[candidate];
        if (distance <= 1.0e-5F || distance >= hit.distance) continue;
        const float3 normal = normalize(subtract(
            add(ray.origin, multiply(ray.direction, distance)), center));
        if (dot(normal, cap_normal) < -0.18F) continue;
        hit = {distance, normal, particle, lobe, fade, cluster_radius};
        break;
    }
}

__device__ void intersect_foam_particle(
    const Ray& ray, const FoamParticle& foam, std::uint32_t particle, FoamHit& hit)
{
    const float age = foam.position_age.w;
    const float life = foam.velocity_life.w;
    const float base_radius = foam.normal_radius.w;
    if (!(age >= 0.0F && life > 0.0F && age < life && base_radius > 0.0F) ||
        !isfinite(foam.position_age.x) || !isfinite(foam.position_age.y) ||
        !isfinite(foam.position_age.z) || !isfinite(life) ||
        !isfinite(foam.normal_radius.x) || !isfinite(foam.normal_radius.y) ||
        !isfinite(foam.normal_radius.z) || !isfinite(base_radius)) return;
    const float normalized_age = age / life;
    const float fade = fminf(1.0F, fmaxf(0.0F, 1.0F - normalized_age));
    const float grow = 0.58F + 0.42F * smooth_unit(normalized_age / 0.12F);
    // A tracer is a local animated foam patch, not a single white particle.
    // Patches spread along the live tangent plane and individual cells pop at
    // different phases. Their anchor and orientation still come from the
    // current fluid simulation.
    const float spread = 0.64F + 0.56F * smooth_unit(normalized_age / 0.55F);
    const float3 anchor = make_float3(
        foam.position_age.x, foam.position_age.y, foam.position_age.z);
    const float3 supplied_normal = make_float3(
        foam.normal_radius.x, foam.normal_radius.y, foam.normal_radius.z);
    const float normal_length_squared = dot(supplied_normal, supplied_normal);
    const float3 outward = normal_length_squared > 1.0e-12F
        ? multiply(supplied_normal, rsqrtf(normal_length_squared))
        : make_float3(0.0F, 1.0F, 0.0F);
    const float3 reference = fabsf(outward.y) < 0.85F
        ? make_float3(0.0F, 1.0F, 0.0F) : make_float3(1.0F, 0.0F, 0.0F);
    const float3 tangent = normalize(cross(reference, outward));
    const float3 bitangent = cross(outward, tangent);
    constexpr std::uint32_t lobe_count = 11U;
    constexpr float cluster_extent = 2.60F;
    const float phase = 2.0F * pi * unit_hash(particle ^ 0x68bc21ebU);
    for (std::uint32_t lobe = 0U; lobe < lobe_count; ++lobe) {
        const std::uint32_t seed = particle * 0x9e3779b9U + lobe * 0x85ebca6bU;
        const float first = unit_hash(seed ^ 0xc2b2ae35U);
        const float second = unit_hash(seed ^ 0x27d4eb2fU);
        const float third = unit_hash(seed ^ 0x165667b1U);
        const float fourth = unit_hash(seed ^ 0xd3a2646cU);
        const float birth = lobe == 0U ? 0.0F : 0.02F + 0.10F * fourth;
        const float death = 0.60F + 0.32F * third;
        const float appearing = smooth_unit((normalized_age - birth) / 0.09F);
        const float disappearing = 1.0F - smooth_unit((normalized_age - death) / 0.10F);
        const float activity = appearing * disappearing;
        if (!(activity > 1.0e-4F)) continue;
        const float radial = lobe == 0U ? 0.0F :
            (0.08F + 1.35F * sqrtf(first)) * base_radius * spread;
        const float angle = phase + 2.0F * pi * second +
            0.16F * sinf(2.0F * pi * (normalized_age + third));
        float3 center = add(anchor, add(multiply(tangent, radial * cosf(angle)),
            multiply(bitangent, radial * sinf(angle))));
        center = add(center, multiply(outward, (0.02F + 0.16F * fourth) * base_radius));
        float radius = (lobe == 0U ? 0.56F : 0.22F + 0.50F * third) * base_radius;
        radius *= grow * (0.72F + 0.28F * activity);
        intersect_foam_lobe(ray, center, radius, outward, particle, lobe,
            fade * activity, cluster_extent * base_radius, hit);
    }
}

__device__ FoamHit trace_foam(const Ray& ray, FluidVisualView visuals)
{
    FoamHit hit;
    if (!visuals.show_foam || visuals.foam_particles == nullptr ||
        visuals.foam_nodes == nullptr || visuals.foam_indices == nullptr ||
        visuals.foam_capacity == 0U || visuals.foam_node_count == 0U) return hit;
    std::uint32_t stack[128];
    int stack_size = 1;
    stack[0] = 0U;
    while (stack_size > 0) {
        const std::uint32_t node_index = stack[--stack_size];
        if (node_index >= visuals.foam_node_count) continue;
        const parallel_mater::HierarchyNode node = visuals.foam_nodes[node_index];
        float node_near = 0.0F;
        const float3 extent = subtract(node.bounds_max, node.bounds_min);
        // Each source bound is a sphere AABB. Expanding by 0.82 of the
        // smallest node extent conservatively covers the 2.60-radius patch.
        const float padding = 0.82F * fmaxf(0.0F,
            fminf(extent.x, fminf(extent.y, extent.z)));
        if (!intersect_bounds(ray, node, hit.distance, node_near, padding)) continue;
        if (node.is_leaf()) {
            for (std::uint32_t item = 0U; item < node.primitive_count; ++item) {
                const std::uint32_t particle =
                    visuals.foam_indices[node.first_primitive + item];
                if (particle >= visuals.foam_capacity) continue;
                intersect_foam_particle(ray, visuals.foam_particles[particle], particle, hit);
            }
            continue;
        }
        for (std::uint32_t child = 0U;
             child < node.child_count && stack_size < 128; ++child) {
            stack[stack_size++] = node.first_child + child;
        }
    }
    return hit;
}

__device__ float3 shade_foam(const Ray& ray, const FoamHit& foam, float3 underneath)
{
    const float3 light = normalize(make_float3(-0.48F, 0.84F, 0.34F));
    const float facing = fmaxf(0.0F, -dot(ray.direction, foam.normal));
    const float rim = powf(1.0F - facing, 2.6F);
    const float3 reflected = subtract(
        ray.direction, multiply(foam.normal, 2.0F * dot(ray.direction, foam.normal)));
    const float highlight = powf(fmaxf(0.0F, dot(normalize(reflected), light)), 34.0F);
    const float variation = unit_hash(
        foam.particle * 0x9e3779b9U + foam.lobe * 0x85ebca6bU);
    const float3 film = make_float3(
        0.80F + 0.16F * variation,
        0.90F + 0.08F * (1.0F - variation),
        0.99F);
    const float age_fade = foam.fade * foam.fade * (3.0F - 2.0F * foam.fade);
    const float opacity = age_fade * fminf(0.96F,
        0.055F + 0.82F * rim + 0.26F * highlight);
    return add(multiply(underneath, 1.0F - opacity), multiply(film, opacity));
}

__device__ float3 shade_debug_particle(const Ray& ray, const Hit& particle)
{
    const float diffuse = 0.25F + 0.75F * fmaxf(
        0.0F, dot(particle.normal, normalize(make_float3(-0.45F, 0.82F, 0.35F))));
    const float facing = fmaxf(0.0F, -dot(ray.direction, particle.normal));
    return add(multiply(make_float3(0.04F, 0.64F, 0.88F), diffuse),
        multiply(make_float3(0.50F, 0.92F, 1.0F), 0.35F * powf(1.0F - facing, 3.0F)));
}

template <bool obstacle_course, GalleryArena arena = GalleryArena::none>
__global__ void render_fluid_kernel(
    uchar4* pixels, std::uint32_t width, std::uint32_t height, Camera camera,
    const float3* positions, const float3* vertex_normals,
    const std::uint32_t* corner_normal_indices, const uint3* triangles,
    const parallel_mater::HierarchyNode* nodes, const std::uint32_t* primitive_indices,
    std::uint32_t node_count, OrientedBox collider,
    const float3* particle_positions, float particle_radius,
    const parallel_mater::HierarchyNode* particle_nodes,
    const std::uint32_t* particle_indices, std::uint32_t particle_node_count,
    FluidVisualView visuals,
    SoftBodyRenderView soft_body)
{
    const std::uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const Ray ray = camera_ray(camera, x, y, width, height);
    const bool wire_display = visuals.display == FluidDisplay::Wireframe;
    const Hit skin = (wire_display || visuals.display == FluidDisplay::Particles ||
        visuals.display == FluidDisplay::Billboards)
        ? Hit{1.0e30F, UINT32_MAX, {}}
        : trace_closest(ray, positions, vertex_normals, corner_normal_indices,
            triangles, nodes, primitive_indices, node_count);
    const bool skin_valid = skin.triangle != UINT32_MAX;
    const bool surface_display = visuals.display == FluidDisplay::Surface;
    // Wireframe is one combined diagnostic context: analytic water particles
    // remain ray traced while the water and soft-body meshes are overlaid by
    // the host OpenGL pass.
    const bool particle_display = visuals.show_particle_layer &&
        (visuals.display == FluidDisplay::Particles || wire_display);
    const FieldHit fluid = surface_display
        ? trace_fluid_surface(ray, visuals.surface) : FieldHit{};
    const Hit particle = particle_display
        ? trace_particle_spheres(ray, particle_positions, particle_radius, particle_nodes,
            particle_indices, particle_node_count)
        : Hit{1.0e30F, UINT32_MAX, {}};
    const bool particle_valid = particle.triangle != UINT32_MAX;
    const bool diagnostic_foam = wire_display && visuals.show_foam;
    const FoamHit foam = (surface_display || diagnostic_foam)
        ? trace_foam(ray, visuals) : FoamHit{};

    float obstacle_distance = 1.0e30F;
    float3 obstacle_color{};
    const bool obstacle_valid = trace_opaque_scene<obstacle_course, arena>(
        ray, collider, soft_body, 1.0e30F, obstacle_distance, obstacle_color);

    const float display_distance = surface_display ? fluid.distance : particle.distance;
    const float nearest_fluid = fminf(skin.distance, display_distance);
    const bool foam_visible = foam.valid() &&
        (!obstacle_valid || foam.distance < obstacle_distance) &&
        (diagnostic_foam || foam.distance <= fluid.distance + 1.5F * foam.radius);
    float3 color{};
    if (obstacle_valid && obstacle_distance < nearest_fluid) {
        color = obstacle_color;
    } else if (visuals.display == FluidDisplay::Wireframe) {
        color = particle_valid
            ? shade_debug_particle(ray, particle)
            : environment<obstacle_course, arena>(ray.origin, ray.direction, collider, soft_body);
    } else if (surface_display && fluid.valid() &&
               (!obstacle_valid || obstacle_distance >= fluid.distance)) {
        const float3 fluid_color = shade_continuous_water<obstacle_course, arena>(
            ray, fluid, collider, soft_body);
        if (skin_valid && skin.distance < fluid.distance) {
            const float3 skin_color = shade_water<obstacle_course, arena>(ray, skin,
                positions, vertex_normals, corner_normal_indices, triangles,
                nodes, primitive_indices, node_count, collider, soft_body);
            color = add(multiply(skin_color, 0.20F), multiply(fluid_color, 0.80F));
        } else {
            color = fluid_color;
        }
    } else if (particle_display && particle_valid &&
               (!obstacle_valid || obstacle_distance >= particle.distance)) {
        const float3 particle_color = shade_debug_particle(ray, particle);
        if (skin_valid && skin.distance < particle.distance) {
            const float3 skin_color = shade_water<obstacle_course, arena>(
                ray, skin, positions, vertex_normals, corner_normal_indices,
                triangles, nodes, primitive_indices, node_count, collider, soft_body);
            color = add(multiply(skin_color, 0.28F), multiply(particle_color, 0.72F));
        } else {
            color = particle_color;
        }
    } else if (skin_valid) {
        // A course object between the transparent skin and its particles blocks
        // the particle layer; the refractive skin still remains visible.
        color = shade_water<obstacle_course, arena>(ray, skin, positions, vertex_normals,
            corner_normal_indices, triangles, nodes, primitive_indices, node_count,
            collider, soft_body);
    } else {
        color = obstacle_valid ? obstacle_color
            : environment<obstacle_course, arena>(ray.origin, ray.direction, collider, soft_body);
    }
    if constexpr (arena == GalleryArena::cloth_basin) {
        float goal_distance{};
        float3 goal_normal{};
        const float foreground=fminf(nearest_fluid,
            obstacle_valid ? obstacle_distance : 1.0e30F);
        if (intersect_box(ray,cloth_goal_center,cloth_goal_half_extents,
                foreground+1.0e-4F,goal_distance,goal_normal)) {
            const float edge_light=0.55F+0.45F*(1.0F-fabsf(
                dot(goal_normal,ray.direction)));
            const float3 goal_color=multiply(make_float3(0.08F,1.0F,0.30F),
                edge_light);
            color=add(multiply(color,0.66F),multiply(goal_color,0.34F));
        }
    }
    // The cap is a thin film over the same composed scene, not another copy
    // of the water/skin/obstacle shading paths.
    if (foam_visible) {
        color = diagnostic_foam
            ? add(multiply(make_float3(0.05F, 1.0F, 0.18F), 0.90F),
                multiply(color, 0.10F))
            : shade_foam(ray, foam, color);
    }
    pixels[static_cast<std::size_t>(y) * width + x] =
        make_uchar4(to_byte(color.x), to_byte(color.y), to_byte(color.z), 255);
}

} // namespace
RayTracer::RayTracer()
{
    check(cudaEventCreate(&render_begin_), "create render event");
    check(cudaEventCreate(&render_end_), "create render event");
    check(cudaMalloc(&bowl_paint_pixels_,
        bowl_paint_pixel_count * sizeof(std::uint32_t)), "allocate bowl paint pixels");
    check(cudaMalloc(&bowl_painted_count_, sizeof(std::uint32_t)),
        "allocate bowl paint count");
    check(cudaMemset(bowl_paint_pixels_, 0,
        bowl_paint_pixel_count * sizeof(std::uint32_t)), "clear bowl paint pixels");
    check(cudaMemset(bowl_painted_count_, 0, sizeof(std::uint32_t)),
        "clear bowl paint count");
    check(cudaMemcpyToSymbol(device_bowl_paint_pixels, &bowl_paint_pixels_,
        sizeof(bowl_paint_pixels_)), "bind bowl paint pixels");
    check(cudaMalloc(&sphere_paint_pixels_,
        sphere_paint_pixel_count * sizeof(std::uint32_t)),
        "allocate sphere paint pixels");
    check(cudaMalloc(&sphere_painted_count_, sizeof(std::uint32_t)),
        "allocate sphere paint count");
    check(cudaMemset(sphere_paint_pixels_, 0,
        sphere_paint_pixel_count * sizeof(std::uint32_t)),
        "clear sphere paint pixels");
    check(cudaMemset(sphere_painted_count_, 0, sizeof(std::uint32_t)),
        "clear sphere paint count");
    check(cudaMemcpyToSymbol(device_sphere_paint_pixels, &sphere_paint_pixels_,
        sizeof(sphere_paint_pixels_)), "bind sphere paint pixels");
    check(cudaMalloc(&cloth_paint_pixels_,
        cloth_paint_pixel_count * sizeof(std::uint32_t)),
        "allocate cloth paint pixels");
    check(cudaMemset(cloth_paint_pixels_, 0,
        cloth_paint_pixel_count * sizeof(std::uint32_t)),
        "clear cloth paint pixels");
    check(cudaMemcpyToSymbol(device_cloth_paint_pixels, &cloth_paint_pixels_,
        sizeof(cloth_paint_pixels_)), "bind cloth paint pixels");
    check(cudaMalloc(&ground_cloth_paint_pixels_,
        cloth_paint_pixel_count*sizeof(std::uint32_t)),
        "allocate ground cloth paint pixels");
    check(cudaMemset(ground_cloth_paint_pixels_,0,
        cloth_paint_pixel_count*sizeof(std::uint32_t)),
        "clear ground cloth paint pixels");
    check(cudaMemcpyToSymbol(device_ground_cloth_paint_pixels,
        &ground_cloth_paint_pixels_,sizeof(ground_cloth_paint_pixels_)),
        "bind ground cloth paint pixels");
}

RayTracer::~RayTracer()
{
    cudaEventDestroy(render_end_);
    cudaEventDestroy(render_begin_);
    cudaFree(ground_cloth_paint_pixels_);
    cudaFree(cloth_paint_pixels_);
    cudaFree(sphere_painted_count_);
    cudaFree(sphere_paint_pixels_);
    cudaFree(bowl_painted_count_);
    cudaFree(bowl_paint_pixels_);
    cudaFreeHost(host_pixels_);
    cudaFree(device_pixels_);
}

__global__ void update_bowl_paint_kernel(const float3* positions,
    std::uint32_t count, float particle_radius, std::uint32_t* pixels,
    std::uint32_t* painted_count)
{
    const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle >= count) return;
    const float3 relative = subtract(positions[particle], bowl_center);
    const float radius = length(relative);
    const float contact_band = particle_radius + 0.025F;
    if (!(radius > 0.0F) ||
        fabsf(radius - bowl_inner_radius) > contact_band ||
        relative.y > 0.0F) return;
    const float3 direction = multiply(relative, 1.0F / radius);
    constexpr float pi = 3.14159265358979323846F;
    const float longitude = (atan2f(direction.z, direction.x) + pi) / (2.0F * pi);
    const float latitude = acosf(fminf(1.0F,
        fmaxf(0.0F, -direction.y))) / (0.5F * pi);
    const std::uint32_t x = min(bowl_paint_width - 1U,
        static_cast<std::uint32_t>(longitude * bowl_paint_width));
    const std::uint32_t y = min(bowl_paint_height - 1U,
        static_cast<std::uint32_t>(latitude * bowl_paint_height));
    if (atomicCAS(pixels + y * bowl_paint_width + x, 0U, 1U) == 0U)
        atomicAdd(painted_count, 1U);
}

__global__ void initialize_bowl_peg_footprints_kernel(
    std::uint32_t* pixels, std::uint32_t* painted_count)
{
    const std::uint32_t texel = blockIdx.x*blockDim.x+threadIdx.x;
    if (texel >= bowl_paint_pixel_count) return;
    const std::uint32_t x = texel%bowl_paint_width;
    const std::uint32_t y = texel/bowl_paint_width;
    const float longitude = (static_cast<float>(x)+0.5F)/bowl_paint_width;
    const float latitude = (static_cast<float>(y)+0.5F)/bowl_paint_height;
    const float azimuth = longitude*(2.0F*pi)-pi;
    const float polar = latitude*(0.5F*pi);
    const float horizontal = sinf(polar);
    const float3 point = add(bowl_center,multiply(make_float3(
        horizontal*cosf(azimuth),-cosf(polar),horizontal*sinf(azimuth)),
        bowl_inner_radius));
    for (std::uint32_t peg=0U; peg<bowl_peg_count; ++peg) {
        const float3 center = bowl_peg(peg);
        const float dx = point.x-center.x;
        const float dz = point.z-center.z;
        if (dx*dx+dz*dz <=
                (bowl_peg_radius+0.018F)*(bowl_peg_radius+0.018F)) {
            pixels[texel]=1U;
            atomicAdd(painted_count,1U);
            return;
        }
    }
}

float RayTracer::update_bowl_paint(const float3* particle_positions,
    std::uint32_t particle_count, float particle_radius, bool reset,
    cudaStream_t stream)
{
    if (reset) {
        check(cudaMemsetAsync(bowl_paint_pixels_, 0,
            bowl_paint_pixel_count * sizeof(std::uint32_t), stream),
            "reset bowl paint pixels");
        check(cudaMemsetAsync(bowl_painted_count_, 0,
            sizeof(std::uint32_t), stream), "reset bowl paint count");
        initialize_bowl_peg_footprints_kernel<<<
            (bowl_paint_pixel_count+255U)/256U,256U,0,stream>>>(
                bowl_paint_pixels_,bowl_painted_count_);
        check(cudaGetLastError(), "initialize bowl peg paint footprints");
    }
    if (particle_positions != nullptr && particle_count != 0U) {
        update_bowl_paint_kernel<<<(particle_count + 255U) / 256U, 256U, 0, stream>>>(
            particle_positions, particle_count, particle_radius,
            bowl_paint_pixels_, bowl_painted_count_);
        check(cudaGetLastError(), "update bowl paint pixels");
    }
    std::uint32_t painted{};
    check(cudaMemcpyAsync(&painted, bowl_painted_count_, sizeof(painted),
        cudaMemcpyDeviceToHost, stream), "download bowl paint count");
    check(cudaStreamSynchronize(stream), "finish bowl paint update");
    return static_cast<float>(painted) /
        static_cast<float>(bowl_paint_pixel_count);
}

__global__ void update_sphere_paint_kernel(const float3* positions,
    std::uint32_t count, float contact_radius, RigidSphereState sphere,
    std::uint32_t* pixels, std::uint32_t* painted_count)
{
    const std::uint32_t item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= count) return;
    const float3 delta = subtract(positions[item], sphere.center);
    const float distance = length(delta);
    const float contact_distance = sphere.radius + contact_radius + 0.012F;
    if (!(distance > 1.0e-8F) || distance > contact_distance) return;
    const float3 direction = inverse_rotate_quaternion(sphere.orientation,
        multiply(delta, 1.0F / distance));
    const float longitude = (atan2f(direction.z, direction.x) + pi) / (2.0F*pi);
    const float latitude = acosf(fminf(1.0F, fmaxf(-1.0F, direction.y))) / pi;
    const std::uint32_t x = min(sphere_paint_width-1U,
        static_cast<std::uint32_t>(longitude*sphere_paint_width));
    const std::uint32_t y = min(sphere_paint_height-1U,
        static_cast<std::uint32_t>(latitude*sphere_paint_height));
    if (atomicCAS(pixels + y*sphere_paint_width + x, 0U, 1U) == 0U)
        atomicAdd(painted_count, 1U);
}

float RayTracer::update_sphere_paint(const float3* contact_positions,
    std::uint32_t contact_count, float contact_radius, RigidSphereState sphere,
    bool reset, cudaStream_t stream)
{
    if (reset) {
        check(cudaMemsetAsync(sphere_paint_pixels_, 0,
            sphere_paint_pixel_count*sizeof(std::uint32_t), stream),
            "reset sphere paint pixels");
        check(cudaMemsetAsync(sphere_painted_count_, 0,
            sizeof(std::uint32_t), stream), "reset sphere paint count");
    }
    if (contact_positions != nullptr && contact_count != 0U) {
        update_sphere_paint_kernel<<<(contact_count+255U)/256U,256U,0,stream>>>(
            contact_positions, contact_count, contact_radius, sphere,
            sphere_paint_pixels_, sphere_painted_count_);
        check(cudaGetLastError(), "update sphere paint pixels");
    }
    std::uint32_t painted{};
    check(cudaMemcpyAsync(&painted, sphere_painted_count_, sizeof(painted),
        cudaMemcpyDeviceToHost, stream), "download sphere paint count");
    check(cudaStreamSynchronize(stream), "finish sphere paint update");
    return static_cast<float>(painted) /
        static_cast<float>(sphere_paint_pixel_count);
}

__global__ void update_goal_cloth_paint_kernel(const float3* positions,
    std::uint32_t count, float source_radius, std::uint32_t* goal_pixels,
    std::uint32_t* ground_pixels)
{
    const std::uint32_t item = blockIdx.x*blockDim.x+threadIdx.x;
    if (item >= count) return;
    const float3 point = positions[item];
    constexpr float half_width = 0.5F*23.0F*0.075F;
    constexpr float bottom = course_floor_y;
    constexpr float top = course_floor_y+23.0F*0.075F;
    constexpr float plane_z = cloth_soft_body_goal_z;
    if (fabsf(point.z-plane_z) <= source_radius+0.02F &&
        point.x >= -half_width-source_radius && point.x <= half_width+source_radius &&
        point.y >= bottom-source_radius && point.y <= top+source_radius) {
        const float u = fminf(0.999999F, fmaxf(0.0F,
            (point.x+half_width)/(2.0F*half_width)));
        // make_cloth_grid assigns v=0 at its bottom row and v=1 at its top.
        // Matching that convention prevents underside contacts from appearing
        // mirrored onto the opposite vertical part of the goal sheet.
        const float v = fminf(0.999999F, fmaxf(0.0F,
            (point.y-bottom)/(top-bottom)));
        const int center_x=static_cast<int>(u*cloth_paint_width);
        const int center_y=static_cast<int>(v*cloth_paint_height);
        // A soft sphere is a finite contact patch, not a point sample. Splat a
        // conservative footprint so rolling across either face paints the
        // complete touched area instead of a sparse trail of single texels.
        const int radius_x=max(1,static_cast<int>(ceilf(
            (source_radius+0.055F)*cloth_paint_width/(2.0F*half_width))));
        const int radius_y=max(1,static_cast<int>(ceilf(
            (source_radius+0.055F)*cloth_paint_height/(top-bottom))));
        for (int oy=-radius_y;oy<=radius_y;++oy) {
            const int py=max(0,min(static_cast<int>(cloth_paint_height)-1,center_y+oy));
            for (int ox=-radius_x;ox<=radius_x;++ox) {
                if (ox*ox*radius_y*radius_y+oy*oy*radius_x*radius_x>
                    radius_x*radius_x*radius_y*radius_y) continue;
                const int px=max(0,min(static_cast<int>(cloth_paint_width)-1,center_x+ox));
                goal_pixels[py*cloth_paint_width+px]=1U;
            }
        }
    }
    constexpr float ground_half = 0.5F*39.0F*0.075F;
    constexpr float ground_center_z = -0.55F;
    constexpr float ground_y = course_floor_y+0.025F;
    if (fabsf(point.y-ground_y) <= source_radius+0.02F &&
        fabsf(point.x) <= ground_half+source_radius &&
        fabsf(point.z-ground_center_z) <= ground_half+source_radius) {
        const float u = fminf(0.999999F,fmaxf(0.0F,
            (point.x+ground_half)/(2.0F*ground_half)));
        // The horizontal conversion maps the cloth's top row to minimum z,
        // so its texture-v direction is reversed in world z.
        const float v = fminf(0.999999F,fmaxf(0.0F,
            (ground_center_z+ground_half-point.z)/(2.0F*ground_half)));
        ground_pixels[static_cast<std::uint32_t>(v*cloth_paint_height)*cloth_paint_width+
            static_cast<std::uint32_t>(u*cloth_paint_width)] = 1U;
    }
}

void RayTracer::update_goal_cloth_paint(const float3* source_positions,
    std::uint32_t source_count, float source_radius, bool reset,
    cudaStream_t stream)
{
    if (reset) {
        check(cudaMemsetAsync(cloth_paint_pixels_, 0,
            cloth_paint_pixel_count*sizeof(std::uint32_t), stream),
            "reset goal cloth paint pixels");
        check(cudaMemsetAsync(ground_cloth_paint_pixels_, 0,
            cloth_paint_pixel_count*sizeof(std::uint32_t), stream),
            "reset ground cloth paint pixels");
    }
    if (source_positions != nullptr && source_count != 0U) {
        update_goal_cloth_paint_kernel<<<(source_count+255U)/256U,256U,0,stream>>>(
            source_positions, source_count, source_radius, cloth_paint_pixels_,
            ground_cloth_paint_pixels_);
        check(cudaGetLastError(), "update goal cloth paint pixels");
    }
}

__global__ void update_rope_bridge_paint_kernel(
    const float3* nodes, std::uint32_t node_count, RigidSphereState sphere,
    std::uint32_t columns, std::uint32_t rows, std::uint32_t* pixels)
{
    const std::uint32_t tile=blockIdx.x*blockDim.x+threadIdx.x;
    const std::uint32_t tile_count=columns*rows;
    const std::uint32_t nodes_per_tile=tile_count==0U ? 0U : node_count/tile_count;
    if (tile>=tile_count || nodes_per_tile<4U ||
        nodes_per_tile*tile+nodes_per_tile-1U>=node_count) return;
    float minimum_x=1.0e30F,maximum_x=-1.0e30F;
    float minimum_z=1.0e30F,maximum_z=-1.0e30F,average_y=0.0F;
    for (std::uint32_t corner=0U;corner<nodes_per_tile;++corner) {
        const float3 point=nodes[nodes_per_tile*tile+corner];
        minimum_x=fminf(minimum_x,point.x); maximum_x=fmaxf(maximum_x,point.x);
        minimum_z=fminf(minimum_z,point.z); maximum_z=fmaxf(maximum_z,point.z);
        average_y+=point.y/static_cast<float>(nodes_per_tile);
    }
    const float closest_x=fminf(maximum_x,fmaxf(minimum_x,sphere.center.x));
    const float closest_z=fminf(maximum_z,fmaxf(minimum_z,sphere.center.z));
    const float dx=sphere.center.x-closest_x;
    const float dy=sphere.center.y-average_y;
    const float dz=sphere.center.z-closest_z;
    if (dx*dx+dy*dy+dz*dz <=
            (sphere.radius+0.045F)*(sphere.radius+0.045F)) pixels[tile]=1U;
}

void RayTracer::update_rope_bridge_paint(const float3* bridge_nodes,
    std::uint32_t node_count, RigidSphereState sphere,
    std::uint32_t columns, std::uint32_t rows, bool reset,
    cudaStream_t stream)
{
    if (reset) check(cudaMemsetAsync(ground_cloth_paint_pixels_,0,
        cloth_paint_pixel_count*sizeof(std::uint32_t),stream),
        "reset rope bridge paint pixels");
    if (bridge_nodes != nullptr && node_count != 0U) {
        const std::uint32_t count=columns*rows;
        update_rope_bridge_paint_kernel<<<(count+63U)/64U,64,0,stream>>>(
            bridge_nodes,node_count,sphere,columns,rows,ground_cloth_paint_pixels_);
        check(cudaGetLastError(),"update rope bridge paint pixels");
    }
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

float RayTracer::render_hybrid(
    parallel_mater::DeviceMeshView skin,
    const parallel_mater::NormalOutput& normals,
    const parallel_mater::Hierarchy& skin_hierarchy,
    const float3* particle_positions,
    float particle_radius,
    const parallel_mater::Hierarchy& particle_hierarchy,
    bool show_particles,
    const OrientedBox& collider,
    const Camera& camera,
    std::uint32_t width,
    std::uint32_t height,
    cudaStream_t stream,
    bool obstacle_course,
    FluidVisualView visuals,
    SoftBodyRenderView soft_bodies,
    GalleryArena arena)
{
    const bool foam_data = visuals.foam_particles != nullptr ||
        visuals.foam_capacity != 0U || visuals.foam_nodes != nullptr ||
        visuals.foam_indices != nullptr || visuals.foam_node_count != 0U;
    const bool modern_visuals = visuals.normal_foam != nullptr ||
        visuals.surface.values != nullptr || visuals.surface.grid != nullptr || foam_data;
    if (!modern_visuals) {
        visuals.display = show_particles ? FluidDisplay::Particles : FluidDisplay::Wireframe;
        visuals.show_foam = false;
    }
    const bool particle_display = visuals.show_particle_layer &&
        (visuals.display == FluidDisplay::Particles ||
            visuals.display == FluidDisplay::Wireframe);
    if (skin_hierarchy.statistics().max_depth > 18U ||
        (particle_display &&
            particle_hierarchy.statistics().max_depth > 18U) ||
        soft_bodies.max_depth > 18U || soft_bodies.member_max_depth > 18U) {
        throw std::runtime_error("hierarchy exceeds ray traversal stack contract");
    }
    const bool any_soft_body = soft_bodies.positions != nullptr ||
        soft_bodies.texcoords != nullptr || soft_bodies.triangles != nullptr ||
        soft_bodies.bindings != nullptr || soft_bodies.triangle_active != nullptr ||
        soft_bodies.nodes != nullptr || soft_bodies.primitive_indices != nullptr ||
        soft_bodies.vertex_count != 0U || soft_bodies.triangle_count != 0U ||
        soft_bodies.node_count != 0U;
    const bool complete_soft_body = soft_bodies.positions != nullptr &&
        soft_bodies.texcoords != nullptr && soft_bodies.triangles != nullptr &&
        soft_bodies.bindings != nullptr && soft_bodies.triangle_active != nullptr &&
        soft_bodies.nodes != nullptr && soft_bodies.primitive_indices != nullptr &&
        soft_bodies.vertex_count != 0U && soft_bodies.triangle_count != 0U &&
        soft_bodies.node_count != 0U;
    const bool any_members = soft_bodies.member_positions != nullptr ||
        soft_bodies.member_edges != nullptr || soft_bodies.member_active != nullptr ||
        soft_bodies.member_nodes != nullptr || soft_bodies.member_indices != nullptr ||
        soft_bodies.member_count != 0U || soft_bodies.member_node_count != 0U;
    const bool complete_members = soft_bodies.member_positions != nullptr &&
        soft_bodies.member_edges != nullptr && soft_bodies.member_active != nullptr &&
        soft_bodies.member_nodes != nullptr && soft_bodies.member_indices != nullptr &&
        soft_bodies.member_count != 0U && soft_bodies.members_per_instance != 0U &&
        soft_bodies.member_voxels_per_instance != 0U &&
        soft_bodies.member_node_count != 0U && soft_bodies.member_half_width > 0.0F;
    if (any_soft_body && !complete_soft_body) {
        throw std::invalid_argument("soft-body rendering requires a complete mesh and hierarchy");
    }
    if (any_members && !complete_members) {
        throw std::invalid_argument("soft-body member rendering requires a complete hierarchy");
    }
    if (particle_display &&
        (particle_positions == nullptr || particle_hierarchy.statistics().node_count == 0U)) {
        throw std::invalid_argument("particle display requires particle data");
    }
    if (visuals.display == FluidDisplay::Surface &&
        (visuals.surface.values == nullptr || visuals.surface.grid == nullptr)) {
        throw std::invalid_argument("fluid surface display requires a scalar grid");
    }
    const bool complete_foam = visuals.foam_particles != nullptr &&
        visuals.foam_capacity != 0U && visuals.foam_nodes != nullptr &&
        visuals.foam_indices != nullptr && visuals.foam_node_count != 0U;
    if (visuals.display == FluidDisplay::Surface && visuals.show_foam &&
        foam_data && !complete_foam) {
        throw std::invalid_argument("foam display requires a complete pool and hierarchy");
    }
    reserve(width, height);
    nvtx3::scoped_range operation_range{"waterlab/hybrid_raytrace"};
    check(cudaEventRecord(render_begin_, stream), "record hybrid render begin");
    const dim3 threads(16, 8);
    const dim3 blocks(
        (width + threads.x - 1U) / threads.x,
        (height + threads.y - 1U) / threads.y);
    const auto launch = [&]<bool course, GalleryArena scene>() {
        render_fluid_kernel<course, scene><<<blocks, threads, 0, stream>>>(
            device_pixels_, width, height, camera,
            skin.positions, normals.vertex_normals(), normals.corner_normal_indices(),
            skin.triangles, skin_hierarchy.nodes(), skin_hierarchy.primitive_indices(),
            skin_hierarchy.statistics().node_count, collider,
            particle_positions, particle_radius,
            particle_hierarchy.nodes(), particle_hierarchy.primitive_indices(),
            particle_hierarchy.statistics().node_count, visuals, soft_bodies);
    };
    if (obstacle_course) launch.operator()<true, GalleryArena::course>();
    else if (arena == GalleryArena::bowl)
        launch.operator()<false, GalleryArena::bowl>();
    else if (arena == GalleryArena::slope)
        launch.operator()<false, GalleryArena::slope>();
    else if (arena == GalleryArena::ground)
        launch.operator()<false, GalleryArena::ground>();
    else if (arena == GalleryArena::grass)
        launch.operator()<false, GalleryArena::grass>();
    else if (arena == GalleryArena::ground_box)
        launch.operator()<false, GalleryArena::ground_box>();
    else if (arena == GalleryArena::low_ceiling_box)
        launch.operator()<false, GalleryArena::low_ceiling_box>();
    else if (arena == GalleryArena::enclosed_box)
        launch.operator()<false, GalleryArena::enclosed_box>();
    else if (arena == GalleryArena::hot_pan)
        launch.operator()<false, GalleryArena::hot_pan>();
    else if (arena == GalleryArena::cloth_basin)
        launch.operator()<false, GalleryArena::cloth_basin>();
    else if (arena == GalleryArena::water_wheel)
        launch.operator()<false, GalleryArena::water_wheel>();
    else if (arena == GalleryArena::rope_post)
        launch.operator()<false, GalleryArena::rope_post>();
    else if (arena == GalleryArena::rope_bridge)
        launch.operator()<false, GalleryArena::rope_bridge>();
    else if (arena == GalleryArena::fishing_tank)
        launch.operator()<false, GalleryArena::fishing_tank>();
    else launch.operator()<false, GalleryArena::none>();
    check(cudaPeekAtLastError(), "hybrid raytrace launch");
    check(cudaEventRecord(render_end_, stream), "record hybrid render end");
    check(
        cudaMemcpyAsync(
            host_pixels_, device_pixels_,
            static_cast<std::size_t>(width) * height * sizeof(uchar4),
            cudaMemcpyDeviceToHost, stream),
        "download hybrid render target");
    check(cudaStreamSynchronize(stream), "synchronize hybrid raytrace");
    float milliseconds = 0.0F;
    check(cudaEventElapsedTime(&milliseconds, render_begin_, render_end_),
        "measure hybrid raytrace");
    return milliseconds;
}
} // namespace waterlab
