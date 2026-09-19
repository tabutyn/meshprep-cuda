// SPDX-License-Identifier: MIT
#pragma once

#include "fluid_visuals.hpp"
#include "soft_body.hpp"
#include "obstacle_course.hpp"

#include <meshprep/meshprep.hpp>

#include <cuda_runtime_api.h>
#include <vector_functions.h>
#include <vector_types.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace waterlab {

struct HostSurfaceMesh {
    std::vector<float3> positions;
    std::vector<uint3> triangles;
    std::vector<std::uint32_t> neighbor_offsets;
    std::vector<std::uint32_t> neighbors;
    std::vector<float> rest_lengths;
};

struct SurfaceVertexEmbedding {
    uint3 source_vertices{};
    float3 barycentric{};
};

[[nodiscard]] HostSurfaceMesh make_geodesic_sphere(std::uint32_t frequency, float radius);
[[nodiscard]] std::vector<SurfaceVertexEmbedding> make_surface_embedding(
    const HostSurfaceMesh& source,
    const std::vector<float3>& target_positions);
struct OrientedBox {
    float3 center{1.45F, 0.0F, 0.0F};
    float3 half_extents{0.30F, 0.65F, 0.42F};
    float yaw{};
    float4 sphere_orientation{0.0F, 0.0F, 0.0F, 1.0F};
};

struct Camera {
    float3 eye{0.0F, 0.0F, 4.2F};
    float3 target{0.0F, 0.0F, 0.0F};
    float3 up{0.0F, 1.0F, 0.0F};
    float vertical_fov_degrees{42.0F};
};

class RayTracer {
public:
    RayTracer();
    ~RayTracer();
    RayTracer(const RayTracer&) = delete;
    RayTracer& operator=(const RayTracer&) = delete;

    // Marks persistent bowl texels only where fluid particle centers are
    // within one particle radius of the inner hemisphere.
    [[nodiscard]] float update_bowl_paint(
        const float3* particle_positions,
        std::uint32_t particle_count,
        float particle_radius,
        bool reset = false,
        cudaStream_t stream = nullptr);
    [[nodiscard]] float update_sphere_paint(
        const float3* contact_positions,
        std::uint32_t contact_count,
        float contact_radius,
        RigidSphereState sphere,
        bool reset = false,
        cudaStream_t stream = nullptr);
    void update_goal_cloth_paint(
        const float3* source_positions,
        std::uint32_t source_count,
        float source_radius,
        bool reset = false,
        cudaStream_t stream = nullptr);
    void update_rope_bridge_paint(
        const float3* bridge_nodes,
        std::uint32_t node_count,
        RigidSphereState sphere,
        std::uint32_t columns,
        std::uint32_t rows,
        bool reset = false,
        cudaStream_t stream = nullptr);

    [[nodiscard]] float render_hybrid(
        meshprep::DeviceMeshView skin,
        const meshprep::NormalOutput& normals,
        const meshprep::Hierarchy& skin_hierarchy,
        const float3* particle_positions,
        float particle_radius,
        const meshprep::Hierarchy& particle_hierarchy,
        bool show_particles,
        const OrientedBox& collider,
        const Camera& camera,
        std::uint32_t width,
        std::uint32_t height,
        cudaStream_t stream = nullptr,
        bool obstacle_course = false,
        FluidVisualView visuals = {},
        SoftBodyRenderView soft_bodies = {},
        GalleryArena arena = GalleryArena::none);
    [[nodiscard]] const uchar4* pixels() const noexcept { return host_pixels_; }

private:
    void reserve(std::uint32_t width, std::uint32_t height);

    uchar4* device_pixels_{};
    uchar4* host_pixels_{};
    cudaEvent_t render_begin_{};
    cudaEvent_t render_end_{};
    std::uint32_t* bowl_paint_pixels_{};
    std::uint32_t* bowl_painted_count_{};
    std::uint32_t* sphere_paint_pixels_{};
    std::uint32_t* sphere_painted_count_{};
    std::uint32_t* cloth_paint_pixels_{};
    std::uint32_t* ground_cloth_paint_pixels_{};
    std::size_t pixel_capacity_{};
};

} // namespace waterlab
