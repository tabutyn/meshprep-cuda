// SPDX-License-Identifier: MIT
#pragma once

#include <meshprep/meshprep.hpp>

#include <cuda_runtime_api.h>
#include <vector_functions.h>
#include <vector_types.h>

#include <cstdint>
#include <vector>

namespace waterlab {

struct HostSurfaceMesh {
    std::vector<float3> positions;
    std::vector<uint3> triangles;
    std::vector<std::uint32_t> neighbor_offsets;
    std::vector<std::uint32_t> neighbors;
    std::vector<float> rest_lengths;
    float rest_volume{};
};

[[nodiscard]] HostSurfaceMesh make_geodesic_sphere(std::uint32_t frequency, float radius);

struct PhysicsOptions {
    float stiffness{260.0F};
    float edge_damping{2.5F};
    float velocity_damping{0.35F};
    float pressure{90.0F};
    float maximum_speed{8.0F};
};

class WaterSurface {
public:
    explicit WaterSurface(const HostSurfaceMesh& mesh);
    ~WaterSurface();
    WaterSurface(const WaterSurface&) = delete;
    WaterSurface& operator=(const WaterSurface&) = delete;

    [[nodiscard]] meshprep::DeviceMeshView mesh_view() const noexcept;
    [[nodiscard]] std::uint32_t vertex_count() const noexcept { return vertex_count_; }
    [[nodiscard]] std::uint32_t triangle_count() const noexcept { return triangle_count_; }
    [[nodiscard]] std::uint32_t edge_count() const noexcept { return directed_edge_count_ / 2U; }

    [[nodiscard]] float step(
        float frame_dt,
        std::uint32_t substeps,
        const PhysicsOptions& options,
        cudaStream_t stream = nullptr);
    void apply_impulse(
        float3 point,
        float3 direction,
        float radius,
        float strength,
        cudaStream_t stream = nullptr);
    void reset(cudaStream_t stream = nullptr);

private:
    float3* positions_a_{};
    float3* positions_b_{};
    float3* velocities_a_{};
    float3* velocities_b_{};
    float3* rest_positions_{};
    uint3* triangles_{};
    std::uint32_t* neighbor_offsets_{};
    std::uint32_t* neighbors_{};
    float* rest_lengths_{};
    float4* constraint_state_{};
    cudaEvent_t step_begin_{};
    cudaEvent_t step_end_{};
    std::uint32_t vertex_count_{};
    std::uint32_t triangle_count_{};
    std::uint32_t directed_edge_count_{};
    float rest_volume_{};
};

struct Camera {
    float3 eye{0.0F, 0.0F, 4.2F};
    float3 target{0.0F, 0.0F, 0.0F};
    float3 up{0.0F, 1.0F, 0.0F};
    float vertical_fov_degrees{42.0F};
};

struct PickResult {
    bool hit{};
    float3 position{};
    float3 normal{};
};

class RayTracer {
public:
    RayTracer();
    ~RayTracer();
    RayTracer(const RayTracer&) = delete;
    RayTracer& operator=(const RayTracer&) = delete;

    [[nodiscard]] float render(
        meshprep::DeviceMeshView mesh,
        const meshprep::Hierarchy& hierarchy,
        const Camera& camera,
        std::uint32_t width,
        std::uint32_t height,
        float time_seconds,
        cudaStream_t stream = nullptr);
    [[nodiscard]] PickResult pick(
        meshprep::DeviceMeshView mesh,
        const meshprep::Hierarchy& hierarchy,
        const Camera& camera,
        std::uint32_t pixel_x,
        std::uint32_t pixel_y,
        std::uint32_t width,
        std::uint32_t height,
        cudaStream_t stream = nullptr);

    [[nodiscard]] const uchar4* pixels() const noexcept { return host_pixels_; }

private:
    void reserve(std::uint32_t width, std::uint32_t height);

    uchar4* device_pixels_{};
    uchar4* host_pixels_{};
    PickResult* device_pick_{};
    cudaEvent_t render_begin_{};
    cudaEvent_t render_end_{};
    std::size_t pixel_capacity_{};
};

} // namespace waterlab
