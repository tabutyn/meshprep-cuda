// SPDX-License-Identifier: MIT
#include <parallel_mater/smoke.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

} // namespace

int main() {
    int devices{};
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::puts("SKIP: CUDA device unavailable");
        return 77;
    }
    try {
        parallel_mater::physics::SmokeOptions options;
        options.capacity = 2'048U;
        options.particle_count = 2'048U;
        options.lifetime = 2.0F;
        options.emitter_center = make_float3(-1.0F, 0.0F, 0.0F);
        options.emitter_half_extents = make_float3(0.02F, 0.10F, 0.10F);
        options.initial_velocity = make_float3(2.0F, 0.0F, 0.0F);
        options.buoyancy = 0.8F;
        parallel_mater::physics::Smoke smoke;
        require(parallel_mater::physics::Smoke::create(options, smoke).ok(),
                "smoke creation failed");
        const auto initial = smoke.particles();
        require(initial.count == options.particle_count && initial.positions != nullptr &&
                    initial.velocities != nullptr && initial.temperatures != nullptr,
                "smoke did not expose its device-resident particle view");
        parallel_mater::physics::Collider sphere;
        sphere.position = make_float3(0.2F, 0.4F, 0.0F);
        sphere.dimensions = make_float3(0.32F, 0.0F, 0.0F);
        sphere.friction = 0.25F;
        parallel_mater::physics::ColliderSet colliders;
        require(
            colliders.update(std::span<const parallel_mater::physics::Collider>(&sphere, 1U)).ok(),
            "smoke collider upload failed");
        parallel_mater::physics::SmokeTimings timing{};
        for (std::uint32_t frame = 0U; frame < 180U; ++frame) {
            require(smoke.step({}, colliders.view(), timing).ok(), "smoke integration failed");
        }
        const auto statistics = smoke.statistics();
        require(statistics.frame_index == 180U && statistics.respawn_count > 0U &&
                    statistics.finite_failure_count == 0U && statistics.maximum_speed > 0.1F &&
                    timing.integrate_ms > 0.0F,
                "smoke integration statistics are invalid");
        std::vector<float3> positions(initial.count);
        require(cudaMemcpy(positions.data(), smoke.particles().positions,
                           positions.size() * sizeof(float3),
                           cudaMemcpyDeviceToHost) == cudaSuccess,
                "smoke positions were not readable");
        std::size_t finite_count{};
        float maximum_y = -INFINITY;
        for (const float3 point : positions) {
            finite_count +=
                std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z);
            maximum_y = std::max(maximum_y, point.y);
        }
        require(finite_count == positions.size() && maximum_y > 0.25F,
                "smoke did not remain finite or respond to buoyancy");

        float3 *body_positions{};
        float3 *body_velocities{};
        float3 *body_impulses{};
        require(cudaMalloc(&body_positions, sizeof(float3)) == cudaSuccess &&
                    cudaMalloc(&body_velocities, sizeof(float3)) == cudaSuccess &&
                    cudaMalloc(&body_impulses, sizeof(float3)) == cudaSuccess,
                "smoke coupling fixture allocation failed");
        const float3 body_point = make_float3(0.0F, 0.2F, 0.0F);
        cudaMemcpy(body_positions, &body_point, sizeof(float3), cudaMemcpyHostToDevice);
        cudaMemset(body_velocities, 0, sizeof(float3));
        cudaMemset(body_impulses, 0, sizeof(float3));
        parallel_mater::physics::PointCouplingView body{
            body_positions, body_velocities, body_impulses, nullptr, 1U, 0.03F, 2.0F};
        require(smoke.couple(body, 1.0F / 60.0F, 8.0F).ok(), "smoke/body coupling failed");
        require(smoke.collect_telemetry().ok(), "smoke coupling telemetry failed");
        timing = smoke.telemetry().timings;
        float3 impulse{};
        cudaMemcpy(&impulse, body_impulses, sizeof(float3), cudaMemcpyDeviceToHost);
        cudaFree(body_impulses);
        cudaFree(body_velocities);
        cudaFree(body_positions);
        require(std::isfinite(impulse.x) && std::isfinite(impulse.y) && std::isfinite(impulse.z) &&
                    std::hypot(std::hypot(impulse.x, impulse.y), impulse.z) > 0.0F &&
                    timing.couple_ms > 0.0F,
                "smoke did not transfer aerodynamic impulse");
        std::puts("smoke tests passed");
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "smoke test failure: %s\n", error.what());
        return 1;
    }
}
