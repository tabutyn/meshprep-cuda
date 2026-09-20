// SPDX-License-Identifier: MIT
#pragma once

#include <parallel_mater/coupling.hpp>

#include <cuda_runtime_api.h>
#include <vector_types.h>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace parallel_mater::physics {

struct SmokeOptions {
    std::uint32_t capacity{8'192U};
    std::uint32_t particle_count{4'096U};
    float timestep{1.0F / 60.0F};
    float lifetime{4.0F};
    float particle_radius{0.035F};
    float3 emitter_center{-2.2F, -0.75F, -1.2F};
    float3 emitter_half_extents{0.08F, 0.45F, 0.45F};
    float3 initial_velocity{2.4F, 0.35F, 0.0F};
    float buoyancy{0.55F};
    float velocity_damping{0.18F};
    float turbulence_strength{1.25F};
    float turbulence_frequency{2.0F};
    float maximum_speed{8.0F};
    std::uint32_t seed{0x51A0C3U};
};

struct SmokeParticleView {
    const float3 *positions{};
    const float3 *velocities{};
    const float *ages{};
    const float *temperatures{};
    std::uint32_t count{};
    float radius{};
};

struct SmokeTimings {
    float integrate_ms{};
    float couple_ms{};

    [[nodiscard]] constexpr float gpu_total_ms() const noexcept { return integrate_ms + couple_ms; }
};

struct SmokeStatistics {
    std::uint64_t frame_index{};
    std::uint64_t respawn_count{};
    std::uint32_t finite_failure_count{};
    float maximum_speed{};
    std::size_t allocated_bytes{};
};

struct SmokeTelemetry {
    SmokeTimings timings{};
    SmokeStatistics statistics{};
};

// Deterministic device-resident smoke/advection particles. This is a compact
// visual and coupling model, not a pressure-projected CFD solver. Advancing is
// enqueue-only; diagnostic readbacks are opt-in through collect_telemetry.
class Smoke {
  public:
    Smoke() noexcept;
    ~Smoke();
    Smoke(Smoke &&) noexcept;
    Smoke &operator=(Smoke &&) noexcept;
    Smoke(const Smoke &) = delete;
    Smoke &operator=(const Smoke &) = delete;

    [[nodiscard]] static Status create(SmokeOptions options, Smoke &output,
                                       cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status initialize(SmokeOptions options = {},
                                    cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status set_colliders(ColliderView colliders) noexcept;
    [[nodiscard]] Status step(float3 acceleration = {}, ColliderView colliders = {},
                              cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status step(float3 acceleration, ColliderView colliders, SmokeTimings &timings,
                              cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status step_async(float3 acceleration, ColliderView colliders,
                                    Completion &completion, cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status begin_frame(FrameOptions frame, cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status prepare_substep(SubstepContext substep,
                                         cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status finish_substep(SubstepContext substep,
                                        cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status finish_frame(Completion &completion,
                                      cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status advance_async(FrameOptions frame, Completion &completion,
                                       cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status advance(FrameOptions frame, cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status couple(PointCouplingView body, float timestep, float drag_coefficient,
                                cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status collect_telemetry_async(Completion &completion,
                                                 cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status resolve_telemetry(SmokeTelemetry &output) noexcept;
    [[nodiscard]] Status collect_telemetry(cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status reset(cudaStream_t stream = nullptr) noexcept;

    [[nodiscard]] bool initialized() const noexcept;
    [[nodiscard]] SmokeOptions options() const noexcept;
    [[nodiscard]] SmokeParticleView particles() const noexcept;
    [[nodiscard]] SmokeTelemetry telemetry() const noexcept;
    [[nodiscard]] SmokeStatistics statistics() const noexcept;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace parallel_mater::physics
