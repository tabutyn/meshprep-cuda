// SPDX-License-Identifier: MIT
#pragma once

#include <parallel_mater/frame.hpp>

#include <vector_types.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace parallel_mater::physics {

struct FluidParticle {
    float3 position{};
    float3 velocity{};
};

struct FluidOptions {
    float particle_radius{0.0225F};
    float interaction_radius{0.09F};
    float particle_mass{1.0F};
    float repulsion{20.0F};
    float viscosity{0.02F};
    float velocity_damping{0.2F};
    float maximum_speed{8.0F};
};

struct FluidView {
    const float3 *positions{};
    const float3 *velocities{};
    float3 *external_impulses{};
    std::uint32_t particle_count{};
    float particle_radius{};
    float inverse_particle_mass{};
};

struct FluidStatistics {
    std::uint32_t particle_count{};
    std::uint32_t maximum_neighbors{};
    std::uint32_t finite_failure_count{};
    std::uint64_t frame_index{};
    std::size_t allocated_bytes{};
};

class Fluid {
  public:
    Fluid() noexcept;
    ~Fluid();
    Fluid(Fluid &&) noexcept;
    Fluid &operator=(Fluid &&) noexcept;
    Fluid(const Fluid &) = delete;
    Fluid &operator=(const Fluid &) = delete;

    [[nodiscard]] static Status create(std::span<const FluidParticle> particles,
                                       FluidOptions options, Fluid &output,
                                       cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status initialize(std::span<const FluidParticle> particles,
                                    FluidOptions options = {},
                                    cudaStream_t stream = nullptr) noexcept;
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
    [[nodiscard]] Status collect_statistics_async(Completion &completion,
                                                  cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status collect_statistics(cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status reset(cudaStream_t stream = nullptr) noexcept;

    [[nodiscard]] bool initialized() const noexcept;
    [[nodiscard]] FluidOptions options() const noexcept;
    [[nodiscard]] FluidView particles() const noexcept;
    [[nodiscard]] PointCouplingView coupling_points() const noexcept;
    // Last completed snapshot; frame submission itself performs no readback.
    [[nodiscard]] FluidStatistics statistics() const noexcept;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace parallel_mater::physics
