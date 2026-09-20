// SPDX-License-Identifier: MIT
#pragma once

#include <parallel_mater/frame.hpp>

#include <vector_types.h>

#include <cstdint>
#include <memory>

namespace parallel_mater::physics {

enum class RigidShape : std::uint8_t { sphere, box };

struct RigidBodyOptions {
    RigidShape shape{RigidShape::sphere};
    float mass{1.0F};
    float radius{0.5F};
    float3 half_extents{0.5F, 0.5F, 0.5F};
    float3 inertia{0.1F, 0.1F, 0.1F};
    float linear_damping{0.1F};
    float angular_damping{0.1F};
    float maximum_linear_speed{100.0F};
    float maximum_angular_speed{100.0F};
};

struct RigidBodyState {
    float3 position{};
    float4 orientation{0.0F, 0.0F, 0.0F, 1.0F};
    float3 linear_velocity{};
    float3 angular_velocity{};
};

class RigidBody {
  public:
    RigidBody() noexcept;
    ~RigidBody();
    RigidBody(RigidBody &&) noexcept;
    RigidBody &operator=(RigidBody &&) noexcept;
    RigidBody(const RigidBody &) = delete;
    RigidBody &operator=(const RigidBody &) = delete;

    [[nodiscard]] static Status create(RigidBodyState state, RigidBodyOptions options,
                                       RigidBody &output) noexcept;
    [[nodiscard]] Status initialize(RigidBodyState state = {},
                                    RigidBodyOptions options = {}) noexcept;
    [[nodiscard]] Status begin_frame(FrameOptions frame, cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status prepare_substep(SubstepContext substep,
                                         cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status apply_force(float3 force, float3 world_point) noexcept;
    [[nodiscard]] Status apply_torque(float3 torque) noexcept;
    [[nodiscard]] Status finish_substep(SubstepContext substep,
                                        cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status finish_frame(Completion &completion,
                                      cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status advance_async(FrameOptions frame, Completion &completion,
                                       cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status advance(FrameOptions frame, cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status reset() noexcept;

    [[nodiscard]] bool initialized() const noexcept;
    [[nodiscard]] RigidBodyOptions options() const noexcept;
    [[nodiscard]] RigidBodyState state() const noexcept;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace parallel_mater::physics
