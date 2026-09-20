// SPDX-License-Identifier: MIT
#pragma once

#include <parallel_mater/soft_body.hpp>

#include <cstdint>
#include <memory>

namespace parallel_mater::physics {

struct RopeOptions {
    std::uint32_t node_count{32U};
    float spacing{0.055F};
    float3 origin{};
    float3 direction{1.0F, 0.0F, 0.0F};
    SoftBodyOptions solver{};
};

class Rope {
  public:
    Rope() noexcept;
    ~Rope();
    Rope(Rope &&) noexcept;
    Rope &operator=(Rope &&) noexcept;
    Rope(const Rope &) = delete;
    Rope &operator=(const Rope &) = delete;

    [[nodiscard]] static Status create(RopeOptions options, Rope &output,
                                       cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status initialize(RopeOptions options = {},
                                    cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status begin_frame(FrameOptions frame, cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status prepare_substep(SubstepContext substep,
                                         cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status finish_substep(SubstepContext substep,
                                        cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status finish_frame(Completion &completion,
                                      cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status abandon_frame(cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status advance_async(FrameOptions frame, Completion &completion,
                                       cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status advance(FrameOptions frame, cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status collect_statistics_async(Completion &completion,
                                                  cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status collect_statistics(cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status reset(cudaStream_t stream = nullptr) noexcept;

    [[nodiscard]] bool initialized() const noexcept;
    [[nodiscard]] RopeOptions options() const noexcept;
    [[nodiscard]] SoftBodyNodeView nodes() const noexcept;
    [[nodiscard]] PointStateView point_state() const noexcept;
    [[nodiscard]] PointCouplingView coupling_points() const noexcept;
    [[nodiscard]] SoftBodyBondView bonds() const noexcept;
    [[nodiscard]] SoftBodySurfaceView surface() const noexcept;
    [[nodiscard]] SoftBodyStatistics statistics() const noexcept;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace parallel_mater::physics
