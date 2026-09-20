// SPDX-License-Identifier: MIT
#pragma once

#include <parallel_mater/soft_body.hpp>

#include <cstdint>
#include <memory>

namespace parallel_mater::physics {

struct ClothOptions {
    std::uint32_t columns{24U};
    std::uint32_t rows{18U};
    float spacing{0.05F};
    float3 top_center{};
    bool shear_springs{true};
    bool bend_springs{true};
    SoftBodyOptions solver{};
};

class Cloth {
public:
    Cloth() noexcept;
    ~Cloth();
    Cloth(Cloth&&) noexcept;
    Cloth& operator=(Cloth&&) noexcept;
    Cloth(const Cloth&) = delete;
    Cloth& operator=(const Cloth&) = delete;

    [[nodiscard]] static Status create(ClothOptions options, Cloth& output,
        cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status initialize(ClothOptions options = {},
        cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status begin_frame(FrameOptions frame,
        cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status prepare_substep(SubstepContext substep,
        cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status finish_substep(SubstepContext substep,
        cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status finish_frame(Completion& completion,
        cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status advance_async(FrameOptions frame,
        Completion& completion, cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status advance(FrameOptions frame,
        cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status collect_statistics_async(Completion& completion,
        cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status collect_statistics(
        cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status reset(cudaStream_t stream = nullptr) noexcept;

    [[nodiscard]] bool initialized() const noexcept;
    [[nodiscard]] ClothOptions options() const noexcept;
    [[nodiscard]] SoftBodyNodeView nodes() const noexcept;
    [[nodiscard]] PointCouplingView coupling_points() const noexcept;
    [[nodiscard]] SoftBodyBondView bonds() const noexcept;
    [[nodiscard]] SoftBodySurfaceView surface() const noexcept;
    [[nodiscard]] SoftBodyStatistics statistics() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace parallel_mater::physics
