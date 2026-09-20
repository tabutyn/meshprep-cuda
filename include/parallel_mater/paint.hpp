// SPDX-License-Identifier: MIT
#pragma once

#include <parallel_mater/geometry.hpp>

#include <cuda_runtime_api.h>
#include <vector_types.h>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace parallel_mater::physics {

// Persistent RGBA texture with an application-defined parameterization. The
// same owner can paint cloth UVs, rigid sphere/box atlases, or other surfaces.
struct PaintSurfaceOptions {
    std::uint32_t width{64U};
    std::uint32_t height{64U};
    float4 clear_color{};
    bool wrap_u{};
};

struct PaintSurfaceView {
    float4 *colors{}; // device memory, row-major
    std::uint32_t width{};
    std::uint32_t height{};
};

// UV coordinates are normalized to [0,1], and radius is normalized to the
// shorter texture axis. Stamps are consumed in array order, so overlapping
// translucent paint remains deterministic.
struct PaintStamp {
    float2 uv{};
    float radius{0.02F};
    float4 color{1.0F, 1.0F, 1.0F, 1.0F};
    float opacity{1.0F};
};

struct PaintStampView {
    const PaintStamp *data{}; // device memory
    std::uint32_t count{};
};

class PaintSurface {
  public:
    PaintSurface() noexcept;
    ~PaintSurface();
    PaintSurface(PaintSurface &&) noexcept;
    PaintSurface &operator=(PaintSurface &&) noexcept;
    PaintSurface(const PaintSurface &) = delete;
    PaintSurface &operator=(const PaintSurface &) = delete;

    [[nodiscard]] static Status create(PaintSurfaceOptions options, PaintSurface &output,
                                       cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status initialize(PaintSurfaceOptions options = {},
                                    cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status clear_async(cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status clear(cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status apply_async(PaintStampView stamps,
                                     cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status apply(PaintStampView stamps,
                               cudaStream_t stream = nullptr) noexcept;

    [[nodiscard]] bool initialized() const noexcept;
    [[nodiscard]] PaintSurfaceOptions options() const noexcept;
    [[nodiscard]] PaintSurfaceView view() const noexcept;
    [[nodiscard]] std::size_t allocated_bytes() const noexcept;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace parallel_mater::physics
