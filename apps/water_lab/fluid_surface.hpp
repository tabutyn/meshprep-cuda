// SPDX-License-Identifier: MIT
#pragma once

#include "particle_cells.hpp"

#include <meshprep/meshprep.hpp>
#include <cuda_runtime_api.h>
#include <vector_types.h>
#include <cstddef>
#include <cstdint>

namespace waterlab {

struct FluidSurfaceGrid {
    float3 minimum{};
    float3 cell_size{};
    uint3 dimensions{};
    float support_radius{};
};

struct FluidSurfaceView {
    const float* values{};
    const FluidSurfaceGrid* grid{}; // Device descriptor; x-major scalar storage.
};

// Rendering-only particle-derived scalar field. No changes to particle physics.
class FluidSurface {
public:
    explicit FluidSurface(std::uint32_t resolution = 48U);
    ~FluidSurface();
    FluidSurface(const FluidSurface&) = delete;
    FluidSurface& operator=(const FluidSurface&) = delete;
    [[nodiscard]] float update(const float3* positions, std::uint32_t particle_count,
        const meshprep::Hierarchy& hierarchy, float support_radius,
        cudaStream_t stream = nullptr, ParticleCellView cells = {});
    [[nodiscard]] FluidSurfaceView view() const noexcept { return {values_, grid_}; }
    [[nodiscard]] std::size_t allocated_bytes() const noexcept {
        return static_cast<std::size_t>(resolution_)*resolution_*resolution_*sizeof(float) +
            sizeof(FluidSurfaceGrid) + sizeof(std::uint32_t);
    }
private:
    float* values_{};
    FluidSurfaceGrid* grid_{};
    std::uint32_t* errors_{};
    std::uint32_t resolution_{};
    cudaEvent_t begin_{}, end_{};
};

} // namespace waterlab
