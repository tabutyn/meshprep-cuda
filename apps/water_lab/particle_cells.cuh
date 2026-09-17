// SPDX-License-Identifier: MIT
#pragma once

#include "particle_cells.hpp"

#include <cuda_runtime.h>

namespace waterlab::detail {

inline constexpr int particle_cell_bias = 1 << 20;

__host__ __device__ inline std::uint64_t particle_cell_key(int x, int y, int z)
{
    const auto encoded_x = static_cast<std::uint64_t>(x + particle_cell_bias);
    const auto encoded_y = static_cast<std::uint64_t>(y + particle_cell_bias);
    const auto encoded_z = static_cast<std::uint64_t>(z + particle_cell_bias);
    return (encoded_x << 42U) | (encoded_y << 21U) | encoded_z;
}

__device__ inline std::uint32_t particle_cell_lower_bound(
    const ParticleCellView& cells, std::uint64_t key)
{
    std::uint32_t first = 0U;
    std::uint32_t last = cells.particle_count;
    while (first < last) {
        const std::uint32_t middle = first + (last - first) / 2U;
        if (cells.keys[middle] < key) first = middle + 1U;
        else last = middle;
    }
    return first;
}

} // namespace waterlab::detail
