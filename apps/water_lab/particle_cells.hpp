// SPDX-License-Identifier: MIT
#pragma once

#include <vector_types.h>

#include <cstdint>

namespace waterlab {

// Borrowed, device-resident spatial ordering for fixed-radius particle queries.
// Keys are sorted; indices are a permutation of [0, particle_count), and each
// key was generated from indexed_positions with cells of cell_size width. Cell
// coordinates must remain in the packed 21-bit range [-2^20, 2^20-1]. The owner
// keeps every buffer alive and unchanged until queued stream work completes.
struct ParticleCellView {
    const float3* indexed_positions{};
    const std::uint64_t* keys{};
    const std::uint32_t* indices{};
    std::uint32_t particle_count{};
    float cell_size{};

    [[nodiscard]] constexpr bool empty() const noexcept {
        return indexed_positions == nullptr && keys == nullptr && indices == nullptr &&
            particle_count == 0U && cell_size == 0.0F;
    }
};

} // namespace waterlab
