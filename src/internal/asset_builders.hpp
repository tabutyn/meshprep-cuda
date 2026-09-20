// SPDX-License-Identifier: MIT
#pragma once

#include "fixed_topology.hpp"

namespace parallel_mater::physics::detail {

[[nodiscard]] waterlab::SoftBodyAsset make_cloth_asset(
    std::uint32_t columns, std::uint32_t rows, float spacing,
    float3 top_center, bool shear_springs, bool bend_springs);

[[nodiscard]] waterlab::SoftBodyAsset make_rope_asset(
    std::uint32_t node_count, float spacing, float3 origin, float3 direction);

} // namespace parallel_mater::physics::detail
