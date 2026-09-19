// SPDX-License-Identifier: MIT
#pragma once

#include "soft_body.hpp"

#include <vector_types.h>

#include <cstdint>

namespace waterlab {

// A vertical, Y-up cloth grid. Vertices are stored top-to-bottom and
// left-to-right. The two top corners are pinned; every grid vertex is also an
// identity-bound render vertex. Structural, shear, and bend springs share the
// same canonical edge/CSR representation as other soft bodies.
struct ClothGridOptions {
    std::uint32_t columns{24U};
    std::uint32_t rows{18U};
    float spacing{0.05F};
    float3 top_center{};
    bool shear_springs{true};
    bool bend_springs{true};
};

[[nodiscard]] SoftBodyAsset make_cloth_grid(const ClothGridOptions& options = {});
[[nodiscard]] SoftBodyAsset make_soft_sphere(
    std::uint32_t surface_nodes = 500U,
    std::uint32_t interior_nodes = 500U,
    float radius = 0.46F);
// Centerline spring chain with a six-sided render tube. Node zero is pinned;
// the last node is the rigid-sphere attachment point used by context 8.
[[nodiscard]] SoftBodyAsset make_soft_rope(
    std::uint32_t node_count = 32U, float spacing = 0.055F);
// A pinned trunk splits into a directly tethered branch and a second branch
// carrying a dodecahedral rope cage around a deforming glass render sphere.
// The direct-attachment endpoint is deliberately stored as the final node.
[[nodiscard]] SoftBodyAsset make_y_rope_cage(
    std::uint32_t node_count = 64U, float total_length = 4.0F);
// Forty independently braced 0.3 m deck tiles. Adjacent tile edges are joined
// by two 0.2 m spring ropes, so an interior tile owns eight bridge ropes.
[[nodiscard]] SoftBodyAsset make_rope_bridge(
    std::uint32_t columns = rope_bridge_columns,
    std::uint32_t rows = rope_bridge_rows);
// Three-layer load-bearing cross used by the water wheel. A small central
// volume is fixed to the axle and its four tips are fixed to the outer rim;
// the material between them remains a connected, shear-braced soft graph.
[[nodiscard]] SoftBodyAsset make_soft_cross(
    std::uint32_t span = 41U, std::uint32_t arm_width = 5U,
    float spacing = 0.10F);

// Column-vector rigid transform: output = translation + x*x_axis + y*y_axis +
// z*z_axis. Axes must form a finite, right-handed orthonormal basis.
struct SoftBodyRigidTransform {
    float3 x_axis{1.0F, 0.0F, 0.0F};
    float3 y_axis{0.0F, 1.0F, 0.0F};
    float3 z_axis{0.0F, 0.0F, 1.0F};
    float3 translation{};
};

[[nodiscard]] SoftBodyAsset transform_soft_body_asset(
    SoftBodyAsset asset, const SoftBodyRigidTransform& transform);
[[nodiscard]] SoftBodyAsset translate_soft_body_asset(
    SoftBodyAsset asset, float3 translation);
// Rotates the generator's XY cloth plane onto XZ with its front normal +Y.
[[nodiscard]] SoftBodyAsset rotate_cloth_to_horizontal(SoftBodyAsset asset);
// Adds a shallow sag to the single flexible floor. Context 5 supplies its
// nine support points separately and uses a rigid perimeter for containment.
[[nodiscard]] SoftBodyAsset shape_cloth_catch_basin(SoftBodyAsset asset);

// Combines two disconnected graphs. Assets must use matching spacing, radius,
// and file conventions because SoftBodyCourse has one set of scalar material
// metadata per graph.
[[nodiscard]] SoftBodyAsset merge_soft_body_assets(
    const SoftBodyAsset& first, const SoftBodyAsset& second);

} // namespace waterlab
