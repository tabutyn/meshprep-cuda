// SPDX-License-Identifier: MIT
#include "asset_builders.hpp"

#include <vector_functions.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace parallel_mater::physics::detail {
namespace {

float distance(float3 a, float3 b)
{
    return std::hypot(std::hypot(b.x - a.x, b.y - a.y), b.z - a.z);
}

void rebuild_adjacency(waterlab::SoftBodyAsset& asset)
{
    std::sort(asset.edges.begin(), asset.edges.end(), [](const auto& a, const auto& b) {
        return a.vertices.x < b.vertices.x ||
            (a.vertices.x == b.vertices.x && a.vertices.y < b.vertices.y);
    });
    std::vector<std::vector<waterlab::SoftBodyNeighbor>> adjacency(
        asset.rest_voxels.size());
    for (std::uint32_t edge = 0U; edge < asset.edges.size(); ++edge) {
        const uint2 vertices = asset.edges[edge].vertices;
        adjacency[vertices.x].push_back({vertices.y, edge});
        adjacency[vertices.y].push_back({vertices.x, edge});
    }
    asset.neighbor_offsets = {0U};
    asset.neighbors.clear();
    for (auto& neighbors : adjacency) {
        std::sort(neighbors.begin(), neighbors.end(), [](const auto& a, const auto& b) {
            return a.vertex < b.vertex;
        });
        asset.neighbors.insert(asset.neighbors.end(), neighbors.begin(), neighbors.end());
        asset.neighbor_offsets.push_back(
            static_cast<std::uint32_t>(asset.neighbors.size()));
    }
}

float3 add(float3 a, float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

float3 multiply(float3 value, float scale)
{
    return make_float3(value.x * scale, value.y * scale, value.z * scale);
}

float3 cross(float3 a, float3 b)
{
    return make_float3(a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

float length(float3 value)
{
    return std::sqrt(value.x * value.x + value.y * value.y + value.z * value.z);
}

} // namespace

waterlab::SoftBodyAsset make_cloth_asset(std::uint32_t columns,
    std::uint32_t rows, float spacing, float3 top_center,
    bool shear_springs, bool bend_springs)
{
    if (columns < 2U || rows < 2U || !std::isfinite(spacing) ||
        spacing <= 0.0F || columns > std::numeric_limits<std::uint32_t>::max() / rows) {
        throw std::invalid_argument("invalid cloth topology");
    }
    const std::uint32_t count = columns * rows;
    const auto index = [columns](std::uint32_t column, std::uint32_t row) {
        return row * columns + column;
    };
    waterlab::SoftBodyAsset asset;
    asset.nominal_spacing = spacing;
    asset.voxel_radius = 0.45F * spacing;
    asset.voxel_flags.assign(count, waterlab::soft_body_voxel_surface);
    const float left = -0.5F * static_cast<float>(columns - 1U) * spacing;
    for (std::uint32_t row = 0U; row < rows; ++row) {
        for (std::uint32_t column = 0U; column < columns; ++column) {
            const std::uint32_t vertex = index(column, row);
            const float3 position = make_float3(
                top_center.x + left + static_cast<float>(column) * spacing,
                top_center.y - static_cast<float>(row) * spacing, top_center.z);
            asset.rest_voxels.push_back(position);
            asset.render_positions.push_back(position);
            asset.render_uvs.push_back(make_float2(
                static_cast<float>(column) / static_cast<float>(columns - 1U),
                1.0F - static_cast<float>(row) / static_cast<float>(rows - 1U)));
            asset.render_bindings.push_back({make_uint4(vertex, vertex, vertex, vertex),
                make_float4(1.0F, 0.0F, 0.0F, 0.0F)});
        }
    }
    asset.voxel_flags[index(0U, 0U)] |= waterlab::soft_body_voxel_pinned;
    asset.voxel_flags[index(columns - 1U, 0U)] |= waterlab::soft_body_voxel_pinned;

    std::vector<uint2> endpoints;
    const auto connect = [&](std::uint32_t a, std::uint32_t b) {
        endpoints.push_back(make_uint2(std::min(a, b), std::max(a, b)));
    };
    for (std::uint32_t row = 0U; row < rows; ++row) {
        for (std::uint32_t column = 0U; column < columns; ++column) {
            const std::uint32_t vertex = index(column, row);
            if (column + 1U < columns) connect(vertex, index(column + 1U, row));
            if (row + 1U < rows) connect(vertex, index(column, row + 1U));
            if (shear_springs && column + 1U < columns && row + 1U < rows) {
                connect(vertex, index(column + 1U, row + 1U));
                connect(index(column + 1U, row), index(column, row + 1U));
            }
            if (bend_springs && column + 2U < columns)
                connect(vertex, index(column + 2U, row));
            if (bend_springs && row + 2U < rows)
                connect(vertex, index(column, row + 2U));
        }
    }
    std::sort(endpoints.begin(), endpoints.end(), [](uint2 a, uint2 b) {
        return a.x < b.x || (a.x == b.x && a.y < b.y);
    });
    endpoints.erase(std::unique(endpoints.begin(), endpoints.end(),
        [](uint2 a, uint2 b) { return a.x == b.x && a.y == b.y; }),
        endpoints.end());
    for (uint2 vertices : endpoints) {
        asset.edges.push_back({vertices,
            distance(asset.rest_voxels[vertices.x], asset.rest_voxels[vertices.y])});
    }
    rebuild_adjacency(asset);
    for (std::uint32_t row = 0U; row + 1U < rows; ++row) {
        for (std::uint32_t column = 0U; column + 1U < columns; ++column) {
            const std::uint32_t a = index(column, row);
            const std::uint32_t b = index(column + 1U, row);
            const std::uint32_t c = index(column, row + 1U);
            const std::uint32_t d = index(column + 1U, row + 1U);
            asset.render_triangles.push_back(make_uint3(a, d, b));
            asset.render_triangles.push_back(make_uint3(a, c, d));
        }
    }
    waterlab::validate_soft_body_asset(asset);
    return asset;
}

waterlab::SoftBodyAsset make_rope_asset(std::uint32_t node_count,
    float spacing, float3 origin, float3 direction)
{
    if (node_count < 8U || node_count > 512U || !std::isfinite(spacing) ||
        spacing <= 0.0F) throw std::invalid_argument("invalid rope topology");
    const float direction_length = length(direction);
    if (!(direction_length > 1.0e-6F))
        throw std::invalid_argument("rope direction must be nonzero");
    const float3 axis = multiply(direction, 1.0F / direction_length);
    const float3 reference = std::abs(axis.y) < 0.9F
        ? make_float3(0.0F, 1.0F, 0.0F) : make_float3(0.0F, 0.0F, 1.0F);
    float3 z_axis = cross(axis, reference);
    z_axis = multiply(z_axis, 1.0F / length(z_axis));
    const float3 y_axis = cross(z_axis, axis);
    constexpr std::uint32_t sides = 6U;
    constexpr float two_pi = 6.28318530717958647692F;
    waterlab::SoftBodyAsset asset;
    asset.nominal_spacing = spacing;
    asset.voxel_radius = 0.36F * spacing;
    asset.voxel_flags.assign(node_count, waterlab::soft_body_voxel_surface);
    asset.voxel_flags.front() |= waterlab::soft_body_voxel_pinned;
    for (std::uint32_t node = 0U; node < node_count; ++node) {
        asset.rest_voxels.push_back(add(origin,
            multiply(axis, spacing * static_cast<float>(node))));
        if (node + 1U < node_count)
            asset.edges.push_back({make_uint2(node, node + 1U), spacing});
        if (node + 2U < node_count)
            asset.edges.push_back({make_uint2(node, node + 2U), 2.0F * spacing});
    }
    rebuild_adjacency(asset);
    const float tube_radius = 0.24F * spacing;
    for (std::uint32_t node = 0U; node < node_count; ++node) {
        for (std::uint32_t side = 0U; side < sides; ++side) {
            const float angle = two_pi * static_cast<float>(side) /
                static_cast<float>(sides);
            const float3 center = asset.rest_voxels[node];
            asset.render_positions.push_back(add(center, add(
                multiply(y_axis, tube_radius * std::cos(angle)),
                multiply(z_axis, tube_radius * std::sin(angle)))));
            asset.render_uvs.push_back(make_float2(
                static_cast<float>(node) / static_cast<float>(node_count - 1U),
                static_cast<float>(side) / static_cast<float>(sides)));
            asset.render_bindings.push_back({make_uint4(node, node, node, node),
                make_float4(1.0F, 0.0F, 0.0F, 0.0F)});
        }
    }
    for (std::uint32_t node = 0U; node + 1U < node_count; ++node) {
        for (std::uint32_t side = 0U; side < sides; ++side) {
            const std::uint32_t next = (side + 1U) % sides;
            const std::uint32_t a = node * sides + side;
            const std::uint32_t b = (node + 1U) * sides + side;
            const std::uint32_t c = (node + 1U) * sides + next;
            const std::uint32_t d = node * sides + next;
            asset.render_triangles.push_back(make_uint3(a, c, b));
            asset.render_triangles.push_back(make_uint3(a, d, c));
        }
    }
    for (std::uint32_t side = 1U; side + 1U < sides; ++side) {
        asset.render_triangles.push_back(make_uint3(0U, side + 1U, side));
        const std::uint32_t base = (node_count - 1U) * sides;
        asset.render_triangles.push_back(make_uint3(base, base + side,
            base + side + 1U));
    }
    waterlab::validate_soft_body_asset(asset);
    return asset;
}

} // namespace parallel_mater::physics::detail
