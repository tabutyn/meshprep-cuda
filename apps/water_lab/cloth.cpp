// SPDX-License-Identifier: MIT
#include "cloth.hpp"
#include "water_lab.hpp"

#include <vector_functions.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace waterlab {
namespace {

[[nodiscard]] bool finite(float3 value)
{
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

[[nodiscard]] float distance(float3 a, float3 b)
{
    return std::hypot(std::hypot(b.x - a.x, b.y - a.y), b.z - a.z);
}

[[nodiscard]] float dot(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

[[nodiscard]] float3 cross(float3 a, float3 b)
{
    return make_float3(a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

[[nodiscard]] float3 apply(const SoftBodyRigidTransform& transform, float3 point)
{
    return make_float3(
        transform.translation.x + point.x * transform.x_axis.x +
            point.y * transform.y_axis.x + point.z * transform.z_axis.x,
        transform.translation.y + point.x * transform.x_axis.y +
            point.y * transform.y_axis.y + point.z * transform.z_axis.y,
        transform.translation.z + point.x * transform.x_axis.z +
            point.y * transform.y_axis.z + point.z * transform.z_axis.z);
}

void validate_transform(const SoftBodyRigidTransform& transform)
{
    constexpr float tolerance = 1.0e-4F;
    if (!finite(transform.x_axis) || !finite(transform.y_axis) ||
        !finite(transform.z_axis) || !finite(transform.translation) ||
        std::abs(dot(transform.x_axis, transform.x_axis) - 1.0F) > tolerance ||
        std::abs(dot(transform.y_axis, transform.y_axis) - 1.0F) > tolerance ||
        std::abs(dot(transform.z_axis, transform.z_axis) - 1.0F) > tolerance ||
        std::abs(dot(transform.x_axis, transform.y_axis)) > tolerance ||
        std::abs(dot(transform.x_axis, transform.z_axis)) > tolerance ||
        std::abs(dot(transform.y_axis, transform.z_axis)) > tolerance ||
        std::abs(dot(cross(transform.x_axis, transform.y_axis), transform.z_axis) - 1.0F) >
            tolerance) {
        throw std::invalid_argument("soft-body transform must be right-handed and orthonormal");
    }
}

[[nodiscard]] std::uint32_t checked_sum(
    std::size_t first, std::size_t second, const char* label)
{
    constexpr std::size_t maximum = std::numeric_limits<std::uint32_t>::max();
    if (first > maximum || second > maximum || first > maximum - second) {
        throw std::invalid_argument(std::string("merged soft-body ") + label + " exceeds uint32");
    }
    return static_cast<std::uint32_t>(first + second);
}

void rebuild_adjacency(SoftBodyAsset& asset)
{
    std::sort(asset.edges.begin(), asset.edges.end(), [](const auto& a, const auto& b) {
        return a.vertices.x < b.vertices.x ||
            (a.vertices.x == b.vertices.x && a.vertices.y < b.vertices.y);
    });
    std::vector<std::vector<SoftBodyNeighbor>> adjacency(asset.rest_voxels.size());
    for (std::uint32_t edge = 0U; edge < asset.edges.size(); ++edge) {
        const uint2 vertices = asset.edges[edge].vertices;
        adjacency[vertices.x].push_back({vertices.y, edge});
        adjacency[vertices.y].push_back({vertices.x, edge});
    }
    asset.neighbor_offsets.clear();
    asset.neighbors.clear();
    asset.neighbor_offsets.reserve(asset.rest_voxels.size() + 1U);
    asset.neighbor_offsets.push_back(0U);
    for (auto& neighbors : adjacency) {
        std::sort(neighbors.begin(), neighbors.end(), [](const auto& a, const auto& b) {
            return a.vertex < b.vertex;
        });
        asset.neighbors.insert(asset.neighbors.end(), neighbors.begin(), neighbors.end());
        if (asset.neighbors.size() > std::numeric_limits<std::uint32_t>::max()) {
            throw std::invalid_argument("merged soft-body neighbor count exceeds uint32");
        }
        asset.neighbor_offsets.push_back(static_cast<std::uint32_t>(asset.neighbors.size()));
    }
}

} // namespace

SoftBodyAsset make_cloth_grid(const ClothGridOptions& options)
{
    if (options.columns < 2U || options.rows < 2U ||
        !std::isfinite(options.spacing) || options.spacing <= 0.0F ||
        !finite(options.top_center) ||
        options.columns > std::numeric_limits<std::uint32_t>::max() / options.rows) {
        throw std::invalid_argument("cloth grid requires finite spacing and at least 2x2 vertices");
    }

    const std::uint32_t vertex_count = options.columns * options.rows;
    const auto index = [&](std::uint32_t column, std::uint32_t row) {
        return row * options.columns + column;
    };

    SoftBodyAsset asset;
    asset.nominal_spacing = options.spacing;
    asset.voxel_radius = options.spacing * 0.45F;
    asset.rest_voxels.reserve(vertex_count);
    asset.voxel_flags.assign(vertex_count, soft_body_voxel_surface);
    asset.render_positions.reserve(vertex_count);
    asset.render_uvs.reserve(vertex_count);
    asset.render_bindings.reserve(vertex_count);

    const float left = -0.5F * static_cast<float>(options.columns - 1U) * options.spacing;
    for (std::uint32_t row = 0U; row < options.rows; ++row) {
        for (std::uint32_t column = 0U; column < options.columns; ++column) {
            const float3 position = make_float3(
                options.top_center.x + left + static_cast<float>(column) * options.spacing,
                options.top_center.y - static_cast<float>(row) * options.spacing,
                options.top_center.z);
            const std::uint32_t vertex = index(column, row);
            asset.rest_voxels.push_back(position);
            asset.render_positions.push_back(position);
            asset.render_uvs.push_back(make_float2(
                static_cast<float>(column) / static_cast<float>(options.columns - 1U),
                1.0F - static_cast<float>(row) / static_cast<float>(options.rows - 1U)));
            asset.render_bindings.push_back({make_uint4(vertex, vertex, vertex, vertex),
                make_float4(1.0F, 0.0F, 0.0F, 0.0F)});
        }
    }
    asset.voxel_flags[index(0U, 0U)] |= soft_body_voxel_pinned;
    asset.voxel_flags[index(options.columns - 1U, 0U)] |= soft_body_voxel_pinned;

    std::vector<uint2> endpoints;
    const auto connect = [&](std::uint32_t a, std::uint32_t b) {
        endpoints.push_back(make_uint2(std::min(a, b), std::max(a, b)));
    };
    for (std::uint32_t row = 0U; row < options.rows; ++row) {
        for (std::uint32_t column = 0U; column < options.columns; ++column) {
            const std::uint32_t vertex = index(column, row);
            if (column + 1U < options.columns) connect(vertex, index(column + 1U, row));
            if (row + 1U < options.rows) connect(vertex, index(column, row + 1U));
            if (options.shear_springs && column + 1U < options.columns &&
                row + 1U < options.rows) {
                connect(vertex, index(column + 1U, row + 1U));
                connect(index(column + 1U, row), index(column, row + 1U));
            }
            if (options.bend_springs && column + 2U < options.columns) {
                connect(vertex, index(column + 2U, row));
            }
            if (options.bend_springs && row + 2U < options.rows) {
                connect(vertex, index(column, row + 2U));
            }
        }
    }
    std::sort(endpoints.begin(), endpoints.end(), [](uint2 a, uint2 b) {
        return a.x < b.x || (a.x == b.x && a.y < b.y);
    });
    endpoints.erase(std::unique(endpoints.begin(), endpoints.end(), [](uint2 a, uint2 b) {
        return a.x == b.x && a.y == b.y;
    }), endpoints.end());

    asset.edges.reserve(endpoints.size());
    std::vector<std::vector<SoftBodyNeighbor>> adjacency(vertex_count);
    for (std::uint32_t edge = 0U; edge < endpoints.size(); ++edge) {
        const uint2 vertices = endpoints[edge];
        asset.edges.push_back({vertices,
            distance(asset.rest_voxels[vertices.x], asset.rest_voxels[vertices.y])});
        adjacency[vertices.x].push_back({vertices.y, edge});
        adjacency[vertices.y].push_back({vertices.x, edge});
    }
    asset.neighbor_offsets.reserve(static_cast<std::size_t>(vertex_count) + 1U);
    asset.neighbor_offsets.push_back(0U);
    for (auto& neighbors : adjacency) {
        std::sort(neighbors.begin(), neighbors.end(), [](const auto& a, const auto& b) {
            return a.vertex < b.vertex;
        });
        asset.neighbors.insert(asset.neighbors.end(), neighbors.begin(), neighbors.end());
        asset.neighbor_offsets.push_back(static_cast<std::uint32_t>(asset.neighbors.size()));
    }

    asset.render_triangles.reserve(
        2U * static_cast<std::size_t>(options.columns - 1U) * (options.rows - 1U));
    for (std::uint32_t row = 0U; row + 1U < options.rows; ++row) {
        for (std::uint32_t column = 0U; column + 1U < options.columns; ++column) {
            const std::uint32_t top_left = index(column, row);
            const std::uint32_t top_right = index(column + 1U, row);
            const std::uint32_t bottom_left = index(column, row + 1U);
            const std::uint32_t bottom_right = index(column + 1U, row + 1U);
            asset.render_triangles.push_back(
                make_uint3(top_left, bottom_right, top_right));
            asset.render_triangles.push_back(
                make_uint3(top_left, bottom_left, bottom_right));
        }
    }

    validate_soft_body_asset(asset);
    return asset;
}

SoftBodyAsset make_soft_sphere(
    std::uint32_t surface_nodes, std::uint32_t interior_nodes, float radius)
{
    if (surface_nodes < 64U || interior_nodes < 64U ||
        !std::isfinite(radius) || radius <= 0.0F) {
        throw std::invalid_argument("soft sphere requires finite radius and interior fill");
    }
    constexpr float pi = 3.14159265358979323846F;
    constexpr float golden_angle = 2.39996322972865332223F;
    SoftBodyAsset asset;
    asset.nominal_spacing = 0.075F;
    asset.voxel_radius = 0.03375F;
    asset.file_flags |= soft_body_asset_free_body;
    asset.rest_voxels.reserve(surface_nodes + interior_nodes);
    asset.voxel_flags.reserve(surface_nodes + interior_nodes);
    for (std::uint32_t index = 0U; index < surface_nodes; ++index) {
        const float y = 1.0F - 2.0F * (static_cast<float>(index) + 0.5F) /
            static_cast<float>(surface_nodes);
        const float planar = sqrtf(1.0F - y*y);
        const float angle = golden_angle * static_cast<float>(index);
        asset.rest_voxels.push_back(make_float3(
            radius * planar * cosf(angle), radius * y,
            radius * planar * sinf(angle)));
        asset.voxel_flags.push_back(soft_body_voxel_surface);
    }
    struct Candidate { float radius_squared; float3 point; };
    std::vector<Candidate> interior;
    const float spacing = asset.nominal_spacing;
    for (int layer = -8; layer <= 8; ++layer) {
        for (int row = -8; row <= 8; ++row) {
            for (int column = -8; column <= 8; ++column) {
                const float3 point = make_float3(
                    spacing * (static_cast<float>(column) + 0.5F * (row & 1)),
                    spacing * 0.81649658F * static_cast<float>(layer),
                    spacing * 0.86602540F * (static_cast<float>(row) +
                        0.33333333F * (layer & 1)));
                const float d2 = dot(point, point);
                if (d2 < (radius - spacing * 0.70F) *
                        (radius - spacing * 0.70F)) interior.push_back({d2, point});
            }
        }
    }
    std::sort(interior.begin(), interior.end(), [](const Candidate& a,
        const Candidate& b) {
        if (a.radius_squared != b.radius_squared)
            return a.radius_squared < b.radius_squared;
        if (a.point.y != b.point.y) return a.point.y < b.point.y;
        if (a.point.z != b.point.z) return a.point.z < b.point.z;
        return a.point.x < b.point.x;
    });
    if (interior.size() < interior_nodes)
        throw std::invalid_argument("soft sphere interior grid is too sparse");
    for (std::uint32_t index = 0U; index < interior_nodes; ++index) {
        asset.rest_voxels.push_back(interior[index].point);
        asset.voxel_flags.push_back(0U);
    }

    const std::uint32_t count = surface_nodes + interior_nodes;
    std::vector<uint2> endpoints;
    endpoints.reserve(13U * count);
    for (std::uint32_t node = 0U; node < count; ++node) {
        std::vector<std::pair<float, std::uint32_t>> nearest;
        nearest.reserve(count - 1U);
        for (std::uint32_t other = 0U; other < count; ++other) {
            if (other == node) continue;
            const float d = distance(asset.rest_voxels[node],
                asset.rest_voxels[other]);
            nearest.emplace_back(d, other);
        }
        std::partial_sort(nearest.begin(), nearest.begin() + 12U, nearest.end());
        for (std::uint32_t slot = 0U; slot < 12U; ++slot) {
            const std::uint32_t other = nearest[slot].second;
            endpoints.push_back(make_uint2(std::min(node, other),
                std::max(node, other)));
        }
        // Every interior voxel has a structural bridge to the surface, and
        // every surface voxel reaches the interior even at a sparse pole.
        const std::uint32_t opposite_first = node < surface_nodes
            ? surface_nodes : 0U;
        const std::uint32_t opposite_last = node < surface_nodes
            ? count : surface_nodes;
        float best = INFINITY;
        std::uint32_t bridge = opposite_first;
        for (std::uint32_t other = opposite_first; other < opposite_last; ++other) {
            const float d = distance(asset.rest_voxels[node],
                asset.rest_voxels[other]);
            if (d < best || (d == best && other < bridge)) {
                best = d;
                bridge = other;
            }
        }
        endpoints.push_back(make_uint2(std::min(node, bridge),
            std::max(node, bridge)));
    }
    std::sort(endpoints.begin(), endpoints.end(), [](uint2 a, uint2 b) {
        return a.x < b.x || (a.x == b.x && a.y < b.y);
    });
    endpoints.erase(std::unique(endpoints.begin(), endpoints.end(), [](uint2 a,
        uint2 b) { return a.x == b.x && a.y == b.y; }), endpoints.end());
    for (const uint2 edge : endpoints) {
        asset.edges.push_back({edge,
            distance(asset.rest_voxels[edge.x], asset.rest_voxels[edge.y])});
    }
    rebuild_adjacency(asset);

    const HostSurfaceMesh render = make_geodesic_sphere(10U, radius);
    asset.render_positions = render.positions;
    asset.render_triangles = render.triangles;
    asset.render_uvs.reserve(render.positions.size());
    asset.render_bindings.reserve(render.positions.size());
    for (const float3 point : render.positions) {
        const float latitude = acosf(std::clamp(point.y / radius, -1.0F, 1.0F));
        asset.render_uvs.push_back(make_float2(
            0.5F + atan2f(point.z, point.x) / (2.0F * pi), latitude / pi));
        std::vector<std::pair<float, std::uint32_t>> nearest;
        nearest.reserve(surface_nodes);
        for (std::uint32_t node = 0U; node < surface_nodes; ++node)
            nearest.emplace_back(distance(point, asset.rest_voxels[node]), node);
        std::partial_sort(nearest.begin(), nearest.begin() + 4U, nearest.end());
        asset.render_bindings.push_back({make_uint4(
            nearest[0].second, nearest[1].second,
            nearest[2].second, nearest[3].second),
            make_float4(1.0F, 0.0F, 0.0F, 0.0F)});
    }
    validate_soft_body_asset(asset);
    return asset;
}

SoftBodyAsset make_soft_rope(std::uint32_t node_count, float spacing)
{
    if (node_count < 8U || node_count > 512U ||
        !std::isfinite(spacing) || spacing <= 0.0F) {
        throw std::invalid_argument("soft rope requires 8-512 nodes and finite spacing");
    }
    constexpr std::uint32_t sides = 6U;
    constexpr float two_pi = 6.28318530717958647692F;
    SoftBodyAsset asset;
    asset.nominal_spacing = spacing;
    asset.voxel_radius = 0.36F * spacing;
    asset.rest_voxels.reserve(node_count);
    asset.voxel_flags.assign(node_count, soft_body_voxel_surface);
    asset.voxel_flags.front() |= soft_body_voxel_pinned;
    for (std::uint32_t node = 0U; node < node_count; ++node)
        asset.rest_voxels.push_back(make_float3(spacing * node, 0.0F, 0.0F));

    for (std::uint32_t node = 0U; node < node_count; ++node) {
        if (node + 1U < node_count) {
            asset.edges.push_back({make_uint2(node, node + 1U), spacing});
        }
        if (node + 2U < node_count) {
            asset.edges.push_back({make_uint2(node, node + 2U), 2.0F * spacing});
        }
    }
    rebuild_adjacency(asset);

    const float tube_radius = 0.24F * spacing;
    asset.render_positions.reserve(static_cast<std::size_t>(node_count) * sides);
    asset.render_uvs.reserve(static_cast<std::size_t>(node_count) * sides);
    asset.render_bindings.reserve(static_cast<std::size_t>(node_count) * sides);
    for (std::uint32_t node = 0U; node < node_count; ++node) {
        for (std::uint32_t side = 0U; side < sides; ++side) {
            const float angle = two_pi * static_cast<float>(side) /
                static_cast<float>(sides);
            asset.render_positions.push_back(make_float3(spacing * node,
                tube_radius * std::cos(angle), tube_radius * std::sin(angle)));
            asset.render_uvs.push_back(make_float2(
                static_cast<float>(node) / static_cast<float>(node_count - 1U),
                static_cast<float>(side) / static_cast<float>(sides)));
            asset.render_bindings.push_back({make_uint4(node, node, node, node),
                make_float4(1.0F, 0.0F, 0.0F, 0.0F)});
        }
    }
    for (std::uint32_t node = 0U; node + 1U < node_count; ++node) {
        for (std::uint32_t side = 0U; side < sides; ++side) {
            const std::uint32_t next_side = (side + 1U) % sides;
            const std::uint32_t a = node * sides + side;
            const std::uint32_t b = (node + 1U) * sides + side;
            const std::uint32_t c = (node + 1U) * sides + next_side;
            const std::uint32_t d = node * sides + next_side;
            asset.render_triangles.push_back(make_uint3(a, c, b));
            asset.render_triangles.push_back(make_uint3(a, d, c));
        }
    }
    // End caps use existing ring vertices, so the render mesh stays watertight
    // without introducing a non-physical center binding.
    for (std::uint32_t side = 1U; side + 1U < sides; ++side) {
        asset.render_triangles.push_back(make_uint3(0U, side + 1U, side));
        const std::uint32_t base = (node_count - 1U) * sides;
        asset.render_triangles.push_back(
            make_uint3(base, base + side, base + side + 1U));
    }
    validate_soft_body_asset(asset);
    return asset;
}

SoftBodyAsset make_y_rope_cage(std::uint32_t node_count,float total_length)
{
    if (node_count<16U || node_count>512U || !std::isfinite(total_length) ||
        total_length<=0.0F)
        throw std::invalid_argument("Y rope requires 16-512 nodes and finite length");
    const std::uint32_t trunk_nodes=std::max(8U,node_count/2U);
    const std::uint32_t branch_nodes=std::max(5U,(node_count-trunk_nodes)/2U+1U);
    const float trunk_length=0.5F*total_length;
    const float branch_length=0.5F*total_length;
    const float trunk_spacing=trunk_length/static_cast<float>(trunk_nodes-1U);
    const float branch_spacing=branch_length/static_cast<float>(branch_nodes-1U);
    const float inverse_direction=1.0F/std::sqrt(1.0F+0.55F*0.55F);
    const float3 primary_direction=make_float3(inverse_direction,0.0F,
        0.55F*inverse_direction);
    const float3 secondary_direction=make_float3(inverse_direction,0.0F,
        -0.55F*inverse_direction);
    const float3 junction=make_float3(trunk_length,0.0F,0.0F);
    const auto vector_add=[](float3 a,float3 b) {
        return make_float3(a.x+b.x,a.y+b.y,a.z+b.z);
    };
    const auto vector_scale=[](float3 value,float scale) {
        return make_float3(value.x*scale,value.y*scale,value.z*scale);
    };
    const auto vector_length=[](float3 value) {
        return std::sqrt(value.x*value.x+value.y*value.y+value.z*value.z);
    };

    SoftBodyAsset asset;
    asset.nominal_spacing=std::min(trunk_spacing,branch_spacing);
    asset.voxel_radius=0.36F*asset.nominal_spacing;
    const auto add_node=[&](float3 point) {
        const std::uint32_t id=static_cast<std::uint32_t>(asset.rest_voxels.size());
        asset.rest_voxels.push_back(point);
        asset.voxel_flags.push_back(soft_body_voxel_surface);
        return id;
    };
    const auto add_edge=[&](std::uint32_t a,std::uint32_t b) {
        asset.edges.push_back({make_uint2(std::min(a,b),std::max(a,b)),
            distance(asset.rest_voxels[a],asset.rest_voxels[b])});
    };
    for (std::uint32_t node=0U;node<trunk_nodes;++node)
        add_node(make_float3(trunk_spacing*node,0.0F,0.0F));
    asset.voxel_flags.front()|=soft_body_voxel_pinned;
    for (std::uint32_t node=0U;node+1U<trunk_nodes;++node) add_edge(node,node+1U);
    for (std::uint32_t node=0U;node+2U<trunk_nodes;++node) add_edge(node,node+2U);

    const std::uint32_t secondary_first=static_cast<std::uint32_t>(
        asset.rest_voxels.size());
    for (std::uint32_t node=1U;node<branch_nodes;++node)
        add_node(vector_add(junction,
            vector_scale(secondary_direction,branch_spacing*node)));
    add_edge(trunk_nodes-1U,secondary_first);
    for (std::uint32_t node=secondary_first;
         node+1U<asset.rest_voxels.size();++node) add_edge(node,node+1U);

    const float3 cage_center=vector_add(junction,
        vector_scale(secondary_direction,branch_length));
    const std::uint32_t cage_first=static_cast<std::uint32_t>(asset.rest_voxels.size());
    constexpr float phi=1.61803398875F;
    constexpr float inv_phi=1.0F/phi;
    const float3 raw[20]{
        {-1,-1,-1},{-1,-1,1},{-1,1,-1},{-1,1,1},
        {1,-1,-1},{1,-1,1},{1,1,-1},{1,1,1},
        {0,-inv_phi,-phi},{0,-inv_phi,phi},{0,inv_phi,-phi},{0,inv_phi,phi},
        {-inv_phi,-phi,0},{-inv_phi,phi,0},{inv_phi,-phi,0},{inv_phi,phi,0},
        {-phi,0,-inv_phi},{-phi,0,inv_phi},{phi,0,-inv_phi},{phi,0,inv_phi}};
    constexpr float cage_radius=0.42F;
    for (float3 point:raw)
        add_node(vector_add(cage_center,
            vector_scale(point,cage_radius/vector_length(point))));
    std::vector<uint2> cage_edges;
    for (std::uint32_t a=0U;a<20U;++a) {
        std::vector<std::pair<float,std::uint32_t>> nearest;
        for (std::uint32_t b=0U;b<20U;++b) if (a!=b)
            nearest.emplace_back(distance(raw[a],raw[b]),b);
        std::partial_sort(nearest.begin(),nearest.begin()+3U,nearest.end());
        for (std::uint32_t slot=0U;slot<3U;++slot)
            cage_edges.push_back(make_uint2(std::min(a,nearest[slot].second),
                std::max(a,nearest[slot].second)));
    }
    std::sort(cage_edges.begin(),cage_edges.end(),[](uint2 a,uint2 b) {
        return a.x<b.x || (a.x==b.x && a.y<b.y);
    });
    cage_edges.erase(std::unique(cage_edges.begin(),cage_edges.end(),
        [](uint2 a,uint2 b){return a.x==b.x && a.y==b.y;}),cage_edges.end());
    for (uint2 edge:cage_edges) add_edge(cage_first+edge.x,cage_first+edge.y);
    const std::uint32_t secondary_end=cage_first-1U;
    for (std::uint32_t link=0U;link<3U;++link)
        add_edge(secondary_end,cage_first+link);

    const std::uint32_t primary_first=static_cast<std::uint32_t>(
        asset.rest_voxels.size());
    for (std::uint32_t node=1U;node<branch_nodes;++node)
        add_node(vector_add(junction,
            vector_scale(primary_direction,branch_spacing*node)));
    add_edge(trunk_nodes-1U,primary_first);
    for (std::uint32_t node=primary_first;
         node+1U<asset.rest_voxels.size();++node) add_edge(node,node+1U);

    std::sort(asset.edges.begin(),asset.edges.end(),[](const SoftBodyEdge& a,
        const SoftBodyEdge& b) {
        return a.vertices.x<b.vertices.x ||
            (a.vertices.x==b.vertices.x && a.vertices.y<b.vertices.y);
    });
    asset.edges.erase(std::unique(asset.edges.begin(),asset.edges.end(),
        [](const SoftBodyEdge& a,const SoftBodyEdge& b) {
            return a.vertices.x==b.vertices.x && a.vertices.y==b.vertices.y;
        }),asset.edges.end());
    rebuild_adjacency(asset);

    const HostSurfaceMesh glass=make_geodesic_sphere(6U,0.31F);
    for (const float3 point:glass.positions) {
        const float3 placed=vector_add(cage_center,point);
        asset.render_positions.push_back(placed);
        asset.render_uvs.push_back(make_float2(
            0.5F+atan2f(point.z,point.x)/(2.0F*3.14159265358979323846F),
            acosf(std::clamp(point.y/0.31F,-1.0F,1.0F))/
                3.14159265358979323846F));
        float best=INFINITY;
        std::uint32_t owner=cage_first;
        for (std::uint32_t node=0U;node<20U;++node) {
            const float d=distance(placed,asset.rest_voxels[cage_first+node]);
            if (d<best) { best=d; owner=cage_first+node; }
        }
        asset.render_bindings.push_back({make_uint4(owner,owner,owner,owner),
            make_float4(1,0,0,0)});
    }
    asset.render_triangles=glass.triangles;
    validate_soft_body_asset(asset);
    return asset;
}

SoftBodyAsset make_rope_bridge(std::uint32_t columns, std::uint32_t rows)
{
    if (columns < 2U || columns > 16U || rows < 2U || rows > 64U)
        throw std::invalid_argument("rope bridge dimensions must be 2..16 by 2..64");
    constexpr std::uint32_t corners = 4U;
    SoftBodyAsset asset;
    asset.nominal_spacing = rope_bridge_gap;
    asset.voxel_radius = 0.035F;
    asset.rest_voxels.reserve(columns * rows * corners);
    asset.voxel_flags.reserve(columns * rows * corners);
    asset.render_positions.reserve(columns * rows * corners);
    asset.render_uvs.reserve(columns * rows * corners);
    asset.render_bindings.reserve(columns * rows * corners);
    const auto tile = [columns](std::uint32_t column, std::uint32_t row) {
        return (row * columns + column) * corners;
    };
    const float width=columns*rope_bridge_tile_size+(columns-1U)*rope_bridge_gap;
    const float bridge_length=rows*rope_bridge_tile_size+(rows-1U)*rope_bridge_gap;
    const float first_x = -0.5F * width +
        0.5F * rope_bridge_tile_size;
    const float first_z = 0.5F * bridge_length -
        0.5F * rope_bridge_tile_size;
    constexpr float half = 0.5F * rope_bridge_tile_size;
    for (std::uint32_t row = 0U; row < rows; ++row) {
        for (std::uint32_t column = 0U; column < columns; ++column) {
            const float center_x = first_x + column * rope_bridge_pitch;
            const float center_z = first_z - row * rope_bridge_pitch;
            const float3 points[corners]{
                {center_x-half, rope_bridge_deck_y, center_z+half},
                {center_x+half, rope_bridge_deck_y, center_z+half},
                {center_x+half, rope_bridge_deck_y, center_z-half},
                {center_x-half, rope_bridge_deck_y, center_z-half}};
            for (std::uint32_t corner = 0U; corner < corners; ++corner) {
                const std::uint32_t node = tile(column, row) + corner;
                asset.rest_voxels.push_back(points[corner]);
                asset.voxel_flags.push_back(soft_body_voxel_surface |
                    ((row == 0U || row + 1U == rows)
                        ? soft_body_voxel_pinned : 0U));
                asset.render_positions.push_back(points[corner]);
                asset.render_uvs.push_back(make_float2(
                    static_cast<float>(column) + (corner == 1U || corner == 2U),
                    static_cast<float>(row) + (corner >= 2U)));
                asset.render_bindings.push_back({make_uint4(node,node,node,node),
                    make_float4(1.0F,0.0F,0.0F,0.0F)});
            }
            const std::uint32_t base = tile(column, row);
            asset.render_triangles.push_back(make_uint3(base, base+2U, base+1U));
            asset.render_triangles.push_back(make_uint3(base, base+3U, base+2U));
        }
    }
    std::vector<uint2> endpoints;
    const auto connect = [&](std::uint32_t a, std::uint32_t b) {
        endpoints.push_back(make_uint2(std::min(a,b), std::max(a,b)));
    };
    for (std::uint32_t row = 0U; row < rows; ++row) {
        for (std::uint32_t column = 0U; column < columns; ++column) {
            const std::uint32_t base = tile(column, row);
            // Six in-tile braces keep each square nearly rigid.
            connect(base, base+1U); connect(base+1U, base+2U);
            connect(base+2U, base+3U); connect(base, base+3U);
            connect(base, base+2U); connect(base+1U, base+3U);
            if (column + 1U < columns) {
                const std::uint32_t right = tile(column+1U, row);
                connect(base+1U, right); connect(base+2U, right+3U);
            }
            if (row + 1U < rows) {
                const std::uint32_t next = tile(column, row+1U);
                connect(base+2U, next+1U); connect(base+3U, next);
            }
        }
    }
    std::sort(endpoints.begin(), endpoints.end(), [](uint2 a, uint2 b) {
        return a.x < b.x || (a.x == b.x && a.y < b.y);
    });
    endpoints.erase(std::unique(endpoints.begin(), endpoints.end(), [](uint2 a, uint2 b) {
        return a.x == b.x && a.y == b.y;
    }), endpoints.end());
    for (const uint2 edge : endpoints)
        asset.edges.push_back({edge,
            distance(asset.rest_voxels[edge.x], asset.rest_voxels[edge.y])});
    rebuild_adjacency(asset);
    validate_soft_body_asset(asset);
    return asset;
}

SoftBodyAsset make_soft_cross(
    std::uint32_t span, std::uint32_t arm_width, float spacing)
{
    if (span < 7U || (span & 1U) == 0U || arm_width < 3U ||
        (arm_width & 1U) == 0U || arm_width >= span ||
        !std::isfinite(spacing) || spacing <= 0.0F) {
        throw std::invalid_argument("soft cross requires odd finite dimensions");
    }
    constexpr std::uint32_t depth = 3U;
    const std::uint32_t middle = span / 2U;
    const std::uint32_t half_arm = arm_width / 2U;
    const std::uint32_t invalid = std::numeric_limits<std::uint32_t>::max();
    std::vector<std::uint32_t> ids(
        static_cast<std::size_t>(depth) * span * span, invalid);
    const auto slot = [span](std::uint32_t x, std::uint32_t y, std::uint32_t z) {
        return (static_cast<std::size_t>(z) * span + y) * span + x;
    };
    const auto occupied = [middle, half_arm](std::uint32_t x, std::uint32_t y) {
        const int dx = static_cast<int>(x) - static_cast<int>(middle);
        const int dy = static_cast<int>(y) - static_cast<int>(middle);
        return std::abs(dx) <= static_cast<int>(half_arm) ||
            std::abs(dy) <= static_cast<int>(half_arm);
    };
    SoftBodyAsset asset;
    asset.nominal_spacing = spacing;
    asset.voxel_radius = 0.38F * spacing;
    for (std::uint32_t z = 0U; z < depth; ++z) {
        for (std::uint32_t y = 0U; y < span; ++y) {
            for (std::uint32_t x = 0U; x < span; ++x) {
                if (!occupied(x, y)) continue;
                const int dx = static_cast<int>(x) - static_cast<int>(middle);
                const int dy = static_cast<int>(y) - static_cast<int>(middle);
                const int dz = static_cast<int>(z) - 1;
                const std::uint32_t id =
                    static_cast<std::uint32_t>(asset.rest_voxels.size());
                ids[slot(x, y, z)] = id;
                const float3 point = make_float3(
                    dx * spacing, -dy * spacing, dz * spacing);
                asset.rest_voxels.push_back(point);
                const bool xy_boundary = x == 0U || x + 1U == span ||
                    y == 0U || y + 1U == span ||
                    !occupied(x == 0U ? x : x - 1U, y) ||
                    !occupied(x + 1U == span ? x : x + 1U, y) ||
                    !occupied(x, y == 0U ? y : y - 1U) ||
                    !occupied(x, y + 1U == span ? y : y + 1U);
                std::uint32_t flags = (z == 0U || z + 1U == depth || xy_boundary)
                    ? soft_body_voxel_surface : 0U;
                const bool axle_anchor =
                    std::abs(dx) <= 1 && std::abs(dy) <= 1;
                const bool rim_anchor =
                    (std::abs(dx) == static_cast<int>(middle) && std::abs(dy) <= 1) ||
                    (std::abs(dy) == static_cast<int>(middle) && std::abs(dx) <= 1);
                if (axle_anchor || rim_anchor)
                    flags |= soft_body_voxel_pinned;
                if (rim_anchor) flags |= soft_body_voxel_rim_anchor;
                asset.voxel_flags.push_back(flags);
            }
        }
    }
    std::vector<std::uint32_t> render_ids(ids.size(), invalid);
    for (std::uint32_t z = 0U; z < depth; ++z) {
        for (std::uint32_t y = 0U; y < span; ++y) {
            for (std::uint32_t x = 0U; x < span; ++x) {
                const std::uint32_t voxel = ids[slot(x, y, z)];
                if (voxel == invalid ||
                    (asset.voxel_flags[voxel] & soft_body_voxel_surface) == 0U)
                    continue;
                render_ids[slot(x, y, z)] =
                    static_cast<std::uint32_t>(asset.render_positions.size());
                asset.render_positions.push_back(asset.rest_voxels[voxel]);
                asset.render_uvs.push_back(make_float2(
                    static_cast<float>(x) / static_cast<float>(span - 1U),
                    static_cast<float>(y) / static_cast<float>(span - 1U)));
                asset.render_bindings.push_back({make_uint4(
                    voxel, voxel, voxel, voxel),
                    make_float4(1.0F, 0.0F, 0.0F, 0.0F)});
            }
        }
    }
    std::vector<uint2> endpoints;
    const auto connect = [&](std::uint32_t ax, std::uint32_t ay, std::uint32_t az,
                             std::uint32_t bx, std::uint32_t by, std::uint32_t bz) {
        if (ax >= span || ay >= span || az >= depth ||
            bx >= span || by >= span || bz >= depth) return;
        const std::uint32_t a = ids[slot(ax, ay, az)];
        const std::uint32_t b = ids[slot(bx, by, bz)];
        if (a == invalid || b == invalid) return;
        endpoints.push_back(make_uint2(std::min(a, b), std::max(a, b)));
    };
    for (std::uint32_t z = 0U; z < depth; ++z) {
        for (std::uint32_t y = 0U; y < span; ++y) {
            for (std::uint32_t x = 0U; x < span; ++x) {
                if (ids[slot(x, y, z)] == invalid) continue;
                if (x + 1U < span) connect(x, y, z, x + 1U, y, z);
                if (y + 1U < span) connect(x, y, z, x, y + 1U, z);
                if (x + 1U < span && y + 1U < span) {
                    connect(x, y, z, x + 1U, y + 1U, z);
                    connect(x + 1U, y, z, x, y + 1U, z);
                }
                if (x + 2U < span) connect(x, y, z, x + 2U, y, z);
                if (y + 2U < span) connect(x, y, z, x, y + 2U, z);
                if (z + 1U < depth) {
                    connect(x, y, z, x, y, z + 1U);
                    if (x + 1U < span) connect(x, y, z, x + 1U, y, z + 1U);
                    if (x > 0U) connect(x, y, z, x - 1U, y, z + 1U);
                    if (y + 1U < span) connect(x, y, z, x, y + 1U, z + 1U);
                    if (y > 0U) connect(x, y, z, x, y - 1U, z + 1U);
                }
            }
        }
    }
    std::sort(endpoints.begin(), endpoints.end(), [](uint2 a, uint2 b) {
        return a.x < b.x || (a.x == b.x && a.y < b.y);
    });
    endpoints.erase(std::unique(endpoints.begin(), endpoints.end(), [](uint2 a, uint2 b) {
        return a.x == b.x && a.y == b.y;
    }), endpoints.end());
    for (const uint2 edge : endpoints) {
        asset.edges.push_back({edge,
            distance(asset.rest_voxels[edge.x], asset.rest_voxels[edge.y])});
    }
    rebuild_adjacency(asset);
    for (std::uint32_t y = 0U; y + 1U < span; ++y) {
        for (std::uint32_t x = 0U; x + 1U < span; ++x) {
            for (const std::uint32_t z : {0U, depth - 1U}) {
                const std::uint32_t a = render_ids[slot(x, y, z)];
                const std::uint32_t b = render_ids[slot(x + 1U, y, z)];
                const std::uint32_t c = render_ids[slot(x, y + 1U, z)];
                const std::uint32_t d = render_ids[slot(x + 1U, y + 1U, z)];
                if (a == invalid || b == invalid || c == invalid || d == invalid)
                    continue;
                if (z == 0U) {
                    asset.render_triangles.push_back(make_uint3(a, d, c));
                    asset.render_triangles.push_back(make_uint3(a, b, d));
                } else {
                    asset.render_triangles.push_back(make_uint3(a, c, d));
                    asset.render_triangles.push_back(make_uint3(a, d, b));
                }
            }
        }
    }
    const auto add_side = [&](std::uint32_t a0, std::uint32_t b0,
                              std::uint32_t a1, std::uint32_t b1, bool reverse) {
        if (a0 == invalid || b0 == invalid || a1 == invalid || b1 == invalid) return;
        if (reverse) {
            asset.render_triangles.push_back(make_uint3(a0, b1, a1));
            asset.render_triangles.push_back(make_uint3(a0, b0, b1));
        } else {
            asset.render_triangles.push_back(make_uint3(a0, a1, b1));
            asset.render_triangles.push_back(make_uint3(a0, b1, b0));
        }
    };
    for (std::uint32_t z = 0U; z + 1U < depth; ++z) {
        for (std::uint32_t y = 0U; y < span; ++y) {
            for (std::uint32_t x = 0U; x + 1U < span; ++x) {
                if (!occupied(x, y) || !occupied(x + 1U, y)) continue;
                const bool top = y == 0U ||
                    (!occupied(x, y - 1U) && !occupied(x + 1U, y - 1U));
                const bool bottom = y + 1U == span ||
                    (!occupied(x, y + 1U) && !occupied(x + 1U, y + 1U));
                if (top) add_side(render_ids[slot(x,y,z)], render_ids[slot(x+1U,y,z)],
                    render_ids[slot(x,y,z+1U)], render_ids[slot(x+1U,y,z+1U)], false);
                if (bottom) add_side(render_ids[slot(x,y,z)], render_ids[slot(x+1U,y,z)],
                    render_ids[slot(x,y,z+1U)], render_ids[slot(x+1U,y,z+1U)], true);
            }
        }
        for (std::uint32_t x = 0U; x < span; ++x) {
            for (std::uint32_t y = 0U; y + 1U < span; ++y) {
                if (!occupied(x, y) || !occupied(x, y + 1U)) continue;
                const bool left = x == 0U ||
                    (!occupied(x - 1U, y) && !occupied(x - 1U, y + 1U));
                const bool right = x + 1U == span ||
                    (!occupied(x + 1U, y) && !occupied(x + 1U, y + 1U));
                if (left) add_side(render_ids[slot(x,y,z)], render_ids[slot(x,y+1U,z)],
                    render_ids[slot(x,y,z+1U)], render_ids[slot(x,y+1U,z+1U)], true);
                if (right) add_side(render_ids[slot(x,y,z)], render_ids[slot(x,y+1U,z)],
                    render_ids[slot(x,y,z+1U)], render_ids[slot(x,y+1U,z+1U)], false);
            }
        }
    }
    validate_soft_body_asset(asset);
    return asset;
}

SoftBodyAsset transform_soft_body_asset(
    SoftBodyAsset asset, const SoftBodyRigidTransform& transform)
{
    validate_soft_body_asset(asset);
    validate_transform(transform);
    for (float3& point : asset.rest_voxels) point = apply(transform, point);
    for (float3& point : asset.render_positions) point = apply(transform, point);
    validate_soft_body_asset(asset);
    return asset;
}

SoftBodyAsset translate_soft_body_asset(SoftBodyAsset asset, float3 translation)
{
    SoftBodyRigidTransform transform;
    transform.translation = translation;
    return transform_soft_body_asset(std::move(asset), transform);
}

SoftBodyAsset rotate_cloth_to_horizontal(SoftBodyAsset asset)
{
    SoftBodyRigidTransform transform;
    transform.y_axis = make_float3(0.0F, 0.0F, -1.0F);
    transform.z_axis = make_float3(0.0F, 1.0F, 0.0F);
    return transform_soft_body_asset(std::move(asset), transform);
}

SoftBodyAsset shape_cloth_catch_basin(SoftBodyAsset asset)
{
    validate_soft_body_asset(asset);
    const auto [minimum_x, maximum_x] = std::minmax_element(
        asset.rest_voxels.begin(), asset.rest_voxels.end(),
        [](float3 a, float3 b) { return a.x < b.x; });
    const auto [minimum_z, maximum_z] = std::minmax_element(
        asset.rest_voxels.begin(), asset.rest_voxels.end(),
        [](float3 a, float3 b) { return a.z < b.z; });
    const float center_x = 0.5F * (minimum_x->x + maximum_x->x);
    const float center_z = 0.5F * (minimum_z->z + maximum_z->z);
    const float half_x = 0.5F * (maximum_x->x - minimum_x->x);
    const float half_z = 0.5F * (maximum_z->z - minimum_z->z);
    const auto vertical_shape = [&](float3 point) {
        const float u = (point.x - center_x) / half_x;
        const float v = (point.z - center_z) / half_z;
        const float edge = std::max(std::fabs(u), std::fabs(v));
        return -0.10F * std::max(0.0F, 1.0F - edge * edge);
    };
    for (std::size_t node = 0U; node < asset.rest_voxels.size(); ++node) {
        float3& point = asset.rest_voxels[node];
        point.y += vertical_shape(point);
    }
    for (float3& point : asset.render_positions)
        point.y += vertical_shape(point);
    for (SoftBodyEdge& edge : asset.edges) edge.rest_length = distance(
        asset.rest_voxels[edge.vertices.x], asset.rest_voxels[edge.vertices.y]);
    validate_soft_body_asset(asset);
    return asset;
}

SoftBodyAsset merge_soft_body_assets(
    const SoftBodyAsset& first, const SoftBodyAsset& second)
{
    validate_soft_body_asset(first);
    validate_soft_body_asset(second);
    constexpr float tolerance = 1.0e-6F;
    if (std::abs(first.nominal_spacing - second.nominal_spacing) > tolerance ||
        std::abs(first.voxel_radius - second.voxel_radius) > tolerance ||
        first.file_flags != second.file_flags) {
        throw std::invalid_argument(
            "soft-body merge requires matching spacing, radius, and file conventions");
    }
    const std::uint32_t voxel_count = checked_sum(
        first.rest_voxels.size(), second.rest_voxels.size(), "voxel count");
    const std::uint32_t render_vertex_count = checked_sum(
        first.render_positions.size(), second.render_positions.size(), "render vertex count");
    (void)checked_sum(first.edges.size(), second.edges.size(), "edge count");
    (void)checked_sum(first.render_triangles.size(), second.render_triangles.size(),
        "render triangle count");

    SoftBodyAsset merged;
    merged.nominal_spacing = first.nominal_spacing;
    merged.voxel_radius = first.voxel_radius;
    merged.relaxation_iterations = std::max(
        first.relaxation_iterations, second.relaxation_iterations);
    merged.file_flags = first.file_flags;
    merged.rest_voxels = first.rest_voxels;
    merged.rest_voxels.insert(merged.rest_voxels.end(),
        second.rest_voxels.begin(), second.rest_voxels.end());
    merged.voxel_flags = first.voxel_flags;
    merged.voxel_flags.insert(merged.voxel_flags.end(),
        second.voxel_flags.begin(), second.voxel_flags.end());

    merged.edges = first.edges;
    merged.edges.reserve(first.edges.size() + second.edges.size());
    for (const SoftBodyEdge edge : second.edges) {
        merged.edges.push_back({make_uint2(
            edge.vertices.x + static_cast<std::uint32_t>(first.rest_voxels.size()),
            edge.vertices.y + static_cast<std::uint32_t>(first.rest_voxels.size())),
            edge.rest_length});
    }
    rebuild_adjacency(merged);

    merged.render_positions = first.render_positions;
    merged.render_positions.insert(merged.render_positions.end(),
        second.render_positions.begin(), second.render_positions.end());
    merged.render_uvs = first.render_uvs;
    merged.render_uvs.insert(merged.render_uvs.end(),
        second.render_uvs.begin(), second.render_uvs.end());
    merged.render_bindings = first.render_bindings;
    merged.render_bindings.reserve(first.render_bindings.size() + second.render_bindings.size());
    const auto offset_binding = [offset =
        static_cast<std::uint32_t>(first.rest_voxels.size())](SoftBodyBinding binding) {
        binding.voxels = make_uint4(binding.voxels.x + offset,
            binding.voxels.y + offset, binding.voxels.z + offset,
            binding.voxels.w + offset);
        return binding;
    };
    for (const SoftBodyBinding binding : second.render_bindings) {
        merged.render_bindings.push_back(offset_binding(binding));
    }
    merged.render_triangles = first.render_triangles;
    merged.render_triangles.reserve(
        first.render_triangles.size() + second.render_triangles.size());
    const std::uint32_t render_offset =
        static_cast<std::uint32_t>(first.render_positions.size());
    for (const uint3 triangle : second.render_triangles) {
        merged.render_triangles.push_back(make_uint3(triangle.x + render_offset,
            triangle.y + render_offset, triangle.z + render_offset));
    }

    if (merged.rest_voxels.size() != voxel_count ||
        merged.render_positions.size() != render_vertex_count) {
        throw std::logic_error("soft-body merge count mismatch");
    }
    validate_soft_body_asset(merged);
    return merged;
}

} // namespace waterlab
