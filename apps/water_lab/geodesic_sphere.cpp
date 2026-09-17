// SPDX-License-Identifier: MIT
#include "water_lab.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace waterlab {
namespace {

float3 normalize(float3 value, float radius)
{
    const float length = std::sqrt(value.x * value.x + value.y * value.y + value.z * value.z);
    return make_float3(
        radius * value.x / length, radius * value.y / length, radius * value.z / length);
}

struct QuantizedPoint {
    std::int64_t x;
    std::int64_t y;
    std::int64_t z;

    bool operator==(const QuantizedPoint&) const = default;
};

struct PointHash {
    std::size_t operator()(const QuantizedPoint& point) const noexcept
    {
        std::size_t seed = std::hash<std::int64_t>{}(point.x);
        seed ^= std::hash<std::int64_t>{}(point.y) + 0x9e3779b9U + (seed << 6U) + (seed >> 2U);
        seed ^= std::hash<std::int64_t>{}(point.z) + 0x9e3779b9U + (seed << 6U) + (seed >> 2U);
        return seed;
    }
};

QuantizedPoint quantize(float3 point)
{
    constexpr double scale = 100000000.0;
    return {
        std::llround(static_cast<double>(point.x) * scale),
        std::llround(static_cast<double>(point.y) * scale),
        std::llround(static_cast<double>(point.z) * scale),
    };
}

float3 barycentric_coordinates(float3 point, float3 a, float3 b, float3 c)
{
    const float3 v0 = make_float3(b.x - a.x, b.y - a.y, b.z - a.z);
    const float3 v1 = make_float3(c.x - a.x, c.y - a.y, c.z - a.z);
    const float3 v2 = make_float3(point.x - a.x, point.y - a.y, point.z - a.z);
    const float d00 = v0.x * v0.x + v0.y * v0.y + v0.z * v0.z;
    const float d01 = v0.x * v1.x + v0.y * v1.y + v0.z * v1.z;
    const float d11 = v1.x * v1.x + v1.y * v1.y + v1.z * v1.z;
    const float d20 = v2.x * v0.x + v2.y * v0.y + v2.z * v0.z;
    const float d21 = v2.x * v1.x + v2.y * v1.y + v2.z * v1.z;
    const float denominator = d00 * d11 - d01 * d01;
    if (!(denominator > 0.0F)) return make_float3(-1.0F, -1.0F, -1.0F);
    const float v = (d11 * d20 - d01 * d21) / denominator;
    const float w = (d00 * d21 - d01 * d20) / denominator;
    return make_float3(1.0F - v - w, v, w);
}

} // namespace

HostSurfaceMesh make_geodesic_sphere(std::uint32_t frequency, float radius)
{
    if (frequency == 0 || radius <= 0.0F) {
        throw std::invalid_argument("sphere frequency and radius must be positive");
    }
    constexpr float phi = 1.6180339887498948482F;
    std::array<float3, 12> base_positions{
        make_float3(-1, phi, 0), make_float3(1, phi, 0), make_float3(-1, -phi, 0),
        make_float3(1, -phi, 0), make_float3(0, -1, phi), make_float3(0, 1, phi),
        make_float3(0, -1, -phi), make_float3(0, 1, -phi), make_float3(phi, 0, -1),
        make_float3(phi, 0, 1), make_float3(-phi, 0, -1), make_float3(-phi, 0, 1),
    };
    for (float3& point : base_positions) point = normalize(point, radius);
    constexpr std::array<std::array<std::uint32_t, 3>, 20> faces{{
        {0, 11, 5}, {0, 5, 1}, {0, 1, 7}, {0, 7, 10}, {0, 10, 11},
        {1, 5, 9}, {5, 11, 4}, {11, 10, 2}, {10, 7, 6}, {7, 1, 8},
        {3, 9, 4}, {3, 4, 2}, {3, 2, 6}, {3, 6, 8}, {3, 8, 9},
        {4, 9, 5}, {2, 4, 11}, {6, 2, 10}, {8, 6, 7}, {9, 8, 1},
    }};

    HostSurfaceMesh mesh;
    mesh.positions.reserve(static_cast<std::size_t>(10U) * frequency * frequency + 2U);
    mesh.triangles.reserve(static_cast<std::size_t>(20U) * frequency * frequency);
    std::unordered_map<QuantizedPoint, std::uint32_t, PointHash> welded;

    auto add_vertex = [&](float3 point) {
        point = normalize(point, radius);
        const QuantizedPoint key = quantize(point);
        const auto found = welded.find(key);
        if (found != welded.end()) return found->second;
        const auto index = static_cast<std::uint32_t>(mesh.positions.size());
        mesh.positions.push_back(point);
        welded.emplace(key, index);
        return index;
    };

    for (const auto& face : faces) {
        const float3 a = base_positions[face[0]];
        const float3 b = base_positions[face[1]];
        const float3 c = base_positions[face[2]];
        std::vector<std::vector<std::uint32_t>> grid(frequency + 1U);
        for (std::uint32_t i = 0; i <= frequency; ++i) {
            grid[i].reserve(frequency - i + 1U);
            for (std::uint32_t j = 0; j <= frequency - i; ++j) {
                const float wa = static_cast<float>(frequency - i - j);
                const float wb = static_cast<float>(i);
                const float wc = static_cast<float>(j);
                const float inverse = 1.0F / static_cast<float>(frequency);
                grid[i].push_back(add_vertex(make_float3(
                    (wa * a.x + wb * b.x + wc * c.x) * inverse,
                    (wa * a.y + wb * b.y + wc * c.y) * inverse,
                    (wa * a.z + wb * b.z + wc * c.z) * inverse)));
            }
        }
        for (std::uint32_t i = 0; i < frequency; ++i) {
            for (std::uint32_t j = 0; j < frequency - i; ++j) {
                mesh.triangles.push_back(make_uint3(grid[i][j], grid[i + 1U][j], grid[i][j + 1U]));
                if (j + 1U < frequency - i) {
                    mesh.triangles.push_back(make_uint3(
                        grid[i + 1U][j], grid[i + 1U][j + 1U], grid[i][j + 1U]));
                }
            }
        }
    }

    std::vector<std::pair<std::uint32_t, std::uint32_t>> edges;
    edges.reserve(mesh.triangles.size() * 3U);
    for (const uint3 triangle : mesh.triangles) {
        for (const auto& [first, second] : std::array{
                 std::pair{triangle.x, triangle.y},
                 std::pair{triangle.y, triangle.z},
                 std::pair{triangle.z, triangle.x}}) {
            edges.emplace_back(std::min(first, second), std::max(first, second));
        }
    }
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());

    mesh.neighbor_offsets.assign(mesh.positions.size() + 1U, 0U);
    for (const auto& [a, b] : edges) {
        ++mesh.neighbor_offsets[a + 1U];
        ++mesh.neighbor_offsets[b + 1U];
    }
    for (std::size_t i = 1; i < mesh.neighbor_offsets.size(); ++i) {
        mesh.neighbor_offsets[i] += mesh.neighbor_offsets[i - 1U];
    }
    mesh.neighbors.resize(edges.size() * 2U);
    mesh.rest_lengths.resize(edges.size() * 2U);
    std::vector<std::uint32_t> cursors = mesh.neighbor_offsets;
    for (const auto& [a, b] : edges) {
        const float3 delta = make_float3(
            mesh.positions[b].x - mesh.positions[a].x,
            mesh.positions[b].y - mesh.positions[a].y,
            mesh.positions[b].z - mesh.positions[a].z);
        const float length = std::sqrt(delta.x * delta.x + delta.y * delta.y + delta.z * delta.z);
        const std::uint32_t ab = cursors[a]++;
        const std::uint32_t ba = cursors[b]++;
        mesh.neighbors[ab] = b;
        mesh.neighbors[ba] = a;
        mesh.rest_lengths[ab] = length;
        mesh.rest_lengths[ba] = length;
    }
    for (std::uint32_t vertex = 0; vertex < mesh.positions.size(); ++vertex) {
        const auto begin = mesh.neighbor_offsets[vertex];
        const auto end = mesh.neighbor_offsets[vertex + 1U];
        std::vector<std::pair<std::uint32_t, float>> local;
        local.reserve(end - begin);
        for (std::uint32_t i = begin; i < end; ++i) {
            local.emplace_back(mesh.neighbors[i], mesh.rest_lengths[i]);
        }
        std::sort(local.begin(), local.end());
        for (std::uint32_t i = begin; i < end; ++i) {
            mesh.neighbors[i] = local[i - begin].first;
            mesh.rest_lengths[i] = local[i - begin].second;
        }
    }
    const std::size_t expected_vertices = static_cast<std::size_t>(10U) * frequency * frequency + 2U;
    const std::size_t expected_triangles = static_cast<std::size_t>(20U) * frequency * frequency;
    const std::size_t expected_edges = static_cast<std::size_t>(30U) * frequency * frequency;
    if (mesh.positions.size() != expected_vertices || mesh.triangles.size() != expected_triangles ||
        edges.size() != expected_edges || mesh.neighbor_offsets.back() != mesh.neighbors.size()) {
        throw std::runtime_error("geodesic sphere welding invariant failed");
    }
    for (std::uint32_t vertex = 0; vertex < mesh.positions.size(); ++vertex) {
        const auto begin = mesh.neighbor_offsets[vertex];
        const auto end = mesh.neighbor_offsets[vertex + 1U];
        for (std::uint32_t edge = begin; edge < end; ++edge) {
            const std::uint32_t neighbor = mesh.neighbors[edge];
            if (neighbor >= mesh.positions.size() || neighbor == vertex ||
                !std::binary_search(
                    mesh.neighbors.begin() + mesh.neighbor_offsets[neighbor],
                    mesh.neighbors.begin() + mesh.neighbor_offsets[neighbor + 1U],
                    vertex) ||
                !(mesh.rest_lengths[edge] > 0.0F)) {
                throw std::runtime_error("geodesic sphere adjacency invariant failed");
            }
        }
    }
    return mesh;
}

std::vector<SurfaceVertexEmbedding> make_surface_embedding(
    const HostSurfaceMesh& source,
    const std::vector<float3>& target_positions)
{
    if (source.positions.empty() || source.triangles.empty()) {
        throw std::invalid_argument("surface embedding source is empty");
    }
    std::vector<SurfaceVertexEmbedding> embedding;
    embedding.reserve(target_positions.size());
    constexpr float inside_tolerance = 2.0e-4F;
    for (const float3 target : target_positions) {
        const float direction_length = std::sqrt(
            target.x * target.x + target.y * target.y + target.z * target.z);
        if (!(direction_length > 0.0F)) {
            throw std::invalid_argument("surface embedding target is at the origin");
        }
        const float3 direction = make_float3(
            target.x / direction_length, target.y / direction_length, target.z / direction_length);
        bool found = false;
        SurfaceVertexEmbedding selected{};
        for (const uint3 triangle : source.triangles) {
            const float3 a = source.positions[triangle.x];
            const float3 b = source.positions[triangle.y];
            const float3 c = source.positions[triangle.z];
            const float3 ab = make_float3(b.x - a.x, b.y - a.y, b.z - a.z);
            const float3 ac = make_float3(c.x - a.x, c.y - a.y, c.z - a.z);
            const float3 normal = make_float3(
                ab.y * ac.z - ab.z * ac.y,
                ab.z * ac.x - ab.x * ac.z,
                ab.x * ac.y - ab.y * ac.x);
            const float denominator =
                normal.x * direction.x + normal.y * direction.y + normal.z * direction.z;
            if (std::fabs(denominator) <= 1.0e-12F) continue;
            const float distance =
                (normal.x * a.x + normal.y * a.y + normal.z * a.z) / denominator;
            if (!(distance > 0.0F)) continue;
            const float3 intersection = make_float3(
                direction.x * distance, direction.y * distance, direction.z * distance);
            float3 weights = barycentric_coordinates(intersection, a, b, c);
            if (weights.x < -inside_tolerance || weights.y < -inside_tolerance ||
                weights.z < -inside_tolerance) continue;
            weights.x = std::max(0.0F, weights.x);
            weights.y = std::max(0.0F, weights.y);
            weights.z = std::max(0.0F, weights.z);
            const float sum = weights.x + weights.y + weights.z;
            if (!(sum > 0.0F)) continue;
            selected.source_vertices = triangle;
            selected.barycentric = make_float3(
                weights.x / sum, weights.y / sum, weights.z / sum);
            found = true;
            break;
        }
        if (!found) {
            throw std::runtime_error("surface embedding could not locate a source triangle");
        }
        embedding.push_back(selected);
    }
    return embedding;
}

} // namespace waterlab
