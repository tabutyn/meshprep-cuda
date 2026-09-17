// SPDX-License-Identifier: MIT
#include "../apps/water_lab/cloth.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

float length(float3 value)
{
    return std::sqrt(value.x * value.x + value.y * value.y + value.z * value.z);
}

float3 subtract(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

float triangle_double_area(float3 a, float3 b, float3 c)
{
    const float3 ab = subtract(b, a);
    const float3 ac = subtract(c, a);
    return length(make_float3(ab.y * ac.z - ab.z * ac.y,
        ab.z * ac.x - ab.x * ac.z, ab.x * ac.y - ab.y * ac.x));
}

bool same_bits(float3 a, float3 b)
{
    return std::bit_cast<std::uint32_t>(a.x) == std::bit_cast<std::uint32_t>(b.x) &&
        std::bit_cast<std::uint32_t>(a.y) == std::bit_cast<std::uint32_t>(b.y) &&
        std::bit_cast<std::uint32_t>(a.z) == std::bit_cast<std::uint32_t>(b.z);
}

bool near(float3 a, float3 b, float tolerance = 1.0e-6F)
{
    return length(subtract(a, b)) <= tolerance;
}

waterlab::SoftBodyState simulate(waterlab::SoftBodyCourse& cloth, std::uint32_t frames)
{
    for (std::uint32_t frame = 0U; frame < frames; ++frame) {
        (void)cloth.step(make_float3(0.0F, -9.81F, 0.0F));
    }
    waterlab::SoftBodyState state;
    cloth.capture_state(state);
    return state;
}

void host_test()
{
    waterlab::ClothGridOptions options;
    options.columns = 9U;
    options.rows = 7U;
    options.spacing = 0.08F;
    options.top_center = make_float3(0.25F, 1.0F, -0.5F);
    const auto cloth = waterlab::make_cloth_grid(options);
    waterlab::validate_soft_body_asset(cloth);

    require(cloth.rest_voxels.size() == 63U && cloth.render_positions.size() == 63U,
        "cloth grid has incorrect vertex count");
    require(cloth.render_triangles.size() == 96U,
        "cloth grid has incorrect triangle count");
    require(std::count_if(cloth.voxel_flags.begin(), cloth.voxel_flags.end(),
        [](std::uint32_t flags) {
            return (flags & waterlab::soft_body_voxel_pinned) != 0U;
        }) == 2, "cloth grid must pin exactly its top corners");
    for (std::uint32_t vertex = 0U; vertex < cloth.render_bindings.size(); ++vertex) {
        const auto binding = cloth.render_bindings[vertex];
        require(binding.voxels.x == vertex && binding.weights.x == 1.0F &&
            binding.weights.y == 0.0F && binding.weights.z == 0.0F &&
            binding.weights.w == 0.0F, "cloth render binding is not identity-bound");
    }

    const float3 translation = make_float3(1.0F, -2.0F, 3.0F);
    const auto translated = waterlab::translate_soft_body_asset(cloth, translation);
    require(near(translated.rest_voxels.front(), make_float3(
        cloth.rest_voxels.front().x + translation.x,
        cloth.rest_voxels.front().y + translation.y,
        cloth.rest_voxels.front().z + translation.z)),
        "soft-body translation did not move simulation vertices");
    require(near(translated.render_positions.back(), make_float3(
        cloth.render_positions.back().x + translation.x,
        cloth.render_positions.back().y + translation.y,
        cloth.render_positions.back().z + translation.z)),
        "soft-body translation did not move render vertices");
    require(translated.edges.front().rest_length == cloth.edges.front().rest_length &&
        translated.neighbor_offsets == cloth.neighbor_offsets,
        "rigid soft-body translation changed its constraint graph");

    const auto horizontal = waterlab::rotate_cloth_to_horizontal(cloth);
    for (const float3 point : horizontal.rest_voxels) {
        require(std::abs(point.y - options.top_center.z) < 1.0e-6F,
            "horizontal cloth does not lie in the XZ plane");
    }
    const uint3 first_triangle = horizontal.render_triangles.front();
    const float3 ab = subtract(horizontal.render_positions[first_triangle.y],
        horizontal.render_positions[first_triangle.x]);
    const float3 ac = subtract(horizontal.render_positions[first_triangle.z],
        horizontal.render_positions[first_triangle.x]);
    require(ab.z * ac.x - ab.x * ac.z > 0.0F,
        "horizontal cloth front normal does not point upward");

    const auto second = waterlab::translate_soft_body_asset(
        cloth, make_float3(2.0F, 0.0F, 0.0F));
    const auto merged = waterlab::merge_soft_body_assets(cloth, second);
    waterlab::validate_soft_body_asset(merged);
    const std::uint32_t voxel_offset = static_cast<std::uint32_t>(cloth.rest_voxels.size());
    const std::uint32_t render_offset =
        static_cast<std::uint32_t>(cloth.render_positions.size());
    require(merged.rest_voxels.size() == 2U * cloth.rest_voxels.size() &&
        merged.edges.size() == 2U * cloth.edges.size() &&
        merged.render_triangles.size() == 2U * cloth.render_triangles.size(),
        "soft-body merge changed component counts");
    const auto merged_binding = merged.render_bindings[render_offset];
    require(merged_binding.voxels.x == voxel_offset &&
        merged.render_triangles[cloth.render_triangles.size()].x ==
            cloth.render_triangles.front().x + render_offset,
        "soft-body merge did not remap render indices");
    const std::uint32_t first_second_edge = static_cast<std::uint32_t>(cloth.edges.size());
    require(merged.edges[first_second_edge].vertices.x >= voxel_offset,
        "soft-body merge did not remap constraint indices");

    auto incompatible = second;
    incompatible.voxel_radius *= 2.0F;
    bool rejected = false;
    try {
        (void)waterlab::merge_soft_body_assets(cloth, incompatible);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "soft-body merge accepted incompatible collision radii");

    const auto cross = waterlab::make_soft_cross();
    waterlab::validate_soft_body_asset(cross);

    const auto rope = waterlab::make_soft_rope(24U, 0.06F);
    waterlab::validate_soft_body_asset(rope);
    require(rope.rest_voxels.size() == 24U &&
            rope.render_positions.size() == 24U * 6U &&
            (rope.voxel_flags.front() & waterlab::soft_body_voxel_pinned) != 0U &&
            (rope.voxel_flags.back() & waterlab::soft_body_voxel_pinned) == 0U,
        "soft rope did not preserve its centerline, tube, and single anchor");
    require(cross.rest_voxels.size() == 1'155U &&
            !cross.render_triangles.empty() && !cross.edges.empty(),
        "procedural axle cross has the wrong three-layer plus-shape topology");
    require(std::count_if(cross.voxel_flags.begin(), cross.voxel_flags.end(),
        [](std::uint32_t flags) {
            return (flags & waterlab::soft_body_voxel_pinned) != 0U;
        }) == 63,
        "procedural axle cross does not pin its axle volume and four rim tips");
    std::uint32_t pinned_axle = 0U;
    std::uint32_t pinned_rim = 0U;
    for (std::size_t node = 0U; node < cross.rest_voxels.size(); ++node) {
        if ((cross.voxel_flags[node] & waterlab::soft_body_voxel_pinned) == 0U) continue;
        const auto point = cross.rest_voxels[node];
        const float radius = std::hypot(point.x, point.y);
        pinned_axle += radius < 0.20F ? 1U : 0U;
        pinned_rim += radius > 1.90F ? 1U : 0U;
    }
    require(pinned_axle == 27U && pinned_rim == 36U,
        "soft cross anchors are not split between the axle and outer rim");
    require(std::count_if(cross.voxel_flags.begin(), cross.voxel_flags.end(),
        [](std::uint32_t flags) {
            return (flags & waterlab::soft_body_voxel_rim_anchor) != 0U;
        }) == 36,
        "soft cross does not identify its inertial rim anchors");
    const auto [minimum_z, maximum_z] = std::minmax_element(
        cross.rest_voxels.begin(), cross.rest_voxels.end(),
        [](float3 a, float3 b) { return a.z < b.z; });
    require(maximum_z->z - minimum_z->z >= 0.19F,
        "procedural axle cross has no physical depth");
    for (std::size_t node = 0U; node + 1U < cross.neighbor_offsets.size(); ++node) {
        require(cross.neighbor_offsets[node + 1U] > cross.neighbor_offsets[node],
            "procedural axle cross contains a disconnected node");
    }
}

void gpu_test()
{
    waterlab::ClothGridOptions grid;
    grid.columns = 12U;
    grid.rows = 10U;
    grid.spacing = 0.05F;
    grid.top_center = make_float3(0.0F, 0.8F, 0.0F);
    const auto asset = waterlab::make_cloth_grid(grid);

    waterlab::SoftBodyOptions solver;
    solver.instance_count = 1U;
    solver.solver_substeps = 4U;
    solver.spring_solver_iterations = 8U;
    solver.spring_stiffness = 900.0F;
    solver.spring_damping_ratio = 0.9F;
    solver.velocity_damping = 0.9F;
    solver.break_strain = 10.0F;
    solver.use_course_layout = false;
    solver.require_1000_voxels = false;
    solver.instance_origins[0] = make_float3(0.0F, 0.0F, 0.0F);
    waterlab::SoftBodyCourse cloth(asset, solver);

    constexpr std::uint32_t frames = 120U;
    const auto first = simulate(cloth, frames);
    require(first.statistics.finite_failure_count == 0U,
        "cloth developed a non-finite state while sagging");
    require(first.statistics.broken_edge_count == 0U,
        "cloth broke an edge during the finite-sag fixture");

    const std::uint32_t top_left = 0U;
    const std::uint32_t top_right = grid.columns - 1U;
    require(same_bits(first.positions[top_left], asset.rest_voxels[top_left]) &&
        same_bits(first.positions[top_right], asset.rest_voxels[top_right]),
        "cloth pinned top corners moved");
    const std::uint32_t bottom_center =
        (grid.rows - 1U) * grid.columns + grid.columns / 2U;
    require(first.positions[bottom_center].y < asset.rest_voxels[bottom_center].y - 0.005F,
        "cloth did not measurably sag under gravity");

    for (const uint3 triangle : asset.render_triangles) {
        require(triangle_double_area(first.positions[triangle.x], first.positions[triangle.y],
            first.positions[triangle.z]) > 1.0e-8F,
            "cloth produced a degenerate render triangle");
    }

    cloth.reset();
    waterlab::SoftBodyState reset;
    cloth.capture_state(reset);
    require(reset.statistics.frame_index == 0U, "cloth reset retained frame history");
    for (std::uint32_t vertex = 0U; vertex < reset.positions.size(); ++vertex) {
        require(same_bits(reset.positions[vertex], asset.rest_voxels[vertex]),
            "cloth reset did not exactly restore its rest pose");
    }
    const auto replay = simulate(cloth, frames);
    require(replay.positions.size() == first.positions.size(),
        "cloth replay changed vertex count");
    for (std::uint32_t vertex = 0U; vertex < first.positions.size(); ++vertex) {
        require(same_bits(replay.positions[vertex], first.positions[vertex]),
            "cloth replay was not bit-identical after reset");
    }
}

} // namespace

int main()
{
    try {
        host_test();
        int devices{};
        const cudaError_t status = cudaGetDeviceCount(&devices);
        if (status != cudaSuccess || devices == 0) {
            std::fprintf(stderr, "SKIP: CUDA device unavailable\n");
            return 77;
        }
        gpu_test();
        std::puts("cloth tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "cloth test failure: %s\n", error.what());
        return 1;
    }
}
