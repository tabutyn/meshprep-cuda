// SPDX-License-Identifier: MIT
#include "../apps/water_lab/soft_body.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

float distance(float3 a, float3 b)
{
    const float x = b.x - a.x;
    const float y = b.y - a.y;
    const float z = b.z - a.z;
    return std::sqrt(x*x + y*y + z*z);
}

bool finite_float3(float3 value)
{
    return std::isfinite(value.x) && std::isfinite(value.y) &&
        std::isfinite(value.z);
}

void require_finite_state(
    const waterlab::SoftBodyState& state, const char* message)
{
    require(state.positions.size() == state.velocities.size(), message);
    for (std::size_t voxel = 0U; voxel < state.positions.size(); ++voxel) {
        if (!finite_float3(state.positions[voxel]) ||
            !finite_float3(state.velocities[voxel])) {
            throw std::runtime_error(message);
        }
    }
    require(state.statistics.finite_failure_count == 0U, message);
}

waterlab::SoftBodyAsset make_fixture()
{
    using namespace waterlab;
    SoftBodyAsset asset;
    asset.nominal_spacing = 0.1F;
    asset.voxel_radius = 0.045F;
    asset.relaxation_iterations = 12U;
    asset.rest_voxels = {
        make_float3(0.0F, 0.10F, 0.0F),
        make_float3(0.1F, 0.10F, 0.0F),
        make_float3(0.0F, 0.20F, 0.0F),
        make_float3(0.0F, 0.10F, 0.1F)};
    asset.voxel_flags = {
        soft_body_voxel_surface | soft_body_voxel_pinned,
        soft_body_voxel_surface,
        soft_body_voxel_surface,
        soft_body_voxel_surface};
    for (std::uint32_t a = 0U; a < 4U; ++a) {
        for (std::uint32_t b = a + 1U; b < 4U; ++b) {
            asset.edges.push_back({make_uint2(a, b),
                distance(asset.rest_voxels[a], asset.rest_voxels[b])});
        }
    }
    asset.neighbor_offsets.push_back(0U);
    for (std::uint32_t vertex = 0U; vertex < 4U; ++vertex) {
        for (std::uint32_t neighbor = 0U; neighbor < 4U; ++neighbor) {
            if (neighbor == vertex) continue;
            const std::uint32_t a = std::min(vertex, neighbor);
            const std::uint32_t b = std::max(vertex, neighbor);
            std::uint32_t edge_id{};
            for (; edge_id < asset.edges.size(); ++edge_id) {
                if (asset.edges[edge_id].vertices.x == a &&
                    asset.edges[edge_id].vertices.y == b) break;
            }
            asset.neighbors.push_back({neighbor, edge_id});
        }
        asset.neighbor_offsets.push_back(
            static_cast<std::uint32_t>(asset.neighbors.size()));
    }
    asset.render_positions = {
        asset.rest_voxels[0], asset.rest_voxels[1], asset.rest_voxels[2]};
    asset.render_uvs = {
        make_float2(0.0F, 0.0F), make_float2(1.0F, 0.0F), make_float2(0.0F, 1.0F)};
    for (std::uint32_t vertex = 0U; vertex < 3U; ++vertex) {
        asset.render_bindings.push_back({make_uint4(vertex, vertex, vertex, vertex),
            make_float4(1.0F, 0.0F, 0.0F, 0.0F)});
    }
    asset.render_triangles = {make_uint3(0U, 1U, 2U)};
    return asset;
}

void host_tests()
{
    auto asset = make_fixture();
    waterlab::validate_soft_body_asset(asset);
    const std::string path = "/tmp/parallel-mater-soft-body-roundtrip.msb";
    waterlab::save_soft_body_asset(asset, path);
    const auto loaded = waterlab::load_soft_body_asset(path);
    std::remove(path.c_str());
    require(loaded.rest_voxels.size() == 4U && loaded.edges.size() == 6U &&
        loaded.neighbors.size() == 12U && loaded.render_positions.size() == 3U,
        "soft-body binary round trip changed counts");
    require(loaded.voxel_flags[0] ==
        (waterlab::soft_body_voxel_surface | waterlab::soft_body_voxel_pinned),
        "soft-body binary round trip changed flags");
    require(std::abs(loaded.voxel_radius - asset.voxel_radius) < 1.0e-7F,
        "soft-body binary round trip changed derived radius");

    auto invalid = asset;
    invalid.neighbors[1] = invalid.neighbors[0];
    bool rejected = false;
    try {
        waterlab::validate_soft_body_asset(invalid);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "soft-body validation accepted duplicate CSR neighbor");

#ifdef PARALLEL_MATER_SOFT_BODY_TEST_ASSET
    const auto generated = waterlab::load_soft_body_asset(PARALLEL_MATER_SOFT_BODY_TEST_ASSET);
    require(generated.rest_voxels.size() == 1'000U &&
        generated.render_positions.size() == 931U &&
        generated.render_triangles.size() == 1'632U,
        "Blender-generated soft-body asset has unexpected counts");
    const auto surface = std::count_if(generated.voxel_flags.begin(),
        generated.voxel_flags.end(), [](std::uint32_t flags) {
            return (flags & waterlab::soft_body_voxel_surface) != 0U;
        });
    require(surface == 500 && generated.relaxation_iterations == 24U,
        "Blender-generated soft-body metadata did not survive loading");
#endif
}

#ifdef PARALLEL_MATER_SOFT_BODY_TEST_ASSET
struct ImpactResult {
    float peak_deformation{};
    float peak_edge_strain{};
    float minimum_triangle_area_ratio{INFINITY};
    float maximum_triangle_area_ratio{};
    float maximum_render_edge_ratio{};
    float maximum_abnormal_triangle_fraction{};
    float minimum_active_triangle_area_ratio{INFINITY};
    float maximum_active_triangle_area_ratio{};
    float maximum_active_render_edge_ratio{};
    float minimum_active_triangle_fraction{1.0F};
    float average_gpu_ms{};
    std::uint32_t broken_edges{};
};

float triangle_double_area(float3 a, float3 b, float3 c)
{
    const float3 ab = make_float3(b.x-a.x, b.y-a.y, b.z-a.z);
    const float3 ac = make_float3(c.x-a.x, c.y-a.y, c.z-a.z);
    const float3 cross = make_float3(ab.y*ac.z-ab.z*ac.y,
        ab.z*ac.x-ab.x*ac.z, ab.x*ac.y-ab.y*ac.x);
    return std::sqrt(cross.x*cross.x + cross.y*cross.y + cross.z*cross.z);
}

void audit_render_geometry(const waterlab::SoftBodyAsset& asset,
    const waterlab::SoftBodyState& state, const std::vector<float3>& current,
    ImpactResult& result)
{
    require(current.size() == asset.render_positions.size(),
        "soft-body runtime render vertex count changed");
    std::uint32_t abnormal{};
    std::uint32_t active_count{};
    for (std::size_t triangle_index = 0U;
         triangle_index < asset.render_triangles.size(); ++triangle_index) {
        const uint3 triangle = asset.render_triangles[triangle_index];
        const float rest_area = triangle_double_area(asset.render_positions[triangle.x],
            asset.render_positions[triangle.y], asset.render_positions[triangle.z]);
        const float current_area = triangle_double_area(current[triangle.x],
            current[triangle.y], current[triangle.z]);
        if (!(rest_area > 1.0e-10F)) continue;
        const float ratio = current_area / rest_area;
        result.minimum_triangle_area_ratio = std::min(
            result.minimum_triangle_area_ratio, ratio);
        result.maximum_triangle_area_ratio = std::max(
            result.maximum_triangle_area_ratio, ratio);
        const bool active = triangle_index >= state.active_render_triangles.size() ||
            state.active_render_triangles[triangle_index] != 0U;
        if (active) {
            ++active_count;
            result.minimum_active_triangle_area_ratio = std::min(
                result.minimum_active_triangle_area_ratio, ratio);
            result.maximum_active_triangle_area_ratio = std::max(
                result.maximum_active_triangle_area_ratio, ratio);
        }
        abnormal += ratio < 0.5F || ratio > 2.0F;
        const std::uint32_t ids[3]{triangle.x, triangle.y, triangle.z};
        for (unsigned edge = 0U; edge < 3U; ++edge) {
            const float rest_length = distance(asset.render_positions[ids[edge]],
                asset.render_positions[ids[(edge + 1U) % 3U]]);
            const float current_length = distance(current[ids[edge]],
                current[ids[(edge + 1U) % 3U]]);
            if (rest_length > 1.0e-8F) {
                const float edge_ratio = current_length / rest_length;
                result.maximum_render_edge_ratio = std::max(
                    result.maximum_render_edge_ratio, edge_ratio);
                if (active) result.maximum_active_render_edge_ratio = std::max(
                    result.maximum_active_render_edge_ratio, edge_ratio);
            }
        }
    }
    result.maximum_abnormal_triangle_fraction = std::max(
        result.maximum_abnormal_triangle_fraction,
        static_cast<float>(abnormal) / static_cast<float>(asset.render_triangles.size()));
    result.minimum_active_triangle_fraction = std::min(
        result.minimum_active_triangle_fraction,
        static_cast<float>(active_count) /
            static_cast<float>(asset.render_triangles.size()));
}

ImpactResult run_impact(float impulse_per_voxel)
{
    const auto asset = waterlab::load_soft_body_asset(PARALLEL_MATER_SOFT_BODY_TEST_ASSET);
    waterlab::SoftBodyOptions options;
    options.instance_count = 1U;
    waterlab::SoftBodyCourse body(asset, options);
    const float3 gravity = make_float3(0.0F, -7.2F, 0.0F);
    for (std::uint32_t frame = 0U; frame < 240U; ++frame) (void)body.step(gravity);
    require(body.statistics().broken_edge_count == 0U,
        "soft-body impact fixture fractured while settling");
    waterlab::SoftBodyState reference;
    body.capture_state(reference);
    require_finite_state(reference,
        "soft-body impact fixture produced non-finite settling state");

    float minimum_x = asset.rest_voxels.front().x;
    for (float3 position : asset.rest_voxels) minimum_x = std::min(minimum_x, position.x);
    std::vector<std::uint32_t> impact_voxels;
    for (std::uint32_t voxel = 0U; voxel < asset.rest_voxels.size(); ++voxel) {
        const float3 p = asset.rest_voxels[voxel];
        if ((asset.voxel_flags[voxel] & waterlab::soft_body_voxel_surface) != 0U &&
            p.x < minimum_x + 0.09F && p.y > -0.25F && p.y < 0.45F) {
            impact_voxels.push_back(voxel);
        }
    }
    require(impact_voxels.size() >= 8U, "soft-body impact patch is empty");

    constexpr std::uint32_t substeps = 4U;
    constexpr float substep_dt = (1.0F / 60.0F) / static_cast<float>(substeps);
    body.begin_frame();
    for (std::uint32_t substep = 0U; substep < substeps; ++substep) {
        body.prepare_substep(substep_dt, gravity);
        if (substep == 0U && impulse_per_voxel != 0.0F) {
            std::vector<float3> impulses(asset.rest_voxels.size());
            for (const std::uint32_t voxel : impact_voxels) {
                impulses[voxel] = make_float3(impulse_per_voxel, 0.0F, 0.0F);
            }
            require(cudaMemcpy(body.voxel_view().external_impulses, impulses.data(),
                impulses.size() * sizeof(float3), cudaMemcpyHostToDevice) == cudaSuccess,
                "cannot upload soft-body impact impulse");
        }
        body.finish_substep(substep_dt, gravity);
    }
    ImpactResult result;
    auto timing = body.finish_frame();
    result.average_gpu_ms += timing.gpu_total_ms();

    for (std::uint32_t frame = 0U; frame < 90U; ++frame) {
        waterlab::SoftBodyState state;
        body.capture_state(state);
        require_finite_state(state,
            "soft-body impact fixture produced non-finite state");
        const waterlab::SoftBodyRenderView render = body.render_view();
        std::vector<float3> render_positions(render.vertex_count);
        require(cudaMemcpy(render_positions.data(), render.positions,
            render_positions.size() * sizeof(float3), cudaMemcpyDeviceToHost) == cudaSuccess,
            "cannot download soft-body runtime render positions");
        audit_render_geometry(asset, state, render_positions, result);
        for (const std::uint32_t voxel : impact_voxels) {
            result.peak_deformation = std::max(result.peak_deformation,
                distance(state.positions[voxel], reference.positions[voxel]));
        }
        for (const auto& edge : asset.edges) {
            result.peak_edge_strain = std::max(result.peak_edge_strain,
                distance(state.positions[edge.vertices.x], state.positions[edge.vertices.y]) /
                    edge.rest_length - 1.0F);
        }
        timing = body.step(gravity);
        require(body.statistics().finite_failure_count == 0U,
            "soft-body impact step reported a non-finite state");
        result.average_gpu_ms += timing.gpu_total_ms();
    }
    result.average_gpu_ms /= 91.0F;
    result.broken_edges = body.statistics().broken_edge_count;
    return result;
}
#endif

void severed_bond_tests()
{
    const auto asset = make_fixture();
    waterlab::SoftBodyOptions options;
    options.instance_count = 1U;
    options.solver_substeps = 1U;
    options.use_course_layout = false;
    options.require_1000_voxels = false;
    options.course_board_collisions = false;
    options.velocity_damping = 0.0F;
    options.break_strain = 10.0F;
    constexpr float dt = 1.0F / 60.0F;
    constexpr float outward_speed = 0.3F;
    constexpr std::uint32_t steps = 30U;
    const auto free_piece = [](std::uint32_t voxel) {
        return voxel == 1U || voxel == 3U;
    };
    const auto gap = [](const waterlab::SoftBodyState& state) {
        return 0.5F * (state.positions[1].x + state.positions[3].x -
            state.positions[0].x - state.positions[2].x);
    };
    float initial_gap{};
    const auto run = [&](bool severed) {
        waterlab::SoftBodyCourse body(asset, options);
        waterlab::SoftBodyState initial;
        body.capture_state(initial);
        for (std::uint32_t voxel = 0U; voxel < initial.positions.size(); ++voxel) {
            if (!free_piece(voxel)) continue;
            initial.positions[voxel].x += 0.3F;
            initial.velocities[voxel] = make_float3(outward_speed, 0.0F, 0.0F);
        }
        if (severed) {
            // Cut the four springs between two pieces; each retains its own
            // internal spring. Both runs start with identical positions/velocities.
            for (std::size_t edge = 0U; edge < asset.edges.size(); ++edge) {
                const uint2 vertices = asset.edges[edge].vertices;
                if (free_piece(vertices.x) == free_piece(vertices.y)) continue;
                initial.active_edges[edge] = 0U;
                ++initial.statistics.broken_edge_count;
            }
            require(initial.statistics.broken_edge_count == 4U,
                "severed-bond fixture did not split into two connected pieces");
        }
        initial_gap = gap(initial);
        body.restore_state(initial);
        waterlab::SoftBodyState current;
        for (std::uint32_t step = 0U; step < steps; ++step) {
            body.begin_frame();
            body.prepare_substep(dt, make_float3(0.0F, 0.0F, 0.0F));
            body.finish_substep(dt, make_float3(0.0F, 0.0F, 0.0F));
            (void)body.finish_frame();
            body.capture_state(current);
            require_finite_state(current, "severed-bond fixture became non-finite");
            require(current.active_edges == initial.active_edges,
                "severed-bond fixture reconnected or broke an unexpected spring");
            if (!severed) continue;
            for (const std::uint32_t voxel : {1U, 3U}) {
                float3 expected = initial.positions[voxel];
                expected.x += outward_speed * dt * static_cast<float>(step + 1U);
                require(distance(current.positions[voxel], expected) < 2.0e-5F &&
                    distance(current.velocities[voxel], initial.velocities[voxel]) < 2.0e-5F,
                    "broken springs still pull or damp a separating piece");
            }
            require(std::abs(distance(current.positions[1], current.positions[3]) -
                distance(initial.positions[1], initial.positions[3])) < 2.0e-5F,
                "separating piece lost its internal spring spacing");
        }
        return gap(current);
    };
    const float severed_gap = run(true);
    const float live_gap = run(false);
    require(severed_gap > initial_gap + 0.1F && live_gap < initial_gap - 0.05F,
        "live springs did not pull while severed pieces drifted apart");
}

void gpu_tests()
{
    auto asset = make_fixture();
    waterlab::SoftBodyOptions options;
    options.instance_count = 1U;
    options.solver_substeps = 1U;
    options.use_course_layout = false;
    options.require_1000_voxels = false;
    options.instance_origins[0] = make_float3(0.0F, 0.0F, 0.0F);
    options.break_strain = 0.10F;
    waterlab::SoftBodyCourse body(std::move(asset), options);

    auto voxels = body.voxel_view();
    auto render = body.render_view();
    require(voxels.voxel_count == 4U && voxels.external_impulses != nullptr &&
        voxels.position_corrections != nullptr && voxels.inverse_voxel_mass > 0.0F,
        "soft-body voxel view is incomplete");
    require(render.vertex_count == 3U && render.triangle_count == 1U &&
        render.bindings != nullptr && render.triangle_active != nullptr &&
        render.node_count != 0U, "soft-body render view is incomplete");

    constexpr float dt = 1.0F / 60.0F;
    body.begin_frame();
    body.prepare_substep(dt, make_float3(0.0F, 0.0F, 0.0F));
    voxels = body.voxel_view();
    const float3 impulse = make_float3(0.02F, 0.0F, 0.0F);
    const float3 correction = make_float3(0.25F, 0.0F, 0.0F);
    require(cudaMemcpy(voxels.external_impulses + 1U, &impulse, sizeof(impulse),
        cudaMemcpyHostToDevice) == cudaSuccess, "cannot upload soft-body test impulse");
    require(cudaMemcpy(voxels.position_corrections + 1U, &correction, sizeof(correction),
        cudaMemcpyHostToDevice) == cudaSuccess, "cannot upload soft-body test correction");
    body.finish_substep(dt, make_float3(0.0F, 0.0F, 0.0F));
    const auto timings = body.finish_frame();
    require(timings.gpu_total_ms() >= 0.0F, "soft-body timing is invalid");

    std::vector<float3> positions(4U);
    require(cudaMemcpy(positions.data(), body.voxel_view().positions,
        positions.size() * sizeof(float3), cudaMemcpyDeviceToHost) == cudaSuccess,
        "cannot read soft-body test positions");
    require(distance(positions[0], make_float3(0.0F, 0.10F, 0.0F)) < 1.0e-7F,
        "pinned soft-body voxel moved");
    require(positions[1].x > 0.11F,
        "soft-body contact correction was not distributed through the graph");
    require(body.statistics().broken_edge_count == 0U,
        "soft-body edge broke before persistent overstrain was established");
    waterlab::SoftBodyState first_step_state;
    body.capture_state(first_step_state);
    require_finite_state(first_step_state,
        "soft-body unit fixture produced non-finite position or velocity");
    for (std::uint32_t frame = 1U;
         frame < options.fracture_persistence_substeps; ++frame) {
        body.begin_frame();
        body.prepare_substep(dt, make_float3(0.0F, 0.0F, 0.0F));
        voxels = body.voxel_view();
        require(cudaMemcpy(voxels.position_corrections + 1U, &correction,
            sizeof(correction), cudaMemcpyHostToDevice) == cudaSuccess,
            "cannot sustain soft-body fracture test correction");
        body.finish_substep(dt, make_float3(0.0F, 0.0F, 0.0F));
        (void)body.finish_frame();
    }
    require(body.statistics().broken_edge_count != 0U,
        "persistently stretched soft-body connection did not break");
    std::uint8_t triangle_active = 1U;
    require(cudaMemcpy(&triangle_active, body.render_view().triangle_active,
        sizeof(triangle_active), cudaMemcpyDeviceToHost) == cudaSuccess,
        "cannot read soft-body render activity");
    require(triangle_active == 1U,
        "broken render support incorrectly removed material triangles");

    waterlab::SoftBodyState captured;
    body.capture_state(captured);
    require_finite_state(captured,
        "fractured soft-body unit fixture produced non-finite state");
    body.reset();
    require(body.statistics().broken_edge_count == 0U &&
        body.statistics().frame_index == 0U, "soft-body reset did not restore statistics");
    require(cudaMemcpy(&triangle_active, body.render_view().triangle_active,
        sizeof(triangle_active), cudaMemcpyDeviceToHost) == cudaSuccess &&
        triangle_active == 1U, "soft-body reset did not restore render triangles");
    body.restore_state(captured);
    require(body.statistics().broken_edge_count == captured.statistics.broken_edge_count &&
        body.statistics().frame_index == captured.statistics.frame_index,
        "soft-body restore did not recover fracture statistics");
    require(cudaMemcpy(&triangle_active, body.render_view().triangle_active,
        sizeof(triangle_active), cudaMemcpyDeviceToHost) == cudaSuccess &&
        triangle_active == 1U, "soft-body restore hid preserved material triangles");
    waterlab::SoftBodyState restored;
    body.capture_state(restored);
    require_finite_state(restored,
        "soft-body restore produced non-finite position or velocity");

#ifdef PARALLEL_MATER_SOFT_BODY_TEST_ASSET
    const auto course_asset = waterlab::load_soft_body_asset(PARALLEL_MATER_SOFT_BODY_TEST_ASSET);
    waterlab::SoftBodyCourse course(PARALLEL_MATER_SOFT_BODY_TEST_ASSET);
    const auto before = course.statistics();
    require(before.instance_count == 8U && before.voxels_per_instance == 1'000U &&
        before.total_voxel_count == 8'000U && before.surface_voxel_count == 4'000U,
        "eight-post soft-body course instantiated incorrect voxel counts");
    waterlab::SoftBodyState rest_start;
    course.capture_state(rest_start);
    require_finite_state(rest_start,
        "eight-post soft-body rest state contains non-finite data");
    const auto course_timing = course.step(make_float3(0.0F, -7.2F, 0.0F));
    require(course_timing.gpu_total_ms() >= 0.0F &&
        course.statistics().finite_failure_count == 0U &&
        course.render_view().triangle_count == 8U * 1'632U,
        "eight-post soft-body course failed its first physical frame");
    float resting_peak_displacement{};
    float resting_peak_strain{};
    for (std::uint32_t frame = 1U; frame < 600U; ++frame) {
        (void)course.step(make_float3(0.0F, -7.2F, 0.0F));
        if (frame == 1U || frame == 4U || frame == 9U || frame == 19U ||
            frame == 39U || frame == 79U || frame == 119U || frame == 159U ||
            frame == 239U || frame == 399U || frame == 599U) {
            waterlab::SoftBodyState snapshot;
            course.capture_state(snapshot);
            require_finite_state(snapshot,
                "resting soft-body course contains non-finite position or velocity");
            float maximum_displacement{};
            float maximum_strain{};
            for (std::size_t voxel = 0U; voxel < snapshot.positions.size(); ++voxel) {
                maximum_displacement = std::max(maximum_displacement,
                    distance(snapshot.positions[voxel], rest_start.positions[voxel]));
            }
            for (std::uint32_t instance = 0U; instance < 8U; ++instance) {
                const std::uint32_t base = instance * 1'000U;
                for (const auto& edge : course_asset.edges) {
                    maximum_strain = std::max(maximum_strain,
                        distance(snapshot.positions[base + edge.vertices.x],
                            snapshot.positions[base + edge.vertices.y]) / edge.rest_length - 1.0F);
                }
            }
            resting_peak_displacement = std::max(
                resting_peak_displacement, maximum_displacement);
            resting_peak_strain = std::max(resting_peak_strain, maximum_strain);
        }
    }
    require(course.statistics().finite_failure_count == 0U,
        "resting soft-body course developed a non-finite state");
    waterlab::SoftBodyState resting_final;
    course.capture_state(resting_final);
    require_finite_state(resting_final,
        "resting soft-body course final state contains non-finite data");
    require(course.statistics().broken_edge_count == 0U,
        "resting soft-body posts broke bonds without an impact");
    std::printf("eight-post soft-body rest: breaks %u, peak displacement %.4f, "
        "peak strain %.3f, first frame %.3f ms (physics %.3f, hierarchies %.3f)\n",
        course.statistics().broken_edge_count, resting_peak_displacement,
        resting_peak_strain,
        course_timing.gpu_total_ms(), course_timing.physics_ms,
        course_timing.render_hierarchy_ms);

    const ImpactResult control = run_impact(0.0F);
    const ImpactResult moderate = run_impact(0.040F);
    // Strong enough to fracture, but below the deliberately catastrophic
    // impulse used by the captured eight-post failure. This fixture checks a
    // localized tear while the recorded replay owns the severe-impact gate.
    const ImpactResult strong = run_impact(0.350F);
    std::printf("soft-body impact control/moderate/strong: deformation %.4f/%.4f/%.4f, "
        "strain %.3f/%.3f/%.3f, breaks %u/%u/%u, GPU %.3f/%.3f/%.3f ms\n",
        control.peak_deformation, moderate.peak_deformation, strong.peak_deformation,
        control.peak_edge_strain, moderate.peak_edge_strain, strong.peak_edge_strain,
        control.broken_edges, moderate.broken_edges, strong.broken_edges,
        control.average_gpu_ms, moderate.average_gpu_ms, strong.average_gpu_ms);
    std::printf("render geometry control/moderate/strong: area-min %.3f/%.3f/%.3f "
        "area-max %.3f/%.3f/%.3f edge-max %.3f/%.3f/%.3f abnormal %.3f/%.3f/%.3f\n",
        control.minimum_triangle_area_ratio, moderate.minimum_triangle_area_ratio,
        strong.minimum_triangle_area_ratio, control.maximum_triangle_area_ratio,
        moderate.maximum_triangle_area_ratio, strong.maximum_triangle_area_ratio,
        control.maximum_render_edge_ratio, moderate.maximum_render_edge_ratio,
        strong.maximum_render_edge_ratio, control.maximum_abnormal_triangle_fraction,
        moderate.maximum_abnormal_triangle_fraction,
        strong.maximum_abnormal_triangle_fraction);
    std::printf("active render control/moderate/strong: area-min %.3f/%.3f/%.3f "
        "area-max %.3f/%.3f/%.3f edge-max %.3f/%.3f/%.3f visible-min %.3f/%.3f/%.3f\n",
        control.minimum_active_triangle_area_ratio,
        moderate.minimum_active_triangle_area_ratio,
        strong.minimum_active_triangle_area_ratio,
        control.maximum_active_triangle_area_ratio,
        moderate.maximum_active_triangle_area_ratio,
        strong.maximum_active_triangle_area_ratio,
        control.maximum_active_render_edge_ratio,
        moderate.maximum_active_render_edge_ratio,
        strong.maximum_active_render_edge_ratio,
        control.minimum_active_triangle_fraction,
        moderate.minimum_active_triangle_fraction,
        strong.minimum_active_triangle_fraction);
    require(control.broken_edges == 0U && moderate.broken_edges == 0U,
        "control or moderate soft-body impact broke a bond");
    require(moderate.peak_deformation >= 0.02F &&
        moderate.peak_deformation >= control.peak_deformation * 5.0F,
        "moderate soft-body impact did not produce a measurable bend");
    require(strong.broken_edges != 0U &&
        strong.peak_deformation >= moderate.peak_deformation * 3.0F,
        "strong soft-body impact did not produce measurable fracture");
    // The deliberately tearing fixture may briefly fold a localized seam;
    // unlike the saved-capture regression, it is not a closed-surface quality
    // gate. Keep every supported expansion/edge bounded and cap the population
    // of transient sub-half-area triangles instead of hiding them as inactive.
    require(strong.minimum_active_triangle_area_ratio >= 0.30F &&
        strong.maximum_active_triangle_area_ratio <= 2.001F &&
        strong.maximum_active_render_edge_ratio <= 2.001F &&
        strong.maximum_abnormal_triangle_fraction <= 0.005F,
        "active fracture geometry escaped the normal-size contract");
    require(strong.minimum_active_triangle_fraction >= 0.90F,
        "localized fracture erased too much of the render surface");
    require(std::isfinite(control.average_gpu_ms) &&
        std::isfinite(moderate.average_gpu_ms) &&
        std::isfinite(strong.average_gpu_ms),
        "soft-body impact timing is non-finite");
#endif
}

} // namespace

int main()
{
    try {
        host_tests();
        int devices{};
        const cudaError_t status = cudaGetDeviceCount(&devices);
        if (status != cudaSuccess || devices == 0) {
            std::fprintf(stderr, "SKIP: CUDA device unavailable\n");
            return 77;
        }
        severed_bond_tests();
        gpu_tests();
        std::puts("soft-body tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "soft-body test failure: %s\n", error.what());
        return 1;
    }
}
