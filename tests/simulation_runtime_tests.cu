// SPDX-License-Identifier: MIT
#include <meshprep/simulation.hpp>

#include "../apps/water_lab/simulation_gallery.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#ifndef MESHPREP_SIMULATION_TEST_ASSET
#define MESHPREP_SIMULATION_TEST_ASSET "assets/softbody/checker_cylinder.msb"
#endif

namespace {

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

bool exactly_equal(float3 left, float3 right)
{
    return left.x == right.x && left.y == right.y && left.z == right.z;
}

template <typename T>
std::vector<T> download(const T* source, std::size_t count)
{
    std::vector<T> result(count);
    require(source != nullptr && cudaMemcpy(result.data(), source,
                count * sizeof(T), cudaMemcpyDeviceToHost) == cudaSuccess,
        "lattice view did not expose a readable device allocation");
    return result;
}

void require_recipe_parity(
    const meshprep::sim::GallerySimulation& simulation,
    const meshprep::sim::GallerySimulationOptions& requested)
{
    waterlab::gallery::ContextPhysicsOverrides overrides;
    overrides.fixed_dt = requested.fixed_step.timestep;
    overrides.physics_iterations = requested.solver_iterations_override;
    overrides.gravity = requested.gravity_override;
    const waterlab::HybridOptions native =
        waterlab::gallery::make_context_physics(requested.context, overrides);
    const meshprep::sim::ResolvedPhysicsOptions installed =
        simulation.resolved_physics();
    require(installed.fixed_step.timestep == native.fixed_dt &&
            installed.solver_iterations == native.physics_iterations &&
            exactly_equal(installed.gravity, native.gravity) &&
            installed.rigid_course_preset == native.obstacle_course,
        "native and installed gallery recipe physics diverged");
}

void test_particle_tub_without_asset()
{
    meshprep::sim::GallerySimulation simulation;
    meshprep::sim::GallerySimulationOptions options;
    options.context = meshprep::sim::ExampleContext::particle_bowl;
    require(options.soft_body_asset_path.empty(),
        "particle tub test accidentally supplied an asset");
    require(simulation.initialize(options).ok(),
        "particle tub must initialize without an asset");
    require(simulation.initialized() &&
            simulation.context() == meshprep::sim::ExampleContext::particle_bowl,
        "particle tub identity was not retained");
    require_recipe_parity(simulation, options);

    auto frame = simulation.render_view();
    require(frame.particle_system_count == 1U && frame.surface_count == 0U &&
            frame.rigid_body_count == 2U + waterlab::bowl_peg_count &&
            frame.lattice_count == 0U &&
            frame.lattices == nullptr,
        "particle tub exposed undeclared components or omitted its arena");
    require(frame.particle_systems[0].count == 20'000U,
        "particle tub exposed the wrong particle count");
    for (std::uint32_t peg = 0U; peg < waterlab::bowl_peg_count; ++peg) {
        const auto& body = frame.rigid_bodies[2U + peg];
        require(body.mesh.vertex_count == 50U && body.mesh.triangle_count == 96U &&
                body.scale.x == waterlab::bowl_peg_radius &&
                body.scale.y == 0.5F * waterlab::bowl_peg_height &&
                body.translation.x == waterlab::bowl_peg(peg).x &&
                body.translation.z == waterlab::bowl_peg(peg).z,
            "particle-bowl peg is not the authored capped cylinder");
    }

    require(simulation.step().ok(), "particle tub fixed step failed");
    frame = simulation.render_view();
    float3 first{};
    const cudaError_t particle_read_status = cudaMemcpy(&first,
        frame.particle_systems[0].positions, sizeof(first), cudaMemcpyDeviceToHost);
    if (particle_read_status != cudaSuccess) std::fprintf(stderr,
        "particle view=%p read=%s\n",
        static_cast<const void*>(frame.particle_systems[0].positions),
        cudaGetErrorString(particle_read_status));
    require(particle_read_status == cudaSuccess,
        "particle tub view was not a readable device allocation");
    require(std::isfinite(first.x) && std::isfinite(first.y) && std::isfinite(first.z),
        "particle tub produced a non-finite position");
    const auto stats = simulation.statistics();
    require(stats.frame_index == 1U && stats.particle_count == 20'000U &&
            stats.surface_count == 0U &&
            stats.rigid_body_count == 2U + waterlab::bowl_peg_count &&
            stats.finite_failure_count == 0U,
        "particle tub statistics violate the public recipe");

    require(simulation.reset().ok() && simulation.statistics().frame_index == 0U,
        "particle tub reset did not restore frame zero");
}

void test_live_particle_resize()
{
    meshprep::sim::GallerySimulationOptions options;
    options.context = meshprep::sim::ExampleContext::particle_bowl;
    options.particle_count_override = 512U;
    meshprep::sim::GallerySimulation simulation;
    require(simulation.initialize(options).ok(), "small active particle prefix failed");
    for (const std::uint32_t count : {2'048U, 256U, 20'000U, 3'000U}) {
        require(simulation.resize_particles(count).ok(),
            "runtime particle spawn/removal failed");
        require(simulation.render_view().particle_systems[0].count == count &&
                simulation.statistics().particle_count == count,
            "runtime resize exposed a stale active count");
        require(simulation.step().ok() &&
                simulation.statistics().finite_failure_count == 0U,
            "resized particle system became non-finite");
    }
    require(!simulation.resize_particles(255U).ok() &&
            !simulation.resize_particles(20'001U).ok(),
        "particle resize failed to enforce its reserved bounds");
}

void test_procedural_soft_body_fluid()
{
    meshprep::sim::GallerySimulationOptions options;
    options.context = meshprep::sim::ExampleContext::soft_body_fluid;
    meshprep::sim::GallerySimulation simulation;
    require(meshprep::sim::GallerySimulation::create(options, simulation).ok(),
        "procedural water-wheel context failed to initialize without an asset");
    require_recipe_parity(simulation, options);
    const auto configured = simulation.options();
    require(configured.context == meshprep::sim::ExampleContext::soft_body_fluid &&
            configured.soft_body_asset_path.empty(),
        "procedural water-wheel context invented an asset dependency");

    auto frame = simulation.render_view();
    require(frame.particle_system_count == 1U && frame.surface_count == 1U &&
            frame.rigid_body_count == 12U + waterlab::water_wheel_fin_count &&
            frame.lattice_count == 1U,
        "soft-body/fluid context exposed an undeclared component view");
    const std::uint32_t first_platform=7U+waterlab::water_wheel_fin_count;
    const auto& left_platform = frame.rigid_bodies[first_platform];
    const auto& right_platform = frame.rigid_bodies[first_platform+1U];
    require(left_platform.translation.z == waterlab::water_wheel_stage_z &&
            right_platform.translation.z == waterlab::water_wheel_stage_z &&
            left_platform.translation.x < waterlab::water_wheel_center.x &&
            right_platform.translation.x > waterlab::water_wheel_center.x &&
            left_platform.scale.z == waterlab::water_wheel_top_platform_half_depth &&
            right_platform.scale.z == waterlab::water_wheel_top_platform_half_depth,
        "public water-wheel recipe omitted or misplaced its two front platforms");
    const auto& near_bumper=frame.rigid_bodies[first_platform+2U];
    const auto& far_bumper=frame.rigid_bodies[first_platform+3U];
    const auto& right_bumper=frame.rigid_bodies[first_platform+4U];
    require(near_bumper.translation.z<waterlab::water_wheel_stage_z &&
            far_bumper.translation.z>waterlab::water_wheel_stage_z &&
            right_bumper.translation.x>waterlab::water_wheel_center.x+
                waterlab::water_wheel_top_platform_outer_x,
        "public water-wheel recipe omitted its stage bumper walls");
    require(frame.surfaces[0].mesh.vertex_count != 0U &&
            frame.surfaces[0].mesh.triangle_count != 0U &&
            frame.surfaces[0].triangle_active != nullptr,
        "soft-body render surface is incomplete");
    const auto& lattice = frame.lattices[0];
    require(lattice.nodes_per_instance == 870U &&
            lattice.node_count == lattice.nodes_per_instance * lattice.instance_count &&
            lattice.bonds_per_instance > 0U && lattice.node_radius > 0.0F,
        "public soft-body lattice omitted volume nodes or topology");
    const auto flags = download(lattice.flags, lattice.node_count);
    require(std::count_if(flags.begin(), flags.end(), [](std::uint32_t flag) {
                return (flag & meshprep::sim::lattice_node_surface) == 0U;
            }) > 0U,
        "public axle cross lattice omitted its interior volume nodes");
    const auto bonds = download(lattice.bonds, lattice.bonds_per_instance);
    require(std::all_of(bonds.begin(), bonds.end(), [&lattice](const auto& bond) {
                return bond.vertices.x < lattice.nodes_per_instance &&
                    bond.vertices.y < lattice.nodes_per_instance && bond.rest_length > 0.0F;
            }), "public lattice bonds do not use local node indices");
    const auto active = download(lattice.bond_active,
        lattice.bonds_per_instance * lattice.instance_count);
    require(std::all_of(active.begin(), active.end(), [](std::uint8_t value) {
                return value == 1U;
            }), "public lattice started with missing live bonds");

    require(simulation.step().ok(), "asset-backed context fixed step failed");
    const auto stats = simulation.statistics();
    require(stats.frame_index == 1U && stats.particle_count == 322U &&
            stats.surface_count == 1U &&
            stats.rigid_body_count == 12U + waterlab::water_wheel_fin_count &&
            stats.finite_failure_count == 0U,
        "procedural water-wheel context statistics are inconsistent");

    meshprep::sim::GallerySimulation moved = std::move(simulation);
    require(moved.initialized() && !simulation.initialized(),
        "GallerySimulation move did not transfer ownership");
    require(moved.reset().ok(), "moved GallerySimulation could not reset");
}

void test_every_public_recipe_steps_declared_components()
{
    for (const auto& recipe : meshprep::sim::example_contexts) {
        meshprep::sim::GallerySimulationOptions options;
        options.context = recipe.id;
        if (meshprep::sim::requires_soft_body_asset(recipe.id)) {
            options.soft_body_asset_path = MESHPREP_SIMULATION_TEST_ASSET;
        }
        meshprep::sim::GallerySimulation simulation;
        require(simulation.initialize(options).ok(),
            "public gallery recipe failed to initialize");
        require_recipe_parity(simulation, options);
        for (std::uint32_t step = 0U; step < 3U; ++step) {
            require(simulation.step().ok(),
                "public gallery recipe failed a multi-step smoke test");
        }

        const bool particles = meshprep::sim::has_component(
                recipe.components, meshprep::sim::Component::fluid_particles) ||
            meshprep::sim::has_component(
                recipe.components, meshprep::sim::Component::hand_particles);
        const bool deformable = meshprep::sim::has_component(
                recipe.components, meshprep::sim::Component::cloth) ||
            meshprep::sim::has_component(
                recipe.components, meshprep::sim::Component::soft_body);
        const bool water_skin = meshprep::sim::has_component(
            recipe.components, meshprep::sim::Component::water_skin);
        const bool rigid = meshprep::sim::has_component(
            recipe.components, meshprep::sim::Component::rigid_bodies);
        const auto frame = simulation.render_view();
        const auto stats = simulation.statistics();
        require(frame.particle_system_count == (particles ? 1U : 0U) &&
                frame.surface_count == static_cast<std::uint32_t>(water_skin) +
                    static_cast<std::uint32_t>(deformable) &&
                frame.rigid_body_count == ([&] {
                    if (!rigid) return 0U;
                    if (recipe.id == meshprep::sim::ExampleContext::water_course)
                        return 5U + waterlab::course_peg_count;
                    if (recipe.id == meshprep::sim::ExampleContext::particle_bowl)
                        return 2U + waterlab::bowl_peg_count;
                    if (recipe.id == meshprep::sim::ExampleContext::soft_body_fluid)
                        return 12U + waterlab::water_wheel_fin_count;
                    if (recipe.id == meshprep::sim::ExampleContext::particles_cloth)
                        return 15U;
                    if (recipe.id == meshprep::sim::ExampleContext::soft_body_cloth)
                        return 6U;
                    if (recipe.id == meshprep::sim::ExampleContext::rope_rigid)
                        return 3U;
                    if (recipe.id == meshprep::sim::ExampleContext::rope_bridge)
                        return 3U;
                    return 7U;
                }()) &&
                frame.lattice_count == (deformable ? 1U : 0U),
            "public gallery recipe exposed an undeclared component");
        for (std::uint32_t body_index = 0U;
             body_index < frame.rigid_body_count; ++body_index) {
            const auto& body = frame.rigid_bodies[body_index];
            const auto triangles = download(
                body.mesh.triangles, body.mesh.triangle_count);
            require(std::all_of(triangles.begin(), triangles.end(),
                [&body](uint3 triangle) {
                    return triangle.x < body.mesh.vertex_count &&
                        triangle.y < body.mesh.vertex_count &&
                        triangle.z < body.mesh.vertex_count;
                }), "public rigid-body view contains non-local triangle indices");
        }
        if (recipe.id == meshprep::sim::ExampleContext::water_course) {
            require(frame.lattice_count == 0U && frame.rigid_body_count == 13U,
                "water course retained a soft lattice or omitted rigid posts");
            for (std::uint32_t peg = 0U; peg < waterlab::course_peg_count; ++peg) {
                const auto& body = frame.rigid_bodies[5U + peg];
                const float3 base = waterlab::course_peg(peg);
                require(body.mesh.vertex_count == 50U &&
                        body.mesh.triangle_count == 96U &&
                        body.translation.x == base.x &&
                        body.translation.z == base.z &&
                        body.scale.x == waterlab::course_peg_radius &&
                        body.scale.y == 0.5F * waterlab::course_peg_height,
                    "public water course rigid-post view is inconsistent");
            }
        }
        require(stats.frame_index == 3U && stats.finite_failure_count == 0U,
            "public gallery recipe produced invalid multi-step statistics");
    }
}

void test_lattice_masks_keep_instance_identity()
{
    waterlab::SoftBodyOptions options;
    options.instance_count = 2U;
    options.use_course_layout = false;
    options.course_board_collisions = false;
    options.instance_origins[1] = make_float3(3.0F, 0.0F, 0.0F);
    waterlab::SoftBodyCourse body(MESHPREP_SIMULATION_TEST_ASSET, options);
    const auto initial = body.lattice_view();
    require(initial.voxel_count == 2000U && initial.voxels_per_instance == 1000U &&
            initial.instance_count == 2U && initial.edges_per_instance > 1U,
        "native lattice view lost instance counts");
    const auto positions = download(initial.positions, initial.voxel_count);
    require(std::abs(positions[1000U].x - positions[0].x - 3.0F) < 1.0e-5F,
        "native lattice positions do not follow the advertised instance stride");

    waterlab::SoftBodyState state;
    body.capture_state(state);
    state.active_edges[0U] = 0U;
    state.active_edges[initial.edges_per_instance + 1U] = 0U;
    state.statistics.broken_edge_count = 2U;
    body.restore_state(state);
    auto view = body.lattice_view();
    auto active = download(view.active_edges,
        view.edges_per_instance * view.instance_count);
    require(active == state.active_edges && active[1U] == 1U &&
            active[view.edges_per_instance] == 1U,
        "native lattice view conflated per-instance broken bonds");
    (void)body.step(make_float3(0.0F, 0.0F, 0.0F));
    view = body.lattice_view();
    active = download(view.active_edges, view.edges_per_instance * view.instance_count);
    require(active[0U] == 0U && active[view.edges_per_instance + 1U] == 0U,
        "native lattice view restored broken bonds after stepping");
    body.reset();
    view = body.lattice_view();
    active = download(view.active_edges, view.edges_per_instance * view.instance_count);
    require(std::all_of(active.begin(), active.end(), [](std::uint8_t value) {
                return value == 1U;
            }), "native lattice view did not restore bonds on reset");
}

void test_explicit_physics_overrides()
{
    meshprep::sim::GallerySimulationOptions options;
    options.context = meshprep::sim::ExampleContext::particles_cloth;
    options.fixed_step.timestep = 1.0F / 120.0F;
    options.solver_iterations_override = 2U;
    options.gravity_override = make_float3(0.25F, -1.5F, 0.5F);
    options.cloth_detail_override = 3U;

    meshprep::sim::GallerySimulation simulation;
    require(simulation.initialize(options).ok(),
        "valid explicit gallery physics overrides were rejected");
    require_recipe_parity(simulation, options);
    const auto resolved = simulation.resolved_physics();
    require(resolved.fixed_step.timestep == 1.0F / 120.0F &&
            resolved.solver_iterations == 2U &&
            exactly_equal(resolved.gravity, *options.gravity_override) &&
            !resolved.rigid_course_preset,
        "public gallery did not report exact resolved overrides");
    const auto detailed_cloth = simulation.render_view().lattices[0];
    require(detailed_cloth.nodes_per_instance == (27U * 3U + 1U) *
            (21U * 3U + 1U),
        "public cloth-detail override did not retessellate context 5");
    require(simulation.step().ok() && simulation.step().ok(),
        "overridden public gallery recipe failed repeated steps");

    options.solver_iterations_override = 0U;
    require(!simulation.initialize(options).ok(),
        "zero solver-iteration override was accepted");
    options.solver_iterations_override =
        waterlab::HybridDroplet::maximum_physics_iterations + 1U;
    require(!simulation.initialize(options).ok(),
        "out-of-range solver-iteration override was accepted");
    options.solver_iterations_override = 2U;
    options.cloth_detail_override = 9U;
    require(!simulation.initialize(options).ok(),
        "out-of-range cloth-detail override was accepted");
    options.cloth_detail_override = 3U;
    options.gravity_override = make_float3(NAN, 0.0F, 0.0F);
    require(!simulation.initialize(options).ok(),
        "non-finite gravity override was accepted");

    meshprep::sim::GallerySimulationOptions hanging_options;
    hanging_options.context = meshprep::sim::ExampleContext::cloth_rigid;
    hanging_options.cloth_detail_override = 2U;
    meshprep::sim::GallerySimulation hanging_cloth;
    require(hanging_cloth.initialize(hanging_options).ok(),
        "context-3 cloth-detail override was rejected");
    const auto hanging_lattice = hanging_cloth.render_view().lattices[0];
    require(hanging_lattice.nodes_per_instance == (27U * 2U + 1U) *
            (29U * 2U + 1U),
        "context-3 cloth-detail override did not retessellate the cloth");
}

meshprep::Status invoke_during_stream_capture(
    meshprep::sim::GallerySimulation& simulation, bool reset)
{
    cudaStream_t stream{};
    require(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess,
        "could not create capture-stream status fixture");
    require(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) == cudaSuccess,
        "could not begin capture-stream status fixture");
    const meshprep::Status status =
        reset ? simulation.reset(stream) : simulation.step(stream);

    cudaGraph_t graph{};
    const cudaError_t end_status = cudaStreamEndCapture(stream, &graph);
    if (graph != nullptr) {
        require(cudaGraphDestroy(graph) == cudaSuccess,
            "could not destroy captured status-fixture graph");
    }
    (void)cudaGetLastError();
    require(end_status == cudaSuccess ||
            end_status == cudaErrorStreamCaptureInvalidated,
        "capture-stream fixture ended with an unexpected CUDA error");
    require(cudaStreamDestroy(stream) == cudaSuccess,
        "could not destroy capture-stream status fixture");
    return status;
}

void test_cuda_failures_retain_the_runtime_error()
{
    meshprep::sim::GallerySimulationOptions options;
    options.context = meshprep::sim::ExampleContext::particle_bowl;

    meshprep::sim::GallerySimulation simulation;
    require(simulation.initialize(options).ok(),
        "status fixture simulation failed to initialize");
    const meshprep::Status step = invoke_during_stream_capture(simulation, false);
    require(step.code == meshprep::StatusCode::cuda_failure &&
            step.cuda_error != cudaSuccess,
        "lower-layer step exception erased the concrete CUDA failure");
    require(simulation.reset().ok(),
        "simulation did not recover after a rejected captured step");

    const meshprep::Status reset = invoke_during_stream_capture(simulation, true);
    require(reset.code == meshprep::StatusCode::cuda_failure &&
            reset.cuda_error != cudaSuccess,
        "lower-layer reset exception erased the concrete CUDA failure");

    require(simulation.reset().ok(),
        "simulation did not recover after a rejected captured reset");
}

} // namespace

int main()
{
    int devices{};
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::fprintf(stderr, "SKIP: CUDA device unavailable\n");
        return 77;
    }
    try {
        test_particle_tub_without_asset();
        test_live_particle_resize();
        test_procedural_soft_body_fluid();
        test_lattice_masks_keep_instance_identity();
        test_every_public_recipe_steps_declared_components();
        test_explicit_physics_overrides();
        test_cuda_failures_retain_the_runtime_error();
        std::puts("simulation runtime API tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "simulation runtime API test failure: %s\n", error.what());
        return 1;
    }
}
