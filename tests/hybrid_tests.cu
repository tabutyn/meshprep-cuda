// SPDX-License-Identifier: MIT
#include "../apps/water_lab/hybrid_lab.hpp"
#include "../apps/water_lab/fluid_visuals.hpp"
#include "../apps/water_lab/fluid_surface.cuh"
#include "../apps/water_lab/obstacle_course.hpp"
#include "../apps/water_lab/soft_body.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

bool finite(float value) { return std::isfinite(value); }
bool finite(float3 value)
{
    return finite(value.x) && finite(value.y) && finite(value.z);
}

std::string soft_body_asset_path()
{
#ifdef MESHPREP_SOFT_BODY_TEST_ASSET
    return MESHPREP_SOFT_BODY_TEST_ASSET;
#else
    return (std::filesystem::path(__FILE__).parent_path().parent_path() /
        "assets/softbody/checker_cylinder.msb").string();
#endif
}

void require_timings(const waterlab::HybridTimings& timings)
{
    require(finite(timings.rebuild_fluid_hierarchy_ms) &&
            timings.rebuild_fluid_hierarchy_ms >= 0.0F,
        "invalid fluid hierarchy timing");
    require(finite(timings.rebuild_skin_hierarchy_ms) &&
            timings.rebuild_skin_hierarchy_ms >= 0.0F,
        "invalid skin hierarchy timing");
    require(finite(timings.update_fluid_physics_ms) &&
            timings.update_fluid_physics_ms >= 0.0F,
        "invalid fluid physics timing");
    require(finite(timings.update_skin_physics_ms) &&
            timings.update_skin_physics_ms >= 0.0F,
        "invalid skin physics timing");
    require(finite(timings.update_rectangle_physics_ms) &&
            timings.update_rectangle_physics_ms >= 0.0F,
        "invalid rectangle physics timing");
    require(finite(timings.update_surface_normals_ms) &&
            timings.update_surface_normals_ms >= 0.0F,
        "invalid normals timing");
    require(finite(timings.update_render_surface_ms) &&
            timings.update_render_surface_ms >= 0.0F,
        "invalid render surface timing");
    require(finite(timings.gpu_total_ms()) && timings.gpu_total_ms() >= 0.0F,
        "invalid total timing");
}

void require_state(const waterlab::HybridDroplet& droplet)
{
    const auto stats = droplet.statistics();
    const auto rectangle = droplet.rectangle();
    require(stats.particle_count == 10'000U, "particle count changed");
    require(stats.physical_skin_vertices == 1'002U, "unexpected frequency-10 skin");
    require(stats.physical_skin_triangles == 2'000U, "unexpected physical triangles");
    require(stats.render_skin_vertices == 20'252U, "unexpected frequency-45 render skin");
    require(stats.render_skin_triangles == 40'500U, "unexpected render triangles");
    require(stats.finite_failures == 0U, "non-finite simulation value");
    require(stats.maximum_particle_neighbors < stats.particle_count,
        "invalid particle neighbor count");
    require(finite(rectangle.center) && finite(rectangle.velocity) &&
            finite(rectangle.yaw) && finite(rectangle.angular_velocity),
        "non-finite rectangle state");
    require(droplet.particle_hierarchy().nodes() != nullptr &&
            droplet.particle_hierarchy().statistics().node_count != 0U,
        "particle hierarchy is empty");
    require(droplet.skin_hierarchy().nodes() != nullptr &&
            droplet.skin_hierarchy().statistics().node_count != 0U,
        "render hierarchy is empty");
}

void require_debug_forces_finite(const waterlab::HybridDroplet& droplet)
{
    const std::uint32_t count = droplet.physical_skin_vertex_count();
    std::vector<float3> values(count);
    const float3* sources[4]{
        droplet.physical_skin_normals(), droplet.skin_box_forces(),
        droplet.skin_particle_forces(), droplet.skin_spring_forces()};
    for (const float3* source : sources) {
        require(cudaMemcpy(values.data(), source, values.size() * sizeof(float3),
                    cudaMemcpyDeviceToHost) == cudaSuccess,
            "failed to download a skin debug vector");
        for (const float3 value : values) {
            require(finite(value), "non-finite skin debug vector");
        }
    }
}

template <typename T>
bool bit_equal(const std::vector<T>& left, const std::vector<T>& right)
{
    return left.size() == right.size() &&
        std::memcmp(left.data(), right.data(), left.size() * sizeof(T)) == 0;
}

void require_same_dynamic_state(
    const waterlab::HybridState& left, const waterlab::HybridState& right)
{
    require(bit_equal(left.particle_positions, right.particle_positions),
        "restored particle positions differ");
    require(bit_equal(left.particle_velocities, right.particle_velocities),
        "restored particle velocities differ");
    require(bit_equal(left.particle_forces, right.particle_forces),
        "restored particle forces differ");
    require(bit_equal(left.particle_skin_owners, right.particle_skin_owners),
        "restored particle owners differ");
    require(bit_equal(left.skin_positions, right.skin_positions),
        "restored skin positions differ");
    require(bit_equal(left.skin_velocities, right.skin_velocities),
        "restored skin velocities differ");
    require(bit_equal(left.skin_forces, right.skin_forces),
        "restored skin forces differ");
    require(bit_equal(left.skin_box_forces, right.skin_box_forces),
        "restored skin box forces differ");
    require(bit_equal(left.skin_box_impulses, right.skin_box_impulses),
        "restored skin box impulses differ");
    require(bit_equal(left.skin_box_pair_work, right.skin_box_pair_work),
        "restored skin box work differs");
    require(bit_equal(left.skin_particle_forces, right.skin_particle_forces),
        "restored particle reactions differ");
    require(bit_equal(left.skin_spring_forces, right.skin_spring_forces),
        "restored spring forces differ");
    require(std::memcmp(&left.rectangle, &right.rectangle,
                sizeof(waterlab::RectangleState)) == 0,
        "restored rectangle state differs");
    require(left.statistics.frame_index == right.statistics.frame_index,
        "restored frame index differs");
}

void test_idle_and_reset()
{
    waterlab::HybridDroplet droplet;
    require_state(droplet);
    for (std::uint32_t frame = 0; frame < 120U; ++frame) {
        require_timings(droplet.step());
        require_state(droplet);
        require(droplet.statistics().particles_outside == 0U,
            "idle particle crossed the skin");
        require(droplet.statistics().frame_index == frame + 1U,
            "one step did not advance exactly one frame");
    }
    droplet.reset();
    require_state(droplet);
    require(droplet.statistics().frame_index == 0U, "reset did not reset frame index");
}

void test_dynamic_rectangle()
{
    waterlab::HybridDroplet droplet;
    const auto initial = droplet.rectangle();
    bool observed_contact = false;
    std::uint32_t maximum_outside = 0U;
    for (std::uint32_t frame = 0; frame < 120U; ++frame) {
        require_timings(droplet.step(make_float3(-400.0F, 0.0F, 0.0F), 5.0F));
        require_state(droplet);
        const auto stats = droplet.statistics();
        maximum_outside = std::max(maximum_outside, stats.particles_outside);
        observed_contact = observed_contact ||
            (stats.skin_vertices_inside_rectangle != 0U &&
             stats.maximum_rectangle_reaction > 0.0F);
    }
    const auto moved = droplet.rectangle();
    require(moved.center.x < initial.center.x - 0.25F,
        "finite-mass rectangle did not respond to force");
    const float speed = std::sqrt(moved.velocity.x * moved.velocity.x +
        moved.velocity.y * moved.velocity.y + moved.velocity.z * moved.velocity.z);
    require(speed <= droplet.options().maximum_rectangle_speed + 1.0e-4F,
        "rectangle speed cap failed");
    require(std::abs(moved.angular_velocity) <=
            droplet.options().maximum_rectangle_angular_speed + 1.0e-4F,
        "rectangle angular speed cap failed");
    require(observed_contact, "dynamic rectangle never exchanged force with the skin");
    if (maximum_outside != 0U) {
        std::printf("dynamic maximum outside particles: %u\n", maximum_outside);
    }
    require(maximum_outside < 128U, "bounded contact produced a large escape burst");
    require(droplet.statistics().particles_outside == 0U,
        "outside particles did not return through the boundary force");
    require_debug_forces_finite(droplet);
}

void test_runtime_physics_controls()
{
    waterlab::HybridOptions initial_options;
    require(initial_options.box_skin_stiffness == 3'000.0F,
        "box-to-skin stiffness is not ten times the original value");
    require(initial_options.box_skin_damping == 64.0F,
        "unexpected box contact damping");
    require(initial_options.box_skin_contact_thickness == 0.002F,
        "unexpected box contact shell thickness");
    require(initial_options.particle_skin_damping == 12.0F,
        "unexpected particle-to-skin damping");
    require(initial_options.maximum_box_ejection_force == 200.0F,
        "box-to-skin force cap is not ten times the original value");
    require(initial_options.maximum_skin_force >=
            initial_options.maximum_box_ejection_force,
        "aggregate skin force cap clips the box ejection force");
    initial_options.physics_iterations = 2U;
    waterlab::HybridDroplet droplet(initial_options);
    require_timings(droplet.step());
    require_state(droplet);
    require(droplet.statistics().frame_index == 1U,
        "substeps advanced more than one displayed physics frame");

    auto updated = droplet.options();
    updated.physics_iterations = 3U;
    updated.particle_repulsion += 0.25F;
    updated.box_skin_stiffness += 250.0F;
    droplet.set_runtime_options(updated);
    require(droplet.options().physics_iterations == 3U,
        "runtime iteration update was not retained");
    require_timings(droplet.step());
    require_state(droplet);
    require(droplet.statistics().frame_index == 2U,
        "runtime substeps changed frame-index semantics");
    require_debug_forces_finite(droplet);
}

void test_particle_only_mode()
{
    waterlab::HybridOptions options = waterlab::course_options();
    options.particle_skin_coupling = false;
    waterlab::HybridDroplet droplet(options);
    waterlab::HybridState before;
    droplet.capture_state(before);
    const auto timings = droplet.step();
    require_timings(timings);
    waterlab::HybridState after;
    droplet.capture_state(after);

    require(bit_equal(before.skin_positions, after.skin_positions) &&
            bit_equal(before.skin_velocities, after.skin_velocities),
        "particle-only mode advanced its hidden compatibility skin");
    require(!bit_equal(before.particle_positions, after.particle_positions),
        "particle-only mode did not advance particles");
    require(std::all_of(after.particle_skin_owners.begin(),
            after.particle_skin_owners.end(), [](std::uint32_t owner) {
                return owner == UINT32_MAX;
            }),
        "particle-only mode emitted a hidden skin owner");
    require(timings.update_skin_physics_ms < 0.05F &&
            timings.rebuild_skin_hierarchy_ms < 0.05F &&
            timings.update_surface_normals_ms < 0.05F &&
            timings.update_render_surface_ms < 0.05F,
        "particle-only mode performed material hidden-skin work");
}

void test_state_capture_restore()
{
    waterlab::HybridDroplet droplet;
    for (std::uint32_t frame = 0U; frame < 4U; ++frame) {
        require_timings(droplet.step(make_float3(-250.0F, 30.0F, 0.0F), 2.0F));
    }
    waterlab::HybridState checkpoint;
    droplet.capture_state(checkpoint);

    require_timings(droplet.step(make_float3(400.0F, -20.0F, 0.0F), -3.0F));
    droplet.restore_state(checkpoint);
    waterlab::HybridState restored;
    droplet.capture_state(restored);
    require_same_dynamic_state(checkpoint, restored);
    require_state(droplet);

    require_timings(droplet.step(make_float3(-300.0F, 10.0F, 0.0F), 1.0F));
    waterlab::HybridState first_continuation;
    droplet.capture_state(first_continuation);
    droplet.restore_state(checkpoint);
    require_timings(droplet.step(make_float3(-300.0F, 10.0F, 0.0F), 1.0F));
    waterlab::HybridState second_continuation;
    droplet.capture_state(second_continuation);
    require_same_dynamic_state(first_continuation, second_continuation);
}

void test_wireframe_box_render()
{
    waterlab::HybridDroplet droplet;
    waterlab::RayTracer renderer;
    waterlab::Camera camera;
    constexpr std::uint32_t width = 320U;
    constexpr std::uint32_t height = 240U;
    const float milliseconds = renderer.render_hybrid(
        droplet.skin_mesh(), droplet.skin_normals(), droplet.skin_hierarchy(),
        droplet.particle_positions(), droplet.particle_radius(),
        droplet.particle_hierarchy(), false, droplet.render_box(), camera, width, height);
    require(finite(milliseconds) && milliseconds >= 0.0F,
        "wireframe render did not complete");
    std::size_t orange_pixels = 0U;
    for (std::size_t index = 0U; index < static_cast<std::size_t>(width) * height; ++index) {
        const uchar4 pixel = renderer.pixels()[index];
        if (pixel.x > pixel.y + 35 && pixel.y > pixel.z + 20) ++orange_pixels;
    }
    require(orange_pixels > 30U, "wireframe rectangle edges are not visible");
    require(orange_pixels < 2'000U, "rectangle rendered as filled faces, not a wireframe");
}

std::size_t differing_pixels(
    const std::vector<uchar4>& left, const std::vector<uchar4>& right)
{
    require(left.size() == right.size(), "render dimensions differ");
    std::size_t count = 0U;
    for (std::size_t i = 0; i < left.size(); ++i) {
        if (std::memcmp(&left[i], &right[i], sizeof(uchar4)) != 0) ++count;
    }
    return count;
}

void test_fluid_visual_render_modes()
{
    waterlab::HybridDroplet droplet;
    waterlab::FluidVisuals visuals(droplet.statistics().particle_count);
    require(finite(visuals.update(droplet.particle_positions(), droplet.particle_velocities(),
                droplet.particle_hierarchy(), droplet.options().particle_support_radius,
                {}, droplet.options().fixed_dt, nullptr, false,
                droplet.particle_cells())),
        "fluid visual update failed");
    waterlab::HybridState before, after;
    droplet.capture_state(before);
    waterlab::RayTracer renderer;
    waterlab::Camera camera;
    // Odd dimensions include a ray exactly through the tiny central bubble;
    // this tests composition independently of subpixel sample placement.
    constexpr std::uint32_t width = 241U, height = 181U;
    const auto render = [&](waterlab::FluidDisplay display, bool foam) {
        const float milliseconds = renderer.render_hybrid(
            droplet.skin_mesh(), droplet.skin_normals(), droplet.skin_hierarchy(),
            droplet.particle_positions(), droplet.particle_radius(),
            droplet.particle_hierarchy(), false, droplet.render_box(), camera,
            width, height, nullptr, false, visuals.view(display, foam));
        require(finite(milliseconds) && milliseconds >= 0.0F, "fluid visual render failed");
        return std::vector<uchar4>(renderer.pixels(), renderer.pixels() + width * height);
    };
    const auto surface = render(waterlab::FluidDisplay::Surface, false);
    const auto particles = render(waterlab::FluidDisplay::Particles, false);
    const auto combined_wire_base = render(waterlab::FluidDisplay::Wireframe, false);
    require(differing_pixels(surface, particles) != 0U, "surface and particles look identical");
    require(differing_pixels(surface, combined_wire_base) != 0U,
        "surface and combined-wire background look identical");
    require(differing_pixels(particles, combined_wire_base) == 0U,
        "combined wire mode must retain the particle layer before host wire overlays");

    meshprep::Hierarchy no_particles;
    const float wire_without_particles = renderer.render_hybrid(
        droplet.skin_mesh(), droplet.skin_normals(), droplet.skin_hierarchy(),
        nullptr, droplet.particle_radius(), no_particles, false,
        droplet.render_box(), camera, width, height, nullptr, false,
        visuals.view(waterlab::FluidDisplay::Wireframe, false, false));
    require(finite(wire_without_particles),
        "component-only wire mode incorrectly required particle data");
    bool rejected_missing_particles = false;
    try {
        (void)renderer.render_hybrid(
            droplet.skin_mesh(), droplet.skin_normals(), droplet.skin_hierarchy(),
            nullptr, droplet.particle_radius(), no_particles, false,
            droplet.render_box(), camera, width, height, nullptr, false,
            visuals.view(waterlab::FluidDisplay::Wireframe, false, true));
    } catch (const std::invalid_argument&) {
        rejected_missing_particles = true;
    }
    require(rejected_missing_particles,
        "combined wire mode accepted a missing requested particle layer");

    // Put actual, independent bubbles on the reconstructed front surface.
    // Changing a water particle's color/coverage is not a foam rendering test.
    waterlab::FluidSurfaceGrid grid;
    const auto device_surface = visuals.view().surface;
    require(cudaMemcpy(&grid, device_surface.grid, sizeof(grid), cudaMemcpyDeviceToHost) == cudaSuccess,
        "failed to download fluid descriptor");
    std::vector<float> field(static_cast<std::size_t>(grid.dimensions.x)*
        grid.dimensions.y*grid.dimensions.z);
    require(cudaMemcpy(field.data(), device_surface.values, field.size()*sizeof(float),
                cudaMemcpyDeviceToHost) == cudaSuccess, "failed to download fluid field");
    const waterlab::FluidSurfaceView host_surface{field.data(), &grid};
    float lower = 0.0F, upper = grid.minimum.z+grid.cell_size.z*(grid.dimensions.z-1U);
    require(waterlab::surface_sample(host_surface, {0,0,lower}) < 0.0F,
        "initial blob has no continuous interior");
    for (int i=0; i<24; ++i) {
        const float middle = (lower+upper)*0.5F;
        if (waterlab::surface_sample(host_surface, {0,0,middle}) < 0.0F) lower=middle;
        else upper=middle;
    }
    std::vector<waterlab::FoamParticle> bubbles(waterlab::FluidVisuals::foam_capacity);
    bubbles[0] = {make_float4(0,0,upper+0.004F,0.1F), make_float4(0,0,0,2.0F),
        make_float4(0,0,1,0.014F)};
    visuals.restore_foam(bubbles, 1U);
    const auto foam = render(waterlab::FluidDisplay::Surface, true);
    const auto foam_off = render(waterlab::FluidDisplay::Surface, false);
    require(differing_pixels(foam, foam_off) != 0U, "independent foam did not affect visible pixels");
    droplet.capture_state(after);
    require_same_dynamic_state(before, after);
}

void test_soft_post_occlusion()
{
    const float3 peg = waterlab::course_peg(0U);
    waterlab::SoftBodyOptions soft_options;
    soft_options.fixed_dt = 1.0F / 60.0F;
    soft_options.solver_substeps = waterlab::course_minimum_iterations;
    waterlab::SoftBodyCourse soft_bodies(soft_body_asset_path(), soft_options);
    require(soft_bodies.statistics().instance_count == waterlab::course_peg_count &&
            soft_bodies.statistics().voxels_per_instance == 1'000U,
        "render fixture did not create eight 1,000-voxel posts");
    const auto soft_view = soft_bodies.render_view();
    require(soft_view.vertex_count == waterlab::course_peg_count * 931U &&
            soft_view.triangle_count == waterlab::course_peg_count * 1'632U &&
            soft_view.node_count != 0U,
        "deformable post render hierarchy is incomplete");
    const float radius = 0.10F;
    const float3 point = make_float3(peg.x, waterlab::course_floor_y + 0.7F, peg.z - 0.8F);
    const meshprep::Aabb bounds{make_float3(point.x-radius, point.y-radius, point.z-radius),
        make_float3(point.x+radius, point.y+radius, point.z+radius)};
    float3* device_point = nullptr;
    meshprep::Aabb* device_bounds = nullptr;
    require(cudaMalloc(reinterpret_cast<void**>(&device_point), 2U*sizeof(point)) == cudaSuccess &&
            cudaMalloc(reinterpret_cast<void**>(&device_bounds), sizeof(bounds)) == cudaSuccess,
        "failed to allocate occlusion fixture");
    require(cudaMemcpy(device_point, &point, sizeof(point), cudaMemcpyHostToDevice) == cudaSuccess &&
            cudaMemcpy(device_bounds, &bounds, sizeof(bounds), cudaMemcpyHostToDevice) == cudaSuccess,
        "failed to upload occlusion fixture");
    meshprep::Workspace workspace;
    meshprep::Hierarchy particles, empty_skin;
    require(meshprep::build_hierarchy({device_bounds, 1U}, {}, workspace, particles).ok(),
        "failed to build occlusion hierarchy");
    waterlab::FluidVisuals visuals(1U);
    require(cudaMemset(device_point+1, 0, sizeof(point)) == cudaSuccess,
        "failed to clear occlusion velocity");
    (void)visuals.update(device_point, device_point+1, particles, 0.11F, {}, 0.0F);
    std::vector<waterlab::FoamParticle> bubbles(waterlab::FluidVisuals::foam_capacity);
    bubbles[0] = {make_float4(point.x,point.y,point.z+0.04F,0.1F),
        make_float4(0,0,0,2.0F), make_float4(0,0,1,0.014F)};
    visuals.restore_foam(bubbles, 1U);
    waterlab::Camera camera;
    camera.eye = make_float3(peg.x, point.y, 1.0F);
    camera.target = point;
    waterlab::RayTracer renderer;
    meshprep::NormalOutput empty_normals;
    constexpr std::uint32_t width = 160U, height = 120U;
    const auto render = [&](waterlab::FluidDisplay display,
                            waterlab::SoftBodyRenderView posts, bool course) {
        (void)renderer.render_hybrid({}, empty_normals, empty_skin, device_point, radius,
            particles, false, {}, camera, width, height, nullptr, course,
            visuals.view(display, true), posts);
        return std::vector<uchar4>(renderer.pixels(), renderer.pixels() + width * height);
    };
    const auto exposed_surface = render(waterlab::FluidDisplay::Surface, {}, false);
    const auto exposed_particles = render(waterlab::FluidDisplay::Particles, {}, false);
    require(differing_pixels(exposed_surface, exposed_particles) != 0U,
        "occlusion fixture did not expose water before adding a post");
    const auto soft_surface = render(waterlab::FluidDisplay::Surface, soft_view, false);
    const auto soft_particles = render(waterlab::FluidDisplay::Particles, soft_view, false);
    require(differing_pixels(soft_surface, soft_particles) == 0U &&
            differing_pixels(exposed_particles, soft_particles) > 500U,
        "water/foam behind a filled soft body remained visible");
    const auto rigid_surface = render(waterlab::FluidDisplay::Surface, {}, true);
    const auto rigid_particles = render(waterlab::FluidDisplay::Particles, {}, true);
    require(differing_pixels(rigid_surface, rigid_particles) == 0U &&
            differing_pixels(exposed_particles, rigid_particles) > 500U,
        "context-1 rigid post failed to occlude water and foam");
    // Structural fracture must not erase the material surface. Broken bonds
    // change dynamics while render triangles remain attached to voxel anchors.
    waterlab::SoftBodyState detached;
    soft_bodies.capture_state(detached);
    std::fill(detached.active_edges.begin(), detached.active_edges.end(), 0U);
    std::fill(detached.active_render_triangles.begin(), detached.active_render_triangles.end(), 0U);
    detached.statistics.broken_edge_count = static_cast<std::uint32_t>(detached.active_edges.size());
    soft_bodies.restore_state(detached);
    for (const bool course : {false, true}) {
        require(differing_pixels(
                render(waterlab::FluidDisplay::Particles, {}, course),
                render(waterlab::FluidDisplay::Particles, soft_bodies.render_view(), course)) > 500U,
            "broken soft-body graph erased its material surface");
    }
    cudaFree(device_bounds);
    cudaFree(device_point);
}

} // namespace

int main()
{
    int device_count = 0;
    const cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess || device_count == 0) {
        std::puts("SKIP meshprep-hybrid-tests: no CUDA device");
        return 77;
    }
    try {
        if (std::getenv("MESHPREP_VISUAL_SANITIZER_SMOKE") != nullptr) {
            test_fluid_visual_render_modes();
            test_soft_post_occlusion();
            std::puts("PASS meshprep-hybrid-tests visual rendering smoke");
            return 0;
        }
        if (std::getenv("MESHPREP_SANITIZER_SMOKE") != nullptr) {
            waterlab::HybridDroplet droplet;
            require_timings(droplet.step(make_float3(-400.0F, 0.0F, 0.0F), 5.0F));
            require_state(droplet);
            require_debug_forces_finite(droplet);
            std::puts("PASS meshprep-hybrid-tests sanitizer smoke");
            return 0;
        }
        test_idle_and_reset();
        test_dynamic_rectangle();
        test_runtime_physics_controls();
        test_particle_only_mode();
        test_state_capture_restore();
        test_wireframe_box_render();
        test_fluid_visual_render_modes();
        test_soft_post_occlusion();
        std::puts("PASS meshprep-hybrid-tests");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL meshprep-hybrid-tests: %s\n", error.what());
        return 1;
    }
}
