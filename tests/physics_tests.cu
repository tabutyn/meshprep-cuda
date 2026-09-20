// SPDX-License-Identifier: MIT
#include <parallel_mater/physics.hpp>

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <type_traits>
#include <vector>

#ifndef PARALLEL_MATER_PHYSICS_TEST_ASSET
#define PARALLEL_MATER_PHYSICS_TEST_ASSET "assets/softbody/checker_cylinder.msb"
#endif

namespace {

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

void test_contract() {
    using parallel_mater::physics::SoftBody;
    static_assert(std::is_default_constructible_v<SoftBody>);
    static_assert(std::is_move_constructible_v<SoftBody>);
    static_assert(!std::is_copy_constructible_v<SoftBody>);

    SoftBody body;
    require(!body.initialized(), "default soft body allocated state");
    require(!body.step({0.0F, -9.81F, 0.0F}), "uninitialized soft body accepted a step");

    parallel_mater::physics::SoftBodyOptions invalid;
    invalid.substeps = 0U;
    require(!body.initialize(PARALLEL_MATER_PHYSICS_TEST_ASSET, invalid),
            "soft body accepted zero substeps");
}

void test_standalone_solver() {
    parallel_mater::physics::SoftBodyOptions options;
    options.instance_count = 2U;
    options.substeps = 2U;
    options.constraint_iterations = 4U;
    options.instance_origins[0] = {-0.4F, 1.5F, 0.0F};
    options.instance_origins[1] = {0.4F, 1.5F, 0.0F};

    parallel_mater::physics::SoftBody body;
    require(
        parallel_mater::physics::SoftBody::create(PARALLEL_MATER_PHYSICS_TEST_ASSET, options, body)
            .ok(),
        "standalone soft body failed to initialize");
    require(body.initialized(), "soft body did not retain initialization");

    const auto nodes = body.nodes();
    const auto bonds = body.bonds();
    const auto surface = body.surface();
    require(nodes.instance_count == 2U && nodes.node_count > 0U &&
                nodes.external_impulses != nullptr && nodes.position_corrections != nullptr &&
                nodes.colors != nullptr,
            "coupling view omitted writable device buffers");
    require(bonds.instance_count == 2U && bonds.bonds_per_instance > 0U && bonds.bonds != nullptr &&
                bonds.bond_active != nullptr,
            "bond view omitted fixed topology");
    require(surface.mesh.vertex_count > 0U && surface.mesh.triangle_count > 0U &&
                surface.vertex_normals != nullptr && surface.vertex_colors != nullptr &&
                surface.hierarchy_node_count > 0U,
            "surface view omitted render or hierarchy data");

    require(body.step({0.0F, -9.81F, 0.0F}).ok(), "standalone fixed step failed");
    auto statistics = body.statistics();
    require(statistics.frame_index == 1U && statistics.instance_count == 2U &&
                statistics.node_count == nodes.node_count && statistics.finite_failure_count == 0U,
            "standalone statistics are inconsistent");

    require(body.begin_frame().ok(), "manual frame did not begin");
    const float dt = options.timestep / static_cast<float>(options.substeps);
    for (std::uint32_t i = 0U; i < options.substeps; ++i) {
        require(body.prepare_substep(dt, {0.0F, 0.0F, 0.0F}).ok(),
                "manual substep prediction failed");
        const auto coupling = body.nodes();
        require(coupling.external_impulses != nullptr,
                "manual coupling phase omitted impulse output");
        require(body.finish_substep(dt, {0.0F, 0.0F, 0.0F}).ok(),
                "manual substep completion failed");
    }
    parallel_mater::physics::SoftBodyTimings timings;
    require(body.finish_frame(timings).ok(), "manual frame completion failed");
    require(timings.gpu_total_ms() >= 0.0F && std::isfinite(timings.gpu_total_ms()),
            "manual frame returned invalid timings");
    require(body.statistics().frame_index == 2U, "manual frame did not advance statistics");

    auto material = body.material();
    material.spring_stiffness *= 1.1F;
    require(body.set_material(material).ok(), "live material update failed");
    require(body.set_constraint_iterations(6U) && body.set_substeps(3U) &&
                body.set_strength_multiplier(1.25F) && body.set_node_mass(0.075F),
            "live solver controls failed");
    require(body.options().constraint_iterations == 6U && body.options().substeps == 3U &&
                body.options().strength_multiplier == 1.25F && body.options().node_mass == 0.075F &&
                body.node_mass() == 0.075F,
            "live solver controls were not reflected in options");
    material.maximum_speed = -1.0F;
    require(!body.set_material(material) && !body.set_substeps(0U) &&
                !body.set_constraint_iterations(0U) && !body.set_strength_multiplier(0.0F) &&
                !body.set_node_mass(0.0F),
            "live solver controls accepted invalid values");
    require(body.reset() && body.statistics().frame_index == 0U, "standalone reset failed");
}

std::vector<std::byte> read_asset_bytes() {
    std::ifstream input(PARALLEL_MATER_PHYSICS_TEST_ASSET, std::ios::binary | std::ios::ate);
    require(static_cast<bool>(input), "could not open soft-body test asset");
    const auto size = static_cast<std::size_t>(input.tellg());
    std::vector<std::byte> bytes(size);
    input.seekg(0, std::ios::beg);
    input.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    require(static_cast<bool>(input), "could not read soft-body test asset");
    return bytes;
}

void test_memory_asset_and_protocol() {
    const auto bytes = read_asset_bytes();
    parallel_mater::physics::SoftBody body;
    const parallel_mater::physics::SoftBodyAssetView asset{bytes.data(), bytes.size()};
    require(body.initialize(asset).ok(), "memory-backed soft body failed to initialize");

    parallel_mater::physics::Completion completion;
    parallel_mater::physics::FrameOptions frame;
    frame.substeps = 2U;
    frame.acceleration = {0.0F, -9.81F, 0.0F};
    require(body.advance_async(frame, completion).ok(), "soft-body async frame failed to enqueue");
    require(completion.pending(), "soft-body completion was not recorded");
    require(completion.wait().ok(), "soft-body async frame failed");
    parallel_mater::physics::Completion telemetry_completion;
    require(body.collect_telemetry_async(telemetry_completion).ok() &&
                telemetry_completion.wait().ok(),
            "soft-body telemetry failed to complete");
    parallel_mater::physics::SoftBodyTelemetry telemetry;
    require(body.resolve_telemetry(telemetry).ok(), "soft-body telemetry failed to resolve");
    require(body.statistics().frame_index == 1U, "soft-body protocol did not advance one frame");
}

void test_independent_owners() {
    using namespace parallel_mater::physics;
    static_assert(FrameSolver<Fluid>);
    static_assert(FrameSolver<Cloth>);
    static_assert(FrameSolver<Rope>);
    static_assert(FrameSolver<RigidBody>);

    std::vector<FluidParticle> particles;
    for (std::uint32_t z = 0U; z < 2U; ++z) {
        for (std::uint32_t y = 0U; y < 2U; ++y) {
            for (std::uint32_t x = 0U; x < 2U; ++x) {
                particles.push_back(
                    {make_float3(0.05F * static_cast<float>(x), 0.05F * static_cast<float>(y),
                                 0.05F * static_cast<float>(z)),
                     {}});
            }
        }
    }
    Fluid fluid;
    require(fluid.initialize(particles).ok(), "fluid owner failed to initialize");
    Completion fluid_completion;
    FrameOptions fluid_frame;
    fluid_frame.substeps = 2U;
    fluid_frame.acceleration = {0.0F, -1.0F, 0.0F};
    require(fluid.advance_async(fluid_frame, fluid_completion).ok() && fluid_completion.wait().ok(),
            "fluid owner failed to advance");
    require(fluid.collect_statistics().ok(), "fluid telemetry failed to collect");
    require(fluid.statistics().frame_index == 1U &&
                fluid.particles().particle_count == particles.size(),
            "fluid owner reported inconsistent state");
    require(fluid.coupling_points().count == particles.size(),
            "fluid owner omitted common coupling view");

    ClothOptions cloth_options;
    cloth_options.columns = 6U;
    cloth_options.rows = 5U;
    cloth_options.solver.substeps = 1U;
    Cloth cloth;
    require(cloth.initialize(cloth_options).ok(), "cloth owner failed to initialize");
    require(cloth.advance(FrameOptions{1.0F / 60.0F, 1U, {0.0F, -1.0F, 0.0F}}).ok(),
            "cloth owner failed to advance");
    require(cloth.nodes().node_count == 30U, "cloth owner produced unexpected topology");
    require(cloth.coupling_points().count == cloth.nodes().node_count,
            "cloth owner omitted common coupling view");

    RopeOptions rope_options;
    rope_options.node_count = 12U;
    rope_options.direction = {0.0F, -1.0F, 0.0F};
    rope_options.solver.substeps = 1U;
    Rope rope;
    require(rope.initialize(rope_options).ok(), "rope owner failed to initialize");
    require(rope.advance(FrameOptions{1.0F / 60.0F, 1U, {0.0F, -1.0F, 0.0F}}).ok(),
            "rope owner failed to advance");
    require(rope.nodes().node_count == rope_options.node_count,
            "rope owner produced unexpected topology");

    RigidBody rigid;
    require(rigid.initialize().ok(), "rigid-body owner failed to initialize");
    const FrameOptions rigid_frame{1.0F / 60.0F, 2U, {0.0F, -9.81F, 0.0F}};
    require(rigid.begin_frame(rigid_frame).ok(), "rigid-body frame failed to begin");
    for (std::uint32_t i = 0U; i < rigid_frame.substeps; ++i) {
        const auto substep = substep_context(rigid_frame, i);
        require(rigid.prepare_substep(substep).ok(), "rigid-body substep failed to prepare");
        require(rigid.apply_force({0.0F, 20.0F, 0.0F}, {0.5F, 0.0F, 0.0F}).ok(),
                "rigid-body force was rejected");
        require(rigid.finish_substep(substep).ok(), "rigid-body substep failed to finish");
    }
    Completion rigid_completion;
    require(rigid.finish_frame(rigid_completion).ok() && rigid_completion.wait().ok(),
            "rigid-body frame failed to finish");
    require(rigid.state().position.y > 0.0F, "rigid-body force did not affect translation");

    const FrameOptions coupled_frame{1.0F / 60.0F, 1U, {}};
    const parallel_mater::Status expected_failure{parallel_mater::StatusCode::invalid_argument,
                                                  cudaSuccess,
                                                  "intentional coupling failure"};
    require(!advance_coupled(
                 coupled_frame,
                 [&](SubstepContext, cudaStream_t) noexcept { return expected_failure; },
                 nullptr, fluid, rigid),
            "coupled frame did not report its coupling failure");
    require(advance_coupled(
                coupled_frame,
                [](SubstepContext, cudaStream_t) noexcept { return parallel_mater::Status{}; },
                nullptr, fluid, rigid)
                .ok(),
            "coupled frame failure left an owner unusable");
}

void test_generic_coupling() {
    using namespace parallel_mater::physics;
    Collider collider;
    collider.shape = ColliderShape::box;
    collider.position = make_float3(1.0F, 2.0F, 3.0F);
    collider.dimensions = make_float3(0.5F, 0.25F, 0.75F);
    ColliderSet colliders;
    require(colliders.update(std::span<const Collider>(&collider, 1U)).ok() &&
                colliders.view().count == 1U,
            "generic collider set failed to upload");
    require(colliders.reserve(4U).ok() && colliders.view().count == 1U,
            "growing collider storage discarded its logical size");
    Collider preserved{};
    cudaMemcpy(&preserved, colliders.view().data, sizeof(Collider), cudaMemcpyDeviceToHost);
    require(preserved.position.x == collider.position.x &&
                preserved.position.y == collider.position.y &&
                preserved.position.z == collider.position.z,
            "growing collider storage discarded uploaded contents");

    const ConstraintRecord host_records[] = {
        {0U, 2U, make_float3(1.0F, 0.0F, 0.0F), {},
         make_float4(0.0F, 0.0F, 1.0F, 1.0F), 0.5F},
        {1U, 0U, make_float3(0.0F, 2.0F, 0.0F), make_float3(0.0F, 0.25F, 0.0F)},
        {0U, 1U, make_float3(3.0F, 0.0F, 0.0F), make_float3(0.5F, 0.0F, 0.0F),
         make_float4(1.0F, 0.0F, 0.0F, 1.0F), 0.5F}};
    ConstraintRecord *records{};
    float3 *positions{};
    float3 *velocities{};
    float3 *impulses{};
    float3 *corrections{};
    float4 *colors{};
    require(cudaMalloc(&records, sizeof(host_records)) == cudaSuccess &&
                cudaMalloc(&positions, 2U * sizeof(float3)) == cudaSuccess &&
                cudaMalloc(&velocities, 2U * sizeof(float3)) == cudaSuccess &&
                cudaMalloc(&impulses, 2U * sizeof(float3)) == cudaSuccess &&
                cudaMalloc(&corrections, 2U * sizeof(float3)) == cudaSuccess &&
                cudaMalloc(&colors, 2U * sizeof(float4)) == cudaSuccess,
            "constraint fixture allocation failed");
    cudaMemcpy(records, host_records, sizeof(host_records), cudaMemcpyHostToDevice);
    cudaMemset(positions, 0, 2U * sizeof(float3));
    cudaMemset(velocities, 0, 2U * sizeof(float3));
    cudaMemset(impulses, 0, 2U * sizeof(float3));
    cudaMemset(corrections, 0, 2U * sizeof(float3));
    cudaMemset(colors, 0, 2U * sizeof(float4));
    ConstraintBatch batch;
    const PointStateView target{positions, velocities, impulses, corrections, 2U, 0.1F, 1.0F,
                                colors};
    require(batch.apply({records, 3U}, target).ok(), "deterministic constraint batch failed");
    float3 host_impulses[2]{};
    float3 host_corrections[2]{};
    float4 host_colors[2]{};
    cudaMemcpy(host_impulses, impulses, sizeof(host_impulses), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_corrections, corrections, sizeof(host_corrections), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_colors, colors, sizeof(host_colors), cudaMemcpyDeviceToHost);
    cudaFree(colors);
    cudaFree(corrections);
    cudaFree(impulses);
    cudaFree(velocities);
    cudaFree(positions);
    cudaFree(records);
    require(host_impulses[0].x == 4.0F && host_impulses[1].y == 2.0F &&
                host_corrections[0].x == 0.5F && host_corrections[1].y == 0.25F,
            "constraint gather did not preserve ordered sums");
    require(host_colors[0].x == 0.25F && host_colors[0].z == 0.5F &&
                host_colors[0].w == 0.75F,
            "constraint gather did not preserve ordered paint blending");
}

void test_contact_painting() {
    using namespace parallel_mater::physics;
    float3 host_position = make_float3(0.9F, 0.0F, 0.0F);
    float3 host_velocity = make_float3(-1.0F, 0.0F, 0.0F);
    float4 host_color{};
    float3 *positions{};
    float3 *velocities{};
    float3 *impulses{};
    float3 *corrections{};
    float4 *colors{};
    require(cudaMalloc(&positions, sizeof(float3)) == cudaSuccess &&
                cudaMalloc(&velocities, sizeof(float3)) == cudaSuccess &&
                cudaMalloc(&impulses, sizeof(float3)) == cudaSuccess &&
                cudaMalloc(&corrections, sizeof(float3)) == cudaSuccess &&
                cudaMalloc(&colors, sizeof(float4)) == cudaSuccess,
            "contact paint fixture allocation failed");
    cudaMemcpy(positions, &host_position, sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(velocities, &host_velocity, sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemset(impulses, 0, sizeof(float3));
    cudaMemset(corrections, 0, sizeof(float3));
    cudaMemset(colors, 0, sizeof(float4));
    Collider collider;
    collider.shape = ColliderShape::sphere;
    collider.dimensions = make_float3(1.0F, 0.0F, 0.0F);
    collider.paint_color = make_float4(0.1F, 0.8F, 0.2F, 1.0F);
    collider.paint_amount = 1.0F;
    ColliderSet set;
    require(set.update(std::span<const Collider>(&collider, 1U)).ok(),
            "paint collider upload failed");
    PointStateView points{positions, velocities, impulses, corrections, 1U, 0.1F, 1.0F, colors};
    require(apply_colliders(points, set.view(), 1.0F / 60.0F).ok(),
            "generic collider contact failed");
    float3 correction{};
    cudaMemcpy(&correction, corrections, sizeof(float3), cudaMemcpyDeviceToHost);
    cudaMemcpy(&host_color, colors, sizeof(float4), cudaMemcpyDeviceToHost);
    require(correction.x > 0.19F && host_color.y == 0.8F,
            "collider contact did not project and paint the point");
    struct ShapeFixture {
        ColliderShape shape;
        float3 point;
        float3 dimensions;
    };
    const ShapeFixture fixtures[] = {
        {ColliderShape::box, make_float3(0.95F, 0.0F, 0.0F), make_float3(1.0F, 1.0F, 1.0F)},
        {ColliderShape::plane, make_float3(0.0F, 0.05F, 0.0F), {}},
        {ColliderShape::capsule, make_float3(0.45F, 0.0F, 0.0F),
         make_float3(0.5F, 0.5F, 0.0F)}};
    for (const ShapeFixture &fixture : fixtures) {
        collider.shape = fixture.shape;
        collider.dimensions = fixture.dimensions;
        require(set.update(std::span<const Collider>(&collider, 1U)).ok(),
                "analytic collider fixture upload failed");
        cudaMemcpy(positions, &fixture.point, sizeof(float3), cudaMemcpyHostToDevice);
        cudaMemset(corrections, 0, sizeof(float3));
        require(apply_colliders(points, set.view(), 1.0F / 60.0F).ok(),
                "analytic collider shape contact failed");
        cudaMemcpy(&correction, corrections, sizeof(float3), cudaMemcpyDeviceToHost);
        require(correction.x * correction.x + correction.y * correction.y +
                        correction.z * correction.z >
                    0.0F,
                "analytic collider shape produced no projection");
    }
    cudaFree(colors);
    cudaFree(corrections);
    cudaFree(impulses);
    cudaFree(velocities);
    cudaFree(positions);

    PaintSurface surface;
    require(surface.initialize({8U, 8U, {}, false}).ok(),
            "paint surface failed to initialize");
    const PaintStamp stamp{make_float2(0.5F, 0.5F), 0.35F,
                           make_float4(0.0F, 1.0F, 0.0F, 1.0F), 1.0F};
    PaintStamp *device_stamp{};
    require(cudaMalloc(&device_stamp, sizeof(PaintStamp)) == cudaSuccess,
            "paint stamp allocation failed");
    cudaMemcpy(device_stamp, &stamp, sizeof(PaintStamp), cudaMemcpyHostToDevice);
    require(surface.apply({device_stamp, 1U}).ok(), "paint surface stamp failed");
    float4 texels[64]{};
    cudaMemcpy(texels, surface.view().colors, sizeof(texels), cudaMemcpyDeviceToHost);
    cudaFree(device_stamp);
    require(texels[4U * 8U + 4U].y > 0.0F && texels[0].y == 0.0F,
            "paint surface did not retain a localized stamp");
}

} // namespace

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::puts("SKIP: no CUDA device");
        return 77;
    }
    try {
        test_contract();
        test_standalone_solver();
        test_memory_asset_and_protocol();
        test_independent_owners();
        test_generic_coupling();
        test_contact_painting();
        std::puts("all public physics tests passed");
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
