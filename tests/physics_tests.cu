// SPDX-License-Identifier: MIT
#include <meshprep/physics.hpp>

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <type_traits>

#ifndef MESHPREP_PHYSICS_TEST_ASSET
#define MESHPREP_PHYSICS_TEST_ASSET "assets/softbody/checker_cylinder.msb"
#endif

namespace {

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

void test_contract()
{
    using meshprep::physics::SoftBody;
    static_assert(std::is_default_constructible_v<SoftBody>);
    static_assert(std::is_move_constructible_v<SoftBody>);
    static_assert(!std::is_copy_constructible_v<SoftBody>);

    SoftBody body;
    require(!body.initialized(), "default soft body allocated state");
    require(!body.step({0.0F, -9.81F, 0.0F}),
        "uninitialized soft body accepted a step");

    meshprep::physics::SoftBodyOptions invalid;
    invalid.substeps = 0U;
    require(!body.initialize(MESHPREP_PHYSICS_TEST_ASSET, invalid),
        "soft body accepted zero substeps");
}

void test_standalone_solver()
{
    meshprep::physics::SoftBodyOptions options;
    options.instance_count = 2U;
    options.substeps = 2U;
    options.constraint_iterations = 4U;
    options.instance_origins[0] = {-0.4F, 1.5F, 0.0F};
    options.instance_origins[1] = {0.4F, 1.5F, 0.0F};
    options.render_internal_members = true;

    meshprep::physics::SoftBody body;
    require(meshprep::physics::SoftBody::create(
                MESHPREP_PHYSICS_TEST_ASSET, options, body).ok(),
        "standalone soft body failed to initialize");
    require(body.initialized(), "soft body did not retain initialization");

    const auto nodes = body.nodes();
    const auto bonds = body.bonds();
    const auto surface = body.surface();
    require(nodes.instance_count == 2U && nodes.node_count > 0U &&
            nodes.external_impulses != nullptr &&
            nodes.position_corrections != nullptr,
        "coupling view omitted writable device buffers");
    require(bonds.instance_count == 2U && bonds.bonds_per_instance > 0U &&
            bonds.bonds != nullptr && bonds.bond_active != nullptr,
        "bond view omitted fixed topology");
    require(surface.mesh.vertex_count > 0U && surface.mesh.triangle_count > 0U &&
            surface.vertex_normals != nullptr &&
            surface.hierarchy_node_count > 0U,
        "surface view omitted render or hierarchy data");

    require(body.step({0.0F, -9.81F, 0.0F}).ok(),
        "standalone fixed step failed");
    auto statistics = body.statistics();
    require(statistics.frame_index == 1U &&
            statistics.instance_count == 2U &&
            statistics.node_count == nodes.node_count &&
            statistics.finite_failure_count == 0U,
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
    meshprep::physics::SoftBodyTimings timings;
    require(body.finish_frame(timings).ok(), "manual frame completion failed");
    require(timings.gpu_total_ms() >= 0.0F && std::isfinite(timings.gpu_total_ms()),
        "manual frame returned invalid timings");
    require(body.statistics().frame_index == 2U,
        "manual frame did not advance statistics");

    auto material = body.material();
    material.spring_stiffness *= 1.1F;
    require(body.set_material(material).ok(), "live material update failed");
    require(body.set_constraint_iterations(6U) && body.set_substeps(3U) &&
            body.set_strength_multiplier(1.25F),
        "live solver controls failed");
    require(body.options().constraint_iterations == 6U &&
            body.options().substeps == 3U &&
            body.options().strength_multiplier == 1.25F,
        "live solver controls were not reflected in options");
    require(body.reset() && body.statistics().frame_index == 0U,
        "standalone reset failed");
}

} // namespace

int main()
{
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::puts("SKIP: no CUDA device");
        return 77;
    }
    try {
        test_contract();
        test_standalone_solver();
        std::puts("all public physics tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
