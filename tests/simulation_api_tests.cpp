// SPDX-License-Identifier: MIT
#include <meshprep/simulation.hpp>

#include <array>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string_view>
#include <type_traits>

namespace {

using meshprep::sim::Component;
using meshprep::sim::ExampleContext;

constexpr std::uint32_t bits(Component value) noexcept
{
    return static_cast<std::uint32_t>(value);
}

struct ExpectedContext {
    ExampleContext id;
    char key;
    std::string_view slug;
    std::string_view title;
    Component components;
};

constexpr std::array<ExpectedContext, 10> expected_contexts{{
    {ExampleContext::water, '1', "water", "Water",
        Component::fluid_particles | Component::rigid_bodies},
    {ExampleContext::cloth, '2', "cloth", "Cloth",
        Component::cloth | Component::rigid_bodies},
    {ExampleContext::soft_body, '3', "softbody", "Softbody",
        Component::soft_body | Component::rigid_bodies},
    {ExampleContext::rope, '4', "rope", "Rope",
        Component::rope | Component::rigid_bodies},
    {ExampleContext::water_cloth, '5', "water-cloth", "Water-Cloth",
        Component::fluid_particles | Component::water_skin |
            Component::cloth | Component::rigid_bodies},
    {ExampleContext::water_soft_body, '6', "water-softbody", "Water-Softbody",
        Component::soft_body | Component::fluid_particles | Component::rigid_bodies},
    {ExampleContext::water_rope, '7', "water-rope", "Water-Rope",
        Component::fluid_particles | Component::rope | Component::rigid_bodies},
    {ExampleContext::cloth_soft_body, '8', "cloth-softbody", "Cloth-Softbody",
        Component::soft_body | Component::cloth | Component::rigid_bodies},
    {ExampleContext::cloth_rope, '9', "cloth-rope", "Cloth-Rope",
        Component::cloth | Component::rope | Component::rigid_bodies},
    {ExampleContext::soft_body_rope, '0', "softbody-rope", "Softbody-Rope",
        Component::soft_body | Component::rope | Component::rigid_bodies},
}};

int failures{};

void expect(bool condition, const char* message)
{
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", message);
    ++failures;
}

void test_context_catalog()
{
    expect(meshprep::sim::example_contexts.size() == expected_contexts.size(),
        "catalog must contain the ten component and pair contexts");

    for (std::size_t index = 0; index < expected_contexts.size(); ++index) {
        const auto& expected = expected_contexts[index];
        const auto& actual = meshprep::sim::example_contexts[index];
        expect(actual.id == expected.id, "context id/order changed");
        expect(actual.key == expected.key, "context key/order changed");
        expect(actual.slug == expected.slug, "context slug changed");
        expect(actual.title == expected.title, "context title changed");
        expect(bits(actual.components) == bits(expected.components),
            "context component set changed or gained an unexpected component");
        expect(meshprep::sim::find_example_context(expected.key) == &actual,
            "lookup must return the catalog entry, not a copy");
    }

    const auto* softbody_rope = meshprep::sim::find_example_context('0');
    expect(softbody_rope != nullptr &&
            softbody_rope->id == ExampleContext::soft_body_rope,
        "context 0 must expose Softbody-Rope");
    const auto* bridge = meshprep::sim::find_example_context('9');
    expect(bridge != nullptr && bridge->id == ExampleContext::cloth_rope,
        "context 9 must expose Cloth-Rope");
    expect(meshprep::sim::find_example_context('x') == nullptr,
        "non-number context key must be rejected");
    expect(!meshprep::sim::has_component(Component::none, Component::cloth),
        "empty component set must not report cloth");
    expect(meshprep::sim::has_component(
               Component::cloth | Component::soft_body, Component::cloth),
        "component composition must retain the left component");
    expect(meshprep::sim::has_component(
               Component::cloth | Component::soft_body, Component::soft_body),
        "component composition must retain the right component");
}

void test_fixed_step_contract()
{
    constexpr meshprep::sim::FixedStepOptions defaults{};
    static_assert(defaults.timestep == 1.0F / 60.0F);
    static_assert(meshprep::sim::valid(defaults));
    static_assert(meshprep::sim::valid({1.0F / 120.0F}));
    static_assert(!meshprep::sim::valid({0.0F}));
    static_assert(!meshprep::sim::valid({-1.0F / 60.0F}));

    expect(meshprep::sim::valid(defaults), "default fixed step must be valid");
    expect(!meshprep::sim::valid(
               {std::numeric_limits<float>::quiet_NaN()}),
        "non-finite fixed timestep must be rejected by a solver boundary");
    expect(!meshprep::sim::valid(
               {std::numeric_limits<float>::infinity()}),
        "infinite fixed timestep must be rejected by a solver boundary");

    meshprep::sim::GallerySimulationOptions options;
    expect(!options.gravity_override.has_value() &&
            !options.solver_iterations_override.has_value(),
        "gallery defaults must select a recipe without implicit overrides");
    options.gravity_override = float3{1.0F, 2.0F, 3.0F};
    options.solver_iterations_override = 3U;
    expect(options.gravity_override->y == 2.0F &&
            *options.solver_iterations_override == 3U,
        "gallery physics overrides must retain explicit values");
}

void test_portable_game_contract()
{
    auto config = meshprep::sim::SimulationConfig::for_level(
        ExampleContext::water_rope);
    config.timestep(1.0F / 120.0F).iterations(8U).particles(12'000U)
        .cloth_resolution(4U);
    expect(meshprep::sim::validate(config) == meshprep::sim::ConfigError::none,
        "fluent portable configuration must validate");
    config.cloth_resolution(0U);
    expect(meshprep::sim::validate(config) ==
            meshprep::sim::ConfigError::invalid_cloth_detail,
        "portable configuration must reject zero cloth detail");

    meshprep::sim::LevelMetrics metrics;
    metrics.painted_fraction = 0.42F;
    const auto paint = meshprep::sim::evaluate(
        ExampleContext::water, metrics);
    expect(paint.normalized == 0.42F && !paint.won,
        "paint goal must expose continuous normalized progress");
    metrics.rope_turns = 3.0F;
    expect(meshprep::sim::evaluate(ExampleContext::rope, metrics).won,
        "rope goal must win at three complete turns");
    metrics.treasure_caught = true;
    metrics.treasure_lift_progress = 1.0F;
    expect(meshprep::sim::evaluate(ExampleContext::water_rope, metrics).won,
        "fishing goal must win after the caught treasure reaches the top");

    meshprep::sim::Campaign campaign(ExampleContext::cloth);
    metrics = {};
    metrics.broken_connections = 1U;
    meshprep::sim::LevelProgress progress;
    for (int frame = 0; frame < 90; ++frame) progress = campaign.update(metrics);
    expect(progress.advanced && campaign.current() == ExampleContext::soft_body,
        "campaign must advance after holding a completed goal");
}

void test_owning_simulation_contract()
{
    static_assert(std::is_default_constructible_v<meshprep::sim::GallerySimulation>);
    static_assert(std::is_move_constructible_v<meshprep::sim::GallerySimulation>);
    static_assert(std::is_move_assignable_v<meshprep::sim::GallerySimulation>);
    static_assert(!std::is_copy_constructible_v<meshprep::sim::GallerySimulation>);
    static_assert(!std::is_copy_assignable_v<meshprep::sim::GallerySimulation>);

    expect(!meshprep::sim::requires_soft_body_asset(ExampleContext::water),
        "procedural particle bowl must not require an asset");
    expect(!meshprep::sim::requires_soft_body_asset(ExampleContext::cloth),
        "procedural cloth must not require an asset");
    expect(!meshprep::sim::requires_soft_body_asset(ExampleContext::water_cloth),
        "rigid water course must not request a soft-body asset");
    expect(meshprep::sim::requires_soft_body_asset(ExampleContext::soft_body),
        "soft-body context must request its authored cylinder asset");
    expect(!meshprep::sim::requires_soft_body_asset(ExampleContext::water_soft_body),
        "procedural water-wheel cross must not request the retired cylinder asset");
    expect(!meshprep::sim::requires_soft_body_asset(ExampleContext::rope),
        "procedural rope must not request an external asset");

    meshprep::sim::GallerySimulation empty;
    expect(!empty.initialized(), "default simulation must not allocate CUDA state");
    expect(!empty.step().ok(), "uninitialized simulation step must return an error");
    expect(empty.render_view().particle_system_count == 0U,
        "uninitialized simulation must expose an empty frame");
}

void test_borrowed_render_views()
{
    static_assert(std::is_trivially_copyable_v<meshprep::sim::ParticleRenderView>);
    static_assert(std::is_trivially_copyable_v<meshprep::sim::SurfaceRenderView>);
    static_assert(std::is_trivially_copyable_v<meshprep::sim::RigidBodyRenderView>);
    static_assert(std::is_trivially_copyable_v<meshprep::sim::LatticeBond>);
    static_assert(std::is_trivially_copyable_v<meshprep::sim::LatticeRenderView>);
    static_assert(std::is_trivially_copyable_v<meshprep::sim::FrameRenderView>);

    float3 positions[3]{};
    float3 velocities[3]{};
    uint3 triangles[1]{};
    float3 normals[3]{};
    float2 texcoords[3]{};
    std::uint8_t triangle_active[1]{1U};
    std::uint32_t flags[3]{};
    meshprep::sim::LatticeBond bonds[1]{{{0U, 1U}, 0.05F}};
    std::uint8_t bond_active[1]{1U};

    const meshprep::sim::ParticleRenderView particles{
        positions, velocities, 3U, 0.025F};
    const meshprep::sim::SurfaceRenderView surface{
        {positions, 3U, triangles, 1U}, normals, texcoords, triangle_active};
    const meshprep::sim::RigidBodyRenderView rigid{
        {positions, 3U, triangles, 1U},
        {1.0F, 2.0F, 3.0F},
        {0.0F, 0.0F, 0.0F, 1.0F},
        {2.0F, 3.0F, 4.0F}};

    const std::array particle_systems{particles};
    const std::array surfaces{surface};
    const std::array rigid_bodies{rigid};
    const std::array lattices{meshprep::sim::LatticeRenderView{
        positions, flags, bonds, bond_active, 3U, 3U, 1U, 1U, 0.025F}};
    const meshprep::sim::FrameRenderView frame{
        particle_systems.data(), 1U,
        surfaces.data(), 1U,
        rigid_bodies.data(), 1U,
        lattices.data(), 1U};

    expect(frame.particle_systems == particle_systems.data(),
        "frame must borrow the particle-view array");
    expect(frame.surfaces == surfaces.data(),
        "frame must borrow the surface-view array");
    expect(frame.rigid_bodies == rigid_bodies.data(),
        "frame must borrow the rigid-body-view array");
    expect(frame.lattices == lattices.data() && frame.lattice_count == 1U &&
            frame.lattices[0].positions == positions &&
            frame.lattices[0].flags == flags && frame.lattices[0].bonds == bonds &&
            frame.lattices[0].bond_active == bond_active,
        "frame must borrow the complete lattice and live bond mask");
    expect(frame.particle_system_count == 1U && frame.surface_count == 1U &&
            frame.rigid_body_count == 1U,
        "frame must preserve borrowed view counts");
    expect(frame.particle_systems[0].positions == positions &&
            frame.particle_systems[0].velocities == velocities &&
            frame.particle_systems[0].count == 3U,
        "particle view must preserve borrowed device pointers and count");
    expect(frame.surfaces[0].mesh.positions == positions &&
            frame.surfaces[0].mesh.triangles == triangles &&
            frame.surfaces[0].triangle_active == triangle_active,
        "surface view must preserve borrowed geometry pointers");
    expect(frame.rigid_bodies[0].translation.x == 1.0F &&
            frame.rigid_bodies[0].scale.z == 4.0F &&
            frame.rigid_bodies[0].orientation.w == 1.0F,
        "rigid view must preserve its transform");

    constexpr meshprep::sim::FrameRenderView empty{};
    static_assert(empty.particle_systems == nullptr && empty.particle_system_count == 0U);
    static_assert(empty.surfaces == nullptr && empty.surface_count == 0U);
    static_assert(empty.rigid_bodies == nullptr && empty.rigid_body_count == 0U);
    static_assert(empty.lattices == nullptr && empty.lattice_count == 0U);
}

} // namespace

int main()
{
    test_context_catalog();
    test_fixed_step_contract();
    test_portable_game_contract();
    test_owning_simulation_contract();
    test_borrowed_render_views();

    if (failures != 0) {
        std::fprintf(stderr, "%d simulation API contract test(s) failed\n", failures);
        return 1;
    }
    std::puts("simulation API contract tests passed");
    return 0;
}
