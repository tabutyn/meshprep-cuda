// SPDX-License-Identifier: MIT
#include "gallery.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string_view>
#include <type_traits>

namespace {

using parallel_mater::examples::Component;
using parallel_mater::examples::SimulationRecipe;

constexpr std::uint32_t bits(Component value) noexcept
{
    return static_cast<std::uint32_t>(value);
}

struct ExpectedRecipe {
    SimulationRecipe recipe;
    std::string_view slug;
    std::string_view title;
    Component components;
};

constexpr std::array<ExpectedRecipe, 15> expected_recipes{{
    {SimulationRecipe::water, "water", "Water",
        Component::fluid_particles | Component::rigid_bodies},
    {SimulationRecipe::cloth, "cloth", "Cloth",
        Component::cloth | Component::rigid_bodies},
    {SimulationRecipe::soft_body, "softbody", "Softbody",
        Component::soft_body | Component::rigid_bodies},
    {SimulationRecipe::rope, "rope", "Rope",
        Component::rope | Component::rigid_bodies},
    {SimulationRecipe::smoke, "smoke", "Smoke",
        Component::smoke | Component::rigid_bodies},
    {SimulationRecipe::water_cloth, "water-cloth", "Water-Cloth",
        Component::fluid_particles | Component::water_skin |
            Component::cloth | Component::rigid_bodies},
    {SimulationRecipe::water_soft_body, "water-softbody", "Water-Softbody",
        Component::soft_body | Component::fluid_particles | Component::rigid_bodies},
    {SimulationRecipe::water_rope, "water-rope", "Water-Rope",
        Component::fluid_particles | Component::rope | Component::rigid_bodies},
    {SimulationRecipe::fluid_smoke, "fluid-smoke", "Fluid-Smoke",
        Component::fluid_particles | Component::smoke},
    {SimulationRecipe::cloth_soft_body, "cloth-softbody", "Cloth-Softbody",
        Component::soft_body | Component::cloth | Component::rigid_bodies},
    {SimulationRecipe::cloth_rope, "cloth-rope", "Cloth-Rope",
        Component::cloth | Component::rope | Component::rigid_bodies},
    {SimulationRecipe::cloth_smoke, "cloth-smoke", "Cloth-Smoke",
        Component::cloth | Component::smoke | Component::rigid_bodies},
    {SimulationRecipe::soft_body_rope, "softbody-rope", "Softbody-Rope",
        Component::soft_body | Component::rope | Component::rigid_bodies},
    {SimulationRecipe::soft_body_smoke, "softbody-smoke", "Softbody-Smoke",
        Component::soft_body | Component::smoke | Component::rigid_bodies},
    {SimulationRecipe::rope_smoke, "rope-smoke", "Rope-Smoke",
        Component::rope | Component::smoke | Component::rigid_bodies},
}};

int failures{};

void expect(bool condition, const char* message)
{
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", message);
    ++failures;
}

void test_recipe_catalog()
{
    expect(parallel_mater::examples::simulation_recipes.size() == expected_recipes.size(),
        "catalog must contain component, pair, and smoke recipes");

    for (std::size_t index = 0; index < expected_recipes.size(); ++index) {
        const auto& expected = expected_recipes[index];
        const auto& actual = parallel_mater::examples::simulation_recipes[index];
        expect(actual.recipe == expected.recipe, "recipe identity/order changed");
        expect(actual.slug == expected.slug, "recipe slug changed");
        expect(actual.title == expected.title, "recipe title changed");
        expect(bits(actual.components) == bits(expected.components),
            "context component set changed or gained an unexpected component");
        expect(parallel_mater::examples::find_simulation_recipe(expected.slug) == &actual,
            "lookup must return the catalog entry, not a copy");
    }

    const auto* softbody_rope = parallel_mater::examples::find_simulation_recipe("softbody-rope");
    expect(softbody_rope != nullptr &&
            softbody_rope->recipe == SimulationRecipe::soft_body_rope,
        "catalog must expose Softbody-Rope by slug");
    const auto* bridge = parallel_mater::examples::find_simulation_recipe("cloth-rope");
    expect(bridge != nullptr && bridge->recipe == SimulationRecipe::cloth_rope,
        "catalog must expose Cloth-Rope by slug");
    expect(parallel_mater::examples::find_simulation_recipe("unknown") == nullptr,
        "unknown recipe slug must be rejected");
    const auto* smoke = parallel_mater::examples::find_simulation_recipe("smoke");
    expect(smoke != nullptr && smoke->recipe == SimulationRecipe::smoke &&
            parallel_mater::examples::has_component(smoke->components, Component::smoke),
        "smoke recipe must expose the smoke component");
    expect(!parallel_mater::examples::has_component(Component::none, Component::cloth),
        "empty component set must not report cloth");
    expect(parallel_mater::examples::has_component(
               Component::cloth | Component::soft_body, Component::cloth),
        "component composition must retain the left component");
    expect(parallel_mater::examples::has_component(
               Component::cloth | Component::soft_body, Component::soft_body),
        "component composition must retain the right component");
}

void test_fixed_step_contract()
{
    constexpr parallel_mater::examples::FixedStepOptions defaults{};
    static_assert(defaults.timestep == 1.0F / 60.0F);
    static_assert(parallel_mater::examples::valid(defaults));
    static_assert(parallel_mater::examples::valid({1.0F / 120.0F}));
    static_assert(!parallel_mater::examples::valid({0.0F}));
    static_assert(!parallel_mater::examples::valid({-1.0F / 60.0F}));

    expect(parallel_mater::examples::valid(defaults), "default fixed step must be valid");
    expect(!parallel_mater::examples::valid(
               {std::numeric_limits<float>::quiet_NaN()}),
        "non-finite fixed timestep must be rejected by a solver boundary");
    expect(!parallel_mater::examples::valid(
               {std::numeric_limits<float>::infinity()}),
        "infinite fixed timestep must be rejected by a solver boundary");

    parallel_mater::examples::GallerySimulationOptions options;
    expect(!options.gravity_override.has_value() &&
            !options.solver_iterations_override.has_value(),
        "gallery defaults must select a recipe without implicit overrides");
    options.gravity_override = float3{1.0F, 2.0F, 3.0F};
    options.solver_iterations_override = 3U;
    expect(options.gravity_override->y == 2.0F &&
            *options.solver_iterations_override == 3U,
        "gallery physics overrides must retain explicit values");
}

void test_recipe_config_contract()
{
    auto config = parallel_mater::examples::RecipeConfig::for_recipe(
        SimulationRecipe::water_rope);
    config.timestep(1.0F / 120.0F).iterations(8U).particles(12'000U)
        .cloth_resolution(4U);
    expect(parallel_mater::examples::validate_recipe_config(config) ==
            parallel_mater::examples::RecipeConfigError::none,
        "fluent portable configuration must validate");
    config.cloth_resolution(0U);
    expect(parallel_mater::examples::validate_recipe_config(config) ==
            parallel_mater::examples::RecipeConfigError::invalid_cloth_detail,
        "portable configuration must reject zero cloth detail");
}

void test_owning_simulation_contract()
{
    static_assert(std::is_default_constructible_v<parallel_mater::examples::GallerySimulation>);
    static_assert(std::is_move_constructible_v<parallel_mater::examples::GallerySimulation>);
    static_assert(std::is_move_assignable_v<parallel_mater::examples::GallerySimulation>);
    static_assert(!std::is_copy_constructible_v<parallel_mater::examples::GallerySimulation>);
    static_assert(!std::is_copy_assignable_v<parallel_mater::examples::GallerySimulation>);

    expect(!parallel_mater::examples::requires_soft_body_asset(SimulationRecipe::water),
        "procedural particle bowl must not require an asset");
    expect(!parallel_mater::examples::requires_soft_body_asset(SimulationRecipe::cloth),
        "procedural cloth must not require an asset");
    expect(!parallel_mater::examples::requires_soft_body_asset(SimulationRecipe::water_cloth),
        "rigid water course must not request a soft-body asset");
    expect(parallel_mater::examples::requires_soft_body_asset(SimulationRecipe::soft_body),
        "soft-body context must request its authored cylinder asset");
    expect(!parallel_mater::examples::requires_soft_body_asset(SimulationRecipe::water_soft_body),
        "procedural water-wheel cross must not request the retired cylinder asset");
    expect(!parallel_mater::examples::requires_soft_body_asset(SimulationRecipe::rope),
        "procedural rope must not request an external asset");

    parallel_mater::examples::GallerySimulation empty;
    expect(!empty.initialized(), "default simulation must not allocate CUDA state");
    expect(!empty.step().ok(), "uninitialized simulation step must return an error");
    expect(empty.render_view().particle_system_count == 0U,
        "uninitialized simulation must expose an empty frame");
}

void test_borrowed_render_views()
{
    static_assert(std::is_trivially_copyable_v<parallel_mater::examples::ParticleRenderView>);
    static_assert(std::is_trivially_copyable_v<parallel_mater::examples::SurfaceRenderView>);
    static_assert(std::is_trivially_copyable_v<parallel_mater::examples::RigidBodyRenderView>);
    static_assert(std::is_trivially_copyable_v<parallel_mater::examples::LatticeBond>);
    static_assert(std::is_trivially_copyable_v<parallel_mater::examples::LatticeRenderView>);
    static_assert(std::is_trivially_copyable_v<parallel_mater::examples::FrameRenderView>);

    float3 positions[3]{};
    float3 velocities[3]{};
    uint3 triangles[1]{};
    float3 normals[3]{};
    float2 texcoords[3]{};
    std::uint8_t triangle_active[1]{1U};
    std::uint32_t flags[3]{};
    parallel_mater::examples::LatticeBond bonds[1]{{{0U, 1U}, 0.05F}};
    std::uint8_t bond_active[1]{1U};

    const parallel_mater::examples::ParticleRenderView particles{
        positions, velocities, 3U, 0.025F};
    const parallel_mater::examples::SurfaceRenderView surface{
        {positions, 3U, triangles, 1U}, normals, texcoords, triangle_active};
    const parallel_mater::examples::RigidBodyRenderView rigid{
        {positions, 3U, triangles, 1U},
        {1.0F, 2.0F, 3.0F},
        {0.0F, 0.0F, 0.0F, 1.0F},
        {2.0F, 3.0F, 4.0F}};

    const std::array particle_systems{particles};
    const std::array surfaces{surface};
    const std::array rigid_bodies{rigid};
    const std::array lattices{parallel_mater::examples::LatticeRenderView{
        positions, flags, bonds, bond_active, 3U, 3U, 1U, 1U, 0.025F}};
    const parallel_mater::examples::FrameRenderView frame{
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

    constexpr parallel_mater::examples::FrameRenderView empty{};
    static_assert(empty.particle_systems == nullptr && empty.particle_system_count == 0U);
    static_assert(empty.surfaces == nullptr && empty.surface_count == 0U);
    static_assert(empty.rigid_bodies == nullptr && empty.rigid_body_count == 0U);
    static_assert(empty.lattices == nullptr && empty.lattice_count == 0U);
}

} // namespace

int main()
{
    test_recipe_catalog();
    test_fixed_step_contract();
    test_recipe_config_contract();
    test_owning_simulation_contract();
    test_borrowed_render_views();

    if (failures != 0) {
        std::fprintf(stderr, "%d simulation API contract test(s) failed\n", failures);
        return 1;
    }
    std::puts("simulation API contract tests passed");
    return 0;
}
