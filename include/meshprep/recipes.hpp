// SPDX-License-Identifier: MIT
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>

namespace meshprep::sim {

// Components are deliberately composable. Recipes describe example solver
// compositions; they do not define gameplay or progression.
enum class Component : std::uint32_t {
    none = 0,
    fluid_particles = 1U << 0U,
    water_skin = 1U << 1U,
    cloth = 1U << 2U,
    soft_body = 1U << 3U,
    rigid_bodies = 1U << 4U,
    hand_particles = 1U << 5U,
    rope = 1U << 6U,
    smoke = 1U << 7U,
};

[[nodiscard]] constexpr Component operator|(Component left, Component right) noexcept
{
    return static_cast<Component>(
        static_cast<std::uint32_t>(left) | static_cast<std::uint32_t>(right));
}

[[nodiscard]] constexpr bool has_component(Component set, Component component) noexcept
{
    return (static_cast<std::uint32_t>(set) &
        static_cast<std::uint32_t>(component)) != 0U;
}

enum class SimulationRecipe : std::uint8_t {
    water = 1,
    cloth = 2,
    soft_body = 3,
    rope = 4,
    water_cloth = 5,
    water_soft_body = 6,
    water_rope = 7,
    cloth_soft_body = 8,
    cloth_rope = 9,
    soft_body_rope = 10,
    smoke = 11,
    fluid_smoke = 12,
    cloth_smoke = 13,
    soft_body_smoke = 14,
    rope_smoke = 15,
};

struct SimulationRecipeInfo {
    SimulationRecipe recipe{};
    std::string_view slug{};
    std::string_view title{};
    Component components{};
};

inline constexpr std::array<SimulationRecipeInfo, 15> simulation_recipes{{
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

[[nodiscard]] constexpr const SimulationRecipeInfo* find_simulation_recipe(
    std::string_view slug) noexcept
{
    for (const auto& recipe : simulation_recipes) {
        if (recipe.slug == slug) return &recipe;
    }
    return nullptr;
}

// Setup shared by gallery clients. Empty optionals select the authored recipe
// preset; objectives and progression remain application policy.
struct RecipeConfig {
    SimulationRecipe recipe{SimulationRecipe::water};
    float fixed_timestep{1.0F / 60.0F};
    std::optional<std::uint32_t> solver_iterations{};
    std::optional<std::uint32_t> particle_count{};
    std::optional<std::uint32_t> physical_skin_frequency{};
    std::optional<std::uint32_t> rope_node_count{};
    std::optional<std::uint32_t> cloth_detail{};

    [[nodiscard]] static constexpr RecipeConfig for_recipe(
        SimulationRecipe selected) noexcept
    {
        RecipeConfig result;
        result.recipe = selected;
        return result;
    }

    constexpr RecipeConfig& timestep(float value) noexcept
    {
        fixed_timestep = value;
        return *this;
    }
    constexpr RecipeConfig& iterations(std::uint32_t value) noexcept
    {
        solver_iterations = value;
        return *this;
    }
    constexpr RecipeConfig& particles(std::uint32_t value) noexcept
    {
        particle_count = value;
        return *this;
    }
    constexpr RecipeConfig& skin_frequency(std::uint32_t value) noexcept
    {
        physical_skin_frequency = value;
        return *this;
    }
    constexpr RecipeConfig& rope_nodes(std::uint32_t value) noexcept
    {
        rope_node_count = value;
        return *this;
    }
    constexpr RecipeConfig& cloth_resolution(std::uint32_t value) noexcept
    {
        cloth_detail = value;
        return *this;
    }
};

enum class RecipeConfigError : std::uint8_t {
    none,
    invalid_recipe,
    invalid_timestep,
    invalid_iterations,
    invalid_particle_count,
    invalid_skin_frequency,
    invalid_rope_nodes,
    invalid_cloth_detail,
};

[[nodiscard]] RecipeConfigError validate_recipe_config(RecipeConfig config) noexcept;
[[nodiscard]] std::string_view recipe_config_error_message(
    RecipeConfigError error) noexcept;

} // namespace meshprep::sim
