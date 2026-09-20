// SPDX-License-Identifier: MIT
#include <meshprep/recipes.hpp>

#include <cmath>

namespace meshprep::sim {

namespace {

[[nodiscard]] bool valid_recipe(SimulationRecipe recipe) noexcept
{
    for (const auto& candidate : simulation_recipes) {
        if (candidate.recipe == recipe) return true;
    }
    return false;
}

} // namespace

RecipeConfigError validate_recipe_config(RecipeConfig config) noexcept
{
    if (!valid_recipe(config.recipe)) return RecipeConfigError::invalid_recipe;
    if (!std::isfinite(config.fixed_timestep) || config.fixed_timestep <= 0.0F)
        return RecipeConfigError::invalid_timestep;
    if (config.solver_iterations &&
        (*config.solver_iterations < 1U || *config.solver_iterations > 16U))
        return RecipeConfigError::invalid_iterations;
    if (config.particle_count &&
        (*config.particle_count < 256U || *config.particle_count > 100'000U))
        return RecipeConfigError::invalid_particle_count;
    if (config.physical_skin_frequency &&
        (*config.physical_skin_frequency < 2U || *config.physical_skin_frequency > 90U))
        return RecipeConfigError::invalid_skin_frequency;
    if (config.rope_node_count &&
        (*config.rope_node_count < 8U || *config.rope_node_count > 512U))
        return RecipeConfigError::invalid_rope_nodes;
    if (config.cloth_detail &&
        (*config.cloth_detail < 1U || *config.cloth_detail > 8U))
        return RecipeConfigError::invalid_cloth_detail;
    return RecipeConfigError::none;
}

std::string_view recipe_config_error_message(RecipeConfigError error) noexcept
{
    switch (error) {
    case RecipeConfigError::none: return "valid";
    case RecipeConfigError::invalid_recipe: return "unknown simulation recipe";
    case RecipeConfigError::invalid_timestep: return "timestep must be finite and positive";
    case RecipeConfigError::invalid_iterations: return "solver iterations must be between 1 and 16";
    case RecipeConfigError::invalid_particle_count: return "particle count must be between 256 and 100000";
    case RecipeConfigError::invalid_skin_frequency: return "skin frequency must be between 2 and 90";
    case RecipeConfigError::invalid_rope_nodes: return "rope node count must be between 8 and 512";
    case RecipeConfigError::invalid_cloth_detail: return "cloth detail must be between 1 and 8";
    }
    return "unknown configuration error";
}

} // namespace meshprep::sim
