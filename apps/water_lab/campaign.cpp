// SPDX-License-Identifier: MIT
#include "campaign.hpp"

#include <algorithm>
#include <cmath>

namespace waterlab {
namespace {

[[nodiscard]] bool valid_recipe(parallel_mater::examples::SimulationRecipe recipe) noexcept
{
    return std::any_of(parallel_mater::examples::simulation_recipes.begin(),
        parallel_mater::examples::simulation_recipes.end(),[recipe](const auto& candidate) {
            return candidate.recipe==recipe;
        });
}

[[nodiscard]] float saturated(float value) noexcept
{
    return std::clamp(std::isfinite(value) ? value : 0.0F,0.0F,1.0F);
}

} // namespace

const GalleryObjective& gallery_objective(
    parallel_mater::examples::SimulationRecipe recipe) noexcept
{
    for (const auto& objective:gallery_objectives) {
        if (objective.recipe==recipe) return objective;
    }
    return gallery_objectives.front();
}

ObjectiveProgress evaluate_objective(
    parallel_mater::examples::SimulationRecipe recipe,const ObjectiveMetrics& metrics) noexcept
{
    ObjectiveProgress result{recipe};
    const GalleryObjective& objective=gallery_objective(recipe);
    switch (objective.kind) {
    case ObjectiveKind::reach_course_goal:
        result.normalized=metrics.course_goal_reached ? 1.0F : 0.0F;
        break;
    case ObjectiveKind::paint_surface:
        result.normalized=saturated(metrics.painted_fraction);
        break;
    case ObjectiveKind::damage_cloth:
        result.normalized=metrics.broken_connections==0U ? 0.0F : 1.0F;
        break;
    case ObjectiveKind::reach_exit:
        result.normalized=metrics.exit_reached ? 1.0F : 0.0F;
        break;
    case ObjectiveKind::ride_lift:
        result.normalized=saturated(metrics.lift_progress);
        break;
    case ObjectiveKind::wrap_post:
        result.normalized=saturated(metrics.rope_turns/objective.target);
        break;
    case ObjectiveKind::catch_treasure:
        result.normalized=metrics.treasure_caught
            ? saturated(metrics.treasure_lift_progress) : 0.0F;
        break;
    case ObjectiveKind::observe:
        result.normalized=0.0F;
        break;
    }
    result.completed=result.normalized>=1.0F;
    return result;
}

void GalleryProgression::select(parallel_mater::examples::SimulationRecipe recipe) noexcept
{
    if (!valid_recipe(recipe)) return;
    recipe_=recipe;
    completed_frames_=0U;
}

ObjectiveProgress GalleryProgression::update(const ObjectiveMetrics& metrics) noexcept
{
    ObjectiveProgress result=evaluate_objective(recipe_,metrics);
    if (!result.completed) {
        completed_frames_=0U;
        return result;
    }
    ++completed_frames_;
    if (completed_frames_>=90U &&
        recipe_!=parallel_mater::examples::simulation_recipes.back().recipe) {
        const auto current=std::find_if(parallel_mater::examples::simulation_recipes.begin(),
            parallel_mater::examples::simulation_recipes.end(),[this](const auto& candidate) {
                return candidate.recipe==recipe_;
            });
        if (current==parallel_mater::examples::simulation_recipes.end() ||
            current+1==parallel_mater::examples::simulation_recipes.end()) return result;
        recipe_=(current+1)->recipe;
        completed_frames_=0U;
        result.advanced=true;
    }
    return result;
}

} // namespace waterlab
