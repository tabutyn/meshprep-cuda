// SPDX-License-Identifier: MIT
#include <meshprep/game.hpp>

#include <algorithm>
#include <cmath>

namespace meshprep::sim {

namespace {

[[nodiscard]] bool valid_context(ExampleContext context) noexcept
{
    return std::any_of(levels.begin(), levels.end(), [context](const auto& item) {
        return item.context == context;
    });
}

[[nodiscard]] float saturated(float value) noexcept
{
    return std::clamp(std::isfinite(value) ? value : 0.0F, 0.0F, 1.0F);
}

} // namespace

ConfigError validate(SimulationConfig config) noexcept
{
    if (!valid_context(config.context)) return ConfigError::invalid_context;
    if (!std::isfinite(config.fixed_timestep) || config.fixed_timestep <= 0.0F)
        return ConfigError::invalid_timestep;
    if (config.solver_iterations &&
        (*config.solver_iterations < 1U || *config.solver_iterations > 16U))
        return ConfigError::invalid_iterations;
    if (config.particle_count &&
        (*config.particle_count < 256U || *config.particle_count > 100'000U))
        return ConfigError::invalid_particle_count;
    if (config.physical_skin_frequency &&
        (*config.physical_skin_frequency < 2U || *config.physical_skin_frequency > 90U))
        return ConfigError::invalid_skin_frequency;
    if (config.rope_node_count &&
        (*config.rope_node_count < 8U || *config.rope_node_count > 512U))
        return ConfigError::invalid_rope_nodes;
    if (config.cloth_detail &&
        (*config.cloth_detail < 1U || *config.cloth_detail > 8U))
        return ConfigError::invalid_cloth_detail;
    return ConfigError::none;
}

std::string_view describe(ConfigError error) noexcept
{
    switch (error) {
    case ConfigError::none: return "valid";
    case ConfigError::invalid_context: return "context must be one of the ten gallery recipes";
    case ConfigError::invalid_timestep: return "timestep must be finite and positive";
    case ConfigError::invalid_iterations: return "solver iterations must be between 1 and 16";
    case ConfigError::invalid_particle_count: return "particle count must be between 256 and 100000";
    case ConfigError::invalid_skin_frequency: return "skin frequency must be between 2 and 90";
    case ConfigError::invalid_rope_nodes: return "rope node count must be between 8 and 512";
    case ConfigError::invalid_cloth_detail: return "cloth detail must be between 1 and 8";
    }
    return "unknown configuration error";
}

LevelProgress evaluate(ExampleContext context, const LevelMetrics& metrics) noexcept
{
    LevelProgress result{context};
    switch (level(context).goal) {
    case GoalKind::reach_course_goal:
        result.normalized = metrics.course_goal_reached ? 1.0F : 0.0F;
        break;
    case GoalKind::paint_surface:
        result.normalized = saturated(metrics.painted_fraction);
        break;
    case GoalKind::damage_cloth:
        result.normalized = metrics.broken_connections == 0U ? 0.0F : 1.0F;
        break;
    case GoalKind::reach_exit:
        result.normalized = metrics.exit_reached ? 1.0F : 0.0F;
        break;
    case GoalKind::reach_hole:
        result.normalized = metrics.hole_reached ? 1.0F : 0.0F;
        break;
    case GoalKind::ride_lift:
        result.normalized = saturated(metrics.lift_progress);
        break;
    case GoalKind::pass_cloth:
        result.normalized = metrics.cloth_passed ? 1.0F : 0.0F;
        break;
    case GoalKind::wrap_post:
        result.normalized = saturated(metrics.rope_turns / level(context).target);
        break;
    case GoalKind::catch_treasure:
        result.normalized = metrics.treasure_caught
            ? saturated(metrics.treasure_lift_progress) : 0.0F;
        break;
    }
    result.won = result.normalized >= 1.0F;
    return result;
}

void Campaign::select(ExampleContext context) noexcept
{
    if (!valid_context(context)) return;
    context_ = context;
    won_frames_ = 0U;
}

LevelProgress Campaign::update(const LevelMetrics& metrics) noexcept
{
    LevelProgress result = evaluate(context_, metrics);
    if (!result.won) {
        won_frames_ = 0U;
        return result;
    }
    ++won_frames_;
    // Hold the win card for 1.5 seconds at the fixed 60 Hz cadence.
    if (won_frames_ >= 90U && context_ != example_contexts.back().id) {
        const auto current = std::find_if(example_contexts.begin(), example_contexts.end(),
            [this](const auto& item) { return item.id == context_; });
        if (current == example_contexts.end() || current + 1 == example_contexts.end())
            return result;
        context_ = (current + 1)->id;
        won_frames_ = 0U;
        result.advanced = true;
    }
    return result;
}

} // namespace meshprep::sim
