// SPDX-License-Identifier: MIT
#pragma once

#include "recipes.hpp"

#include <array>
#include <cstdint>
#include <string_view>

namespace waterlab {

// Native-gallery objectives are application policy, not simulation API.
enum class ObjectiveKind : std::uint8_t {
    reach_course_goal,
    paint_surface,
    damage_cloth,
    reach_exit,
    ride_lift,
    wrap_post,
    catch_treasure,
    observe,
};

struct GalleryObjective {
    parallel_mater::examples::SimulationRecipe recipe{};
    ObjectiveKind kind{};
    std::string_view description{};
    float target{};
};

inline constexpr std::array<GalleryObjective, 15> gallery_objectives{{
    {parallel_mater::examples::SimulationRecipe::water,ObjectiveKind::paint_surface,
        "Cover every bowl tile with blue water.",1.0F},
    {parallel_mater::examples::SimulationRecipe::cloth,ObjectiveKind::damage_cloth,
        "Damage the cloth marked GOAL.",1.0F},
    {parallel_mater::examples::SimulationRecipe::soft_body,ObjectiveKind::paint_surface,
        "Cover the rolling sphere in blue paint from the cylinders.",1.0F},
    {parallel_mater::examples::SimulationRecipe::rope,ObjectiveKind::wrap_post,
        "Wrap the tether three complete turns around the post.",3.0F},
    {parallel_mater::examples::SimulationRecipe::smoke,ObjectiveKind::observe,
        "Roll through the stream and reveal the turbulent wake.",0.0F},
    {parallel_mater::examples::SimulationRecipe::water_cloth,ObjectiveKind::reach_course_goal,
        "Roll the water sphere through the obstacle course.",1.0F},
    {parallel_mater::examples::SimulationRecipe::water_soft_body,ObjectiveKind::ride_lift,
        "Cross the water wheel from the right stage to the left exit.",1.0F},
    {parallel_mater::examples::SimulationRecipe::water_rope,ObjectiveKind::catch_treasure,
        "Hook the submerged treasure and reel it to the top.",1.0F},
    {parallel_mater::examples::SimulationRecipe::fluid_smoke,ObjectiveKind::observe,
        "Heat the water into rising steam.",0.0F},
    {parallel_mater::examples::SimulationRecipe::cloth_soft_body,ObjectiveKind::damage_cloth,
        "Paint and damage the cloth marked GOAL.",1.0F},
    {parallel_mater::examples::SimulationRecipe::cloth_rope,ObjectiveKind::reach_exit,
        "Roll across the cloth-and-rope floor.",1.0F},
    {parallel_mater::examples::SimulationRecipe::cloth_smoke,ObjectiveKind::observe,
        "Turn the pitched cloth windmill with smoke.",0.0F},
    {parallel_mater::examples::SimulationRecipe::soft_body_rope,ObjectiveKind::reach_exit,
        "Roll the soft-body sphere across the suspended rope bridge.",1.0F},
    {parallel_mater::examples::SimulationRecipe::soft_body_smoke,ObjectiveKind::observe,
        "Bend the soft grass with smoke and the rolling sphere.",0.0F},
    {parallel_mater::examples::SimulationRecipe::rope_smoke,ObjectiveKind::observe,
        "Cross the rope bridge while the smoke stream loads it.",0.0F},
}};

[[nodiscard]] const GalleryObjective& gallery_objective(
    parallel_mater::examples::SimulationRecipe recipe) noexcept;

struct ObjectiveMetrics {
    bool course_goal_reached{};
    float painted_fraction{};
    std::uint32_t broken_connections{};
    bool exit_reached{};
    float lift_progress{};
    float rope_turns{};
    bool treasure_caught{};
    float treasure_lift_progress{};
};

struct ObjectiveProgress {
    parallel_mater::examples::SimulationRecipe recipe{parallel_mater::examples::SimulationRecipe::water};
    float normalized{};
    bool completed{};
    bool advanced{};
};

[[nodiscard]] ObjectiveProgress evaluate_objective(
    parallel_mater::examples::SimulationRecipe recipe,const ObjectiveMetrics& metrics) noexcept;

class GalleryProgression {
public:
    explicit constexpr GalleryProgression(
        parallel_mater::examples::SimulationRecipe recipe=parallel_mater::examples::SimulationRecipe::water) noexcept
        : recipe_(recipe) {}

    void select(parallel_mater::examples::SimulationRecipe recipe) noexcept;
    [[nodiscard]] ObjectiveProgress update(const ObjectiveMetrics& metrics) noexcept;
    [[nodiscard]] constexpr parallel_mater::examples::SimulationRecipe current() const noexcept
    {
        return recipe_;
    }
    [[nodiscard]] constexpr std::uint32_t completed_frames() const noexcept
    {
        return completed_frames_;
    }

private:
    parallel_mater::examples::SimulationRecipe recipe_{parallel_mater::examples::SimulationRecipe::water};
    std::uint32_t completed_frames_{};
};

} // namespace waterlab
