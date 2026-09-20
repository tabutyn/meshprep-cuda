// SPDX-License-Identifier: MIT
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>

namespace meshprep::sim {

// Components are deliberately composable. Levels are recipes built from these
// flags; they are not separate solver implementations.
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

enum class ExampleContext : std::uint8_t {
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

struct ExampleContextInfo {
    ExampleContext id{};
    std::string_view slug{};
    std::string_view title{};
    Component components{};
};

inline constexpr std::array<ExampleContextInfo, 15> example_contexts{{
    {ExampleContext::water, "water", "Water",
        Component::fluid_particles | Component::rigid_bodies},
    {ExampleContext::cloth, "cloth", "Cloth",
        Component::cloth | Component::rigid_bodies},
    {ExampleContext::soft_body, "softbody", "Softbody",
        Component::soft_body | Component::rigid_bodies},
    {ExampleContext::rope, "rope", "Rope",
        Component::rope | Component::rigid_bodies},
    {ExampleContext::smoke, "smoke", "Smoke",
        Component::smoke | Component::rigid_bodies},
    {ExampleContext::water_cloth, "water-cloth", "Water-Cloth",
        Component::fluid_particles | Component::water_skin |
            Component::cloth | Component::rigid_bodies},
    {ExampleContext::water_soft_body, "water-softbody", "Water-Softbody",
        Component::soft_body | Component::fluid_particles | Component::rigid_bodies},
    {ExampleContext::water_rope, "water-rope", "Water-Rope",
        Component::fluid_particles | Component::rope | Component::rigid_bodies},
    {ExampleContext::fluid_smoke, "fluid-smoke", "Fluid-Smoke",
        Component::fluid_particles | Component::smoke},
    {ExampleContext::cloth_soft_body, "cloth-softbody", "Cloth-Softbody",
        Component::soft_body | Component::cloth | Component::rigid_bodies},
    {ExampleContext::cloth_rope, "cloth-rope", "Cloth-Rope",
        Component::cloth | Component::rope | Component::rigid_bodies},
    {ExampleContext::cloth_smoke, "cloth-smoke", "Cloth-Smoke",
        Component::cloth | Component::smoke | Component::rigid_bodies},
    {ExampleContext::soft_body_rope, "softbody-rope", "Softbody-Rope",
        Component::soft_body | Component::rope | Component::rigid_bodies},
    {ExampleContext::soft_body_smoke, "softbody-smoke", "Softbody-Smoke",
        Component::soft_body | Component::smoke | Component::rigid_bodies},
    {ExampleContext::rope_smoke, "rope-smoke", "Rope-Smoke",
        Component::rope | Component::smoke | Component::rigid_bodies},
}};

[[nodiscard]] constexpr const ExampleContextInfo* find_example_context(
    std::string_view slug) noexcept
{
    for (const auto& context : example_contexts) {
        if (context.slug == slug) return &context;
    }
    return nullptr;
}

enum class GoalKind : std::uint8_t {
    reach_course_goal,
    paint_surface,
    damage_cloth,
    reach_exit,
    reach_hole,
    ride_lift,
    pass_cloth,
    wrap_post,
    catch_treasure,
    observe,
};

struct LevelDefinition {
    ExampleContext context{};
    GoalKind goal{};
    std::string_view objective{};
    float target{};
};

inline constexpr std::array<LevelDefinition, 15> levels{{
    {ExampleContext::water, GoalKind::paint_surface,
        "Cover every bowl tile with blue water.", 1.0F},
    {ExampleContext::cloth, GoalKind::damage_cloth,
        "Damage the cloth marked GOAL.", 1.0F},
    {ExampleContext::soft_body, GoalKind::paint_surface,
        "Cover the rolling sphere in blue paint from the cylinders.", 1.0F},
    {ExampleContext::rope, GoalKind::wrap_post,
        "Wrap the tether three complete turns around the post.", 3.0F},
    {ExampleContext::smoke, GoalKind::observe,
        "Roll through the stream and reveal the turbulent wake.", 0.0F},
    {ExampleContext::water_cloth, GoalKind::reach_course_goal,
        "Roll the water sphere through the obstacle course.", 1.0F},
    {ExampleContext::water_soft_body, GoalKind::ride_lift,
        "Cross the water wheel from the right stage to the left exit.", 1.0F},
    {ExampleContext::water_rope, GoalKind::catch_treasure,
        "Hook the submerged treasure and reel it to the top.", 1.0F},
    {ExampleContext::fluid_smoke, GoalKind::observe,
        "Heat the water into rising steam.", 0.0F},
    {ExampleContext::cloth_soft_body, GoalKind::damage_cloth,
        "Paint and damage the cloth marked GOAL.", 1.0F},
    {ExampleContext::cloth_rope, GoalKind::reach_exit,
        "Roll across the cloth-and-rope floor.", 1.0F},
    {ExampleContext::cloth_smoke, GoalKind::observe,
        "Turn the pitched cloth windmill with smoke.", 0.0F},
    {ExampleContext::soft_body_rope, GoalKind::reach_exit,
        "Roll the soft-body sphere across the suspended rope bridge.", 1.0F},
    {ExampleContext::soft_body_smoke, GoalKind::observe,
        "Bend the soft grass with smoke and the rolling sphere.", 0.0F},
    {ExampleContext::rope_smoke, GoalKind::observe,
        "Cross the rope bridge while the smoke stream loads it.", 0.0F},
}};

[[nodiscard]] constexpr const LevelDefinition& level(ExampleContext context) noexcept
{
    for (const auto& definition : levels) {
        if (definition.context == context) return definition;
    }
    return levels.front();
}

// Setup shared by the CUDA simulation and native applications. Empty optionals
// select the authored level preset.
struct SimulationConfig {
    ExampleContext context{ExampleContext::water};
    float fixed_timestep{1.0F / 60.0F};
    std::optional<std::uint32_t> solver_iterations{};
    std::optional<std::uint32_t> particle_count{};
    std::optional<std::uint32_t> physical_skin_frequency{};
    std::optional<std::uint32_t> rope_node_count{};
    std::optional<std::uint32_t> cloth_detail{};

    [[nodiscard]] static constexpr SimulationConfig for_level(
        ExampleContext selected) noexcept
    {
        SimulationConfig result;
        result.context = selected;
        return result;
    }

    constexpr SimulationConfig& timestep(float value) noexcept
    {
        fixed_timestep = value;
        return *this;
    }
    constexpr SimulationConfig& iterations(std::uint32_t value) noexcept
    {
        solver_iterations = value;
        return *this;
    }
    constexpr SimulationConfig& particles(std::uint32_t value) noexcept
    {
        particle_count = value;
        return *this;
    }
    constexpr SimulationConfig& skin_frequency(std::uint32_t value) noexcept
    {
        physical_skin_frequency = value;
        return *this;
    }
    constexpr SimulationConfig& rope_nodes(std::uint32_t value) noexcept
    {
        rope_node_count = value;
        return *this;
    }
    constexpr SimulationConfig& cloth_resolution(std::uint32_t value) noexcept
    {
        cloth_detail = value;
        return *this;
    }
};

enum class ConfigError : std::uint8_t {
    none,
    invalid_context,
    invalid_timestep,
    invalid_iterations,
    invalid_particle_count,
    invalid_skin_frequency,
    invalid_rope_nodes,
    invalid_cloth_detail,
};

[[nodiscard]] ConfigError validate(SimulationConfig config) noexcept;
[[nodiscard]] std::string_view describe(ConfigError error) noexcept;

// Applications report a few semantic metrics; Campaign owns win evaluation
// and numbered-level progression. Backends remain free to represent physics
// and rendering differently.
struct LevelMetrics {
    bool course_goal_reached{};
    float painted_fraction{};
    std::uint32_t broken_connections{};
    bool exit_reached{};
    bool hole_reached{};
    float lift_progress{};
    bool cloth_passed{};
    float rope_turns{};
    bool treasure_caught{};
    // Zero until the hook latches. One means the treasure has reached the
    // authored recovery height at the top of the tank.
    float treasure_lift_progress{};
};

struct LevelProgress {
    ExampleContext context{ExampleContext::water};
    float normalized{};
    bool won{};
    bool advanced{};
};

[[nodiscard]] LevelProgress evaluate(
    ExampleContext context, const LevelMetrics& metrics) noexcept;

class Campaign {
public:
    explicit constexpr Campaign(
        ExampleContext context = ExampleContext::water) noexcept
        : context_(context) {}

    void select(ExampleContext context) noexcept;
    [[nodiscard]] LevelProgress update(const LevelMetrics& metrics) noexcept;
    [[nodiscard]] constexpr ExampleContext current() const noexcept { return context_; }
    [[nodiscard]] constexpr std::uint32_t won_frames() const noexcept
    {
        return won_frames_;
    }

private:
    ExampleContext context_{ExampleContext::water};
    std::uint32_t won_frames_{};
};

} // namespace meshprep::sim
