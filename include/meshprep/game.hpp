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
    water_course = 1,
    particle_bowl = 2,
    cloth_rigid = 3,
    soft_body_rigid = 4,
    particles_cloth = 5,
    soft_body_fluid = 6,
    soft_body_cloth = 7,
    rope_rigid = 8,
    rope_bridge = 9,
};

struct ExampleContextInfo {
    ExampleContext id{};
    char key{};
    std::string_view slug{};
    std::string_view title{};
    Component components{};
};

inline constexpr std::array<ExampleContextInfo, 9> example_contexts{{
    {ExampleContext::water_course, '1', "water-course", "Water obstacle course",
        Component::fluid_particles | Component::water_skin | Component::rigid_bodies},
    {ExampleContext::particle_bowl, '2', "particle-bowl", "Paint the bowl blue",
        Component::fluid_particles | Component::rigid_bodies},
    {ExampleContext::cloth_rigid, '3', "cloth-rigid", "Break the goal cloth",
        Component::cloth | Component::rigid_bodies},
    {ExampleContext::soft_body_rigid, '4', "soft-body-rigid", "Cylinder curtain",
        Component::soft_body | Component::rigid_bodies},
    {ExampleContext::particles_cloth, '5', "particles-cloth", "Water snake",
        Component::fluid_particles | Component::cloth | Component::rigid_bodies},
    {ExampleContext::soft_body_fluid, '6', "soft-body-fluid", "Water wheel crossing",
        Component::soft_body | Component::fluid_particles | Component::rigid_bodies},
    {ExampleContext::soft_body_cloth, '7', "soft-body-cloth", "Cloth gate",
        Component::soft_body | Component::cloth | Component::rigid_bodies},
    {ExampleContext::rope_rigid, '8', "rope-rigid", "Wrap the post",
        Component::soft_body | Component::rigid_bodies},
    {ExampleContext::rope_bridge, '9', "rope-bridge", "Cross the rope bridge",
        Component::soft_body | Component::rigid_bodies},
}};

[[nodiscard]] constexpr const ExampleContextInfo* find_example_context(char key) noexcept
{
    for (const auto& context : example_contexts) {
        if (context.key == key) return &context;
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
};

struct LevelDefinition {
    ExampleContext context{};
    GoalKind goal{};
    std::string_view objective{};
    float target{};
};

inline constexpr std::array<LevelDefinition, 9> levels{{
    {ExampleContext::water_course, GoalKind::reach_course_goal,
        "Roll the water sphere through the obstacle course.", 1.0F},
    {ExampleContext::particle_bowl, GoalKind::paint_surface,
        "Cover every bowl tile with blue water.", 1.0F},
    {ExampleContext::cloth_rigid, GoalKind::damage_cloth,
        "Damage the cloth marked GOAL.", 1.0F},
    {ExampleContext::soft_body_rigid, GoalKind::paint_surface,
        "Cover the rolling sphere in blue paint from the cylinders.", 1.0F},
    {ExampleContext::particles_cloth, GoalKind::reach_hole,
        "Guide the sphere and water through the snake to the top-right hole.", 1.0F},
    {ExampleContext::soft_body_fluid, GoalKind::ride_lift,
        "Cross the water wheel from the right stage to the left exit.", 1.0F},
    {ExampleContext::soft_body_cloth, GoalKind::damage_cloth,
        "Paint and damage the cloth marked GOAL.", 1.0F},
    {ExampleContext::rope_rigid, GoalKind::wrap_post,
        "Wrap the tether three complete turns around the post.", 3.0F},
    {ExampleContext::rope_bridge, GoalKind::reach_exit,
        "Roll the soft-body sphere across the suspended rope bridge.", 1.0F},
}};

[[nodiscard]] constexpr const LevelDefinition& level(ExampleContext context) noexcept
{
    return levels[static_cast<std::size_t>(context) - 1U];
}

// Setup shared by the CUDA simulation and native applications. Empty optionals
// select the authored level preset.
struct SimulationConfig {
    ExampleContext context{ExampleContext::water_course};
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
};

struct LevelProgress {
    ExampleContext context{ExampleContext::water_course};
    float normalized{};
    bool won{};
    bool advanced{};
};

[[nodiscard]] LevelProgress evaluate(
    ExampleContext context, const LevelMetrics& metrics) noexcept;

class Campaign {
public:
    explicit constexpr Campaign(
        ExampleContext context = ExampleContext::water_course) noexcept
        : context_(context) {}

    void select(ExampleContext context) noexcept;
    [[nodiscard]] LevelProgress update(const LevelMetrics& metrics) noexcept;
    [[nodiscard]] constexpr ExampleContext current() const noexcept { return context_; }
    [[nodiscard]] constexpr std::uint32_t won_frames() const noexcept
    {
        return won_frames_;
    }

private:
    ExampleContext context_{ExampleContext::water_course};
    std::uint32_t won_frames_{};
};

} // namespace meshprep::sim
