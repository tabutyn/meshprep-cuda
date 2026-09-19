// SPDX-License-Identifier: MIT
#pragma once

#include "fluid_visuals.hpp"
#include "hybrid_lab.hpp"
#include "soft_body.hpp"

#include <meshprep/simulation.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>

namespace waterlab::gallery {

// Context 5 keeps the same physical 3x3 catch area as the original 28x22
// sheet, but defaults to twice the tessellation in each direction. This gives
// the small rigid sphere a continuous contact surface instead of gaps wider
// than its useful contact footprint.
inline constexpr std::uint32_t default_hanging_cloth_detail = 1U;
inline constexpr std::uint32_t default_catch_cloth_detail = 3U;
inline constexpr std::uint32_t catch_cloth_columns =
    27U * default_catch_cloth_detail + 1U;
inline constexpr std::uint32_t catch_cloth_rows =
    21U * default_catch_cloth_detail + 1U;
inline constexpr float catch_cloth_spacing =
    0.13F / static_cast<float>(default_catch_cloth_detail);
inline constexpr float3 catch_cloth_center = cloth_basin_center;
inline constexpr std::uint32_t default_rope_nodes = 64U;
inline constexpr float rope_length = 4.0F;

[[nodiscard]] constexpr float gravity_tilt_degrees(
    meshprep::sim::ExampleContext context) noexcept
{
    return context == meshprep::sim::ExampleContext::particles_cloth ? 76.0F :
        context == meshprep::sim::ExampleContext::soft_body_cloth ? 38.0F : 20.0F;
}

[[nodiscard]] const meshprep::sim::ExampleContextInfo& context_info(
    meshprep::sim::ExampleContext context) noexcept;

[[nodiscard]] bool context_has(
    meshprep::sim::ExampleContext context,
    meshprep::sim::Component component) noexcept;

[[nodiscard]] constexpr std::uint32_t default_cloth_detail(
    meshprep::sim::ExampleContext context) noexcept
{
    return context == meshprep::sim::ExampleContext::particles_cloth
        ? default_catch_cloth_detail : default_hanging_cloth_detail;
}

// The only recipe-to-solver translation used by both the installed headless
// API and the native gallery. Optional values are explicit caller overrides;
// an empty value keeps the selected recipe's preset.
struct ContextPhysicsOverrides {
    float fixed_dt{1.0F / 60.0F};
    std::optional<std::uint32_t> physics_iterations{};
    std::optional<float3> gravity{};
    std::optional<std::uint32_t> particle_count{};
    std::optional<std::uint32_t> physical_skin_frequency{};
};

[[nodiscard]] HybridOptions make_context_physics(
    meshprep::sim::ExampleContext context,
    const ContextPhysicsOverrides& overrides = {}) noexcept;

// Builds the deformable fixture used by an example context. Context 1 uses
// analytic rigid posts and context 2 is particle-only, so both return null.
// The asset path remains a caller choice so this sample layer has no
// build-system macro or install-layout dependency.
[[nodiscard]] std::unique_ptr<SoftBodyCourse> make_context_deformable(
    meshprep::sim::ExampleContext context,
    const HybridOptions& physics,
    std::string_view soft_body_asset_path,
    std::uint32_t rope_node_count = default_rope_nodes,
    std::uint32_t cloth_detail = 0U,
    std::uint32_t bridge_columns = rope_bridge_columns,
    std::uint32_t bridge_rows = rope_bridge_rows);

[[nodiscard]] FluidDisplay default_context_display(
    meshprep::sim::ExampleContext context) noexcept;

[[nodiscard]] RigidSphereState initial_rigid_sphere(
    meshprep::sim::ExampleContext context,
    std::uint32_t rope_node_count = default_rope_nodes) noexcept;

void initialize_context_motion(
    meshprep::sim::ExampleContext context, SoftBodyCourse& body);

} // namespace waterlab::gallery
