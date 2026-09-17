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

// Context 5 uses one doubled-spacing cloth. Four support rows crossed with four
// support columns divide that sheet into nine independently flexible cells.
inline constexpr std::uint32_t catch_cloth_columns = 28U;
inline constexpr std::uint32_t catch_cloth_rows = 22U;
inline constexpr float catch_cloth_spacing = 0.13F;
inline constexpr float3 catch_cloth_center = cloth_basin_center;
inline constexpr std::uint32_t default_rope_nodes = 32U;
inline constexpr float rope_length = 1.75F;

[[nodiscard]] const meshprep::sim::ExampleContextInfo& context_info(
    meshprep::sim::ExampleContext context) noexcept;

[[nodiscard]] bool context_has(
    meshprep::sim::ExampleContext context,
    meshprep::sim::Component component) noexcept;

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
    std::uint32_t rope_node_count = default_rope_nodes);

[[nodiscard]] FluidDisplay default_context_display(
    meshprep::sim::ExampleContext context) noexcept;

[[nodiscard]] RigidSphereState initial_rigid_sphere(
    meshprep::sim::ExampleContext context,
    std::uint32_t rope_node_count = default_rope_nodes) noexcept;

void initialize_context_motion(
    meshprep::sim::ExampleContext context, SoftBodyCourse& body);

} // namespace waterlab::gallery
