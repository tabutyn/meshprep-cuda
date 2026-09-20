// SPDX-License-Identifier: MIT
#pragma once

#include "recipes.hpp"
#include <parallel_mater/geometry.hpp>
#include <parallel_mater/physics.hpp>

#include <vector_types.h>

#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace parallel_mater::examples {

enum class ParticleMaterial : std::uint8_t {
    fluid,
    smoke,
    steam,
};

// Read-only CUDA views form the rendering boundary. Simulation implementations
// retain ownership and may replace their allocations between frames; callers
// must reacquire views after every simulation step.
struct ParticleRenderView {
    const float3 *positions{};
    const float3 *velocities{};
    std::uint32_t count{};
    float radius{};
    ParticleMaterial material{ParticleMaterial::fluid};
};

struct SurfaceRenderView {
    DeviceMeshView mesh{};
    const float3 *vertex_normals{};
    const float2 *texcoords{};
    const std::uint8_t *triangle_active{};
};

struct RigidBodyRenderView {
    DeviceMeshView mesh{};
    float3 translation{};
    // Unit quaternion stored as (x, y, z, w).
    float4 orientation{0.0F, 0.0F, 0.0F, 1.0F};
    float3 scale{1.0F, 1.0F, 1.0F};
};

inline constexpr std::uint32_t lattice_node_surface = 1U << 0U;
inline constexpr std::uint32_t lattice_node_pinned = 1U << 1U;

using LatticeBond = physics::Bond;

// Read-only CUDA arrays include every node, including the solid interior.
// Bonds are shared local topology: instance i uses positions at
// i * nodes_per_instance + bonds[b].vertices.{x,y}. Its live connection mask
// is bond_active[i * bonds_per_instance + b]; zero means permanently broken
// until reset. Flags use lattice_node_surface and lattice_node_pinned.
struct LatticeRenderView {
    const float3 *positions{};
    const std::uint32_t *flags{};
    const LatticeBond *bonds{};
    const std::uint8_t *bond_active{};
    std::uint32_t node_count{};
    std::uint32_t nodes_per_instance{};
    std::uint32_t bonds_per_instance{};
    std::uint32_t instance_count{};
    float node_radius{};
};

struct FrameRenderView {
    const ParticleRenderView *particle_systems{};
    std::uint32_t particle_system_count{};
    const SurfaceRenderView *surfaces{};
    std::uint32_t surface_count{};
    const RigidBodyRenderView *rigid_bodies{};
    std::uint32_t rigid_body_count{};
    const LatticeRenderView *lattices{};
    std::uint32_t lattice_count{};
};

struct FixedStepOptions {
    float timestep{1.0F / 60.0F};
};

// Header-only validation keeps the fixed-step contract identical for sample
// applications and future solver backends without introducing a windowing or
// renderer dependency into the installed package.
[[nodiscard]] constexpr bool valid(FixedStepOptions options) noexcept {
    return options.timestep > 0.0F && options.timestep == options.timestep &&
           options.timestep <= std::numeric_limits<float>::max();
}

// Configuration for the experimental, headless gallery backend. The asset
// path is read only by initialize(); GallerySimulation owns its own copy after
// initialization. Softbody requires the converted cylinder .msb asset; the
// water wheel, rigid course, and other contexts are generated procedurally.
struct GallerySimulationOptions {
    SimulationRecipe recipe{SimulationRecipe::water};
    FixedStepOptions fixed_step{};
    // Empty overrides select recipe defaults. An explicit particle count also
    // sizes the reserved capacity, up to the 100,000-particle stress ceiling.
    std::optional<std::uint32_t> solver_iterations_override{};
    std::optional<float3> gravity_override{};
    std::optional<std::uint32_t> particle_count_override{};
    std::optional<std::uint32_t> physical_skin_frequency_override{};
    std::optional<std::uint32_t> rope_node_count_override{};
    // Multiplies each authored cloth interval in Cloth while
    // preserving the fixture's physical dimensions. Range: 1..8.
    std::optional<std::uint32_t> cloth_detail_override{};
    // Cloth-Rope and Softbody-Rope bridge dimensions. Empty selects 4x10.
    std::optional<std::uint32_t> bridge_columns_override{};
    std::optional<std::uint32_t> bridge_rows_override{};
    // Softbody cylinder-field dimensions. The same authored volume is
    // repacked with thinner cylinders as either dimension grows.
    std::optional<std::uint32_t> cylinder_columns_override{};
    std::optional<std::uint32_t> cylinder_rows_override{};
    std::string_view soft_body_asset_path{};
};

// Concrete values selected after applying the context preset and explicit
// overrides. This is useful for logging and for configuring a renderer or
// another system with the exact timestep/gravity used by the simulation.
struct ResolvedPhysicsOptions {
    FixedStepOptions fixed_step{};
    std::uint32_t solver_iterations{};
    float3 gravity{};
    bool rigid_course_preset{};
};

struct GallerySimulationStatistics {
    std::uint64_t frame_index{};
    std::uint32_t particle_count{};
    std::uint32_t surface_count{};
    std::uint32_t rigid_body_count{};
    std::uint32_t finite_failure_count{};
    std::uint32_t broken_connection_count{};
    float last_gpu_time_ms{};
    std::size_t allocated_bytes{};
};

[[nodiscard]] constexpr bool requires_soft_body_asset(SimulationRecipe context) noexcept {
    return context == SimulationRecipe::soft_body || context == SimulationRecipe::water_soft_body ||
           context == SimulationRecipe::cloth_soft_body ||
           context == SimulationRecipe::soft_body_rope ||
           context == SimulationRecipe::soft_body_smoke;
}

// Owning, synchronous fixed-step simulation without a window or renderer.
// Device pointers in render_view() remain owned by this object. Reacquire the
// view after every step/reset and do not retain it across initialize() or move.
// All public operations translate implementation exceptions into Status.
// CUDA and ParallelMater dependency failures preserve their StatusCode and
// cudaError_t; unexpected implementation exceptions become internal_error.
class GallerySimulation {
  public:
    GallerySimulation() noexcept;
    ~GallerySimulation();
    GallerySimulation(GallerySimulation &&) noexcept;
    GallerySimulation &operator=(GallerySimulation &&) noexcept;
    GallerySimulation(const GallerySimulation &) = delete;
    GallerySimulation &operator=(const GallerySimulation &) = delete;

    [[nodiscard]] static Status create(GallerySimulationOptions options, GallerySimulation &output,
                                       cudaStream_t stream = nullptr) noexcept;

    [[nodiscard]] Status initialize(GallerySimulationOptions options,
                                    cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status step(cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status reset(cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status resize_particles(std::uint32_t active_count,
                                          cudaStream_t stream = nullptr) noexcept;

    [[nodiscard]] bool initialized() const noexcept;
    [[nodiscard]] SimulationRecipe recipe() const noexcept;
    // The returned asset-path view refers to memory owned by this simulation.
    [[nodiscard]] GallerySimulationOptions options() const noexcept;
    [[nodiscard]] ResolvedPhysicsOptions resolved_physics() const noexcept;
    [[nodiscard]] GallerySimulationStatistics statistics() const noexcept;
    [[nodiscard]] FrameRenderView render_view() const noexcept;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Fluent convenience layer for applications. Portable values live in
// RecipeConfig; CUDA-specific gravity, asset path, stream, and ownership
// enter only at build(). Existing GallerySimulationOptions remains available
// for callers that prefer aggregate initialization.
class SimulationBuilder {
  public:
    explicit SimulationBuilder(SimulationRecipe recipe) noexcept
        : config_(RecipeConfig::for_recipe(recipe)) {}

    SimulationBuilder &timestep(float value) noexcept;
    SimulationBuilder &iterations(std::uint32_t value) noexcept;
    SimulationBuilder &particles(std::uint32_t value) noexcept;
    SimulationBuilder &skin_frequency(std::uint32_t value) noexcept;
    SimulationBuilder &rope_nodes(std::uint32_t value) noexcept;
    SimulationBuilder &cloth_detail(std::uint32_t value) noexcept;
    SimulationBuilder &bridge_grid(std::uint32_t columns, std::uint32_t rows) noexcept;
    SimulationBuilder &cylinder_grid(std::uint32_t columns, std::uint32_t rows) noexcept;
    SimulationBuilder &gravity(float3 value) noexcept;
    SimulationBuilder &soft_body_asset(std::string path);

    [[nodiscard]] const RecipeConfig &config() const noexcept { return config_; }
    [[nodiscard]] Status build(GallerySimulation &output,
                               cudaStream_t stream = nullptr) const noexcept;

  private:
    RecipeConfig config_{};
    std::optional<float3> gravity_{};
    std::optional<std::uint32_t> bridge_columns_{};
    std::optional<std::uint32_t> bridge_rows_{};
    std::optional<std::uint32_t> cylinder_columns_{};
    std::optional<std::uint32_t> cylinder_rows_{};
    std::string asset_path_{};
};

} // namespace parallel_mater::examples
