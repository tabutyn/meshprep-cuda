// SPDX-License-Identifier: MIT
#pragma once

#include <meshprep/meshprep.hpp>

#include <vector_types.h>

#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string_view>

namespace meshprep::sim {

// Components are deliberately composable. Example contexts are recipes built
// from these flags; they are not separate solver implementations.
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
    return (static_cast<std::uint32_t>(set) & static_cast<std::uint32_t>(component)) != 0U;
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
};

struct ExampleContextInfo {
    ExampleContext id{};
    char key{};
    std::string_view slug{};
    std::string_view title{};
    Component components{};
};

inline constexpr std::array<ExampleContextInfo, 8> example_contexts{{
    {ExampleContext::water_course, '1', "water-course", "Water obstacle course",
        Component::fluid_particles | Component::water_skin | Component::rigid_bodies},
    {ExampleContext::particle_bowl, '2', "particle-bowl", "Particles in a hemispherical bowl",
        Component::fluid_particles | Component::rigid_bodies},
    {ExampleContext::cloth_rigid, '3', "cloth-rigid", "Rolling rigid sphere and hanging cloth",
        Component::cloth | Component::rigid_bodies},
    {ExampleContext::soft_body_rigid, '4', "soft-body-rigid", "Rolling rigid sphere and soft post",
        Component::soft_body | Component::rigid_bodies},
    {ExampleContext::particles_cloth, '5', "particles-cloth", "Rigid sphere, particles, and catching cloth",
        Component::fluid_particles | Component::cloth | Component::rigid_bodies},
    {ExampleContext::soft_body_fluid, '6', "soft-body-fluid", "Water wheel with a soft axle cross",
        Component::soft_body | Component::fluid_particles | Component::rigid_bodies},
    {ExampleContext::soft_body_cloth, '7', "soft-body-cloth", "Rolling soft sphere and cloth",
        Component::soft_body | Component::cloth | Component::rigid_bodies},
    {ExampleContext::rope_rigid, '8', "rope-rigid", "Rigid sphere tethered to a center post",
        Component::soft_body | Component::rigid_bodies},
}};

[[nodiscard]] constexpr const ExampleContextInfo* find_example_context(char key) noexcept
{
    for (const auto& context : example_contexts) {
        if (context.key == key) return &context;
    }
    return nullptr;
}

// Read-only CUDA views form the rendering boundary. Simulation implementations
// retain ownership and may replace their allocations between frames; callers
// must reacquire views after every simulation step.
struct ParticleRenderView {
    const float3* positions{};
    const float3* velocities{};
    std::uint32_t count{};
    float radius{};
};

struct SurfaceRenderView {
    DeviceMeshView mesh{};
    const float3* vertex_normals{};
    const float2* texcoords{};
    const std::uint8_t* triangle_active{};
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

struct LatticeBond {
    uint2 vertices{};
    float rest_length{};
};

// Read-only CUDA arrays include every node, including the solid interior.
// Bonds are shared local topology: instance i uses positions at
// i * nodes_per_instance + bonds[b].vertices.{x,y}. Its live connection mask
// is bond_active[i * bonds_per_instance + b]; zero means permanently broken
// until reset. Flags use lattice_node_surface and lattice_node_pinned.
struct LatticeRenderView {
    const float3* positions{};
    const std::uint32_t* flags{};
    const LatticeBond* bonds{};
    const std::uint8_t* bond_active{};
    std::uint32_t node_count{};
    std::uint32_t nodes_per_instance{};
    std::uint32_t bonds_per_instance{};
    std::uint32_t instance_count{};
    float node_radius{};
};

struct FrameRenderView {
    const ParticleRenderView* particle_systems{};
    std::uint32_t particle_system_count{};
    const SurfaceRenderView* surfaces{};
    std::uint32_t surface_count{};
    const RigidBodyRenderView* rigid_bodies{};
    std::uint32_t rigid_body_count{};
    const LatticeRenderView* lattices{};
    std::uint32_t lattice_count{};
};

struct FixedStepOptions {
    float timestep{1.0F / 60.0F};
};

// Header-only validation keeps the fixed-step contract identical for sample
// applications and future solver backends without introducing a windowing or
// renderer dependency into the installed package.
[[nodiscard]] constexpr bool valid(FixedStepOptions options) noexcept
{
    return options.timestep > 0.0F && options.timestep == options.timestep &&
        options.timestep <= std::numeric_limits<float>::max();
}

// Configuration for the experimental, headless gallery backend. The asset
// path is read only by initialize(); GallerySimulation owns its own copy after
// initialization. Context 4 requires the converted cylinder .msb asset; the
// water wheel, rigid course, and other contexts are generated procedurally.
struct GallerySimulationOptions {
    ExampleContext context{ExampleContext::particle_bowl};
    FixedStepOptions fixed_step{};
    // Empty overrides select recipe defaults. An explicit particle count also
    // sizes the reserved capacity, up to the 100,000-particle stress ceiling.
    std::optional<std::uint32_t> solver_iterations_override{};
    std::optional<float3> gravity_override{};
    std::optional<std::uint32_t> particle_count_override{};
    std::optional<std::uint32_t> physical_skin_frequency_override{};
    std::optional<std::uint32_t> rope_node_count_override{};
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

[[nodiscard]] constexpr bool requires_soft_body_asset(
    ExampleContext context) noexcept
{
    return context == ExampleContext::soft_body_rigid;
}

// Owning, synchronous fixed-step simulation without a window or renderer.
// Device pointers in render_view() remain owned by this object. Reacquire the
// view after every step/reset and do not retain it across initialize() or move.
// All public operations translate implementation exceptions into Status.
// CUDA and meshprep dependency failures preserve their StatusCode and
// cudaError_t; unexpected implementation exceptions become internal_error.
class GallerySimulation {
public:
    GallerySimulation() noexcept;
    ~GallerySimulation();
    GallerySimulation(GallerySimulation&&) noexcept;
    GallerySimulation& operator=(GallerySimulation&&) noexcept;
    GallerySimulation(const GallerySimulation&) = delete;
    GallerySimulation& operator=(const GallerySimulation&) = delete;

    [[nodiscard]] static Status create(
        GallerySimulationOptions options,
        GallerySimulation& output,
        cudaStream_t stream = nullptr) noexcept;

    [[nodiscard]] Status initialize(
        GallerySimulationOptions options,
        cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status step(cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status reset(cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status resize_particles(
        std::uint32_t active_count, cudaStream_t stream = nullptr) noexcept;

    [[nodiscard]] bool initialized() const noexcept;
    [[nodiscard]] ExampleContext context() const noexcept;
    // The returned asset-path view refers to memory owned by this simulation.
    [[nodiscard]] GallerySimulationOptions options() const noexcept;
    [[nodiscard]] ResolvedPhysicsOptions resolved_physics() const noexcept;
    [[nodiscard]] GallerySimulationStatistics statistics() const noexcept;
    [[nodiscard]] FrameRenderView render_view() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace meshprep::sim
