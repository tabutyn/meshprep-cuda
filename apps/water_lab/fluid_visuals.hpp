// SPDX-License-Identifier: MIT
#pragma once

#include "fluid_surface.hpp"

#include <parallel_mater/geometry.hpp>
#include <cuda_runtime_api.h>
#include <vector_types.h>
#include <cstdint>
#include <vector>

namespace waterlab {

// Keep Wireframe second so one V press from the default surface view enters
// the geometry diagnostic requested by the interactive workflow.
enum class FluidDisplay : std::uint32_t {
    Surface, Wireframe, Particles, Billboards
};

struct FoamParticle {
    float4 position_age{};
    float4 velocity_life{};
    float4 normal_radius{};

    // Active records have a nonnegative age and a positive lifetime.
    [[nodiscard]] __host__ __device__ bool active() const noexcept {
        return position_age.w >= 0.0F && velocity_life.w > 0.0F;
    }
};

struct FluidVisualView {
    // xyz: outward particle-distribution normal; w: instantaneous foam source [0,1].
    const float4* normal_foam{};
    std::uint32_t particle_count{};
    FluidDisplay display{FluidDisplay::Surface};
    bool show_foam{true};
    // Wireframe mode normally retains the particle layer beneath host-drawn
    // meshes. Component-only gallery contexts can disable that layer without
    // inventing another display mode.
    bool show_particle_layer{true};
    FluidSurfaceView surface{};
    const FoamParticle* foam_particles{};
    std::uint32_t foam_capacity{};
    const parallel_mater::HierarchyNode* foam_nodes{};
    const std::uint32_t* foam_indices{};
    std::uint32_t foam_node_count{};
};

struct FoamSettings {
    float emission_rate{8.0F};
    float radius_scale{1.6F};
    float lifetime_scale{1.25F};
};

// Rendering-only state. Never changes particle positions, velocities, or forces.
// Foam is a fixed-capacity, rendering-only particle layer on the derived surface.
class FluidVisuals {
public:
    static constexpr std::uint32_t foam_capacity{2'048U};

    explicit FluidVisuals(std::uint32_t particle_count,
        std::uint32_t surface_resolution = 48U);
    ~FluidVisuals();
    FluidVisuals(const FluidVisuals&) = delete;
    FluidVisuals& operator=(const FluidVisuals&) = delete;

    [[nodiscard]] float update(const float3* positions, const float3* velocities,
        const parallel_mater::Hierarchy& hierarchy, float support_radius, float3 gravity,
        float dt, cudaStream_t stream = nullptr, bool obstacle_course = false,
        ParticleCellView cells = {});
    void reset(cudaStream_t stream = nullptr);
    void set_active_count(std::uint32_t count);
    void set_foam_settings(FoamSettings settings);
    [[nodiscard]] FoamSettings foam_settings() const noexcept { return foam_settings_; }
    void capture(std::vector<float4>& output, cudaStream_t stream = nullptr) const;
    void restore(const std::vector<float4>& input, cudaStream_t stream = nullptr);
    void capture_foam(std::vector<FoamParticle>& output, std::uint64_t& tick,
        cudaStream_t stream = nullptr) const;
    void restore_foam(const std::vector<FoamParticle>& input, std::uint64_t tick,
        cudaStream_t stream = nullptr);
    [[nodiscard]] FluidVisualView view(FluidDisplay display = FluidDisplay::Surface,
        bool show_foam = true, bool show_particle_layer = true) const noexcept {
        return {normal_foam_, count_, display, show_foam, show_particle_layer, surface_.view(),
            foam_particles_, foam_capacity, foam_hierarchy_.nodes(),
            foam_hierarchy_.primitive_indices(), foam_hierarchy_.statistics().node_count};
    }
    [[nodiscard]] std::size_t allocated_bytes() const noexcept {
        return static_cast<std::size_t>(capacity_) * sizeof(float4) +
            foam_capacity * (sizeof(FoamParticle) + sizeof(parallel_mater::Aabb)) +
            sizeof(float3) + sizeof(std::uint32_t) + surface_.allocated_bytes() +
            foam_workspace_.capacity_bytes() + foam_hierarchy_.allocated_bytes();
    }
private:
    FluidSurface surface_;
    float4* normal_foam_{};
    FoamParticle* foam_particles_{};
    parallel_mater::Aabb* foam_bounds_{};
    float3* foam_anchor_{};
    std::uint32_t* errors_{};
    std::uint32_t count_{};
    std::uint32_t capacity_{};
    std::uint64_t tick_{};
    FoamSettings foam_settings_{};
    parallel_mater::Workspace foam_workspace_;
    parallel_mater::Hierarchy foam_hierarchy_;
    cudaEvent_t begin_{}, end_{};
};

} // namespace waterlab
