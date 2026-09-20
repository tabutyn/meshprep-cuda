// SPDX-License-Identifier: MIT
#include "gallery.hpp"

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdio>
#include <stdexcept>

#ifndef PARALLEL_MATER_SIMULATION_TEST_ASSET
#define PARALLEL_MATER_SIMULATION_TEST_ASSET "assets/softbody/checker_cylinder.msb"
#endif

namespace {

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

using parallel_mater::examples::Component;
using parallel_mater::examples::SimulationRecipe;

std::uint32_t deformable_count(Component components) {
    return static_cast<std::uint32_t>(
               parallel_mater::examples::has_component(components, Component::cloth)) +
           static_cast<std::uint32_t>(
               parallel_mater::examples::has_component(components, Component::rope)) +
           static_cast<std::uint32_t>(
               parallel_mater::examples::has_component(components, Component::soft_body));
}

void test_every_recipe() {
    for (const auto &recipe : parallel_mater::examples::simulation_recipes) {
        parallel_mater::examples::GallerySimulationOptions options;
        options.recipe = recipe.recipe;
        options.solver_iterations_override = 2U;
        options.soft_body_asset_path = PARALLEL_MATER_SIMULATION_TEST_ASSET;
        if (parallel_mater::examples::has_component(recipe.components, Component::fluid_particles))
            options.particle_count_override = 256U;

        parallel_mater::examples::GallerySimulation simulation;
        const parallel_mater::Status initialized = simulation.initialize(options);
        if (!initialized) {
            std::fprintf(stderr, "recipe %.*s initialization failed: %s\n",
                         static_cast<int>(recipe.slug.size()), recipe.slug.data(),
                         initialized.message);
        }
        require(initialized.ok(), "public gallery recipe failed initialization");
        const auto initial = simulation.render_view();
        const bool fluid =
            parallel_mater::examples::has_component(recipe.components, Component::fluid_particles);
        const bool smoke =
            parallel_mater::examples::has_component(recipe.components, Component::smoke);
        const bool rigid =
            parallel_mater::examples::has_component(recipe.components, Component::rigid_bodies);
        require(initial.particle_system_count ==
                    static_cast<std::uint32_t>(fluid) + static_cast<std::uint32_t>(smoke),
                "gallery particle owners do not match recipe components");
        require(initial.surface_count == deformable_count(recipe.components) &&
                    initial.lattice_count == deformable_count(recipe.components),
                "gallery deformable owners do not match recipe components");
        require(initial.rigid_body_count == static_cast<std::uint32_t>(rigid),
                "gallery rigid-body owner does not match recipe components");

        for (std::uint32_t frame = 0U; frame < 3U; ++frame) {
            const parallel_mater::Status stepped = simulation.step();
            if (!stepped) {
                std::fprintf(stderr, "recipe %.*s frame %u failed: %s (cuda=%s)\n",
                             static_cast<int>(recipe.slug.size()), recipe.slug.data(), frame,
                             stepped.message, cudaGetErrorString(stepped.cuda_error));
            }
            require(stepped.ok(), "public gallery recipe failed a multi-step test");
        }
        const auto statistics = simulation.statistics();
        require(statistics.frame_index == 3U && statistics.finite_failure_count == 0U,
                "gallery recipe returned inconsistent statistics");
        const auto current = simulation.render_view();
        for (std::uint32_t index = 0U; index < current.particle_system_count; ++index) {
            float3 first{};
            require(cudaMemcpy(&first, current.particle_systems[index].positions, sizeof(first),
                               cudaMemcpyDeviceToHost) == cudaSuccess &&
                        std::isfinite(first.x) && std::isfinite(first.y) && std::isfinite(first.z),
                    "gallery particle owner produced unreadable or non-finite state");
        }
        require(simulation.reset().ok() && simulation.statistics().frame_index == 0U,
                "gallery recipe did not reset to frame zero");
    }
}

void test_particle_resize() {
    parallel_mater::examples::GallerySimulation simulation;
    parallel_mater::examples::GallerySimulationOptions options;
    options.recipe = SimulationRecipe::water;
    options.particle_count_override = 64U;
    require(simulation.initialize(options).ok(), "fluid gallery failed initialization");
    require(simulation.resize_particles(512U).ok(), "fluid owner failed to resize");
    require(simulation.render_view().particle_systems[0].count == 512U,
            "fluid resize did not replace the public owner");
    require(!simulation.resize_particles(0U), "fluid resize accepted zero particles");
}

} // namespace

int main() {
    int devices{};
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::puts("SKIP: CUDA device unavailable");
        return 77;
    }
    try {
        test_every_recipe();
        test_particle_resize();
        std::puts("all public gallery runtime tests passed");
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "simulation runtime API test failure: %s\n", error.what());
        return 1;
    }
}
