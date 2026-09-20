// SPDX-License-Identifier: MIT
#include <parallel_mater/gallery.hpp>

#include <cuda_runtime_api.h>

#include <cstdio>

int main()
{
    // Water is fully procedural: no application window, renderer, or asset
    // path is required. The simulation owns all CUDA allocations.
    parallel_mater::sim::GallerySimulation simulation;
    // One fluent builder is enough to select a complete authored simulation.
    // Omitted properties retain the level preset; no window or renderer is
    // pulled into the installed headless library.
    const parallel_mater::Status initialized =
        parallel_mater::sim::SimulationBuilder(
            parallel_mater::sim::ExampleContext::water)
            .particles(512U)
            .timestep(1.0F / 60.0F)
            .build(simulation);
    if (!initialized) {
        std::fprintf(stderr, "initialize failed: %s\n", initialized.message);
        return 1;
    }
    const parallel_mater::Status stepped = simulation.step();
    if (!stepped) {
        std::fprintf(stderr, "step failed: %s\n", stepped.message);
        return 1;
    }
    // Add active particle IDs without reallocating the reserved GPU capacity.
    const parallel_mater::Status resized = simulation.resize_particles(2'048U);
    if (!resized) {
        std::fprintf(stderr, "particle resize failed: %s\n", resized.message);
        return 1;
    }
    const parallel_mater::Status second_step = simulation.step();
    if (!second_step) {
        std::fprintf(stderr, "second step failed: %s\n", second_step.message);
        return 1;
    }

    // Views are borrowed device pointers and must be reacquired after a step.
    const parallel_mater::sim::FrameRenderView frame = simulation.render_view();
    if (frame.particle_system_count != 1U || frame.particle_systems == nullptr ||
        frame.particle_systems[0].count != 2'048U) {
        std::fprintf(stderr, "unexpected Water render view\n");
        return 1;
    }
    float3 first_particle{};
    const cudaError_t copied = cudaMemcpy(&first_particle,
        frame.particle_systems[0].positions, sizeof(first_particle),
        cudaMemcpyDeviceToHost);
    if (copied != cudaSuccess) {
        std::fprintf(stderr, "particle download failed: %s\n", cudaGetErrorString(copied));
        return 1;
    }

    const auto stats = simulation.statistics();
    const auto physics = simulation.resolved_physics();
    std::printf(
        "Water: %u particles, %u rigid bodies, frame %llu, "
        "dt=%.5f, iterations=%u, gravity=(%.2f, %.2f, %.2f), "
        "first=(%.3f, %.3f, %.3f)\n",
        stats.particle_count, stats.rigid_body_count,
        static_cast<unsigned long long>(stats.frame_index),
        physics.fixed_step.timestep, physics.solver_iterations,
        physics.gravity.x, physics.gravity.y, physics.gravity.z,
        first_particle.x, first_particle.y, first_particle.z);
    return 0;
}
