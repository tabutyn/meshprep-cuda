// SPDX-License-Identifier: MIT
#include <meshprep/simulation.hpp>

#include <cuda_runtime_api.h>

#include <cstdio>

int main()
{
    // Context 2 is fully procedural: no application window, renderer, or asset
    // path is required. The simulation owns all CUDA allocations.
    meshprep::sim::GallerySimulationOptions options;
    options.context = meshprep::sim::ExampleContext::particle_bowl;
    options.fixed_step.timestep = 1.0F / 60.0F;
    options.particle_count_override = 512U;
    // Leave gravity_override and solver_iterations_override empty to use the
    // context-2 Earth-gravity preset. Set either optional field when a caller
    // deliberately wants to depart from the catalog recipe.

    meshprep::sim::GallerySimulation simulation;
    const meshprep::Status initialized = simulation.initialize(options);
    if (!initialized) {
        std::fprintf(stderr, "initialize failed: %s\n", initialized.message);
        return 1;
    }
    const meshprep::Status stepped = simulation.step();
    if (!stepped) {
        std::fprintf(stderr, "step failed: %s\n", stepped.message);
        return 1;
    }
    // Add active particle IDs without reallocating the reserved GPU capacity.
    const meshprep::Status resized = simulation.resize_particles(2'048U);
    if (!resized) {
        std::fprintf(stderr, "particle resize failed: %s\n", resized.message);
        return 1;
    }
    const meshprep::Status second_step = simulation.step();
    if (!second_step) {
        std::fprintf(stderr, "second step failed: %s\n", second_step.message);
        return 1;
    }

    // Views are borrowed device pointers and must be reacquired after a step.
    const meshprep::sim::FrameRenderView frame = simulation.render_view();
    if (frame.particle_system_count != 1U || frame.particle_systems == nullptr ||
        frame.particle_systems[0].count != 2'048U) {
        std::fprintf(stderr, "unexpected context-2 render view\n");
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
        "context 2: %u particles, %u rigid bodies, frame %llu, "
        "dt=%.5f, iterations=%u, gravity=(%.2f, %.2f, %.2f), "
        "first=(%.3f, %.3f, %.3f)\n",
        stats.particle_count, stats.rigid_body_count,
        static_cast<unsigned long long>(stats.frame_index),
        physics.fixed_step.timestep, physics.solver_iterations,
        physics.gravity.x, physics.gravity.y, physics.gravity.z,
        first_particle.x, first_particle.y, first_particle.z);
    return 0;
}
