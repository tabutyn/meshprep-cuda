// SPDX-License-Identifier: MIT
#include <meshprep/meshprep.hpp>
#include <meshprep/simulation.hpp>

int main()
{
    meshprep::Workspace workspace;
    meshprep::sim::GallerySimulation simulation;
    meshprep::sim::GallerySimulationOptions options;
    options.context = meshprep::sim::ExampleContext::particle_bowl;
    if (!simulation.initialize(options) || !simulation.step()) return 1;
    const auto frame = simulation.render_view();
    const auto physics = simulation.resolved_physics();
    return workspace.capacity_bytes() == 0 &&
            frame.particle_system_count == 1U &&
            frame.particle_systems[0].count == 10'000U &&
            frame.surface_count == 0U && frame.rigid_body_count == 1U &&
            physics.solver_iterations == 4U &&
            physics.gravity.x == 0.0F && physics.gravity.y == -9.81F &&
            physics.gravity.z == 0.0F && !physics.rigid_course_preset
        ? 0 : 1;
}
