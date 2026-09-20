// SPDX-License-Identifier: MIT
#include <parallel_mater/geometry.hpp>
#include <parallel_mater/gallery.hpp>
#include <parallel_mater/physics.hpp>
#include <parallel_mater/recipes.hpp>
#include <parallel_mater/smoke.hpp>

#include <type_traits>

int main()
{
    static_assert(std::is_move_constructible_v<parallel_mater::physics::SoftBody>);
    static_assert(!std::is_copy_constructible_v<parallel_mater::physics::SoftBody>);
    static_assert(std::is_move_constructible_v<parallel_mater::physics::Smoke>);
    static_assert(!std::is_copy_constructible_v<parallel_mater::physics::Smoke>);
    static_assert(std::is_move_constructible_v<parallel_mater::sim::GallerySimulation>);

    parallel_mater::physics::SoftBody soft_body;
    parallel_mater::physics::SoftBodyOptions options;
    parallel_mater::physics::Smoke smoke;
    parallel_mater::sim::GallerySimulation gallery;
    const auto* recipe=parallel_mater::sim::find_simulation_recipe("water");
    parallel_mater::Workspace workspace;
    return !soft_body.initialized() && !smoke.initialized() && !gallery.initialized() &&
            recipe!=nullptr && recipe->recipe==parallel_mater::sim::SimulationRecipe::water &&
            options.instance_count == 1U &&
            options.substeps == 4U && workspace.capacity_bytes() == 0U
        ? 0
        : 1;
}
