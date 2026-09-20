// SPDX-License-Identifier: MIT
#include <parallel_mater/geometry.hpp>
#include <parallel_mater/physics.hpp>
#include <parallel_mater/smoke.hpp>

#include <type_traits>

int main()
{
    static_assert(std::is_move_constructible_v<parallel_mater::physics::SoftBody>);
    static_assert(!std::is_copy_constructible_v<parallel_mater::physics::SoftBody>);
    static_assert(std::is_move_constructible_v<parallel_mater::physics::Smoke>);
    static_assert(!std::is_copy_constructible_v<parallel_mater::physics::Smoke>);

    parallel_mater::physics::SoftBody soft_body;
    parallel_mater::physics::SoftBodyOptions options;
    parallel_mater::physics::Smoke smoke;
    parallel_mater::Workspace workspace;
    return !soft_body.initialized() && !smoke.initialized() &&
            options.instance_count == 1U &&
            options.substeps == 4U && workspace.capacity_bytes() == 0U
        ? 0
        : 1;
}
