// SPDX-License-Identifier: MIT
#include <parallel_mater/cloth.hpp>
#include <parallel_mater/fluid.hpp>
#include <parallel_mater/geometry.hpp>
#include <parallel_mater/frame.hpp>
#include <parallel_mater/physics.hpp>
#include <parallel_mater/rigid_body.hpp>
#include <parallel_mater/rope.hpp>
#include <parallel_mater/smoke.hpp>

#include <type_traits>

int main()
{
    static_assert(std::is_move_constructible_v<parallel_mater::physics::SoftBody>);
    static_assert(!std::is_copy_constructible_v<parallel_mater::physics::SoftBody>);
    static_assert(std::is_move_constructible_v<parallel_mater::physics::Smoke>);
    static_assert(!std::is_copy_constructible_v<parallel_mater::physics::Smoke>);
    static_assert(std::is_move_constructible_v<parallel_mater::physics::Fluid>);
    static_assert(std::is_move_constructible_v<parallel_mater::physics::Cloth>);
    static_assert(std::is_move_constructible_v<parallel_mater::physics::Rope>);
    static_assert(std::is_move_constructible_v<parallel_mater::physics::RigidBody>);

    parallel_mater::physics::SoftBody soft_body;
    parallel_mater::physics::SoftBodyOptions options;
    parallel_mater::physics::Smoke smoke;
    parallel_mater::physics::Fluid fluid;
    parallel_mater::physics::Cloth cloth;
    parallel_mater::physics::Rope rope;
    parallel_mater::physics::RigidBody rigid_body;
    parallel_mater::Workspace workspace;
    return !soft_body.initialized() && !smoke.initialized() &&
            !fluid.initialized() && !cloth.initialized() &&
            !rope.initialized() && !rigid_body.initialized() &&
            options.instance_count == 1U &&
            options.substeps == 4U && workspace.capacity_bytes() == 0U
        ? 0
        : 1;
}
