// SPDX-License-Identifier: MIT
#include <parallel_mater/rigid_body.hpp>
#include <type_traits>
static_assert(std::is_move_constructible_v<parallel_mater::physics::RigidBody>);
