// SPDX-License-Identifier: MIT
#pragma once

#include "../../src/internal/status_exception.hpp"

namespace waterlab::detail {
using parallel_mater::physics::detail::StatusException;
using parallel_mater::physics::detail::cuda_status;
using parallel_mater::physics::detail::throw_if_failed;
} // namespace waterlab::detail
