// SPDX-License-Identifier: MIT
#pragma once

#include <parallel_mater/geometry.hpp>

#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>

namespace parallel_mater::physics::detail {

// Internal adapter for legacy solver implementations that report failures by
// exception. Retaining the complete Status prevents the installed noexcept API
// from having to infer CUDA errors from human-readable exception text.
class StatusException final : public std::runtime_error {
  public:
    StatusException(parallel_mater::Status status, const char *operation)
        : std::runtime_error(std::string(operation) + ": " + status.message), status_(status) {}

    [[nodiscard]] parallel_mater::Status status() const noexcept { return status_; }

  private:
    parallel_mater::Status status_{};
};

[[nodiscard]] inline parallel_mater::Status cuda_status(cudaError_t error,
                                                        const char *message) noexcept {
    return error == cudaSuccess
               ? parallel_mater::Status{}
               : parallel_mater::Status{error == cudaErrorMemoryAllocation
                                            ? parallel_mater::StatusCode::allocation_failure
                                            : parallel_mater::StatusCode::cuda_failure,
                                        error, message};
}

inline void throw_if_failed(cudaError_t error, const char *operation) {
    if (error != cudaSuccess) {
        throw StatusException(cuda_status(error, cudaGetErrorString(error)), operation);
    }
}

inline void throw_if_failed(parallel_mater::Status status, const char *operation) {
    if (!status) throw StatusException(status, operation);
}

} // namespace parallel_mater::physics::detail
