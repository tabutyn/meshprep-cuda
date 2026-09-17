// SPDX-License-Identifier: MIT
#pragma once

#include <meshprep/meshprep.hpp>

#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>

namespace waterlab::detail {

// Internal adapter for legacy solver implementations that report failures by
// exception. Retaining the complete Status prevents the installed noexcept API
// from having to infer CUDA errors from human-readable exception text.
class StatusException final : public std::runtime_error {
public:
    StatusException(meshprep::Status status, const char* operation)
        : std::runtime_error(std::string(operation) + ": " + status.message),
          status_(status)
    {
    }

    [[nodiscard]] meshprep::Status status() const noexcept { return status_; }

private:
    meshprep::Status status_{};
};

[[nodiscard]] inline meshprep::Status cuda_status(
    cudaError_t error, const char* message) noexcept
{
    return {
        error == cudaErrorMemoryAllocation
            ? meshprep::StatusCode::allocation_failure
            : meshprep::StatusCode::cuda_failure,
        error,
        message};
}

inline void throw_if_failed(cudaError_t error, const char* operation)
{
    if (error != cudaSuccess) {
        throw StatusException(cuda_status(error, cudaGetErrorString(error)), operation);
    }
}

inline void throw_if_failed(meshprep::Status status, const char* operation)
{
    if (!status) throw StatusException(status, operation);
}

} // namespace waterlab::detail
