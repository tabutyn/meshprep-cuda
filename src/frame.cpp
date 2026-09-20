// SPDX-License-Identifier: MIT
#include <parallel_mater/frame.hpp>

#include <cmath>
#include <utility>

namespace parallel_mater::physics {
namespace {

[[nodiscard]] constexpr Status invalid(const char *message) noexcept {
    return {StatusCode::invalid_argument, cudaSuccess, message};
}

[[nodiscard]] Status cuda_status(cudaError_t error, const char *message) noexcept {
    return error == cudaSuccess
               ? Status{}
               : Status{error == cudaErrorMemoryAllocation ? StatusCode::allocation_failure
                                                           : StatusCode::cuda_failure,
                        error, message};
}

[[nodiscard]] bool finite(float3 value) noexcept {
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

} // namespace

bool valid(FrameOptions options) noexcept {
    return std::isfinite(options.timestep) && options.timestep > 0.0F && options.substeps > 0U &&
           options.substeps <= 256U && finite(options.acceleration);
}

SubstepContext substep_context(FrameOptions options, std::uint32_t index) noexcept {
    return {index, options.substeps, options.timestep / static_cast<float>(options.substeps),
            options.acceleration};
}

Completion::~Completion() {
    if (event_ != nullptr) cudaEventDestroy(event_);
}

Completion::Completion(Completion &&other) noexcept
    : event_(std::exchange(other.event_, nullptr)), status_(other.status_),
      pending_(std::exchange(other.pending_, false)) {}

Completion &Completion::operator=(Completion &&other) noexcept {
    if (this == &other) return *this;
    if (event_ != nullptr) cudaEventDestroy(event_);
    event_ = std::exchange(other.event_, nullptr);
    status_ = other.status_;
    pending_ = std::exchange(other.pending_, false);
    return *this;
}

Status Completion::record(cudaStream_t stream) noexcept {
    if (pending_) return invalid("completion token is already pending");
    if (event_ == nullptr) {
        status_ = cuda_status(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming),
                              "could not create completion event");
        if (!status_) return status_;
    }
    status_ = cuda_status(cudaEventRecord(event_, stream), "could not record completion event");
    pending_ = status_.ok();
    return status_;
}

Status Completion::wait() noexcept {
    if (!pending_) return status_;
    status_ = cuda_status(cudaEventSynchronize(event_), "asynchronous CUDA operation failed");
    pending_ = false;
    return status_;
}

bool Completion::ready() noexcept {
    if (!pending_) return true;
    const cudaError_t result = cudaEventQuery(event_);
    if (result == cudaErrorNotReady) return false;
    status_ = cuda_status(result, "asynchronous CUDA operation failed");
    pending_ = false;
    return true;
}

} // namespace parallel_mater::physics
