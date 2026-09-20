// SPDX-License-Identifier: MIT
#pragma once

#include <parallel_mater/geometry.hpp>

#include <cuda_runtime_api.h>
#include <vector_types.h>

#include <concepts>
#include <cstdint>
#include <utility>

namespace parallel_mater::physics {

// Shared fixed-step description used by every owning solver.  A frame is
// divided into equal physical substeps; render cadence remains application
// policy and never changes the amount of simulated time.
struct FrameOptions {
    float timestep{1.0F / 60.0F};
    std::uint32_t substeps{1U};
    float3 acceleration{};
};

// Immutable context passed through one prepare/couple/finish sequence.
struct SubstepContext {
    std::uint32_t index{};
    std::uint32_t count{1U};
    float timestep{1.0F / 60.0F};
    float3 acceleration{};
};

// Common borrowed point interface for deterministic contact kernels. Solvers
// retain positions/velocities; couplers gather one impulse per point into the
// writable buffer during the prepared substep.
struct PointCouplingView {
    const float3* positions{};
    const float3* velocities{};
    float3* external_impulses{};
    std::uint32_t count{};
    float radius{};
    float inverse_mass{};
};

[[nodiscard]] bool valid(FrameOptions options) noexcept;
[[nodiscard]] SubstepContext substep_context(
    FrameOptions options, std::uint32_t index) noexcept;

// Movable CUDA completion event.  Async solver calls enqueue work, record this
// token, and return without synchronizing.  wait() is the synchronous
// convenience boundary and reports deferred CUDA execution failures.
class Completion {
public:
    Completion() noexcept = default;
    ~Completion();
    Completion(Completion&& other) noexcept;
    Completion& operator=(Completion&& other) noexcept;
    Completion(const Completion&) = delete;
    Completion& operator=(const Completion&) = delete;

    [[nodiscard]] Status record(cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status wait() noexcept;
    [[nodiscard]] bool ready() noexcept;
    [[nodiscard]] bool pending() const noexcept { return pending_; }
    [[nodiscard]] Status status() const noexcept { return status_; }

private:
    cudaEvent_t event_{};
    Status status_{};
    bool pending_{};
};

// Structural protocol for independently owned solvers.  Coupling kernels may
// read a solver's borrowed view and write its documented coupling buffers only
// after prepare_substep and before finish_substep.
template <typename Solver>
concept FrameSolver = requires(
    Solver& solver, FrameOptions frame, SubstepContext substep,
    Completion& completion, cudaStream_t stream) {
    { solver.begin_frame(frame, stream) } -> std::same_as<Status>;
    { solver.prepare_substep(substep, stream) } -> std::same_as<Status>;
    { solver.finish_substep(substep, stream) } -> std::same_as<Status>;
    { solver.finish_frame(completion, stream) } -> std::same_as<Status>;
};

// Convenience driver for uncoupled frames.  Applications that couple two or
// more solvers use the same calls explicitly and insert coupling work between
// prepare_substep() and finish_substep().
template <FrameSolver Solver>
[[nodiscard]] Status step_async(
    Solver& solver, FrameOptions frame, Completion& completion,
    cudaStream_t stream = nullptr) noexcept
{
    Status status = solver.begin_frame(frame, stream);
    if (!status) return status;
    for (std::uint32_t i = 0U; i < frame.substeps; ++i) {
        const SubstepContext substep = substep_context(frame, i);
        status = solver.prepare_substep(substep, stream);
        if (!status) return status;
        status = solver.finish_substep(substep, stream);
        if (!status) return status;
    }
    return solver.finish_frame(completion, stream);
}

template <FrameSolver Solver>
[[nodiscard]] Status step(
    Solver& solver, FrameOptions frame,
    cudaStream_t stream = nullptr) noexcept
{
    Completion completion;
    Status status = step_async(solver, frame, completion, stream);
    return status ? completion.wait() : status;
}

} // namespace parallel_mater::physics
