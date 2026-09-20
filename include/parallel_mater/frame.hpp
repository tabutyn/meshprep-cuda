// SPDX-License-Identifier: MIT
#pragma once

#include <parallel_mater/geometry.hpp>

#include <cuda_runtime_api.h>
#include <vector_types.h>

#include <array>
#include <concepts>
#include <cstdint>
#include <type_traits>
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
struct PointStateView {
    const float3 *positions{};
    const float3 *velocities{};
    float3 *external_impulses{};
    // Optional. Solvers that expose positional projection accept deterministic
    // corrections here; force-only solvers leave it null.
    float3 *position_corrections{};
    std::uint32_t count{};
    float radius{};
    float inverse_mass{};
    // Optional linear RGBA color owned by the solver. Contact and constraint
    // kernels may update it during the prepared substep.
    float4 *colors{};
};

// Source-compatible name retained from the force-only view.
using PointCouplingView = PointStateView;

[[nodiscard]] bool valid(FrameOptions options) noexcept;
[[nodiscard]] SubstepContext substep_context(FrameOptions options, std::uint32_t index) noexcept;

// Movable CUDA completion event.  Async solver calls enqueue work, record this
// token, and return without synchronizing.  wait() is the synchronous
// convenience boundary and reports deferred CUDA execution failures.
class Completion {
  public:
    Completion() noexcept = default;
    ~Completion();
    Completion(Completion &&other) noexcept;
    Completion &operator=(Completion &&other) noexcept;
    Completion(const Completion &) = delete;
    Completion &operator=(const Completion &) = delete;

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
concept FrameSolver = requires(Solver &solver, FrameOptions frame, SubstepContext substep,
                               Completion &completion, cudaStream_t stream) {
    { solver.begin_frame(frame, stream) } -> std::same_as<Status>;
    { solver.prepare_substep(substep, stream) } -> std::same_as<Status>;
    { solver.finish_substep(substep, stream) } -> std::same_as<Status>;
    { solver.finish_frame(completion, stream) } -> std::same_as<Status>;
};

template <typename Solver>
concept RecoverableFrameSolver = FrameSolver<Solver> &&
    requires(Solver &solver, cudaStream_t stream) {
        { solver.abandon_frame(stream) } -> std::same_as<Status>;
    };

// Convenience driver for uncoupled frames.  Applications that couple two or
// more solvers use the same calls explicitly and insert coupling work between
// prepare_substep() and finish_substep().
template <FrameSolver Solver>
[[nodiscard]] Status step_async(Solver &solver, FrameOptions frame, Completion &completion,
                                cudaStream_t stream = nullptr) noexcept {
    Status status = solver.begin_frame(frame, stream);
    if (!status) return status;
    for (std::uint32_t i = 0U; i < frame.substeps; ++i) {
        const SubstepContext substep = substep_context(frame, i);
        status = solver.prepare_substep(substep, stream);
        if (!status) {
            if constexpr (requires { solver.abandon_frame(stream); })
                static_cast<void>(solver.abandon_frame(stream));
            return status;
        }
        status = solver.finish_substep(substep, stream);
        if (!status) {
            if constexpr (requires { solver.abandon_frame(stream); })
                static_cast<void>(solver.abandon_frame(stream));
            return status;
        }
    }
    status = solver.finish_frame(completion, stream);
    if (!status) {
        if constexpr (requires { solver.abandon_frame(stream); })
            static_cast<void>(solver.abandon_frame(stream));
    }
    return status;
}

// Synchronous, error-safe composition driver for the existing example
// contexts and small applications. The callback enqueues pair/contact work
// after every solver has prepared a substep. A failed stage abandons all active
// protocol state so the owners remain reusable; already enqueued CUDA work is
// not rolled back.
template <typename Coupler, RecoverableFrameSolver... Solvers>
    requires std::is_nothrow_invocable_r_v<Status, Coupler &, SubstepContext, cudaStream_t>
[[nodiscard]] Status advance_coupled(FrameOptions frame, Coupler &&couple,
                                     cudaStream_t stream, Solvers &...solvers) noexcept {
    if (!valid(frame))
        return {StatusCode::invalid_argument, cudaSuccess, "invalid frame options"};
    Status status{};
    auto abandon = [&] { (static_cast<void>(solvers.abandon_frame(stream)), ...); };
    auto begin_one = [&](auto &solver) {
        if (status) status = solver.begin_frame(frame, stream);
    };
    (begin_one(solvers), ...);
    if (!status) {
        abandon();
        return status;
    }
    for (std::uint32_t index = 0U; index < frame.substeps; ++index) {
        const SubstepContext substep = substep_context(frame, index);
        auto prepare_one = [&](auto &solver) {
            if (status) status = solver.prepare_substep(substep, stream);
        };
        (prepare_one(solvers), ...);
        if (status) status = couple(substep, stream);
        auto finish_one = [&](auto &solver) {
            if (status) status = solver.finish_substep(substep, stream);
        };
        (finish_one(solvers), ...);
        if (!status) {
            abandon();
            return status;
        }
    }
    std::array<Completion, sizeof...(Solvers)> completions;
    std::size_t completion_index{};
    auto complete_one = [&](auto &solver) {
        if (status) status = solver.finish_frame(completions[completion_index++], stream);
    };
    (complete_one(solvers), ...);
    if (!status) {
        abandon();
        return status;
    }
    for (Completion &completion : completions) {
        status = completion.wait();
        if (!status) return status;
    }
    return {};
}

template <FrameSolver Solver>
[[nodiscard]] Status step(Solver &solver, FrameOptions frame,
                          cudaStream_t stream = nullptr) noexcept {
    Completion completion;
    Status status = step_async(solver, frame, completion, stream);
    return status ? completion.wait() : status;
}

} // namespace parallel_mater::physics
