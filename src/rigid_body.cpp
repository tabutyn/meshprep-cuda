// SPDX-License-Identifier: MIT
#include <parallel_mater/physics.hpp>

#include <algorithm>
#include <cmath>
#include <memory>
#include <new>
#include <utility>

namespace parallel_mater::physics {
namespace {

[[nodiscard]] constexpr Status invalid(const char *message) noexcept {
    return {StatusCode::invalid_argument, cudaSuccess, message};
}

[[nodiscard]] bool finite(float value) noexcept { return std::isfinite(value); }
[[nodiscard]] bool finite(float3 value) noexcept {
    return finite(value.x) && finite(value.y) && finite(value.z);
}
[[nodiscard]] bool finite(float4 value) noexcept {
    return finite(value.x) && finite(value.y) && finite(value.z) && finite(value.w);
}
[[nodiscard]] float3 add(float3 a, float3 b) noexcept {
    return float3{a.x + b.x, a.y + b.y, a.z + b.z};
}
[[nodiscard]] float3 subtract(float3 a, float3 b) noexcept {
    return float3{a.x - b.x, a.y - b.y, a.z - b.z};
}
[[nodiscard]] float3 multiply(float3 value, float scale) noexcept {
    return float3{value.x * scale, value.y * scale, value.z * scale};
}
[[nodiscard]] float3 cross(float3 a, float3 b) noexcept {
    return float3{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
[[nodiscard]] float length(float3 value) noexcept {
    return std::sqrt(value.x * value.x + value.y * value.y + value.z * value.z);
}
[[nodiscard]] float3 clamp_length(float3 value, float maximum) noexcept {
    const float magnitude = length(value);
    return magnitude > maximum && magnitude > 0.0F ? multiply(value, maximum / magnitude) : value;
}
[[nodiscard]] float4 quaternion_product(float4 a, float4 b) noexcept {
    return float4{a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
                  a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
                  a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
                  a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z};
}
[[nodiscard]] float4 normalize(float4 value) noexcept {
    const float magnitude =
        std::sqrt(value.x * value.x + value.y * value.y + value.z * value.z + value.w * value.w);
    return magnitude > 1.0e-8F ? float4{value.x / magnitude, value.y / magnitude,
                                        value.z / magnitude, value.w / magnitude}
                               : float4{0.0F, 0.0F, 0.0F, 1.0F};
}
[[nodiscard]] bool valid(RigidBodyOptions options) noexcept {
    const bool shape_valid = options.shape == RigidShape::sphere
                                 ? finite(options.radius) && options.radius > 0.0F
                                 : finite(options.half_extents) && options.half_extents.x > 0.0F &&
                                       options.half_extents.y > 0.0F &&
                                       options.half_extents.z > 0.0F;
    return shape_valid && finite(options.mass) && options.mass > 0.0F && finite(options.inertia) &&
           options.inertia.x > 0.0F && options.inertia.y > 0.0F && options.inertia.z > 0.0F &&
           finite(options.linear_damping) && options.linear_damping >= 0.0F &&
           finite(options.angular_damping) && options.angular_damping >= 0.0F &&
           finite(options.maximum_linear_speed) && options.maximum_linear_speed > 0.0F &&
           finite(options.maximum_angular_speed) && options.maximum_angular_speed > 0.0F;
}

} // namespace

struct RigidBody::Impl {
    RigidBodyOptions options{};
    RigidBodyState state{};
    RigidBodyState initial_state{};
    FrameOptions frame{};
    float3 force{};
    float3 torque{};
    std::uint32_t next_substep{};
    bool frame_active{};
    bool substep_prepared{};
};

RigidBody::RigidBody() noexcept = default;
RigidBody::~RigidBody() = default;
RigidBody::RigidBody(RigidBody &&) noexcept = default;
RigidBody &RigidBody::operator=(RigidBody &&) noexcept = default;

Status RigidBody::create(RigidBodyState state, RigidBodyOptions options,
                         RigidBody &output) noexcept {
    return output.initialize(state, options);
}

Status RigidBody::initialize(RigidBodyState state, RigidBodyOptions options) noexcept {
    if (!valid(options) || !finite(state.position) || !finite(state.orientation) ||
        !finite(state.linear_velocity) || !finite(state.angular_velocity))
        return invalid("invalid rigid-body state");
    auto replacement = std::unique_ptr<Impl>(new (std::nothrow) Impl);
    if (!replacement) {
        return {StatusCode::allocation_failure, cudaErrorMemoryAllocation,
                "could not allocate rigid body"};
    }
    state.orientation = normalize(state.orientation);
    replacement->options = options;
    replacement->state = state;
    replacement->initial_state = state;
    impl_ = std::move(replacement);
    return {};
}

Status RigidBody::begin_frame(FrameOptions frame, cudaStream_t) noexcept {
    if (!impl_) return invalid("rigid body is not initialized");
    if (!parallel_mater::physics::valid(frame)) return invalid("invalid frame options");
    if (impl_->frame_active) return invalid("rigid-body frame is already active");
    impl_->frame = frame;
    impl_->next_substep = 0U;
    impl_->frame_active = true;
    return {};
}

Status RigidBody::prepare_substep(SubstepContext value, cudaStream_t) noexcept {
    if (!impl_ || !impl_->frame_active) return invalid("rigid-body frame is not active");
    if (impl_->substep_prepared || value.index != impl_->next_substep ||
        value.count != impl_->frame.substeps || value.timestep <= 0.0F || !finite(value.timestep) ||
        !finite(value.acceleration)) {
        return invalid("invalid rigid-body substep");
    }
    impl_->force = {};
    impl_->torque = {};
    impl_->substep_prepared = true;
    return {};
}

Status RigidBody::apply_force(float3 force, float3 world_point) noexcept {
    if (!impl_ || !impl_->substep_prepared || !finite(force) || !finite(world_point))
        return invalid("rigid-body force phase is not active");
    impl_->force = add(impl_->force, force);
    impl_->torque = add(impl_->torque, cross(subtract(world_point, impl_->state.position), force));
    return {};
}

Status RigidBody::apply_torque(float3 torque) noexcept {
    if (!impl_ || !impl_->substep_prepared || !finite(torque)) {
        return invalid("rigid-body force phase is not active");
    }
    impl_->torque = add(impl_->torque, torque);
    return {};
}

Status RigidBody::finish_substep(SubstepContext value, cudaStream_t) noexcept {
    if (!impl_ || !impl_->frame_active || !impl_->substep_prepared ||
        value.index != impl_->next_substep || value.count != impl_->frame.substeps) {
        return invalid("rigid-body substep is out of order");
    }
    const float dt = value.timestep;
    const float inverse_mass = 1.0F / impl_->options.mass;
    impl_->state.linear_velocity =
        add(impl_->state.linear_velocity,
            multiply(add(value.acceleration, multiply(impl_->force, inverse_mass)), dt));
    impl_->state.linear_velocity =
        multiply(impl_->state.linear_velocity, std::exp(-impl_->options.linear_damping * dt));
    impl_->state.linear_velocity =
        clamp_length(impl_->state.linear_velocity, impl_->options.maximum_linear_speed);
    impl_->state.position = add(impl_->state.position, multiply(impl_->state.linear_velocity, dt));

    const float3 angular_acceleration{impl_->torque.x / impl_->options.inertia.x,
                                      impl_->torque.y / impl_->options.inertia.y,
                                      impl_->torque.z / impl_->options.inertia.z};
    impl_->state.angular_velocity =
        add(impl_->state.angular_velocity, multiply(angular_acceleration, dt));
    impl_->state.angular_velocity =
        multiply(impl_->state.angular_velocity, std::exp(-impl_->options.angular_damping * dt));
    impl_->state.angular_velocity =
        clamp_length(impl_->state.angular_velocity, impl_->options.maximum_angular_speed);
    const float4 spin =
        quaternion_product(float4{impl_->state.angular_velocity.x, impl_->state.angular_velocity.y,
                                  impl_->state.angular_velocity.z, 0.0F},
                           impl_->state.orientation);
    impl_->state.orientation = normalize(float4{impl_->state.orientation.x + 0.5F * dt * spin.x,
                                                impl_->state.orientation.y + 0.5F * dt * spin.y,
                                                impl_->state.orientation.z + 0.5F * dt * spin.z,
                                                impl_->state.orientation.w + 0.5F * dt * spin.w});
    if (!finite(impl_->state.position) || !finite(impl_->state.orientation) ||
        !finite(impl_->state.linear_velocity) || !finite(impl_->state.angular_velocity)) {
        return {StatusCode::internal_error, cudaSuccess,
                "rigid-body integration produced non-finite state"};
    }
    impl_->substep_prepared = false;
    ++impl_->next_substep;
    return {};
}

Status RigidBody::finish_frame(Completion &completion, cudaStream_t stream) noexcept {
    if (!impl_ || !impl_->frame_active || impl_->substep_prepared ||
        impl_->next_substep != impl_->frame.substeps) {
        return invalid("rigid-body frame has incomplete substeps");
    }
    if (completion.pending()) return invalid("completion token is already pending");
    Status status = completion.record(stream);
    if (status) impl_->frame_active = false;
    return status;
}

Status RigidBody::abandon_frame(cudaStream_t stream) noexcept {
    if (!impl_) return invalid("rigid body is not initialized");
    const cudaError_t error = cudaStreamSynchronize(stream);
    impl_->frame_active = false;
    impl_->substep_prepared = false;
    impl_->next_substep = 0U;
    impl_->force = {};
    impl_->torque = {};
    return error == cudaSuccess
               ? Status{}
               : Status{error == cudaErrorMemoryAllocation ? StatusCode::allocation_failure
                                                           : StatusCode::cuda_failure,
                        error, "could not drain abandoned rigid-body frame"};
}

Status RigidBody::advance_async(FrameOptions frame, Completion &completion,
                                cudaStream_t stream) noexcept {
    return parallel_mater::physics::step_async(*this, frame, completion, stream);
}
Status RigidBody::advance(FrameOptions frame, cudaStream_t stream) noexcept {
    return parallel_mater::physics::step(*this, frame, stream);
}
Status RigidBody::reset() noexcept {
    if (!impl_) return invalid("rigid body is not initialized");
    if (impl_->frame_active) return invalid("cannot reset an active rigid-body frame");
    impl_->state = impl_->initial_state;
    return {};
}
bool RigidBody::initialized() const noexcept { return impl_ != nullptr; }
RigidBodyOptions RigidBody::options() const noexcept {
    return impl_ ? impl_->options : RigidBodyOptions{};
}
RigidBodyState RigidBody::state() const noexcept { return impl_ ? impl_->state : RigidBodyState{}; }

static_assert(FrameSolver<RigidBody>);

} // namespace parallel_mater::physics
