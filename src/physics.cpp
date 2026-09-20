// SPDX-License-Identifier: MIT
#include <parallel_mater/cloth.hpp>
#include <parallel_mater/rope.hpp>

#include "internal/asset_builders.hpp"
#include "internal/fixed_topology.hpp"
#include "internal/status_exception.hpp"

#include <cmath>
#include <memory>
#include <new>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

namespace parallel_mater::physics {
namespace {

constexpr Status invalid(const char *message) noexcept {
    return {StatusCode::invalid_argument, cudaSuccess, message};
}

constexpr Status internal(const char *message) noexcept {
    return {StatusCode::internal_error, cudaSuccess, message};
}

constexpr Status allocation_failure(const char *message) noexcept {
    return {StatusCode::allocation_failure, cudaErrorMemoryAllocation, message};
}

[[nodiscard]] bool finite(float value) noexcept { return std::isfinite(value); }

[[nodiscard]] bool finite(float3 value) noexcept {
    return finite(value.x) && finite(value.y) && finite(value.z);
}

[[nodiscard]] bool valid(const SoftBodyOptions &value) noexcept {
    if (value.instance_count == 0U || value.instance_count > SoftBodyOptions::maximum_instances ||
        value.substeps == 0U || value.substeps > 32U || value.constraint_iterations == 0U ||
        value.constraint_iterations > 256U || !finite(value.timestep) || value.timestep <= 0.0F ||
        !finite(value.node_mass) || value.node_mass <= 0.0F || !finite(value.spring_stiffness) ||
        value.spring_stiffness < 100.0F || value.spring_stiffness > 160'000.0F ||
        !finite(value.velocity_damping) || value.velocity_damping < 0.0F ||
        value.velocity_damping > 30.0F || !finite(value.maximum_projection_fraction) ||
        value.maximum_projection_fraction <= 0.0F || value.maximum_projection_fraction > 1.0F ||
        !finite(value.constraint_velocity_response) || value.constraint_velocity_response < 0.0F ||
        value.constraint_velocity_response > 1.0F || !finite(value.break_strain) ||
        value.break_strain <= 0.0F || value.fracture_persistence_substeps == 0U ||
        value.fracture_persistence_substeps > 64U || !finite(value.maximum_speed) ||
        value.maximum_speed < 0.5F || value.maximum_speed > 30.0F ||
        !finite(value.strength_multiplier) || value.strength_multiplier < 0.0625F ||
        value.strength_multiplier > 64.0F || value.hierarchy_leaf_size == 0U ||
        value.hierarchy_leaf_size > 32U) {
        return false;
    }
    for (std::uint32_t i = 0U; i < value.instance_count; ++i) {
        if (!finite(value.instance_origins[i])) return false;
    }
    return true;
}

[[nodiscard]] bool valid(const SoftBodyMaterial &value) noexcept {
    return finite(value.spring_stiffness) && value.spring_stiffness >= 100.0F &&
           value.spring_stiffness <= 160'000.0F && finite(value.velocity_damping) &&
           value.velocity_damping >= 0.0F && value.velocity_damping <= 30.0F &&
           finite(value.maximum_speed) && value.maximum_speed >= 0.5F &&
           value.maximum_speed <= 30.0F;
}

template <typename Function> Status invoke(Function &&function, const char *failure) noexcept {
    try {
        std::forward<Function>(function)();
        return {};
    } catch (const detail::StatusException &error) {
        Status status = error.status();
        status.message = failure;
        return status;
    } catch (const std::bad_alloc &) {
        return allocation_failure(failure);
    } catch (const std::invalid_argument &) {
        return invalid(failure);
    } catch (const std::exception &) {
        return internal(failure);
    }
}

} // namespace

struct SoftBody::Impl {
    explicit Impl(std::string path, SoftBodyOptions selected)
        : options(std::move(selected)), solver(path, options) {}
    explicit Impl(detail::FixedTopologyAsset asset, SoftBodyOptions selected)
        : options(std::move(selected)), solver(std::move(asset), options) {}

    SoftBodyOptions options{};
    detail::FixedTopology solver;
    FrameOptions frame{};
    std::uint32_t next_substep{};
    bool frame_active{};
    SoftBodyTelemetry telemetry{};
};

SoftBody::SoftBody() noexcept = default;
SoftBody::~SoftBody() = default;
SoftBody::SoftBody(SoftBody &&) noexcept = default;
SoftBody &SoftBody::operator=(SoftBody &&) noexcept = default;

Status SoftBody::create(std::string_view asset_path, SoftBodyOptions options, SoftBody &output,
                        cudaStream_t stream) noexcept {
    return output.initialize(asset_path, options, stream);
}

Status SoftBody::create(SoftBodyAssetView asset, SoftBodyOptions options, SoftBody &output,
                        cudaStream_t stream) noexcept {
    return output.initialize(asset, options, stream);
}

Status SoftBody::initialize(std::string_view asset_path, SoftBodyOptions options,
                            cudaStream_t stream) noexcept {
    if (asset_path.empty()) return invalid("soft-body asset path is empty");
    if (!valid(options)) return invalid("invalid soft-body options");
    return invoke(
        [&] {
            auto replacement = std::make_unique<Impl>(std::string(asset_path), options);
            const cudaError_t status = cudaStreamSynchronize(stream);
            if (status != cudaSuccess) {
                throw detail::StatusException({status == cudaErrorMemoryAllocation
                                                   ? StatusCode::allocation_failure
                                                   : StatusCode::cuda_failure,
                                               status, cudaGetErrorString(status)},
                                              "initialize public soft body");
            }
            impl_ = std::move(replacement);
        },
        "could not initialize soft body");
}

Status SoftBody::initialize(SoftBodyAssetView asset, SoftBodyOptions options,
                            cudaStream_t stream) noexcept {
    if (asset.data == nullptr || asset.size == 0U) {
        return invalid("soft-body asset bytes are empty");
    }
    if (!valid(options)) return invalid("invalid soft-body options");
    return invoke(
        [&] {
            auto native = detail::load_fixed_topology_asset(
                std::span<const std::byte>(asset.data, asset.size));
            auto replacement = std::make_unique<Impl>(std::move(native), options);
            const cudaError_t status = cudaStreamSynchronize(stream);
            if (status != cudaSuccess) {
                throw detail::StatusException({status == cudaErrorMemoryAllocation
                                                   ? StatusCode::allocation_failure
                                                   : StatusCode::cuda_failure,
                                               status, cudaGetErrorString(status)},
                                              "initialize public soft body from memory");
            }
            impl_ = std::move(replacement);
        },
        "could not initialize soft body from memory");
}

Status SoftBody::step(float3 gravity, cudaStream_t stream) noexcept {
    if (!impl_) return invalid("soft body is not initialized");
    if (!finite(gravity)) return invalid("gravity must be finite");
    return advance({impl_->options.timestep, impl_->options.substeps, gravity}, stream);
}

Status SoftBody::step(float3 gravity, SoftBodyTimings &timings, cudaStream_t stream) noexcept {
    if (!impl_) return invalid("soft body is not initialized");
    if (!finite(gravity)) return invalid("gravity must be finite");
    Status status = advance({impl_->options.timestep, impl_->options.substeps, gravity}, stream);
    if (status) status = collect_telemetry(stream);
    if (status) timings = impl_->telemetry.timings;
    return status;
}

Status SoftBody::begin_frame(cudaStream_t stream) noexcept {
    if (!impl_) return invalid("soft body is not initialized");
    return invoke([&] { impl_->solver.begin_frame(stream); }, "soft-body begin_frame failed");
}

Status SoftBody::prepare_substep(float dt, float3 gravity, cudaStream_t stream) noexcept {
    if (!impl_) return invalid("soft body is not initialized");
    return invoke([&] { impl_->solver.prepare_substep(dt, gravity, stream); },
                  "soft-body prepare_substep failed");
}

Status SoftBody::finish_substep(float dt, float3 gravity, cudaStream_t stream) noexcept {
    if (!impl_) return invalid("soft body is not initialized");
    return invoke([&] { impl_->solver.finish_substep(dt, gravity, stream); },
                  "soft-body finish_substep failed");
}

Status SoftBody::finish_frame(SoftBodyTimings &timings, cudaStream_t stream) noexcept {
    if (!impl_) return invalid("soft body is not initialized");
    return invoke(
        [&] {
            timings = impl_->solver.finish_frame(stream);
            impl_->telemetry.timings = timings;
        },
        "soft-body finish_frame failed");
}

Status SoftBody::begin_frame(FrameOptions frame, cudaStream_t stream) noexcept {
    if (!impl_) return invalid("soft body is not initialized");
    if (!parallel_mater::physics::valid(frame)) return invalid("invalid frame options");
    if (impl_->frame_active) return invalid("soft-body frame is already active");
    Status status = begin_frame(stream);
    if (status) {
        impl_->frame = frame;
        impl_->next_substep = 0U;
        impl_->frame_active = true;
    }
    return status;
}

Status SoftBody::prepare_substep(SubstepContext substep, cudaStream_t stream) noexcept {
    if (!impl_ || !impl_->frame_active) {
        return invalid("soft-body frame is not active");
    }
    if (substep.index != impl_->next_substep || substep.count != impl_->frame.substeps) {
        return invalid("soft-body substep is out of order");
    }
    return prepare_substep(substep.timestep, substep.acceleration, stream);
}

Status SoftBody::finish_substep(SubstepContext substep, cudaStream_t stream) noexcept {
    if (!impl_ || !impl_->frame_active) {
        return invalid("soft-body frame is not active");
    }
    if (substep.index != impl_->next_substep || substep.count != impl_->frame.substeps) {
        return invalid("soft-body substep is out of order");
    }
    Status status = finish_substep(substep.timestep, substep.acceleration, stream);
    if (status) ++impl_->next_substep;
    return status;
}

Status SoftBody::finish_frame(Completion &completion, cudaStream_t stream) noexcept {
    if (!impl_ || !impl_->frame_active) {
        return invalid("soft-body frame is not active");
    }
    if (impl_->next_substep != impl_->frame.substeps) {
        return invalid("soft-body frame has incomplete substeps");
    }
    if (completion.pending()) return invalid("completion token is already pending");
    Status status =
        invoke([&] { impl_->solver.finish_frame_async(stream); }, "soft-body finish_frame failed");
    if (!status) return status;
    status = completion.record(stream);
    if (status) impl_->frame_active = false;
    return status;
}

Status SoftBody::collect_telemetry_async(Completion &completion, cudaStream_t stream) noexcept {
    if (!impl_) return invalid("soft body is not initialized");
    if (impl_->frame_active) return invalid("cannot collect an active soft-body frame");
    if (completion.pending()) return invalid("completion token is already pending");
    Status status = invoke([&] { impl_->solver.collect_telemetry_async(stream); },
                           "soft-body telemetry request failed");
    return status ? completion.record(stream) : status;
}

Status SoftBody::resolve_telemetry(SoftBodyTelemetry &output) noexcept {
    if (!impl_) return invalid("soft body is not initialized");
    Status status = invoke(
        [&] {
            impl_->telemetry.timings = impl_->solver.resolve_telemetry();
            impl_->telemetry.statistics = statistics();
            output = impl_->telemetry;
        },
        "soft-body telemetry is not ready");
    return status;
}

Status SoftBody::collect_telemetry(cudaStream_t stream) noexcept {
    Completion completion;
    Status status = collect_telemetry_async(completion, stream);
    if (status) status = completion.wait();
    SoftBodyTelemetry output;
    if (status) status = resolve_telemetry(output);
    return status;
}

SoftBodyTelemetry SoftBody::telemetry() const noexcept {
    return impl_ ? impl_->telemetry : SoftBodyTelemetry{};
}

Status SoftBody::advance_async(FrameOptions frame, Completion &completion,
                               cudaStream_t stream) noexcept {
    return parallel_mater::physics::step_async(*this, frame, completion, stream);
}

Status SoftBody::advance(FrameOptions frame, cudaStream_t stream) noexcept {
    return parallel_mater::physics::step(*this, frame, stream);
}

Status SoftBody::reset(cudaStream_t stream) noexcept {
    if (!impl_) return invalid("soft body is not initialized");
    return invoke([&] { impl_->solver.reset(stream); }, "soft-body reset failed");
}

Status SoftBody::set_material(SoftBodyMaterial material) noexcept {
    if (!impl_) return invalid("soft body is not initialized");
    if (!valid(material)) return invalid("invalid soft-body material");
    return invoke(
        [&] {
            impl_->solver.set_material(material);
            impl_->options.spring_stiffness = material.spring_stiffness;
            impl_->options.velocity_damping = material.velocity_damping;
            impl_->options.maximum_speed = material.maximum_speed;
        },
        "invalid soft-body material");
}

Status SoftBody::set_substeps(std::uint32_t substeps) noexcept {
    if (!impl_) return invalid("soft body is not initialized");
    if (substeps == 0U || substeps > 32U) return invalid("invalid soft-body substep count");
    return invoke(
        [&] {
            impl_->solver.set_substeps(substeps);
            impl_->options.substeps = substeps;
        },
        "invalid soft-body substep count");
}

Status SoftBody::set_constraint_iterations(std::uint32_t iterations) noexcept {
    if (!impl_) return invalid("soft body is not initialized");
    if (iterations == 0U || iterations > 256U)
        return invalid("invalid soft-body constraint iteration count");
    return invoke(
        [&] {
            impl_->solver.set_constraint_iterations(iterations);
            impl_->options.constraint_iterations = iterations;
        },
        "invalid soft-body constraint iteration count");
}

Status SoftBody::set_strength_multiplier(float multiplier) noexcept {
    if (!impl_) return invalid("soft body is not initialized");
    if (!finite(multiplier) || multiplier < 0.0625F || multiplier > 64.0F)
        return invalid("invalid soft-body strength multiplier");
    return invoke(
        [&] {
            impl_->solver.set_strength_multiplier(multiplier);
            impl_->options.strength_multiplier = multiplier;
        },
        "invalid soft-body strength multiplier");
}

Status SoftBody::set_node_mass(float mass) noexcept {
    if (!impl_) return invalid("soft body is not initialized");
    if (!finite(mass) || mass <= 0.0F) return invalid("invalid soft-body node mass");
    return invoke(
        [&] {
            impl_->solver.set_node_mass(mass);
            impl_->options.node_mass = mass;
        },
        "invalid soft-body node mass");
}

bool SoftBody::initialized() const noexcept { return impl_ != nullptr; }

SoftBodyOptions SoftBody::options() const noexcept {
    return impl_ ? impl_->options : SoftBodyOptions{};
}

SoftBodyMaterial SoftBody::material() const noexcept {
    if (!impl_) return {};
    return impl_->solver.material();
}

float SoftBody::node_mass() const noexcept { return impl_ ? impl_->options.node_mass : 0.0F; }

SoftBodyNodeView SoftBody::nodes() const noexcept {
    if (!impl_) return {};
    return impl_->solver.nodes();
}

PointCouplingView SoftBody::coupling_points() const noexcept {
    const SoftBodyNodeView view = nodes();
    return {view.positions,  view.velocities,  view.external_impulses, view.position_corrections,
            view.node_count, view.node_radius, view.inverse_node_mass};
}

SoftBodyBondView SoftBody::bonds() const noexcept {
    if (!impl_) return {};
    return impl_->solver.bonds();
}

SoftBodySurfaceView SoftBody::surface() const noexcept {
    if (!impl_) return {};
    return impl_->solver.surface();
}

SoftBodyStatistics SoftBody::statistics() const noexcept {
    if (!impl_) return {};
    return impl_->solver.statistics();
}

std::size_t SoftBody::allocated_bytes() const noexcept {
    return impl_ ? impl_->solver.allocated_bytes() : 0U;
}

struct Cloth::Impl {
    ClothOptions options{};
    SoftBody body{};
};

Cloth::Cloth() noexcept = default;
Cloth::~Cloth() = default;
Cloth::Cloth(Cloth &&) noexcept = default;
Cloth &Cloth::operator=(Cloth &&) noexcept = default;

Status Cloth::create(ClothOptions options, Cloth &output, cudaStream_t stream) noexcept {
    return output.initialize(options, stream);
}

Status Cloth::initialize(ClothOptions options, cudaStream_t stream) noexcept {
    if (options.columns < 2U || options.columns > 2048U || options.rows < 2U ||
        options.rows > 2048U || !finite(options.spacing) || options.spacing <= 0.0F ||
        !finite(options.top_center) || options.solver.instance_count != 1U ||
        !valid(options.solver))
        return invalid("invalid cloth options");
    return invoke(
        [&] {
            auto asset = detail::make_cloth_asset(options.columns, options.rows, options.spacing,
                                                  options.top_center, options.shear_springs,
                                                  options.bend_springs);
            auto replacement = std::make_unique<Impl>();
            replacement->options = options;
            replacement->body.impl_ =
                std::make_unique<SoftBody::Impl>(std::move(asset), options.solver);
            const cudaError_t status = cudaStreamSynchronize(stream);
            if (status != cudaSuccess) {
                throw detail::StatusException({status == cudaErrorMemoryAllocation
                                                   ? StatusCode::allocation_failure
                                                   : StatusCode::cuda_failure,
                                               status, cudaGetErrorString(status)},
                                              "initialize cloth");
            }
            impl_ = std::move(replacement);
        },
        "could not initialize cloth");
}

Status Cloth::begin_frame(FrameOptions frame, cudaStream_t stream) noexcept {
    return impl_ ? impl_->body.begin_frame(frame, stream) : invalid("cloth is not initialized");
}
Status Cloth::prepare_substep(SubstepContext value, cudaStream_t stream) noexcept {
    return impl_ ? impl_->body.prepare_substep(value, stream) : invalid("cloth is not initialized");
}
Status Cloth::finish_substep(SubstepContext value, cudaStream_t stream) noexcept {
    return impl_ ? impl_->body.finish_substep(value, stream) : invalid("cloth is not initialized");
}
Status Cloth::finish_frame(Completion &completion, cudaStream_t stream) noexcept {
    return impl_ ? impl_->body.finish_frame(completion, stream)
                 : invalid("cloth is not initialized");
}
Status Cloth::advance_async(FrameOptions frame, Completion &completion,
                            cudaStream_t stream) noexcept {
    return parallel_mater::physics::step_async(*this, frame, completion, stream);
}
Status Cloth::advance(FrameOptions frame, cudaStream_t stream) noexcept {
    return parallel_mater::physics::step(*this, frame, stream);
}
Status Cloth::collect_statistics_async(Completion &completion, cudaStream_t stream) noexcept {
    return impl_ ? impl_->body.collect_telemetry_async(completion, stream)
                 : invalid("cloth is not initialized");
}
Status Cloth::collect_statistics(cudaStream_t stream) noexcept {
    return impl_ ? impl_->body.collect_telemetry(stream) : invalid("cloth is not initialized");
}
Status Cloth::reset(cudaStream_t stream) noexcept {
    return impl_ ? impl_->body.reset(stream) : invalid("cloth is not initialized");
}
bool Cloth::initialized() const noexcept { return impl_ != nullptr && impl_->body.initialized(); }
ClothOptions Cloth::options() const noexcept { return impl_ ? impl_->options : ClothOptions{}; }
SoftBodyNodeView Cloth::nodes() const noexcept {
    return impl_ ? impl_->body.nodes() : SoftBodyNodeView{};
}
PointCouplingView Cloth::coupling_points() const noexcept {
    return impl_ ? impl_->body.coupling_points() : PointCouplingView{};
}
SoftBodyBondView Cloth::bonds() const noexcept {
    return impl_ ? impl_->body.bonds() : SoftBodyBondView{};
}
SoftBodySurfaceView Cloth::surface() const noexcept {
    return impl_ ? impl_->body.surface() : SoftBodySurfaceView{};
}
SoftBodyStatistics Cloth::statistics() const noexcept {
    return impl_ ? impl_->body.statistics() : SoftBodyStatistics{};
}

struct Rope::Impl {
    RopeOptions options{};
    SoftBody body{};
};

Rope::Rope() noexcept = default;
Rope::~Rope() = default;
Rope::Rope(Rope &&) noexcept = default;
Rope &Rope::operator=(Rope &&) noexcept = default;

Status Rope::create(RopeOptions options, Rope &output, cudaStream_t stream) noexcept {
    return output.initialize(options, stream);
}

Status Rope::initialize(RopeOptions options, cudaStream_t stream) noexcept {
    const float direction_length = std::sqrt(options.direction.x * options.direction.x +
                                             options.direction.y * options.direction.y +
                                             options.direction.z * options.direction.z);
    if (options.node_count < 8U || options.node_count > 512U || !finite(options.spacing) ||
        options.spacing <= 0.0F || !finite(options.origin) || !finite(options.direction) ||
        !(direction_length > 1.0e-6F) || options.solver.instance_count != 1U ||
        !valid(options.solver))
        return invalid("invalid rope options");
    return invoke(
        [&] {
            auto asset = detail::make_rope_asset(options.node_count, options.spacing,
                                                 options.origin, options.direction);
            auto replacement = std::make_unique<Impl>();
            replacement->options = options;
            replacement->body.impl_ =
                std::make_unique<SoftBody::Impl>(std::move(asset), options.solver);
            const cudaError_t status = cudaStreamSynchronize(stream);
            if (status != cudaSuccess) {
                throw detail::StatusException({status == cudaErrorMemoryAllocation
                                                   ? StatusCode::allocation_failure
                                                   : StatusCode::cuda_failure,
                                               status, cudaGetErrorString(status)},
                                              "initialize rope");
            }
            impl_ = std::move(replacement);
        },
        "could not initialize rope");
}

Status Rope::begin_frame(FrameOptions frame, cudaStream_t stream) noexcept {
    return impl_ ? impl_->body.begin_frame(frame, stream) : invalid("rope is not initialized");
}
Status Rope::prepare_substep(SubstepContext value, cudaStream_t stream) noexcept {
    return impl_ ? impl_->body.prepare_substep(value, stream) : invalid("rope is not initialized");
}
Status Rope::finish_substep(SubstepContext value, cudaStream_t stream) noexcept {
    return impl_ ? impl_->body.finish_substep(value, stream) : invalid("rope is not initialized");
}
Status Rope::finish_frame(Completion &completion, cudaStream_t stream) noexcept {
    return impl_ ? impl_->body.finish_frame(completion, stream)
                 : invalid("rope is not initialized");
}
Status Rope::advance_async(FrameOptions frame, Completion &completion,
                           cudaStream_t stream) noexcept {
    return parallel_mater::physics::step_async(*this, frame, completion, stream);
}
Status Rope::advance(FrameOptions frame, cudaStream_t stream) noexcept {
    return parallel_mater::physics::step(*this, frame, stream);
}
Status Rope::collect_statistics_async(Completion &completion, cudaStream_t stream) noexcept {
    return impl_ ? impl_->body.collect_telemetry_async(completion, stream)
                 : invalid("rope is not initialized");
}
Status Rope::collect_statistics(cudaStream_t stream) noexcept {
    return impl_ ? impl_->body.collect_telemetry(stream) : invalid("rope is not initialized");
}
Status Rope::reset(cudaStream_t stream) noexcept {
    return impl_ ? impl_->body.reset(stream) : invalid("rope is not initialized");
}
bool Rope::initialized() const noexcept { return impl_ != nullptr && impl_->body.initialized(); }
RopeOptions Rope::options() const noexcept { return impl_ ? impl_->options : RopeOptions{}; }
SoftBodyNodeView Rope::nodes() const noexcept {
    return impl_ ? impl_->body.nodes() : SoftBodyNodeView{};
}
PointCouplingView Rope::coupling_points() const noexcept {
    return impl_ ? impl_->body.coupling_points() : PointCouplingView{};
}
SoftBodyBondView Rope::bonds() const noexcept {
    return impl_ ? impl_->body.bonds() : SoftBodyBondView{};
}
SoftBodySurfaceView Rope::surface() const noexcept {
    return impl_ ? impl_->body.surface() : SoftBodySurfaceView{};
}
SoftBodyStatistics Rope::statistics() const noexcept {
    return impl_ ? impl_->body.statistics() : SoftBodyStatistics{};
}

static_assert(FrameSolver<SoftBody>);
static_assert(FrameSolver<Cloth>);
static_assert(FrameSolver<Rope>);

} // namespace parallel_mater::physics
