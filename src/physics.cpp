// SPDX-License-Identifier: MIT
#include <meshprep/physics.hpp>

#include "soft_body.hpp"
#include "status_exception.hpp"

#include <cmath>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>

namespace meshprep::physics {
namespace {

constexpr Status invalid(const char* message) noexcept
{
    return {StatusCode::invalid_argument, cudaSuccess, message};
}

constexpr Status internal(const char* message) noexcept
{
    return {StatusCode::internal_error, cudaSuccess, message};
}

constexpr Status allocation_failure(const char* message) noexcept
{
    return {StatusCode::allocation_failure, cudaErrorMemoryAllocation, message};
}

[[nodiscard]] bool finite(float value) noexcept
{
    return std::isfinite(value);
}

[[nodiscard]] bool finite(float3 value) noexcept
{
    return finite(value.x) && finite(value.y) && finite(value.z);
}

[[nodiscard]] bool valid(const SoftBodyOptions& value) noexcept
{
    if (value.instance_count == 0U ||
        value.instance_count > SoftBodyOptions::maximum_instances ||
        value.substeps == 0U || value.substeps > 32U ||
        value.constraint_iterations == 0U ||
        value.constraint_iterations > 256U ||
        !finite(value.timestep) || value.timestep <= 0.0F ||
        !finite(value.node_mass) || value.node_mass <= 0.0F ||
        !finite(value.spring_stiffness) || value.spring_stiffness < 100.0F ||
        value.spring_stiffness > 160'000.0F ||
        !finite(value.spring_damping_ratio) || value.spring_damping_ratio < 0.0F ||
        value.spring_damping_ratio > 4.0F ||
        !finite(value.velocity_damping) || value.velocity_damping < 0.0F ||
        value.velocity_damping > 30.0F ||
        !finite(value.maximum_projection_fraction) ||
        value.maximum_projection_fraction <= 0.0F ||
        value.maximum_projection_fraction > 1.0F ||
        !finite(value.constraint_velocity_response) ||
        value.constraint_velocity_response < 0.0F ||
        value.constraint_velocity_response > 1.0F ||
        !finite(value.break_strain) || value.break_strain <= 0.0F ||
        value.fracture_persistence_substeps == 0U ||
        value.fracture_persistence_substeps > 64U ||
        !finite(value.maximum_speed) || value.maximum_speed < 0.5F ||
        value.maximum_speed > 30.0F ||
        !finite(value.strength_multiplier) ||
        value.strength_multiplier < 0.0625F ||
        value.strength_multiplier > 64.0F ||
        !finite(value.ground_friction) || value.ground_friction < 0.0F ||
        value.ground_friction > 50.0F || value.hierarchy_leaf_size == 0U ||
        value.hierarchy_leaf_size > 32U) {
        return false;
    }
    for (std::uint32_t i = 0U; i < value.instance_count; ++i) {
        if (!finite(value.instance_origins[i])) return false;
    }
    return true;
}

[[nodiscard]] waterlab::SoftBodyOptions translated(SoftBodyOptions value)
{
    waterlab::SoftBodyOptions result;
    result.instance_count = value.instance_count;
    result.solver_substeps = value.substeps;
    result.spring_solver_iterations = value.constraint_iterations;
    result.fixed_dt = value.timestep;
    result.voxel_mass = value.node_mass;
    result.spring_stiffness = value.spring_stiffness;
    result.spring_damping_ratio = value.spring_damping_ratio;
    result.velocity_damping = value.velocity_damping;
    result.maximum_projection_fraction = value.maximum_projection_fraction;
    result.constraint_velocity_response = value.constraint_velocity_response;
    result.break_strain = value.break_strain;
    result.fracture_persistence_substeps = value.fracture_persistence_substeps;
    result.maximum_speed = value.maximum_speed;
    result.strength_multiplier = value.strength_multiplier;
    result.ground_friction = value.ground_friction;
    result.hierarchy_leaf_size = value.hierarchy_leaf_size;
    result.use_course_layout = false;
    result.require_1000_voxels = false;
    result.course_board_collisions = false;
    result.arena = waterlab::GalleryArena::none;
    result.unbonded_voxel_collisions = value.unbonded_node_collisions;
    result.preserve_fractured_triangle_shape =
        value.preserve_fractured_triangle_shape;
    result.render_internal_members = value.render_internal_members;
    result.instance_origins = value.instance_origins;
    return result;
}

template <typename Function>
Status invoke(Function&& function, const char* failure) noexcept
{
    try {
        std::forward<Function>(function)();
        return {};
    } catch (const waterlab::detail::StatusException& error) {
        Status status = error.status();
        status.message = failure;
        return status;
    } catch (const std::bad_alloc&) {
        return allocation_failure(failure);
    } catch (const std::invalid_argument&) {
        return invalid(failure);
    } catch (const std::exception&) {
        return internal(failure);
    }
}

} // namespace

struct SoftBody::Impl {
    explicit Impl(std::string path, SoftBodyOptions selected)
        : options(std::move(selected)), solver(path, translated(options)) {}

    SoftBodyOptions options{};
    waterlab::SoftBodyCourse solver;
};

SoftBody::SoftBody() noexcept = default;
SoftBody::~SoftBody() = default;
SoftBody::SoftBody(SoftBody&&) noexcept = default;
SoftBody& SoftBody::operator=(SoftBody&&) noexcept = default;

Status SoftBody::create(
    std::string_view asset_path, SoftBodyOptions options, SoftBody& output,
    cudaStream_t stream) noexcept
{
    return output.initialize(asset_path, options, stream);
}

Status SoftBody::initialize(
    std::string_view asset_path, SoftBodyOptions options, cudaStream_t stream) noexcept
{
    if (asset_path.empty()) return invalid("soft-body asset path is empty");
    if (!valid(options)) return invalid("invalid soft-body options");
    return invoke([&] {
        auto replacement = std::make_unique<Impl>(std::string(asset_path), options);
        const cudaError_t status = cudaStreamSynchronize(stream);
        if (status != cudaSuccess) {
            throw waterlab::detail::StatusException(
                {status == cudaErrorMemoryAllocation
                        ? StatusCode::allocation_failure : StatusCode::cuda_failure,
                    status, cudaGetErrorString(status)},
                "initialize public soft body");
        }
        impl_ = std::move(replacement);
    }, "could not initialize soft body");
}

Status SoftBody::step(float3 gravity, cudaStream_t stream) noexcept
{
    SoftBodyTimings ignored{};
    return step(gravity, ignored, stream);
}

Status SoftBody::step(
    float3 gravity, SoftBodyTimings& timings, cudaStream_t stream) noexcept
{
    if (!impl_) return invalid("soft body is not initialized");
    if (!finite(gravity)) return invalid("gravity must be finite");
    return invoke([&] {
        const auto measured = impl_->solver.step(gravity, stream);
        timings = {measured.physics_ms, measured.render_deformation_ms,
            measured.render_hierarchy_ms};
    }, "soft-body step failed");
}

Status SoftBody::begin_frame(cudaStream_t stream) noexcept
{
    if (!impl_) return invalid("soft body is not initialized");
    return invoke([&] { impl_->solver.begin_frame(stream); },
        "soft-body begin_frame failed");
}

Status SoftBody::prepare_substep(float dt, float3 gravity, cudaStream_t stream) noexcept
{
    if (!impl_) return invalid("soft body is not initialized");
    return invoke([&] { impl_->solver.prepare_substep(dt, gravity, stream); },
        "soft-body prepare_substep failed");
}

Status SoftBody::finish_substep(float dt, float3 gravity, cudaStream_t stream) noexcept
{
    if (!impl_) return invalid("soft body is not initialized");
    return invoke([&] { impl_->solver.finish_substep(dt, gravity, stream); },
        "soft-body finish_substep failed");
}

Status SoftBody::finish_frame(SoftBodyTimings& timings, cudaStream_t stream) noexcept
{
    if (!impl_) return invalid("soft body is not initialized");
    return invoke([&] {
        const auto measured = impl_->solver.finish_frame(stream);
        timings = {measured.physics_ms, measured.render_deformation_ms,
            measured.render_hierarchy_ms};
    }, "soft-body finish_frame failed");
}

Status SoftBody::reset(cudaStream_t stream) noexcept
{
    if (!impl_) return invalid("soft body is not initialized");
    return invoke([&] { impl_->solver.reset(stream); }, "soft-body reset failed");
}

Status SoftBody::set_material(SoftBodyMaterial material) noexcept
{
    if (!impl_) return invalid("soft body is not initialized");
    return invoke([&] {
        impl_->solver.set_material({material.spring_stiffness,
            material.spring_damping_ratio, material.velocity_damping,
            material.maximum_speed, material.ground_friction});
        impl_->options.spring_stiffness = material.spring_stiffness;
        impl_->options.spring_damping_ratio = material.spring_damping_ratio;
        impl_->options.velocity_damping = material.velocity_damping;
        impl_->options.maximum_speed = material.maximum_speed;
        impl_->options.ground_friction = material.ground_friction;
    }, "invalid soft-body material");
}

Status SoftBody::set_substeps(std::uint32_t substeps) noexcept
{
    if (!impl_) return invalid("soft body is not initialized");
    return invoke([&] {
        impl_->solver.set_solver_substeps(substeps);
        impl_->options.substeps = substeps;
    }, "invalid soft-body substep count");
}

Status SoftBody::set_constraint_iterations(std::uint32_t iterations) noexcept
{
    if (!impl_) return invalid("soft body is not initialized");
    return invoke([&] {
        impl_->solver.set_spring_solver_iterations(iterations);
        impl_->options.constraint_iterations = iterations;
    }, "invalid soft-body constraint iteration count");
}

Status SoftBody::set_strength_multiplier(float multiplier) noexcept
{
    if (!impl_) return invalid("soft body is not initialized");
    return invoke([&] {
        impl_->solver.set_strength_multiplier(multiplier);
        impl_->options.strength_multiplier = multiplier;
    }, "invalid soft-body strength multiplier");
}

bool SoftBody::initialized() const noexcept { return impl_ != nullptr; }

SoftBodyOptions SoftBody::options() const noexcept
{
    return impl_ ? impl_->options : SoftBodyOptions{};
}

SoftBodyMaterial SoftBody::material() const noexcept
{
    if (!impl_) return {};
    const auto value = impl_->solver.material();
    return {value.spring_stiffness, value.spring_damping_ratio,
        value.velocity_damping, value.maximum_speed, value.ground_friction};
}

SoftBodyNodeView SoftBody::nodes() const noexcept
{
    if (!impl_) return {};
    const auto value = impl_->solver.voxel_view();
    return {value.positions, value.velocities, value.rest_positions, value.flags,
        value.external_impulses, value.position_corrections, value.voxel_count,
        value.voxels_per_instance, value.instance_count, value.voxel_radius,
        value.inverse_voxel_mass};
}

SoftBodyBondView SoftBody::bonds() const noexcept
{
    if (!impl_) return {};
    const auto value = impl_->solver.lattice_view();
    static_assert(sizeof(Bond) == sizeof(waterlab::SoftBodyEdge));
    static_assert(alignof(Bond) == alignof(waterlab::SoftBodyEdge));
    return {value.positions, value.flags,
        reinterpret_cast<const Bond*>(value.edges), value.active_edges,
        value.voxel_count, value.voxels_per_instance, value.edges_per_instance,
        value.instance_count, value.voxel_radius};
}

SoftBodySurfaceView SoftBody::surface() const noexcept
{
    if (!impl_) return {};
    const auto value = impl_->solver.render_view();
    return {{value.positions, value.vertex_count, value.triangles,
                value.triangle_count},
        value.vertex_normals, value.corner_normal_indices, value.texcoords,
        value.triangle_active, value.nodes, value.primitive_indices,
        value.node_count, value.max_depth};
}

SoftBodyStatistics SoftBody::statistics() const noexcept
{
    if (!impl_) return {};
    const auto value = impl_->solver.statistics();
    return {value.instance_count, value.voxels_per_instance,
        value.total_voxel_count, value.surface_voxel_count,
        value.edges_per_instance, value.total_edge_count,
        value.broken_edge_count, value.finite_failure_count, value.frame_index};
}

std::size_t SoftBody::allocated_bytes() const noexcept
{
    return impl_ ? impl_->solver.allocated_bytes() : 0U;
}

} // namespace meshprep::physics
