// SPDX-License-Identifier: MIT
#include "gallery.hpp"

#include <parallel_mater/cloth.hpp>
#include <parallel_mater/coupling.hpp>
#include <parallel_mater/fluid.hpp>
#include <parallel_mater/rigid_body.hpp>
#include <parallel_mater/rope.hpp>
#include <parallel_mater/smoke.hpp>
#include <parallel_mater/soft_body.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <numeric>
#include <span>
#include <string>
#include <utility>
#include <vector>

#ifndef PARALLEL_MATER_GALLERY_ASSET_PATH
#define PARALLEL_MATER_GALLERY_ASSET_PATH ""
#endif

namespace parallel_mater::examples {
namespace {

[[nodiscard]] constexpr Status invalid(const char *message) noexcept {
    return {StatusCode::invalid_argument, cudaSuccess, message};
}
[[nodiscard]] constexpr Status allocation_failure(const char *message) noexcept {
    return {StatusCode::allocation_failure, cudaErrorMemoryAllocation, message};
}
[[nodiscard]] constexpr Status cuda_failure(cudaError_t error, const char *message) noexcept {
    return error == cudaSuccess
               ? Status{}
               : Status{error == cudaErrorMemoryAllocation ? StatusCode::allocation_failure
                                                           : StatusCode::cuda_failure,
                        error, message};
}

[[nodiscard]] bool finite(float3 value) noexcept {
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

[[nodiscard]] const SimulationRecipeInfo *recipe_info(SimulationRecipe recipe) noexcept {
    for (const auto &candidate : simulation_recipes)
        if (candidate.recipe == recipe) return &candidate;
    return nullptr;
}

[[nodiscard]] bool valid_options(const GallerySimulationOptions &options) noexcept {
    return valid(options.fixed_step) && recipe_info(options.recipe) != nullptr &&
           (!options.solver_iterations_override || (*options.solver_iterations_override >= 1U &&
                                                    *options.solver_iterations_override <= 32U)) &&
           (!options.gravity_override || finite(*options.gravity_override)) &&
           (!options.particle_count_override || (*options.particle_count_override >= 8U &&
                                                 *options.particle_count_override <= 100'000U)) &&
           (!options.physical_skin_frequency_override ||
            (*options.physical_skin_frequency_override >= 2U &&
             *options.physical_skin_frequency_override <= 45U)) &&
           (!options.rope_node_count_override || (*options.rope_node_count_override >= 8U &&
                                                  *options.rope_node_count_override <= 512U)) &&
           (!options.cloth_detail_override ||
            (*options.cloth_detail_override >= 1U && *options.cloth_detail_override <= 8U)) &&
           (!options.bridge_columns_override ||
            (*options.bridge_columns_override >= 2U && *options.bridge_columns_override <= 16U)) &&
           (!options.bridge_rows_override ||
            (*options.bridge_rows_override >= 2U && *options.bridge_rows_override <= 64U)) &&
           (!options.cylinder_columns_override || (*options.cylinder_columns_override >= 1U &&
                                                   *options.cylinder_columns_override <= 16U)) &&
           (!options.cylinder_rows_override ||
            (*options.cylinder_rows_override >= 1U && *options.cylinder_rows_override <= 16U));
}

[[nodiscard]] float3 default_gravity(SimulationRecipe recipe) noexcept {
    switch (recipe) {
    case SimulationRecipe::smoke:
    case SimulationRecipe::fluid_smoke:
    case SimulationRecipe::cloth_smoke:
    case SimulationRecipe::soft_body_smoke:
    case SimulationRecipe::rope_smoke:
        return {0.0F, -2.0F, 0.0F};
    default:
        return {0.0F, -9.81F, 0.0F};
    }
}

[[nodiscard]] std::uint32_t default_particles(SimulationRecipe recipe) noexcept {
    switch (recipe) {
    case SimulationRecipe::water:
        return 20'000U;
    case SimulationRecipe::water_rope:
        return 40'000U;
    case SimulationRecipe::fluid_smoke:
        return 8'000U;
    case SimulationRecipe::water_cloth:
    case SimulationRecipe::water_soft_body:
        return 10'000U;
    default:
        return 0U;
    }
}

[[nodiscard]] std::vector<physics::FluidParticle> make_fluid(std::uint32_t count, float radius) {
    std::vector<physics::FluidParticle> result;
    result.reserve(count);
    const std::uint32_t side =
        static_cast<std::uint32_t>(std::ceil(std::cbrt(static_cast<double>(count))));
    const float spacing = 2.15F * radius;
    const float half = 0.5F * spacing * static_cast<float>(side - 1U);
    for (std::uint32_t index = 0U; index < count; ++index) {
        const std::uint32_t x = index % side;
        const std::uint32_t y = (index / side) % side;
        const std::uint32_t z = index / (side * side);
        result.push_back(
            {{spacing * static_cast<float>(x) - half, 0.8F + spacing * static_cast<float>(y),
              spacing * static_cast<float>(z) - half},
             {}});
    }
    return result;
}

[[nodiscard]] physics::SmokeOptions smoke_options(SimulationRecipe recipe,
                                                  float timestep) noexcept {
    physics::SmokeOptions options;
    options.timestep = timestep;
    options.particle_count = 6'000U;
    options.capacity = 12'000U;
    options.emitter_center = {-2.0F, 0.2F, 0.0F};
    options.emitter_half_extents = {0.04F, 0.45F, 0.45F};
    options.initial_velocity = {2.8F, 0.1F, 0.0F};
    if (recipe == SimulationRecipe::fluid_smoke) {
        options.particle_count = 7'500U;
        options.emitter_center = {0.0F, -0.7F, 0.0F};
        options.emitter_half_extents = {0.8F, 0.04F, 0.8F};
        options.initial_velocity = {0.0F, 0.5F, 0.0F};
        options.buoyancy = 1.8F;
    }
    return options;
}

template <typename T> class DeviceBuffer {
  public:
    DeviceBuffer() noexcept = default;
    ~DeviceBuffer() { cudaFree(data_); }
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;
    [[nodiscard]] Status upload(std::span<const T> values, cudaStream_t stream) {
        if (values.empty()) return {};
        if (values.size() > capacity_) {
            cudaFree(data_);
            data_ = nullptr;
            count_ = capacity_ = 0U;
            cudaError_t error = cudaMalloc(reinterpret_cast<void **>(&data_), values.size_bytes());
            if (error != cudaSuccess) return cuda_failure(error, "allocate gallery buffer");
            capacity_ = values.size();
        }
        cudaError_t error = cudaMemcpyAsync(data_, values.data(), values.size_bytes(),
                                            cudaMemcpyHostToDevice, stream);
        if (error != cudaSuccess) return cuda_failure(error, "upload gallery buffer");
        count_ = values.size();
        return {};
    }
    [[nodiscard]] T *get() const noexcept { return data_; }
    [[nodiscard]] std::size_t size() const noexcept { return count_; }

  private:
    T *data_{};
    std::size_t count_{};
    std::size_t capacity_{};
};

constexpr std::array<float3, 6> sphere_vertices{
    {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}}};
constexpr std::array<uint3, 8> sphere_triangles{
    {{2, 0, 4}, {2, 4, 1}, {2, 1, 5}, {2, 5, 0}, {3, 4, 0}, {3, 1, 4}, {3, 5, 1}, {3, 0, 5}}};

} // namespace

struct GallerySimulation::Impl {
    [[nodiscard]] Status initialize(GallerySimulationOptions requested,
                                    cudaStream_t stream) noexcept {
        if (!valid_options(requested)) return invalid("invalid gallery options");
        options = requested;
        info = recipe_info(options.recipe);
        asset_path = options.soft_body_asset_path;
        if (asset_path.empty()) asset_path = PARALLEL_MATER_GALLERY_ASSET_PATH;
        options.soft_body_asset_path = asset_path;
        resolved = {{options.fixed_step.timestep},
                    options.solver_iterations_override.value_or(4U),
                    options.gravity_override.value_or(default_gravity(options.recipe)),
                    false};
        frame = {resolved.fixed_step.timestep, resolved.solver_iterations, resolved.gravity};

        Status status;
        if (has_component(info->components, Component::fluid_particles)) {
            status = initialize_fluid(
                options.particle_count_override.value_or(default_particles(options.recipe)),
                stream);
            if (!status) return status;
        }
        if (has_component(info->components, Component::cloth)) {
            cloth = std::make_unique<physics::Cloth>();
            physics::ClothOptions selected;
            const std::uint32_t detail = options.cloth_detail_override.value_or(1U);
            selected.columns = 12U * detail;
            selected.rows = 10U * detail;
            selected.spacing = 0.08F / static_cast<float>(detail);
            selected.top_center = {0.0F, 1.2F, 0.0F};
            selected.solver.substeps = resolved.solver_iterations;
            selected.solver.timestep = resolved.fixed_step.timestep;
            status = cloth->initialize(selected, stream);
            if (!status) return status;
        }
        if (has_component(info->components, Component::rope)) {
            rope = std::make_unique<physics::Rope>();
            physics::RopeOptions selected;
            selected.node_count = options.rope_node_count_override.value_or(64U);
            selected.spacing = 0.055F;
            selected.origin = {-1.4F, 1.4F, 0.0F};
            selected.direction = {1.0F, -0.25F, 0.0F};
            selected.solver.substeps = resolved.solver_iterations;
            selected.solver.timestep = resolved.fixed_step.timestep;
            status = rope->initialize(selected, stream);
            if (!status) return status;
        }
        if (has_component(info->components, Component::soft_body)) {
            if (asset_path.empty())
                return invalid("gallery soft-body recipe requires an asset path");
            soft_body = std::make_unique<physics::SoftBody>();
            physics::SoftBodyOptions selected;
            selected.substeps = resolved.solver_iterations;
            selected.timestep = resolved.fixed_step.timestep;
            selected.instance_origins[0] = {0.0F, 0.8F, 0.0F};
            status = soft_body->initialize(asset_path, selected, stream);
            if (!status) return status;
        }
        if (has_component(info->components, Component::rigid_bodies)) {
            rigid = std::make_unique<physics::RigidBody>();
            physics::RigidBodyState state;
            state.position = {-0.8F, 1.4F, 0.0F};
            state.linear_velocity = {0.35F, 0.0F, 0.0F};
            physics::RigidBodyOptions selected;
            selected.radius = 0.35F;
            status = rigid->initialize(state, selected);
            if (!status) return status;
            status = rigid_vertices.upload(sphere_vertices, stream);
            if (!status) return status;
            status = rigid_triangles.upload(sphere_triangles, stream);
            if (!status) return status;
        }
        if (has_component(info->components, Component::smoke)) {
            smoke = std::make_unique<physics::Smoke>();
            status = smoke->initialize(smoke_options(options.recipe, resolved.fixed_step.timestep),
                                       stream);
            if (!status) return status;
        }
        status = colliders.reserve(1U);
        if (!status) return status;
        status = constraints.reserve(1U, stream);
        if (!status) return status;
        status = constraint_records.upload(
            std::span<const physics::ConstraintRecord>(&host_constraint, 1U), stream);
        if (!status) return status;
        const cudaError_t synchronized = cudaStreamSynchronize(stream);
        if (synchronized != cudaSuccess)
            return cuda_failure(synchronized, "complete gallery initialization");
        refresh_views();
        refresh_statistics();
        return {};
    }

    [[nodiscard]] Status initialize_fluid(std::uint32_t count, cudaStream_t stream) noexcept {
        physics::FluidOptions selected;
        selected.repulsion = 20.0F;
        selected.particle_mass = 0.08F;
        auto particles = make_fluid(count, selected.particle_radius);
        auto replacement = std::make_unique<physics::Fluid>();
        Status status = replacement->initialize(particles, selected, stream);
        if (status) fluid = std::move(replacement);
        return status;
    }

    template <typename Solver>
    [[nodiscard]] Status begin(Solver *solver, cudaStream_t stream) noexcept {
        return solver ? solver->begin_frame(frame, stream) : Status{};
    }
    template <typename Solver>
    [[nodiscard]] Status prepare(Solver *solver, physics::SubstepContext substep,
                                 cudaStream_t stream) noexcept {
        return solver ? solver->prepare_substep(substep, stream) : Status{};
    }
    template <typename Solver>
    [[nodiscard]] Status finish(Solver *solver, physics::SubstepContext substep,
                                cudaStream_t stream) noexcept {
        return solver ? solver->finish_substep(substep, stream) : Status{};
    }
    template <typename Solver>
    [[nodiscard]] Status complete(Solver *solver, cudaStream_t stream) noexcept {
        if (!solver) return {};
        physics::Completion completion;
        Status status = solver->finish_frame(completion, stream);
        return status ? completion.wait() : status;
    }

    [[nodiscard]] Status apply_coupling(physics::PointCouplingView target, std::uint32_t order,
                                        cudaStream_t stream) noexcept {
        if (target.count == 0U) return {};
        host_constraint = {target.count / 2U, order, {0.0F, 0.002F, 0.0F}, {}};
        Status status = constraint_records.upload(
            std::span<const physics::ConstraintRecord>(&host_constraint, 1U), stream);
        if (!status) return status;
        return constraints.apply_async({constraint_records.get(), 1U}, target, stream);
    }

    [[nodiscard]] Status step(cudaStream_t stream) noexcept {
        Status status;
        physics::ColliderView collider_view;
        if (rigid) {
            const auto state = rigid->state();
            physics::Collider collider;
            collider.shape = physics::ColliderShape::sphere;
            collider.position = state.position;
            collider.orientation = state.orientation;
            collider.linear_velocity = state.linear_velocity;
            collider.angular_velocity = state.angular_velocity;
            collider.dimensions = {rigid->options().radius, 0.0F, 0.0F};
            status =
                colliders.update_async(std::span<const physics::Collider>(&collider, 1U), stream);
            if (!status) return status;
            collider_view = colliders.view();
        }
        if (smoke) {
            status = smoke->step({}, collider_view, stream);
            if (!status) return status;
        }
        if (!(status = begin(fluid.get(), stream)) || !(status = begin(cloth.get(), stream)) ||
            !(status = begin(rope.get(), stream)) || !(status = begin(soft_body.get(), stream)) ||
            !(status = begin(rigid.get(), stream)))
            return status;

        for (std::uint32_t index = 0U; index < frame.substeps; ++index) {
            const auto substep = physics::substep_context(frame, index);
            if (!(status = prepare(fluid.get(), substep, stream)) ||
                !(status = prepare(cloth.get(), substep, stream)) ||
                !(status = prepare(rope.get(), substep, stream)) ||
                !(status = prepare(soft_body.get(), substep, stream)) ||
                !(status = prepare(rigid.get(), substep, stream)))
                return status;
            std::uint32_t order{};
            if (fluid && !(status = apply_coupling(fluid->coupling_points(), order++, stream)))
                return status;
            if (cloth && !(status = apply_coupling(cloth->coupling_points(), order++, stream)))
                return status;
            if (rope && !(status = apply_coupling(rope->coupling_points(), order++, stream)))
                return status;
            if (soft_body &&
                !(status = apply_coupling(soft_body->coupling_points(), order++, stream)))
                return status;
            if (smoke) {
                if (fluid && !(status = smoke->couple(fluid->coupling_points(), substep.timestep,
                                                      3.0F, stream)))
                    return status;
                if (cloth && !(status = smoke->couple(cloth->coupling_points(), substep.timestep,
                                                      3.0F, stream)))
                    return status;
                if (rope && !(status = smoke->couple(rope->coupling_points(), substep.timestep,
                                                     3.0F, stream)))
                    return status;
                if (soft_body && !(status = smoke->couple(soft_body->coupling_points(),
                                                          substep.timestep, 3.0F, stream)))
                    return status;
            }
            if (!(status = finish(fluid.get(), substep, stream)) ||
                !(status = finish(cloth.get(), substep, stream)) ||
                !(status = finish(rope.get(), substep, stream)) ||
                !(status = finish(soft_body.get(), substep, stream)) ||
                !(status = finish(rigid.get(), substep, stream)))
                return status;
        }
        if (!(status = complete(fluid.get(), stream)) ||
            !(status = complete(cloth.get(), stream)) || !(status = complete(rope.get(), stream)) ||
            !(status = complete(soft_body.get(), stream)) ||
            !(status = complete(rigid.get(), stream)))
            return status;
        if (fluid && !(status = fluid->collect_statistics(stream))) return status;
        if (cloth && !(status = cloth->collect_statistics(stream))) return status;
        if (rope && !(status = rope->collect_statistics(stream))) return status;
        if (soft_body && !(status = soft_body->collect_telemetry(stream))) return status;
        if (smoke && !(status = smoke->collect_telemetry(stream))) return status;
        refresh_views();
        refresh_statistics();
        return statistics.finite_failure_count == 0U
                   ? Status{}
                   : Status{StatusCode::internal_error, cudaSuccess,
                            "gallery simulation produced non-finite state"};
    }

    [[nodiscard]] Status reset(cudaStream_t stream) noexcept {
        Status status;
        if (fluid && !(status = fluid->reset(stream))) return status;
        if (cloth && !(status = cloth->reset(stream))) return status;
        if (rope && !(status = rope->reset(stream))) return status;
        if (soft_body && !(status = soft_body->reset(stream))) return status;
        if (smoke && !(status = smoke->reset(stream))) return status;
        if (rigid && !(status = rigid->reset())) return status;
        refresh_views();
        refresh_statistics();
        return {};
    }

    void append_surface(physics::SoftBodySurfaceView view) noexcept {
        if (view.mesh.vertex_count == 0U) return;
        surfaces[surface_count++] = {view.mesh, view.vertex_normals, view.texcoords,
                                     view.triangle_active};
    }
    void append_lattice(physics::SoftBodyNodeView nodes, physics::SoftBodyBondView bonds) noexcept {
        if (nodes.node_count == 0U) return;
        lattices[lattice_count++] = {
            nodes.positions,          nodes.flags,          bonds.bonds,
            bonds.bond_active,        nodes.node_count,     nodes.nodes_per_instance,
            bonds.bonds_per_instance, nodes.instance_count, nodes.node_radius};
    }
    void refresh_views() noexcept {
        particle_count = surface_count = rigid_count = lattice_count = 0U;
        if (fluid) {
            const auto view = fluid->particles();
            particles[particle_count++] = {view.positions, view.velocities, view.particle_count,
                                           view.particle_radius, ParticleMaterial::fluid};
        }
        if (smoke) {
            const auto view = smoke->particles();
            particles[particle_count++] = {view.positions, view.velocities, view.count, 0.018F,
                                           options.recipe == SimulationRecipe::fluid_smoke
                                               ? ParticleMaterial::steam
                                               : ParticleMaterial::smoke};
        }
        if (cloth) {
            append_surface(cloth->surface());
            append_lattice(cloth->nodes(), cloth->bonds());
        }
        if (rope) {
            append_surface(rope->surface());
            append_lattice(rope->nodes(), rope->bonds());
        }
        if (soft_body) {
            append_surface(soft_body->surface());
            append_lattice(soft_body->nodes(), soft_body->bonds());
        }
        if (rigid) {
            const auto state = rigid->state();
            const float radius = rigid->options().radius;
            rigid_views[rigid_count++] = {{rigid_vertices.get(), sphere_vertices.size(),
                                           rigid_triangles.get(), sphere_triangles.size()},
                                          state.position,
                                          state.orientation,
                                          {radius, radius, radius}};
        }
    }
    void refresh_statistics() noexcept {
        statistics = {};
        statistics.particle_count =
            particle_count == 0U
                ? 0U
                : static_cast<std::uint32_t>(std::accumulate(
                      particles.begin(), particles.begin() + particle_count, std::size_t{},
                      [](std::size_t sum, const ParticleRenderView &view) {
                          return sum + view.count;
                      }));
        statistics.surface_count = surface_count;
        statistics.rigid_body_count = rigid_count;
        if (fluid) {
            const auto value = fluid->statistics();
            statistics.frame_index = std::max(statistics.frame_index, value.frame_index);
            statistics.finite_failure_count += value.finite_failure_count;
            statistics.allocated_bytes += value.allocated_bytes;
        }
        const auto add_deformable = [this](physics::SoftBodyStatistics value) {
            statistics.frame_index = std::max(statistics.frame_index, value.frame_index);
            statistics.finite_failure_count += value.finite_failure_count;
            statistics.broken_connection_count += value.broken_bond_count;
        };
        if (cloth) add_deformable(cloth->statistics());
        if (rope) add_deformable(rope->statistics());
        if (soft_body) {
            add_deformable(soft_body->statistics());
            statistics.allocated_bytes += soft_body->allocated_bytes();
            statistics.last_gpu_time_ms += soft_body->telemetry().timings.gpu_total_ms();
        }
        if (smoke) {
            const auto value = smoke->statistics();
            statistics.frame_index = std::max(statistics.frame_index, value.frame_index);
            statistics.finite_failure_count += value.finite_failure_count;
            statistics.allocated_bytes += value.allocated_bytes;
            statistics.last_gpu_time_ms += smoke->telemetry().timings.gpu_total_ms();
        }
        statistics.allocated_bytes += rigid_vertices.size() * sizeof(float3) +
                                      rigid_triangles.size() * sizeof(uint3) +
                                      constraints.allocated_bytes();
    }

    GallerySimulationOptions options{};
    ResolvedPhysicsOptions resolved{};
    const SimulationRecipeInfo *info{};
    std::string asset_path;
    physics::FrameOptions frame{};
    std::unique_ptr<physics::Fluid> fluid;
    std::unique_ptr<physics::Cloth> cloth;
    std::unique_ptr<physics::Rope> rope;
    std::unique_ptr<physics::SoftBody> soft_body;
    std::unique_ptr<physics::RigidBody> rigid;
    std::unique_ptr<physics::Smoke> smoke;
    physics::ColliderSet colliders;
    physics::ConstraintBatch constraints;
    physics::ConstraintRecord host_constraint{};
    DeviceBuffer<physics::ConstraintRecord> constraint_records;
    DeviceBuffer<float3> rigid_vertices;
    DeviceBuffer<uint3> rigid_triangles;
    std::array<ParticleRenderView, 2U> particles{};
    std::array<SurfaceRenderView, 3U> surfaces{};
    std::array<RigidBodyRenderView, 1U> rigid_views{};
    std::array<LatticeRenderView, 3U> lattices{};
    std::uint32_t particle_count{};
    std::uint32_t surface_count{};
    std::uint32_t rigid_count{};
    std::uint32_t lattice_count{};
    GallerySimulationStatistics statistics{};
};

GallerySimulation::GallerySimulation() noexcept = default;
GallerySimulation::~GallerySimulation() = default;
GallerySimulation::GallerySimulation(GallerySimulation &&) noexcept = default;
GallerySimulation &GallerySimulation::operator=(GallerySimulation &&) noexcept = default;

Status GallerySimulation::create(GallerySimulationOptions options, GallerySimulation &output,
                                 cudaStream_t stream) noexcept {
    return output.initialize(options, stream);
}

Status GallerySimulation::initialize(GallerySimulationOptions options,
                                     cudaStream_t stream) noexcept {
    try {
        auto replacement = std::make_unique<Impl>();
        Status status = replacement->initialize(options, stream);
        if (status) impl_ = std::move(replacement);
        return status;
    } catch (const std::bad_alloc &) {
        return allocation_failure("host allocation failed while initializing gallery");
    } catch (...) {
        return {StatusCode::internal_error, cudaSuccess,
                "gallery initialization raised an unexpected exception"};
    }
}

Status GallerySimulation::step(cudaStream_t stream) noexcept {
    return impl_ ? impl_->step(stream) : invalid("gallery simulation is not initialized");
}
Status GallerySimulation::reset(cudaStream_t stream) noexcept {
    return impl_ ? impl_->reset(stream) : invalid("gallery simulation is not initialized");
}

Status GallerySimulation::resize_particles(std::uint32_t active_count,
                                           cudaStream_t stream) noexcept {
    if (!impl_ || !impl_->fluid) return invalid("gallery recipe has no fluid owner");
    if (active_count < 8U || active_count > 100'000U)
        return invalid("fluid particle count must be in [8, 100000]");
    Status status = impl_->initialize_fluid(active_count, stream);
    if (status) {
        impl_->refresh_views();
        impl_->refresh_statistics();
    }
    return status;
}

bool GallerySimulation::initialized() const noexcept { return impl_ != nullptr; }
SimulationRecipe GallerySimulation::recipe() const noexcept {
    return impl_ ? impl_->options.recipe : SimulationRecipe::water;
}
GallerySimulationOptions GallerySimulation::options() const noexcept {
    return impl_ ? impl_->options : GallerySimulationOptions{};
}
ResolvedPhysicsOptions GallerySimulation::resolved_physics() const noexcept {
    return impl_ ? impl_->resolved : ResolvedPhysicsOptions{};
}
GallerySimulationStatistics GallerySimulation::statistics() const noexcept {
    return impl_ ? impl_->statistics : GallerySimulationStatistics{};
}
FrameRenderView GallerySimulation::render_view() const noexcept {
    return impl_ ? FrameRenderView{impl_->particles.data(),   impl_->particle_count,
                                   impl_->surfaces.data(),    impl_->surface_count,
                                   impl_->rigid_views.data(), impl_->rigid_count,
                                   impl_->lattices.data(),    impl_->lattice_count}
                 : FrameRenderView{};
}

SimulationBuilder &SimulationBuilder::timestep(float value) noexcept {
    config_.fixed_timestep = value;
    return *this;
}
SimulationBuilder &SimulationBuilder::iterations(std::uint32_t value) noexcept {
    config_.solver_iterations = value;
    return *this;
}
SimulationBuilder &SimulationBuilder::particles(std::uint32_t value) noexcept {
    config_.particle_count = value;
    return *this;
}
SimulationBuilder &SimulationBuilder::skin_frequency(std::uint32_t value) noexcept {
    config_.physical_skin_frequency = value;
    return *this;
}
SimulationBuilder &SimulationBuilder::rope_nodes(std::uint32_t value) noexcept {
    config_.rope_node_count = value;
    return *this;
}
SimulationBuilder &SimulationBuilder::cloth_detail(std::uint32_t value) noexcept {
    config_.cloth_detail = value;
    return *this;
}
SimulationBuilder &SimulationBuilder::bridge_grid(std::uint32_t columns,
                                                  std::uint32_t rows) noexcept {
    bridge_columns_ = columns;
    bridge_rows_ = rows;
    return *this;
}
SimulationBuilder &SimulationBuilder::cylinder_grid(std::uint32_t columns,
                                                    std::uint32_t rows) noexcept {
    cylinder_columns_ = columns;
    cylinder_rows_ = rows;
    return *this;
}
SimulationBuilder &SimulationBuilder::gravity(float3 value) noexcept {
    gravity_ = value;
    return *this;
}
SimulationBuilder &SimulationBuilder::soft_body_asset(std::string path) {
    asset_path_ = std::move(path);
    return *this;
}

Status SimulationBuilder::build(GallerySimulation &output, cudaStream_t stream) const noexcept {
    const RecipeConfigError error = validate_recipe_config(config_);
    if (error != RecipeConfigError::none) return invalid(recipe_config_error_message(error).data());
    GallerySimulationOptions options;
    options.recipe = config_.recipe;
    options.fixed_step = {config_.fixed_timestep};
    options.solver_iterations_override = config_.solver_iterations;
    options.gravity_override = gravity_;
    options.particle_count_override = config_.particle_count;
    options.physical_skin_frequency_override = config_.physical_skin_frequency;
    options.rope_node_count_override = config_.rope_node_count;
    options.cloth_detail_override = config_.cloth_detail;
    options.bridge_columns_override = bridge_columns_;
    options.bridge_rows_override = bridge_rows_;
    options.cylinder_columns_override = cylinder_columns_;
    options.cylinder_rows_override = cylinder_rows_;
    options.soft_body_asset_path = asset_path_;
    return output.initialize(options, stream);
}

} // namespace parallel_mater::examples
