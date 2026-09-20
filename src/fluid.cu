// SPDX-License-Identifier: MIT
#include <parallel_mater/physics.hpp>

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <utility>

namespace parallel_mater::physics {
namespace {

constexpr std::uint32_t block_size = 256U;
constexpr int cell_bias = 1 << 20;

[[nodiscard]] constexpr Status invalid(const char* message) noexcept
{
    return {StatusCode::invalid_argument, cudaSuccess, message};
}

[[nodiscard]] Status cuda_status(cudaError_t error, const char* message) noexcept
{
    return error == cudaSuccess
        ? Status{}
        : Status{error == cudaErrorMemoryAllocation
                ? StatusCode::allocation_failure : StatusCode::cuda_failure,
            error, message};
}

[[nodiscard]] bool finite(float value) noexcept { return std::isfinite(value); }
[[nodiscard]] bool finite(float3 value) noexcept
{
    return finite(value.x) && finite(value.y) && finite(value.z);
}

[[nodiscard]] bool valid(FluidOptions options) noexcept
{
    return finite(options.particle_radius) && options.particle_radius > 0.0F &&
        finite(options.interaction_radius) &&
        options.interaction_radius >= 2.0F * options.particle_radius &&
        finite(options.particle_mass) && options.particle_mass > 0.0F &&
        finite(options.repulsion) && options.repulsion >= 0.0F &&
        finite(options.viscosity) && options.viscosity >= 0.0F &&
        finite(options.velocity_damping) && options.velocity_damping >= 0.0F &&
        finite(options.maximum_speed) && options.maximum_speed > 0.0F;
}

__host__ __device__ float3 add(float3 a, float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ float3 subtract(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ float3 multiply(float3 value, float scale)
{
    return make_float3(value.x * scale, value.y * scale, value.z * scale);
}

__host__ __device__ float dot(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ std::uint64_t cell_key(int x, int y, int z)
{
    const auto encoded_x = static_cast<std::uint64_t>(x + cell_bias);
    const auto encoded_y = static_cast<std::uint64_t>(y + cell_bias);
    const auto encoded_z = static_cast<std::uint64_t>(z + cell_bias);
    return (encoded_x << 42U) | (encoded_y << 21U) | encoded_z;
}

__device__ std::uint32_t lower_bound(
    const std::uint64_t* keys, std::uint32_t count, std::uint64_t key)
{
    std::uint32_t first = 0U;
    std::uint32_t last = count;
    while (first < last) {
        const std::uint32_t middle = first + (last - first) / 2U;
        if (keys[middle] < key) first = middle + 1U;
        else last = middle;
    }
    return first;
}

__global__ void emit_cells(
    const float3* positions, std::uint64_t* keys, std::uint32_t* indices,
    std::uint32_t count, float inverse_cell_size)
{
    const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle >= count) return;
    const float3 point = positions[particle];
    keys[particle] = cell_key(__float2int_rd(point.x * inverse_cell_size),
        __float2int_rd(point.y * inverse_cell_size),
        __float2int_rd(point.z * inverse_cell_size));
    indices[particle] = particle;
}

__global__ void compute_forces(
    const float3* positions, const float3* velocities,
    const std::uint64_t* keys, const std::uint32_t* indices,
    std::uint32_t count, FluidOptions options, float3* forces,
    std::uint32_t* diagnostics)
{
    const std::uint32_t work = blockIdx.x * blockDim.x + threadIdx.x;
    if (work >= count) return;
    const std::uint32_t particle = indices[work];
    const float3 point = positions[particle];
    const float3 velocity = velocities[particle];
    const float inverse_radius = 1.0F / options.interaction_radius;
    const int cell_x = __float2int_rd(point.x * inverse_radius);
    const int cell_y = __float2int_rd(point.y * inverse_radius);
    const int cell_z = __float2int_rd(point.z * inverse_radius);
    float3 force{};
    std::uint32_t neighbors = 0U;
    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                const std::uint64_t key = cell_key(
                    cell_x + dx, cell_y + dy, cell_z + dz);
                const std::uint32_t first = lower_bound(keys, count, key);
                for (std::uint32_t item = first;
                     item < count && keys[item] == key; ++item) {
                    const std::uint32_t other = indices[item];
                    if (other == particle) continue;
                    const float3 delta = subtract(point, positions[other]);
                    const float distance_squared = dot(delta, delta);
                    if (!(distance_squared > 1.0e-12F) ||
                        distance_squared >= options.interaction_radius *
                            options.interaction_radius) continue;
                    ++neighbors;
                    const float distance = sqrtf(distance_squared);
                    const float q = 1.0F - distance * inverse_radius;
                    const float3 normal = multiply(delta, 1.0F / distance);
                    force = add(force, multiply(normal,
                        options.repulsion * q * q));
                    force = add(force, multiply(
                        subtract(velocities[other], velocity),
                        options.viscosity * q));
                }
            }
        }
    }
    forces[particle] = force;
    atomicMax(diagnostics, neighbors);
}

__global__ void integrate(
    float3* positions, float3* velocities, const float3* forces,
    const float3* external_impulses, std::uint32_t count,
    FluidOptions options, SubstepContext substep,
    std::uint32_t* diagnostics)
{
    const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle >= count) return;
    const float inverse_mass = 1.0F / options.particle_mass;
    float3 velocity = velocities[particle];
    velocity = add(velocity, multiply(add(substep.acceleration,
        multiply(forces[particle], inverse_mass)), substep.timestep));
    velocity = add(velocity,
        multiply(external_impulses[particle], inverse_mass));
    velocity = multiply(velocity,
        expf(-options.velocity_damping * substep.timestep));
    const float speed_squared = dot(velocity, velocity);
    if (speed_squared > options.maximum_speed * options.maximum_speed) {
        velocity = multiply(velocity,
            options.maximum_speed / sqrtf(speed_squared));
    }
    const float3 position = add(
        positions[particle], multiply(velocity, substep.timestep));
    if (!isfinite(position.x) || !isfinite(position.y) ||
        !isfinite(position.z) || !isfinite(velocity.x) ||
        !isfinite(velocity.y) || !isfinite(velocity.z)) {
        atomicAdd(diagnostics + 1U, 1U);
        return;
    }
    positions[particle] = position;
    velocities[particle] = velocity;
}

template <typename T>
Status allocate(T*& pointer, std::size_t count) noexcept
{
    if (count == 0U) return {};
    return cuda_status(cudaMalloc(&pointer, count * sizeof(T)),
        "could not allocate fluid storage");
}

} // namespace

struct Fluid::Impl {
    ~Impl()
    {
        if (host_diagnostics != nullptr) cudaFreeHost(host_diagnostics);
        cudaFree(cub_storage);
        cudaFree(diagnostics);
        cudaFree(indices_b);
        cudaFree(indices_a);
        cudaFree(keys_b);
        cudaFree(keys_a);
        cudaFree(external_impulses);
        cudaFree(forces);
        cudaFree(initial_velocities);
        cudaFree(initial_positions);
        cudaFree(velocities);
        cudaFree(positions);
    }

    FluidOptions options{};
    std::uint32_t count{};
    float3* positions{};
    float3* velocities{};
    float3* initial_positions{};
    float3* initial_velocities{};
    float3* forces{};
    float3* external_impulses{};
    std::uint64_t* keys_a{};
    std::uint64_t* keys_b{};
    std::uint32_t* indices_a{};
    std::uint32_t* indices_b{};
    std::uint32_t* diagnostics{};
    std::uint32_t* host_diagnostics{};
    void* cub_storage{};
    std::size_t cub_storage_bytes{};
    FrameOptions frame{};
    std::uint32_t next_substep{};
    bool frame_active{};
    std::uint64_t frame_index{};
};

Fluid::Fluid() noexcept = default;
Fluid::~Fluid() = default;
Fluid::Fluid(Fluid&&) noexcept = default;
Fluid& Fluid::operator=(Fluid&&) noexcept = default;

Status Fluid::create(
    std::span<const FluidParticle> particles, FluidOptions options,
    Fluid& output, cudaStream_t stream) noexcept
{
    return output.initialize(particles, options, stream);
}

Status Fluid::initialize(
    std::span<const FluidParticle> particles, FluidOptions options,
    cudaStream_t stream) noexcept
{
    if (particles.empty() ||
        particles.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        return invalid("fluid particle count exceeds the CUB sort limit");
    }
    if (!valid(options)) return invalid("invalid fluid options");
    for (const FluidParticle& particle : particles) {
        if (!finite(particle.position) || !finite(particle.velocity)) {
            return invalid("fluid particle state must be finite");
        }
    }
    auto replacement = std::unique_ptr<Impl>(new (std::nothrow) Impl);
    if (!replacement) {
        return {StatusCode::allocation_failure, cudaErrorMemoryAllocation,
            "could not allocate fluid owner"};
    }
    replacement->options = options;
    replacement->count = static_cast<std::uint32_t>(particles.size());
    Status status{};
    const std::size_t count = particles.size();
    if (!(status = allocate(replacement->positions, count)) ||
        !(status = allocate(replacement->velocities, count)) ||
        !(status = allocate(replacement->initial_positions, count)) ||
        !(status = allocate(replacement->initial_velocities, count)) ||
        !(status = allocate(replacement->forces, count)) ||
        !(status = allocate(replacement->external_impulses, count)) ||
        !(status = allocate(replacement->keys_a, count)) ||
        !(status = allocate(replacement->keys_b, count)) ||
        !(status = allocate(replacement->indices_a, count)) ||
        !(status = allocate(replacement->indices_b, count)) ||
        !(status = allocate(replacement->diagnostics, 2U))) return status;
    if (!(status = cuda_status(cudaHostAlloc(&replacement->host_diagnostics,
            2U * sizeof(std::uint32_t), cudaHostAllocPortable),
            "could not allocate fluid diagnostic staging"))) return status;

    std::unique_ptr<float3[]> host_positions(new (std::nothrow) float3[count]);
    std::unique_ptr<float3[]> host_velocities(new (std::nothrow) float3[count]);
    if (!host_positions || !host_velocities) {
        return {StatusCode::allocation_failure, cudaErrorMemoryAllocation,
            "could not stage fluid particles"};
    }
    for (std::size_t i = 0U; i < count; ++i) {
        host_positions[i] = particles[i].position;
        host_velocities[i] = particles[i].velocity;
    }
    const std::size_t bytes = count * sizeof(float3);
    if (!(status = cuda_status(cudaMemcpyAsync(replacement->positions,
            host_positions.get(), bytes, cudaMemcpyHostToDevice, stream),
            "could not upload fluid positions")) ||
        !(status = cuda_status(cudaMemcpyAsync(replacement->velocities,
            host_velocities.get(), bytes, cudaMemcpyHostToDevice, stream),
            "could not upload fluid velocities")) ||
        !(status = cuda_status(cudaMemcpyAsync(replacement->initial_positions,
            host_positions.get(), bytes, cudaMemcpyHostToDevice, stream),
            "could not upload initial fluid positions")) ||
        !(status = cuda_status(cudaMemcpyAsync(replacement->initial_velocities,
            host_velocities.get(), bytes, cudaMemcpyHostToDevice, stream),
            "could not upload initial fluid velocities"))) return status;

    if (!(status = cuda_status(cub::DeviceRadixSort::SortPairs(nullptr,
            replacement->cub_storage_bytes, replacement->keys_a,
            replacement->keys_b, replacement->indices_a,
            replacement->indices_b, static_cast<int>(count), 0, 64, stream),
            "could not size fluid sort workspace"))) return status;
    if (!(status = cuda_status(cudaMalloc(&replacement->cub_storage,
            replacement->cub_storage_bytes),
            "could not allocate fluid sort workspace"))) return status;
    if (!(status = cuda_status(cudaMemsetAsync(replacement->diagnostics, 0,
            2U * sizeof(std::uint32_t), stream),
            "could not clear fluid diagnostics")) ||
        !(status = cuda_status(cudaStreamSynchronize(stream),
            "could not complete fluid initialization"))) return status;
    replacement->host_diagnostics[0] = 0U;
    replacement->host_diagnostics[1] = 0U;
    impl_ = std::move(replacement);
    return {};
}

Status Fluid::begin_frame(FrameOptions frame, cudaStream_t stream) noexcept
{
    if (!impl_) return invalid("fluid is not initialized");
    if (!parallel_mater::physics::valid(frame)) return invalid("invalid frame options");
    if (impl_->frame_active) return invalid("fluid frame is already active");
    Status status = cuda_status(cudaMemsetAsync(impl_->diagnostics, 0,
        2U * sizeof(std::uint32_t), stream), "could not clear fluid diagnostics");
    if (status) {
        impl_->frame = frame;
        impl_->next_substep = 0U;
        impl_->frame_active = true;
    }
    return status;
}

Status Fluid::prepare_substep(SubstepContext substep, cudaStream_t stream) noexcept
{
    if (!impl_ || !impl_->frame_active) return invalid("fluid frame is not active");
    if (substep.index != impl_->next_substep ||
        substep.count != impl_->frame.substeps ||
        !finite(substep.timestep) || substep.timestep <= 0.0F ||
        !finite(substep.acceleration)) return invalid("invalid fluid substep");
    const std::uint32_t blocks = (impl_->count + block_size - 1U) / block_size;
    emit_cells<<<blocks, block_size, 0, stream>>>(impl_->positions,
        impl_->keys_a, impl_->indices_a, impl_->count,
        1.0F / impl_->options.interaction_radius);
    Status status = cuda_status(cudaGetLastError(), "could not emit fluid cells");
    if (!status) return status;
    status = cuda_status(cub::DeviceRadixSort::SortPairs(impl_->cub_storage,
        impl_->cub_storage_bytes, impl_->keys_a, impl_->keys_b,
        impl_->indices_a, impl_->indices_b, static_cast<int>(impl_->count),
        0, 64, stream), "could not sort fluid cells");
    if (!status) return status;
    status = cuda_status(cudaMemsetAsync(impl_->external_impulses, 0,
        static_cast<std::size_t>(impl_->count) * sizeof(float3), stream),
        "could not clear fluid coupling impulses");
    if (!status) return status;
    compute_forces<<<blocks, block_size, 0, stream>>>(impl_->positions,
        impl_->velocities, impl_->keys_b, impl_->indices_b, impl_->count,
        impl_->options, impl_->forces, impl_->diagnostics);
    return cuda_status(cudaGetLastError(), "could not compute fluid forces");
}

Status Fluid::finish_substep(SubstepContext substep, cudaStream_t stream) noexcept
{
    if (!impl_ || !impl_->frame_active) return invalid("fluid frame is not active");
    if (substep.index != impl_->next_substep ||
        substep.count != impl_->frame.substeps) return invalid("fluid substep is out of order");
    const std::uint32_t blocks = (impl_->count + block_size - 1U) / block_size;
    integrate<<<blocks, block_size, 0, stream>>>(impl_->positions,
        impl_->velocities, impl_->forces, impl_->external_impulses,
        impl_->count, impl_->options, substep, impl_->diagnostics);
    Status status = cuda_status(cudaGetLastError(), "could not integrate fluid");
    if (status) ++impl_->next_substep;
    return status;
}

Status Fluid::finish_frame(Completion& completion, cudaStream_t stream) noexcept
{
    if (!impl_ || !impl_->frame_active) return invalid("fluid frame is not active");
    if (impl_->next_substep != impl_->frame.substeps) {
        return invalid("fluid frame has incomplete substeps");
    }
    if (completion.pending()) return invalid("completion token is already pending");
    Status status = completion.record(stream);
    if (status) {
        impl_->frame_active = false;
        ++impl_->frame_index;
    }
    return status;
}

Status Fluid::collect_statistics_async(
    Completion& completion, cudaStream_t stream) noexcept
{
    if (!impl_) return invalid("fluid is not initialized");
    if (impl_->frame_active) return invalid("cannot collect an active fluid frame");
    if (completion.pending()) return invalid("completion token is already pending");
    Status status = cuda_status(cudaMemcpyAsync(impl_->host_diagnostics,
        impl_->diagnostics, 2U * sizeof(std::uint32_t),
        cudaMemcpyDeviceToHost, stream), "could not download fluid diagnostics");
    return status ? completion.record(stream) : status;
}

Status Fluid::collect_statistics(cudaStream_t stream) noexcept
{
    Completion completion;
    Status status = collect_statistics_async(completion, stream);
    return status ? completion.wait() : status;
}

Status Fluid::advance_async(
    FrameOptions frame, Completion& completion, cudaStream_t stream) noexcept
{
    return parallel_mater::physics::step_async(*this, frame, completion, stream);
}

Status Fluid::advance(FrameOptions frame, cudaStream_t stream) noexcept
{
    return parallel_mater::physics::step(*this, frame, stream);
}

Status Fluid::reset(cudaStream_t stream) noexcept
{
    if (!impl_) return invalid("fluid is not initialized");
    if (impl_->frame_active) return invalid("cannot reset an active fluid frame");
    const std::size_t bytes = static_cast<std::size_t>(impl_->count) * sizeof(float3);
    Status status = cuda_status(cudaMemcpyAsync(impl_->positions,
        impl_->initial_positions, bytes, cudaMemcpyDeviceToDevice, stream),
        "could not reset fluid positions");
    if (!status) return status;
    status = cuda_status(cudaMemcpyAsync(impl_->velocities,
        impl_->initial_velocities, bytes, cudaMemcpyDeviceToDevice, stream),
        "could not reset fluid velocities");
    if (!status) return status;
    status = cuda_status(cudaStreamSynchronize(stream),
        "could not complete fluid reset");
    if (status) {
        impl_->frame_index = 0U;
        impl_->host_diagnostics[0] = 0U;
        impl_->host_diagnostics[1] = 0U;
    }
    return status;
}

bool Fluid::initialized() const noexcept { return impl_ != nullptr; }
FluidOptions Fluid::options() const noexcept
{
    return impl_ ? impl_->options : FluidOptions{};
}
FluidView Fluid::particles() const noexcept
{
    return impl_ ? FluidView{impl_->positions, impl_->velocities,
        impl_->external_impulses, impl_->count, impl_->options.particle_radius,
        1.0F / impl_->options.particle_mass} : FluidView{};
}
PointCouplingView Fluid::coupling_points() const noexcept
{
    if (!impl_) return {};
    return {impl_->positions, impl_->velocities, impl_->external_impulses,
        nullptr, impl_->count, impl_->options.particle_radius,
        1.0F / impl_->options.particle_mass};
}
FluidStatistics Fluid::statistics() const noexcept
{
    if (!impl_) return {};
    const std::size_t count = impl_->count;
    const std::size_t arrays = 6U * count * sizeof(float3) +
        2U * count * sizeof(std::uint64_t) +
        2U * count * sizeof(std::uint32_t) + 2U * sizeof(std::uint32_t) +
        impl_->cub_storage_bytes;
    return {impl_->count, impl_->host_diagnostics[0],
        impl_->host_diagnostics[1], impl_->frame_index, arrays};
}

static_assert(FrameSolver<Fluid>);

} // namespace parallel_mater::physics
