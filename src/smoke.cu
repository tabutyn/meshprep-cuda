// SPDX-License-Identifier: MIT
#include <parallel_mater/smoke.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <memory>
#include <new>
#include <utility>

namespace parallel_mater::physics {
namespace {

constexpr std::uint32_t block_size = 256U;
constexpr std::uint32_t maximum_colliders = 1'024U;

__host__ __device__ float3 add(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ float3 subtract(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ float3 multiply(float3 value, float scale) {
    return make_float3(value.x * scale, value.y * scale, value.z * scale);
}

__host__ __device__ float dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

__host__ __device__ float length(float3 value) { return sqrtf(dot(value, value)); }

__host__ __device__ bool finite3(float3 value) {
    return isfinite(value.x) && isfinite(value.y) && isfinite(value.z);
}

__device__ float4 blend(float4 destination, float4 source, float amount) {
    amount = fminf(1.0F, fmaxf(0.0F, amount));
    return make_float4(destination.x + (source.x - destination.x) * amount,
                       destination.y + (source.y - destination.y) * amount,
                       destination.z + (source.z - destination.z) * amount,
                       destination.w + (source.w - destination.w) * amount);
}

__device__ std::uint32_t hash(std::uint32_t value) {
    value ^= value >> 16U;
    value *= 0x7feb352dU;
    value ^= value >> 15U;
    value *= 0x846ca68bU;
    return value ^ (value >> 16U);
}

__device__ float random_signed(std::uint32_t value) {
    return 2.0F * static_cast<float>(hash(value) & 0x00ffffffU) / 16777215.0F - 1.0F;
}

__device__ float3 spawn_position(std::uint32_t particle, std::uint64_t generation,
                                 SmokeOptions options) {
    const std::uint32_t base =
        options.seed ^ particle ^ static_cast<std::uint32_t>(generation * 0x9e3779b9ULL);
    return add(options.emitter_center,
               make_float3(options.emitter_half_extents.x * random_signed(base + 0x1234U),
                           options.emitter_half_extents.y * random_signed(base + 0x5678U),
                           options.emitter_half_extents.z * random_signed(base + 0x9abcU)));
}

__device__ float3 turbulence(float3 p, float time, SmokeOptions options) {
    const float f = options.turbulence_frequency;
    const float x = sinf(f * (1.7F * p.y + 0.9F * p.z) + 1.3F * time);
    const float y = sinf(f * (1.1F * p.z + 1.5F * p.x) - 0.7F * time);
    const float z = sinf(f * (1.3F * p.x + 0.8F * p.y) + 0.9F * time);
    return multiply(make_float3(y - z, z - x, x - y), options.turbulence_strength);
}

__global__ void initialize_smoke(float3 *positions, float3 *velocities, float *ages,
                                 float *temperatures, float4 *colors, SmokeOptions options) {
    const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle >= options.particle_count) return;
    const float age = options.lifetime * static_cast<float>(particle) /
                      static_cast<float>(options.particle_count);
    const float3 emitter = spawn_position(particle, 0U, options);
    positions[particle] = add(emitter, multiply(options.initial_velocity, age));
    velocities[particle] = options.initial_velocity;
    ages[particle] = age;
    temperatures[particle] = 1.0F - age / options.lifetime;
    colors[particle] = options.initial_color;
}

__global__ void integrate_smoke(float3 *positions, float3 *velocities, float *ages,
                                float *temperatures, float4 *colors, SmokeOptions options,
                                float3 acceleration,
                                ColliderView colliders, std::uint64_t frame,
                                unsigned long long *respawns, std::uint32_t *counters) {
    const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle >= options.particle_count) return;
    float age = ages[particle] + options.timestep;
    float3 position = positions[particle];
    float3 velocity = velocities[particle];
    float4 color = colors[particle];
    if (age >= options.lifetime) {
        age = fmodf(age, options.lifetime);
        position = spawn_position(particle, frame, options);
        velocity = options.initial_velocity;
        color = options.initial_color;
        atomicAdd(respawns, 1ULL);
    }
    const float time = static_cast<float>(frame) * options.timestep;
    acceleration = add(acceleration, make_float3(0.0F, options.buoyancy, 0.0F));
    acceleration = add(acceleration, turbulence(position, time, options));
    velocity = add(velocity, multiply(acceleration, options.timestep));
    velocity = multiply(velocity, 1.0F / (1.0F + options.velocity_damping * options.timestep));
    const float speed = length(velocity);
    if (speed > options.maximum_speed) velocity = multiply(velocity, options.maximum_speed / speed);
    position = add(position, multiply(velocity, options.timestep));

    for (std::uint32_t index = 0U; index < colliders.count; ++index) {
        const Collider collider = colliders.data[index];
        if (collider.shape != ColliderShape::sphere) continue;
        const float radius = collider.dimensions.x;
        const float3 delta = subtract(position, collider.position);
        const float distance = length(delta);
        const float target = radius + collider.contact_offset + options.particle_radius;
        if (distance < target) {
            const float3 normal =
                distance > 1.0e-7F ? multiply(delta, 1.0F / distance) : make_float3(1, 0, 0);
            position = add(collider.position, multiply(normal, target));
            const float3 relative = subtract(velocity, collider.linear_velocity);
            const float normal_speed = dot(relative, normal);
            const float3 tangent = subtract(relative, multiply(normal, normal_speed));
            velocity = add(collider.linear_velocity,
                           add(multiply(normal, fmaxf(-collider.restitution * normal_speed, 0.0F)),
                               multiply(tangent, 1.0F - fminf(collider.friction, 1.0F))));
            if (collider.paint_amount > 0.0F)
                color = blend(color, collider.paint_color, collider.paint_amount);
        }
        // Vortex pair in the downstream wake makes obstacle-induced
        // turbulence visible instead of merely deleting smoke at the sphere.
        const float3 wake_delta = subtract(position, collider.position);
        const float wake_distance = length(wake_delta);
        const float stream_length = length(options.initial_velocity);
        if (wake_distance < 4.0F * radius && stream_length > 1.0e-6F) {
            const float3 stream = multiply(options.initial_velocity, 1.0F / stream_length);
            const float downstream = dot(wake_delta, stream);
            if (downstream > 0.0F) {
                const float3 swirl = make_float3(0.0F, -wake_delta.z, wake_delta.y);
                const float swirl_length = length(swirl);
                if (swirl_length > 1.0e-6F)
                    velocity =
                        add(velocity, multiply(swirl, options.turbulence_strength *
                                                          options.timestep * downstream /
                                                          (swirl_length * (wake_distance + 0.1F))));
            }
        }
    }
    if (!finite3(position) || !finite3(velocity) || !isfinite(age)) {
        position = spawn_position(particle, frame, options);
        velocity = options.initial_velocity;
        age = 0.0F;
        atomicAdd(counters, 1U);
    }
    positions[particle] = position;
    velocities[particle] = velocity;
    ages[particle] = age;
    temperatures[particle] = fmaxf(0.0F, 1.0F - age / options.lifetime);
    colors[particle] = color;
    atomicMax(counters + 1U, __float_as_uint(length(velocity)));
}

__global__ void couple_smoke(SmokeOptions options, std::uint64_t frame, PointCouplingView body,
                             float timestep, float drag) {
    const std::uint32_t node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= body.count || body.external_impulses == nullptr) return;
    const float3 point = body.positions[node];
    const float stream_speed = length(options.initial_velocity);
    if (!(stream_speed > 1.0e-6F) || !(body.inverse_mass > 0.0F)) return;
    const float3 direction = multiply(options.initial_velocity, 1.0F / stream_speed);
    const float3 from_emitter = subtract(point, options.emitter_center);
    const float downstream = dot(from_emitter, direction);
    const float reach = stream_speed * options.lifetime;
    if (downstream < -body.radius || downstream > reach + body.radius) return;
    const float3 lateral = subtract(from_emitter, multiply(direction, downstream));
    const float lateral_distance = length(lateral);
    const float emitter_radius =
        fmaxf(options.emitter_half_extents.x,
              fmaxf(options.emitter_half_extents.y, options.emitter_half_extents.z));
    const float spread = emitter_radius + body.radius + 0.12F * sqrtf(fmaxf(downstream, 0.0F));
    if (!(lateral_distance < spread)) return;
    const float radial = 1.0F - lateral_distance / spread;
    const float axial =
        fminf(1.0F, fmaxf(0.0F, (reach - downstream) / fmaxf(0.15F * reach, 1.0e-4F)));
    const float weight = radial * radial * axial;
    const float time = static_cast<float>(frame) * options.timestep;
    float3 sampled =
        add(options.initial_velocity, multiply(turbulence(point, time, options), 0.45F));
    sampled.y += options.buoyancy * fmaxf(downstream, 0.0F) / stream_speed;
    const float response = 1.0F - expf(-drag * timestep);
    const float mass = 1.0F / body.inverse_mass;
    const float3 impulse =
        multiply(subtract(sampled, body.velocities[node]), mass * response * weight);
    body.external_impulses[node] = add(body.external_impulses[node], impulse);
}

bool finite(float value) { return std::isfinite(value); }
bool finite(float3 value) { return finite(value.x) && finite(value.y) && finite(value.z); }
bool finite(float4 value) {
    return finite(value.x) && finite(value.y) && finite(value.z) && finite(value.w);
}

bool valid(SmokeOptions value) {
    return value.capacity >= 256U && value.capacity <= 1'000'000U && value.particle_count >= 1U &&
           value.particle_count <= value.capacity && finite(value.timestep) &&
           value.timestep > 0.0F && finite(value.lifetime) && value.lifetime > value.timestep &&
           finite(value.particle_radius) && value.particle_radius > 0.0F &&
           finite(value.emitter_center) && finite(value.emitter_half_extents) &&
           value.emitter_half_extents.x >= 0.0F && value.emitter_half_extents.y >= 0.0F &&
           value.emitter_half_extents.z >= 0.0F && finite(value.initial_velocity) &&
           finite(value.buoyancy) && finite(value.velocity_damping) &&
           value.velocity_damping >= 0.0F && finite(value.turbulence_strength) &&
           value.turbulence_strength >= 0.0F && finite(value.turbulence_frequency) &&
           value.turbulence_frequency > 0.0F && finite(value.maximum_speed) &&
           value.maximum_speed > 0.0F && finite(value.initial_color);
}

Status cuda_status(cudaError_t error, const char *message) {
    return error == cudaSuccess
               ? Status{}
               : Status{error == cudaErrorMemoryAllocation ? StatusCode::allocation_failure
                                                           : StatusCode::cuda_failure,
                        error, message};
}

} // namespace

struct Smoke::Impl {
    SmokeOptions options{};
    float3 *positions{};
    float3 *velocities{};
    float *ages{};
    float *temperatures{};
    float4 *colors{};
    unsigned long long *respawns{};
    std::uint32_t *counters{};
    unsigned long long *host_respawns{};
    std::uint32_t *host_counters{};
    cudaEvent_t begin{};
    cudaEvent_t end{};
    cudaEvent_t couple_begin{};
    cudaEvent_t couple_end{};
    cudaEvent_t telemetry_ready{};
    SmokeStatistics statistics{};
    SmokeTelemetry telemetry{};
    ColliderView colliders{};
    FrameOptions frame{};
    std::uint32_t next_substep{};
    bool frame_active{};
    bool telemetry_pending{};
    bool couple_recorded{};
    std::uint64_t tick_index{};
    std::uint64_t frame_start_index{};

    explicit Impl(SmokeOptions selected) : options(selected) {}
    ~Impl() {
        if (begin) cudaEventDestroy(begin);
        if (end) cudaEventDestroy(end);
        if (couple_begin) cudaEventDestroy(couple_begin);
        if (couple_end) cudaEventDestroy(couple_end);
        if (telemetry_ready) cudaEventDestroy(telemetry_ready);
        cudaFreeHost(host_counters);
        cudaFreeHost(host_respawns);
        cudaFree(counters);
        cudaFree(respawns);
        cudaFree(temperatures);
        cudaFree(colors);
        cudaFree(ages);
        cudaFree(velocities);
        cudaFree(positions);
    }
};

Smoke::Smoke() noexcept = default;
Smoke::~Smoke() = default;
Smoke::Smoke(Smoke &&) noexcept = default;
Smoke &Smoke::operator=(Smoke &&) noexcept = default;

Status Smoke::create(SmokeOptions options, Smoke &output, cudaStream_t stream) noexcept {
    return output.initialize(options, stream);
}

Status Smoke::initialize(SmokeOptions options, cudaStream_t stream) noexcept {
    if (!valid(options))
        return {StatusCode::invalid_argument, cudaSuccess, "invalid smoke options"};
    try {
        auto replacement = std::make_unique<Impl>(options);
        const std::size_t vector_bytes = options.capacity * sizeof(float3);
        const std::size_t scalar_bytes = options.capacity * sizeof(float);
        cudaError_t error = cudaMalloc(&replacement->positions, vector_bytes);
        if (error != cudaSuccess) return cuda_status(error, "allocate smoke positions");
        error = cudaMalloc(&replacement->velocities, vector_bytes);
        if (error != cudaSuccess) return cuda_status(error, "allocate smoke velocities");
        error = cudaMalloc(&replacement->ages, scalar_bytes);
        if (error != cudaSuccess) return cuda_status(error, "allocate smoke ages");
        error = cudaMalloc(&replacement->temperatures, scalar_bytes);
        if (error != cudaSuccess) return cuda_status(error, "allocate smoke temperatures");
        error = cudaMalloc(&replacement->colors, options.capacity * sizeof(float4));
        if (error != cudaSuccess) return cuda_status(error, "allocate smoke colors");
        error = cudaMalloc(&replacement->respawns, sizeof(unsigned long long));
        if (error != cudaSuccess) return cuda_status(error, "allocate smoke statistics");
        error = cudaMalloc(&replacement->counters, 2U * sizeof(std::uint32_t));
        if (error != cudaSuccess) return cuda_status(error, "allocate smoke counters");
        error = cudaMallocHost(&replacement->host_respawns, sizeof(unsigned long long));
        if (error != cudaSuccess) return cuda_status(error, "allocate smoke telemetry");
        error = cudaMallocHost(&replacement->host_counters, 2U * sizeof(std::uint32_t));
        if (error != cudaSuccess) return cuda_status(error, "allocate smoke telemetry counters");
        if ((error = cudaEventCreate(&replacement->begin)) != cudaSuccess ||
            (error = cudaEventCreate(&replacement->end)) != cudaSuccess ||
            (error = cudaEventCreate(&replacement->couple_begin)) != cudaSuccess ||
            (error = cudaEventCreate(&replacement->couple_end)) != cudaSuccess ||
            (error = cudaEventCreateWithFlags(&replacement->telemetry_ready,
                                              cudaEventDisableTiming)) != cudaSuccess)
            return cuda_status(error, "create smoke timing events");
        replacement->statistics.allocated_bytes = 2U * vector_bytes + 2U * scalar_bytes +
                                                  options.capacity * sizeof(float4) +
                                                  sizeof(unsigned long long) +
                                                  2U * sizeof(std::uint32_t);
        replacement->telemetry.statistics.allocated_bytes = replacement->statistics.allocated_bytes;
        impl_ = std::move(replacement);
        return reset(stream);
    } catch (const std::bad_alloc &) {
        return {StatusCode::allocation_failure, cudaErrorMemoryAllocation,
                "allocate smoke host state"};
    } catch (...) {
        return {StatusCode::internal_error, cudaSuccess, "initialize smoke"};
    }
}

Status Smoke::set_colliders(ColliderView colliders) noexcept {
    if (!impl_) return {StatusCode::invalid_argument, cudaSuccess, "smoke is not initialized"};
    if (colliders.count > maximum_colliders || (colliders.count != 0U && colliders.data == nullptr))
        return {StatusCode::invalid_argument, cudaSuccess, "invalid collider view"};
    impl_->colliders = colliders;
    return {};
}

Status Smoke::step(float3 acceleration, ColliderView colliders, cudaStream_t stream) noexcept {
    Completion completion;
    Status status = step_async(acceleration, colliders, completion, stream);
    return status ? completion.wait() : status;
}

Status Smoke::step(float3 acceleration, ColliderView colliders, SmokeTimings &timings,
                   cudaStream_t stream) noexcept {
    Status status = step(acceleration, colliders, stream);
    if (status) status = collect_telemetry(stream);
    if (status) timings = impl_->telemetry.timings;
    return status;
}

Status Smoke::step_async(float3 acceleration, ColliderView colliders, Completion &completion,
                         cudaStream_t stream) noexcept {
    Status status = set_colliders(colliders);
    if (!status) return status;
    return advance_async({impl_->options.timestep, 1U, acceleration}, completion, stream);
}

Status Smoke::begin_frame(FrameOptions frame, cudaStream_t) noexcept {
    if (!impl_) return {StatusCode::invalid_argument, cudaSuccess, "smoke is not initialized"};
    if (!parallel_mater::physics::valid(frame))
        return {StatusCode::invalid_argument, cudaSuccess, "invalid frame options"};
    if (impl_->frame_active)
        return {StatusCode::invalid_argument, cudaSuccess, "smoke frame is already active"};
    if (impl_->telemetry_pending)
        return {StatusCode::invalid_argument, cudaSuccess,
                "resolve pending smoke telemetry before beginning another frame"};
    impl_->frame = frame;
    impl_->next_substep = 0U;
    impl_->frame_active = true;
    impl_->frame_start_index = impl_->statistics.frame_index;
    return {};
}

Status Smoke::prepare_substep(SubstepContext substep, cudaStream_t) noexcept {
    if (!impl_ || !impl_->frame_active || substep.index != impl_->next_substep ||
        substep.count != impl_->frame.substeps)
        return {StatusCode::invalid_argument, cudaSuccess, "invalid smoke substep"};
    return {};
}

Status Smoke::finish_substep(SubstepContext substep, cudaStream_t stream) noexcept {
    if (!impl_ || !impl_->frame_active || substep.index != impl_->next_substep ||
        substep.count != impl_->frame.substeps)
        return {StatusCode::invalid_argument, cudaSuccess, "smoke substep is out of order"};
    SmokeOptions options = impl_->options;
    options.timestep = substep.timestep;
    cudaError_t error = cudaEventRecord(impl_->begin, stream);
    if (error != cudaSuccess) return cuda_status(error, "record smoke integration start");
    integrate_smoke<<<(options.particle_count + block_size - 1U) / block_size, block_size, 0,
                      stream>>>(impl_->positions, impl_->velocities, impl_->ages,
                                impl_->temperatures, impl_->colors, options, substep.acceleration,
                                impl_->colliders, impl_->tick_index + 1U, impl_->respawns,
                                impl_->counters);
    error = cudaGetLastError();
    if (error != cudaSuccess) return cuda_status(error, "launch smoke integration");
    error = cudaEventRecord(impl_->end, stream);
    if (error != cudaSuccess) return cuda_status(error, "record smoke integration end");
    ++impl_->tick_index;
    ++impl_->next_substep;
    return {};
}

Status Smoke::finish_frame(Completion &completion, cudaStream_t stream) noexcept {
    if (!impl_ || !impl_->frame_active || impl_->next_substep != impl_->frame.substeps)
        return {StatusCode::invalid_argument, cudaSuccess, "smoke frame has incomplete substeps"};
    if (completion.pending())
        return {StatusCode::invalid_argument, cudaSuccess, "completion token is already pending"};
    Status status = completion.record(stream);
    if (status) {
        impl_->statistics.frame_index = impl_->frame_start_index + 1U;
        impl_->frame_active = false;
    }
    return status;
}

Status Smoke::abandon_frame(cudaStream_t stream) noexcept {
    if (!impl_)
        return {StatusCode::invalid_argument, cudaSuccess, "smoke is not initialized"};
    const cudaError_t error = cudaStreamSynchronize(stream);
    impl_->frame_active = false;
    impl_->next_substep = 0U;
    return cuda_status(error, "could not drain abandoned smoke frame");
}

Status Smoke::advance_async(FrameOptions frame, Completion &completion,
                            cudaStream_t stream) noexcept {
    return parallel_mater::physics::step_async(*this, frame, completion, stream);
}

Status Smoke::advance(FrameOptions frame, cudaStream_t stream) noexcept {
    return parallel_mater::physics::step(*this, frame, stream);
}

Status Smoke::couple(PointCouplingView body, float timestep, float drag,
                     cudaStream_t stream) noexcept {
    if (!impl_) return {StatusCode::invalid_argument, cudaSuccess, "smoke is not initialized"};
    if (body.count == 0U) return {};
    if (body.positions == nullptr || body.velocities == nullptr ||
        body.external_impulses == nullptr || !finite(body.inverse_mass) ||
        body.inverse_mass <= 0.0F || !finite(body.radius) || body.radius < 0.0F ||
        !finite(timestep) || timestep <= 0.0F || !finite(drag) || drag < 0.0F)
        return {StatusCode::invalid_argument, cudaSuccess, "invalid smoke coupling view"};
    cudaEventRecord(impl_->couple_begin, stream);
    couple_smoke<<<(body.count + block_size - 1U) / block_size, block_size, 0, stream>>>(
        impl_->options, impl_->tick_index, body, timestep, drag);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return cuda_status(error, "launch smoke coupling");
    const Status status =
        cuda_status(cudaEventRecord(impl_->couple_end, stream), "record smoke coupling end");
    if (status) impl_->couple_recorded = true;
    return status;
}

Status Smoke::collect_telemetry_async(Completion &completion, cudaStream_t stream) noexcept {
    if (!impl_) return {StatusCode::invalid_argument, cudaSuccess, "smoke is not initialized"};
    if (impl_->frame_active || impl_->telemetry_pending)
        return {StatusCode::invalid_argument, cudaSuccess,
                "smoke telemetry requires one completed frame"};
    if (completion.pending())
        return {StatusCode::invalid_argument, cudaSuccess, "completion token is already pending"};
    cudaError_t error = cudaMemcpyAsync(impl_->host_respawns, impl_->respawns,
                                        sizeof(unsigned long long), cudaMemcpyDeviceToHost, stream);
    if (error != cudaSuccess) return cuda_status(error, "download smoke respawn count");
    error = cudaMemcpyAsync(impl_->host_counters, impl_->counters, 2U * sizeof(std::uint32_t),
                            cudaMemcpyDeviceToHost, stream);
    if (error != cudaSuccess) return cuda_status(error, "download smoke counters");
    error = cudaEventRecord(impl_->telemetry_ready, stream);
    if (error != cudaSuccess) return cuda_status(error, "record smoke telemetry completion");
    Status status = completion.record(stream);
    if (status) impl_->telemetry_pending = true;
    return status;
}

Status Smoke::resolve_telemetry(SmokeTelemetry &output) noexcept {
    if (!impl_ || !impl_->telemetry_pending)
        return {StatusCode::invalid_argument, cudaSuccess, "smoke telemetry was not requested"};
    const cudaError_t ready = cudaEventQuery(impl_->telemetry_ready);
    if (ready == cudaErrorNotReady)
        return {StatusCode::invalid_argument, cudaSuccess, "smoke telemetry is not ready"};
    if (ready != cudaSuccess) return cuda_status(ready, "query smoke telemetry");
    SmokeTimings timings{};
    cudaError_t error = cudaEventElapsedTime(&timings.integrate_ms, impl_->begin, impl_->end);
    if (error != cudaSuccess) return cuda_status(error, "resolve smoke integration timing");
    if (impl_->couple_recorded) {
        error = cudaEventElapsedTime(&timings.couple_ms, impl_->couple_begin, impl_->couple_end);
        if (error != cudaSuccess) return cuda_status(error, "resolve smoke coupling timing");
    }
    impl_->statistics.respawn_count = *impl_->host_respawns;
    impl_->statistics.finite_failure_count = impl_->host_counters[0];
    impl_->statistics.maximum_speed = std::bit_cast<float>(impl_->host_counters[1]);
    impl_->telemetry = {timings, impl_->statistics};
    impl_->telemetry_pending = false;
    impl_->couple_recorded = false;
    output = impl_->telemetry;
    return {};
}

Status Smoke::collect_telemetry(cudaStream_t stream) noexcept {
    Completion completion;
    Status status = collect_telemetry_async(completion, stream);
    if (status) status = completion.wait();
    SmokeTelemetry output;
    return status ? resolve_telemetry(output) : status;
}

Status Smoke::reset(cudaStream_t stream) noexcept {
    if (!impl_) return {StatusCode::invalid_argument, cudaSuccess, "smoke is not initialized"};
    cudaMemsetAsync(impl_->respawns, 0, sizeof(unsigned long long), stream);
    cudaMemsetAsync(impl_->counters, 0, 2U * sizeof(std::uint32_t), stream);
    initialize_smoke<<<(impl_->options.particle_count + block_size - 1U) / block_size, block_size,
                       0, stream>>>(impl_->positions, impl_->velocities, impl_->ages,
                                    impl_->temperatures, impl_->colors, impl_->options);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return cuda_status(error, "launch smoke reset");
    error = cudaStreamSynchronize(stream);
    if (error != cudaSuccess) return cuda_status(error, "complete smoke reset");
    const std::size_t bytes = impl_->statistics.allocated_bytes;
    impl_->statistics = {};
    impl_->statistics.allocated_bytes = bytes;
    impl_->telemetry = {};
    impl_->telemetry.statistics.allocated_bytes = bytes;
    *impl_->host_respawns = 0U;
    impl_->host_counters[0] = 0U;
    impl_->host_counters[1] = 0U;
    impl_->telemetry_pending = false;
    impl_->couple_recorded = false;
    impl_->tick_index = 0U;
    return {};
}

bool Smoke::initialized() const noexcept { return impl_ != nullptr; }
SmokeOptions Smoke::options() const noexcept { return impl_ ? impl_->options : SmokeOptions{}; }
SmokeParticleView Smoke::particles() const noexcept {
    if (!impl_) return {};
    return {impl_->positions,
            impl_->velocities,
            impl_->ages,
            impl_->temperatures,
            impl_->options.particle_count,
            impl_->options.particle_radius,
            impl_->colors};
}
PointStateView Smoke::point_state() const noexcept {
    if (!impl_) return {};
    return {impl_->positions, impl_->velocities, nullptr, nullptr,
            impl_->options.particle_count, impl_->options.particle_radius, 0.0F, impl_->colors};
}
SmokeTelemetry Smoke::telemetry() const noexcept {
    return impl_ ? impl_->telemetry : SmokeTelemetry{};
}
SmokeStatistics Smoke::statistics() const noexcept {
    return impl_ ? impl_->statistics : SmokeStatistics{};
}

static_assert(FrameSolver<Smoke>);

} // namespace parallel_mater::physics
