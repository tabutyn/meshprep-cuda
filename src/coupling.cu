// SPDX-License-Identifier: MIT
#include <parallel_mater/coupling.hpp>

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <cmath>
#include <memory>
#include <new>
#include <utility>

namespace parallel_mater::physics {
namespace {

constexpr std::uint32_t block_size = 256U;

Status invalid(const char *message) noexcept {
    return {StatusCode::invalid_argument, cudaSuccess, message};
}

Status cuda_status(cudaError_t error, const char *message) noexcept {
    return error == cudaSuccess
               ? Status{}
               : Status{error == cudaErrorMemoryAllocation ? StatusCode::allocation_failure
                                                           : StatusCode::cuda_failure,
                        error, message};
}

bool finite(float value) { return std::isfinite(value); }
bool finite(float3 value) { return finite(value.x) && finite(value.y) && finite(value.z); }
bool finite(float4 value) {
    return finite(value.x) && finite(value.y) && finite(value.z) && finite(value.w);
}

bool valid(const Collider &value) {
    const float orientation_length_squared =
        value.orientation.x * value.orientation.x + value.orientation.y * value.orientation.y +
        value.orientation.z * value.orientation.z + value.orientation.w * value.orientation.w;
    bool valid_dimensions{};
    switch (value.shape) {
    case ColliderShape::sphere:
        valid_dimensions = value.dimensions.x > 0.0F;
        break;
    case ColliderShape::box:
        valid_dimensions =
            value.dimensions.x > 0.0F && value.dimensions.y > 0.0F && value.dimensions.z > 0.0F;
        break;
    case ColliderShape::plane:
        valid_dimensions = true;
        break;
    case ColliderShape::capsule:
        valid_dimensions = value.dimensions.x > 0.0F && value.dimensions.y >= 0.0F;
        break;
    default:
        return false;
    }
    return finite(value.position) && finite(value.orientation) &&
           finite(orientation_length_squared) && orientation_length_squared > 1.0e-12F &&
           finite(value.linear_velocity) && finite(value.angular_velocity) &&
           finite(value.dimensions) && value.dimensions.x >= 0.0F && value.dimensions.y >= 0.0F &&
           value.dimensions.z >= 0.0F && valid_dimensions && finite(value.friction) &&
           value.friction >= 0.0F && finite(value.restitution) && value.restitution >= 0.0F &&
           finite(value.contact_offset) && value.contact_offset >= 0.0F;
}

__global__ void prepare_records(const ConstraintRecord *input, std::uint64_t *keys,
                                ConstraintRecord *values, std::uint32_t count) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const ConstraintRecord record = input[index];
    keys[index] = (static_cast<std::uint64_t>(record.point) << 32U) |
                  static_cast<std::uint64_t>(record.order);
    values[index] = record;
}

__device__ std::uint32_t lower_bound_point(const std::uint64_t *keys, std::uint32_t count,
                                           std::uint32_t point) {
    const std::uint64_t key = static_cast<std::uint64_t>(point) << 32U;
    std::uint32_t first = 0U;
    std::uint32_t last = count;
    while (first < last) {
        const std::uint32_t middle = first + (last - first) / 2U;
        if (keys[middle] < key)
            first = middle + 1U;
        else
            last = middle;
    }
    return first;
}

__global__ void gather_records(const std::uint64_t *keys, const ConstraintRecord *records,
                               std::uint32_t record_count, PointCouplingView target) {
    const std::uint32_t point = blockIdx.x * blockDim.x + threadIdx.x;
    if (point >= target.count) return;
    float3 impulse{};
    float3 correction{};
    std::uint32_t record = lower_bound_point(keys, record_count, point);
    while (record < record_count && records[record].point == point) {
        const ConstraintRecord value = records[record++];
        impulse.x += value.impulse.x;
        impulse.y += value.impulse.y;
        impulse.z += value.impulse.z;
        correction.x += value.position_correction.x;
        correction.y += value.position_correction.y;
        correction.z += value.position_correction.z;
    }
    if (target.external_impulses != nullptr) {
        target.external_impulses[point].x += impulse.x;
        target.external_impulses[point].y += impulse.y;
        target.external_impulses[point].z += impulse.z;
    }
    if (target.position_corrections != nullptr) {
        target.position_corrections[point].x += correction.x;
        target.position_corrections[point].y += correction.y;
        target.position_corrections[point].z += correction.z;
    }
}

} // namespace

struct ColliderSet::Impl {
    Collider *data{};
    std::uint32_t size{};
    std::uint32_t capacity{};
    ~Impl() { cudaFree(data); }
};

ColliderSet::ColliderSet() noexcept = default;
ColliderSet::~ColliderSet() = default;
ColliderSet::ColliderSet(ColliderSet &&) noexcept = default;
ColliderSet &ColliderSet::operator=(ColliderSet &&) noexcept = default;

Status ColliderSet::reserve(std::uint32_t capacity) noexcept {
    if (capacity == 0U) return invalid("collider capacity must be nonzero");
    if (impl_ && impl_->capacity >= capacity) return {};
    try {
        auto replacement = std::make_unique<Impl>();
        Status status = cuda_status(
            cudaMalloc(&replacement->data, static_cast<std::size_t>(capacity) * sizeof(Collider)),
            "could not allocate collider set");
        if (!status) return status;
        replacement->capacity = capacity;
        impl_ = std::move(replacement);
        return {};
    } catch (const std::bad_alloc &) {
        return {StatusCode::allocation_failure, cudaErrorMemoryAllocation,
                "could not allocate collider host state"};
    }
}

Status ColliderSet::update_async(std::span<const Collider> colliders,
                                 cudaStream_t stream) noexcept {
    for (const Collider &collider : colliders) {
        if (!valid(collider)) return invalid("invalid collider");
    }
    if (colliders.empty()) {
        if (impl_) impl_->size = 0U;
        return {};
    }
    if (!impl_ || impl_->capacity < colliders.size()) {
        Status status = reserve(static_cast<std::uint32_t>(colliders.size()));
        if (!status) return status;
    }
    Status status =
        cuda_status(cudaMemcpyAsync(impl_->data, colliders.data(), colliders.size_bytes(),
                                    cudaMemcpyHostToDevice, stream),
                    "could not upload colliders");
    if (status) impl_->size = static_cast<std::uint32_t>(colliders.size());
    return status;
}

Status ColliderSet::update(std::span<const Collider> colliders, cudaStream_t stream) noexcept {
    Status status = update_async(colliders, stream);
    return status ? cuda_status(cudaStreamSynchronize(stream), "could not complete collider upload")
                  : status;
}

ColliderView ColliderSet::view() const noexcept {
    return impl_ ? ColliderView{impl_->data, impl_->size} : ColliderView{};
}

std::uint32_t ColliderSet::capacity() const noexcept { return impl_ ? impl_->capacity : 0U; }

struct ConstraintBatch::Impl {
    std::uint64_t *keys_a{};
    std::uint64_t *keys_b{};
    ConstraintRecord *records_a{};
    ConstraintRecord *records_b{};
    void *sort_storage{};
    std::size_t sort_storage_bytes{};
    std::uint32_t capacity{};
    ~Impl() {
        cudaFree(sort_storage);
        cudaFree(records_b);
        cudaFree(records_a);
        cudaFree(keys_b);
        cudaFree(keys_a);
    }
};

ConstraintBatch::ConstraintBatch() noexcept = default;
ConstraintBatch::~ConstraintBatch() = default;
ConstraintBatch::ConstraintBatch(ConstraintBatch &&) noexcept = default;
ConstraintBatch &ConstraintBatch::operator=(ConstraintBatch &&) noexcept = default;

Status ConstraintBatch::reserve(std::uint32_t capacity, cudaStream_t stream) noexcept {
    if (capacity == 0U) return invalid("constraint capacity must be nonzero");
    if (impl_ && impl_->capacity >= capacity) return {};
    try {
        auto replacement = std::make_unique<Impl>();
        const std::size_t key_bytes = static_cast<std::size_t>(capacity) * sizeof(std::uint64_t);
        const std::size_t value_bytes =
            static_cast<std::size_t>(capacity) * sizeof(ConstraintRecord);
        Status status = cuda_status(cudaMalloc(&replacement->keys_a, key_bytes),
                                    "could not allocate constraint keys");
        if (!status) return status;
        if (!(status = cuda_status(cudaMalloc(&replacement->keys_b, key_bytes),
                                   "could not allocate sorted constraint keys")))
            return status;
        if (!(status = cuda_status(cudaMalloc(&replacement->records_a, value_bytes),
                                   "could not allocate constraint records")))
            return status;
        if (!(status = cuda_status(cudaMalloc(&replacement->records_b, value_bytes),
                                   "could not allocate sorted constraint records")))
            return status;
        cudaError_t error = cub::DeviceRadixSort::SortPairs(
            nullptr, replacement->sort_storage_bytes, replacement->keys_a, replacement->keys_b,
            replacement->records_a, replacement->records_b, static_cast<int>(capacity), 0, 64,
            stream);
        if (!(status = cuda_status(error, "could not size constraint sort workspace")))
            return status;
        if (!(status = cuda_status(
                  cudaMalloc(&replacement->sort_storage, replacement->sort_storage_bytes),
                  "could not allocate constraint sort workspace")))
            return status;
        replacement->capacity = capacity;
        impl_ = std::move(replacement);
        return {};
    } catch (const std::bad_alloc &) {
        return {StatusCode::allocation_failure, cudaErrorMemoryAllocation,
                "could not allocate constraint host state"};
    }
}

Status ConstraintBatch::apply_async(ConstraintRecordView records, PointCouplingView target,
                                    cudaStream_t stream) noexcept {
    if (records.count == 0U) return {};
    if (records.data == nullptr || target.count == 0U || target.positions == nullptr ||
        target.velocities == nullptr ||
        (target.external_impulses == nullptr && target.position_corrections == nullptr)) {
        return invalid("invalid constraint batch view");
    }
    if (!impl_ || impl_->capacity < records.count) {
        Status status = reserve(records.count, stream);
        if (!status) return status;
    }
    const std::uint32_t record_blocks = (records.count + block_size - 1U) / block_size;
    prepare_records<<<record_blocks, block_size, 0, stream>>>(records.data, impl_->keys_a,
                                                              impl_->records_a, records.count);
    Status status = cuda_status(cudaGetLastError(), "could not prepare deterministic constraints");
    if (!status) return status;
    status = cuda_status(cub::DeviceRadixSort::SortPairs(
                             impl_->sort_storage, impl_->sort_storage_bytes, impl_->keys_a,
                             impl_->keys_b, impl_->records_a, impl_->records_b,
                             static_cast<int>(records.count), 0, 64, stream),
                         "could not sort deterministic constraints");
    if (!status) return status;
    gather_records<<<(target.count + block_size - 1U) / block_size, block_size, 0, stream>>>(
        impl_->keys_b, impl_->records_b, records.count, target);
    return cuda_status(cudaGetLastError(), "could not gather deterministic constraints");
}

Status ConstraintBatch::apply(ConstraintRecordView records, PointCouplingView target,
                              cudaStream_t stream) noexcept {
    Status status = apply_async(records, target, stream);
    return status ? cuda_status(cudaStreamSynchronize(stream),
                                "could not complete deterministic constraints")
                  : status;
}

std::uint32_t ConstraintBatch::capacity() const noexcept { return impl_ ? impl_->capacity : 0U; }

std::size_t ConstraintBatch::allocated_bytes() const noexcept {
    return impl_ ? impl_->sort_storage_bytes +
                       static_cast<std::size_t>(impl_->capacity) *
                           (2U * sizeof(std::uint64_t) + 2U * sizeof(ConstraintRecord))
                 : 0U;
}

} // namespace parallel_mater::physics
