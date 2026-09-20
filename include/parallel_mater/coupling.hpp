// SPDX-License-Identifier: MIT
#pragma once

#include <parallel_mater/frame.hpp>

#include <cuda_runtime_api.h>
#include <vector_types.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace parallel_mater::physics {

enum class ColliderShape : std::uint8_t { sphere, box, plane, capsule };

// Solver-neutral analytic collider. Dimensions mean radius for a sphere,
// half-extents for a box, (radius, half-height, 0) for a Y-axis capsule, and
// zero for a plane whose local +Y axis is its outward normal.
struct Collider {
    ColliderShape shape{ColliderShape::sphere};
    float3 position{};
    // Local-to-world quaternion (x, y, z, w); callers should keep it unit length.
    float4 orientation{0.0F, 0.0F, 0.0F, 1.0F};
    float3 linear_velocity{};
    float3 angular_velocity{};
    float3 dimensions{0.5F, 0.5F, 0.5F};
    float friction{0.15F};
    float restitution{};
    float contact_offset{};
    float4 paint_color{1.0F, 1.0F, 1.0F, 1.0F};
    float paint_amount{};
    std::uint32_t user_id{};
};

struct ColliderView {
    const Collider *data{}; // device memory
    std::uint32_t count{};
};

// Owns a device collider array. update_async copies the caller's host span;
// the source may be released once the supplied stream reaches completion.
class ColliderSet {
  public:
    ColliderSet() noexcept;
    ~ColliderSet();
    ColliderSet(ColliderSet &&) noexcept;
    ColliderSet &operator=(ColliderSet &&) noexcept;
    ColliderSet(const ColliderSet &) = delete;
    ColliderSet &operator=(const ColliderSet &) = delete;

    [[nodiscard]] Status reserve(std::uint32_t capacity) noexcept;
    [[nodiscard]] Status update_async(std::span<const Collider> colliders,
                                      cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status update(std::span<const Collider> colliders,
                                cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] ColliderView view() const noexcept;
    [[nodiscard]] std::uint32_t capacity() const noexcept;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// One coupling contribution. `order` is a caller-defined stable tie-breaker;
// batches sort by (point, order) before gathering, so results never depend on
// kernel scheduling or floating-point atomics.
struct ConstraintRecord {
    std::uint32_t point{};
    std::uint32_t order{};
    float3 impulse{};
    float3 position_correction{};
    float4 paint_color{};
    float paint_amount{};
};

struct ConstraintRecordView {
    const ConstraintRecord *data{}; // device memory
    std::uint32_t count{};
};

class ConstraintBatch {
  public:
    ConstraintBatch() noexcept;
    ~ConstraintBatch();
    ConstraintBatch(ConstraintBatch &&) noexcept;
    ConstraintBatch &operator=(ConstraintBatch &&) noexcept;
    ConstraintBatch(const ConstraintBatch &) = delete;
    ConstraintBatch &operator=(const ConstraintBatch &) = delete;

    [[nodiscard]] Status reserve(std::uint32_t capacity, cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status apply_async(ConstraintRecordView records, PointCouplingView target,
                                     cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status apply(ConstraintRecordView records, PointCouplingView target,
                               cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] std::uint32_t capacity() const noexcept;
    [[nodiscard]] std::size_t allocated_bytes() const noexcept;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Applies analytic solid contacts to a common point view. One thread owns one
// point and visits colliders in array order, avoiding contact atomics while
// combining projection, velocity response, friction, and paint transfer.
[[nodiscard]] Status apply_colliders_async(PointStateView points, ColliderView colliders,
                                           float timestep,
                                           cudaStream_t stream = nullptr) noexcept;
[[nodiscard]] Status apply_colliders(PointStateView points, ColliderView colliders,
                                     float timestep,
                                     cudaStream_t stream = nullptr) noexcept;

} // namespace parallel_mater::physics
