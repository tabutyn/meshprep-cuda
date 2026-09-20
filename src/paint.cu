// SPDX-License-Identifier: MIT
#include <parallel_mater/paint.hpp>

#include <cuda_runtime.h>

#include <cmath>
#include <limits>
#include <memory>
#include <new>
#include <utility>

namespace parallel_mater::physics {
namespace {

constexpr std::uint32_t block_size = 256U;

[[nodiscard]] constexpr Status invalid(const char *message) noexcept {
    return {StatusCode::invalid_argument, cudaSuccess, message};
}

[[nodiscard]] Status cuda_status(cudaError_t error, const char *message) noexcept {
    return error == cudaSuccess
               ? Status{}
               : Status{error == cudaErrorMemoryAllocation ? StatusCode::allocation_failure
                                                           : StatusCode::cuda_failure,
                        error, message};
}

[[nodiscard]] bool finite(float value) noexcept { return std::isfinite(value); }
[[nodiscard]] bool finite(float4 value) noexcept {
    return finite(value.x) && finite(value.y) && finite(value.z) && finite(value.w);
}

__device__ float clamp01(float value) { return fminf(1.0F, fmaxf(0.0F, value)); }

__global__ void clear_paint(float4 *colors, std::uint32_t count, float4 value) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) colors[index] = value;
}

__global__ void apply_stamps(float4 *colors, std::uint32_t width, std::uint32_t height,
                             bool wrap_u, PaintStampView stamps) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t count = width * height;
    if (index >= count) return;
    const std::uint32_t x = index % width;
    const std::uint32_t y = index / width;
    const float u = (static_cast<float>(x) + 0.5F) / static_cast<float>(width);
    const float v = (static_cast<float>(y) + 0.5F) / static_cast<float>(height);
    const float aspect_u = static_cast<float>(width) /
                           static_cast<float>(width < height ? width : height);
    const float aspect_v = static_cast<float>(height) /
                           static_cast<float>(width < height ? width : height);
    float4 result = colors[index];
    for (std::uint32_t stamp_index = 0U; stamp_index < stamps.count; ++stamp_index) {
        const PaintStamp stamp = stamps.data[stamp_index];
        if (!isfinite(stamp.uv.x) || !isfinite(stamp.uv.y) || !isfinite(stamp.radius) ||
            stamp.radius <= 0.0F || !isfinite(stamp.opacity) || stamp.opacity <= 0.0F ||
            !isfinite(stamp.color.x) || !isfinite(stamp.color.y) ||
            !isfinite(stamp.color.z) || !isfinite(stamp.color.w))
            continue;
        float du = fabsf(u - stamp.uv.x);
        if (wrap_u) du = fminf(du, 1.0F - du);
        const float dv = v - stamp.uv.y;
        const float distance = sqrtf(du * du * aspect_u * aspect_u +
                                     dv * dv * aspect_v * aspect_v);
        if (distance > stamp.radius) continue;
        const float falloff = stamp.radius > 1.0e-8F
                                  ? clamp01(1.0F - distance / stamp.radius)
                                  : (distance == 0.0F ? 1.0F : 0.0F);
        const float alpha = clamp01(stamp.opacity * falloff);
        result.x += (stamp.color.x - result.x) * alpha;
        result.y += (stamp.color.y - result.y) * alpha;
        result.z += (stamp.color.z - result.z) * alpha;
        result.w += (stamp.color.w - result.w) * alpha;
    }
    colors[index] = result;
}

} // namespace

struct PaintSurface::Impl {
    ~Impl() { cudaFree(colors); }
    PaintSurfaceOptions options{};
    float4 *colors{};
};

PaintSurface::PaintSurface() noexcept = default;
PaintSurface::~PaintSurface() = default;
PaintSurface::PaintSurface(PaintSurface &&) noexcept = default;
PaintSurface &PaintSurface::operator=(PaintSurface &&) noexcept = default;

Status PaintSurface::create(PaintSurfaceOptions options, PaintSurface &output,
                            cudaStream_t stream) noexcept {
    return output.initialize(options, stream);
}

Status PaintSurface::initialize(PaintSurfaceOptions options, cudaStream_t stream) noexcept {
    if (options.width == 0U || options.height == 0U || !finite(options.clear_color))
        return invalid("invalid paint surface options");
    const std::size_t count = static_cast<std::size_t>(options.width) * options.height;
    if (count > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()))
        return {StatusCode::unsupported_size, cudaSuccess, "paint surface is too large"};
    auto replacement = std::unique_ptr<Impl>(new (std::nothrow) Impl);
    if (!replacement)
        return {StatusCode::allocation_failure, cudaErrorMemoryAllocation,
                "could not allocate paint surface owner"};
    replacement->options = options;
    Status status = cuda_status(cudaMalloc(&replacement->colors, count * sizeof(float4)),
                                "could not allocate paint surface");
    if (!status) return status;
    const auto device_count = static_cast<std::uint32_t>(count);
    clear_paint<<<(device_count + block_size - 1U) / block_size, block_size, 0, stream>>>(
        replacement->colors, device_count, options.clear_color);
    status = cuda_status(cudaGetLastError(), "could not initialize paint surface");
    if (status)
        status = cuda_status(cudaStreamSynchronize(stream),
                             "could not complete paint surface initialization");
    if (!status) return status;
    impl_ = std::move(replacement);
    return {};
}

Status PaintSurface::clear_async(cudaStream_t stream) noexcept {
    if (!impl_) return invalid("paint surface is not initialized");
    const std::uint32_t count = impl_->options.width * impl_->options.height;
    clear_paint<<<(count + block_size - 1U) / block_size, block_size, 0, stream>>>(
        impl_->colors, count, impl_->options.clear_color);
    return cuda_status(cudaGetLastError(), "could not clear paint surface");
}

Status PaintSurface::clear(cudaStream_t stream) noexcept {
    Status status = clear_async(stream);
    return status ? cuda_status(cudaStreamSynchronize(stream),
                                "could not complete paint surface clear")
                  : status;
}

Status PaintSurface::apply_async(PaintStampView stamps, cudaStream_t stream) noexcept {
    if (!impl_) return invalid("paint surface is not initialized");
    if (stamps.count == 0U) return {};
    if (stamps.data == nullptr) return invalid("paint stamp storage is null");
    const std::uint32_t count = impl_->options.width * impl_->options.height;
    apply_stamps<<<(count + block_size - 1U) / block_size, block_size, 0, stream>>>(
        impl_->colors, impl_->options.width, impl_->options.height, impl_->options.wrap_u, stamps);
    return cuda_status(cudaGetLastError(), "could not apply paint stamps");
}

Status PaintSurface::apply(PaintStampView stamps, cudaStream_t stream) noexcept {
    Status status = apply_async(stamps, stream);
    return status ? cuda_status(cudaStreamSynchronize(stream),
                                "could not complete paint stamps")
                  : status;
}

bool PaintSurface::initialized() const noexcept { return impl_ != nullptr; }
PaintSurfaceOptions PaintSurface::options() const noexcept {
    return impl_ ? impl_->options : PaintSurfaceOptions{};
}
PaintSurfaceView PaintSurface::view() const noexcept {
    return impl_ ? PaintSurfaceView{impl_->colors, impl_->options.width, impl_->options.height}
                 : PaintSurfaceView{};
}
std::size_t PaintSurface::allocated_bytes() const noexcept {
    return impl_ ? static_cast<std::size_t>(impl_->options.width) * impl_->options.height *
                       sizeof(float4)
                 : 0U;
}

} // namespace parallel_mater::physics
