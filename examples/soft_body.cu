// SPDX-License-Identifier: MIT
#include <parallel_mater/physics.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

__global__ void collide_ground(parallel_mater::physics::SoftBodyNodeView nodes) {
    const std::uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= nodes.node_count || (nodes.flags[id] & parallel_mater::physics::node_pinned) != 0U)
        return;
    const float penetration = nodes.node_radius - nodes.positions[id].y;
    if (penetration > 0.0F) {
        // Each thread owns one node, so this simple example needs no atomics.
        nodes.position_corrections[id].y += penetration;
    }
}

} // namespace

int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s converted-asset.msb\n", argv[0]);
        return 2;
    }

    parallel_mater::physics::SoftBodyOptions options;
    options.instance_origins[0] = {0.0F, 1.0F, 0.0F};

    parallel_mater::physics::SoftBody body;
    parallel_mater::Status status = body.initialize(argv[1], options);
    if (!status) {
        std::fprintf(stderr, "initialize: %s\n", status.message);
        return 1;
    }

    const float dt = options.timestep / static_cast<float>(options.substeps);
    constexpr std::uint32_t warmup_frames{10U};
    constexpr std::uint32_t measured_frames{120U};
    parallel_mater::physics::SoftBodyTimings timings;
    std::vector<float> samples;
    samples.reserve(measured_frames);
    for (std::uint32_t frame = 0U; frame < warmup_frames + measured_frames; ++frame) {
        status = body.begin_frame();
        for (std::uint32_t substep = 0U; status && substep < options.substeps; ++substep) {
            status = body.prepare_substep(dt, {0.0F, -9.81F, 0.0F});
            if (!status) break;
            const auto nodes = body.nodes();
            collide_ground<<<(nodes.node_count + 255U) / 256U, 256U>>>(nodes);
            if (cudaGetLastError() != cudaSuccess) return 1;
            status = body.finish_substep(dt, {0.0F, -9.81F, 0.0F});
        }
        if (status) status = body.finish_frame(timings);
        if (!status) {
            std::fprintf(stderr, "frame %u: %s\n", frame, status.message);
            return 1;
        }
        if (frame >= warmup_frames) samples.push_back(timings.gpu_total_ms());
    }

    std::sort(samples.begin(), samples.end());
    const auto percentile = [&samples](float fraction) {
        const std::size_t index =
            static_cast<std::size_t>(fraction * static_cast<float>(samples.size() - 1U));
        return samples[index];
    };
    const auto stats = body.statistics();
    const auto surface = body.surface();
    std::printf("%u nodes, %llu triangles, %u broken bonds, GPU p5/median/p95 "
                "%.3f/%.3f/%.3f ms, %.2f MiB\n",
                stats.node_count, static_cast<unsigned long long>(surface.mesh.triangle_count),
                stats.broken_bond_count, percentile(0.05F), percentile(0.50F), percentile(0.95F),
                static_cast<double>(body.allocated_bytes()) / (1024.0 * 1024.0));
    return stats.finite_failure_count == 0U ? 0 : 1;
}
