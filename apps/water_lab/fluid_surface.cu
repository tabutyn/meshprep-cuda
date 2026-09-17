// SPDX-License-Identifier: MIT
#include "fluid_surface.hpp"
#include "particle_cells.cuh"

#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>
#include <string>

namespace waterlab {
namespace {

void check(cudaError_t status, const char* operation)
{
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation)+": "+cudaGetErrorString(status));
    }
}

__device__ bool finite3(float3 p)
{
    return isfinite(p.x) && isfinite(p.y) && isfinite(p.z);
}

__global__ void initialize_grid(const meshprep::HierarchyNode* nodes, float radius,
    std::uint32_t resolution, FluidSurfaceGrid* descriptor, std::uint32_t* errors)
{
    const auto root = nodes[0];
    FluidSurfaceGrid grid{};
    grid.dimensions = make_uint3(resolution, resolution, resolution);
    grid.support_radius = radius;
    grid.minimum = make_float3(root.bounds_min.x-radius, root.bounds_min.y-radius, root.bounds_min.z-radius);
    const float scale = 1.0F/static_cast<float>(resolution-1U);
    grid.cell_size = make_float3((root.bounds_max.x-root.bounds_min.x+2*radius)*scale,
        (root.bounds_max.y-root.bounds_min.y+2*radius)*scale,
        (root.bounds_max.z-root.bounds_min.z+2*radius)*scale);
    if (!finite3(grid.minimum) || !finite3(grid.cell_size) ||
        !(grid.cell_size.x>0) || !(grid.cell_size.y>0) || !(grid.cell_size.z>0)) {
        grid.cell_size = {};
        atomicOr(errors, 1U);
    }
    *descriptor = grid;
}

__device__ float bounds_distance_squared(float3 p, const meshprep::HierarchyNode& node)
{
    const float x = fmaxf(fmaxf(node.bounds_min.x-p.x, 0), p.x-node.bounds_max.x);
    const float y = fmaxf(fmaxf(node.bounds_min.y-p.y, 0), p.y-node.bounds_max.y);
    const float z = fmaxf(fmaxf(node.bounds_min.z-p.z, 0), p.z-node.bounds_max.z);
    return x*x+y*y+z*z;
}

__device__ bool accumulate_field_particle(float3 sample, const float3* positions,
    std::uint32_t particle, std::uint32_t particle_count, float radius_squared,
    float& weights, float3& offset, std::uint32_t* errors)
{
    if (particle >= particle_count) { atomicOr(errors, 2U); return false; }
    const float3 q = positions[particle];
    if (!finite3(q)) { atomicOr(errors, 1U); return false; }
    const float3 d = make_float3(sample.x-q.x, sample.y-q.y, sample.z-q.z);
    const float distance_squared = d.x*d.x+d.y*d.y+d.z*d.z;
    if (distance_squared >= radius_squared) return true;
    const float a = 1.0F-distance_squared/radius_squared;
    const float w = a*a*a;
    weights += w;
    offset.x += w*d.x;
    offset.y += w*d.y;
    offset.z += w*d.z;
    return true;
}

__global__ void build_field(const float3* positions, std::uint32_t particle_count,
    const meshprep::HierarchyNode* nodes, const std::uint32_t* indices,
    std::uint32_t node_count, const FluidSurfaceGrid* descriptor,
    ParticleCellView cells, std::uint32_t sample_count, float* values,
    std::uint32_t* errors)
{
    const std::uint32_t i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= sample_count) return;
    const auto grid = *descriptor;
    if (!(grid.cell_size.x>0)) return;
    const std::uint32_t x = i%grid.dimensions.x;
    const std::uint32_t y = (i/grid.dimensions.x)%grid.dimensions.y;
    const std::uint32_t z = i/(grid.dimensions.x*grid.dimensions.y);
    const float3 p = make_float3(grid.minimum.x+x*grid.cell_size.x,
        grid.minimum.y+y*grid.cell_size.y, grid.minimum.z+z*grid.cell_size.z);
    const float radius_squared = grid.support_radius*grid.support_radius;
    float weights = 0;
    float3 offset{};
    if (cells.keys != nullptr) {
        const float inverse_cell_size = 1.0F/cells.cell_size;
        const int cell_x = __float2int_rd(p.x*inverse_cell_size);
        const int cell_y = __float2int_rd(p.y*inverse_cell_size);
        const int cell_z = __float2int_rd(p.z*inverse_cell_size);
        for (int dz=-1; dz<=1; ++dz) {
            for (int dy=-1; dy<=1; ++dy) {
                for (int dx=-1; dx<=1; ++dx) {
                    const std::uint64_t key = detail::particle_cell_key(
                        cell_x+dx, cell_y+dy, cell_z+dz);
                    const std::uint32_t first = detail::particle_cell_lower_bound(cells, key);
                    for (std::uint32_t item=first;
                         item<cells.particle_count && cells.keys[item]==key; ++item) {
                        if (!accumulate_field_particle(p, positions, cells.indices[item],
                                particle_count, radius_squared, weights, offset, errors)) return;
                    }
                }
            }
        }
    } else {
        std::uint32_t stack[128], pending = 1U;
        stack[0] = 0U;
        while (pending != 0U) {
            const std::uint32_t node_id = stack[--pending];
            if (node_id >= node_count) { atomicOr(errors, 2U); return; }
            const auto node = nodes[node_id];
            if (bounds_distance_squared(p, node) >= radius_squared) continue;
            if (node.is_leaf()) {
                if (node.first_primitive > particle_count ||
                    node.primitive_count > particle_count-node.first_primitive) {
                    atomicOr(errors, 2U); return;
                }
                for (std::uint32_t item=0; item<node.primitive_count; ++item) {
                    if (!accumulate_field_particle(p, positions,
                            indices[node.first_primitive+item], particle_count,
                            radius_squared, weights, offset, errors)) return;
                }
            } else {
                // An eight-way depth-18 tree needs at most 127 DFS stack entries.
                if (node.child_count>8U || node.first_child>node_count ||
                    node.child_count>node_count-node.first_child ||
                    pending+node.child_count>128U) {
                    atomicOr(errors, 2U); return;
                }
                for (std::uint32_t child=0; child<node.child_count; ++child) {
                    stack[pending++] = node.first_child+child;
                }
            }
        }
    }
    // Zhu-Bridson-style distance from the weighted local particle center.
    // Accumulate offsets to avoid cancellation from large world coordinates.
    float value = grid.support_radius;
    if (weights > 1.0e-10F) {
        offset.x /= weights;
        offset.y /= weights;
        offset.z /= weights;
        const float surface_radius = grid.support_radius/3.0F;
        value = sqrtf(offset.x*offset.x+offset.y*offset.y+offset.z*offset.z)-surface_radius;
    }
    if (!isfinite(value)) { atomicOr(errors, 1U); return; }
    values[i] = value;
}

} // namespace

FluidSurface::FluidSurface(std::uint32_t resolution) : resolution_(resolution)
{
    if (resolution < 2U || resolution > 256U) throw std::invalid_argument("fluid grid resolution must be 2..256");
    try {
        check(cudaMalloc(&values_, static_cast<std::size_t>(resolution)*resolution*resolution*sizeof(float)), "allocate fluid surface");
        check(cudaMalloc(&grid_, sizeof(FluidSurfaceGrid)), "allocate fluid grid descriptor");
        check(cudaMalloc(&errors_, sizeof(std::uint32_t)), "allocate fluid surface audit");
        check(cudaMemset(grid_, 0, sizeof(FluidSurfaceGrid)), "initialize fluid grid descriptor");
        check(cudaEventCreate(&begin_), "create surface begin event");
        check(cudaEventCreate(&end_), "create surface end event");
    } catch (...) {
        if (begin_) cudaEventDestroy(begin_);
        if (end_) cudaEventDestroy(end_);
        cudaFree(errors_);
        cudaFree(grid_);
        cudaFree(values_);
        throw;
    }
}

FluidSurface::~FluidSurface()
{
    cudaEventDestroy(begin_);
    cudaEventDestroy(end_);
    cudaFree(errors_);
    cudaFree(grid_);
    cudaFree(values_);
}

float FluidSurface::update(const float3* positions, std::uint32_t particle_count,
    const meshprep::Hierarchy& hierarchy, float support_radius, cudaStream_t stream,
    ParticleCellView cells)
{
    const auto stats = hierarchy.statistics();
    if (!positions || !particle_count || !hierarchy.nodes() || !hierarchy.primitive_indices() ||
        !stats.node_count || stats.max_depth>18U || !std::isfinite(support_radius) ||
        support_radius<=0 || !(support_radius*support_radius>0) ||
        !std::isfinite(support_radius*support_radius)) {
        throw std::invalid_argument("invalid fluid surface input or hierarchy depth above 18");
    }
    if (hierarchy.primitive_count() != particle_count) {
        throw std::runtime_error("fluid surface hierarchy/particle count mismatch");
    }
    if (!cells.empty() && (cells.indexed_positions != positions || !cells.keys || !cells.indices ||
            cells.particle_count != particle_count || !std::isfinite(cells.cell_size) ||
            cells.cell_size != support_radius)) {
        throw std::invalid_argument("fluid surface particle cell view mismatch");
    }
    check(cudaEventRecord(begin_, stream), "begin fluid surface");
    check(cudaMemsetAsync(errors_, 0, sizeof(std::uint32_t), stream),
        "clear fluid surface audit");
    initialize_grid<<<1,1,0,stream>>>(
        hierarchy.nodes(), support_radius, resolution_, grid_, errors_);
    check(cudaGetLastError(), "initialize fluid grid");
    const std::uint32_t samples = resolution_*resolution_*resolution_;
    build_field<<<(samples+255U)/256U,256,0,stream>>>(positions, particle_count,
        hierarchy.nodes(), hierarchy.primitive_indices(), stats.node_count, grid_, cells,
        samples, values_, errors_);
    check(cudaGetLastError(), "build fluid surface field");
    check(cudaEventRecord(end_, stream), "end fluid surface");
    std::uint32_t errors{};
    check(cudaMemcpyAsync(&errors, errors_, sizeof(errors), cudaMemcpyDeviceToHost, stream), "read surface audit");
    check(cudaStreamSynchronize(stream), "complete fluid surface");
    if (errors) throw std::runtime_error("fluid surface: nonfinite sample or invalid hierarchy traversal");
    float milliseconds{};
    check(cudaEventElapsedTime(&milliseconds, begin_, end_), "measure fluid surface");
    return milliseconds;
}

} // namespace waterlab
