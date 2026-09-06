// SPDX-License-Identifier: MIT
#include <meshprep/meshprep.hpp>

#include <cuda_runtime.h>

#include <cstdio>

int main()
{
    const float3 positions[] = {
        make_float3(0.0F, 0.0F, 0.0F),
        make_float3(1.0F, 0.0F, 0.0F),
        make_float3(0.0F, 1.0F, 0.0F),
    };
    const uint3 triangles[] = {make_uint3(0, 1, 2)};
    float3* device_positions = nullptr;
    uint3* device_triangles = nullptr;
    cudaMalloc(&device_positions, sizeof(positions));
    cudaMalloc(&device_triangles, sizeof(triangles));
    cudaMemcpy(device_positions, positions, sizeof(positions), cudaMemcpyHostToDevice);
    cudaMemcpy(device_triangles, triangles, sizeof(triangles), cudaMemcpyHostToDevice);

    meshprep::Workspace workspace;
    meshprep::NormalOutput normals;
    meshprep::Hierarchy hierarchy;
    const meshprep::DeviceMeshView mesh{device_positions, 3, device_triangles, 1};
    const meshprep::Status normal_status =
        meshprep::compute_normals(mesh, {}, workspace, normals);
    const meshprep::Status hierarchy_status =
        meshprep::build_hierarchy(mesh, {}, workspace, hierarchy);
    if (!normal_status || !hierarchy_status) {
        std::fprintf(
            stderr,
            "meshprep failed: %s / %s\n",
            normal_status.message,
            hierarchy_status.message);
        cudaFree(device_triangles);
        cudaFree(device_positions);
        return 1;
    }
    std::printf(
        "face normals: 1, vertex normals: %u, hierarchy nodes: %u\n",
        normals.statistics().vertex_normal_count,
        hierarchy.statistics().node_count);
    cudaFree(device_triangles);
    cudaFree(device_positions);
    return 0;
}
