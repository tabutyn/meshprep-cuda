// SPDX-License-Identifier: MIT
#include <parallel_mater/geometry.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <utility>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* expression, int line)
{
    if (condition) return;
    std::fprintf(stderr, "FAIL line %d: %s\n", line, expression);
    ++failures;
}

#define CHECK(expression) check(static_cast<bool>(expression), #expression, __LINE__)

void cuda_check(cudaError_t error, const char* operation)
{
    if (error == cudaSuccess) return;
    std::fprintf(stderr, "CUDA failure in %s: %s\n", operation, cudaGetErrorString(error));
    std::exit(2);
}

template <typename T>
class DeviceArray {
public:
    DeviceArray() = default;
    explicit DeviceArray(const std::vector<T>& source) { upload(source); }
    ~DeviceArray() { cudaFree(data_); }
    DeviceArray(DeviceArray&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)), count_(std::exchange(other.count_, 0))
    {
    }
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;

    void upload(const std::vector<T>& source)
    {
        cudaFree(data_);
        data_ = nullptr;
        count_ = source.size();
        if (source.empty()) return;
        cuda_check(cudaMalloc(&data_, source.size() * sizeof(T)), "cudaMalloc");
        cuda_check(
            cudaMemcpy(data_, source.data(), source.size() * sizeof(T), cudaMemcpyHostToDevice),
            "cudaMemcpy host to device");
    }

    [[nodiscard]] T* data() const { return data_; }

private:
    T* data_{};
    std::size_t count_{};
};

template <typename T>
std::vector<T> download(const T* source, std::size_t count)
{
    std::vector<T> result(count);
    if (count > 0) {
        cuda_check(
            cudaMemcpy(result.data(), source, count * sizeof(T), cudaMemcpyDeviceToHost),
            "cudaMemcpy device to host");
    }
    return result;
}

bool near(float a, float b, float tolerance = 1.0e-5F)
{
    return std::fabs(a - b) <= tolerance;
}

bool near(float3 a, float3 b, float tolerance = 1.0e-5F)
{
    return near(a.x, b.x, tolerance) && near(a.y, b.y, tolerance) &&
        near(a.z, b.z, tolerance);
}

struct HostMesh {
    std::vector<float3> positions;
    std::vector<uint3> triangles;
};

struct DeviceMesh {
    explicit DeviceMesh(const HostMesh& host)
        : positions(host.positions),
          triangles(host.triangles),
          position_count(host.positions.size()),
          triangle_count(host.triangles.size())
    {
    }

    [[nodiscard]] parallel_mater::DeviceMeshView view() const
    {
        return {positions.data(), position_count, triangles.data(), triangle_count};
    }

    DeviceArray<float3> positions;
    DeviceArray<uint3> triangles;
    std::uint64_t position_count{};
    std::uint64_t triangle_count{};
};

HostMesh quad_mesh()
{
    return {
        {
            make_float3(0.0F, 0.0F, 0.0F),
            make_float3(1.0F, 0.0F, 0.0F),
            make_float3(1.0F, 1.0F, 0.0F),
            make_float3(0.0F, 1.0F, 0.0F),
        },
        {make_uint3(0, 1, 2), make_uint3(0, 2, 3)},
    };
}

float3 cpu_face_normal(const HostMesh& mesh, std::size_t triangle_index)
{
    const uint3 triangle = mesh.triangles[triangle_index];
    const float3 a = mesh.positions[triangle.x];
    const float3 b = mesh.positions[triangle.y];
    const float3 c = mesh.positions[triangle.z];
    const float3 ab = make_float3(b.x - a.x, b.y - a.y, b.z - a.z);
    const float3 ac = make_float3(c.x - a.x, c.y - a.y, c.z - a.z);
    const float3 cross = make_float3(
        ab.y * ac.z - ab.z * ac.y,
        ab.z * ac.x - ab.x * ac.z,
        ab.x * ac.y - ab.y * ac.x);
    const float length = std::sqrt(cross.x * cross.x + cross.y * cross.y + cross.z * cross.z);
    if (length == 0.0F) return make_float3(0.0F, 0.0F, 0.0F);
    return make_float3(cross.x / length, cross.y / length, cross.z / length);
}

void test_empty_input()
{
    parallel_mater::Workspace workspace;
    parallel_mater::NormalOutput normals;
    parallel_mater::Hierarchy hierarchy;
    CHECK(
        parallel_mater::compute_normals({}, {}, workspace, normals).code ==
        parallel_mater::StatusCode::invalid_argument);
    CHECK(
        parallel_mater::build_hierarchy(parallel_mater::DeviceMeshView{}, {}, workspace, hierarchy).code ==
        parallel_mater::StatusCode::invalid_argument);
    CHECK(
        parallel_mater::build_hierarchy(parallel_mater::DeviceAabbView{}, {}, workspace, hierarchy).code ==
        parallel_mater::StatusCode::invalid_argument);
}

void test_triangle_and_degenerate_normals()
{
    const HostMesh host{
        {
            make_float3(0.0F, 0.0F, 0.0F),
            make_float3(1.0F, 0.0F, 0.0F),
            make_float3(0.0F, 1.0F, 0.0F),
        },
        {make_uint3(0, 1, 2), make_uint3(0, 0, 0)},
    };
    const DeviceMesh device(host);
    parallel_mater::Workspace workspace;
    parallel_mater::NormalOutput output;
    const parallel_mater::Status status =
        parallel_mater::compute_normals(device.view(), {}, workspace, output);
    CHECK(status.ok());
    CHECK(output.statistics().degenerate_triangle_count == 1);
    const auto faces = download(output.face_normals(), 2);
    CHECK(near(faces[0], make_float3(0.0F, 0.0F, 1.0F)));
    CHECK(near(faces[1], make_float3(0.0F, 0.0F, 0.0F)));
}

void test_smooth_and_sharp_quad()
{
    const HostMesh host = quad_mesh();
    const DeviceMesh device(host);
    parallel_mater::Workspace workspace;
    parallel_mater::NormalOutput smooth;
    parallel_mater::Status status = parallel_mater::compute_normals(device.view(), {}, workspace, smooth);
    CHECK(status.ok());
    CHECK(smooth.statistics().vertex_normal_count == 4);
    const auto smooth_normals = download(
        smooth.vertex_normals(), smooth.statistics().vertex_normal_count);
    for (const float3 normal : smooth_normals) {
        CHECK(near(normal, make_float3(0.0F, 0.0F, 1.0F)));
    }
    const auto smooth_indices = download(smooth.corner_normal_indices(), 6);
    const std::vector<std::uint32_t> expected{0, 1, 2, 0, 2, 3};
    CHECK(smooth_indices == expected);

    const std::vector<uint2> host_edges{make_uint2(2, 0), make_uint2(0, 2)};
    const DeviceArray<uint2> device_edges(host_edges);
    parallel_mater::NormalOutput sharp;
    status = parallel_mater::compute_normals(
        device.view(), {device_edges.data(), host_edges.size()}, workspace, sharp);
    CHECK(status.ok());
    CHECK(sharp.statistics().vertex_normal_count == 6);
    const auto sharp_indices = download(sharp.corner_normal_indices(), 6);
    CHECK(sharp_indices[0] != sharp_indices[3]);
    CHECK(sharp_indices[2] != sharp_indices[4]);

    const auto reference_indices = sharp_indices;
    for (int iteration = 0; iteration < 100; ++iteration) {
        status = parallel_mater::compute_normals(
            device.view(), {device_edges.data(), host_edges.size()}, workspace, sharp);
        CHECK(status.ok());
        CHECK(download(sharp.corner_normal_indices(), 6) == reference_indices);
    }
}

void test_normal_topology_fixtures()
{
    const HostMesh cube{
        {
            make_float3(-1, -1, -1), make_float3(1, -1, -1),
            make_float3(1, 1, -1), make_float3(-1, 1, -1),
            make_float3(-1, -1, 1), make_float3(1, -1, 1),
            make_float3(1, 1, 1), make_float3(-1, 1, 1),
        },
        {
            make_uint3(0, 2, 1), make_uint3(0, 3, 2),
            make_uint3(4, 5, 6), make_uint3(4, 6, 7),
            make_uint3(0, 1, 5), make_uint3(0, 5, 4),
            make_uint3(1, 2, 6), make_uint3(1, 6, 5),
            make_uint3(2, 3, 7), make_uint3(2, 7, 6),
            make_uint3(3, 0, 4), make_uint3(3, 4, 7),
        },
    };
    const std::vector<uint2> cube_edges{
        make_uint2(0, 1), make_uint2(1, 2), make_uint2(2, 3), make_uint2(0, 3),
        make_uint2(4, 5), make_uint2(5, 6), make_uint2(6, 7), make_uint2(4, 7),
        make_uint2(0, 4), make_uint2(1, 5), make_uint2(2, 6), make_uint2(3, 7),
    };
    const DeviceMesh cube_device(cube);
    const DeviceArray<uint2> cube_edge_device(cube_edges);
    parallel_mater::Workspace workspace;
    parallel_mater::NormalOutput output;
    CHECK(parallel_mater::compute_normals(
              cube_device.view(), {cube_edge_device.data(), cube_edges.size()}, workspace, output)
              .ok());
    CHECK(output.statistics().vertex_normal_count == 24);

    const HostMesh disconnected_fans{
        {
            make_float3(0, 0, 0), make_float3(1, 0, 0), make_float3(0, 1, 0),
            make_float3(0, 0, 1), make_float3(0, 1, 1),
        },
        {make_uint3(0, 1, 2), make_uint3(0, 3, 4)},
    };
    const DeviceMesh disconnected_device(disconnected_fans);
    CHECK(parallel_mater::compute_normals(disconnected_device.view(), {}, workspace, output).ok());
    CHECK(output.statistics().vertex_normal_count == 6);

    const HostMesh non_manifold{
        {
            make_float3(0, 0, 0), make_float3(1, 0, 0), make_float3(0, 1, 0),
            make_float3(0, 0, 1), make_float3(0, -1, 0),
        },
        {make_uint3(0, 1, 2), make_uint3(1, 0, 3), make_uint3(0, 1, 4)},
    };
    const DeviceMesh non_manifold_device(non_manifold);
    CHECK(parallel_mater::compute_normals(non_manifold_device.view(), {}, workspace, output).ok());
    CHECK(output.statistics().vertex_normal_count == 5);
}

void test_invalid_meshes()
{
    HostMesh bad_index = quad_mesh();
    bad_index.triangles[0].z = 99;
    const DeviceMesh device_bad_index(bad_index);
    parallel_mater::Workspace workspace;
    parallel_mater::NormalOutput normals;
    CHECK(
        parallel_mater::compute_normals(device_bad_index.view(), {}, workspace, normals).code ==
        parallel_mater::StatusCode::invalid_mesh);

    HostMesh nonfinite = quad_mesh();
    nonfinite.positions[0].x = std::numeric_limits<float>::quiet_NaN();
    const DeviceMesh device_nonfinite(nonfinite);
    parallel_mater::Hierarchy hierarchy;
    CHECK(
        parallel_mater::build_hierarchy(device_nonfinite.view(), {}, workspace, hierarchy).code ==
        parallel_mater::StatusCode::invalid_mesh);

    const std::vector<uint2> self_edge{make_uint2(0, 0)};
    const DeviceArray<uint2> device_edge(self_edge);
    const DeviceMesh valid_device(quad_mesh());
    CHECK(
        parallel_mater::compute_normals(
            valid_device.view(), {device_edge.data(), 1}, workspace, normals).code ==
        parallel_mater::StatusCode::invalid_mesh);

    const std::vector<uint2> out_of_range_edge{make_uint2(0, 99)};
    const DeviceArray<uint2> out_of_range_edge_device(out_of_range_edge);
    CHECK(
        parallel_mater::compute_normals(
            valid_device.view(), {out_of_range_edge_device.data(), 1}, workspace, normals).code ==
        parallel_mater::StatusCode::invalid_mesh);

    parallel_mater::Hierarchy invalid_options_output;
    CHECK(
        parallel_mater::build_hierarchy(
            valid_device.view(), parallel_mater::HierarchyOptions{0}, workspace, invalid_options_output)
            .code == parallel_mater::StatusCode::invalid_argument);
}

bool same_node(const parallel_mater::HierarchyNode& a, const parallel_mater::HierarchyNode& b)
{
    return std::bit_cast<std::uint32_t>(a.bounds_min.x) ==
            std::bit_cast<std::uint32_t>(b.bounds_min.x) &&
        std::bit_cast<std::uint32_t>(a.bounds_min.y) ==
            std::bit_cast<std::uint32_t>(b.bounds_min.y) &&
        std::bit_cast<std::uint32_t>(a.bounds_min.z) ==
            std::bit_cast<std::uint32_t>(b.bounds_min.z) &&
        std::bit_cast<std::uint32_t>(a.bounds_max.x) ==
            std::bit_cast<std::uint32_t>(b.bounds_max.x) &&
        std::bit_cast<std::uint32_t>(a.bounds_max.y) ==
            std::bit_cast<std::uint32_t>(b.bounds_max.y) &&
        std::bit_cast<std::uint32_t>(a.bounds_max.z) ==
            std::bit_cast<std::uint32_t>(b.bounds_max.z) &&
        a.first_child == b.first_child && a.child_count == b.child_count &&
        a.first_primitive == b.first_primitive && a.primitive_count == b.primitive_count;
}

bool bounds_contain(float3 minimum, float3 maximum, float3 point)
{
    constexpr float tolerance = 1.0e-5F;
    return point.x >= minimum.x - tolerance && point.x <= maximum.x + tolerance &&
        point.y >= minimum.y - tolerance && point.y <= maximum.y + tolerance &&
        point.z >= minimum.z - tolerance && point.z <= maximum.z + tolerance;
}

void validate_hierarchy(
    const HostMesh& mesh,
    const std::vector<parallel_mater::HierarchyNode>& nodes,
    const std::vector<std::uint32_t>& permutation,
    std::uint32_t max_leaf_size)
{
    CHECK(!nodes.empty());
    CHECK(permutation.size() == mesh.triangles.size());
    std::vector<std::uint32_t> sorted = permutation;
    std::sort(sorted.begin(), sorted.end());
    for (std::uint32_t i = 0; i < sorted.size(); ++i) CHECK(sorted[i] == i);

    std::vector<bool> visited(nodes.size(), false);
    std::vector<std::uint32_t> stack{0};
    std::uint32_t primitive_total = 0;
    while (!stack.empty()) {
        const std::uint32_t node_index = stack.back();
        stack.pop_back();
        CHECK(node_index < nodes.size());
        if (node_index >= nodes.size()) continue;
        CHECK(!visited[node_index]);
        visited[node_index] = true;
        const auto& node = nodes[node_index];
        CHECK(node.child_count <= 8);
        if (node.is_leaf()) {
            CHECK(node.primitive_count > 0);
            CHECK(node.primitive_count <= max_leaf_size);
            CHECK(node.first_primitive + node.primitive_count <= permutation.size());
            primitive_total += node.primitive_count;
            for (std::uint32_t i = 0; i < node.primitive_count; ++i) {
                const std::uint32_t triangle_index = permutation[node.first_primitive + i];
                const uint3 triangle = mesh.triangles[triangle_index];
                CHECK(bounds_contain(node.bounds_min, node.bounds_max, mesh.positions[triangle.x]));
                CHECK(bounds_contain(node.bounds_min, node.bounds_max, mesh.positions[triangle.y]));
                CHECK(bounds_contain(node.bounds_min, node.bounds_max, mesh.positions[triangle.z]));
            }
        } else {
            CHECK(node.primitive_count == 0);
            CHECK(node.first_child + node.child_count <= nodes.size());
            for (std::uint32_t i = 0; i < node.child_count; ++i) {
                const std::uint32_t child_index = node.first_child + i;
                const auto& child = nodes[child_index];
                CHECK(bounds_contain(node.bounds_min, node.bounds_max, child.bounds_min));
                CHECK(bounds_contain(node.bounds_min, node.bounds_max, child.bounds_max));
                stack.push_back(child_index);
            }
        }
    }
    CHECK(primitive_total == mesh.triangles.size());
    CHECK(std::all_of(visited.begin(), visited.end(), [](bool value) { return value; }));
}

HostMesh identical_centroid_mesh(std::uint32_t count)
{
    HostMesh result;
    result.positions.reserve(static_cast<std::size_t>(count) * 3U);
    result.triangles.reserve(count);
    for (std::uint32_t i = 0; i < count; ++i) {
        const float scale = 1.0F + static_cast<float>(i) * 0.001F;
        const std::uint32_t base = static_cast<std::uint32_t>(result.positions.size());
        result.positions.push_back(make_float3(scale, 0.0F, 0.0F));
        result.positions.push_back(make_float3(0.0F, scale, 0.0F));
        result.positions.push_back(make_float3(-scale, -scale, 0.0F));
        result.triangles.push_back(make_uint3(base, base + 1U, base + 2U));
    }
    return result;
}

void test_hierarchy_and_determinism()
{
    const HostMesh host = identical_centroid_mesh(257);
    const DeviceMesh device(host);
    parallel_mater::Workspace workspace;
    parallel_mater::Hierarchy hierarchy;
    const parallel_mater::HierarchyOptions options{4};
    parallel_mater::Status status =
        parallel_mater::build_hierarchy(device.view(), options, workspace, hierarchy);
    CHECK(status.ok());
    CHECK(hierarchy.statistics().max_depth > 0);
    const auto reference_nodes = download(hierarchy.nodes(), hierarchy.statistics().node_count);
    const auto reference_permutation =
        download(hierarchy.primitive_indices(), host.triangles.size());
    validate_hierarchy(host, reference_nodes, reference_permutation, options.max_leaf_size);

    for (int iteration = 0; iteration < 100; ++iteration) {
        status = parallel_mater::build_hierarchy(device.view(), options, workspace, hierarchy);
        CHECK(status.ok());
        CHECK(hierarchy.statistics().node_count == reference_nodes.size());
        const auto nodes = download(hierarchy.nodes(), hierarchy.statistics().node_count);
        const auto permutation = download(hierarchy.primitive_indices(), host.triangles.size());
        CHECK(permutation == reference_permutation);
        CHECK(nodes.size() == reference_nodes.size());
        for (std::size_t i = 0; i < nodes.size() && i < reference_nodes.size(); ++i) {
            CHECK(same_node(nodes[i], reference_nodes[i]));
        }
    }
}

void test_aabb_hierarchy()
{
    std::vector<parallel_mater::Aabb> host_bounds;
    HostMesh proxy_mesh;
    for (std::uint32_t index = 0; index < 257U; ++index) {
        const float x = static_cast<float>(index % 17U) * 0.25F;
        const float y = static_cast<float>((index / 17U) % 17U) * 0.2F;
        const float z = static_cast<float>(index / 289U) * 0.3F;
        const float3 minimum = make_float3(x - 0.03F, y - 0.04F, z - 0.05F);
        const float3 maximum = make_float3(x + 0.03F, y + 0.04F, z + 0.05F);
        host_bounds.push_back({minimum, maximum});
        const std::uint32_t base = static_cast<std::uint32_t>(proxy_mesh.positions.size());
        proxy_mesh.positions.push_back(minimum);
        proxy_mesh.positions.push_back(maximum);
        proxy_mesh.positions.push_back(make_float3(x, y, z));
        proxy_mesh.triangles.push_back(make_uint3(base, base + 1U, base + 2U));
    }
    DeviceArray<parallel_mater::Aabb> device_bounds(host_bounds);
    parallel_mater::Workspace workspace;
    parallel_mater::Hierarchy hierarchy;
    const parallel_mater::HierarchyOptions options{4};
    parallel_mater::Status status = parallel_mater::build_hierarchy(
        parallel_mater::DeviceAabbView{device_bounds.data(), host_bounds.size()},
        options,
        workspace,
        hierarchy);
    CHECK(status.ok());
    const auto reference_nodes = download(hierarchy.nodes(), hierarchy.statistics().node_count);
    const auto reference_permutation = download(
        hierarchy.primitive_indices(), host_bounds.size());
    validate_hierarchy(proxy_mesh, reference_nodes, reference_permutation, options.max_leaf_size);
    for (int iteration = 0; iteration < 20; ++iteration) {
        status = parallel_mater::build_hierarchy(
            parallel_mater::DeviceAabbView{device_bounds.data(), host_bounds.size()},
            options,
            workspace,
            hierarchy);
        CHECK(status.ok());
        CHECK(download(hierarchy.primitive_indices(), host_bounds.size()) == reference_permutation);
        const auto nodes = download(hierarchy.nodes(), hierarchy.statistics().node_count);
        CHECK(nodes.size() == reference_nodes.size());
        for (std::size_t node = 0; node < nodes.size() && node < reference_nodes.size(); ++node) {
            CHECK(same_node(nodes[node], reference_nodes[node]));
        }
    }

    for (auto& bounds : host_bounds) {
        bounds.minimum.x += 10.0F;
        bounds.maximum.x += 10.0F;
    }
    device_bounds.upload(host_bounds);
    status = parallel_mater::refit_hierarchy(
        parallel_mater::DeviceAabbView{device_bounds.data(), host_bounds.size()}, hierarchy);
    CHECK(status.ok());
    CHECK(download(hierarchy.primitive_indices(), host_bounds.size()) == reference_permutation);
    const auto refitted_nodes = download(
        hierarchy.nodes(), hierarchy.statistics().node_count);
    CHECK(refitted_nodes.size() == reference_nodes.size());
    for (std::size_t node = 0; node < refitted_nodes.size(); ++node) {
        CHECK(refitted_nodes[node].first_child == reference_nodes[node].first_child);
        CHECK(refitted_nodes[node].child_count == reference_nodes[node].child_count);
        CHECK(refitted_nodes[node].first_primitive == reference_nodes[node].first_primitive);
        CHECK(refitted_nodes[node].primitive_count == reference_nodes[node].primitive_count);
    }
    CHECK(std::abs(refitted_nodes[0].bounds_min.x - 9.97F) < 1.0e-6F);
    CHECK(std::abs(refitted_nodes[0].bounds_max.x - 14.03F) < 1.0e-6F);
    CHECK(
        parallel_mater::refit_hierarchy(
            parallel_mater::DeviceAabbView{device_bounds.data(), host_bounds.size() - 1U}, hierarchy).code ==
        parallel_mater::StatusCode::invalid_argument);

    host_bounds[0] = {make_float3(1.0F, 0.0F, 0.0F), make_float3(-1.0F, 0.0F, 0.0F)};
    device_bounds.upload(host_bounds);
    status = parallel_mater::refit_hierarchy(
        parallel_mater::DeviceAabbView{device_bounds.data(), host_bounds.size()}, hierarchy);
    CHECK(status.code == parallel_mater::StatusCode::invalid_mesh);
    const auto nodes_after_failed_refit = download(
        hierarchy.nodes(), hierarchy.statistics().node_count);
    for (std::size_t node = 0; node < refitted_nodes.size(); ++node) {
        CHECK(same_node(nodes_after_failed_refit[node], refitted_nodes[node]));
    }
    status = parallel_mater::build_hierarchy(
        parallel_mater::DeviceAabbView{device_bounds.data(), host_bounds.size()},
        options,
        workspace,
        hierarchy);
    CHECK(status.code == parallel_mater::StatusCode::invalid_mesh);
}

void test_root_leaf()
{
    const HostMesh host = quad_mesh();
    const DeviceMesh device(host);
    parallel_mater::Workspace workspace;
    parallel_mater::Hierarchy hierarchy;
    const parallel_mater::Status status =
        parallel_mater::build_hierarchy(device.view(), {}, workspace, hierarchy);
    CHECK(status.ok());
    CHECK(hierarchy.statistics().node_count == 1);
    CHECK(hierarchy.statistics().leaf_count == 1);
    const auto nodes = download(hierarchy.nodes(), 1);
    const auto permutation = download(hierarchy.primitive_indices(), 2);
    validate_hierarchy(host, nodes, permutation, 8);
}

void test_seeded_triangle_soup_against_cpu()
{
    constexpr std::uint32_t triangle_count = 4096;
    std::mt19937 generator(0x4d455348U);
    std::uniform_real_distribution<float> distribution(-100.0F, 100.0F);
    HostMesh host;
    host.positions.reserve(triangle_count * 3U);
    host.triangles.reserve(triangle_count);
    for (std::uint32_t triangle_index = 0; triangle_index < triangle_count; ++triangle_index) {
        const std::uint32_t base = static_cast<std::uint32_t>(host.positions.size());
        host.positions.push_back(make_float3(distribution(generator), distribution(generator), distribution(generator)));
        host.positions.push_back(make_float3(distribution(generator), distribution(generator), distribution(generator)));
        host.positions.push_back(make_float3(distribution(generator), distribution(generator), distribution(generator)));
        host.triangles.push_back(make_uint3(base, base + 1U, base + 2U));
    }

    const DeviceMesh device(host);
    parallel_mater::Workspace workspace;
    parallel_mater::NormalOutput normals;
    CHECK(parallel_mater::compute_normals(device.view(), {}, workspace, normals).ok());
    CHECK(normals.statistics().vertex_normal_count == triangle_count * 3U);
    const auto gpu_faces = download(normals.face_normals(), triangle_count);
    for (std::size_t i = 0; i < gpu_faces.size(); ++i) {
        CHECK(near(gpu_faces[i], cpu_face_normal(host, i), 2.0e-5F));
    }

    parallel_mater::Hierarchy hierarchy;
    CHECK(parallel_mater::build_hierarchy(device.view(), {}, workspace, hierarchy).ok());
    const auto nodes = download(hierarchy.nodes(), hierarchy.statistics().node_count);
    const auto permutation = download(hierarchy.primitive_indices(), triangle_count);
    validate_hierarchy(host, nodes, permutation, 8);
}

} // namespace

int main()
{
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::puts("SKIP: no CUDA device");
        return 77;
    }
    test_empty_input();
    test_triangle_and_degenerate_normals();
    test_smooth_and_sharp_quad();
    test_normal_topology_fixtures();
    test_invalid_meshes();
    test_root_leaf();
    test_hierarchy_and_determinism();
    test_aabb_hierarchy();
    test_seeded_triangle_soup_against_cpu();
    if (failures != 0) {
        std::fprintf(stderr, "%d test assertions failed\n", failures);
        return 1;
    }
    std::puts("all meshprep tests passed");
    return 0;
}
