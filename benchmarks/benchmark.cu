// SPDX-License-Identifier: MIT
#include <meshprep/meshprep.hpp>

#include <cuda_runtime.h>

#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

struct HostMesh {
    std::vector<float3> positions;
    std::vector<uint3> triangles;
};

struct Options {
    std::filesystem::path object_path;
    std::uint32_t generated_triangles{262144};
    int warmups{5};
    int iterations{30};
    std::string operation{"hierarchy"};
};

template <typename T>
class DeviceArray {
public:
    explicit DeviceArray(const std::vector<T>& values)
    {
        if (values.empty()) return;
        check(cudaMalloc(&data_, values.size() * sizeof(T)), "cudaMalloc");
        check(
            cudaMemcpy(data_, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice),
            "cudaMemcpy");
    }
    ~DeviceArray() { cudaFree(data_); }
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;

    [[nodiscard]] T* data() const { return data_; }

private:
    static void check(cudaError_t error, const char* operation)
    {
        if (error == cudaSuccess) return;
        std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
        std::exit(2);
    }

    T* data_{};
};

void usage(const char* executable)
{
    std::fprintf(
        stderr,
        "usage: %s [--obj FILE | --triangles N] [--warmups N] [--iterations N] "
        "[--operation hierarchy|normals|pipeline]\n",
        executable);
}

bool parse_u32(std::string_view text, std::uint32_t& value)
{
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    return result.ec == std::errc{} && result.ptr == text.data() + text.size();
}

bool parse_int(std::string_view text, int& value)
{
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    return result.ec == std::errc{} && result.ptr == text.data() + text.size();
}

bool parse_options(int argc, char** argv, Options& options)
{
    for (int i = 1; i < argc; ++i) {
        const std::string_view argument = argv[i];
        if (argument == "--help") return false;
        if (i + 1 >= argc) return false;
        const std::string_view value = argv[++i];
        if (argument == "--obj") options.object_path = value;
        else if (argument == "--triangles") {
            if (!parse_u32(value, options.generated_triangles) || options.generated_triangles == 0) {
                return false;
            }
        } else if (argument == "--warmups") {
            if (!parse_int(value, options.warmups) || options.warmups < 0) return false;
        } else if (argument == "--iterations") {
            if (!parse_int(value, options.iterations) || options.iterations <= 0) return false;
        } else if (argument == "--operation") {
            options.operation = value;
            if (options.operation != "hierarchy" && options.operation != "normals" &&
                options.operation != "pipeline") {
                return false;
            }
        } else {
            return false;
        }
    }
    return true;
}

HostMesh make_grid(std::uint32_t target_triangles)
{
    const auto cells = static_cast<std::uint32_t>((target_triangles + 1U) / 2U);
    const auto side = static_cast<std::uint32_t>(std::ceil(std::sqrt(static_cast<double>(cells)))) + 1U;
    HostMesh mesh;
    mesh.positions.reserve(static_cast<std::size_t>(side) * side);
    for (std::uint32_t y = 0; y < side; ++y) {
        for (std::uint32_t x = 0; x < side; ++x) {
            const float xf = static_cast<float>(x) * 0.01F;
            const float yf = static_cast<float>(y) * 0.01F;
            mesh.positions.push_back(make_float3(xf, yf, 0.1F * std::sin(xf) * std::cos(yf)));
        }
    }
    mesh.triangles.reserve(target_triangles);
    for (std::uint32_t y = 0; y + 1U < side && mesh.triangles.size() < target_triangles; ++y) {
        for (std::uint32_t x = 0; x + 1U < side && mesh.triangles.size() < target_triangles; ++x) {
            const std::uint32_t a = y * side + x;
            const std::uint32_t b = a + 1U;
            const std::uint32_t c = a + side;
            const std::uint32_t d = c + 1U;
            mesh.triangles.push_back(make_uint3(a, b, d));
            if (mesh.triangles.size() < target_triangles) {
                mesh.triangles.push_back(make_uint3(a, d, c));
            }
        }
    }
    return mesh;
}

bool parse_obj_index(std::string_view token, std::size_t vertex_count, std::uint32_t& index)
{
    const std::size_t slash = token.find('/');
    if (slash != std::string_view::npos) token = token.substr(0, slash);
    int parsed = 0;
    if (!parse_int(token, parsed) || parsed == 0) return false;
    const std::int64_t resolved = parsed > 0 ? static_cast<std::int64_t>(parsed - 1)
                                             : static_cast<std::int64_t>(vertex_count) + parsed;
    if (resolved < 0 || resolved >= static_cast<std::int64_t>(vertex_count)) return false;
    index = static_cast<std::uint32_t>(resolved);
    return true;
}

bool read_obj(const std::filesystem::path& path, HostMesh& mesh)
{
    std::ifstream input(path);
    if (!input) return false;
    std::string line;
    std::vector<std::uint32_t> face;
    while (std::getline(input, line)) {
        if (line.starts_with("v ")) {
            float x = 0.0F;
            float y = 0.0F;
            float z = 0.0F;
            if (std::sscanf(line.c_str() + 2, "%f %f %f", &x, &y, &z) == 3) {
                mesh.positions.push_back(make_float3(x, y, z));
            }
        } else if (line.starts_with("f ")) {
            face.clear();
            std::string_view remaining(line.c_str() + 2);
            while (!remaining.empty()) {
                const std::size_t first = remaining.find_first_not_of(' ');
                if (first == std::string_view::npos) break;
                remaining.remove_prefix(first);
                const std::size_t end = remaining.find(' ');
                const std::string_view token = remaining.substr(0, end);
                std::uint32_t index = 0;
                if (!parse_obj_index(token, mesh.positions.size(), index)) return false;
                face.push_back(index);
                if (end == std::string_view::npos) break;
                remaining.remove_prefix(end + 1U);
            }
            for (std::size_t i = 1; i + 1 < face.size(); ++i) {
                mesh.triangles.push_back(make_uint3(face[0], face[i], face[i + 1]));
            }
        }
    }
    return !mesh.positions.empty() && !mesh.triangles.empty();
}

meshprep::Status run_operation(
    const std::string& operation,
    meshprep::DeviceMeshView mesh,
    meshprep::Workspace& workspace,
    meshprep::NormalOutput& normals,
    meshprep::Hierarchy& hierarchy)
{
    if (operation == "normals" || operation == "pipeline") {
        const meshprep::Status status =
            meshprep::compute_normals(mesh, {}, workspace, normals);
        if (!status) return status;
    }
    if (operation == "hierarchy" || operation == "pipeline") {
        return meshprep::build_hierarchy(mesh, {}, workspace, hierarchy);
    }
    return {};
}

} // namespace

int main(int argc, char** argv)
{
    Options options;
    if (!parse_options(argc, argv, options)) {
        usage(argv[0]);
        return argc > 1 && std::string_view(argv[1]) == "--help" ? 0 : 1;
    }
    const auto load_start = std::chrono::steady_clock::now();
    HostMesh host_mesh;
    std::string source;
    if (options.object_path.empty()) {
        host_mesh = make_grid(options.generated_triangles);
        source = "procedural-grid";
    } else {
        if (!read_obj(options.object_path, host_mesh)) {
            std::fprintf(stderr, "failed to read OBJ: %s\n", options.object_path.c_str());
            return 1;
        }
        source = options.object_path.filename().string();
    }
    const double load_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - load_start).count();
    std::fprintf(
        stderr,
        "loaded %s: %zu vertices, %zu triangles in %.3fs\n",
        source.c_str(),
        host_mesh.positions.size(),
        host_mesh.triangles.size(),
        load_seconds);

    DeviceArray<float3> positions(host_mesh.positions);
    DeviceArray<uint3> triangles(host_mesh.triangles);
    const meshprep::DeviceMeshView mesh{
        positions.data(), host_mesh.positions.size(), triangles.data(), host_mesh.triangles.size()};
    meshprep::Workspace workspace;
    meshprep::NormalOutput normals;
    meshprep::Hierarchy hierarchy;
    for (int i = 0; i < options.warmups; ++i) {
        const meshprep::Status status =
            run_operation(options.operation, mesh, workspace, normals, hierarchy);
        if (!status) {
            std::fprintf(stderr, "warmup failed: %s (%s)\n", status.message, cudaGetErrorString(status.cuda_error));
            return 1;
        }
    }

    cudaEvent_t begin{};
    cudaEvent_t end{};
    cudaEventCreate(&begin);
    cudaEventCreate(&end);
    std::puts(
        "RESULT_HEADER,operation,source,vertices,triangles,iteration,gpu_ms,wall_ms,"
        "mtriangles_per_second,nodes,leaves,branches,max_depth,workspace_bytes,output_bytes");
    for (int iteration = 0; iteration < options.iterations; ++iteration) {
        cudaEventRecord(begin);
        const auto wall_begin = std::chrono::steady_clock::now();
        const meshprep::Status status =
            run_operation(options.operation, mesh, workspace, normals, hierarchy);
        const auto wall_end = std::chrono::steady_clock::now();
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        if (!status) {
            std::fprintf(stderr, "benchmark failed: %s (%s)\n", status.message, cudaGetErrorString(status.cuda_error));
            return 1;
        }
        float gpu_ms = 0.0F;
        cudaEventElapsedTime(&gpu_ms, begin, end);
        const double wall_ms =
            std::chrono::duration<double, std::milli>(wall_end - wall_begin).count();
        const double throughput = host_mesh.triangles.size() / (static_cast<double>(gpu_ms) * 1000.0);
        const auto stats = hierarchy.statistics();
        std::printf(
            "RESULT,%s,%s,%zu,%zu,%d,%.6f,%.6f,%.6f,%u,%u,%u,%u,%zu,%zu\n",
            options.operation.c_str(),
            source.c_str(),
            host_mesh.positions.size(),
            host_mesh.triangles.size(),
            iteration,
            gpu_ms,
            wall_ms,
            throughput,
            stats.node_count,
            stats.leaf_count,
            stats.branch_count,
            stats.max_depth,
            workspace.capacity_bytes(),
            hierarchy.allocated_bytes() + normals.allocated_bytes());
    }
    cudaEventDestroy(end);
    cudaEventDestroy(begin);
    return 0;
}
