// SPDX-License-Identifier: MIT
#include "fixed_topology.hpp"
#include "status_exception.hpp"

#include <parallel_mater/geometry.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace parallel_mater::physics::detail {
namespace {

constexpr std::uint32_t block_size = 256U;
constexpr std::array<char, 8> file_magic{'M', 'S', 'B', 'O', 'D', 'Y', '1', '\0'};
constexpr std::uint32_t file_version = 1U;
constexpr std::uint32_t file_endian = 0x01020304U;

struct FileHeader {
    char magic[8];
    std::uint32_t version;
    std::uint32_t endian;
    std::uint32_t header_bytes;
    std::uint32_t node_count;
    std::uint32_t bond_count;
    std::uint32_t neighbor_count;
    std::uint32_t surface_vertex_count;
    std::uint32_t surface_triangle_count;
    std::uint32_t surface_node_count;
    std::uint32_t pinned_node_count;
    std::uint32_t relaxation_iterations;
    std::uint32_t flags;
    float nominal_spacing;
    std::uint32_t reserved;
};
static_assert(sizeof(FileHeader) == 64U);

struct NodeRecord {
    float x, y, z;
    std::uint8_t flags;
    std::uint8_t padding[3];
};
struct BondRecord {
    std::uint32_t a, b;
    float rest_length;
};
struct NeighborRecord {
    std::uint32_t node, bond;
};
struct SurfaceVertexRecord {
    float x, y, z, u, v;
    std::uint32_t nodes[4];
    float weights[4];
};
struct TriangleRecord {
    std::uint32_t a, b, c;
};
static_assert(sizeof(NodeRecord) == 16U);
static_assert(sizeof(BondRecord) == 12U);
static_assert(sizeof(NeighborRecord) == 8U);
static_assert(sizeof(SurfaceVertexRecord) == 52U);
static_assert(sizeof(TriangleRecord) == 12U);

[[nodiscard]] bool finite(float value) noexcept { return std::isfinite(value); }
[[nodiscard]] bool finite(float2 value) noexcept { return finite(value.x) && finite(value.y); }
[[nodiscard]] bool finite(float3 value) noexcept {
    return finite(value.x) && finite(value.y) && finite(value.z);
}
[[nodiscard]] bool finite(float4 value) noexcept {
    return finite(value.x) && finite(value.y) && finite(value.z) && finite(value.w);
}

void check(cudaError_t error, const char *operation) { throw_if_failed(error, operation); }

void check(Status status, const char *operation) { throw_if_failed(status, operation); }

template <typename T> class DeviceArray {
  public:
    DeviceArray() noexcept = default;
    ~DeviceArray() { cudaFree(data_); }
    DeviceArray(const DeviceArray &) = delete;
    DeviceArray &operator=(const DeviceArray &) = delete;

    void allocate(std::size_t count) {
        if (count == 0U) return;
        check(cudaMalloc(&data_, count * sizeof(T)), "allocate fixed-topology storage");
        count_ = count;
    }
    void upload(const std::vector<T> &values) {
        allocate(values.size());
        if (!values.empty())
            check(
                cudaMemcpy(data_, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice),
                "upload fixed-topology storage");
    }
    [[nodiscard]] T *get() noexcept { return data_; }
    [[nodiscard]] const T *get() const noexcept { return data_; }
    [[nodiscard]] std::size_t size() const noexcept { return count_; }
    [[nodiscard]] std::size_t bytes() const noexcept { return count_ * sizeof(T); }

  private:
    T *data_{};
    std::size_t count_{};
};

class Event {
  public:
    Event() { check(cudaEventCreate(&value_), "create fixed-topology timing event"); }
    ~Event() {
        if (value_) cudaEventDestroy(value_);
    }
    Event(const Event &) = delete;
    Event &operator=(const Event &) = delete;
    operator cudaEvent_t() const noexcept { return value_; }

  private:
    cudaEvent_t value_{};
};

template <typename T>
void read_records(std::istream &input, std::vector<T> &records, const char *label) {
    if (records.empty()) return;
    input.read(reinterpret_cast<char *>(records.data()),
               static_cast<std::streamsize>(records.size() * sizeof(T)));
    if (!input) throw std::runtime_error(std::string("truncated soft-body ") + label);
}

[[nodiscard]] std::uint32_t checked_count(std::size_t value, const char *label) {
    if (value > std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument(std::string("soft-body ") + label + " exceeds uint32");
    return static_cast<std::uint32_t>(value);
}

[[nodiscard]] std::size_t checked_product(std::size_t first, std::size_t second,
                                          const char *label) {
    if (first != 0U && second > std::numeric_limits<std::size_t>::max() / first)
        throw std::invalid_argument(std::string("soft-body ") + label + " overflows size_t");
    const std::size_t result = first * second;
    (void)checked_count(result, label);
    return result;
}

[[nodiscard]] std::array<std::uint32_t, 4> binding_nodes(SurfaceBinding value) {
    return {value.nodes.x, value.nodes.y, value.nodes.z, value.nodes.w};
}
[[nodiscard]] std::array<float, 4> binding_weights(SurfaceBinding value) {
    return {value.weights.x, value.weights.y, value.weights.z, value.weights.w};
}

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
__host__ __device__ float3 cross(float3 a, float3 b) {
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

__global__ void predict_nodes(const float3 *positions, const float3 *velocities, const float3 *rest,
                              const std::uint32_t *flags, float3 *predicted, std::uint32_t count,
                              float3 acceleration, float timestep) {
    const std::uint32_t node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= count) return;
    if ((flags[node] & node_pinned) != 0U) {
        predicted[node] = rest[node];
        return;
    }
    predicted[node] =
        add(positions[node],
            multiply(add(velocities[node], multiply(acceleration, timestep)), timestep));
}

__global__ void apply_external(float3 *predicted, const float3 *rest, const std::uint32_t *flags,
                               const float3 *impulses, const float3 *corrections,
                               std::uint32_t count, float inverse_mass, float timestep,
                               float max_projection) {
    const std::uint32_t node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= count) return;
    if ((flags[node] & node_pinned) != 0U) {
        predicted[node] = rest[node];
        return;
    }
    float3 correction = corrections[node];
    const float length_squared = dot(correction, correction);
    if (length_squared > max_projection * max_projection)
        correction = multiply(correction, max_projection / sqrtf(length_squared));
    predicted[node] =
        add(add(predicted[node], correction), multiply(impulses[node], inverse_mass * timestep));
}

__global__ void gather_constraints(const float3 *predicted, const float3 *rest,
                                   const std::uint32_t *flags, const Bond *bonds,
                                   const std::uint8_t *active, const std::uint32_t *offsets,
                                   const FixedTopologyNeighbor *neighbors, float3 *corrections,
                                   std::uint32_t count, std::uint32_t nodes_per_instance,
                                   std::uint32_t bonds_per_instance, float inverse_mass,
                                   float compliance, float max_projection) {
    const std::uint32_t node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= count) return;
    if ((flags[node] & node_pinned) != 0U) {
        corrections[node] = {};
        return;
    }
    const std::uint32_t instance = node / nodes_per_instance;
    const std::uint32_t local = node - instance * nodes_per_instance;
    float3 correction{};
    std::uint32_t contributions{};
    for (std::uint32_t row = offsets[local]; row < offsets[local + 1U]; ++row) {
        const FixedTopologyNeighbor neighbor = neighbors[row];
        if (active[instance * bonds_per_instance + neighbor.bond] == 0U) continue;
        const std::uint32_t other = instance * nodes_per_instance + neighbor.node;
        const float3 delta = subtract(predicted[node], predicted[other]);
        const float length_squared = dot(delta, delta);
        if (!(length_squared > 1.0e-12F)) continue;
        const float length = sqrtf(length_squared);
        const float constraint = length - bonds[neighbor.bond].rest_length;
        const float other_inverse_mass = (flags[other] & node_pinned) != 0U ? 0.0F : inverse_mass;
        const float denominator = inverse_mass + other_inverse_mass + compliance;
        correction =
            add(correction, multiply(delta, -inverse_mass * constraint / (denominator * length)));
        ++contributions;
    }
    if (contributions != 0U)
        correction = multiply(correction, 1.0F / static_cast<float>(contributions));
    const float length_squared = dot(correction, correction);
    if (length_squared > max_projection * max_projection)
        correction = multiply(correction, max_projection / sqrtf(length_squared));
    corrections[node] = correction;
}

__global__ void apply_constraints(float3 *predicted, const float3 *rest, const std::uint32_t *flags,
                                  const float3 *corrections, std::uint32_t count) {
    const std::uint32_t node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= count) return;
    predicted[node] =
        (flags[node] & node_pinned) != 0U ? rest[node] : add(predicted[node], corrections[node]);
}

__global__ void update_bonds(const float3 *predicted, const Bond *bonds, std::uint8_t *active,
                             std::uint8_t *damage, std::uint32_t bonds_per_instance,
                             std::uint32_t nodes_per_instance, std::uint32_t total_bonds,
                             float break_strain, std::uint32_t persistence,
                             unsigned int *counters) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total_bonds || active[index] == 0U) return;
    const std::uint32_t instance = index / bonds_per_instance;
    const Bond bond = bonds[index - instance * bonds_per_instance];
    const float3 delta = subtract(predicted[instance * nodes_per_instance + bond.vertices.x],
                                  predicted[instance * nodes_per_instance + bond.vertices.y]);
    const float strain = fabsf(sqrtf(dot(delta, delta)) / bond.rest_length - 1.0F);
    std::uint8_t next = damage[index];
    if (strain > break_strain)
        next = static_cast<std::uint8_t>(min(static_cast<std::uint32_t>(next) + 1U, 255U));
    else if (next != 0U)
        --next;
    damage[index] = next;
    if (next >= persistence) {
        active[index] = 0U;
        atomicAdd(counters + 1U, 1U);
    }
}

__global__ void commit_nodes(float3 *positions, float3 *velocities, const float3 *predicted,
                             const float3 *rest, const std::uint32_t *flags, std::uint32_t count,
                             float timestep, float damping, float maximum_speed,
                             float velocity_response, unsigned int *counters) {
    const std::uint32_t node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= count) return;
    if ((flags[node] & node_pinned) != 0U) {
        positions[node] = rest[node];
        velocities[node] = {};
        return;
    }
    const float3 old = positions[node];
    float3 velocity = multiply(subtract(predicted[node], old), velocity_response / timestep);
    velocity = multiply(velocity, expf(-damping * timestep));
    const float speed_squared = dot(velocity, velocity);
    if (speed_squared > maximum_speed * maximum_speed)
        velocity = multiply(velocity, maximum_speed / sqrtf(speed_squared));
    const float3 position = predicted[node];
    if (!isfinite(position.x) || !isfinite(position.y) || !isfinite(position.z) ||
        !isfinite(velocity.x) || !isfinite(velocity.y) || !isfinite(velocity.z)) {
        positions[node] = rest[node];
        velocities[node] = {};
        atomicAdd(counters, 1U);
    } else {
        positions[node] = position;
        velocities[node] = velocity;
    }
}

__global__ void deform_surface(const float3 *nodes, const float3 *rest_nodes,
                               const float3 *authored, const SurfaceBinding *bindings,
                               float3 *positions, std::uint32_t count) {
    const std::uint32_t vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex >= count) return;
    const SurfaceBinding binding = bindings[vertex];
    const std::uint32_t ids[4]{binding.nodes.x, binding.nodes.y, binding.nodes.z, binding.nodes.w};
    const float weights[4]{binding.weights.x, binding.weights.y, binding.weights.z,
                           binding.weights.w};
    float3 displacement{};
    for (unsigned slot = 0U; slot < 4U; ++slot)
        displacement = add(displacement, multiply(subtract(nodes[ids[slot]], rest_nodes[ids[slot]]),
                                                  weights[slot]));
    positions[vertex] = add(authored[vertex], displacement);
}

__global__ void compute_surface_normals(const float3 *positions, const uint3 *triangles,
                                        const std::uint32_t *offsets, const std::uint32_t *incident,
                                        float3 *normals, std::uint32_t count) {
    const std::uint32_t vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex >= count) return;
    float3 sum{};
    for (std::uint32_t row = offsets[vertex]; row < offsets[vertex + 1U]; ++row) {
        const uint3 triangle = triangles[incident[row]];
        sum = add(sum, cross(subtract(positions[triangle.y], positions[triangle.x]),
                             subtract(positions[triangle.z], positions[triangle.x])));
    }
    const float length_squared = dot(sum, sum);
    normals[vertex] = length_squared > 1.0e-20F ? multiply(sum, rsqrtf(length_squared))
                                                : make_float3(0.0F, 1.0F, 0.0F);
}

__global__ void emit_triangle_bounds(const float3 *positions, const uint3 *triangles, Aabb *bounds,
                                     std::uint32_t count) {
    const std::uint32_t triangle = blockIdx.x * blockDim.x + threadIdx.x;
    if (triangle >= count) return;
    const uint3 indices = triangles[triangle];
    const float3 a = positions[indices.x];
    const float3 b = positions[indices.y];
    const float3 c = positions[indices.z];
    bounds[triangle] = {
        {fminf(a.x, fminf(b.x, c.x)), fminf(a.y, fminf(b.y, c.y)), fminf(a.z, fminf(b.z, c.z))},
        {fmaxf(a.x, fmaxf(b.x, c.x)), fmaxf(a.y, fmaxf(b.y, c.y)), fmaxf(a.z, fmaxf(b.z, c.z))}};
}

} // namespace

void validate_fixed_topology_asset(const FixedTopologyAsset &asset) {
    const std::uint32_t node_count = checked_count(asset.rest_nodes.size(), "node count");
    const std::uint32_t bond_count = checked_count(asset.bonds.size(), "bond count");
    if (node_count == 0U || bond_count == 0U || !finite(asset.nominal_spacing) ||
        asset.nominal_spacing <= 0.0F || !finite(asset.node_radius) || asset.node_radius <= 0.0F ||
        (asset.file_flags & (asset_y_up | asset_delta_skinning)) !=
            (asset_y_up | asset_delta_skinning))
        throw std::invalid_argument("soft-body asset requires nodes, bonds, and positive spacing");
    if (asset.node_flags.size() != node_count ||
        asset.neighbor_offsets.size() != static_cast<std::size_t>(node_count) + 1U ||
        asset.neighbor_offsets.front() != 0U ||
        asset.neighbor_offsets.back() != asset.neighbors.size())
        throw std::invalid_argument("soft-body node or CSR arrays are inconsistent");
    bool surface{};
    bool pinned{};
    for (std::uint32_t node = 0U; node < node_count; ++node) {
        if (!finite(asset.rest_nodes[node]) ||
            (asset.node_flags[node] & ~(asset_node_surface | asset_node_pinned)) != 0U ||
            asset.neighbor_offsets[node] > asset.neighbor_offsets[node + 1U])
            throw std::invalid_argument("soft-body node data is invalid");
        surface |= (asset.node_flags[node] & asset_node_surface) != 0U;
        pinned |= (asset.node_flags[node] & asset_node_pinned) != 0U;
    }
    if (!surface || (!pinned && (asset.file_flags & asset_free_body) == 0U))
        throw std::invalid_argument(
            "soft-body asset requires surface nodes and anchors unless free");
    std::vector<std::uint32_t> occurrences(bond_count);
    uint2 previous{};
    for (std::uint32_t index = 0U; index < bond_count; ++index) {
        const Bond bond = asset.bonds[index];
        if (bond.vertices.x >= node_count || bond.vertices.y >= node_count ||
            bond.vertices.x >= bond.vertices.y || !finite(bond.rest_length) ||
            bond.rest_length <= 0.0F ||
            (index != 0U && (bond.vertices.x < previous.x ||
                             (bond.vertices.x == previous.x && bond.vertices.y <= previous.y))))
            throw std::invalid_argument("soft-body bonds must be valid, unique, and sorted");
        previous = bond.vertices;
    }
    for (std::uint32_t node = 0U; node < node_count; ++node) {
        std::uint32_t previous_neighbor{};
        bool first = true;
        for (std::uint32_t row = asset.neighbor_offsets[node];
             row < asset.neighbor_offsets[node + 1U]; ++row) {
            const auto neighbor = asset.neighbors[row];
            if (neighbor.node >= node_count || neighbor.node == node ||
                neighbor.bond >= bond_count || (!first && neighbor.node <= previous_neighbor))
                throw std::invalid_argument("soft-body neighbor CSR is invalid");
            const uint2 endpoints = asset.bonds[neighbor.bond].vertices;
            if (!((endpoints.x == node && endpoints.y == neighbor.node) ||
                  (endpoints.y == node && endpoints.x == neighbor.node)))
                throw std::invalid_argument("soft-body neighbor references another bond");
            ++occurrences[neighbor.bond];
            previous_neighbor = neighbor.node;
            first = false;
        }
    }
    if (std::any_of(occurrences.begin(), occurrences.end(),
                    [](std::uint32_t count) { return count != 2U; }))
        throw std::invalid_argument("every soft-body bond must occur twice in CSR");
    const std::size_t vertices = asset.surface_positions.size();
    if (vertices == 0U || asset.surface_triangles.empty() || asset.surface_uvs.size() != vertices ||
        asset.surface_bindings.size() != vertices)
        throw std::invalid_argument("soft-body surface arrays are inconsistent");
    for (std::size_t vertex = 0U; vertex < vertices; ++vertex) {
        if (!finite(asset.surface_positions[vertex]) || !finite(asset.surface_uvs[vertex]) ||
            !finite(asset.surface_bindings[vertex].weights))
            throw std::invalid_argument("soft-body surface vertex is non-finite");
        const auto nodes = binding_nodes(asset.surface_bindings[vertex]);
        const auto weights = binding_weights(asset.surface_bindings[vertex]);
        float sum{};
        for (unsigned slot = 0U; slot < 4U; ++slot) {
            if (nodes[slot] >= node_count || weights[slot] < 0.0F)
                throw std::invalid_argument("soft-body surface binding is invalid");
            sum += weights[slot];
        }
        if (std::abs(sum - 1.0F) > 1.0e-4F)
            throw std::invalid_argument("soft-body surface weights do not sum to one");
    }
    for (const uint3 triangle : asset.surface_triangles)
        if (triangle.x >= vertices || triangle.y >= vertices || triangle.z >= vertices ||
            triangle.x == triangle.y || triangle.y == triangle.z || triangle.z == triangle.x)
            throw std::invalid_argument("soft-body surface triangle is invalid");
}

FixedTopologyAsset load_fixed_topology_asset(std::span<const std::byte> bytes) {
    if constexpr (std::endian::native != std::endian::little)
        throw std::runtime_error("soft-body assets require a little-endian host");
    if (bytes.empty()) throw std::invalid_argument("soft-body asset bytes are empty");
    const std::string storage(reinterpret_cast<const char *>(bytes.data()), bytes.size());
    std::istringstream input(storage, std::ios::in | std::ios::binary);
    FileHeader header{};
    input.read(reinterpret_cast<char *>(&header), sizeof(header));
    if (!input || !std::equal(file_magic.begin(), file_magic.end(), header.magic) ||
        header.version != file_version || header.endian != file_endian ||
        header.header_bytes != sizeof(FileHeader))
        throw std::runtime_error("unsupported or corrupt soft-body asset header");
    constexpr std::uint32_t maximum_records = 50'000'000U;
    if (header.node_count > maximum_records || header.bond_count > maximum_records ||
        header.neighbor_count > maximum_records || header.surface_vertex_count > maximum_records ||
        header.surface_triangle_count > maximum_records)
        throw std::runtime_error("soft-body asset count exceeds safety limit");
    std::vector<NodeRecord> nodes(header.node_count);
    std::vector<BondRecord> bonds(header.bond_count);
    std::vector<std::uint32_t> offsets(static_cast<std::size_t>(header.node_count) + 1U);
    std::vector<NeighborRecord> neighbors(header.neighbor_count);
    std::vector<SurfaceVertexRecord> vertices(header.surface_vertex_count);
    std::vector<TriangleRecord> triangles(header.surface_triangle_count);
    read_records(input, nodes, "nodes");
    read_records(input, bonds, "bonds");
    read_records(input, offsets, "neighbor offsets");
    read_records(input, neighbors, "neighbors");
    read_records(input, vertices, "surface vertices");
    read_records(input, triangles, "surface triangles");
    if (input.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("soft-body asset has unexpected trailing bytes");
    FixedTopologyAsset asset;
    asset.nominal_spacing = header.nominal_spacing;
    asset.node_radius = 0.45F * header.nominal_spacing;
    asset.relaxation_iterations = header.relaxation_iterations;
    asset.file_flags = header.flags;
    for (const auto &node : nodes) {
        asset.rest_nodes.push_back(make_float3(node.x, node.y, node.z));
        asset.node_flags.push_back(node.flags);
    }
    for (const auto &bond : bonds)
        asset.bonds.push_back({make_uint2(bond.a, bond.b), bond.rest_length});
    asset.neighbor_offsets = std::move(offsets);
    for (const auto &neighbor : neighbors)
        asset.neighbors.push_back({neighbor.node, neighbor.bond});
    for (const auto &vertex : vertices) {
        asset.surface_positions.push_back(make_float3(vertex.x, vertex.y, vertex.z));
        asset.surface_uvs.push_back(make_float2(vertex.u, vertex.v));
        asset.surface_bindings.push_back(
            {make_uint4(vertex.nodes[0], vertex.nodes[1], vertex.nodes[2], vertex.nodes[3]),
             make_float4(vertex.weights[0], vertex.weights[1], vertex.weights[2],
                         vertex.weights[3])});
    }
    for (const auto &triangle : triangles)
        asset.surface_triangles.push_back(make_uint3(triangle.a, triangle.b, triangle.c));
    validate_fixed_topology_asset(asset);
    const auto surface_count = static_cast<std::uint32_t>(
        std::count_if(asset.node_flags.begin(), asset.node_flags.end(),
                      [](std::uint32_t flags) { return (flags & asset_node_surface) != 0U; }));
    const auto pinned_count = static_cast<std::uint32_t>(
        std::count_if(asset.node_flags.begin(), asset.node_flags.end(),
                      [](std::uint32_t flags) { return (flags & asset_node_pinned) != 0U; }));
    if (surface_count != header.surface_node_count || pinned_count != header.pinned_node_count)
        throw std::runtime_error("soft-body asset flag counts do not match payload");
    return asset;
}

FixedTopologyAsset load_fixed_topology_asset(const std::string &path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("cannot open soft-body asset: " + path);
    const std::streampos end = input.tellg();
    if (end <= 0) throw std::runtime_error("soft-body asset is empty: " + path);
    std::vector<std::byte> bytes(static_cast<std::size_t>(end));
    input.seekg(0, std::ios::beg);
    input.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!input) throw std::runtime_error("could not read soft-body asset: " + path);
    return load_fixed_topology_asset(bytes);
}

struct FixedTopology::Impl {
    Impl(FixedTopologyAsset source, SoftBodyOptions selected)
        : asset(std::move(source)), options(selected) {
        validate_fixed_topology_asset(asset);
        nodes_per_instance = checked_count(asset.rest_nodes.size(), "nodes per instance");
        bonds_per_instance = checked_count(asset.bonds.size(), "bonds per instance");
        surface_vertices_per_instance =
            checked_count(asset.surface_positions.size(), "surface vertices per instance");
        surface_triangles_per_instance =
            checked_count(asset.surface_triangles.size(), "surface triangles per instance");
        node_count = checked_count(
            checked_product(nodes_per_instance, options.instance_count, "expanded node count"),
            "expanded node count");
        bond_count = checked_count(
            checked_product(bonds_per_instance, options.instance_count, "expanded bond count"),
            "expanded bond count");
        surface_vertex_count =
            checked_count(checked_product(surface_vertices_per_instance, options.instance_count,
                                          "expanded surface vertices"),
                          "expanded surface vertices");
        surface_triangle_count =
            checked_count(checked_product(surface_triangles_per_instance, options.instance_count,
                                          "expanded surface triangles"),
                          "expanded surface triangles");
        surface_node_count =
            static_cast<std::uint32_t>(std::count_if(
                asset.node_flags.begin(), asset.node_flags.end(),
                [](std::uint32_t flags) { return (flags & asset_node_surface) != 0U; })) *
            options.instance_count;

        std::vector<float3> host_rest;
        std::vector<std::uint32_t> host_flags;
        std::vector<float3> host_surface;
        std::vector<float2> host_uvs;
        std::vector<SurfaceBinding> host_bindings;
        std::vector<uint3> host_triangles;
        host_rest.reserve(node_count);
        host_flags.reserve(node_count);
        host_surface.reserve(surface_vertex_count);
        host_uvs.reserve(surface_vertex_count);
        host_bindings.reserve(surface_vertex_count);
        host_triangles.reserve(surface_triangle_count);
        for (std::uint32_t instance = 0U; instance < options.instance_count; ++instance) {
            const float3 origin = options.instance_origins[instance];
            const std::uint32_t node_base = instance * nodes_per_instance;
            const std::uint32_t surface_base = instance * surface_vertices_per_instance;
            for (std::uint32_t node = 0U; node < nodes_per_instance; ++node) {
                host_rest.push_back(add(asset.rest_nodes[node], origin));
                host_flags.push_back(asset.node_flags[node]);
            }
            for (std::uint32_t vertex = 0U; vertex < surface_vertices_per_instance; ++vertex) {
                host_surface.push_back(add(asset.surface_positions[vertex], origin));
                host_uvs.push_back(asset.surface_uvs[vertex]);
                SurfaceBinding binding = asset.surface_bindings[vertex];
                binding.nodes.x += node_base;
                binding.nodes.y += node_base;
                binding.nodes.z += node_base;
                binding.nodes.w += node_base;
                host_bindings.push_back(binding);
            }
            for (uint3 triangle : asset.surface_triangles) {
                triangle.x += surface_base;
                triangle.y += surface_base;
                triangle.z += surface_base;
                host_triangles.push_back(triangle);
            }
        }
        std::vector<std::vector<std::uint32_t>> adjacency(surface_vertex_count);
        std::vector<std::uint32_t> host_corner_indices;
        host_corner_indices.reserve(3U * surface_triangle_count);
        for (std::uint32_t triangle = 0U; triangle < surface_triangle_count; ++triangle) {
            const uint3 value = host_triangles[triangle];
            adjacency[value.x].push_back(triangle);
            adjacency[value.y].push_back(triangle);
            adjacency[value.z].push_back(triangle);
            host_corner_indices.push_back(value.x);
            host_corner_indices.push_back(value.y);
            host_corner_indices.push_back(value.z);
        }
        std::vector<std::uint32_t> host_surface_offsets{0U};
        std::vector<std::uint32_t> host_incident;
        for (auto &row : adjacency) {
            std::sort(row.begin(), row.end());
            host_incident.insert(host_incident.end(), row.begin(), row.end());
            host_surface_offsets.push_back(static_cast<std::uint32_t>(host_incident.size()));
        }
        rest.upload(host_rest);
        positions.upload(host_rest);
        initial_positions.upload(host_rest);
        velocities.allocate(node_count);
        check(cudaMemset(velocities.get(), 0, velocities.bytes()),
              "clear fixed-topology velocities");
        predicted.allocate(node_count);
        corrections.allocate(node_count);
        external_impulses.allocate(node_count);
        external_corrections.allocate(node_count);
        flags.upload(host_flags);
        topology_bonds.upload(asset.bonds);
        neighbor_offsets.upload(asset.neighbor_offsets);
        neighbors.upload(asset.neighbors);
        active_bonds.allocate(bond_count);
        damage.allocate(bond_count);
        check(cudaMemset(active_bonds.get(), 1, active_bonds.bytes()), "activate bonds");
        check(cudaMemset(damage.get(), 0, damage.bytes()), "clear bond damage");
        authored_surface.upload(host_surface);
        surface_positions.allocate(surface_vertex_count);
        surface_normals.allocate(surface_vertex_count);
        surface_uvs.upload(host_uvs);
        surface_bindings.upload(host_bindings);
        surface_triangles.upload(host_triangles);
        surface_triangle_active.allocate(surface_triangle_count);
        check(cudaMemset(surface_triangle_active.get(), 1, surface_triangle_active.bytes()),
              "activate surface triangles");
        corner_normal_indices.upload(host_corner_indices);
        surface_offsets.upload(host_surface_offsets);
        incident_triangles.upload(host_incident);
        triangle_bounds.allocate(surface_triangle_count);
        counters.allocate(2U);
        check(cudaMemset(counters.get(), 0, counters.bytes()), "clear fixed-topology counters");
        launch_surface(nullptr);
        check(build_hierarchy(DeviceMeshView{surface_positions.get(), surface_vertex_count,
                                             surface_triangles.get(), surface_triangle_count},
                              {options.hierarchy_leaf_size}, workspace, hierarchy),
              "build fixed-topology surface hierarchy");
    }

    void launch_surface(cudaStream_t stream) {
        deform_surface<<<(surface_vertex_count + block_size - 1U) / block_size, block_size, 0,
                         stream>>>(positions.get(), rest.get(), authored_surface.get(),
                                   surface_bindings.get(), surface_positions.get(),
                                   surface_vertex_count);
        compute_surface_normals<<<(surface_vertex_count + block_size - 1U) / block_size, block_size,
                                  0, stream>>>(surface_positions.get(), surface_triangles.get(),
                                               surface_offsets.get(), incident_triangles.get(),
                                               surface_normals.get(), surface_vertex_count);
        emit_triangle_bounds<<<(surface_triangle_count + block_size - 1U) / block_size, block_size,
                               0, stream>>>(surface_positions.get(), surface_triangles.get(),
                                            triangle_bounds.get(), surface_triangle_count);
        check(cudaGetLastError(), "launch fixed-topology surface update");
    }

    FixedTopologyAsset asset;
    SoftBodyOptions options{};
    std::uint32_t nodes_per_instance{};
    std::uint32_t bonds_per_instance{};
    std::uint32_t surface_vertices_per_instance{};
    std::uint32_t surface_triangles_per_instance{};
    std::uint32_t node_count{};
    std::uint32_t bond_count{};
    std::uint32_t surface_vertex_count{};
    std::uint32_t surface_triangle_count{};
    std::uint32_t surface_node_count{};
    DeviceArray<float3> rest, positions, initial_positions, velocities, predicted, corrections;
    DeviceArray<float3> external_impulses, external_corrections;
    DeviceArray<std::uint32_t> flags, neighbor_offsets;
    DeviceArray<Bond> topology_bonds;
    DeviceArray<FixedTopologyNeighbor> neighbors;
    DeviceArray<std::uint8_t> active_bonds, damage, surface_triangle_active;
    DeviceArray<float3> authored_surface, surface_positions, surface_normals;
    DeviceArray<float2> surface_uvs;
    DeviceArray<SurfaceBinding> surface_bindings;
    DeviceArray<uint3> surface_triangles;
    DeviceArray<std::uint32_t> corner_normal_indices, surface_offsets, incident_triangles;
    DeviceArray<Aabb> triangle_bounds;
    DeviceArray<unsigned int> counters;
    Workspace workspace;
    Hierarchy hierarchy;
    Event physics_begin, physics_end, surface_begin, surface_end, hierarchy_begin, hierarchy_end;
    std::array<unsigned int, 2> host_counters{};
    SoftBodyTimings timings{};
    std::uint64_t frame_index{};
    bool telemetry_pending{};
};

FixedTopology::FixedTopology(FixedTopologyAsset asset, SoftBodyOptions options)
    : impl_(std::make_unique<Impl>(std::move(asset), options)) {}
FixedTopology::FixedTopology(const std::string &path, SoftBodyOptions options)
    : FixedTopology(load_fixed_topology_asset(path), options) {}
FixedTopology::~FixedTopology() = default;
FixedTopology::FixedTopology(FixedTopology &&) noexcept = default;
FixedTopology &FixedTopology::operator=(FixedTopology &&) noexcept = default;

void FixedTopology::begin_frame(cudaStream_t stream) {
    check(cudaEventRecord(impl_->physics_begin, stream), "record fixed-topology frame start");
    check(cudaMemsetAsync(impl_->counters.get(), 0, sizeof(unsigned int), stream),
          "clear fixed-topology finite counter");
}

void FixedTopology::prepare_substep(float timestep, float3 acceleration, cudaStream_t stream) {
    check(cudaMemsetAsync(impl_->external_impulses.get(), 0, impl_->external_impulses.bytes(),
                          stream),
          "clear external impulses");
    check(cudaMemsetAsync(impl_->external_corrections.get(), 0, impl_->external_corrections.bytes(),
                          stream),
          "clear external corrections");
    predict_nodes<<<(impl_->node_count + block_size - 1U) / block_size, block_size, 0, stream>>>(
        impl_->positions.get(), impl_->velocities.get(), impl_->rest.get(), impl_->flags.get(),
        impl_->predicted.get(), impl_->node_count, acceleration, timestep);
    check(cudaGetLastError(), "predict fixed-topology nodes");
}

void FixedTopology::finish_substep(float timestep, float3, cudaStream_t stream) {
    const float inverse_mass = 1.0F / impl_->options.node_mass;
    const float maximum_projection =
        impl_->options.maximum_projection_fraction * impl_->asset.nominal_spacing;
    apply_external<<<(impl_->node_count + block_size - 1U) / block_size, block_size, 0, stream>>>(
        impl_->predicted.get(), impl_->rest.get(), impl_->flags.get(),
        impl_->external_impulses.get(), impl_->external_corrections.get(), impl_->node_count,
        inverse_mass, timestep, maximum_projection);
    const float stiffness = impl_->options.spring_stiffness * impl_->options.strength_multiplier;
    const float compliance = 1.0F / (stiffness * timestep * timestep);
    for (std::uint32_t iteration = 0U; iteration < impl_->options.constraint_iterations;
         ++iteration) {
        gather_constraints<<<(impl_->node_count + block_size - 1U) / block_size, block_size, 0,
                             stream>>>(
            impl_->predicted.get(), impl_->rest.get(), impl_->flags.get(),
            impl_->topology_bonds.get(), impl_->active_bonds.get(), impl_->neighbor_offsets.get(),
            impl_->neighbors.get(), impl_->corrections.get(), impl_->node_count,
            impl_->nodes_per_instance, impl_->bonds_per_instance, inverse_mass, compliance,
            maximum_projection);
        apply_constraints<<<(impl_->node_count + block_size - 1U) / block_size, block_size, 0,
                            stream>>>(impl_->predicted.get(), impl_->rest.get(), impl_->flags.get(),
                                      impl_->corrections.get(), impl_->node_count);
    }
    update_bonds<<<(impl_->bond_count + block_size - 1U) / block_size, block_size, 0, stream>>>(
        impl_->predicted.get(), impl_->topology_bonds.get(), impl_->active_bonds.get(),
        impl_->damage.get(), impl_->bonds_per_instance, impl_->nodes_per_instance,
        impl_->bond_count, impl_->options.break_strain,
        impl_->options.fracture_persistence_substeps, impl_->counters.get());
    commit_nodes<<<(impl_->node_count + block_size - 1U) / block_size, block_size, 0, stream>>>(
        impl_->positions.get(), impl_->velocities.get(), impl_->predicted.get(), impl_->rest.get(),
        impl_->flags.get(), impl_->node_count, timestep, impl_->options.velocity_damping,
        impl_->options.maximum_speed, impl_->options.constraint_velocity_response,
        impl_->counters.get());
    check(cudaGetLastError(), "finish fixed-topology substep");
}

void FixedTopology::finish_frame_async(cudaStream_t stream) {
    check(cudaEventRecord(impl_->physics_end, stream), "record fixed-topology physics end");
    check(cudaEventRecord(impl_->surface_begin, stream), "record surface update start");
    impl_->launch_surface(stream);
    check(cudaEventRecord(impl_->surface_end, stream), "record surface update end");
    check(cudaEventRecord(impl_->hierarchy_begin, stream), "record hierarchy refit start");
    check(refit_hierarchy_unchecked_async(
              {impl_->triangle_bounds.get(), impl_->surface_triangle_count}, impl_->hierarchy,
              stream),
          "refit fixed-topology hierarchy");
    check(cudaEventRecord(impl_->hierarchy_end, stream), "record hierarchy refit end");
    ++impl_->frame_index;
}

void FixedTopology::collect_telemetry_async(cudaStream_t stream) {
    check(cudaMemcpyAsync(impl_->host_counters.data(), impl_->counters.get(),
                          impl_->counters.bytes(), cudaMemcpyDeviceToHost, stream),
          "download fixed-topology counters");
    impl_->telemetry_pending = true;
}

SoftBodyTimings FixedTopology::resolve_telemetry() {
    if (!impl_->telemetry_pending)
        throw std::logic_error("fixed-topology telemetry was not requested");
    check(
        cudaEventElapsedTime(&impl_->timings.physics_ms, impl_->physics_begin, impl_->physics_end),
        "resolve fixed-topology physics timing");
    check(cudaEventElapsedTime(&impl_->timings.surface_deformation_ms, impl_->surface_begin,
                               impl_->surface_end),
          "resolve surface timing");
    check(cudaEventElapsedTime(&impl_->timings.hierarchy_ms, impl_->hierarchy_begin,
                               impl_->hierarchy_end),
          "resolve hierarchy timing");
    impl_->telemetry_pending = false;
    return impl_->timings;
}

SoftBodyTimings FixedTopology::finish_frame(cudaStream_t stream) {
    finish_frame_async(stream);
    collect_telemetry_async(stream);
    check(cudaStreamSynchronize(stream), "complete fixed-topology frame");
    return resolve_telemetry();
}

void FixedTopology::reset(cudaStream_t stream) {
    check(cudaMemcpyAsync(impl_->positions.get(), impl_->initial_positions.get(),
                          impl_->positions.bytes(), cudaMemcpyDeviceToDevice, stream),
          "reset nodes");
    check(cudaMemcpyAsync(impl_->rest.get(), impl_->initial_positions.get(), impl_->rest.bytes(),
                          cudaMemcpyDeviceToDevice, stream),
          "reset rest nodes");
    check(cudaMemsetAsync(impl_->velocities.get(), 0, impl_->velocities.bytes(), stream),
          "reset velocities");
    check(cudaMemsetAsync(impl_->active_bonds.get(), 1, impl_->active_bonds.bytes(), stream),
          "reset bonds");
    check(cudaMemsetAsync(impl_->damage.get(), 0, impl_->damage.bytes(), stream),
          "reset bond damage");
    check(cudaMemsetAsync(impl_->counters.get(), 0, impl_->counters.bytes(), stream),
          "reset counters");
    impl_->launch_surface(stream);
    check(refit_hierarchy_unchecked_async(
              {impl_->triangle_bounds.get(), impl_->surface_triangle_count}, impl_->hierarchy,
              stream),
          "reset hierarchy");
    check(cudaStreamSynchronize(stream), "complete fixed-topology reset");
    impl_->frame_index = 0U;
    impl_->host_counters = {};
    impl_->timings = {};
}

void FixedTopology::set_material(SoftBodyMaterial value) {
    impl_->options.spring_stiffness = value.spring_stiffness;
    impl_->options.velocity_damping = value.velocity_damping;
    impl_->options.maximum_speed = value.maximum_speed;
}
void FixedTopology::set_substeps(std::uint32_t value) { impl_->options.substeps = value; }
void FixedTopology::set_constraint_iterations(std::uint32_t value) {
    impl_->options.constraint_iterations = value;
}
void FixedTopology::set_strength_multiplier(float value) {
    impl_->options.strength_multiplier = value;
}
void FixedTopology::set_node_mass(float value) { impl_->options.node_mass = value; }

SoftBodyMaterial FixedTopology::material() const noexcept {
    return {impl_->options.spring_stiffness, impl_->options.velocity_damping,
            impl_->options.maximum_speed};
}

SoftBodyNodeView FixedTopology::nodes() const noexcept {
    return {
        impl_->positions.get(),   impl_->velocities.get(),        impl_->rest.get(),
        impl_->flags.get(),       impl_->external_impulses.get(), impl_->external_corrections.get(),
        impl_->node_count,        impl_->nodes_per_instance,      impl_->options.instance_count,
        impl_->asset.node_radius, 1.0F / impl_->options.node_mass};
}

SoftBodyBondView FixedTopology::bonds() const noexcept {
    return {impl_->positions.get(),      impl_->flags.get(),
            impl_->topology_bonds.get(), impl_->active_bonds.get(),
            impl_->node_count,           impl_->nodes_per_instance,
            impl_->bonds_per_instance,   impl_->options.instance_count,
            impl_->asset.node_radius};
}

SoftBodySurfaceView FixedTopology::surface() const noexcept {
    const HierarchyStatistics hierarchy_statistics = impl_->hierarchy.statistics();
    return {{impl_->surface_positions.get(), impl_->surface_vertex_count,
             impl_->surface_triangles.get(), impl_->surface_triangle_count},
            impl_->surface_normals.get(),
            impl_->corner_normal_indices.get(),
            impl_->surface_uvs.get(),
            impl_->surface_triangle_active.get(),
            impl_->hierarchy.nodes(),
            impl_->hierarchy.primitive_indices(),
            hierarchy_statistics.node_count,
            hierarchy_statistics.max_depth};
}

SoftBodyStatistics FixedTopology::statistics() const noexcept {
    return {impl_->options.instance_count, impl_->nodes_per_instance, impl_->node_count,
            impl_->surface_node_count,     impl_->bonds_per_instance, impl_->bond_count,
            impl_->host_counters[1],       impl_->host_counters[0],   impl_->frame_index};
}

std::size_t FixedTopology::allocated_bytes() const noexcept {
    return impl_->rest.bytes() + impl_->positions.bytes() + impl_->initial_positions.bytes() +
           impl_->velocities.bytes() + impl_->predicted.bytes() + impl_->corrections.bytes() +
           impl_->external_impulses.bytes() + impl_->external_corrections.bytes() +
           impl_->flags.bytes() + impl_->topology_bonds.bytes() + impl_->neighbor_offsets.bytes() +
           impl_->neighbors.bytes() + impl_->active_bonds.bytes() + impl_->damage.bytes() +
           impl_->authored_surface.bytes() + impl_->surface_positions.bytes() +
           impl_->surface_normals.bytes() + impl_->surface_uvs.bytes() +
           impl_->surface_bindings.bytes() + impl_->surface_triangles.bytes() +
           impl_->surface_triangle_active.bytes() + impl_->corner_normal_indices.bytes() +
           impl_->surface_offsets.bytes() + impl_->incident_triangles.bytes() +
           impl_->triangle_bounds.bytes() + impl_->counters.bytes() +
           impl_->workspace.capacity_bytes() + impl_->hierarchy.allocated_bytes();
}

} // namespace parallel_mater::physics::detail
