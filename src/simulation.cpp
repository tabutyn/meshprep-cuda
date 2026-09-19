// SPDX-License-Identifier: MIT
#include <meshprep/simulation.hpp>

#include "hybrid_lab.hpp"
#include "obstacle_course.hpp"
#include "simulation_gallery.hpp"
#include "soft_body.hpp"
#include "status_exception.hpp"
#include "water_lab.hpp"

#include <cuda_runtime_api.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace meshprep::sim {
namespace {

constexpr Status success() noexcept
{
    return {};
}

constexpr Status invalid(const char* message) noexcept
{
    return {StatusCode::invalid_argument, cudaSuccess, message};
}

constexpr Status internal(const char* message) noexcept
{
    return {StatusCode::internal_error, cudaSuccess, message};
}

constexpr Status allocation_failure(const char* message) noexcept
{
    return {StatusCode::allocation_failure, cudaErrorMemoryAllocation, message};
}

void check_cuda(cudaError_t status, const char* operation)
{
    waterlab::detail::throw_if_failed(status, operation);
}

[[nodiscard]] Status translated(
    const waterlab::detail::StatusException& error,
    const char* message) noexcept
{
    Status status = error.status();
    status.message = message;
    return status;
}

[[nodiscard]] bool finite(float3 value) noexcept
{
    return std::isfinite(value.x) && std::isfinite(value.y) &&
        std::isfinite(value.z);
}

[[nodiscard]] bool valid(const GallerySimulationOptions& options) noexcept
{
    return meshprep::sim::valid(options.fixed_step) &&
        (!options.solver_iterations_override.has_value() ||
            (*options.solver_iterations_override >= 1U &&
             *options.solver_iterations_override <=
                waterlab::HybridDroplet::maximum_physics_iterations)) &&
        (!options.gravity_override.has_value() ||
            finite(*options.gravity_override)) &&
        (!options.particle_count_override.has_value() ||
            (*options.particle_count_override >= 256U &&
             *options.particle_count_override <= 100'000U)) &&
        (!options.physical_skin_frequency_override.has_value() ||
            (*options.physical_skin_frequency_override >= 2U &&
             *options.physical_skin_frequency_override <= 45U)) &&
        (!options.rope_node_count_override.has_value() ||
            (*options.rope_node_count_override >= 8U &&
             *options.rope_node_count_override <= 512U)) &&
        (!options.cloth_detail_override.has_value() ||
            (*options.cloth_detail_override >= 1U &&
             *options.cloth_detail_override <= 8U));
}

[[nodiscard]] const ExampleContextInfo* context_info(ExampleContext context) noexcept
{
    for (const auto& candidate : example_contexts) {
        if (candidate.id == context) return &candidate;
    }
    return nullptr;
}

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() noexcept = default;
    ~DeviceBuffer() { cudaFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void upload(const T* source, std::size_t count, cudaStream_t stream)
    {
        if (count == 0U) return;
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)),
            "allocate simulation rigid mesh");
        check_cuda(cudaMemcpyAsync(data_, source, count * sizeof(T),
            cudaMemcpyHostToDevice, stream), "upload simulation rigid mesh");
        count_ = count;
    }

    [[nodiscard]] T* get() const noexcept { return data_; }
    [[nodiscard]] std::size_t size() const noexcept { return count_; }

private:
    T* data_{};
    std::size_t count_{};
};

} // namespace

struct GallerySimulation::Impl {
    [[nodiscard]] std::uint32_t rope_node_count() const noexcept
    {
        return options_.rope_node_count_override.value_or(
            waterlab::gallery::default_rope_nodes);
    }

    [[nodiscard]] std::uint32_t cloth_detail() const noexcept
    {
        return options_.cloth_detail_override.value_or(
            waterlab::gallery::default_cloth_detail(options_.context));
    }

    explicit Impl(GallerySimulationOptions requested, cudaStream_t stream)
        : options_(requested), asset_path_(requested.soft_body_asset_path)
    {
        options_.soft_body_asset_path = asset_path_;
        info_ = context_info(options_.context);
        if (info_ == nullptr) throw std::invalid_argument("unknown gallery context");
        if (!valid(options_)) {
            throw std::invalid_argument("invalid gallery fixed-step options");
        }
        if (requires_soft_body_asset(options_.context) && asset_path_.empty()) {
            throw std::invalid_argument("gallery context requires a soft-body asset");
        }

        const bool particles = has_component(
            info_->components, Component::fluid_particles) ||
            has_component(info_->components, Component::hand_particles);
        const bool water_skin = has_component(info_->components, Component::water_skin);
        const bool deformable = has_component(info_->components, Component::cloth) ||
            has_component(info_->components, Component::soft_body);
        const bool rigid = has_component(info_->components, Component::rigid_bodies);

        const waterlab::gallery::ContextPhysicsOverrides overrides{
            options_.fixed_step.timestep,
            options_.solver_iterations_override,
            options_.gravity_override,
            options_.particle_count_override,
            options_.physical_skin_frequency_override};
        const waterlab::HybridOptions physics =
            waterlab::gallery::make_context_physics(options_.context, overrides);
        resolved_physics_ = {
            {physics.fixed_dt}, physics.physics_iterations, physics.gravity,
            physics.obstacle_course};

        if (particles || water_skin) {
            hybrid_ = std::make_unique<waterlab::HybridDroplet>(physics);
        }
        if (deformable) {
            deformable_ = waterlab::gallery::make_context_deformable(
                options_.context, physics, asset_path_, rope_node_count(),
                cloth_detail());
            if (!deformable_) {
                throw std::runtime_error("gallery recipe omitted a declared deformable");
            }
        }
        if (rigid) initialize_rigid_arena(stream);
        check_cuda(cudaStreamSynchronize(stream), "complete gallery initialization");
        refresh_views();
        refresh_statistics(0.0F);
    }

    void initialize_rigid_arena(cudaStream_t stream)
    {
        if (options_.context == ExampleContext::particle_bowl) {
            constexpr std::uint32_t segments = 40U;
            constexpr std::uint32_t rings = 16U;
            std::vector<float3> vertices;
            std::vector<uint3> triangles;
            vertices.reserve((rings + 1U) * segments);
            for (std::uint32_t ring = 0U; ring <= rings; ++ring) {
                const float polar = 0.5F * 3.14159265358979323846F +
                    0.5F * 3.14159265358979323846F *
                        static_cast<float>(ring) / static_cast<float>(rings);
                for (std::uint32_t segment = 0U; segment < segments; ++segment) {
                    const float azimuth = 2.0F * 3.14159265358979323846F *
                        static_cast<float>(segment) / static_cast<float>(segments);
                    vertices.push_back({sinf(polar) * cosf(azimuth), cosf(polar),
                        sinf(polar) * sinf(azimuth)});
                }
            }
            for (std::uint32_t ring = 0U; ring < rings; ++ring) {
                for (std::uint32_t segment = 0U; segment < segments; ++segment) {
                    const std::uint32_t a = ring * segments + segment;
                    const std::uint32_t b = ring * segments + (segment + 1U) % segments;
                    const std::uint32_t c = (ring + 1U) * segments + segment;
                    const std::uint32_t d = (ring + 1U) * segments +
                        (segment + 1U) % segments;
                    triangles.push_back({a, b, c});
                    triangles.push_back({b, d, c});
                }
            }
            const std::uint32_t bowl_vertex_count =
                static_cast<std::uint32_t>(vertices.size());
            const std::uint32_t bowl_triangle_count =
                static_cast<std::uint32_t>(triangles.size());
            const auto sphere = waterlab::make_geodesic_sphere(6U, 1.0F);
            const std::uint32_t sphere_vertex_first = bowl_vertex_count;
            const std::uint32_t sphere_triangle_first = bowl_triangle_count;
            vertices.insert(vertices.end(), sphere.positions.begin(), sphere.positions.end());
            for (const uint3 triangle : sphere.triangles) {
                triangles.push_back(triangle);
            }
            constexpr std::uint32_t peg_segments = 24U;
            const std::uint32_t peg_vertex_first =
                static_cast<std::uint32_t>(vertices.size());
            const std::uint32_t peg_triangle_first =
                static_cast<std::uint32_t>(triangles.size());
            for (std::uint32_t segment = 0U; segment < peg_segments; ++segment) {
                const float angle = 2.0F * 3.14159265358979323846F *
                    static_cast<float>(segment) / static_cast<float>(peg_segments);
                vertices.push_back({std::cos(angle), -1.0F, std::sin(angle)});
                vertices.push_back({std::cos(angle),  1.0F, std::sin(angle)});
            }
            vertices.push_back({0.0F, -1.0F, 0.0F});
            vertices.push_back({0.0F,  1.0F, 0.0F});
            const std::uint32_t peg_bottom_center = 2U * peg_segments;
            const std::uint32_t peg_top_center = peg_bottom_center + 1U;
            for (std::uint32_t segment = 0U; segment < peg_segments; ++segment) {
                const std::uint32_t next = (segment + 1U) % peg_segments;
                const std::uint32_t bottom = 2U * segment;
                const std::uint32_t top = bottom + 1U;
                const std::uint32_t next_bottom = 2U * next;
                const std::uint32_t next_top = next_bottom + 1U;
                triangles.push_back({bottom, next_bottom, top});
                triangles.push_back({top, next_bottom, next_top});
                triangles.push_back({peg_bottom_center, bottom, next_bottom});
                triangles.push_back({peg_top_center, next_top, top});
            }
            rigid_vertices_.upload(vertices.data(), vertices.size(), stream);
            rigid_triangles_.upload(triangles.data(), triangles.size(), stream);
            rigid_views_[0] = {{rigid_vertices_.get(), bowl_vertex_count,
                    rigid_triangles_.get(), bowl_triangle_count},
                waterlab::bowl_center, {0.0F, 0.0F, 0.0F, 1.0F},
                {waterlab::bowl_inner_radius + waterlab::bowl_wall_thickness,
                 waterlab::bowl_inner_radius + waterlab::bowl_wall_thickness,
                 waterlab::bowl_inner_radius + waterlab::bowl_wall_thickness}};
            rigid_sphere_ = waterlab::gallery::initial_rigid_sphere(options_.context);
            rigid_views_[1] = {{rigid_vertices_.get() + sphere_vertex_first,
                    sphere.positions.size(),
                    rigid_triangles_.get() + sphere_triangle_first,
                    sphere.triangles.size()},
                rigid_sphere_.center, {0.0F, 0.0F, 0.0F, 1.0F},
                {rigid_sphere_.radius, rigid_sphere_.radius, rigid_sphere_.radius}};
            rigid_sphere_view_index_ = 1U;
            const DeviceMeshView peg_mesh{
                rigid_vertices_.get() + peg_vertex_first,
                2U * peg_segments + 2U,
                rigid_triangles_.get() + peg_triangle_first,
                4U * peg_segments};
            for (std::uint32_t peg = 0U; peg < waterlab::bowl_peg_count; ++peg) {
                rigid_views_[2U + peg] = {peg_mesh,
                    waterlab::bowl_peg(peg), {0.0F, 0.0F, 0.0F, 1.0F},
                    {waterlab::bowl_peg_radius, 0.5F * waterlab::bowl_peg_height,
                     waterlab::bowl_peg_radius}};
            }
            rigid_count_ = 2U + waterlab::bowl_peg_count;
            return;
        }
        if (options_.context == ExampleContext::rope_rigid ||
            options_.context == ExampleContext::rope_bridge) {
            constexpr std::array<float3, 8U> cube_vertices{{
                {-1.0F, -1.0F, -1.0F}, {1.0F, -1.0F, -1.0F},
                {1.0F, 1.0F, -1.0F}, {-1.0F, 1.0F, -1.0F},
                {-1.0F, -1.0F, 1.0F}, {1.0F, -1.0F, 1.0F},
                {1.0F, 1.0F, 1.0F}, {-1.0F, 1.0F, 1.0F},
            }};
            constexpr std::array<uint3, 12U> cube_triangles{{
                {0U,2U,1U}, {0U,3U,2U}, {4U,5U,6U}, {4U,6U,7U},
                {0U,1U,5U}, {0U,5U,4U}, {3U,7U,6U}, {3U,6U,2U},
                {0U,4U,7U}, {0U,7U,3U}, {1U,2U,6U}, {1U,6U,5U},
            }};
            const auto sphere = waterlab::make_geodesic_sphere(6U, 1.0F);
            std::vector<float3> vertices(sphere.positions.begin(), sphere.positions.end());
            std::vector<uint3> triangles(sphere.triangles.begin(), sphere.triangles.end());
            const std::uint32_t cube_vertex_first =
                static_cast<std::uint32_t>(vertices.size());
            const std::uint32_t cube_triangle_first =
                static_cast<std::uint32_t>(triangles.size());
            vertices.insert(vertices.end(), cube_vertices.begin(), cube_vertices.end());
            triangles.insert(triangles.end(), cube_triangles.begin(), cube_triangles.end());
            rigid_vertices_.upload(vertices.data(), vertices.size(), stream);
            rigid_triangles_.upload(triangles.data(), triangles.size(), stream);
            rigid_sphere_ = waterlab::gallery::initial_rigid_sphere(
                options_.context, rope_node_count());
            rigid_views_[0] = {{rigid_vertices_.get(), sphere.positions.size(),
                    rigid_triangles_.get(), sphere.triangles.size()},
                rigid_sphere_.center, {0,0,0,1},
                {rigid_sphere_.radius, rigid_sphere_.radius, rigid_sphere_.radius}};
            rigid_sphere_view_index_ = 0U;
            const DeviceMeshView cube{rigid_vertices_.get() + cube_vertex_first,
                cube_vertices.size(), rigid_triangles_.get() + cube_triangle_first,
                cube_triangles.size()};
            if (options_.context == ExampleContext::rope_rigid) {
                rigid_views_[1] = {cube,
                    {0.0F, waterlab::course_floor_y - 0.05F, -1.2F}, {0,0,0,1},
                    {3.1F, 0.05F, 2.7F}};
                rigid_views_[2] = {cube, waterlab::rope_post_center, {0,0,0,1},
                    {waterlab::rope_post_radius, 0.5F * waterlab::rope_post_height,
                     waterlab::rope_post_radius}};
            } else {
                for (std::uint32_t side = 0U; side < 2U; ++side) {
                    const float sign = side == 0U ? -1.0F : 1.0F;
                    const float inner = sign * waterlab::rope_bridge_land_inner_z;
                    const float outer = sign * waterlab::rope_bridge_land_outer_z;
                    rigid_views_[1U+side] = {cube,
                        {0.0F,waterlab::rope_bridge_land_y-0.06F,
                         0.5F*(inner+outer)}, {0,0,0,1},
                        {waterlab::rope_bridge_land_half_width,0.06F,
                         0.5F*std::fabs(outer-inner)}};
                }
            }
            rigid_count_ = 3U;
            return;
        }
        if (options_.context == ExampleContext::cloth_rigid ||
            options_.context == ExampleContext::soft_body_rigid ||
            options_.context == ExampleContext::particles_cloth ||
            options_.context == ExampleContext::soft_body_cloth) {
            constexpr std::array<float3, 8U> cube_vertices{{
                {-1.0F, -1.0F, -1.0F}, {1.0F, -1.0F, -1.0F},
                {1.0F, 1.0F, -1.0F}, {-1.0F, 1.0F, -1.0F},
                {-1.0F, -1.0F, 1.0F}, {1.0F, -1.0F, 1.0F},
                {1.0F, 1.0F, 1.0F}, {-1.0F, 1.0F, 1.0F},
            }};
            constexpr std::array<uint3, 12U> cube_triangles{{
                {0U, 2U, 1U}, {0U, 3U, 2U}, {4U, 5U, 6U}, {4U, 6U, 7U},
                {0U, 1U, 5U}, {0U, 5U, 4U}, {3U, 7U, 6U}, {3U, 6U, 2U},
                {0U, 4U, 7U}, {0U, 7U, 3U}, {1U, 2U, 6U}, {1U, 6U, 5U},
            }};
            const auto sphere = waterlab::make_geodesic_sphere(6U, 1.0F);
            std::vector<float3> vertices(sphere.positions.begin(), sphere.positions.end());
            std::vector<uint3> triangles(sphere.triangles.begin(), sphere.triangles.end());
            const std::uint32_t cube_vertex_first =
                static_cast<std::uint32_t>(vertices.size());
            const std::uint32_t cube_triangle_first =
                static_cast<std::uint32_t>(triangles.size());
            vertices.insert(vertices.end(), cube_vertices.begin(), cube_vertices.end());
            for (const uint3 triangle : cube_triangles) {
                triangles.push_back(triangle);
            }
            rigid_vertices_.upload(vertices.data(), vertices.size(), stream);
            rigid_triangles_.upload(triangles.data(), triangles.size(), stream);
            rigid_sphere_ = waterlab::gallery::initial_rigid_sphere(
                options_.context, rope_node_count());
            rigid_views_[0] = {{rigid_vertices_.get(), sphere.positions.size(),
                    rigid_triangles_.get(), sphere.triangles.size()},
                rigid_sphere_.center, {0.0F, 0.0F, 0.0F, 1.0F},
                {rigid_sphere_.radius, rigid_sphere_.radius, rigid_sphere_.radius}};
            rigid_sphere_view_index_ = 0U;
            const DeviceMeshView cube{rigid_vertices_.get() + cube_vertex_first,
                cube_vertices.size(), rigid_triangles_.get() + cube_triangle_first,
                cube_triangles.size()};
            const float3 c = options_.context == ExampleContext::soft_body_rigid
                ? waterlab::low_gallery_box_center : waterlab::gallery_box_center;
            const float3 h = options_.context == ExampleContext::soft_body_rigid
                ? waterlab::low_gallery_box_half_extents : waterlab::gallery_box_half_extents;
            constexpr float thickness = 0.045F;
            rigid_views_[1] = {cube, {c.x, c.y - h.y - thickness, c.z}, {0,0,0,1},
                {h.x, thickness, h.z}};
            rigid_views_[2] = {cube, {c.x, c.y + h.y + thickness, c.z}, {0,0,0,1},
                {h.x, thickness, h.z}};
            rigid_views_[3] = {cube, {c.x - h.x - thickness, c.y, c.z}, {0,0,0,1},
                {thickness, h.y, h.z}};
            rigid_views_[4] = {cube, {c.x + h.x + thickness, c.y, c.z}, {0,0,0,1},
                {thickness, h.y, h.z}};
            rigid_views_[5] = {cube, {c.x, c.y, c.z - h.z - thickness}, {0,0,0,1},
                {h.x, h.y, thickness}};
            rigid_views_[6] = {cube, {c.x, c.y, c.z + h.z + thickness}, {0,0,0,1},
                {h.x, h.y, thickness}};
            rigid_count_ = 7U;
            if (options_.context == ExampleContext::particles_cloth) {
                constexpr float support_half_width = 0.018F;
                constexpr float support_half_height = 0.025F;
                const float support_y = waterlab::cloth_basin_center.y - 0.040F;
                for (std::uint32_t line = 0U; line < 4U; ++line) {
                    const float alpha = static_cast<float>(line) / 3.0F;
                    const float x = waterlab::cloth_basin_center.x +
                        (2.0F * alpha - 1.0F) *
                            waterlab::cloth_basin_inner_half_extents.x;
                    const float z = waterlab::cloth_basin_center.z +
                        (2.0F * alpha - 1.0F) *
                            waterlab::cloth_basin_inner_half_extents.y;
                    rigid_views_[7U + line] = {cube,
                        {x, support_y, waterlab::cloth_basin_center.z}, {0,0,0,1},
                        {support_half_width, support_half_height,
                         waterlab::cloth_basin_inner_half_extents.y + support_half_width}};
                    rigid_views_[11U + line] = {cube,
                        {waterlab::cloth_basin_center.x, support_y, z}, {0,0,0,1},
                        {waterlab::cloth_basin_inner_half_extents.x + support_half_width,
                         support_half_height, support_half_width}};
                }
                rigid_count_ = 15U;
            }
            if (options_.context == ExampleContext::soft_body_cloth) {
                for (std::uint32_t wall = 0U; wall < 6U; ++wall)
                    rigid_views_[wall] = rigid_views_[wall + 1U];
                rigid_sphere_view_index_ = std::numeric_limits<std::uint32_t>::max();
                rigid_count_ = 6U;
            }
            return;
        }
        if (options_.context == ExampleContext::soft_body_fluid) {
            constexpr std::uint32_t segments = 64U;
            constexpr float shell_half_thickness = 0.045F;
            const float outer = waterlab::water_wheel_shell_radius + shell_half_thickness;
            const float inner = waterlab::water_wheel_shell_radius - shell_half_thickness;
            std::vector<float3> vertices;
            std::vector<uint3> triangles;
            vertices.reserve(4U * segments + 8U);
            triangles.reserve(8U * segments + 12U);
            for (std::uint32_t segment = 0U; segment < segments; ++segment) {
                const float angle = 2.0F * 3.14159265358979323846F *
                    static_cast<float>(segment) / static_cast<float>(segments);
                const float cosine = std::cos(angle);
                const float sine = std::sin(angle);
                vertices.push_back({outer * cosine, outer * sine,
                    -waterlab::water_wheel_half_depth});
                vertices.push_back({outer * cosine, outer * sine,
                    waterlab::water_wheel_half_depth});
                vertices.push_back({inner * cosine, inner * sine,
                    -waterlab::water_wheel_half_depth});
                vertices.push_back({inner * cosine, inner * sine,
                    waterlab::water_wheel_half_depth});
            }
            for (std::uint32_t segment = 0U; segment < segments; ++segment) {
                const std::uint32_t next = (segment + 1U) % segments;
                const float middle_angle = 2.0F * 3.14159265358979323846F *
                    (static_cast<float>(segment) + 0.5F) /
                    static_cast<float>(segments);
                if (waterlab::water_wheel_shell_opening({
                    waterlab::water_wheel_shell_radius * std::cos(middle_angle),
                    waterlab::water_wheel_shell_radius * std::sin(middle_angle), 0.0F}))
                    continue;
                const std::uint32_t a = 4U * segment;
                const std::uint32_t b = 4U * next;
                triangles.insert(triangles.end(), {
                    {a, b, a + 1U}, {a + 1U, b, b + 1U},
                    {a + 2U, a + 3U, b + 2U}, {a + 3U, b + 3U, b + 2U},
                    {a, a + 2U, b}, {a + 2U, b + 2U, b},
                    {a + 1U, b + 1U, a + 3U}, {a + 3U, b + 1U, b + 3U}});
            }
            const std::uint32_t shell_vertex_count =
                static_cast<std::uint32_t>(vertices.size());
            const std::uint32_t shell_triangle_count =
                static_cast<std::uint32_t>(triangles.size());
            const std::uint32_t core_vertex_first = shell_vertex_count;
            const std::uint32_t core_triangle_first = shell_triangle_count;
            for (std::uint32_t segment = 0U; segment < segments; ++segment) {
                const float angle = 2.0F * 3.14159265358979323846F *
                    static_cast<float>(segment) / static_cast<float>(segments);
                const float x = waterlab::water_wheel_hub_radius * std::cos(angle);
                const float y = waterlab::water_wheel_hub_radius * std::sin(angle);
                vertices.push_back({x, y, -waterlab::water_wheel_half_depth});
                vertices.push_back({x, y, waterlab::water_wheel_half_depth});
            }
            vertices.push_back({0.0F, 0.0F, -waterlab::water_wheel_half_depth});
            vertices.push_back({0.0F, 0.0F, waterlab::water_wheel_half_depth});
            const std::uint32_t bottom_center = 2U * segments;
            const std::uint32_t top_center = bottom_center + 1U;
            for (std::uint32_t segment = 0U; segment < segments; ++segment) {
                const std::uint32_t next = (segment + 1U) % segments;
                const std::uint32_t bottom = 2U * segment;
                const std::uint32_t top = bottom + 1U;
                const std::uint32_t next_bottom = 2U * next;
                const std::uint32_t next_top = next_bottom + 1U;
                triangles.push_back({bottom, next_bottom, top});
                triangles.push_back({top, next_bottom, next_top});
                triangles.push_back({bottom_center, bottom, next_bottom});
                triangles.push_back({top_center, next_top, top});
            }
            const std::uint32_t core_vertex_count =
                static_cast<std::uint32_t>(vertices.size()) - core_vertex_first;
            const std::uint32_t core_triangle_count =
                static_cast<std::uint32_t>(triangles.size()) - core_triangle_first;
            const std::uint32_t cube_vertex_first =
                static_cast<std::uint32_t>(vertices.size());
            const std::uint32_t cube_triangle_first =
                static_cast<std::uint32_t>(triangles.size());
            constexpr std::array<float3, 8U> cube_vertices{{
                {-1,-1,-1}, {1,-1,-1}, {1,1,-1}, {-1,1,-1},
                {-1,-1,1}, {1,-1,1}, {1,1,1}, {-1,1,1}}};
            constexpr std::array<uint3, 12U> cube_triangles{{
                {0,2,1}, {0,3,2}, {4,5,6}, {4,6,7}, {0,1,5}, {0,5,4},
                {3,7,6}, {3,6,2}, {0,4,7}, {0,7,3}, {1,2,6}, {1,6,5}}};
            vertices.insert(vertices.end(), cube_vertices.begin(), cube_vertices.end());
            for (const uint3 triangle : cube_triangles) triangles.push_back(triangle);
            rigid_vertices_.upload(vertices.data(), vertices.size(), stream);
            rigid_triangles_.upload(triangles.data(), triangles.size(), stream);
            rigid_views_[0] = {{rigid_vertices_.get(), shell_vertex_count,
                    rigid_triangles_.get(), shell_triangle_count},
                waterlab::water_wheel_center, {0,0,0,1}, {1.0F, 1.0F, 1.0F}};
            rigid_views_[1] = {{rigid_vertices_.get() + core_vertex_first,
                    core_vertex_count,
                    rigid_triangles_.get() + core_triangle_first,
                    core_triangle_count},
                waterlab::water_wheel_center, {0,0,0,1}, {1.0F, 1.0F, 1.0F}};
            const DeviceMeshView cube{rigid_vertices_.get() + cube_vertex_first,
                cube_vertices.size(), rigid_triangles_.get() + cube_triangle_first,
                cube_triangles.size()};
            rigid_wheel_first_fin_ = 2U;
            for (std::uint32_t fin = 0U; fin < waterlab::water_wheel_fin_count; ++fin) {
                const float angle = 2.0F * 3.14159265358979323846F *
                    static_cast<float>(fin) /
                    static_cast<float>(waterlab::water_wheel_fin_count);
                const float middle = 0.5F * (waterlab::water_wheel_radius +
                    waterlab::water_wheel_fin_outer_radius);
                rigid_views_[2U + fin] = {cube,
                    {waterlab::water_wheel_center.x + middle * std::cos(angle),
                     waterlab::water_wheel_center.y + middle * std::sin(angle),
                     waterlab::water_wheel_center.z},
                    {0.0F, 0.0F, std::sin(0.5F * angle), std::cos(0.5F * angle)},
                    {0.5F * (waterlab::water_wheel_fin_outer_radius -
                        waterlab::water_wheel_radius),
                     waterlab::water_wheel_fin_thickness,
                     waterlab::water_wheel_half_depth}};
            }
            constexpr float ramp_half_thickness = 0.045F;
            constexpr float ramp_half_width = waterlab::water_wheel_ground_half_depth;
            const float ramp_angle = std::atan(waterlab::water_wheel_ramp_gradient);
            const float4 ramp_orientation{0.0F, 0.0F,
                std::sin(0.5F * ramp_angle), std::cos(0.5F * ramp_angle)};
            const std::uint32_t inlet_view = 2U + waterlab::water_wheel_fin_count;
            const float inlet_middle = 0.5F *
                (waterlab::water_wheel_inlet_start_x + waterlab::water_wheel_entry_x);
            rigid_views_[inlet_view] = {cube,
                {inlet_middle, waterlab::water_wheel_inlet_height(inlet_middle) -
                    ramp_half_thickness, waterlab::water_wheel_center.z},
                ramp_orientation,
                {0.5F * (waterlab::water_wheel_entry_x -
                    waterlab::water_wheel_inlet_start_x),
                 ramp_half_thickness, ramp_half_width}};
            const std::uint32_t collector_view = inlet_view + 1U;
            const float collector_middle = 0.5F *
                (waterlab::water_wheel_collector_start_x +
                 waterlab::water_wheel_collector_end_x);
            rigid_views_[collector_view] = {cube,
                {collector_middle,
                 waterlab::water_wheel_collector_height(collector_middle) -
                    ramp_half_thickness,
                 waterlab::water_wheel_center.z},
                ramp_orientation,
                {0.5F * (waterlab::water_wheel_collector_end_x -
                    waterlab::water_wheel_collector_start_x),
                 ramp_half_thickness, ramp_half_width}};
            const DeviceMeshView cylinder{rigid_vertices_.get() + core_vertex_first,
                core_vertex_count, rigid_triangles_.get() + core_triangle_first,
                core_triangle_count};
            const std::uint32_t axle_view = collector_view + 1U;
            rigid_views_[axle_view] = {cylinder, waterlab::water_wheel_center,
                {0,0,0,1},
                {0.11F / waterlab::water_wheel_hub_radius,
                 0.11F / waterlab::water_wheel_hub_radius,
                 waterlab::water_wheel_axle_half_length /
                    waterlab::water_wheel_half_depth}};
            for (std::uint32_t side = 0U; side < 2U; ++side) {
                const float sign = side == 0U ? -1.0F : 1.0F;
                rigid_views_[axle_view + 1U + side] = {cylinder,
                    {waterlab::water_wheel_center.x, waterlab::water_wheel_center.y,
                     waterlab::water_wheel_center.z + sign *
                        waterlab::water_wheel_outer_disk_offset},
                    {0,0,0,1},
                    {waterlab::water_wheel_radius / waterlab::water_wheel_hub_radius,
                     waterlab::water_wheel_radius / waterlab::water_wheel_hub_radius,
                     waterlab::water_wheel_outer_disk_half_thickness /
                        waterlab::water_wheel_half_depth}};
            }
            const std::uint32_t first_platform = axle_view + 3U;
            for (std::uint32_t side = 0U; side < 2U; ++side) {
                const float sign = side == 0U ? -1.0F : 1.0F;
                const float inner = waterlab::water_wheel_center.x + sign *
                    waterlab::water_wheel_top_platform_gap_half_width;
                const float outer = waterlab::water_wheel_center.x + sign *
                    waterlab::water_wheel_top_platform_outer_x;
                rigid_views_[first_platform + side] = {cube,
                    {0.5F * (inner + outer),
                     waterlab::water_wheel_top_platform_y - 0.035F,
                     waterlab::water_wheel_stage_z},
                    {0,0,0,1},
                    {0.5F * std::fabs(outer - inner), 0.035F,
                     waterlab::water_wheel_top_platform_half_depth}};
            }
            const std::uint32_t first_bumper=first_platform+2U;
            for (std::uint32_t side=0U;side<2U;++side) {
                const float sign=side==0U ? -1.0F : 1.0F;
                rigid_views_[first_bumper+side]={cube,
                    {waterlab::water_wheel_center.x,
                     waterlab::water_wheel_top_platform_y+
                        waterlab::water_wheel_top_bumper_height,
                     waterlab::water_wheel_stage_z+sign*(
                        waterlab::water_wheel_top_platform_half_depth+
                        waterlab::water_wheel_top_bumper_thickness)},
                    {0,0,0,1},
                    {waterlab::water_wheel_top_platform_outer_x,
                     waterlab::water_wheel_top_bumper_height,
                     waterlab::water_wheel_top_bumper_thickness}};
            }
            rigid_views_[first_bumper+2U]={cube,
                {waterlab::water_wheel_center.x+
                    waterlab::water_wheel_top_platform_outer_x+
                    waterlab::water_wheel_top_bumper_thickness,
                 waterlab::water_wheel_top_platform_y+
                    waterlab::water_wheel_top_bumper_height,
                 waterlab::water_wheel_stage_z},
                {0,0,0,1},
                {waterlab::water_wheel_top_bumper_thickness,
                 waterlab::water_wheel_top_bumper_height,
                 waterlab::water_wheel_top_platform_half_depth+
                    2.0F*waterlab::water_wheel_top_bumper_thickness}};
            rigid_count_ = 12U + waterlab::water_wheel_fin_count;
            return;
        }
        // The course exposes the same board, rails, and capped posts as the
        // native analytic renderer/collider. Boxes share one unit cube; posts
        // share one unit cylinder.
        constexpr std::array<float3, 8U> vertices{{
            {-1.0F, -1.0F, -1.0F}, {1.0F, -1.0F, -1.0F},
            {1.0F, 1.0F, -1.0F}, {-1.0F, 1.0F, -1.0F},
            {-1.0F, -1.0F, 1.0F}, {1.0F, -1.0F, 1.0F},
            {1.0F, 1.0F, 1.0F}, {-1.0F, 1.0F, 1.0F},
        }};
        constexpr std::array<uint3, 12U> triangles{{
            {0U, 2U, 1U}, {0U, 3U, 2U}, {4U, 5U, 6U}, {4U, 6U, 7U},
            {0U, 1U, 5U}, {0U, 5U, 4U}, {3U, 7U, 6U}, {3U, 6U, 2U},
            {0U, 4U, 7U}, {0U, 7U, 3U}, {1U, 2U, 6U}, {1U, 6U, 5U},
        }};
        std::vector<float3> course_vertices(vertices.begin(), vertices.end());
        std::vector<uint3> course_triangles(triangles.begin(), triangles.end());
        constexpr std::uint32_t cylinder_segments = 24U;
        const std::uint32_t cylinder_vertex_first =
            static_cast<std::uint32_t>(course_vertices.size());
        const std::uint32_t cylinder_triangle_first =
            static_cast<std::uint32_t>(course_triangles.size());
        for (std::uint32_t segment = 0U; segment < cylinder_segments; ++segment) {
            const float angle = 2.0F * 3.14159265358979323846F *
                static_cast<float>(segment) / static_cast<float>(cylinder_segments);
            const float x = std::cos(angle);
            const float z = std::sin(angle);
            course_vertices.push_back({x, -1.0F, z});
            course_vertices.push_back({x, 1.0F, z});
        }
        course_vertices.push_back({0.0F, -1.0F, 0.0F});
        course_vertices.push_back({0.0F, 1.0F, 0.0F});
        const std::uint32_t bottom_center = 2U * cylinder_segments;
        const std::uint32_t top_center = bottom_center + 1U;
        for (std::uint32_t segment = 0U; segment < cylinder_segments; ++segment) {
            const std::uint32_t next = (segment + 1U) % cylinder_segments;
            const std::uint32_t bottom = 2U * segment;
            const std::uint32_t top = bottom + 1U;
            const std::uint32_t next_bottom = 2U * next;
            const std::uint32_t next_top = next_bottom + 1U;
            course_triangles.push_back({bottom, top, next_bottom});
            course_triangles.push_back({top, next_top, next_bottom});
            course_triangles.push_back({bottom_center, bottom, next_bottom});
            course_triangles.push_back({top_center, next_top, top});
        }
        rigid_vertices_.upload(
            course_vertices.data(), course_vertices.size(), stream);
        rigid_triangles_.upload(
            course_triangles.data(), course_triangles.size(), stream);

        const DeviceMeshView cube{rigid_vertices_.get(), vertices.size(),
            rigid_triangles_.get(), triangles.size()};
        const float middle_z =
            0.5F * (waterlab::course_near_z + waterlab::course_far_z);
        const float half_length =
            0.5F * (waterlab::course_near_z - waterlab::course_far_z);
        const float rail_y =
            waterlab::course_floor_y + 0.5F * waterlab::course_rail_height;
        const float side_x =
            waterlab::course_half_width + 0.5F * waterlab::course_rail_thickness;
        const float end_half_width =
            waterlab::course_half_width + waterlab::course_rail_thickness;
        rigid_views_[0] = {cube,
            {0.0F, waterlab::course_floor_y -
                0.5F * waterlab::course_floor_thickness, middle_z},
            {0.0F, 0.0F, 0.0F, 1.0F},
            {waterlab::course_half_width + waterlab::course_rail_thickness,
                0.5F * waterlab::course_floor_thickness,
                half_length + waterlab::course_rail_thickness}};
        rigid_views_[1] = {cube, {-side_x, rail_y, middle_z},
            {0.0F, 0.0F, 0.0F, 1.0F},
            {0.5F * waterlab::course_rail_thickness,
                0.5F * waterlab::course_rail_height,
                half_length + waterlab::course_rail_thickness}};
        rigid_views_[2] = rigid_views_[1];
        rigid_views_[2].translation.x = side_x;
        rigid_views_[3] = {cube,
            {0.0F, rail_y,
                waterlab::course_near_z + 0.5F * waterlab::course_rail_thickness},
            {0.0F, 0.0F, 0.0F, 1.0F},
            {end_half_width, 0.5F * waterlab::course_rail_height,
                0.5F * waterlab::course_rail_thickness}};
        rigid_views_[4] = rigid_views_[3];
        rigid_views_[4].translation.z =
            waterlab::course_far_z - 0.5F * waterlab::course_rail_thickness;
        const DeviceMeshView cylinder{
            rigid_vertices_.get() + cylinder_vertex_first,
            2U * cylinder_segments + 2U,
            rigid_triangles_.get() + cylinder_triangle_first,
            4U * cylinder_segments};
        for (std::uint32_t peg = 0U; peg < waterlab::course_peg_count; ++peg) {
            const float3 base = waterlab::course_peg(peg);
            rigid_views_[5U + peg] = {cylinder,
                {base.x, waterlab::course_floor_y +
                    0.5F * waterlab::course_peg_height, base.z},
                {0.0F, 0.0F, 0.0F, 1.0F},
                {waterlab::course_peg_radius,
                    0.5F * waterlab::course_peg_height,
                    waterlab::course_peg_radius}};
        }
        rigid_count_ = 5U + waterlab::course_peg_count;
    }

    void refresh_views() noexcept
    {
        if (rigid_sphere_view_index_ != std::numeric_limits<std::uint32_t>::max()) {
            rigid_views_[rigid_sphere_view_index_].translation = rigid_sphere_.center;
        }
        if (rigid_wheel_first_fin_ != std::numeric_limits<std::uint32_t>::max()) {
            constexpr float two_pi = 6.28318530717958647692F;
            const float middle = 0.5F * (waterlab::water_wheel_radius +
                waterlab::water_wheel_fin_outer_radius);
            for (std::uint32_t fin = 0U; fin < waterlab::water_wheel_fin_count; ++fin) {
                const float angle = water_wheel_.angle + two_pi * static_cast<float>(fin) /
                    static_cast<float>(waterlab::water_wheel_fin_count);
                auto& view = rigid_views_[rigid_wheel_first_fin_ + fin];
                view.translation = {
                    waterlab::water_wheel_center.x + middle * std::cos(angle),
                    waterlab::water_wheel_center.y + middle * std::sin(angle),
                    waterlab::water_wheel_center.z};
                view.orientation = {0.0F, 0.0F,
                    std::sin(0.5F * angle), std::cos(0.5F * angle)};
            }
            const std::uint32_t first_outer_rim =
                rigid_wheel_first_fin_ + waterlab::water_wheel_fin_count + 3U;
            for (std::uint32_t side = 0U; side < 2U; ++side) {
                rigid_views_[first_outer_rim + side].orientation = {0.0F, 0.0F,
                    std::sin(0.5F * water_wheel_.angle),
                    std::cos(0.5F * water_wheel_.angle)};
            }
        }
        particle_count_ = 0U;
        surface_count_ = 0U;
        lattice_count_ = 0U;
        if (hybrid_ && (has_component(info_->components, Component::fluid_particles) ||
                           has_component(info_->components, Component::hand_particles))) {
            particle_views_[particle_count_++] = {hybrid_->particle_positions(),
                hybrid_->particle_velocities(), hybrid_->statistics().particle_count,
                hybrid_->particle_radius()};
        }
        if (hybrid_ && has_component(info_->components, Component::water_skin)) {
            surface_views_[surface_count_++] = {hybrid_->skin_mesh(),
                hybrid_->skin_normals().vertex_normals(), nullptr, nullptr};
        }
        if (deformable_ && (has_component(info_->components, Component::cloth) ||
                               has_component(info_->components, Component::soft_body))) {
            const waterlab::SoftBodyRenderView view = deformable_->render_view();
            surface_views_[surface_count_++] = {
                {view.positions, view.vertex_count, view.triangles, view.triangle_count},
                view.vertex_normals, view.texcoords, view.triangle_active};
            const waterlab::SoftBodyLatticeView lattice = deformable_->lattice_view();
            lattice_views_[lattice_count_++] = {lattice.positions, lattice.flags,
                lattice.edges, lattice.active_edges, lattice.voxel_count,
                lattice.voxels_per_instance, lattice.edges_per_instance,
                lattice.instance_count, lattice.voxel_radius};
        }
    }

    void refresh_statistics(float gpu_time_ms) noexcept
    {
        statistics_ = {};
        statistics_.surface_count = surface_count_;
        statistics_.rigid_body_count = rigid_count_;
        statistics_.last_gpu_time_ms = gpu_time_ms;
        statistics_.allocated_bytes = rigid_vertices_.size() * sizeof(float3) +
            rigid_triangles_.size() * sizeof(uint3);
        if (hybrid_) {
            const auto hybrid = hybrid_->statistics();
            statistics_.frame_index = hybrid.frame_index;
            statistics_.particle_count = particle_count_ == 0U
                ? 0U : hybrid.particle_count;
            statistics_.finite_failure_count += hybrid.finite_failures;
            statistics_.allocated_bytes += hybrid_->allocated_bytes();
        }
        if (deformable_) {
            const auto deformable = deformable_->statistics();
            if (!hybrid_) statistics_.frame_index = deformable.frame_index;
            statistics_.finite_failure_count += deformable.finite_failure_count;
            statistics_.broken_connection_count = deformable.broken_edge_count;
            statistics_.allocated_bytes += deformable_->allocated_bytes();
        }
    }

    void advance(cudaStream_t stream)
    {
        float gpu_time = 0.0F;
        if (hybrid_) {
            const bool dynamic_sphere =
                options_.context == ExampleContext::particle_bowl ||
                options_.context == ExampleContext::particles_cloth;
            const auto timing = hybrid_->step(
                {}, 0.0F, stream, deformable_.get(), false,
                dynamic_sphere ? &rigid_sphere_ : nullptr,
                options_.context == ExampleContext::soft_body_fluid
                    ? &water_wheel_ : nullptr);
            gpu_time = timing.gpu_total_ms();
        } else if (deformable_) {
            const bool rolling_rigid = options_.context == ExampleContext::cloth_rigid ||
                options_.context == ExampleContext::soft_body_rigid ||
                options_.context == ExampleContext::rope_rigid ||
                options_.context == ExampleContext::rope_bridge;
            waterlab::SoftBodyTimings timing{};
            if (options_.context == ExampleContext::rope_rigid) {
                const auto lattice = deformable_->lattice_view();
                timing = deformable_->step_with_tethered_rigid_sphere(
                    rigid_sphere_, lattice.voxels_per_instance - 1U,
                    rigid_sphere_.radius + lattice.voxel_radius,
                    resolved_physics_.gravity, stream);
            } else if (rolling_rigid) {
                timing = deformable_->step_with_rigid_sphere(
                    rigid_sphere_, resolved_physics_.gravity, stream);
            } else {
                timing = deformable_->step(resolved_physics_.gravity, stream);
            }
            gpu_time = timing.gpu_total_ms();
        }
        refresh_views();
        refresh_statistics(gpu_time);
    }

    void reset(cudaStream_t stream)
    {
        if (hybrid_) hybrid_->reset(stream);
        if (deformable_) {
            if (options_.context == ExampleContext::soft_body_fluid) {
                deformable_->set_pinned_rotation_z(
                    waterlab::water_wheel_center, 0.0F, stream);
            }
            deformable_->reset(stream);
            waterlab::gallery::initialize_context_motion(
                options_.context, *deformable_);
        }
        if (options_.context == ExampleContext::cloth_rigid ||
            options_.context == ExampleContext::soft_body_rigid ||
            options_.context == ExampleContext::particle_bowl ||
            options_.context == ExampleContext::particles_cloth ||
            options_.context == ExampleContext::rope_rigid ||
            options_.context == ExampleContext::rope_bridge) {
            rigid_sphere_ = waterlab::gallery::initial_rigid_sphere(
                options_.context, rope_node_count());
        }
        if (options_.context == ExampleContext::soft_body_fluid)
            water_wheel_ = {};
        refresh_views();
        refresh_statistics(0.0F);
    }

    GallerySimulationOptions options_{};
    ResolvedPhysicsOptions resolved_physics_{};
    std::string asset_path_;
    const ExampleContextInfo* info_{};
    std::unique_ptr<waterlab::HybridDroplet> hybrid_;
    std::unique_ptr<waterlab::SoftBodyCourse> deformable_;
    waterlab::RigidSphereState rigid_sphere_{};
    waterlab::WaterWheelState water_wheel_{};
    DeviceBuffer<float3> rigid_vertices_;
    DeviceBuffer<uint3> rigid_triangles_;
    std::array<ParticleRenderView, 1U> particle_views_{};
    std::array<SurfaceRenderView, 2U> surface_views_{};
    std::array<RigidBodyRenderView, 12U + waterlab::water_wheel_fin_count> rigid_views_{};
    std::array<LatticeRenderView, 1U> lattice_views_{};
    std::uint32_t particle_count_{};
    std::uint32_t surface_count_{};
    std::uint32_t rigid_count_{};
    std::uint32_t rigid_sphere_view_index_{
        std::numeric_limits<std::uint32_t>::max()};
    std::uint32_t rigid_wheel_first_fin_{
        std::numeric_limits<std::uint32_t>::max()};
    std::uint32_t lattice_count_{};
    GallerySimulationStatistics statistics_{};
};

GallerySimulation::GallerySimulation() noexcept = default;
GallerySimulation::~GallerySimulation() = default;
GallerySimulation::GallerySimulation(GallerySimulation&&) noexcept = default;
GallerySimulation& GallerySimulation::operator=(GallerySimulation&&) noexcept = default;

Status GallerySimulation::create(
    GallerySimulationOptions options,
    GallerySimulation& output,
    cudaStream_t stream) noexcept
{
    return output.initialize(options, stream);
}

Status GallerySimulation::initialize(
    GallerySimulationOptions options, cudaStream_t stream) noexcept
{
    if (context_info(options.context) == nullptr) {
        return invalid("unknown gallery simulation context");
    }
    if (!valid(options)) {
        return invalid("invalid gallery simulation options");
    }
    if (requires_soft_body_asset(options.context) &&
        options.soft_body_asset_path.empty()) {
        return invalid("this gallery context requires a soft-body .msb asset path");
    }
    try {
        auto candidate = std::make_unique<Impl>(options, stream);
        impl_ = std::move(candidate);
        return success();
    } catch (const waterlab::detail::StatusException& error) {
        return translated(error,
            "a simulation dependency failed during gallery initialization");
    } catch (const std::bad_alloc&) {
        return allocation_failure("host allocation failed while initializing simulation");
    } catch (const std::invalid_argument&) {
        return invalid("invalid gallery simulation options or soft-body asset");
    } catch (...) {
        return internal("gallery simulation initialization failed");
    }
}

Status GallerySimulation::step(cudaStream_t stream) noexcept
{
    if (!impl_) return invalid("gallery simulation is not initialized");
    try {
        impl_->advance(stream);
        if (impl_->statistics_.finite_failure_count != 0U) {
            return internal("gallery simulation produced non-finite state");
        }
        return success();
    } catch (const waterlab::detail::StatusException& error) {
        return translated(error,
            "a simulation dependency failed while stepping gallery simulation");
    } catch (...) {
        return internal("gallery simulation step failed");
    }
}

Status GallerySimulation::reset(cudaStream_t stream) noexcept
{
    if (!impl_) return invalid("gallery simulation is not initialized");
    try {
        impl_->reset(stream);
        return success();
    } catch (const waterlab::detail::StatusException& error) {
        return translated(error,
            "a simulation dependency failed while resetting gallery simulation");
    } catch (...) {
        return internal("gallery simulation reset failed");
    }
}

Status GallerySimulation::resize_particles(
    std::uint32_t active_count, cudaStream_t stream) noexcept
{
    if (!impl_ || !impl_->hybrid_ || impl_->particle_count_ == 0U) {
        return invalid("gallery context has no fluid particles");
    }
    const std::uint32_t capacity = impl_->hybrid_->options().particle_capacity;
    if (active_count < 256U || active_count > capacity) {
        return invalid("active particle count is outside the reserved capacity");
    }
    try {
        impl_->hybrid_->resize_particles(active_count, stream);
        impl_->refresh_views();
        impl_->refresh_statistics(0.0F);
        return success();
    } catch (const waterlab::detail::StatusException& error) {
        return translated(error, "a CUDA dependency failed while resizing particles");
    } catch (...) {
        return internal("gallery particle resize failed");
    }
}

bool GallerySimulation::initialized() const noexcept
{
    return static_cast<bool>(impl_);
}

ExampleContext GallerySimulation::context() const noexcept
{
    return impl_ ? impl_->options_.context : ExampleContext::particle_bowl;
}

GallerySimulationOptions GallerySimulation::options() const noexcept
{
    return impl_ ? impl_->options_ : GallerySimulationOptions{};
}

ResolvedPhysicsOptions GallerySimulation::resolved_physics() const noexcept
{
    return impl_ ? impl_->resolved_physics_ : ResolvedPhysicsOptions{};
}

GallerySimulationStatistics GallerySimulation::statistics() const noexcept
{
    return impl_ ? impl_->statistics_ : GallerySimulationStatistics{};
}

FrameRenderView GallerySimulation::render_view() const noexcept
{
    if (!impl_) return {};
    return {
        impl_->particle_count_ == 0U ? nullptr : impl_->particle_views_.data(),
        impl_->particle_count_,
        impl_->surface_count_ == 0U ? nullptr : impl_->surface_views_.data(),
        impl_->surface_count_,
        impl_->rigid_count_ == 0U ? nullptr : impl_->rigid_views_.data(),
        impl_->rigid_count_,
        impl_->lattice_count_ == 0U ? nullptr : impl_->lattice_views_.data(),
        impl_->lattice_count_};
}

SimulationBuilder& SimulationBuilder::timestep(float value) noexcept
{
    config_.timestep(value);
    return *this;
}

SimulationBuilder& SimulationBuilder::iterations(std::uint32_t value) noexcept
{
    config_.iterations(value);
    return *this;
}

SimulationBuilder& SimulationBuilder::particles(std::uint32_t value) noexcept
{
    config_.particles(value);
    return *this;
}

SimulationBuilder& SimulationBuilder::skin_frequency(std::uint32_t value) noexcept
{
    config_.skin_frequency(value);
    return *this;
}

SimulationBuilder& SimulationBuilder::rope_nodes(std::uint32_t value) noexcept
{
    config_.rope_nodes(value);
    return *this;
}

SimulationBuilder& SimulationBuilder::cloth_detail(std::uint32_t value) noexcept
{
    config_.cloth_resolution(value);
    return *this;
}

SimulationBuilder& SimulationBuilder::gravity(float3 value) noexcept
{
    gravity_ = value;
    return *this;
}

SimulationBuilder& SimulationBuilder::soft_body_asset(std::string path)
{
    asset_path_ = std::move(path);
    return *this;
}

Status SimulationBuilder::build(
    GallerySimulation& output, cudaStream_t stream) const noexcept
{
    const ConfigError error = validate(config_);
    if (error != ConfigError::none) return invalid(describe(error).data());
    GallerySimulationOptions options;
    options.context = config_.context;
    options.fixed_step.timestep = config_.fixed_timestep;
    options.solver_iterations_override = config_.solver_iterations;
    options.gravity_override = gravity_;
    options.particle_count_override = config_.particle_count;
    options.physical_skin_frequency_override = config_.physical_skin_frequency;
    options.rope_node_count_override = config_.rope_node_count;
    options.cloth_detail_override = config_.cloth_detail;
    options.soft_body_asset_path = asset_path_;
    return output.initialize(options, stream);
}

} // namespace meshprep::sim
