// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Thomas Butyn

#include "fluid_visuals.hpp"
#include "hybrid_lab.hpp"
#include "obstacle_course.hpp"
#include "soft_body.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

#ifndef MESHPREP_SOFT_BODY_ASSET_PATH
#define MESHPREP_SOFT_BODY_ASSET_PATH "assets/softbody/checker_cylinder.msb"
#endif

constexpr std::string_view latest_capture_file =
    "/tmp/meshprep-hybrid-captures/LAST_CAPTURE.txt";
constexpr std::array<char, 8> capture_magic{'M','P','H','C','A','P','1','\0'};
constexpr std::uint32_t minimum_capture_version = 4U;
constexpr std::uint32_t maximum_capture_version = 6U;
constexpr std::uint32_t maximum_capture_frames = 10'000U;
constexpr std::uint32_t maximum_records = 10'000'000U;
constexpr double minimum_active_area_ratio = 0.5;
constexpr double maximum_active_area_ratio = 2.0;
constexpr double maximum_active_edge_ratio = 2.0;
constexpr double minimum_active_triangle_fraction = 0.90;
// Conservative sampled guard against gross water-skin collapse.  This is a
// containment-audit safety gate, not a material-quality specification.
constexpr double minimum_skin_triangle_area_ratio = 0.10;
constexpr double minimum_skin_volume_ratio = 0.80;
constexpr double maximum_skin_volume_ratio = 1.20;

static_assert(sizeof(float3) == 12U);
static_assert(sizeof(float4) == 16U);
static_assert(sizeof(waterlab::FoamParticle) == 48U,
    "capture parser must be updated when the foam record changes");

struct CaptureFileHeader {
    char magic[8]{};
    std::uint32_t version{};
    std::uint32_t frame_count{};
    std::uint32_t particle_count{};
    std::uint32_t skin_vertex_count{};
    std::uint32_t options_size{};
    std::uint32_t rectangle_size{};
    std::uint32_t statistics_size{};
    std::uint32_t timings_size{};
    std::uint32_t float3_size{};
};

struct SoftBodyCaptureHeader {
    std::uint32_t present{};
    std::uint32_t statistics_size{};
    std::uint32_t voxel_count{};
    std::uint32_t edge_count{};
    std::uint32_t render_triangle_count{};
};

struct CaptureFrame {
    waterlab::HybridOptions options{};
    waterlab::HybridStatistics statistics{};
    float3 rectangle_control_force{};
    float rectangle_control_torque{};
    bool has_soft_body{};
    waterlab::SoftBodyState soft_body;
    // Retained verbatim for the optional saved-start replay and skin-contact
    // audit.  The original audit intentionally skipped these large arrays.
    waterlab::HybridState state;
};

struct Capture {
    std::filesystem::path path;
    CaptureFileHeader header{};
    std::vector<CaptureFrame> frames;
};

struct CommandLine {
    std::filesystem::path capture_path;
    std::filesystem::path asset_path{MESHPREP_SOFT_BODY_ASSET_PATH};
    std::filesystem::path csv_path;
    bool baseline_only{};
    bool skin_contact{};
    bool render_contact{};
    bool resimulate_saved_start{};
    std::uint32_t stride{1U};
};

template <typename T>
void read_value(std::ifstream& input, T& value, std::string_view label)
{
    input.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!input) throw std::runtime_error("capture ended while reading " + std::string(label));
}

template <typename T>
void read_sized_value(
    std::ifstream& input, T& value, std::uint32_t byte_count, std::string_view label)
{
    if (byte_count == 0U || byte_count > sizeof(value)) {
        throw std::runtime_error("unsupported " + std::string(label) + " byte size");
    }
    value = {};
    input.read(reinterpret_cast<char*>(&value), byte_count);
    if (!input) throw std::runtime_error("capture ended while reading " + std::string(label));
}

template <typename T>
void read_values(std::ifstream& input, std::vector<T>& values,
    std::uint32_t count, std::string_view label)
{
    if (count > maximum_records) {
        throw std::runtime_error(std::string(label) + " count exceeds safety limit");
    }
    values.resize(count);
    input.read(reinterpret_cast<char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(T)));
    if (!input) throw std::runtime_error("capture ended while reading " + std::string(label));
}

void skip_bytes(std::ifstream& input, std::uint64_t bytes, std::string_view label)
{
    if (bytes > static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max())) {
        throw std::runtime_error(std::string(label) + " byte count exceeds stream limit");
    }
    input.seekg(static_cast<std::streamoff>(bytes), std::ios::cur);
    if (!input) throw std::runtime_error("capture ended while skipping " + std::string(label));
}

std::uint64_t checked_bytes(
    std::uint64_t count, std::uint64_t size, std::string_view label)
{
    if (size != 0U && count > std::numeric_limits<std::uint64_t>::max() / size) {
        throw std::runtime_error(std::string(label) + " size overflow");
    }
    return count * size;
}

std::filesystem::path resolve_capture_path(std::filesystem::path path)
{
    if (path.empty()) {
        std::ifstream latest{std::filesystem::path(latest_capture_file)};
        std::string latest_path;
        if (!latest || !std::getline(latest, latest_path) || latest_path.empty()) {
            throw std::runtime_error(
                "no capture specified and LAST_CAPTURE.txt is unavailable");
        }
        path = latest_path;
    }
    if (std::filesystem::is_directory(path)) path /= "capture.bin";
    return path;
}

Capture load_capture(std::filesystem::path requested_path)
{
    Capture capture;
    capture.path = resolve_capture_path(std::move(requested_path));
    std::ifstream input(capture.path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open capture " + capture.path.string());
    read_value(input, capture.header, "file header");
    if (!std::equal(capture_magic.begin(), capture_magic.end(), capture.header.magic) ||
        capture.header.version < minimum_capture_version ||
        capture.header.version > maximum_capture_version ||
        capture.header.frame_count == 0U ||
        capture.header.frame_count > maximum_capture_frames ||
        capture.header.particle_count == 0U ||
        capture.header.particle_count > maximum_records ||
        capture.header.skin_vertex_count == 0U ||
        capture.header.skin_vertex_count > maximum_records ||
        capture.header.float3_size != sizeof(float3)) {
        throw std::runtime_error("capture header is unsupported or invalid");
    }
    if (capture.header.rectangle_size == 0U ||
        capture.header.rectangle_size > sizeof(waterlab::RectangleState) ||
        capture.header.statistics_size == 0U ||
        capture.header.statistics_size > sizeof(waterlab::HybridStatistics) ||
        capture.header.timings_size == 0U ||
        capture.header.timings_size > sizeof(waterlab::HybridTimings)) {
        throw std::runtime_error("capture ABI is newer than this audit build");
    }

    capture.frames.reserve(capture.header.frame_count);
    for (std::uint32_t frame_index = 0U;
         frame_index < capture.header.frame_count; ++frame_index) {
        CaptureFrame frame;
        read_sized_value(input, frame.options, capture.header.options_size, "HybridOptions");
        waterlab::RectangleState rectangle;
        read_sized_value(input, rectangle, capture.header.rectangle_size, "RectangleState");
        read_sized_value(input, frame.statistics,
            capture.header.statistics_size, "HybridStatistics");
        waterlab::HybridTimings timings;
        read_sized_value(input, timings, capture.header.timings_size, "HybridTimings");
        float3 rectangle_target{};
        read_value(input, rectangle_target, "rectangle target");
        read_value(input, frame.rectangle_control_force, "rectangle control force");
        read_value(input, frame.rectangle_control_torque, "rectangle control torque");
        frame.state.options = frame.options;
        frame.state.rectangle = rectangle;
        frame.state.statistics = frame.statistics;

        const std::uint64_t particles = capture.header.particle_count;
        const std::uint64_t skin = capture.header.skin_vertex_count;
        // Keep this exact order synchronized with main.cpp::write_capture_frame.
        read_values(input, frame.state.particle_positions, particles, "particle positions");
        read_values(input, frame.state.particle_velocities, particles, "particle velocities");
        read_values(input, frame.state.particle_forces, particles, "particle forces");
        read_values(input, frame.state.particle_skin_owners, particles, "particle owners");
        read_values(input, frame.state.skin_positions, skin, "skin positions");
        read_values(input, frame.state.skin_velocities, skin, "skin velocities");
        read_values(input, frame.state.skin_forces, skin, "skin forces");
        read_values(input, frame.state.skin_box_forces, skin, "skin box forces");
        read_values(input, frame.state.skin_box_impulses, skin, "skin box impulses");
        read_values(input, frame.state.skin_box_pair_work, skin, "skin box work");
        read_values(input, frame.state.skin_particle_forces, skin, "skin particle forces");
        read_values(input, frame.state.skin_spring_forces, skin, "skin spring forces");

        float visual_ms{};
        read_value(input, visual_ms, "visual time");
        skip_bytes(input, checked_bytes(particles, sizeof(float4), "normal foam"),
            "normal foam array");
        std::uint64_t foam_tick{};
        read_value(input, foam_tick, "foam tick");
        skip_bytes(input, checked_bytes(waterlab::FluidVisuals::foam_capacity,
            sizeof(waterlab::FoamParticle), "foam particles"), "foam particles");

        SoftBodyCaptureHeader soft_header;
        read_value(input, soft_header, "soft-body header");
        std::uint32_t damage_count{};
        if (capture.header.version >= 5U) {
            read_value(input, damage_count, "soft-body damage count");
        }
        if (soft_header.present > 1U ||
            soft_header.statistics_size == 0U ||
            soft_header.statistics_size > sizeof(waterlab::SoftBodyStatistics) ||
            soft_header.voxel_count > maximum_records ||
            soft_header.edge_count > maximum_records ||
            soft_header.render_triangle_count > maximum_records ||
            damage_count > maximum_records ||
            (damage_count != 0U && damage_count != soft_header.edge_count)) {
            throw std::runtime_error("soft-body capture header is invalid");
        }
        frame.has_soft_body = soft_header.present != 0U;
        if (!frame.has_soft_body) {
            if (soft_header.voxel_count != 0U || soft_header.edge_count != 0U ||
                soft_header.render_triangle_count != 0U || damage_count != 0U) {
                throw std::runtime_error("absent soft-body frame has nonzero payload counts");
            }
            capture.frames.push_back(std::move(frame));
            continue;
        }

        read_value(input, frame.soft_body.strength_multiplier,
            "soft-body strength multiplier");
        read_sized_value(input, frame.soft_body.statistics,
            soft_header.statistics_size, "SoftBodyStatistics");
        read_values(input, frame.soft_body.positions,
            soft_header.voxel_count, "soft-body positions");
        read_values(input, frame.soft_body.velocities,
            soft_header.voxel_count, "soft-body velocities");
        read_values(input, frame.soft_body.active_edges,
            soft_header.edge_count, "soft-body active edges");
        if (capture.header.version >= 5U) {
            read_values(input, frame.soft_body.edge_damage,
                damage_count, "soft-body edge damage");
        }
        read_values(input, frame.soft_body.active_render_triangles,
            soft_header.render_triangle_count, "soft-body render activity");
        capture.frames.push_back(std::move(frame));
    }
    if (input.peek() != std::char_traits<char>::eof()) {
        throw std::runtime_error("capture has unexpected trailing data");
    }
    return capture;
}

bool finite(float3 value)
{
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

double length(float3 value)
{
    return std::hypot(std::hypot(
        static_cast<double>(value.x), static_cast<double>(value.y)),
        static_cast<double>(value.z));
}

float3 subtract(float3 first, float3 second)
{
    return make_float3(first.x - second.x, first.y - second.y, first.z - second.z);
}

float3 add(float3 first, float3 second)
{
    return make_float3(first.x + second.x, first.y + second.y, first.z + second.z);
}

float3 scale(float3 value, double factor)
{
    return make_float3(static_cast<float>(static_cast<double>(value.x) * factor),
        static_cast<float>(static_cast<double>(value.y) * factor),
        static_cast<float>(static_cast<double>(value.z) * factor));
}

float3 cross(float3 first, float3 second)
{
    return make_float3(
        first.y * second.z - first.z * second.y,
        first.z * second.x - first.x * second.z,
        first.x * second.y - first.y * second.x);
}

double dot(float3 first, float3 second)
{
    return static_cast<double>(first.x) * second.x +
        static_cast<double>(first.y) * second.y +
        static_cast<double>(first.z) * second.z;
}

double squared_distance(float3 first, float3 second)
{
    const float3 delta = subtract(first, second);
    return dot(delta, delta);
}

double signed_tetrahedron_six_volume(
    float3 a, float3 b, float3 c, float3 anchor)
{
    return dot(cross(subtract(b, a), subtract(c, a)), subtract(anchor, a));
}

double triangle_double_area(float3 a, float3 b, float3 c)
{
    return length(cross(subtract(b, a), subtract(c, a)));
}

struct SkinContactMetrics {
    std::uint32_t centers_inside{};
    std::uint32_t surface_centers_inside{};
    std::uint32_t spheres_inside{};
    std::uint32_t nonfinite{};
    double maximum_depth{};
    double total_depth{};
    double minimum_area_ratio{std::numeric_limits<double>::infinity()};
    double signed_volume{};
    bool winding_agrees{true};
};

// Watertight, oriented physical-skin triangles are regenerated from the recorded
// options.  Parity is cheap enough after the skin AABB rejects the other seven
// course posts; the solid-angle check is deliberately sampled as an independent
// validation, not used as a sphere approximation.
bool ray_inside_x(float3 point, const std::vector<float3>& positions,
    const std::vector<uint3>& triangles)
{
    std::uint32_t crossings{};
    for (const uint3 t : triangles) {
        const float3 a = positions[t.x], b = positions[t.y], c = positions[t.z];
        const double ay = a.y - point.y, az = a.z - point.z;
        const double by = b.y - point.y, bz = b.z - point.z;
        const double cy = c.y - point.y, cz = c.z - point.z;
        // Moller-Trumbore specialized to +X; the tiny fixed y/z offset makes
        // a ray through a shared edge deterministic instead of double-counted.
        const double py = 0.6180339887498948e-7;
        const double pz = 0.4142135623730950e-7;
        const double e1y = by - ay, e1z = bz - az;
        const double e2y = cy - ay, e2z = cz - az;
        const double determinant = e1y * e2z - e1z * e2y;
        if (std::abs(determinant) < 1.0e-14) continue;
        const double qy = -ay + py, qz = -az + pz;
        const double u = (qy * e2z - qz * e2y) / determinant;
        const double v = (e1y * qz - e1z * qy) / determinant;
        if (u < 0.0 || v < 0.0 || u + v > 1.0) continue;
        const double x = a.x + u * (b.x - a.x) + v * (c.x - a.x);
        if (x > point.x) ++crossings;
    }
    return (crossings & 1U) != 0U;
}

double point_triangle_distance(float3 p, float3 a, float3 b, float3 c)
{
    // Ericson's closest-point region tests: float vector storage, double dot products.
    const auto sub = [](float3 x, float3 y) { return make_float3(x.x-y.x,x.y-y.y,x.z-y.z); };
    const auto dotd = [](float3 x, float3 y) { return static_cast<double>(x.x)*y.x + static_cast<double>(x.y)*y.y + static_cast<double>(x.z)*y.z; };
    const float3 ab=sub(b,a), ac=sub(c,a), ap=sub(p,a);
    const double d1=dotd(ab,ap), d2=dotd(ac,ap);
    if (d1<=0.0 && d2<=0.0) return length(ap);
    const float3 bp=sub(p,b); const double d3=dotd(ab,bp), d4=dotd(ac,bp);
    if (d3>=0.0 && d4<=d3) return length(bp);
    const double vc=d1*d4-d3*d2;
    if (vc<=0.0 && d1>=0.0 && d3<=0.0) return length(sub(p, add(a, scale(ab,d1/(d1-d3)))));
    const float3 cp=sub(p,c); const double d5=dotd(ab,cp), d6=dotd(ac,cp);
    if (d6>=0.0 && d5<=d6) return length(cp);
    const double vb=d5*d2-d1*d6;
    if (vb<=0.0 && d2>=0.0 && d6<=0.0) return length(sub(p, add(a, scale(ac,d2/(d2-d6)))));
    const double va=d3*d6-d5*d4;
    if (va<=0.0 && (d4-d3)>=0.0 && (d5-d6)>=0.0) {
        const float3 bc=sub(c,b); return length(sub(p, add(b, scale(bc,(d4-d3)/((d4-d3)+(d5-d6))))));
    }
    const double denom=1.0/(va+vb+vc);
    return length(sub(p, add(a, add(scale(ab,vb*denom),scale(ac,vc*denom)))));
}

double winding_number(float3 p, const std::vector<float3>& positions,
    const std::vector<uint3>& triangles)
{
    double sum{};
    for (const uint3 t : triangles) {
        const float3 a=subtract(positions[t.x],p), b=subtract(positions[t.y],p), c=subtract(positions[t.z],p);
        const double la=length(a), lb=length(b), lc=length(c);
        sum += 2.0 * std::atan2(dot(a,cross(b,c)), la*lb*lc + dot(a,b)*lc + dot(b,c)*la + dot(c,a)*lb);
    }
    return sum / (4.0 * std::acos(-1.0));
}

[[nodiscard]] bool skin_quality_pass(double minimum_area_ratio,
    double first_signed_volume, double minimum_signed_volume,
    double maximum_signed_volume, bool skin_geometry_valid) noexcept
{
    return skin_geometry_valid && std::isfinite(minimum_area_ratio) &&
        minimum_area_ratio >= minimum_skin_triangle_area_ratio &&
        std::isfinite(first_signed_volume) && first_signed_volume > 0.0 &&
        std::isfinite(minimum_signed_volume) && std::isfinite(maximum_signed_volume) &&
        minimum_signed_volume / first_signed_volume >= minimum_skin_volume_ratio &&
        maximum_signed_volume / first_signed_volume <= maximum_skin_volume_ratio;
}

std::vector<float3> active_render_points(const std::vector<float3>& positions,
    const std::vector<uint3>& triangles, const std::vector<std::uint8_t>& activity)
{
    if (triangles.size()!=activity.size())
        throw std::runtime_error("render contact activity count mismatch");
    std::vector<std::uint8_t> used(positions.size());
    for (std::size_t i=0;i<triangles.size();++i) if (activity[i]) {
        const auto t=triangles[i];
        if (t.x>=used.size() || t.y>=used.size() || t.z>=used.size())
            throw std::runtime_error("render contact triangle index out of bounds");
        used[t.x]=used[t.y]=used[t.z]=1U;
    }
    std::vector<float3> points;
    for (std::size_t i=0;i<used.size();++i) if (used[i]) points.push_back(positions[i]);
    return points;
}

void skin_contact_self_test()
{
    const std::vector<float3> vertices{make_float3(0,0,0), make_float3(1,0,0),
        make_float3(0,1,0), make_float3(0,0,1)};
    const std::vector<uint3> triangles{make_uint3(0,2,1), make_uint3(0,1,3),
        make_uint3(0,3,2), make_uint3(1,2,3)};
    const float3 inside=make_float3(.25F,.25F,.25F);
    const float3 outside=make_float3(1.1F,.1F,.1F);
    const auto selected=active_render_points(vertices,triangles,{1,0,0,0});
    if (selected.size()!=3U || selected[2].y!=1.F ||
        !active_render_points(vertices,triangles,{0,0,0,0}).empty() ||
        active_render_points(vertices,triangles,{1,1,1,1}).size()!=4U)
        throw std::runtime_error("active render-vertex selection self-test failed");
    if (!ray_inside_x(inside,vertices,triangles) ||
        !(std::abs(winding_number(inside,vertices,triangles)) > .5) ||
        ray_inside_x(outside,vertices,triangles) ||
        !(std::abs(winding_number(outside,vertices,triangles)) < 1.0e-6) ||
        std::abs(point_triangle_distance(inside,vertices[1],vertices[2],vertices[3]) -
            0.1443375672974064) > 1.0e-6) {
        throw std::runtime_error("skin-contact tetrahedron self-test failed");
    }
    if (!skin_quality_pass(0.5, 1.0, 1.0, 1.0, true) ||
        skin_quality_pass(0.005, 1.0, 1.0, 1.0, true) ||
        skin_quality_pass(0.5, 1.0, 0.65, 1.0, true) ||
        skin_quality_pass(0.5, 1.0, 1.0, 1.21, true) ||
        skin_quality_pass(std::numeric_limits<double>::quiet_NaN(), 1.0, 1.0, 1.0, true)) {
        throw std::runtime_error("skin-contact quality self-test failed");
    }
}

struct DriftMetrics {
    double maximum_position{};
    double maximum_velocity{};
    double position_square_sum{};
    double velocity_square_sum{};
    std::uint64_t position_count{};
    std::uint64_t velocity_count{};
    bool positions_identical{true};
    bool velocities_identical{true};
    void add(const std::vector<float3>& actual, const std::vector<float3>& recorded) {
        if (actual.size()!=recorded.size()) throw std::runtime_error("replay state size mismatch");
        if (!actual.empty()) positions_identical &= std::memcmp(actual.data(),
            recorded.data(), actual.size()*sizeof(float3)) == 0;
        for (std::size_t i=0;i<actual.size();++i) {
            const double d=length(subtract(actual[i],recorded[i]));
            maximum_position=std::max(maximum_position,d); position_square_sum+=d*d; ++position_count;
        }
    }
    void add_velocity(const std::vector<float3>& actual, const std::vector<float3>& recorded) {
        if (actual.size()!=recorded.size()) throw std::runtime_error("replay velocity size mismatch");
        if (!actual.empty()) velocities_identical &= std::memcmp(actual.data(),
            recorded.data(), actual.size()*sizeof(float3)) == 0;
        for (std::size_t i=0;i<actual.size();++i) {
            const double d=length(subtract(actual[i],recorded[i]));
            maximum_velocity=std::max(maximum_velocity,d); velocity_square_sum+=d*d; ++velocity_count;
        }
    }
    [[nodiscard]] double rms_position() const { return position_count ? std::sqrt(position_square_sum/position_count) : 0.0; }
    [[nodiscard]] double rms_velocity() const { return velocity_count ? std::sqrt(velocity_square_sum/velocity_count) : 0.0; }
};

void add_drift(DriftMetrics& drift, const waterlab::HybridState& actual,
    const waterlab::HybridState& recorded, const waterlab::SoftBodyState& actual_soft,
    const waterlab::SoftBodyState& recorded_soft)
{
    drift.add(actual.particle_positions,recorded.particle_positions);
    drift.add(actual.skin_positions,recorded.skin_positions);
    drift.add(actual_soft.positions,recorded_soft.positions);
    drift.add_velocity(actual.particle_velocities,recorded.particle_velocities);
    drift.add_velocity(actual.skin_velocities,recorded.skin_velocities);
    drift.add_velocity(actual_soft.velocities,recorded_soft.velocities);
}

SkinContactMetrics measure_skin_points(const waterlab::HybridOptions& options,
    const std::vector<float3>& skin_positions, const std::vector<float3>& points,
    bool validate_winding, const waterlab::SoftBodyAsset* asset = nullptr)
{
    const auto mesh = waterlab::make_geodesic_sphere(
        options.physical_skin_frequency, options.skin_radius);
    if (mesh.positions.size() != skin_positions.size())
        throw std::runtime_error("recorded skin vertices do not match recorded skin options");
    SkinContactMetrics out;
    float3 lo=skin_positions.front(), hi=lo;
    for (const float3 p : skin_positions) {
        if (!finite(p)) { ++out.nonfinite; continue; }
        lo.x=std::min(lo.x,p.x); lo.y=std::min(lo.y,p.y); lo.z=std::min(lo.z,p.z);
        hi.x=std::max(hi.x,p.x); hi.y=std::max(hi.y,p.y); hi.z=std::max(hi.z,p.z);
    }
    for (const uint3 t : mesh.triangles) {
        const float3 a=skin_positions[t.x], b=skin_positions[t.y], c=skin_positions[t.z];
        if (!finite(a)||!finite(b)||!finite(c)) { ++out.nonfinite; continue; }
        const double rest=triangle_double_area(mesh.positions[t.x],mesh.positions[t.y],mesh.positions[t.z]);
        out.minimum_area_ratio=std::min(out.minimum_area_ratio,triangle_double_area(a,b,c)/rest);
        out.signed_volume += dot(a,cross(b,c))/6.0;
    }
    bool checked_winding{};
    for (std::size_t i=0; i<points.size(); ++i) {
        const float3 p=points[i];
        if (!finite(p)) { ++out.nonfinite; continue; }
        if (p.x<lo.x || p.x>hi.x || p.y<lo.y || p.y>hi.y || p.z<lo.z || p.z>hi.z) continue;
        const bool inside=ray_inside_x(p, skin_positions, mesh.triangles);
        if (validate_winding && !checked_winding) {
            const bool winding_inside=std::abs(winding_number(p,skin_positions,mesh.triangles))>0.5;
            out.winding_agrees = inside == winding_inside; checked_winding=true;
        }
        if (!inside) continue;
        ++out.centers_inside;
        if (asset != nullptr && (asset->voxel_flags[i % asset->rest_voxels.size()] & waterlab::soft_body_voxel_surface) != 0U) ++out.surface_centers_inside;
        double clearance=std::numeric_limits<double>::infinity();
        for (const uint3 t : mesh.triangles) clearance=std::min(clearance,point_triangle_distance(p,skin_positions[t.x],skin_positions[t.y],skin_positions[t.z]));
        out.maximum_depth=std::max(out.maximum_depth,clearance); out.total_depth+=clearance;
        if (asset != nullptr && clearance >= asset->voxel_radius) ++out.spheres_inside;
    }
    return out;
}

SkinContactMetrics measure_skin_contact(const CaptureFrame& frame,
    const waterlab::SoftBodyAsset& asset, bool validate_winding)
{
    return measure_skin_points(frame.options, frame.state.skin_positions,
        frame.soft_body.positions, validate_winding, &asset);
}

struct RenderTemplate {
    const waterlab::SoftBodyAsset* asset{};
    std::uint32_t instance_count{};
    std::vector<float3> rest_positions;
    std::vector<uint3> triangles;
    // Each surface triangle is paired with a nearby interior voxel. Comparing
    // the signed tetrahedron volume at rest and after deformation detects a
    // material-relative flip while remaining invariant under rigid motion.
    std::vector<std::uint32_t> orientation_anchor_voxels;
    std::vector<double> rest_orientation_volumes;
};

struct OrientationAnchor {
    std::uint32_t voxel{};
    double rest_volume{};
};

OrientationAnchor choose_orientation_anchor(
    const waterlab::SoftBodyAsset& asset, uint3 triangle)
{
    const float3 a = asset.render_positions[triangle.x];
    const float3 b = asset.render_positions[triangle.y];
    const float3 c = asset.render_positions[triangle.z];
    const float3 normal = cross(subtract(b, a), subtract(c, a));
    const double normal_length = length(normal);
    if (!(normal_length > 1.0e-12)) {
        throw std::runtime_error("cannot orient a degenerate rest triangle");
    }

    float3 voxel_center{};
    for (const float3 voxel : asset.rest_voxels) voxel_center = add(voxel_center, voxel);
    voxel_center = scale(voxel_center, 1.0 / static_cast<double>(asset.rest_voxels.size()));
    const float3 triangle_center = scale(add(add(a, b), c), 1.0 / 3.0);
    double interior_sign = dot(normal, subtract(voxel_center, triangle_center));
    interior_sign = interior_sign < 0.0 ? -1.0 : 1.0;
    const double anchor_depth = std::max(
        static_cast<double>(asset.nominal_spacing),
        2.0 * static_cast<double>(asset.voxel_radius));
    const float3 target = add(triangle_center,
        scale(normal, interior_sign * anchor_depth / normal_length));
    const double minimum_plane_distance = std::max(
        1.0e-6, 0.05 * static_cast<double>(asset.nominal_spacing));

    std::optional<std::uint32_t> best;
    double best_distance = std::numeric_limits<double>::infinity();
    for (std::uint32_t voxel = 0U; voxel < asset.rest_voxels.size(); ++voxel) {
        const float3 position = asset.rest_voxels[voxel];
        const double plane_distance =
            dot(normal, subtract(position, triangle_center)) / normal_length;
        if (plane_distance * interior_sign < minimum_plane_distance) continue;
        const double candidate_distance = squared_distance(position, target);
        if (candidate_distance < best_distance) {
            best = voxel;
            best_distance = candidate_distance;
        }
    }

    // A thin/sheet asset may have no voxel clearly on the centroid-facing side.
    // In that case choose the closest sufficiently non-coplanar material point;
    // the sign comparison remains valid even though "inside" is undefined.
    if (!best.has_value()) {
        for (std::uint32_t voxel = 0U; voxel < asset.rest_voxels.size(); ++voxel) {
            const float3 position = asset.rest_voxels[voxel];
            const double plane_distance = std::abs(
                dot(normal, subtract(position, triangle_center)) / normal_length);
            if (plane_distance < minimum_plane_distance) continue;
            const double candidate_distance = squared_distance(position, triangle_center);
            if (candidate_distance < best_distance) {
                best = voxel;
                best_distance = candidate_distance;
            }
        }
    }
    if (!best.has_value()) {
        throw std::runtime_error(
            "cannot find a non-coplanar deformation-following orientation anchor");
    }
    const double rest_volume = signed_tetrahedron_six_volume(
        a, b, c, asset.rest_voxels[*best]);
    if (!std::isfinite(rest_volume) || std::abs(rest_volume) <= 1.0e-12) {
        throw std::runtime_error("soft-body orientation anchor is numerically degenerate");
    }
    return {*best, rest_volume};
}

RenderTemplate make_render_template(
    const waterlab::SoftBodyAsset& asset, std::uint32_t instance_count)
{
    if (instance_count == 0U ||
        instance_count > waterlab::SoftBodyOptions::maximum_instances) {
        throw std::runtime_error("captured soft-body instance count is unsupported");
    }
    const float local_min_y = std::min_element(asset.render_positions.begin(),
        asset.render_positions.end(), [](float3 first, float3 second) {
            return first.y < second.y;
        })->y;
    std::vector<OrientationAnchor> local_orientation;
    local_orientation.reserve(asset.render_triangles.size());
    for (const uint3 triangle : asset.render_triangles) {
        local_orientation.push_back(choose_orientation_anchor(asset, triangle));
    }
    RenderTemplate result{&asset, instance_count, {}, {}, {}, {}};
    result.rest_positions.reserve(asset.render_positions.size() * instance_count);
    result.triangles.reserve(asset.render_triangles.size() * instance_count);
    result.orientation_anchor_voxels.reserve(
        asset.render_triangles.size() * instance_count);
    result.rest_orientation_volumes.reserve(
        asset.render_triangles.size() * instance_count);
    for (std::uint32_t instance = 0U; instance < instance_count; ++instance) {
        float3 origin = waterlab::course_peg(instance);
        origin.y -= local_min_y;
        for (const float3 local : asset.render_positions) {
            result.rest_positions.push_back(make_float3(
                local.x + origin.x, local.y + origin.y, local.z + origin.z));
        }
        const std::uint32_t vertex_base = instance *
            static_cast<std::uint32_t>(asset.render_positions.size());
        for (std::size_t triangle_index = 0U;
             triangle_index < asset.render_triangles.size(); ++triangle_index) {
            const uint3 triangle = asset.render_triangles[triangle_index];
            result.triangles.push_back(make_uint3(
                triangle.x + vertex_base, triangle.y + vertex_base,
                triangle.z + vertex_base));
            result.orientation_anchor_voxels.push_back(
                local_orientation[triangle_index].voxel +
                instance * static_cast<std::uint32_t>(asset.rest_voxels.size()));
            result.rest_orientation_volumes.push_back(
                local_orientation[triangle_index].rest_volume);
        }
    }
    return result;
}

std::vector<float3> reconstruct_render_positions(
    const RenderTemplate& geometry, const waterlab::SoftBodyState& state)
{
    const auto& asset = *geometry.asset;
    const std::size_t voxels_per_instance = asset.rest_voxels.size();
    const std::size_t vertices_per_instance = asset.render_positions.size();
    if (state.positions.size() != voxels_per_instance * geometry.instance_count ||
        geometry.rest_positions.size() != vertices_per_instance * geometry.instance_count) {
        throw std::runtime_error("soft-body state does not match render asset");
    }
    std::vector<float3> result(geometry.rest_positions.size());
    for (std::uint32_t instance = 0U; instance < geometry.instance_count; ++instance) {
        const std::size_t voxel_base = instance * voxels_per_instance;
        const std::size_t vertex_base = instance * vertices_per_instance;
        for (std::size_t vertex = 0U; vertex < vertices_per_instance; ++vertex) {
            const waterlab::SoftBodyBinding binding = asset.render_bindings[vertex];
            const std::uint32_t ids[4]{
                binding.voxels.x, binding.voxels.y, binding.voxels.z, binding.voxels.w};
            const float weights[4]{
                binding.weights.x, binding.weights.y,
                binding.weights.z, binding.weights.w};
            float3 current{};
            float3 rest{};
            for (unsigned slot = 0U; slot < 4U; ++slot) {
                const float weight = weights[slot];
                const float3 current_voxel = state.positions[voxel_base + ids[slot]];
                const float3 rest_voxel = asset.rest_voxels[ids[slot]];
                current.x += weight * current_voxel.x;
                current.y += weight * current_voxel.y;
                current.z += weight * current_voxel.z;
                rest.x += weight * rest_voxel.x;
                rest.y += weight * rest_voxel.y;
                rest.z += weight * rest_voxel.z;
            }
            const float3 local = asset.render_positions[vertex];
            result[vertex_base + vertex] = make_float3(
                current.x + local.x - rest.x,
                current.y + local.y - rest.y,
                current.z + local.z - rest.z);
        }
    }
    return result;
}

double percentile(std::vector<double>& values, double fraction)
{
    if (values.empty()) return 0.0;
    const std::size_t index = static_cast<std::size_t>(
        fraction * static_cast<double>(values.size() - 1U));
    std::nth_element(values.begin(), values.begin() +
        static_cast<std::ptrdiff_t>(index), values.end());
    return values[index];
}

struct FrameMetrics {
    std::uint64_t physics_frame{};
    std::uint32_t broken_edges{};
    std::uint32_t active_triangles{};
    std::uint32_t measured_triangles{};
    std::uint32_t abnormal_triangles{};
    std::uint32_t measured_active_triangles{};
    std::uint32_t abnormal_active_triangles{};
    std::uint32_t inversion_count{};
    std::uint32_t active_inversion_count{};
    std::vector<std::uint32_t> inverted_triangle_ids;
    std::vector<std::uint32_t> active_inverted_triangle_ids;
    std::uint32_t voxel_position_finite_failures{};
    std::uint32_t voxel_velocity_finite_failures{};
    std::uint32_t geometry_finite_failures{};
    std::uint32_t reported_fluid_finite_failures{};
    std::uint32_t reported_soft_finite_failures{};
    double maximum_penetration{};
    double maximum_triangle_area_ratio{};
    std::uint32_t maximum_triangle_area_id{};
    double p99_triangle_area_ratio{};
    double maximum_render_edge_ratio{};
    std::uint32_t maximum_render_edge_triangle_id{};
    double minimum_active_triangle_area_ratio{
        std::numeric_limits<double>::infinity()};
    double maximum_active_triangle_area_ratio{};
    double p99_active_triangle_area_ratio{};
    double maximum_active_render_edge_ratio{};
    double maximum_voxel_speed{};
    double p99_voxel_speed{};
};

FrameMetrics measure_frame(const RenderTemplate& geometry,
    const waterlab::SoftBodyState& soft_body,
    const waterlab::HybridStatistics& hybrid_statistics,
    std::vector<double>& all_area_ratios,
    std::vector<double>& all_active_area_ratios,
    std::vector<double>& all_speeds,
    const std::vector<float3>* runtime_render_positions = nullptr)
{
    if (soft_body.positions.size() != soft_body.velocities.size()) {
        throw std::runtime_error("soft-body position/velocity counts do not match");
    }
    if (soft_body.active_render_triangles.size() != geometry.triangles.size() ||
        geometry.orientation_anchor_voxels.size() != geometry.triangles.size() ||
        geometry.rest_orientation_volumes.size() != geometry.triangles.size()) {
        throw std::runtime_error("captured render activity does not match the asset");
    }
    FrameMetrics metrics;
    metrics.physics_frame = hybrid_statistics.frame_index;
    metrics.broken_edges = static_cast<std::uint32_t>(std::count(
        soft_body.active_edges.begin(), soft_body.active_edges.end(), std::uint8_t{0U}));
    metrics.active_triangles = static_cast<std::uint32_t>(std::count(
        soft_body.active_render_triangles.begin(),
        soft_body.active_render_triangles.end(), std::uint8_t{1U}));
    metrics.reported_fluid_finite_failures = hybrid_statistics.finite_failures;
    metrics.reported_soft_finite_failures = soft_body.statistics.finite_failure_count;
    metrics.maximum_penetration = hybrid_statistics.maximum_soft_body_penetration;

    std::vector<double> frame_speeds;
    frame_speeds.reserve(soft_body.velocities.size());
    for (std::size_t voxel = 0U; voxel < soft_body.positions.size(); ++voxel) {
        if (!finite(soft_body.positions[voxel])) {
            ++metrics.voxel_position_finite_failures;
        }
        const float3 velocity = soft_body.velocities[voxel];
        if (!finite(velocity)) {
            ++metrics.voxel_velocity_finite_failures;
            continue;
        }
        const double speed = length(velocity);
        metrics.maximum_voxel_speed = std::max(metrics.maximum_voxel_speed, speed);
        frame_speeds.push_back(speed);
        all_speeds.push_back(speed);
    }

    const std::vector<float3> reconstructed = runtime_render_positions == nullptr
        ? reconstruct_render_positions(geometry, soft_body) : std::vector<float3>{};
    const std::vector<float3>& current = runtime_render_positions == nullptr
        ? reconstructed : *runtime_render_positions;
    if (current.size() != geometry.rest_positions.size()) {
        throw std::runtime_error("runtime render position count does not match the asset");
    }
    std::vector<double> frame_area_ratios;
    std::vector<double> frame_active_area_ratios;
    frame_area_ratios.reserve(geometry.triangles.size());
    frame_active_area_ratios.reserve(metrics.active_triangles);
    for (std::size_t triangle_index = 0U;
         triangle_index < geometry.triangles.size(); ++triangle_index) {
        const uint3 triangle = geometry.triangles[triangle_index];
        const float3 current_vertices[3]{
            current[triangle.x], current[triangle.y], current[triangle.z]};
        const float3 rest_vertices[3]{geometry.rest_positions[triangle.x],
            geometry.rest_positions[triangle.y], geometry.rest_positions[triangle.z]};
        if (!finite(current_vertices[0]) || !finite(current_vertices[1]) ||
            !finite(current_vertices[2])) {
            ++metrics.geometry_finite_failures;
            continue;
        }
        const double rest_area = triangle_double_area(
            rest_vertices[0], rest_vertices[1], rest_vertices[2]);
        const double current_area = triangle_double_area(
            current_vertices[0], current_vertices[1], current_vertices[2]);
        if (!(rest_area > 1.0e-12) || !std::isfinite(current_area)) {
            ++metrics.geometry_finite_failures;
            continue;
        }
        const double area_ratio = current_area / rest_area;
        ++metrics.measured_triangles;
        metrics.abnormal_triangles += area_ratio < 0.5 || area_ratio > 2.0;
        if (area_ratio > metrics.maximum_triangle_area_ratio) {
            metrics.maximum_triangle_area_ratio = area_ratio;
            metrics.maximum_triangle_area_id =
                static_cast<std::uint32_t>(triangle_index);
        }
        frame_area_ratios.push_back(area_ratio);
        all_area_ratios.push_back(area_ratio);
        const bool active = soft_body.active_render_triangles[triangle_index] != 0U;
        const std::uint32_t anchor_voxel =
            geometry.orientation_anchor_voxels[triangle_index];
        if (anchor_voxel >= soft_body.positions.size()) {
            throw std::runtime_error("orientation anchor exceeds soft-body voxel state");
        }
        const double current_orientation = signed_tetrahedron_six_volume(
            current_vertices[0], current_vertices[1], current_vertices[2],
            soft_body.positions[anchor_voxel]);
        const double rest_orientation =
            geometry.rest_orientation_volumes[triangle_index];
        if (!std::isfinite(current_orientation)) {
            ++metrics.geometry_finite_failures;
        } else if (current_orientation * rest_orientation <= 0.0) {
            ++metrics.inversion_count;
            metrics.inverted_triangle_ids.push_back(
                static_cast<std::uint32_t>(triangle_index));
            if (active) {
                ++metrics.active_inversion_count;
                metrics.active_inverted_triangle_ids.push_back(
                    static_cast<std::uint32_t>(triangle_index));
            }
        }
        if (active) {
            ++metrics.measured_active_triangles;
            metrics.abnormal_active_triangles +=
                area_ratio < minimum_active_area_ratio ||
                area_ratio > maximum_active_area_ratio;
            metrics.minimum_active_triangle_area_ratio = std::min(
                metrics.minimum_active_triangle_area_ratio, area_ratio);
            metrics.maximum_active_triangle_area_ratio = std::max(
                metrics.maximum_active_triangle_area_ratio, area_ratio);
            frame_active_area_ratios.push_back(area_ratio);
            all_active_area_ratios.push_back(area_ratio);
        }
        for (unsigned edge = 0U; edge < 3U; ++edge) {
            const unsigned next = (edge + 1U) % 3U;
            const double rest_length = length(subtract(
                rest_vertices[next], rest_vertices[edge]));
            const double current_length = length(subtract(
                current_vertices[next], current_vertices[edge]));
            if (rest_length > 1.0e-12 && std::isfinite(current_length)) {
                const double edge_ratio = current_length / rest_length;
                if (edge_ratio > metrics.maximum_render_edge_ratio) {
                    metrics.maximum_render_edge_ratio = edge_ratio;
                    metrics.maximum_render_edge_triangle_id =
                        static_cast<std::uint32_t>(triangle_index);
                }
                if (active) {
                    metrics.maximum_active_render_edge_ratio = std::max(
                        metrics.maximum_active_render_edge_ratio, edge_ratio);
                }
            } else {
                ++metrics.geometry_finite_failures;
            }
        }
    }
    metrics.p99_triangle_area_ratio = percentile(frame_area_ratios, 0.99);
    metrics.p99_active_triangle_area_ratio = percentile(
        frame_active_area_ratios, 0.99);
    metrics.p99_voxel_speed = percentile(frame_speeds, 0.99);
    return metrics;
}

struct Summary {
    std::string mode;
    std::uint64_t first_physics_frame{};
    std::uint64_t last_physics_frame{};
    std::uint32_t starting_broken_edges{};
    std::uint32_t ending_broken_edges{};
    std::uint64_t observed_new_breaks{};
    std::uint32_t maximum_breaks_in_one_frame{};
    std::uint32_t minimum_active_triangles{std::numeric_limits<std::uint32_t>::max()};
    std::uint32_t triangle_count{};
    std::uint64_t measured_triangles{};
    std::uint64_t abnormal_triangles{};
    double maximum_frame_abnormal_fraction{};
    std::uint64_t measured_active_triangles{};
    std::uint64_t abnormal_active_triangles{};
    double maximum_frame_active_abnormal_fraction{};
    std::uint64_t inversion_count{};
    std::uint64_t active_inversion_count{};
    std::uint32_t maximum_frame_inversion_count{};
    std::uint32_t unique_inverted_triangles{};
    std::uint32_t unique_active_inverted_triangles{};
    std::vector<std::uint32_t> unique_active_inversions_per_instance;
    std::uint64_t voxel_position_finite_failures{};
    std::uint64_t voxel_velocity_finite_failures{};
    std::uint64_t geometry_finite_failures{};
    std::uint64_t reported_fluid_finite_failures{};
    std::uint32_t reported_soft_finite_failures{};
    double maximum_penetration{};
    double maximum_triangle_area_ratio{};
    std::uint64_t maximum_triangle_area_frame{};
    std::uint32_t maximum_triangle_area_id{};
    double p99_triangle_area_ratio{};
    double maximum_render_edge_ratio{};
    std::uint64_t maximum_render_edge_frame{};
    std::uint32_t maximum_render_edge_triangle_id{};
    double minimum_active_triangle_area_ratio{
        std::numeric_limits<double>::infinity()};
    double maximum_active_triangle_area_ratio{};
    double p99_active_triangle_area_ratio{};
    double maximum_active_render_edge_ratio{};
    double maximum_voxel_speed{};
    double p99_voxel_speed{};
    double wall_ms{};
    std::size_t frame_count{};
};

class SummaryBuilder {
public:
    SummaryBuilder(std::string mode, std::uint32_t triangle_count,
        std::uint32_t instance_count,
        std::optional<std::uint32_t> previous_broken = std::nullopt)
        : instance_count_(instance_count),
          inverted_triangles_(triangle_count),
          active_inverted_triangles_(triangle_count),
          previous_broken_(previous_broken)
    {
        summary_.mode = std::move(mode);
        summary_.triangle_count = triangle_count;
        if (instance_count_ == 0U || triangle_count % instance_count_ != 0U) {
            throw std::invalid_argument(
                "triangle count must divide evenly across soft-body instances");
        }
    }

    void add(const FrameMetrics& frame, double wall_ms)
    {
        if (summary_.frame_count == 0U) {
            summary_.first_physics_frame = frame.physics_frame;
            summary_.starting_broken_edges = frame.broken_edges;
        }
        summary_.last_physics_frame = frame.physics_frame;
        summary_.ending_broken_edges = frame.broken_edges;
        if (previous_broken_.has_value()) {
            const std::uint32_t added = frame.broken_edges >= *previous_broken_
                ? frame.broken_edges - *previous_broken_ : 0U;
            summary_.observed_new_breaks += added;
            summary_.maximum_breaks_in_one_frame = std::max(
                summary_.maximum_breaks_in_one_frame, added);
        }
        previous_broken_ = frame.broken_edges;
        summary_.minimum_active_triangles = std::min(
            summary_.minimum_active_triangles, frame.active_triangles);
        summary_.geometry_finite_failures += frame.geometry_finite_failures;
        summary_.reported_fluid_finite_failures += frame.reported_fluid_finite_failures;
        summary_.reported_soft_finite_failures = std::max(
            summary_.reported_soft_finite_failures,
            frame.reported_soft_finite_failures);
        summary_.maximum_penetration = std::max(
            summary_.maximum_penetration, frame.maximum_penetration);
        summary_.measured_triangles += frame.measured_triangles;
        summary_.abnormal_triangles += frame.abnormal_triangles;
        summary_.measured_active_triangles += frame.measured_active_triangles;
        summary_.abnormal_active_triangles += frame.abnormal_active_triangles;
        summary_.inversion_count += frame.inversion_count;
        summary_.active_inversion_count += frame.active_inversion_count;
        summary_.maximum_frame_inversion_count = std::max(
            summary_.maximum_frame_inversion_count, frame.inversion_count);
        for (const std::uint32_t triangle : frame.inverted_triangle_ids) {
            inverted_triangles_[triangle] = 1U;
        }
        for (const std::uint32_t triangle : frame.active_inverted_triangle_ids) {
            active_inverted_triangles_[triangle] = 1U;
        }
        summary_.voxel_position_finite_failures +=
            frame.voxel_position_finite_failures;
        summary_.voxel_velocity_finite_failures +=
            frame.voxel_velocity_finite_failures;
        if (frame.measured_triangles != 0U) {
            summary_.maximum_frame_abnormal_fraction = std::max(
                summary_.maximum_frame_abnormal_fraction,
                static_cast<double>(frame.abnormal_triangles) /
                    static_cast<double>(frame.measured_triangles));
        }
        if (frame.measured_active_triangles != 0U) {
            summary_.maximum_frame_active_abnormal_fraction = std::max(
                summary_.maximum_frame_active_abnormal_fraction,
                static_cast<double>(frame.abnormal_active_triangles) /
                    static_cast<double>(frame.measured_active_triangles));
            summary_.minimum_active_triangle_area_ratio = std::min(
                summary_.minimum_active_triangle_area_ratio,
                frame.minimum_active_triangle_area_ratio);
        }
        summary_.maximum_active_triangle_area_ratio = std::max(
            summary_.maximum_active_triangle_area_ratio,
            frame.maximum_active_triangle_area_ratio);
        summary_.maximum_active_render_edge_ratio = std::max(
            summary_.maximum_active_render_edge_ratio,
            frame.maximum_active_render_edge_ratio);
        if (frame.maximum_triangle_area_ratio > summary_.maximum_triangle_area_ratio) {
            summary_.maximum_triangle_area_ratio = frame.maximum_triangle_area_ratio;
            summary_.maximum_triangle_area_frame = frame.physics_frame;
            summary_.maximum_triangle_area_id = frame.maximum_triangle_area_id;
        }
        if (frame.maximum_render_edge_ratio > summary_.maximum_render_edge_ratio) {
            summary_.maximum_render_edge_ratio = frame.maximum_render_edge_ratio;
            summary_.maximum_render_edge_frame = frame.physics_frame;
            summary_.maximum_render_edge_triangle_id =
                frame.maximum_render_edge_triangle_id;
        }
        summary_.maximum_voxel_speed = std::max(
            summary_.maximum_voxel_speed, frame.maximum_voxel_speed);
        summary_.wall_ms += wall_ms;
        ++summary_.frame_count;
    }

    Summary finish(std::vector<double>& area_ratios,
        std::vector<double>& active_area_ratios, std::vector<double>& speeds)
    {
        summary_.p99_triangle_area_ratio = percentile(area_ratios, 0.99);
        summary_.p99_active_triangle_area_ratio = percentile(
            active_area_ratios, 0.99);
        summary_.p99_voxel_speed = percentile(speeds, 0.99);
        if (summary_.minimum_active_triangles ==
            std::numeric_limits<std::uint32_t>::max()) {
            summary_.minimum_active_triangles = 0U;
        }
        summary_.unique_inverted_triangles = static_cast<std::uint32_t>(std::count(
            inverted_triangles_.begin(), inverted_triangles_.end(), std::uint8_t{1U}));
        summary_.unique_active_inverted_triangles = static_cast<std::uint32_t>(std::count(
            active_inverted_triangles_.begin(), active_inverted_triangles_.end(),
            std::uint8_t{1U}));
        summary_.unique_active_inversions_per_instance.assign(instance_count_, 0U);
        const std::uint32_t triangles_per_instance =
            summary_.triangle_count / instance_count_;
        for (std::uint32_t triangle = 0U;
             triangle < active_inverted_triangles_.size(); ++triangle) {
            if (active_inverted_triangles_[triangle] != 0U) {
                ++summary_.unique_active_inversions_per_instance[
                    triangle / triangles_per_instance];
            }
        }
        return summary_;
    }

private:
    Summary summary_;
    std::uint32_t instance_count_{};
    std::vector<std::uint8_t> inverted_triangles_;
    std::vector<std::uint8_t> active_inverted_triangles_;
    std::optional<std::uint32_t> previous_broken_;
};

void write_csv_header(std::ofstream& output)
{
    output << "mode,capture_index,physics_frame,broken_edges,active_triangles,"
        "measured_triangles,abnormal_triangles,"
        "measured_active_triangles,abnormal_active_triangles,"
        "inversion_count,active_inversion_count,"
        "max_penetration,max_triangle_area_ratio,p99_triangle_area_ratio,"
        "max_triangle_area_id,max_render_edge_ratio,max_render_edge_triangle_id,"
        "min_active_triangle_area_ratio,max_active_triangle_area_ratio,"
        "p99_active_triangle_area_ratio,max_active_render_edge_ratio,"
        "max_voxel_speed,p99_voxel_speed,"
        "voxel_position_finite_failures,voxel_velocity_finite_failures,"
        "geometry_finite_failures,fluid_finite_failures,soft_finite_failures,wall_ms,"
        "skin_center_inside,skin_surface_inside,skin_sphere_inside,skin_max_clearance,"
        "skin_total_clearance,skin_min_area_ratio,skin_signed_volume,skin_nonfinite,"
        "skin_winding_agrees,step_gpu_total_ms\n";
}

void write_csv_frame(std::ofstream& output, std::string_view mode,
    std::size_t index, const FrameMetrics& frame, double wall_ms, double step_gpu_total_ms = 0.0)
{
    if (!output) return;
    output << mode << ',' << index << ',' << frame.physics_frame << ','
        << frame.broken_edges << ',' << frame.active_triangles << ','
        << frame.measured_triangles << ',' << frame.abnormal_triangles << ','
        << frame.measured_active_triangles << ','
        << frame.abnormal_active_triangles << ','
        << frame.inversion_count << ',' << frame.active_inversion_count << ','
        << frame.maximum_penetration << ','
        << frame.maximum_triangle_area_ratio << ','
        << frame.p99_triangle_area_ratio << ',' << frame.maximum_triangle_area_id << ','
        << frame.maximum_render_edge_ratio << ','
        << frame.maximum_render_edge_triangle_id << ','
        << frame.minimum_active_triangle_area_ratio << ','
        << frame.maximum_active_triangle_area_ratio << ','
        << frame.p99_active_triangle_area_ratio << ','
        << frame.maximum_active_render_edge_ratio << ','
        << frame.maximum_voxel_speed << ',' << frame.p99_voxel_speed << ','
        << frame.voxel_position_finite_failures << ','
        << frame.voxel_velocity_finite_failures << ','
        << frame.geometry_finite_failures << ','
        << frame.reported_fluid_finite_failures << ','
        << frame.reported_soft_finite_failures << ',' << wall_ms
        << ",,,,,,,,,," << step_gpu_total_ms << '\n';
}

void print_summary(const Summary& summary)
{
    const double transitions = static_cast<double>(
        summary.frame_count > 1U ? summary.frame_count - 1U : summary.frame_count);
    const double active_fraction = summary.triangle_count != 0U
        ? static_cast<double>(summary.minimum_active_triangles) /
            static_cast<double>(summary.triangle_count) : 0.0;
    const double abnormal_fraction = summary.measured_triangles != 0U
        ? static_cast<double>(summary.abnormal_triangles) /
            static_cast<double>(summary.measured_triangles) : 0.0;
    const double active_abnormal_fraction = summary.measured_active_triangles != 0U
        ? static_cast<double>(summary.abnormal_active_triangles) /
            static_cast<double>(summary.measured_active_triangles) : 0.0;
    std::cout << std::fixed << std::setprecision(6)
        << "METRICS,mode=" << summary.mode
        << ",frames=" << summary.frame_count
        << ",first_physics_frame=" << summary.first_physics_frame
        << ",last_physics_frame=" << summary.last_physics_frame
        << ",start_broken=" << summary.starting_broken_edges
        << ",end_broken=" << summary.ending_broken_edges
        << ",observed_new_breaks=" << summary.observed_new_breaks
        << ",mean_breaks_per_frame="
        << (transitions > 0.0 ? static_cast<double>(summary.observed_new_breaks) /
            transitions : 0.0)
        << ",max_breaks_in_frame=" << summary.maximum_breaks_in_one_frame
        << ",max_penetration=" << summary.maximum_penetration
        << ",max_triangle_area_ratio=" << summary.maximum_triangle_area_ratio
        << ",max_triangle_area_frame=" << summary.maximum_triangle_area_frame
        << ",max_triangle_area_id=" << summary.maximum_triangle_area_id
        << ",p99_triangle_area_ratio=" << summary.p99_triangle_area_ratio
        << ",max_render_edge_ratio=" << summary.maximum_render_edge_ratio
        << ",max_render_edge_frame=" << summary.maximum_render_edge_frame
        << ",max_render_edge_triangle_id="
        << summary.maximum_render_edge_triangle_id
        << ",abnormal_triangle_fraction=" << abnormal_fraction
        << ",max_frame_abnormal_fraction="
        << summary.maximum_frame_abnormal_fraction
        << ",min_active_triangle_area_ratio="
        << summary.minimum_active_triangle_area_ratio
        << ",max_active_triangle_area_ratio="
        << summary.maximum_active_triangle_area_ratio
        << ",p99_active_triangle_area_ratio="
        << summary.p99_active_triangle_area_ratio
        << ",active_abnormal_triangle_fraction="
        << active_abnormal_fraction
        << ",max_frame_active_abnormal_fraction="
        << summary.maximum_frame_active_abnormal_fraction
        << ",inversion_count=" << summary.inversion_count
        << ",active_inversion_count=" << summary.active_inversion_count
        << ",max_frame_inversion_count="
        << summary.maximum_frame_inversion_count
        << ",unique_inverted_triangles=" << summary.unique_inverted_triangles
        << ",unique_active_inverted_triangles="
        << summary.unique_active_inverted_triangles
        << ",max_active_render_edge_ratio="
        << summary.maximum_active_render_edge_ratio
        << ",max_voxel_speed=" << summary.maximum_voxel_speed
        << ",p99_voxel_speed=" << summary.p99_voxel_speed
        << ",minimum_active_triangle_fraction=" << active_fraction
        << ",voxel_position_finite_failures="
        << summary.voxel_position_finite_failures
        << ",voxel_velocity_finite_failures="
        << summary.voxel_velocity_finite_failures
        << ",geometry_finite_failures=" << summary.geometry_finite_failures
        << ",reported_fluid_finite_failures="
        << summary.reported_fluid_finite_failures
        << ",reported_soft_finite_failures="
        << summary.reported_soft_finite_failures
        << ",wall_ms=" << summary.wall_ms
        << ",mean_wall_ms="
        << (summary.frame_count != 0U ? summary.wall_ms /
            static_cast<double>(summary.frame_count) : 0.0)
        << '\n';
    std::cout << "INVERSION_INSTANCES,mode=" << summary.mode
        << ",unique_active_triangles=";
    for (std::size_t instance = 0U;
         instance < summary.unique_active_inversions_per_instance.size(); ++instance) {
        if (instance != 0U) std::cout << '|';
        std::cout << instance << ':'
            << summary.unique_active_inversions_per_instance[instance];
    }
    std::cout << '\n';
}

struct GateResult {
    bool finite{};
    bool triangle_normal{};
    bool orientation{};
};

GateResult evaluate_gates(const Summary& summary)
{
    const double active_fraction = summary.triangle_count != 0U
        ? static_cast<double>(summary.minimum_active_triangles) /
            static_cast<double>(summary.triangle_count) : 0.0;
    GateResult result;
    result.finite = summary.voxel_position_finite_failures == 0U &&
        summary.voxel_velocity_finite_failures == 0U &&
        summary.geometry_finite_failures == 0U &&
        summary.reported_fluid_finite_failures == 0U &&
        summary.reported_soft_finite_failures == 0U;
    result.triangle_normal = summary.measured_active_triangles != 0U &&
        summary.minimum_active_triangle_area_ratio >= minimum_active_area_ratio &&
        summary.maximum_active_triangle_area_ratio <= maximum_active_area_ratio &&
        summary.maximum_active_render_edge_ratio <= maximum_active_edge_ratio &&
        active_fraction >= minimum_active_triangle_fraction;
    result.orientation = summary.active_inversion_count == 0U;
    return result;
}

void print_gates(const Summary& summary, GateResult gates)
{
    const double active_fraction = summary.triangle_count != 0U
        ? static_cast<double>(summary.minimum_active_triangles) /
            static_cast<double>(summary.triangle_count) : 0.0;
    std::cout << std::fixed << std::setprecision(6)
        << "GATE,mode=" << summary.mode
        << ",finite=" << (gates.finite ? "PASS" : "FAIL")
        << ",triangle_normal=" << (gates.triangle_normal ? "PASS" : "FAIL")
        << ",orientation=" << (gates.orientation ? "PASS" : "FAIL")
        << ",inversion_count=" << summary.inversion_count
        << ",active_inversion_count=" << summary.active_inversion_count
        << ",unique_active_inverted_triangles="
        << summary.unique_active_inverted_triangles
        << ",required_active_inversion_count=0"
        << ",active_area_min=" << summary.minimum_active_triangle_area_ratio
        << ",active_area_max=" << summary.maximum_active_triangle_area_ratio
        << ",active_edge_max=" << summary.maximum_active_render_edge_ratio
        << ",active_triangle_fraction=" << active_fraction
        << ",required_area_min=" << minimum_active_area_ratio
        << ",required_area_max=" << maximum_active_area_ratio
        << ",required_edge_max=" << maximum_active_edge_ratio
        << ",required_active_fraction=" << minimum_active_triangle_fraction
        << '\n';
}

Summary audit_recording(const Capture& capture, const RenderTemplate& geometry,
    std::ofstream& csv)
{
    SummaryBuilder builder("captured",
        static_cast<std::uint32_t>(geometry.triangles.size()),
        geometry.instance_count);
    std::vector<double> area_ratios;
    std::vector<double> active_area_ratios;
    std::vector<double> speeds;
    area_ratios.reserve(geometry.triangles.size() * capture.frames.size());
    active_area_ratios.reserve(geometry.triangles.size() * capture.frames.size());
    speeds.reserve(capture.frames.front().soft_body.positions.size() * capture.frames.size());
    for (std::size_t index = 0U; index < capture.frames.size(); ++index) {
        const CaptureFrame& frame = capture.frames[index];
        if (!frame.has_soft_body) {
            throw std::runtime_error("capture contains a frame without soft-body state");
        }
        const FrameMetrics metrics = measure_frame(
            geometry, frame.soft_body, frame.statistics,
            area_ratios, active_area_ratios, speeds);
        builder.add(metrics, 0.0);
        write_csv_frame(csv, "captured", index, metrics, 0.0);
    }
    return builder.finish(area_ratios, active_area_ratios, speeds);
}

bool audit_skin_contact(const Capture& capture, const waterlab::SoftBodyAsset& asset,
    std::uint32_t stride, std::ofstream& csv)
{
    std::uint64_t centers{}, surface{}, spheres{}, nonfinite{};
    double maximum_depth{}, total_depth{};
    double minimum_area_ratio{std::numeric_limits<double>::infinity()};
    double first_signed_volume{std::numeric_limits<double>::quiet_NaN()};
    double minimum_signed_volume{std::numeric_limits<double>::infinity()};
    double maximum_signed_volume{-std::numeric_limits<double>::infinity()};
    bool skin_geometry_valid=true;
    std::optional<std::uint64_t> first_inside;
    bool winding_agrees=true;
    for (std::size_t index=0; index<capture.frames.size(); index+=stride) {
        const CaptureFrame& frame=capture.frames[index];
        const SkinContactMetrics m=measure_skin_contact(frame, asset,
            index == 0U || index + stride >= capture.frames.size() || !first_inside.has_value());
        centers += m.centers_inside; surface += m.surface_centers_inside;
        spheres += m.spheres_inside; nonfinite += m.nonfinite;
        maximum_depth=std::max(maximum_depth,m.maximum_depth); total_depth+=m.total_depth;
        minimum_area_ratio=std::min(minimum_area_ratio,m.minimum_area_ratio);
        if (!std::isfinite(first_signed_volume)) first_signed_volume=m.signed_volume;
        minimum_signed_volume=std::min(minimum_signed_volume,m.signed_volume);
        maximum_signed_volume=std::max(maximum_signed_volume,m.signed_volume);
        skin_geometry_valid = skin_geometry_valid && std::isfinite(m.minimum_area_ratio) &&
            m.minimum_area_ratio > 0.0 && std::isfinite(m.signed_volume) && m.signed_volume > 0.0;
        winding_agrees = winding_agrees && m.winding_agrees;
        if (m.centers_inside != 0U && !first_inside.has_value()) first_inside=frame.statistics.frame_index;
        std::cout << std::fixed << std::setprecision(6)
            << "SKIN_CONTACT,mode=captured,index=" << index
            << ",physics_frame=" << frame.statistics.frame_index
            << ",center_inside=" << m.centers_inside
            << ",surface_inside=" << m.surface_centers_inside
            << ",sphere_inside=" << m.spheres_inside
            << ",max_depth=" << m.maximum_depth << ",total_depth=" << m.total_depth
            << ",min_area_ratio=" << m.minimum_area_ratio
            << ",signed_volume=" << m.signed_volume
            << ",nonfinite=" << m.nonfinite
            << ",winding_agrees=" << (m.winding_agrees ? "true" : "false")
            << ",winding_check=sampled-first-aabb-candidate" << '\n';
        if (csv) {
            csv << "skin-contact," << index << ',' << frame.statistics.frame_index;
            for (unsigned column=0; column<27U; ++column) csv << ',';
            csv << m.centers_inside << ','
            << m.surface_centers_inside << ',' << m.spheres_inside << ',' << m.maximum_depth
            << ',' << m.total_depth << ',' << m.minimum_area_ratio << ',' << m.signed_volume
            << ',' << m.nonfinite << ',' << (m.winding_agrees ? 1 : 0) << ",\n";
        }
    }
    const bool precontact_valid = measure_skin_contact(capture.frames.front(), asset, true).centers_inside == 0U;
    const bool full_coverage = stride == 1U;
    const bool skin_quality = skin_quality_pass(minimum_area_ratio, first_signed_volume,
        minimum_signed_volume, maximum_signed_volume, skin_geometry_valid);
    const bool containment_pass = full_coverage && centers == 0U && nonfinite == 0U &&
        winding_agrees && precontact_valid && skin_geometry_valid && skin_quality;
    const char* gate_status = containment_pass ? "PASS" :
        (!full_coverage && centers == 0U && nonfinite == 0U && winding_agrees &&
            precontact_valid && skin_geometry_valid && skin_quality ? "INCONCLUSIVE" : "FAIL");
    std::cout << std::fixed << std::setprecision(6)
        << "SKIN_CONTACT_SUMMARY,mode=captured,stride=" << stride
        << ",sampled_frames=" << ((capture.frames.size()+stride-1U)/stride)
        << ",center_inside_total=" << centers << ",surface_inside_total=" << surface
        << ",sphere_inside_total=" << spheres << ",max_depth=" << maximum_depth
        << ",total_depth=" << total_depth << ",nonfinite=" << nonfinite
        << ",first_inside_frame=" << (first_inside.has_value() ? std::to_string(*first_inside) : "none")
        << ",first_frame_inside=" << (precontact_valid ? 0 : 1)
        << ",winding_agrees=" << (winding_agrees ? "true" : "false")
        << ",winding_check=sampled-first-aabb-candidate"
        << ",skin_min_area_ratio=" << minimum_area_ratio
        << ",skin_first_signed_volume=" << first_signed_volume
        << ",skin_min_signed_volume=" << minimum_signed_volume
        << ",skin_max_signed_volume=" << maximum_signed_volume
        << ",skin_geometry_valid=" << (skin_geometry_valid ? "true" : "false")
        << ",skin_quality_area_min_limit=" << minimum_skin_triangle_area_ratio
        << ",skin_quality_volume_ratio_limits=" << minimum_skin_volume_ratio
        << ':' << maximum_skin_volume_ratio
        << ",skin_quality_pass=" << (skin_quality ? "true" : "false")
        << ",full_frame_coverage=" << (full_coverage ? "true" : "false") << '\n';
    std::cout << "CONTAINMENT_GATE,status=" << gate_status
        << ",reason=" << (!precontact_valid ? "invalid-precontact-start" :
            nonfinite != 0U ? "nonfinite" : !winding_agrees ? "winding-disagreement" :
            !skin_geometry_valid ? "invalid-skin-geometry" : centers != 0U ? "inside-voxels" :
            !skin_quality ? "skin-quality" :
            !full_coverage ? "sampled-frames" : "no-inside-voxels")
        << ",measurement=actual-physical-triangle-parity-and-closest-distance\n";
    return containment_pass;
}

Summary resimulate_from_rest(const Capture& capture,
    const waterlab::SoftBodyAsset& asset, const RenderTemplate& geometry,
    std::ofstream& csv, bool saved_start = false, bool skin_contact = false,
    std::uint32_t stride = 1U, bool* containment_pass_out = nullptr,
    bool render_contact = false)
{
    const CaptureFrame& first = capture.frames.front();
    if (!first.options.obstacle_course) {
        throw std::runtime_error("soft-body capture is not an obstacle-course simulation");
    }
    waterlab::SoftBodyOptions soft_options;
    soft_options.instance_count = first.soft_body.statistics.instance_count;
    soft_options.fixed_dt = first.options.fixed_dt;
    soft_options.solver_substeps = first.options.physics_iterations;
    // These app composition settings are not serialized in SoftBodyState.
    // They match gallery::make_context_deformable(water_course).
    soft_options.spring_solver_iterations = 16U;
    soft_options.maximum_speed = first.options.maximum_skin_speed;
    soft_options.strength_multiplier = first.soft_body.strength_multiplier;
    soft_options.use_course_layout = true;
    soft_options.require_1000_voxels = asset.rest_voxels.size() == 1'000U;
    soft_options.course_board_collisions = true;
    soft_options.arena = first.options.arena;
    soft_options.unbonded_voxel_collisions = true;

    // The normal mode deliberately starts from authored rest.  Saved-start mode
    // restores only frame zero (the pre-contact capture state), then advances;
    // it never repeatedly snaps to recorded later states.
    waterlab::HybridDroplet droplet(first.options);
    waterlab::SoftBodyCourse soft_body(asset, soft_options);
    SummaryBuilder builder(saved_start ? "saved-start-resimulation" : "fresh-rest-resimulation",
        static_cast<std::uint32_t>(geometry.triangles.size()),
        geometry.instance_count, 0U);
    std::vector<double> area_ratios;
    std::vector<double> active_area_ratios;
    std::vector<double> speeds;
    area_ratios.reserve(geometry.triangles.size() * capture.frames.size());
    active_area_ratios.reserve(geometry.triangles.size() * capture.frames.size());
    speeds.reserve(first.soft_body.positions.size() * capture.frames.size());
    DriftMetrics drift;
    std::uint64_t contact_centers{}, contact_surface{}, contact_spheres{}, contact_nonfinite{};
    double contact_max_depth{}, contact_total_depth{}, physics_ms{}, wall_ms_total{};
    double contact_minimum_area_ratio{std::numeric_limits<double>::infinity()};
    double contact_first_signed_volume{std::numeric_limits<double>::quiet_NaN()};
    double contact_minimum_signed_volume{std::numeric_limits<double>::infinity()};
    double contact_maximum_signed_volume{-std::numeric_limits<double>::infinity()};
    bool contact_skin_geometry_valid=true;
    std::vector<double> gpu_step_ms, wall_step_ms;
    std::optional<std::uint64_t> first_contact;
    bool contact_winding=true;
    std::uint64_t render_inside{}, render_peak{}, render_frames{}, render_nonfinite{};
    std::size_t minimum_render_vertices=std::numeric_limits<std::size_t>::max();
    double render_depth{};
    bool render_winding=true;
    const auto record_render_contact = [&](const waterlab::HybridState& state,
        const waterlab::SoftBodyState& soft, const std::vector<float3>& positions,
        std::uint64_t physics_frame) {
        if (!render_contact) return;
        if (positions.size()!=geometry.rest_positions.size())
            throw std::runtime_error("render contact vertex count mismatch");
        const auto points=active_render_points(positions,geometry.triangles,soft.active_render_triangles);
        minimum_render_vertices=std::min(minimum_render_vertices,points.size());
        const auto m=measure_skin_points(state.options,state.skin_positions,points,true);
        render_inside+=m.centers_inside; render_nonfinite+=m.nonfinite;
        render_peak=std::max<std::uint64_t>(render_peak,m.centers_inside);
        render_depth=std::max(render_depth,m.maximum_depth);
        render_winding=render_winding && m.winding_agrees; ++render_frames;
        std::cout<<"RENDER_CONTACT,frame="<<physics_frame<<",active_vertices="<<points.size()
            <<",count="<<m.centers_inside<<",maxdepth="<<m.maximum_depth
            <<",nonfinite="<<m.nonfinite<<'\n';
    };
    const auto record_contact = [&](std::size_t index, const waterlab::HybridState& state,
        const waterlab::SoftBodyState& soft, const waterlab::HybridStatistics& statistics) {
        if (!skin_contact || index % stride != 0U) return;
        CaptureFrame frame; frame.options=state.options; frame.statistics=statistics;
        frame.state=state; frame.soft_body=soft;
        const SkinContactMetrics m=measure_skin_contact(frame,asset,
            index==0U || index+stride>=capture.frames.size() || !first_contact.has_value());
        contact_centers+=m.centers_inside; contact_surface+=m.surface_centers_inside;
        contact_spheres+=m.spheres_inside; contact_nonfinite+=m.nonfinite;
        contact_max_depth=std::max(contact_max_depth,m.maximum_depth); contact_total_depth+=m.total_depth;
        contact_minimum_area_ratio=std::min(contact_minimum_area_ratio,m.minimum_area_ratio);
        if (!std::isfinite(contact_first_signed_volume)) contact_first_signed_volume=m.signed_volume;
        contact_minimum_signed_volume=std::min(contact_minimum_signed_volume,m.signed_volume);
        contact_maximum_signed_volume=std::max(contact_maximum_signed_volume,m.signed_volume);
        contact_skin_geometry_valid = contact_skin_geometry_valid &&
            std::isfinite(m.minimum_area_ratio) && m.minimum_area_ratio > 0.0 &&
            std::isfinite(m.signed_volume) && m.signed_volume > 0.0;
        contact_winding=contact_winding && m.winding_agrees;
        if (m.centers_inside && !first_contact.has_value()) first_contact=statistics.frame_index;
        std::cout << "SKIN_CONTACT,mode=" << (saved_start ? "saved-start-resimulation" : "fresh-rest-resimulation")
            << ",index=" << index << ",physics_frame=" << statistics.frame_index
            << ",center_inside=" << m.centers_inside << ",surface_inside=" << m.surface_centers_inside
            << ",sphere_inside=" << m.spheres_inside << ",max_depth=" << m.maximum_depth
            << ",total_depth=" << m.total_depth << ",min_area_ratio=" << m.minimum_area_ratio
            << ",signed_volume=" << m.signed_volume << ",nonfinite=" << m.nonfinite
            << ",winding_agrees=" << (m.winding_agrees ? "true" : "false")
            << ",winding_check=sampled-first-aabb-candidate" << '\n';
        if (csv) {
            csv << "skin-contact-" << (saved_start ? "saved-start-resimulation" : "fresh-rest-resimulation")
                << ',' << index << ',' << statistics.frame_index;
            for (unsigned column=0; column<27U; ++column) csv << ',';
            csv << m.centers_inside << ',' << m.surface_centers_inside << ',' << m.spheres_inside
                << ',' << m.maximum_depth << ',' << m.total_depth << ',' << m.minimum_area_ratio
                << ',' << m.signed_volume << ',' << m.nonfinite << ',' << (m.winding_agrees ? 1 : 0)
                << ",\n";
        }
    };

    std::size_t start_index = 0U;
    if (saved_start) {
        droplet.restore_state(first.state);
        soft_body.restore_state(first.soft_body);
        waterlab::HybridState restored; waterlab::SoftBodyState restored_soft;
        droplet.capture_state(restored); soft_body.capture_state(restored_soft);
        add_drift(drift,restored,first.state,restored_soft,first.soft_body);
        const bool restore_positions_identical=drift.positions_identical;
        const bool restore_velocities_identical=drift.velocities_identical;
        std::cout << "RESTORE_CHECK,position_max=" << drift.maximum_position
            << ",velocity_max=" << drift.maximum_velocity
            << ",positions_bit_identical=" << (restore_positions_identical ? "true" : "false")
            << ",velocities_bit_identical=" << (restore_velocities_identical ? "true" : "false")
            << ",status=" << ((drift.maximum_position <= 1.0e-6 && drift.maximum_velocity <= 1.0e-6) ? "PASS" : "WARN") << '\n';
        record_contact(0U,restored,restored_soft,droplet.statistics());
        if (render_contact) {
            const auto view=soft_body.render_view();
            std::vector<float3> positions(view.vertex_count);
            const auto result=cudaMemcpy(positions.data(),view.positions,
                positions.size()*sizeof(float3),cudaMemcpyDeviceToHost);
            if (result!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
            record_render_contact(restored,restored_soft,positions,droplet.statistics().frame_index);
        }
        const FrameMetrics metrics = measure_frame(geometry, restored_soft, droplet.statistics(),
            area_ratios, active_area_ratios, speeds);
        builder.add(metrics, 0.0);
        write_csv_frame(csv, "saved-start-resimulation", 0U, metrics, 0.0);
        start_index = 1U;
    }
    for (std::size_t index = start_index; index < capture.frames.size(); ++index) {
        const CaptureFrame& recorded = capture.frames[index];
        droplet.set_runtime_options(recorded.options);
        soft_body.set_strength_multiplier(recorded.soft_body.strength_multiplier);
        soft_body.set_solver_substeps(recorded.options.physics_iterations);
        const auto begin = std::chrono::steady_clock::now();
        const waterlab::HybridTimings step_timings=droplet.step(recorded.rectangle_control_force,
            recorded.rectangle_control_torque, nullptr, &soft_body);
        const double step_wall_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - begin).count();
        waterlab::SoftBodyState state;
        soft_body.capture_state(state);
        waterlab::HybridState replayed;
        if (saved_start || skin_contact || render_contact) droplet.capture_state(replayed);
        const waterlab::SoftBodyRenderView render = soft_body.render_view();
        std::vector<float3> render_positions(render.vertex_count);
        if (!render_positions.empty()) {
            const cudaError_t status = cudaMemcpy(render_positions.data(), render.positions,
                render_positions.size() * sizeof(float3), cudaMemcpyDeviceToHost);
            if (status != cudaSuccess) {
                throw std::runtime_error(std::string("download runtime render positions: ") +
                    cudaGetErrorString(status));
            }
        }
        const double wall_ms = step_wall_ms;
        physics_ms += step_timings.gpu_total_ms(); wall_ms_total += wall_ms;
        gpu_step_ms.push_back(step_timings.gpu_total_ms()); wall_step_ms.push_back(wall_ms);
        if (saved_start) add_drift(drift,replayed,recorded.state,state,recorded.soft_body);
        if (skin_contact) record_contact(index,replayed,state,droplet.statistics());
        record_render_contact(replayed,state,render_positions,droplet.statistics().frame_index);
        const FrameMetrics metrics = measure_frame(
            geometry, state, droplet.statistics(),
            area_ratios, active_area_ratios, speeds, &render_positions);
        builder.add(metrics, wall_ms);
        write_csv_frame(csv, saved_start ? "saved-start-resimulation" : "fresh-rest-resimulation",
            index, metrics, wall_ms, step_timings.gpu_total_ms());
    }
    if (skin_contact) {
        const bool precontact_valid = !saved_start || contact_centers == 0U || first_contact != first.statistics.frame_index;
        const bool full_coverage=stride==1U;
        const bool contact_skin_quality = skin_quality_pass(contact_minimum_area_ratio,
            contact_first_signed_volume, contact_minimum_signed_volume,
            contact_maximum_signed_volume, contact_skin_geometry_valid);
        const bool pass=full_coverage && contact_centers==0U && contact_nonfinite==0U &&
            contact_winding && precontact_valid && contact_skin_geometry_valid && contact_skin_quality;
        const char* gate_status=pass ? "PASS" :
            (!full_coverage && contact_centers==0U && contact_nonfinite==0U && contact_winding &&
                precontact_valid && contact_skin_geometry_valid && contact_skin_quality ? "INCONCLUSIVE" : "FAIL");
        if (containment_pass_out != nullptr) *containment_pass_out = pass;
        const double step_count = static_cast<double>(gpu_step_ms.size());
        const double gpu_median = percentile(gpu_step_ms, 0.5);
        const double gpu_p95 = percentile(gpu_step_ms, 0.95);
        std::cout << "SKIN_CONTACT_SUMMARY,mode=" << (saved_start ? "saved-start-resimulation" : "fresh-rest-resimulation")
            << ",stride=" << stride << ",center_inside_total=" << contact_centers
            << ",surface_inside_total=" << contact_surface << ",sphere_inside_total=" << contact_spheres
            << ",max_depth=" << contact_max_depth << ",total_depth=" << contact_total_depth
            << ",nonfinite=" << contact_nonfinite << ",first_inside_frame="
            << (first_contact ? std::to_string(*first_contact) : "none")
            << ",winding_agrees=" << (contact_winding ? "true" : "false")
            << ",winding_check=sampled-first-aabb-candidate"
            << ",skin_min_area_ratio=" << contact_minimum_area_ratio
            << ",skin_first_signed_volume=" << contact_first_signed_volume
            << ",skin_min_signed_volume=" << contact_minimum_signed_volume
            << ",skin_max_signed_volume=" << contact_maximum_signed_volume
            << ",skin_geometry_valid=" << (contact_skin_geometry_valid ? "true" : "false")
            << ",skin_quality_area_min_limit=" << minimum_skin_triangle_area_ratio
            << ",skin_quality_volume_ratio_limits=" << minimum_skin_volume_ratio
            << ':' << maximum_skin_volume_ratio
            << ",skin_quality_pass=" << (contact_skin_quality ? "true" : "false")
            << ",full_frame_coverage=" << (full_coverage ? "true" : "false")
            << ",physics_gpu_total_ms_sum=" << physics_ms
            << ",physics_gpu_total_ms_mean=" << (step_count ? physics_ms/step_count : 0.0)
            << ",physics_gpu_total_ms_median=" << gpu_median
            << ",physics_gpu_total_ms_p95=" << gpu_p95
            << ",step_wall_ms_sum=" << wall_ms_total
            << ",step_wall_ms_mean=" << (step_count ? wall_ms_total/step_count : 0.0) << '\n';
        std::cout << "CONTAINMENT_GATE,mode=" << (saved_start ? "saved-start-resimulation" : "fresh-rest-resimulation")
            << ",status=" << gate_status << '\n';
    }
    if (!skin_contact && containment_pass_out != nullptr) *containment_pass_out = true;
    if (render_contact) {
        const bool pass=render_inside==0U && render_nonfinite==0U && render_winding &&
            render_frames==capture.frames.size() && minimum_render_vertices>0U;
        std::cout<<"RENDER_CONTACT_SUMMARY,frames="<<render_frames<<",inside_total="<<render_inside
            <<",peak_inside="<<render_peak<<",maxdepth="<<render_depth
            <<",nonfinite="<<render_nonfinite<<",winding_agrees="<<render_winding
            <<",minimum_active_vertices="<<minimum_render_vertices<<'\n'
            <<"RENDER_CONTACT_GATE,status="<<(pass?"PASS":"FAIL")
            <<",measurement=active-render-vertices-versus-physical-skin\n";
        if (!pass && containment_pass_out!=nullptr) *containment_pass_out=false;
    }
    if (saved_start) std::cout << "STATE_DRIFT,position_max=" << drift.maximum_position
        << ",position_rms=" << drift.rms_position() << ",velocity_max=" << drift.maximum_velocity
        << ",velocity_rms=" << drift.rms_velocity()
        << ",all_samples_positions_bit_identical=" << (drift.positions_identical ? "true" : "false")
        << ",all_samples_velocities_bit_identical=" << (drift.velocities_identical ? "true" : "false") << '\n';
    return builder.finish(area_ratios, active_area_ratios, speeds);
}

CommandLine parse_command_line(int argc, char** argv)
{
    CommandLine options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument = argv[index];
        if (argument == "--skin-contact-self-test") {
            skin_contact_self_test();
            std::cout << "skin-contact analytic reference passed\n";
            std::exit(0);
        }
        if (argument == "--help") {
            std::cout << "usage: meshprep-soft-body-capture-audit [CAPTURE] "
                "[--asset FILE] [--csv FILE] [--baseline-only] [--skin-contact] "
                "[--resimulate-saved-start] [--stride N] [--render-contact]\n"
                "CAPTURE defaults to /tmp/meshprep-hybrid-captures/LAST_CAPTURE.txt.\n"
                "--skin-contact-self-test checks the analytic CPU reference without a capture.\n";
            std::exit(0);
        }
        if (argument == "--baseline-only") {
            options.baseline_only = true;
            continue;
        }
        if (argument == "--skin-contact") { options.skin_contact = true; continue; }
        if (argument == "--render-contact") {
            options.render_contact=true; options.skin_contact=true; continue;
        }
        if (argument == "--resimulate-saved-start") {
            options.resimulate_saved_start = true;
            continue;
        }
        if (argument == "--stride") {
            if (++index >= argc) throw std::invalid_argument("--stride requires a positive integer");
            const std::string_view value=argv[index];
            const auto parsed=std::from_chars(value.data(),value.data()+value.size(),options.stride);
            if (parsed.ec != std::errc{} || parsed.ptr != value.data()+value.size() || options.stride == 0U)
                throw std::invalid_argument("--stride requires a positive integer");
            continue;
        }
        if (argument == "--asset" || argument == "--csv") {
            if (index + 1 >= argc) {
                throw std::invalid_argument(std::string(argument) + " requires a path");
            }
            const std::filesystem::path value = argv[++index];
            if (argument == "--asset") options.asset_path = value;
            else options.csv_path = value;
            continue;
        }
        if (!argument.empty() && argument.front() == '-') {
            throw std::invalid_argument("unknown option " + std::string(argument));
        }
        if (!options.capture_path.empty()) {
            throw std::invalid_argument("only one capture path may be specified");
        }
        options.capture_path = argument;
    }
    if (options.render_contact && options.baseline_only)
        throw std::invalid_argument("--render-contact requires GPU resimulation, not --baseline-only");
    if (options.render_contact && options.stride!=1U)
        throw std::invalid_argument("--render-contact requires --stride 1 for full coverage");
    return options;
}

int run(int argc, char** argv)
{
    const CommandLine options = parse_command_line(argc, argv);
    const Capture capture = load_capture(options.capture_path);
    if (capture.frames.empty() || !capture.frames.front().has_soft_body) {
        throw std::runtime_error("capture has no soft-body frames to audit");
    }
    const waterlab::SoftBodyAsset asset =
        waterlab::load_soft_body_asset(options.asset_path.string());
    const auto instances = capture.frames.front().soft_body.statistics.instance_count;
    const RenderTemplate geometry = make_render_template(asset, instances);

    std::ofstream csv;
    if (!options.csv_path.empty()) {
        if (options.csv_path.has_parent_path()) {
            std::filesystem::create_directories(options.csv_path.parent_path());
        }
        csv.open(options.csv_path, std::ios::trunc);
        if (!csv) throw std::runtime_error("cannot create " + options.csv_path.string());
        write_csv_header(csv);
    }

    std::cout << "CAPTURE,path=" << capture.path.string()
        << ",version=" << capture.header.version
        << ",frames=" << capture.frames.size()
        << ",particles=" << capture.header.particle_count
        << ",skin_vertices=" << capture.header.skin_vertex_count
        << ",soft_instances=" << instances
        << ",soft_voxels=" << capture.frames.front().soft_body.positions.size()
        << ",render_triangles_per_instance=" << asset.render_triangles.size()
        << ",render_triangles=" << geometry.triangles.size() << '\n';
    if (capture.header.version == 4U) {
        std::cout << "NOTICE,version=4,edge_damage=unavailable,"
            "fresh_future_fracture_timing_may_diverge=true\n";
    }
    const Summary captured = audit_recording(capture, geometry, csv);
    print_summary(captured);
    const GateResult captured_gates = evaluate_gates(captured);
    print_gates(captured, captured_gates);
    bool captured_containment_pass = true;
    if (options.skin_contact) {
        skin_contact_self_test();
        captured_containment_pass = audit_skin_contact(capture, asset, options.stride, csv);
    }
    if (options.baseline_only) {
        const bool pass = captured_gates.finite && captured_gates.triangle_normal &&
            captured_gates.orientation && captured_containment_pass;
        std::cout << "RESULT," << (pass ? "PASS" : "FAIL")
            << ",mode=captured"
            << ",finite_gate=" << (captured_gates.finite ? "PASS" : "FAIL")
            << ",triangle_normal_gate="
            << (captured_gates.triangle_normal ? "PASS" : "FAIL")
            << ",orientation_gate="
            << (captured_gates.orientation ? "PASS" : "FAIL")
            << ",containment_gate=" << (options.skin_contact
                ? (captured_containment_pass ? "PASS" : "FAIL") : "NOT_CHECKED") << '\n';
        return pass ? 0 : 2;
    }

    int device_count{};
    const cudaError_t device_status = cudaGetDeviceCount(&device_count);
    if (device_status != cudaSuccess || device_count == 0) {
        std::cout << "RESIMULATION,status=SKIP,reason=no-CUDA-device,"
            "captured_state_was_not_used_as_initial_state=true\n";
        return 77;
    }
    std::cout << "RESIMULATION,status=BEGIN,initial_state="
        << (options.resimulate_saved_start ? "captured-frame-0" : "authored-rest")
        << ",captured_state_was_not_used_as_initial_state="
        << (options.resimulate_saved_start ? "false" : "true")
        << ",soft_options_assumption=course-gallery-16-iterations-unbonded-collision\n";
    bool resimulated_containment_pass = true;
    const Summary resimulated = resimulate_from_rest(capture, asset, geometry, csv,
        options.resimulate_saved_start, options.skin_contact, options.stride,
        &resimulated_containment_pass, options.render_contact);
    print_summary(resimulated);
    const GateResult resimulated_gates = evaluate_gates(resimulated);
    print_gates(resimulated, resimulated_gates);
    std::cout << std::fixed << std::setprecision(6)
        << "COMPARISON,p99_triangle_area_ratio_delta="
        << resimulated.p99_triangle_area_ratio - captured.p99_triangle_area_ratio
        << ",max_triangle_area_ratio_delta="
        << resimulated.maximum_triangle_area_ratio - captured.maximum_triangle_area_ratio
        << ",max_render_edge_ratio_delta="
        << resimulated.maximum_render_edge_ratio - captured.maximum_render_edge_ratio
        << ",inversion_count_delta="
        << static_cast<std::int64_t>(resimulated.inversion_count) -
            static_cast<std::int64_t>(captured.inversion_count)
        << ",new_break_delta="
        << static_cast<std::int64_t>(resimulated.observed_new_breaks) -
            static_cast<std::int64_t>(captured.observed_new_breaks)
        << '\n';
    const bool pass = resimulated_gates.finite && resimulated_gates.triangle_normal &&
        resimulated_gates.orientation && resimulated_containment_pass;
    std::cout << "RESULT," << (pass ? "PASS" : "FAIL")
        << ",mode=" << (options.resimulate_saved_start ? "saved-start-resimulation" : "fresh-rest-resimulation")
        << ",finite_gate=" << (resimulated_gates.finite ? "PASS" : "FAIL")
        << ",triangle_normal_gate="
        << (resimulated_gates.triangle_normal ? "PASS" : "FAIL")
        << ",orientation_gate="
        << (resimulated_gates.orientation ? "PASS" : "FAIL")
        << ",containment_gate=" << (options.skin_contact
            ? (resimulated_containment_pass ? "PASS" : "FAIL") : "NOT_CHECKED")
        << ",captured_triangle_normal_gate="
        << (captured_gates.triangle_normal ? "PASS" : "FAIL")
        << ",captured_orientation_gate="
        << (captured_gates.orientation ? "PASS" : "FAIL") << '\n';
    return pass ? 0 : 2;
}

} // namespace

int main(int argc, char** argv)
{
    try {
        return run(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "capture audit failed: " << error.what() << '\n';
        return 1;
    }
}
