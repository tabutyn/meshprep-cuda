// SPDX-License-Identifier: MIT
#include "hybrid_lab.hpp"
#include "water_lab.hpp"

#include <cuda_runtime_api.h>
#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr std::uint32_t settle = 600, press = 780, face_hold = 960, slide = 1140,
    edge_hold = 1440, withdraw = 1620, total_frames = 1980, smooth_half_width = 7;
constexpr float edge_radius = 0.08F, recovery_shape = 0.0375F, recovery_speed = 0.05F;
constexpr double reference_vibration_rms = 0.004002695;
constexpr double reference_press_vector_rms = 0.00908244;
constexpr double reference_max_penetration = 0.00444844;
constexpr double reference_intersection_frames = 777.0;
constexpr double reference_contact_min_area = 0.852770;
constexpr double reference_contact_max_stretch = 1.02565;
constexpr double reference_recovery_seconds = 1.63333342;
constexpr double reference_skin_physics_ms = 0.060416;
constexpr double fold_reference_face_hold_rms = 0.0371224074;
constexpr double fold_reference_slide_rms = 0.0301510385;
constexpr double fold_reference_max_penetration = 0.0173502266;
constexpr double fold_reference_contact_min_area = 0.398994952;
constexpr double fold_reference_contact_max_stretch = 1.93132201;
constexpr double fold_reference_recovery_seconds = 2.80000015;

struct Args {
    std::uint32_t runs{5}, iterations{1};
    std::filesystem::path output{"/tmp/parallel-mater-contact-experiment"};
    bool stiff_material{};
    bool verbose{};
    float box_damping{64.0F};
    float pressed_x{0.72F};
    float particle_skin_distance{0.05F};
    float particle_skin_damping{12.0F};
    std::string label{"selected"};
};

struct PhaseMetrics {
    double vector_rms{}, vector_p99{}, vector_peak{}, worst_vertex_rms{},
        tangential_rms{};
};

enum Phase : std::size_t { press_phase, face_hold_phase, slide_phase,
    edge_hold_phase, withdraw_phase, recovery_phase, phase_count };

constexpr std::array<const char*, phase_count> phase_names{
    "press", "face_hold", "slide", "edge_hold", "withdraw", "recovery"};
constexpr std::array<std::pair<std::uint32_t, std::uint32_t>, phase_count> phase_ranges{{
    {settle, press}, {press, face_hold}, {face_hold, slide},
    {slide, edge_hold}, {edge_hold, withdraw}, {withdraw, total_frames}}};

struct Frame {
    float3 target{}, control{};
    waterlab::RectangleState box{};
    double work{}, cumulative_work{}, penetration{}, penetration_sum{}, min_area{INFINITY},
        max_stretch{}, shape_error{}, speed_rms{}, vibration{}, vibration_peak{}, diagnostic_ms{},
        tracked_vector_rms{}, tracked_vector_p99{}, tracked_vector_peak{}, contact_impulse{},
        contact_pair_work{}, contact_min_area{INFINITY}, contact_max_stretch{}, normal_angle_max{},
        edge_support_fraction{NAN}, edge_max_particle_distance{}, particle_edge_circulation_rms{},
        particle_edge_circulation_peak{};
    std::uint32_t penetrated{}, intersections{}, onsets{}, skips{}, switches{}, switches90{},
        switches180{}, edge_vertices{}, outside{}, finite_failures{}, bad_positions{}, bad_forces{},
        tracked_vertices{}, owner_changes{}, normal_angle45{}, rest_orientation_reversals{},
        particle_cap_hits{}, skin_cap_hits{}, box_cap_hits{}, circulating_particles{};
    float skin_ms{}, physics_ms{};
};

struct Result {
    double vibration{}, vibration_peak{}, hold_ratio{INFINITY}, penetration{}, penetration_integral{},
        min_area{INFINITY}, max_stretch{}, recovery{INFINITY}, skin_ms{}, physics_ms{},
        contact_impulse{}, contact_pair_work{}, contact_min_area{INFINITY}, contact_max_stretch{},
        normal_angle_max{}, minimum_edge_support_fraction{1.0}, maximum_edge_particle_distance{},
        particle_edge_circulation_rms{}, particle_edge_circulation_peak{};
    std::uint32_t intersection_frames{}, max_intersections{}, outside{}, finite_failures{};
    std::uint64_t onsets{}, skips{}, switches{}, switches90{}, switches180{}, owner_changes{},
        normal_angle45{}, rest_orientation_reversals{}, particle_cap_hits{}, skin_cap_hits{}, box_cap_hits{};
    std::uint32_t tracked_vertices{};
    std::array<PhaseMetrics, phase_count> phases{}, full_phases{};
};

float3 add(float3 a, float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
float3 sub(float3 a, float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
float3 mul(float3 a, float s) { return make_float3(a.x * s, a.y * s, a.z * s); }
double dot(float3 a, float3 b) { return double(a.x) * b.x + double(a.y) * b.y + double(a.z) * b.z; }
float3 cross(float3 a, float3 b) { return make_float3(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x); }
double norm(float3 a) { return std::sqrt(dot(a, a)); }
bool finite(float3 a) { return std::isfinite(a.x) && std::isfinite(a.y) && std::isfinite(a.z); }
float3 rotate_y(float3 a, float angle) {
    const float c = std::cos(angle), s = std::sin(angle);
    return make_float3(c*a.x+s*a.z, a.y, -s*a.x+c*a.z);
}
float3 lerp(float3 a, float3 b, float t) { return add(a, mul(sub(b, a), t)); }
float3 clamp_norm(float3 a, float limit) { const double n = norm(a); return n > limit ? mul(a, limit / float(n)) : a; }

bool parse_u32(std::string_view text, std::uint32_t& value) {
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    return result.ec == std::errc{} && result.ptr == text.data() + text.size();
}

bool parse_float(std::string_view text, float& value) {
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    return result.ec == std::errc{} && result.ptr == text.data() + text.size() &&
        std::isfinite(value);
}

bool valid_label(std::string_view value) {
    return !value.empty() && std::all_of(value.begin(), value.end(), [](char c) {
        return std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_';
    });
}

bool parse(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; ++i) {
        const std::string_view key = argv[i];
        if (key == "--verbose") {
            args.verbose = true;
            continue;
        }
        if (key == "--help" || i + 1 == argc) return false;
        const std::string_view value = argv[++i];
        if (key == "--runs") {
            if (!parse_u32(value, args.runs) || args.runs == 0) return false;
        } else if (key == "--iterations") {
            if (!parse_u32(value, args.iterations) || args.iterations == 0 ||
                args.iterations > waterlab::HybridDroplet::maximum_physics_iterations) return false;
        } else if (key == "--output") args.output = value;
        else if (key == "--label") {
            if (!valid_label(value)) return false;
            args.label = value;
        } else if (key == "--box-damping") {
            if (!parse_float(value, args.box_damping) || args.box_damping < 0.0F) return false;
        } else if (key == "--pressed-x") {
            if (!parse_float(value, args.pressed_x) || args.pressed_x < 0.20F ||
                args.pressed_x > 0.90F) return false;
        } else if (key == "--particle-skin-distance") {
            if (!parse_float(value, args.particle_skin_distance) ||
                args.particle_skin_distance <= 0.0F ||
                args.particle_skin_distance >= 0.30F) return false;
        } else if (key == "--particle-skin-damping") {
            if (!parse_float(value, args.particle_skin_damping) ||
                args.particle_skin_damping < 0.0F) return false;
        }
        else if (key == "--material") {
            if (value == "stiff") args.stiff_material = true;
            else if (value != "default") return false;
        }
        else return false;
    }
    return true;
}

void cuda_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

float3 target_at(std::uint32_t frame, float pressed_x) {
    const float3 start{1.45F,0,0}, pressed{pressed_x,0,0}, edge{pressed_x,0.30F,0}, out{1.45F,0.30F,0};
    if (frame < settle) return start;
    if (frame < press) return lerp(start, pressed, float(frame-settle+1) / float(press-settle));
    if (frame < face_hold) return pressed;
    if (frame < slide) return lerp(pressed, edge, float(frame-face_hold+1) / float(slide-face_hold));
    if (frame < edge_hold) return edge;
    if (frame < withdraw) return lerp(edge, out, float(frame-edge_hold+1) / float(withdraw-edge_hold));
    return out;
}

float3 controller(float3 target, const waterlab::RectangleState& box, const waterlab::HybridOptions& options) {
    const float c = 2 * std::sqrt(options.rectangle_target_stiffness * options.rectangle_mass);
    return clamp_norm(add(mul(sub(target, box.center), options.rectangle_target_stiffness),
        mul(box.velocity, -options.rectangle_target_damping_ratio * c)), options.maximum_rectangle_control_force);
}

struct BoxSample { double distance{}; float3 normal{}; };

BoxSample box_sample(float3 p, const waterlab::RectangleState& box) {
    const float3 local = rotate_y(sub(p, box.center), -box.yaw);
    const double x = std::abs(double(local.x))-box.half_extents.x,
        y = std::abs(double(local.y))-box.half_extents.y,
        z = std::abs(double(local.z))-box.half_extents.z;
    const float3 outside{float(std::max(x,0.0)),float(std::max(y,0.0)),float(std::max(z,0.0))};
    const double outside_length=norm(outside); float3 local_normal{};
    if (outside_length>1e-8) local_normal=make_float3(std::copysign(outside.x,local.x)/float(outside_length),
        std::copysign(outside.y,local.y)/float(outside_length),std::copysign(outside.z,local.z)/float(outside_length));
    else { local_normal=make_float3(std::copysign(1.0F,local.x),0,0); double nearest=x;
        if (y>nearest) { nearest=y; local_normal=make_float3(0,std::copysign(1.0F,local.y),0); }
        if (z>nearest) local_normal=make_float3(0,0,std::copysign(1.0F,local.z)); }
    return {outside_length + std::min(std::max({x,y,z}),0.0),
        rotate_y(local_normal,box.yaw)};
}

int force_face(float3 force, float yaw) {
    const float3 f = rotate_y(force, -yaw); const std::array<float,3> m{std::abs(f.x),std::abs(f.y),std::abs(f.z)};
    const int axis = int(std::max_element(m.begin(), m.end()) - m.begin());
    const float component = axis == 0 ? f.x : axis == 1 ? f.y : f.z;
    return 2 * axis + (component < 0 ? 1 : 0);
}

double segment_distance2(float3 p, float3 a, float3 b) {
    const float3 ab = sub(b,a); const double d = dot(ab,ab), t = d > 0 ? std::clamp(dot(sub(p,a),ab)/d, 0.0, 1.0) : 0.0;
    const float3 delta = sub(p,add(a,mul(ab,float(t)))); return dot(delta,delta);
}

bool edge_region(float3 p, const waterlab::RectangleState& box) {
    p = rotate_y(sub(p,box.center), -box.yaw);
    const float x=-box.half_extents.x, y=box.half_extents.y, z=box.half_extents.z;
    const std::array<std::pair<float3,float3>,4> edges{{
        {{x,-y,-z},{x,-y,z}}, {{x,y,-z},{x,y,z}}, {{x,-y,-z},{x,y,-z}}, {{x,-y,z},{x,y,z}}}};
    double nearest = INFINITY; for (const auto& [a,b] : edges) nearest = std::min(nearest, segment_distance2(p,a,b));
    return nearest <= double(edge_radius) * edge_radius;
}

struct GridCell {
    int x{}, y{}, z{};
    bool operator==(const GridCell&) const = default;
};

struct GridCellHash {
    std::size_t operator()(const GridCell& c) const noexcept {
        std::size_t h=std::hash<int>{}(c.x);
        h^=std::hash<int>{}(c.y)+0x9e3779b9U+(h<<6U)+(h>>2U);
        h^=std::hash<int>{}(c.z)+0x9e3779b9U+(h<<6U)+(h>>2U);
        return h;
    }
};

GridCell grid_cell(float3 p,float size) {
    return {int(std::floor(p.x/size)),int(std::floor(p.y/size)),int(std::floor(p.z/size))};
}

struct ParticleEdgeMetrics {
    double support_fraction{NAN},maximum_particle_distance{},circulation_rms{},circulation_peak{};
    std::uint32_t circulating_particles{};
};

ParticleEdgeMetrics particle_edge_metrics(const std::vector<float3>& skin,
    const std::vector<float3>& particle_positions,const std::vector<float3>& particle_velocities,
    const waterlab::RectangleState& box,float support_radius) {
    ParticleEdgeMetrics result;
    std::unordered_map<GridCell,std::vector<std::uint32_t>,GridCellHash> grid;
    grid.reserve(particle_positions.size()/4U);
    for (std::uint32_t i=0;i<particle_positions.size();++i)
        grid[grid_cell(particle_positions[i],support_radius)].push_back(i);
    std::uint32_t edge_vertices=0,supported=0;
    for (float3 point:skin) {
        if (!edge_region(point,box)) continue;
        ++edge_vertices; double nearest2=INFINITY; const GridCell cell=grid_cell(point,support_radius);
        for (int z=-1;z<=1;++z) for (int y=-1;y<=1;++y) for (int x=-1;x<=1;++x) {
            const auto found=grid.find({cell.x+x,cell.y+y,cell.z+z});
            if (found==grid.end()) continue;
            for (std::uint32_t particle:found->second)
                nearest2=std::min(nearest2,dot(sub(point,particle_positions[particle]),
                    sub(point,particle_positions[particle])));
        }
        if (nearest2<=double(support_radius)*support_radius) ++supported;
        if (std::isfinite(nearest2)) result.maximum_particle_distance=std::max(
            result.maximum_particle_distance,std::sqrt(nearest2));
        else result.maximum_particle_distance=INFINITY;
    }
    if (edge_vertices) result.support_fraction=double(supported)/double(edge_vertices);

    const float x=-box.half_extents.x,y=box.half_extents.y,z=box.half_extents.z;
    const std::array<std::pair<float3,float3>,4> edges{{
        {{x,-y,-z},{x,-y,z}},{{x,y,-z},{x,y,z}},{{x,-y,-z},{x,y,-z}},{{x,-y,z},{x,y,z}}}};
    double circulation2=0;
    for (std::size_t i=0;i<particle_positions.size();++i) {
        const float3 local=rotate_y(sub(particle_positions[i],box.center),-box.yaw);
        std::size_t nearest_edge=0; double nearest2=INFINITY; float3 closest{};
        for (std::size_t e=0;e<edges.size();++e) {
            const float3 ab=sub(edges[e].second,edges[e].first); const double denom=dot(ab,ab);
            const float t=float(std::clamp(dot(sub(local,edges[e].first),ab)/denom,0.0,1.0));
            const float3 q=add(edges[e].first,mul(ab,t)); const double d2=dot(sub(local,q),sub(local,q));
            if (d2<nearest2) { nearest2=d2; nearest_edge=e; closest=q; }
        }
        if (nearest2>double(support_radius)*support_radius || nearest2<1e-12) continue;
        const float3 axis=mul(sub(edges[nearest_edge].second,edges[nearest_edge].first),
            1.0F/float(norm(sub(edges[nearest_edge].second,edges[nearest_edge].first))));
        float3 tangent=cross(axis,sub(local,closest)); tangent=mul(tangent,1.0F/float(norm(tangent)));
        const float3 relative=sub(particle_positions[i],box.center);
        const float3 angular{box.angular_velocity*relative.z,0,-box.angular_velocity*relative.x};
        const float3 local_velocity=rotate_y(sub(particle_velocities[i],add(box.velocity,angular)),-box.yaw);
        const double circulation=dot(local_velocity,tangent);
        circulation2+=circulation*circulation;
        result.circulation_peak=std::max(result.circulation_peak,std::abs(circulation));
        ++result.circulating_particles;
    }
    result.circulation_rms=result.circulating_particles
        ? std::sqrt(circulation2/double(result.circulating_particles)) : NAN;
    return result;
}

bool triangle_box(float3 a, float3 b, float3 c, float3 half) {
    const std::array<float3,3> vertices{a,b,c}, edges{sub(b,a),sub(c,b),sub(a,c)}, basis{{{1,0,0},{0,1,0},{0,0,1}}};
    std::array<float3,13> axes{basis[0],basis[1],basis[2],cross(edges[0],edges[1])};
    std::size_t next=4; for (float3 edge : edges) for (float3 axis : basis) axes[next++]=cross(edge,axis);
    for (float3 axis : axes) {
        const double length2=dot(axis,axis); if (length2 < 1e-20) continue;
        double low=dot(vertices[0],axis), high=low;
        for (std::size_t i=1;i<3;++i) { const double p=dot(vertices[i],axis); low=std::min(low,p); high=std::max(high,p); }
        const double radius=half.x*std::abs(axis.x)+half.y*std::abs(axis.y)+half.z*std::abs(axis.z), epsilon=1e-7*std::sqrt(length2);
        if (low > radius+epsilon || high < -radius-epsilon) return false;
    }
    return true;
}

std::uint32_t intersection_count(const std::vector<float3>& p, const std::vector<uint3>& triangles,
    const waterlab::RectangleState& box) {
    std::uint32_t count=0;
    for (uint3 t : triangles) if (triangle_box(rotate_y(sub(p[t.x],box.center),-box.yaw),
        rotate_y(sub(p[t.y],box.center),-box.yaw), rotate_y(sub(p[t.z],box.center),-box.yaw), box.half_extents)) ++count;
    return count;
}

double area(float3 a,float3 b,float3 c) { return 0.5*norm(cross(sub(b,a),sub(c,a))); }
double rms(double sum, std::uint64_t count) { return count ? std::sqrt(sum/double(count)) : NAN; }
double median(std::vector<double> values) { if (values.empty()) return NAN; std::sort(values.begin(),values.end()); return values[values.size()/2]; }
double percentile(std::vector<double> values, double fraction) {
    if (values.empty()) return NAN;
    std::sort(values.begin(),values.end());
    const double position=fraction*double(values.size()-1); const auto low=std::size_t(position);
    const auto high=std::min(low+1,values.size()-1); const double blend=position-double(low);
    return values[low]*(1.0-blend)+values[high]*blend;
}

double shape_error(const std::vector<float3>& p,const std::vector<float3>& rest) {
    if (rest.empty()) return 0;
    float3 cp{},cr{}; for (std::size_t i=0;i<p.size();++i) { cp=add(cp,p[i]); cr=add(cr,rest[i]); }
    cp=mul(cp,1.0F/float(p.size())); cr=mul(cr,1.0F/float(p.size())); double sum=0;
    for (std::size_t i=0;i<p.size();++i) { const float3 d=sub(sub(p[i],cp),sub(rest[i],cr)); sum+=dot(d,d); }
    return std::sqrt(sum/double(p.size()));
}

void write_frames(const std::filesystem::path& path,const std::vector<Frame>& frames) {
    std::ofstream out(path); if (!out) throw std::runtime_error("cannot write "+path.string());
    out << "frame,target_x,target_y,target_z,box_x,box_y,box_z,box_vx,box_vy,box_vz,box_yaw,box_angular_velocity,"
        "control_fx,control_fy,control_fz,control_work,cumulative_control_work,max_penetration,penetration_sum,penetrated_vertices,"
        "triangle_intersections,min_area_ratio,max_edge_stretch_ratio,contact_onsets,shell_skips,normal_switches,normal_90_switches,"
        "normal_180_switches,edge_vertices,particles_outside,finite_failures,bad_positions,bad_forces,shape_error,skin_speed_rms,"
        "corner_vibration_rms,corner_vibration_peak,tracked_vertices,tracked_vector_rms,tracked_vector_p99,tracked_vector_peak,"
        "contact_impulse,contact_pair_work,particle_owner_changes,contact_normal_angle_max_deg,contact_normal_angle45_count,"
        "contact_min_area_ratio,contact_max_edge_stretch_ratio,rest_orientation_reversals,particle_force_cap_hits,skin_force_cap_hits,"
        "box_force_cap_hits,edge_support_fraction,edge_max_particle_distance,particle_edge_circulation_rms,"
        "particle_edge_circulation_peak,circulating_particles,skin_physics_ms,physics_gpu_ms,diagnostic_ms\n" << std::setprecision(9);
    for (std::size_t i=0;i<frames.size();++i) { const Frame& f=frames[i];
        out << i<<','<<f.target.x<<','<<f.target.y<<','<<f.target.z<<','<<f.box.center.x<<','<<f.box.center.y<<','<<f.box.center.z<<','
            <<f.box.velocity.x<<','<<f.box.velocity.y<<','<<f.box.velocity.z<<','<<f.box.yaw<<','<<f.box.angular_velocity<<','
            <<f.control.x<<','<<f.control.y<<','<<f.control.z<<','<<f.work<<','<<f.cumulative_work<<','<<f.penetration<<','
            <<f.penetration_sum<<','<<f.penetrated<<','<<f.intersections<<','<<f.min_area<<','<<f.max_stretch<<','<<f.onsets<<','
            <<f.skips<<','<<f.switches<<','<<f.switches90<<','<<f.switches180<<','<<f.edge_vertices<<','<<f.outside<<','
            <<f.finite_failures<<','<<f.bad_positions<<','<<f.bad_forces<<','<<f.shape_error<<','<<f.speed_rms<<','<<f.vibration<<','
            <<f.vibration_peak<<','<<f.tracked_vertices<<','<<f.tracked_vector_rms<<','<<f.tracked_vector_p99<<','
            <<f.tracked_vector_peak<<','<<f.contact_impulse<<','<<f.contact_pair_work<<','<<f.owner_changes<<','
            <<f.normal_angle_max<<','<<f.normal_angle45<<','<<f.contact_min_area<<','<<f.contact_max_stretch<<','
            <<f.rest_orientation_reversals<<','<<f.particle_cap_hits<<','<<f.skin_cap_hits<<','<<f.box_cap_hits<<','
            <<f.edge_support_fraction<<','<<f.edge_max_particle_distance<<','<<f.particle_edge_circulation_rms<<','
            <<f.particle_edge_circulation_peak<<','<<f.circulating_particles<<','
            <<f.skin_ms<<','<<f.physics_ms<<','<<f.diagnostic_ms<<'\n';
    }
}

Result one_run(const Args& args,std::uint32_t run,const waterlab::HostSurfaceMesh& mesh,
    const std::vector<std::pair<std::uint32_t,std::uint32_t>>& edges,const std::vector<double>& rest_areas,
    const std::vector<double>& rest_lengths) {
    waterlab::HybridOptions options; options.physics_iterations=args.iterations;
    options.box_skin_damping=args.box_damping;
    options.particle_skin_distance=args.particle_skin_distance;
    options.particle_skin_damping=args.particle_skin_damping;
    options.collect_contact_diagnostics=true;
    if (args.stiff_material) { options.particle_repulsion=12.0F; options.skin_spring_stiffness=740.0F; }
    waterlab::HybridDroplet droplet(options); const std::size_t n=droplet.physical_skin_vertex_count();
    if (n != mesh.positions.size()) throw std::runtime_error("physical topology mismatch");
    const auto directory=args.output/(args.label+(args.stiff_material?"-stiff":""))/
        ("iterations-"+std::to_string(args.iterations))/("run-"+std::to_string(run));
    std::filesystem::create_directories(directory);
    std::ofstream raw_skin(directory/"skin.bin",std::ios::binary);
    if (!raw_skin) throw std::runtime_error("cannot write skin.bin");
    const std::array<std::uint32_t,4> header{
        0x4d504353U,2U,total_frames,std::uint32_t(n)};
    raw_skin.write(reinterpret_cast<const char*>(header.data()),sizeof(header));

    std::vector<float3> p(n), previous=mesh.positions, forces(n), impulses(n), settled_positions,
        particle_positions(options.particle_count),particle_velocities(options.particle_count);
    std::vector<float> pair_work(n);
    std::vector<std::uint32_t> owners(options.particle_count,UINT32_MAX),
        old_owners(options.particle_count,UINT32_MAX);
    std::vector<double> old_distance(n,INFINITY); std::vector<int> old_face(n,-1);
    std::vector<float3> old_normals(n), relative_velocities(std::size_t(total_frames)*n),
        contact_normals(std::size_t(total_frames)*n), position_history(std::size_t(total_frames)*n);
    std::vector<float> velocities(std::size_t(total_frames)*n);
    std::vector<std::uint8_t> membership(std::size_t(total_frames)*n), tracked(n);
    std::vector<Frame> frames; frames.reserve(total_frames); double accumulated_work=0;
    for (std::uint32_t frame=0;frame<total_frames;++frame) {
        Frame f; f.target=target_at(frame,args.pressed_x); const auto old_box=droplet.rectangle();
        f.control=controller(f.target,old_box,options); const auto timing=droplet.step(f.control); f.box=droplet.rectangle();
        const auto diagnostic_start=std::chrono::steady_clock::now();
        cuda_check(cudaMemcpy(p.data(),droplet.physical_skin_positions(),n*sizeof(float3),cudaMemcpyDeviceToHost),"skin positions");
        cuda_check(cudaMemcpy(forces.data(),droplet.skin_box_forces(),n*sizeof(float3),cudaMemcpyDeviceToHost),"box forces");
        cuda_check(cudaMemcpy(impulses.data(),droplet.skin_box_impulses(),n*sizeof(float3),cudaMemcpyDeviceToHost),"box impulses");
        cuda_check(cudaMemcpy(pair_work.data(),droplet.skin_box_pair_work(),n*sizeof(float),cudaMemcpyDeviceToHost),"box pair work");
        cuda_check(cudaMemcpy(owners.data(),droplet.particle_skin_owners(),owners.size()*sizeof(std::uint32_t),cudaMemcpyDeviceToHost),"particle skin owners");
        cuda_check(cudaMemcpy(particle_positions.data(),droplet.particle_positions(),
            particle_positions.size()*sizeof(float3),cudaMemcpyDeviceToHost),"particle positions");
        cuda_check(cudaMemcpy(particle_velocities.data(),droplet.particle_velocities(),
            particle_velocities.size()*sizeof(float3),cudaMemcpyDeviceToHost),"particle velocities");
        f.work=dot(f.control,mul(add(old_box.velocity,f.box.velocity),0.5F))*options.fixed_dt;
        accumulated_work+=f.work; f.cumulative_work=accumulated_work; f.skin_ms=timing.update_skin_physics_ms;
        f.physics_ms=timing.rebuild_fluid_hierarchy_ms+timing.rebuild_skin_hierarchy_ms+
            timing.update_fluid_physics_ms+timing.update_skin_physics_ms+
            timing.update_rectangle_physics_ms;
        double speed2=0;
        for (std::size_t v=0;v<n;++v) {
            const BoxSample box_point=box_sample(p[v],f.box); const double d=box_point.distance, depth=std::max(0.0,-d);
            f.penetration=std::max(f.penetration,depth);
            f.penetration_sum+=depth; f.penetrated+=depth>0;
            const double previous_distance=old_distance[v];
            f.onsets+=previous_distance>options.box_skin_contact_thickness && d<=options.box_skin_contact_thickness;
            f.skips+=previous_distance>options.box_skin_contact_thickness && d<0;
            if (previous_distance<=options.box_skin_contact_thickness &&
                d<=options.box_skin_contact_thickness && norm(old_normals[v])>0.5) {
                const double cosine=std::clamp(dot(old_normals[v],box_point.normal),-1.0,1.0);
                const double angle=std::acos(cosine)*180.0/3.14159265358979323846;
                f.normal_angle_max=std::max(f.normal_angle_max,angle);
                f.normal_angle45+=angle>=45.0;
            }
            old_distance[v]=d;
            old_normals[v]=box_point.normal;
            const bool valid_force=finite(forces[v]); const int face=valid_force && norm(forces[v])>1e-6 ? force_face(forces[v],old_box.yaw):-1;
            if (face>=0 && old_face[v]>=0 && face!=old_face[v]) { ++f.switches; face/2==old_face[v]/2 ? ++f.switches180:++f.switches90; }
            old_face[v]=face; const float3 velocity=mul(sub(p[v],previous[v]),1/options.fixed_dt);
            const float3 relative=sub(p[v],f.box.center), angular{f.box.angular_velocity*relative.z,0,-f.box.angular_velocity*relative.x};
            const float relative_speed=float(dot(sub(velocity,add(f.box.velocity,angular)),box_point.normal));
            const float3 relative_velocity=sub(velocity,add(f.box.velocity,angular));
            const std::size_t sample=std::size_t(frame)*n+v; velocities[sample]=relative_speed;
            relative_velocities[sample]=relative_velocity; contact_normals[sample]=box_point.normal;
            position_history[sample]=p[v];
            membership[sample]=std::uint8_t(edge_region(p[v],f.box)); f.edge_vertices+=membership[sample];
            if (frame>=settle && frame<withdraw && membership[sample]) tracked[v]=1U;
            speed2+=dot(velocity,velocity);
            f.bad_positions+=!finite(p[v]); f.bad_forces+=!valid_force;
            f.contact_impulse+=norm(impulses[v]); f.contact_pair_work+=pair_work[v];
        }
        for (std::size_t particle=0;particle<owners.size();++particle) {
            f.owner_changes+=owners[particle]!=UINT32_MAX && old_owners[particle]!=UINT32_MAX &&
                owners[particle]!=old_owners[particle];
        }
        old_owners=owners;
        raw_skin.write(reinterpret_cast<const char*>(p.data()),std::streamsize(n*sizeof(float3)));
        raw_skin.write(reinterpret_cast<const char*>(forces.data()),std::streamsize(n*sizeof(float3)));
        raw_skin.write(reinterpret_cast<const char*>(impulses.data()),std::streamsize(n*sizeof(float3)));
        f.speed_rms=std::sqrt(speed2/double(n)); f.intersections=intersection_count(p,mesh.triangles,f.box);
        for (std::size_t i=0;i<mesh.triangles.size();++i) { const uint3 t=mesh.triangles[i];
            f.min_area=std::min(f.min_area,area(p[t.x],p[t.y],p[t.z])/rest_areas[i]); }
        for (std::size_t i=0;i<edges.size();++i) f.max_stretch=std::max(f.max_stretch,
            norm(sub(p[edges[i].first],p[edges[i].second]))/rest_lengths[i]);
        const auto stats=droplet.statistics(); f.outside=stats.particles_outside; f.finite_failures=stats.finite_failures;
        f.particle_cap_hits=stats.particle_force_cap_hits;
        f.skin_cap_hits=stats.skin_force_cap_hits;
        f.box_cap_hits=stats.box_force_cap_hits;
        const ParticleEdgeMetrics particle_edge=particle_edge_metrics(
            p,particle_positions,particle_velocities,f.box,options.particle_support_radius);
        f.edge_support_fraction=particle_edge.support_fraction;
        f.edge_max_particle_distance=particle_edge.maximum_particle_distance;
        f.particle_edge_circulation_rms=particle_edge.circulation_rms;
        f.particle_edge_circulation_peak=particle_edge.circulation_peak;
        f.circulating_particles=particle_edge.circulating_particles;
        if (frame==settle-1) settled_positions=p;
        f.shape_error=shape_error(p,settled_positions);
        f.diagnostic_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-diagnostic_start).count();
        frames.push_back(f); previous=p;
    }

    const std::uint32_t tracked_count=static_cast<std::uint32_t>(
        std::count(tracked.begin(),tracked.end(),std::uint8_t{1}));
    std::vector<float3> highpass(relative_velocities.size());
    for (std::uint32_t frame=0;frame<total_frames;++frame) {
        std::vector<double> frame_speeds; frame_speeds.reserve(tracked_count);
        double frame_sum2=0;
        const std::uint32_t window_begin=frame>smooth_half_width?frame-smooth_half_width:0U;
        const std::uint32_t window_end=std::min(total_frames-1U,frame+smooth_half_width);
        for (std::size_t v=0;v<n;++v) {
            float3 mean{};
            for (std::uint32_t sample_frame=window_begin;
                 sample_frame<=window_end;++sample_frame) {
                mean=add(mean,relative_velocities[std::size_t(sample_frame)*n+v]);
            }
            mean=mul(mean,1.0F/float(window_end-window_begin+1U));
            const std::size_t sample=std::size_t(frame)*n+v;
            highpass[sample]=sub(relative_velocities[sample],mean);
            if (!tracked[v]) continue;
            const double speed=norm(highpass[sample]);
            frame_speeds.push_back(speed); frame_sum2+=speed*speed;
        }
        Frame& f=frames[frame]; f.tracked_vertices=tracked_count;
        f.tracked_vector_rms=rms(frame_sum2,frame_speeds.size());
        f.tracked_vector_p99=percentile(frame_speeds,0.99);
        f.tracked_vector_peak=frame_speeds.empty()?0:*std::max_element(frame_speeds.begin(),frame_speeds.end());
    }

    auto measure_phase=[&](std::uint32_t begin,std::uint32_t end,bool edge_scope) {
        PhaseMetrics metric; std::vector<double> speeds;
        std::vector<double> vertex_sum2(n); std::vector<std::uint32_t> vertex_samples(n);
        double sum2=0,tangent2=0; std::uint64_t samples=0;
        for (std::uint32_t frame=begin;frame<end;++frame) for (std::size_t v=0;v<n;++v) {
            if (edge_scope && !tracked[v]) continue;
            const std::size_t sample=std::size_t(frame)*n+v;
            const double speed=norm(highpass[sample]);
            const double normal_speed=dot(highpass[sample],contact_normals[sample]);
            const double tangential=std::max(0.0,speed*speed-normal_speed*normal_speed);
            speeds.push_back(speed); sum2+=speed*speed; tangent2+=tangential; ++samples;
            vertex_sum2[v]+=speed*speed; ++vertex_samples[v];
        }
        metric.vector_rms=rms(sum2,samples); metric.tangential_rms=rms(tangent2,samples);
        metric.vector_p99=percentile(speeds,0.99);
        metric.vector_peak=speeds.empty()?0:*std::max_element(speeds.begin(),speeds.end());
        for (std::size_t v=0;v<n;++v) if (vertex_samples[v]) metric.worst_vertex_rms=std::max(
            metric.worst_vertex_rms,std::sqrt(vertex_sum2[v]/double(vertex_samples[v])));
        return metric;
    };

    std::array<PhaseMetrics,phase_count> phase_metrics,full_phase_metrics;
    for (std::size_t phase=0;phase<phase_count;++phase) {
        phase_metrics[phase]=measure_phase(
            phase_ranges[phase].first,phase_ranges[phase].second,true);
        full_phase_metrics[phase]=measure_phase(
            phase_ranges[phase].first,phase_ranges[phase].second,false);
    }

    std::vector<std::uint8_t> patch_triangles(mesh.triangles.size()),patch_edges(edges.size());
    for (std::size_t i=0;i<mesh.triangles.size();++i) { const uint3 t=mesh.triangles[i];
        patch_triangles[i]=tracked[t.x]||tracked[t.y]||tracked[t.z]; }
    for (std::size_t i=0;i<edges.size();++i) patch_edges[i]=
        tracked[edges[i].first]||tracked[edges[i].second];
    std::vector<double> settled_areas(mesh.triangles.size()),settled_lengths(edges.size());
    for (std::size_t i=0;i<mesh.triangles.size();++i) { const uint3 t=mesh.triangles[i];
        settled_areas[i]=area(settled_positions[t.x],settled_positions[t.y],settled_positions[t.z]); }
    for (std::size_t i=0;i<edges.size();++i) settled_lengths[i]=
        norm(sub(settled_positions[edges[i].first],settled_positions[edges[i].second]));
    for (std::uint32_t frame=settle;frame<withdraw;++frame) {
        Frame& f=frames[frame]; float3 center{};
        const float3* frame_positions=position_history.data()+std::size_t(frame)*n;
        for (std::size_t v=0;v<n;++v) center=add(center,frame_positions[v]);
        center=mul(center,1.0F/float(n));
        for (std::size_t i=0;i<mesh.triangles.size();++i) if (patch_triangles[i]) {
            const uint3 t=mesh.triangles[i]; const float3 a=frame_positions[t.x],b=frame_positions[t.y],c=frame_positions[t.z];
            const float3 normal=cross(sub(b,a),sub(c,a)); const float3 centroid=mul(add(a,add(b,c)),1.0F/3.0F);
            const double ratio=0.5*norm(normal)/settled_areas[i];
            f.contact_min_area=std::min(f.contact_min_area,ratio);
            f.rest_orientation_reversals+=dot(normal,sub(centroid,center))<=0.0;
        }
        for (std::size_t i=0;i<edges.size();++i) if (patch_edges[i]) f.contact_max_stretch=std::max(
            f.contact_max_stretch,norm(sub(frame_positions[edges[i].first],frame_positions[edges[i].second]))/settled_lengths[i]);
    }

    double hold2=0,peak=0,first2=0,last2=0; std::uint64_t hold_n=0,first_n=0,last_n=0;
    for (std::uint32_t frame=smooth_half_width;frame+smooth_half_width<total_frames;++frame) {
        double frame2=0,frame_peak=0; std::uint64_t frame_n=0;
        for (std::size_t v=0;v<n;++v) { const std::size_t sample=std::size_t(frame)*n+v; if (!membership[sample]) continue;
            double mean=0; for (std::uint32_t f=frame-smooth_half_width;f<=frame+smooth_half_width;++f) mean+=velocities[std::size_t(f)*n+v];
            const double residual=velocities[sample]-mean/double(2*smooth_half_width+1); frame2+=residual*residual;
            frame_peak=std::max(frame_peak,std::abs(residual)); ++frame_n;
            if (frame>=slide && frame<edge_hold) { hold2+=residual*residual; peak=std::max(peak,std::abs(residual)); ++hold_n;
                if (frame<slide+60) { first2+=residual*residual; ++first_n; }
                if (frame>=edge_hold-60) { last2+=residual*residual; ++last_n; } }
        }
        frames[frame].vibration=rms(frame2,frame_n); frames[frame].vibration_peak=frame_peak;
    }

    Result result; result.vibration=rms(hold2,hold_n); result.vibration_peak=peak;
    result.tracked_vertices=tracked_count; result.phases=phase_metrics;
    result.full_phases=full_phase_metrics;
    const double first=rms(first2,first_n),last=rms(last2,last_n); result.hold_ratio=first>0?last/first:INFINITY;
    std::vector<double> times,physics_times; times.reserve(frames.size()); physics_times.reserve(frames.size());
    for (const Frame& f:frames) { result.penetration=std::max(result.penetration,f.penetration);
        result.penetration_integral+=f.penetration_sum*options.fixed_dt; result.intersection_frames+=f.intersections>0;
        result.max_intersections=std::max(result.max_intersections,f.intersections); result.min_area=std::min(result.min_area,f.min_area);
        result.max_stretch=std::max(result.max_stretch,f.max_stretch); result.onsets+=f.onsets; result.skips+=f.skips;
        result.switches+=f.switches; result.switches90+=f.switches90; result.switches180+=f.switches180;
        result.owner_changes+=f.owner_changes; result.normal_angle45+=f.normal_angle45;
        result.normal_angle_max=std::max(result.normal_angle_max,f.normal_angle_max);
        result.contact_impulse+=f.contact_impulse; result.contact_pair_work+=f.contact_pair_work;
        result.particle_cap_hits+=f.particle_cap_hits; result.skin_cap_hits+=f.skin_cap_hits;
        result.box_cap_hits+=f.box_cap_hits;
        result.outside=std::max(result.outside,f.outside); result.finite_failures=std::max(result.finite_failures,
            f.finite_failures+f.bad_positions+f.bad_forces); times.push_back(f.skin_ms); physics_times.push_back(f.physics_ms); }
    for (std::uint32_t frame=settle;frame<withdraw;++frame) {
        result.contact_min_area=std::min(result.contact_min_area,frames[frame].contact_min_area);
        result.contact_max_stretch=std::max(result.contact_max_stretch,frames[frame].contact_max_stretch);
        result.rest_orientation_reversals+=frames[frame].rest_orientation_reversals;
    }
    double circulation2=0.0; std::uint64_t circulation_samples=0;
    for (std::uint32_t frame=settle;frame<edge_hold;++frame) {
        const Frame& f=frames[frame];
        if (std::isfinite(f.edge_support_fraction)) result.minimum_edge_support_fraction=
            std::min(result.minimum_edge_support_fraction,f.edge_support_fraction);
        result.maximum_edge_particle_distance=std::max(
            result.maximum_edge_particle_distance,f.edge_max_particle_distance);
        if (std::isfinite(f.particle_edge_circulation_rms)) {
            circulation2+=f.particle_edge_circulation_rms*f.particle_edge_circulation_rms*
                double(f.circulating_particles);
            circulation_samples+=f.circulating_particles;
        }
        result.particle_edge_circulation_peak=std::max(
            result.particle_edge_circulation_peak,f.particle_edge_circulation_peak);
    }
    result.particle_edge_circulation_rms=rms(circulation2,circulation_samples);
    result.skin_ms=median(times); result.physics_ms=median(physics_times);
    std::uint32_t last_unstable=edge_hold-1U;
    for (std::uint32_t frame=edge_hold;frame<total_frames;++frame) if (
        frames[frame].shape_error>recovery_shape || frames[frame].speed_rms>recovery_speed) last_unstable=frame;
    if (last_unstable+1U<total_frames) result.recovery=double(last_unstable+1U-edge_hold)*options.fixed_dt;
    write_frames(directory/"frames.csv",frames);
    if (args.verbose) {
        std::cout<<std::setprecision(9)<<"CONTACT_EXPERIMENT_RUN,model="<<args.label<<",material="<<(args.stiff_material?"stiff":"default")
            <<",iterations="<<args.iterations<<",run="<<run
            <<",box_damping="<<args.box_damping<<",tracked_vertices="<<result.tracked_vertices
            <<",vibration_rms="<<result.vibration<<",vibration_peak="<<result.vibration_peak<<",hold_ratio="<<result.hold_ratio
            <<",max_penetration="<<result.penetration<<",integrated_penetration="<<result.penetration_integral
            <<",intersection_frames="<<result.intersection_frames<<",max_intersections="<<result.max_intersections
            <<",min_area_ratio="<<result.min_area<<",max_edge_stretch_ratio="<<result.max_stretch<<",contact_onsets="<<result.onsets
            <<",shell_skips="<<result.skips<<",normal_switches="<<result.switches<<",normal_90_switches="<<result.switches90
            <<",normal_180_switches="<<result.switches180<<",max_particles_outside="<<result.outside
            <<",contact_impulse="<<result.contact_impulse<<",contact_pair_work="<<result.contact_pair_work
            <<",particle_owner_changes="<<result.owner_changes<<",contact_normal_angle_max_deg="<<result.normal_angle_max
            <<",contact_normal_angle45_count="<<result.normal_angle45
            <<",contact_min_area_ratio="<<result.contact_min_area<<",contact_max_edge_stretch_ratio="<<result.contact_max_stretch
            <<",rest_orientation_reversal_samples="<<result.rest_orientation_reversals
            <<",particle_force_cap_hits="<<result.particle_cap_hits<<",skin_force_cap_hits="<<result.skin_cap_hits
            <<",box_force_cap_hits="<<result.box_cap_hits
            <<",minimum_edge_support_fraction="<<result.minimum_edge_support_fraction
            <<",maximum_edge_particle_distance="<<result.maximum_edge_particle_distance
            <<",particle_edge_circulation_rms="<<result.particle_edge_circulation_rms
            <<",particle_edge_circulation_peak="<<result.particle_edge_circulation_peak
            <<",finite_failures="<<result.finite_failures<<",recovery_seconds="<<result.recovery
            <<",median_skin_physics_ms="<<result.skin_ms<<",median_physics_gpu_ms="<<result.physics_ms<<'\n';
        for (std::size_t phase=0;phase<phase_count;++phase) {
            const PhaseMetrics& metric=result.phases[phase];
            std::cout<<"CONTACT_EXPERIMENT_PHASE,model="<<args.label<<",iterations="<<args.iterations
                <<",run="<<run<<",scope=edge,phase="<<phase_names[phase]
                <<",vector_rms="<<metric.vector_rms<<",vector_p99="<<metric.vector_p99
                <<",vector_peak="<<metric.vector_peak<<",worst_vertex_rms="<<metric.worst_vertex_rms
                <<",tangential_rms="<<metric.tangential_rms<<'\n';
            const PhaseMetrics& full=result.full_phases[phase];
            std::cout<<"CONTACT_EXPERIMENT_PHASE,model="<<args.label<<",iterations="<<args.iterations
                <<",run="<<run<<",scope=full,phase="<<phase_names[phase]
                <<",vector_rms="<<full.vector_rms<<",vector_p99="<<full.vector_p99
                <<",vector_peak="<<full.vector_peak<<",worst_vertex_rms="<<full.worst_vertex_rms
                <<",tangential_rms="<<full.tangential_rms<<'\n';
        }
    }
    return result;
}

double variation(const std::vector<double>& values) {
    const auto [low,high]=std::minmax_element(values.begin(),values.end());
    const double mean=std::accumulate(values.begin(),values.end(),0.0)/double(values.size());
    return std::abs(mean)>1e-20 ? (*high-*low)/std::abs(mean) : (*high==*low?0:INFINITY);
}

std::vector<std::pair<std::uint32_t,std::uint32_t>> make_edges(const std::vector<uint3>& triangles) {
    std::vector<std::pair<std::uint32_t,std::uint32_t>> edges; edges.reserve(3*triangles.size());
    for (uint3 t:triangles) for (auto [a,b]:std::array{std::pair{t.x,t.y},std::pair{t.y,t.z},std::pair{t.z,t.x}})
        edges.emplace_back(std::min(a,b),std::max(a,b));
    std::sort(edges.begin(),edges.end()); edges.erase(std::unique(edges.begin(),edges.end()),edges.end()); return edges;
}

int execute(const Args& args) {
    int device=0; cuda_check(cudaGetDevice(&device),"get CUDA device");
    cudaDeviceProp device_properties{};
    cuda_check(cudaGetDeviceProperties(&device_properties,device),"get CUDA device properties");
    const bool reference_timing_device=
        std::string_view(device_properties.name).find("RTX 3050 Ti")!=std::string_view::npos;
    const waterlab::HybridOptions defaults; const auto mesh=waterlab::make_geodesic_sphere(defaults.physical_skin_frequency,defaults.skin_radius);
    const auto edges=make_edges(mesh.triangles); std::vector<double> rest_areas,rest_lengths;
    for (uint3 t:mesh.triangles) rest_areas.push_back(area(mesh.positions[t.x],mesh.positions[t.y],mesh.positions[t.z]));
    for (auto [a,b]:edges) rest_lengths.push_back(norm(sub(mesh.positions[a],mesh.positions[b])));
    std::vector<Result> results; for (std::uint32_t run=0;run<args.runs;++run) results.push_back(one_run(args,run,mesh,edges,rest_areas,rest_lengths));
    std::vector<double> vibration,vibration_peak,hold_ratio,penetration,penetration_integral,intersection_frames,
        max_intersections,min_area,max_stretch,recovery,timing,physics_timing,skips,switches,outside,
        contact_impulse,contact_pair_work,contact_min_area,contact_max_stretch,inverted,
        owner_changes,normal_angle45,particle_caps,skin_caps,box_caps,tracked_vertices,
        minimum_edge_support,maximum_edge_particle_distance,particle_edge_circulation,
        particle_edge_circulation_peak;
    std::array<std::array<std::vector<double>,5>,phase_count> phase_values,full_phase_values;
    std::uint32_t failures=0; bool interaction=false;
    for (const Result& r:results) { vibration.push_back(r.vibration); vibration_peak.push_back(r.vibration_peak); hold_ratio.push_back(r.hold_ratio);
        penetration.push_back(r.penetration); penetration_integral.push_back(r.penetration_integral); intersection_frames.push_back(r.intersection_frames);
        max_intersections.push_back(r.max_intersections); recovery.push_back(r.recovery); skips.push_back(double(r.skips)); switches.push_back(double(r.switches));
        outside.push_back(double(r.outside));
        contact_impulse.push_back(r.contact_impulse); contact_pair_work.push_back(r.contact_pair_work);
        contact_min_area.push_back(r.contact_min_area); contact_max_stretch.push_back(r.contact_max_stretch);
        inverted.push_back(double(r.rest_orientation_reversals)); owner_changes.push_back(double(r.owner_changes));
        normal_angle45.push_back(double(r.normal_angle45)); particle_caps.push_back(double(r.particle_cap_hits));
        skin_caps.push_back(double(r.skin_cap_hits)); box_caps.push_back(double(r.box_cap_hits));
        minimum_edge_support.push_back(r.minimum_edge_support_fraction);
        maximum_edge_particle_distance.push_back(r.maximum_edge_particle_distance);
        particle_edge_circulation.push_back(r.particle_edge_circulation_rms);
        particle_edge_circulation_peak.push_back(r.particle_edge_circulation_peak);
        tracked_vertices.push_back(double(r.tracked_vertices));
        for (std::size_t phase=0;phase<phase_count;++phase) {
            phase_values[phase][0].push_back(r.phases[phase].vector_rms);
            phase_values[phase][1].push_back(r.phases[phase].vector_p99);
            phase_values[phase][2].push_back(r.phases[phase].vector_peak);
            phase_values[phase][3].push_back(r.phases[phase].worst_vertex_rms);
            phase_values[phase][4].push_back(r.phases[phase].tangential_rms);
            full_phase_values[phase][0].push_back(r.full_phases[phase].vector_rms);
            full_phase_values[phase][1].push_back(r.full_phases[phase].vector_p99);
            full_phase_values[phase][2].push_back(r.full_phases[phase].vector_peak);
            full_phase_values[phase][3].push_back(r.full_phases[phase].worst_vertex_rms);
            full_phase_values[phase][4].push_back(r.full_phases[phase].tangential_rms);
        }
        min_area.push_back(r.min_area); max_stretch.push_back(r.max_stretch); timing.push_back(r.skin_ms); physics_timing.push_back(r.physics_ms);
        failures=std::max(failures,r.finite_failures); interaction|=r.contact_impulse>0; }
    const double reproducibility=std::max({variation(vibration),variation(penetration),variation(intersection_frames),
        variation(contact_min_area),variation(contact_max_stretch),variation(phase_values[press_phase][1]),
        variation(phase_values[press_phase][3]),variation(phase_values[edge_hold_phase][1]),
        variation(phase_values[edge_hold_phase][3])});
    const double vibration_value=median(vibration), hold_value=median(hold_ratio), penetration_value=median(penetration),
        intersection_value=median(intersection_frames), area_value=median(min_area), stretch_value=median(max_stretch),
        recovery_value=median(recovery), timing_value=median(timing), outside_value=median(outside);
    const double press_rms=median(phase_values[press_phase][0]);
    const double contact_area=median(contact_min_area);
    const double contact_stretch=median(contact_max_stretch);
    const double timing_allowance=std::max(0.02,0.05*reference_skin_physics_ms);
    const bool timing_gate=!reference_timing_device ||
        timing_value<=reference_skin_physics_ms+timing_allowance;
    const bool shallow_stability_gate=
        vibration_value<=0.70*reference_vibration_rms &&
        press_rms<=0.70*reference_press_vector_rms && hold_value<=1.0 &&
        penetration_value<=1.05*reference_max_penetration &&
        intersection_value<=1.05*reference_intersection_frames &&
        contact_area>=0.95*reference_contact_min_area &&
        contact_stretch<=1.05*reference_contact_max_stretch &&
        recovery_value<=reference_recovery_seconds+1.0/60.0+1.0e-6 &&
        timing_gate;
    const bool fold_scenario=args.pressed_x<0.60F;
    const bool fold_stability_gate=
        median(phase_values[face_hold_phase][0])<=0.70*fold_reference_face_hold_rms &&
        median(phase_values[slide_phase][0])<=0.70*fold_reference_slide_rms &&
        penetration_value<=1.05*fold_reference_max_penetration &&
        contact_area>=0.95*fold_reference_contact_min_area &&
        contact_stretch<=1.05*fold_reference_contact_max_stretch &&
        recovery_value<=fold_reference_recovery_seconds+1.0/60.0+1.0e-6 &&
        timing_gate;
    const bool stability_gate=fold_scenario?fold_stability_gate:shallow_stability_gate;
    const bool validity_gate=failures==0&&outside_value==0&&
        median(particle_caps)==0&&median(skin_caps)==0&&median(box_caps)==0&&
        std::isfinite(recovery_value);
    const bool pass=interaction&&reproducibility<=0.05&&stability_gate&&validity_gate;
    std::cout<<std::setprecision(9)<<"CONTACT_EXPERIMENT_SUMMARY,model="<<args.label<<",material="<<(args.stiff_material?"stiff":"default")
        <<",iterations="<<args.iterations<<",runs="<<args.runs
        <<",box_damping="<<args.box_damping<<",pressed_x="<<args.pressed_x
        <<",scenario="<<(fold_scenario?"fold":"shallow")
        <<",particle_skin_distance="<<args.particle_skin_distance
        <<",particle_skin_damping="<<args.particle_skin_damping
        <<",tracked_vertices_median="<<median(tracked_vertices)
        <<",vibration_rms_median="<<vibration_value<<",vibration_peak_median="<<median(vibration_peak)<<",hold_ratio_median="<<hold_value
        <<",max_penetration_median="<<penetration_value<<",integrated_penetration_median="<<median(penetration_integral)
        <<",intersection_frames_median="<<intersection_value<<",max_intersections_median="<<median(max_intersections)
        <<",min_area_ratio_median="<<area_value
        <<",max_edge_stretch_ratio_median="<<stretch_value<<",skin_physics_ms_median="<<timing_value
        <<",physics_gpu_ms_median="<<median(physics_timing)
        <<",shell_skips_median="<<median(skips)<<",normal_switches_median="<<median(switches)<<",recovery_seconds_median="<<median(recovery)
        <<",contact_impulse_median="<<median(contact_impulse)<<",contact_pair_work_median="<<median(contact_pair_work)
        <<",particle_owner_changes_median="<<median(owner_changes)<<",contact_normal_angle45_median="<<median(normal_angle45)
        <<",contact_min_area_ratio_median="<<median(contact_min_area)
        <<",contact_max_edge_stretch_ratio_median="<<median(contact_max_stretch)
        <<",rest_orientation_reversal_samples_median="<<median(inverted)
        <<",particle_force_cap_hits_median="<<median(particle_caps)
        <<",skin_force_cap_hits_median="<<median(skin_caps)<<",box_force_cap_hits_median="<<median(box_caps)
        <<",minimum_edge_support_fraction_median="<<median(minimum_edge_support)
        <<",maximum_edge_particle_distance_median="<<median(maximum_edge_particle_distance)
        <<",particle_edge_circulation_rms_median="<<median(particle_edge_circulation)
        <<",particle_edge_circulation_peak_median="<<median(particle_edge_circulation_peak)
        <<",max_particles_outside="<<outside_value<<",primary_variation="<<reproducibility
        <<",edge_interaction="<<interaction<<",stability_gate="<<stability_gate
        <<",validity_gate="<<validity_gate<<",timing_gate="<<timing_gate
        <<",timing_reference_gpu="<<reference_timing_device<<",finite_failures="<<failures
        <<",status="<<(pass?"PASS":"FAIL")<<'\n';
    for (std::size_t phase=0;phase<phase_count;++phase) {
        std::cout<<"CONTACT_EXPERIMENT_PHASE_SUMMARY,model="<<args.label<<",iterations="<<args.iterations
            <<",scope=edge,phase="<<phase_names[phase]<<",vector_rms_median="<<median(phase_values[phase][0])
            <<",vector_p99_median="<<median(phase_values[phase][1])
            <<",vector_peak_median="<<median(phase_values[phase][2])
            <<",worst_vertex_rms_median="<<median(phase_values[phase][3])
            <<",tangential_rms_median="<<median(phase_values[phase][4])<<'\n';
        std::cout<<"CONTACT_EXPERIMENT_PHASE_SUMMARY,model="<<args.label<<",iterations="<<args.iterations
            <<",scope=full,phase="<<phase_names[phase]<<",vector_rms_median="<<median(full_phase_values[phase][0])
            <<",vector_p99_median="<<median(full_phase_values[phase][1])
            <<",vector_peak_median="<<median(full_phase_values[phase][2])
            <<",worst_vertex_rms_median="<<median(full_phase_values[phase][3])
            <<",tangential_rms_median="<<median(full_phase_values[phase][4])<<'\n';
    }
    return pass?0:2;
}

} // namespace

int main(int argc,char** argv) {
    int device_count=0; const cudaError_t device_status=cudaGetDeviceCount(&device_count);
    if (device_status!=cudaSuccess || device_count==0) {
        std::puts("SKIP parallel-mater-contact-experiment: no CUDA device");
        return 77;
    }
    try { Args args; if (!parse(argc,argv,args)) { std::fprintf(stderr,"usage: %s [--runs N] [--iterations 1..8] [--output DIR] "
        "[--material default|stiff] [--label NAME] [--box-damping D] [--pressed-x X] "
        "[--particle-skin-distance D] [--particle-skin-damping D] [--verbose]\n",argv[0]);
        return argc>1&&std::string_view(argv[1])=="--help"?0:1; } return execute(args); }
    catch (const std::exception& e) { std::fprintf(stderr,"contact experiment failed: %s\n",e.what()); return 1; }
}
