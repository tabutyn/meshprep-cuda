// SPDX-License-Identifier: MIT
#include "../apps/water_lab/hybrid_lab.hpp"
#include "../apps/water_lab/obstacle_course.hpp"
#include "../apps/water_lab/course_rotation.cuh"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>
namespace {
constexpr float particle_radius = 0.0225F;
constexpr float skin_radius = 0.006F;
constexpr float clearance_tolerance = 2.0e-4F;
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
float length(float3 v) { return std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z); }
float energy(float3 v) { return 0.5F * (v.x*v.x + v.y*v.y + v.z*v.z); }
bool finite(float3 v) { return std::isfinite(v.x)&&std::isfinite(v.y)&&std::isfinite(v.z); }
void require_close(float a,float e,float t,const char* message) { require(std::abs(a-e)<=t,message); }
void test_analytic_course_boundaries() {
    struct Case { float3 p; unsigned contact; float distance; float3 normal; };
    const std::array<Case,5> cases{{
        {make_float3(0,waterlab::course_floor_y+0.25F,0),0U,0.25F,make_float3(0,1,0)},
        {make_float3(-waterlab::course_half_width+0.4F,0,0),1U,0.4F,make_float3(1,0,0)},
        {make_float3(waterlab::course_half_width-0.4F,0,0),2U,0.4F,make_float3(-1,0,0)},
        {make_float3(0,0,waterlab::course_near_z-0.3F),3U,0.3F,make_float3(0,0,-1)},
        {make_float3(0,0,waterlab::course_far_z+0.3F),4U,0.3F,make_float3(0,0,1)}}};
    for (const auto& expected : cases) {
        const auto actual=waterlab::course_contact(expected.p,expected.contact);
        require_close(actual.distance,expected.distance,2.0e-6F,"analytic contact distance is wrong");
        require(length(make_float3(actual.normal.x-expected.normal.x,
            actual.normal.y-expected.normal.y,actual.normal.z-expected.normal.z))<2.0e-6F,
            "analytic contact normal is wrong");
    }
    require(waterlab::course_contact_count ==
            waterlab::course_boundary_contact_count + waterlab::course_peg_count,
        "analytic course does not expose all rigid post colliders");
    const float3 peg = waterlab::course_peg(0U);
    const auto side = waterlab::course_contact(make_float3(
        peg.x + waterlab::course_peg_radius + 0.2F,
        waterlab::course_floor_y + 0.5F * waterlab::course_peg_height, peg.z),
        waterlab::course_boundary_contact_count);
    require_close(side.distance, 0.2F, 2.0e-6F,
        "rigid post side distance is wrong");
    require_close(side.normal.x, 1.0F, 2.0e-6F,
        "rigid post side normal is wrong");
    const auto invalid = waterlab::course_contact({}, waterlab::course_contact_count);
    require(std::isinf(invalid.distance) && length(invalid.normal) == 0.0F,
        "out-of-range course contact was not inert");
}
void require_rotation(float3 r0, float3 r1, float3 r2,
    float3 input, float3 expected, const char* message) {
    const float3 actual = waterlab::rotate_course(waterlab::best_fit_rotation(r0,r1,r2), input);
    require(length(make_float3(actual.x-expected.x, actual.y-expected.y,
                actual.z-expected.z)) <= 2.0e-5F, message);
}
void test_best_fit_rotation() {
    const float3 x=make_float3(1,0,0), y=make_float3(0,1,0), z=make_float3(0,0,1);
    require_rotation(make_float3(0,-1,0),x,z, x,y, "90-degree Z rotation is wrong");
    require_rotation(x,make_float3(0,-1,0),make_float3(0,0,-1),
        y,make_float3(0,-1,0), "180-degree X rotation is wrong");
    require_rotation(make_float3(0,-2,0),x,make_float3(0,0,3),
        x,y, "anisotropic Z rotation is wrong");
    const float3 v=make_float3(0.2F,-0.3F,0.4F);
    require_rotation({}, {}, {}, v,v, "zero covariance fallback is not identity");
}
void test_course_preset() {
    const waterlab::HybridOptions lab;
    const waterlab::HybridOptions course=waterlab::course_options();
    require(!lab.obstacle_course && lab.physics_iterations==1U &&
            lab.gravity.x==0.0F && lab.gravity.y==0.0F && lab.gravity.z==0.0F,
        "course preset changed ordinary lab defaults");
    require(lab.maximum_particle_speed==3.0F && lab.maximum_skin_speed==2.0F,
        "ordinary lab speed caps changed");
    require(course.obstacle_course && course.physics_iterations==waterlab::course_minimum_iterations,
        "course preset did not select stable substeps");
    require(waterlab::HybridDroplet::maximum_physics_iterations == 16U,
        "interactive physics iteration ceiling is not 16");
    require(waterlab::course_peg_count == 8U,
        "authored rigid course no longer contains eight pegs");
    require_close(waterlab::course_motion_scale,4.0F,0.0F,"course did not start at 4x");
    require_close(course.gravity.y,-waterlab::course_gravity_magnitude,0.0F,
        "course gravity was not scaled");
    require_close(course.maximum_particle_speed,
        lab.maximum_particle_speed*waterlab::course_motion_scale,0.0F,"particle cap was not scaled");
    require_close(course.maximum_skin_speed,
        lab.maximum_skin_speed*waterlab::course_motion_scale,0.0F,"skin cap was not scaled");
    require_close(course.particle_repulsion,
        lab.particle_repulsion*waterlab::course_material_scale,0.0F,"particle repulsion was not scaled");
    require_close(course.skin_spring_stiffness,
        lab.skin_spring_stiffness*waterlab::course_material_scale,0.0F,"skin spring was not scaled");
    require_close(course.particle_skin_stiffness,2'000.0F*waterlab::course_material_scale,0.0F,
        "particle boundary stiffness was not scaled");
    require_close(course.maximum_particle_force,240.0F*waterlab::course_material_scale,0.0F,
        "particle force cap was not scaled");
    require_close(course.maximum_skin_force,
        lab.maximum_skin_force*waterlab::course_material_scale,0.0F,"skin force cap was not scaled");
}
void test_motion_controls() {
    auto original = waterlab::course_options();
    original.gravity = make_float3(0, -0.8F*waterlab::course_gravity_magnitude,
        -0.6F*waterlab::course_gravity_magnitude);
    for (int scale=1; scale<=8; ++scale) {
        const auto changed = waterlab::with_course_motion_multiplier(original, static_cast<float>(scale));
        require_close(waterlab::course_motion_multiplier(changed), static_cast<float>(scale), 0,
            "course multiplier did not change");
        require_close(length(changed.gravity), 1.8F*scale, 2.0e-6F, "acceleration did not scale");
        require_close(changed.gravity.z/changed.gravity.y, 0.75F, 1.0e-6F, "tilt direction changed");
        require_close(changed.maximum_particle_speed, 3.0F*scale, 0, "particle speed did not scale");
        require_close(changed.maximum_skin_speed, 2.0F*scale, 0, "skin speed did not scale");
        require(changed.physics_iterations==static_cast<unsigned>(std::max(4,scale)) &&
            changed.physics_iterations<=waterlab::HybridDroplet::maximum_physics_iterations,
            "course multiplier did not select bounded substeps");
        require(changed.fixed_dt==original.fixed_dt &&
            changed.fixed_dt*changed.maximum_particle_speed/changed.physics_iterations<=0.050001F,
            "course controls changed physical time or increased substep travel");
        const float material_ratio = static_cast<float>(scale) /
            waterlab::course_motion_scale;
        require_close(changed.skin_spring_stiffness,
            original.skin_spring_stiffness*material_ratio,1.0e-5F,
            "skin stiffness did not track gravity");
        require_close(changed.particle_repulsion,
            original.particle_repulsion*material_ratio,1.0e-5F,
            "particle repulsion did not track gravity");
        require_close(changed.particle_skin_stiffness,
            original.particle_skin_stiffness*material_ratio,1.0e-4F,
            "boundary stiffness did not track gravity");
        require_close(changed.maximum_particle_force,
            original.maximum_particle_force*material_ratio,1.0e-4F,
            "particle force cap did not track gravity");
        require_close(changed.maximum_skin_force,
            original.maximum_skin_force*material_ratio,1.0e-4F,
            "skin force cap did not track gravity");
    }
    const auto upper=waterlab::with_course_motion_multiplier(original,99.0F);
    const auto lower=waterlab::with_course_motion_multiplier(upper,-99.0F);
    require(waterlab::course_motion_multiplier(upper)==8 &&
        waterlab::course_motion_multiplier(lower)==1 && lower.physics_iterations==4,
        "motion clamp or downward substep adjustment failed");
    original.physics_iterations=8;
    require(waterlab::with_course_motion_multiplier(original,5).physics_iterations==8,
        "motion controls discarded manually selected extra substeps");
}

void test_runtime_motion_controls() {
    waterlab::HybridDroplet droplet(waterlab::course_options());
    for (const float scale : {8.0F, 1.0F, 4.0F}) {
        auto options=waterlab::with_course_motion_multiplier(droplet.options(), scale);
        droplet.set_runtime_options(options);
        for (unsigned frame=0; frame<60; ++frame) {
            const auto previous_frame=droplet.statistics().frame_index;
            (void)droplet.step();
            require(droplet.statistics().frame_index==previous_frame+1,
                "motion controls changed the one-tick-per-frame contract");
        }
        require(droplet.statistics().finite_failures==0,
            "bounded runtime motion change produced non-finite state");
        waterlab::HybridState saved;
        droplet.capture_state(saved);
        require_close(waterlab::course_motion_multiplier(saved.options), scale, 0,
            "capture lost motion multiplier");
        droplet.set_runtime_options(waterlab::with_course_motion_multiplier(
            droplet.options(), scale==8 ? 1 : 8));
        droplet.restore_state(saved);
        require_close(waterlab::course_motion_multiplier(droplet.options()), scale, 0,
            "replay failed to restore captured motion multiplier");
        droplet.reset();
        require_close(waterlab::course_motion_multiplier(droplet.options()), scale, 0,
            "reset lost motion multiplier");
    }
}
float3 check_projection(float3 position, float3 velocity, float radius, unsigned contact) {
    const float before = energy(velocity);
    waterlab::project_course_contact(position, velocity, radius);
    require(finite(position) && finite(velocity), "projection produced non-finite state");
    require(energy(velocity) <= before + 1.0e-6F,
        "zero-restitution projection added kinetic energy");
    require(waterlab::course_contact(position, contact).distance >=
            radius - clearance_tolerance,
        "projection left a point inside its obstacle");
    return position;
}
void test_projection() {
    check_projection(make_float3(0.0F, waterlab::course_floor_y - 0.01F, 0.0F),
        make_float3(0.3F, -1.0F, 0.2F), particle_radius, 0U);
    check_projection(make_float3(-waterlab::course_half_width - 0.01F, 0.0F, 0.0F),
        make_float3(-1.0F, 0.2F, 0.3F), particle_radius, 1U);
    check_projection(make_float3(waterlab::course_half_width + 0.01F, 0.0F, 0.0F),
        make_float3(1.0F, 0.2F, 0.3F), particle_radius, 2U);
    check_projection(make_float3(0.0F, 0.0F, waterlab::course_near_z + 0.01F),
        make_float3(0.1F, 0.2F, 1.0F), particle_radius, 3U);
    check_projection(make_float3(0.0F, 0.0F, waterlab::course_far_z - 0.01F),
        make_float3(0.1F, 0.2F, -1.0F), particle_radius, 4U);
    const float3 peg = waterlab::course_peg(0U);
    check_projection(make_float3(peg.x, waterlab::course_floor_y +
            0.5F * waterlab::course_peg_height, peg.z),
        make_float3(-1.0F, 0.0F, 0.0F), particle_radius,
        waterlab::course_boundary_contact_count);
    float3 seam = make_float3(
        peg.x, waterlab::course_floor_y + 0.01F, peg.z);
    float3 seam_velocity = make_float3(-1.0F, -1.0F, 0.0F);
    waterlab::project_course_contact(seam, seam_velocity, particle_radius);
    require(waterlab::course_contact(seam, 0U).distance >=
                particle_radius - clearance_tolerance &&
            waterlab::course_contact(seam,
                waterlab::course_boundary_contact_count).distance >=
                particle_radius - clearance_tolerance,
        "post/floor seam projection re-entered one rigid solid");
    constexpr float travel=0.1F, dt=1.0F/60.0F;
    check_projection(make_float3(0,waterlab::course_floor_y+particle_radius+0.01F-travel,0),
        make_float3(0,-travel/dt,0),particle_radius,0U);
    check_projection(make_float3(waterlab::course_half_width-particle_radius-0.01F+travel,
        0,0),make_float3(travel/dt,0,0),particle_radius,2U);
}

void test_ground_cloth_pit_projection() {
    float3 velocity = make_float3(0.0F, -1.0F, 0.0F);
    float3 inside = make_float3(waterlab::ground_pit_center.x,
        waterlab::course_floor_y - 0.20F, waterlab::ground_pit_center.y);
    waterlab::project_gallery_contact(
        inside, velocity, particle_radius, waterlab::GalleryArena::ground_box);
    require_close(inside.y, waterlab::course_floor_y - 0.20F, 1.0e-6F,
        "ground-box pit was still covered by the room floor");
    float3 outside = make_float3(
        waterlab::ground_pit_center.x + waterlab::ground_pit_half_extents.x + 0.2F,
        waterlab::course_floor_y - 0.20F, waterlab::ground_pit_center.y);
    velocity = make_float3(0.0F, -1.0F, 0.0F);
    waterlab::project_gallery_contact(
        outside, velocity, particle_radius, waterlab::GalleryArena::ground_box);
    require_close(outside.y, waterlab::course_floor_y + particle_radius, 1.0e-6F,
        "solid floor outside the context-7 pit stopped containing bodies");
    float3 bottom = make_float3(waterlab::ground_pit_center.x,
        waterlab::ground_pit_bottom_y - 0.20F, waterlab::ground_pit_center.y);
    velocity = make_float3(0.0F, -1.0F, 0.0F);
    waterlab::project_gallery_contact(
        bottom, velocity, particle_radius, waterlab::GalleryArena::ground_box);
    require_close(bottom.y, waterlab::ground_pit_bottom_y + particle_radius, 1.0e-6F,
        "context-7 pit has no finite bottom containment");
}
struct Audit {
    float particle_board{1.0e30F};
    float skin_board{1.0e30F};
    float particle_post{1.0e30F};
    float skin_post{1.0e30F};
    float initial_z{}, frame300_height{}; float3 final_center{};
    float maximum_particle_speed{}, maximum_skin_speed{}, maximum_edge_stretch{};
    float maximum_com_horizontal_speed{}, sum_com_horizontal_speed{};
    float minimum_triangle_area{1.0e30F};
    float minimum_relative_triangle_area{1.0e30F}, minimum_face_alignment{1.0F};
    unsigned maximum_particles_outside{}, maximum_finite_failures{};
    unsigned first_outside_frame{~0U};
    float3 first_outside_particle_center{}, first_outside_skin_center{};
    bool counts_valid{true}, positions_finite{true};
};
float3 centroid(const std::vector<float3>& points) {
    float3 center{};
    for (float3 p : points) { center.x += p.x; center.y += p.y; center.z += p.z; }
    const float inverse = 1.0F / static_cast<float>(points.size());
    return make_float3(center.x*inverse,center.y*inverse,center.z*inverse);
}
float3 droplet_center(const waterlab::HybridState& state, const waterlab::HybridOptions& options) {
    double x=0.0,y=0.0,z=0.0;
    for (float3 p : state.particle_positions) { x+=p.x*options.particle_mass;
        y+=p.y*options.particle_mass; z+=p.z*options.particle_mass; }
    for (float3 p : state.skin_positions) { x+=p.x*options.skin_vertex_mass;
        y+=p.y*options.skin_vertex_mass; z+=p.z*options.skin_vertex_mass; }
    const double mass=state.particle_positions.size()*options.particle_mass+
        state.skin_positions.size()*options.skin_vertex_mass;
    return make_float3(static_cast<float>(x/mass),static_cast<float>(y/mass),
        static_cast<float>(z/mass));
}
void audit_points(const std::vector<float3>&, float, float&, float&, bool&);
float droplet_velocity_z(const waterlab::HybridState& state,
    const waterlab::HybridOptions& options) {
    double momentum=0.0;
    for (float3 v : state.particle_velocities) momentum+=v.z*options.particle_mass;
    for (float3 v : state.skin_velocities) momentum+=v.z*options.skin_vertex_mass;
    const double mass=state.particle_velocities.size()*options.particle_mass+
        state.skin_velocities.size()*options.skin_vertex_mass;
    return static_cast<float>(momentum/mass);
}
float free_motion_velocity(waterlab::HybridOptions options) {
    waterlab::HybridDroplet droplet(options);
    waterlab::HybridState state; droplet.capture_state(state);
    float board=1.0e30F, post=1.0e30F; bool values_finite=true;
    audit_points(state.particle_positions,particle_radius,board,post,values_finite);
    audit_points(state.skin_positions,skin_radius,board,post,values_finite);
    require(values_finite && board>0.0F,
        "free-motion fixture begins in contact with the course");
    for (unsigned frame=0; frame<4U; ++frame) (void)droplet.step();
    droplet.capture_state(state);
    audit_points(state.particle_positions,particle_radius,board,post,values_finite);
    audit_points(state.skin_positions,skin_radius,board,post,values_finite);
    require(values_finite && board>0.0F,
        "free-motion fixture reached a course obstacle");
    return droplet_velocity_z(state,options);
}
void test_free_motion_ratio(waterlab::HybridOptions current) {
    current.gravity=make_float3(0,0,-0.65F*waterlab::course_motion_scale);
    auto baseline=current;
    baseline.gravity=make_float3(0,0,-0.65F);
    baseline.maximum_particle_speed/=waterlab::course_motion_scale;
    baseline.maximum_skin_speed/=waterlab::course_motion_scale;
    baseline.particle_repulsion/=waterlab::course_material_scale;
    baseline.skin_spring_stiffness/=waterlab::course_material_scale;
    baseline.particle_skin_stiffness/=waterlab::course_material_scale;
    baseline.maximum_particle_force/=waterlab::course_material_scale;
    baseline.maximum_skin_force/=waterlab::course_material_scale;
    const float old_speed=std::abs(free_motion_velocity(baseline));
    const float new_speed=std::abs(free_motion_velocity(current));
    const float ratio=new_speed/old_speed;
    std::printf("four-tick free COM velocity-z old=%.6f new=%.6f ratio=%.4f\n",
        old_speed,new_speed,ratio);
    require(std::isfinite(ratio) && ratio>=3.8F && ratio<=4.2F,
        "runtime acceleration did not scale free-motion COM velocity by 4x within 5%");
}
void audit_points(const std::vector<float3>& points, float radius,
    float& board_min, float& post_min, bool& all_finite) {
    for (float3 p : points) {
        if (!finite(p)) { all_finite = false; continue; }
        for (unsigned i = 0; i < waterlab::course_contact_count; ++i)
            board_min = std::min(board_min, waterlab::course_contact(p, i).distance-radius);
        for (unsigned i = waterlab::course_boundary_contact_count;
             i < waterlab::course_contact_count; ++i)
            post_min = std::min(post_min, waterlab::course_contact(p, i).distance-radius);
    }
}
float3 triangle_normal(float3 a, float3 b, float3 c) {
    const float3 u = make_float3(b.x-a.x, b.y-a.y, b.z-a.z);
    const float3 v = make_float3(c.x-a.x, c.y-a.y, c.z-a.z);
    return make_float3(u.y*v.z-u.z*v.y,u.z*v.x-u.x*v.z,u.x*v.y-u.y*v.x);
}
float audit_skin_geometry(const waterlab::HybridState& state,
    const waterlab::HostSurfaceMesh& rest, Audit& audit) {
    for (float3 v : state.skin_velocities) { audit.positions_finite&=finite(v);
        if (finite(v)) audit.maximum_skin_speed=std::max(audit.maximum_skin_speed,length(v)); }
    float frame_stretch = 0.0F;
    for (std::size_t v = 0; v+1U < rest.neighbor_offsets.size(); ++v)
        for (std::size_t i=rest.neighbor_offsets[v]; i<rest.neighbor_offsets[v+1U]; ++i) {
            const float3 a=state.skin_positions[v], b=state.skin_positions[rest.neighbors[i]];
            const float ratio=length(make_float3(b.x-a.x,b.y-a.y,b.z-a.z))/rest.rest_lengths[i];
            frame_stretch=std::max(frame_stretch,std::abs(ratio-1.0F));
        }
    audit.maximum_edge_stretch = std::max(audit.maximum_edge_stretch, frame_stretch);
    const float3 center=centroid(state.skin_positions), rest_center=centroid(rest.positions);
    float3 row0{},row1{},row2{};
    for (std::size_t i=0; i<rest.positions.size(); ++i) {
        const float3 c=make_float3(state.skin_positions[i].x-center.x,
            state.skin_positions[i].y-center.y,state.skin_positions[i].z-center.z);
        const float3 r=make_float3(rest.positions[i].x-rest_center.x,
            rest.positions[i].y-rest_center.y,rest.positions[i].z-rest_center.z);
        row0.x+=c.x*r.x; row0.y+=c.x*r.y; row0.z+=c.x*r.z;
        row1.x+=c.y*r.x; row1.y+=c.y*r.y; row1.z+=c.y*r.z;
        row2.x+=c.z*r.x; row2.y+=c.z*r.y; row2.z+=c.z*r.z;
    }
    const float4 rotation=waterlab::best_fit_rotation(row0,row1,row2);
    for (uint3 tri : rest.triangles) {
        const float3 current=triangle_normal(state.skin_positions[tri.x],
            state.skin_positions[tri.y],state.skin_positions[tri.z]);
        const float3 original=triangle_normal(rest.positions[tri.x],
            rest.positions[tri.y],rest.positions[tri.z]);
        const float current_length=length(current), original_length=length(original);
        const float relative_area=current_length/original_length;
        const float3 expected=waterlab::rotate_course(rotation,original);
        const float alignment=current_length>1.0e-20F
            ? (current.x*expected.x+current.y*expected.y+current.z*expected.z)/
                (current_length*original_length) : -1.0F;
        audit.minimum_triangle_area=std::min(audit.minimum_triangle_area,0.5F*current_length);
        audit.minimum_relative_triangle_area=std::min(
            audit.minimum_relative_triangle_area,relative_area);
        audit.minimum_face_alignment=std::min(audit.minimum_face_alignment,alignment);
    }
    return frame_stretch;
}
float median(std::vector<float> values) {
    const auto middle = values.begin() + static_cast<std::ptrdiff_t>(values.size()/2U);
    std::nth_element(values.begin(), middle, values.end()); return *middle;
}
bool same_motion(const waterlab::HybridState& a, const waterlab::HybridState& b) {
    const auto equal = [](const auto& x, const auto& y) {
        return x.size() == y.size() &&
            std::memcmp(x.data(), y.data(),
                x.size()*sizeof(typename std::decay_t<decltype(x)>::value_type)) == 0;
    };
    return equal(a.particle_positions, b.particle_positions) &&
        equal(a.particle_velocities, b.particle_velocities) &&
        equal(a.skin_positions, b.skin_positions) && equal(a.skin_velocities, b.skin_velocities);
}
void run_gpu(unsigned frames, unsigned iterations) {
    waterlab::HybridOptions options = waterlab::course_options();
    if (iterations != 0U) options.physics_iterations = iterations;
    options.obstacle_course = true;
    options.gravity = make_float3(0.0F, -waterlab::course_gravity_magnitude,
        -0.65F * waterlab::course_motion_scale);
    test_free_motion_ratio(options);
    waterlab::HybridDroplet droplet(options);
    auto unsafe=options; unsafe.physics_iterations=1U;
    bool rejected=false;
    try { droplet.set_runtime_options(unsafe); } catch (const std::invalid_argument&) { rejected=true; }
    require(rejected,"unsafe one-substep course configuration was accepted at the 12 m/s cap");
    const auto rest = waterlab::make_geodesic_sphere(
        options.physical_skin_frequency, options.skin_radius);
    const auto short_run = [&] {
        droplet.reset();
        for (unsigned i = 0; i < 24U; ++i) (void)droplet.step();
        waterlab::HybridState state; droplet.capture_state(state);
        return state;
    };
    const auto first=short_run(), second=short_run();
    require(same_motion(first,second),
        "reset course sequence is not bit-identical");
    droplet.reset();
    waterlab::HybridState state; droplet.capture_state(state);
    Audit audit;
    audit.initial_z = centroid(state.skin_positions).z;
    float3 previous_com=droplet_center(state,options);
    bool steering_released = false;
    std::array<std::vector<float>,8> timing;
    for (unsigned frame = 0; frame < frames; ++frame) {
        const auto t = droplet.step();
        const float stages[]{t.rebuild_fluid_hierarchy_ms,t.rebuild_skin_hierarchy_ms,
            t.update_fluid_physics_ms,t.update_skin_physics_ms,t.update_rectangle_physics_ms,
            t.update_surface_normals_ms,t.update_render_surface_ms,t.gpu_total_ms()};
        for (unsigned i=0; i<timing.size(); ++i) timing[i].push_back(stages[i]);
        droplet.capture_state(state);
        const float3 current_skin_center = centroid(state.skin_positions);
        if (!steering_released && current_skin_center.z < -8.0F) {
            auto level_options = droplet.options();
            level_options.gravity = make_float3(
                0.0F, -waterlab::course_gravity_magnitude, 0.0F);
            droplet.set_runtime_options(level_options);
            steering_released = true;
        }
        audit.counts_valid = audit.counts_valid && state.statistics.particle_count == 10'000U &&
            state.statistics.physical_skin_vertices == 1'002U;
        audit.maximum_particles_outside=std::max(audit.maximum_particles_outside,
            state.statistics.particles_outside);
        if (state.statistics.particles_outside!=0U && audit.first_outside_frame==~0U) {
            audit.first_outside_frame=frame+1U;
            audit.first_outside_particle_center=centroid(state.particle_positions);
            audit.first_outside_skin_center=centroid(state.skin_positions);
        }
        audit.maximum_finite_failures=std::max(audit.maximum_finite_failures,
            state.statistics.finite_failures);
        for (float3 v : state.particle_velocities) { audit.positions_finite&=finite(v);
            if (finite(v)) audit.maximum_particle_speed=std::max(
                audit.maximum_particle_speed,length(v)); }
        const float3 com=droplet_center(state,options);
        const float horizontal_speed=std::hypot(com.x-previous_com.x,com.z-previous_com.z)/options.fixed_dt;
        audit.maximum_com_horizontal_speed=std::max(audit.maximum_com_horizontal_speed,horizontal_speed);
        audit.sum_com_horizontal_speed+=horizontal_speed; previous_com=com;
        audit_points(state.particle_positions,particle_radius,audit.particle_board,
            audit.particle_post,
            audit.positions_finite);
        audit_points(state.skin_positions,skin_radius,audit.skin_board,
            audit.skin_post,
            audit.positions_finite);
        (void)audit_skin_geometry(state, rest, audit);
        if (frame == 299U) {
            float low = 1.0e30F, high = -1.0e30F;
            for (float3 p : state.skin_positions) { low = std::min(low, p.y); high = std::max(high, p.y); }
            audit.frame300_height = high - low;
        }
    }
    audit.final_center = centroid(state.skin_positions);
    std::printf("course frames=%u center (%.3f %.3f %.3f) frame300-height=%.3f "
        "outside-max=%u speed-max(particle %.3f skin %.3f) com-horizontal(mean %.3f max %.3f) "
        "edge-strain-max=%.6f "
        "tri-area-min=%.9f relative-area-min=%.6f face-alignment-min=%.6f\n",
        frames,audit.final_center.x,audit.final_center.y,audit.final_center.z,audit.frame300_height,
        audit.maximum_particles_outside,audit.maximum_particle_speed,audit.maximum_skin_speed,
        audit.sum_com_horizontal_speed/frames,audit.maximum_com_horizontal_speed,audit.maximum_edge_stretch,
        audit.minimum_triangle_area,audit.minimum_relative_triangle_area,audit.minimum_face_alignment);
    std::printf("steering-released=%u min-clearance "
        "particle(all %.6f post %.6f) skin(all %.6f post %.6f)\n",
        steering_released ? 1U : 0U,audit.particle_board,audit.particle_post,
        audit.skin_board,audit.skin_post);
    if (audit.first_outside_frame!=~0U)
        std::printf("first-outside-frame=%u particle-center=(%.3f %.3f %.3f) "
            "skin-center=(%.3f %.3f %.3f)\n",audit.first_outside_frame,
            audit.first_outside_particle_center.x,audit.first_outside_particle_center.y,
            audit.first_outside_particle_center.z,audit.first_outside_skin_center.x,
            audit.first_outside_skin_center.y,audit.first_outside_skin_center.z);
    std::printf("median ms hierarchy(fluid %.3f skin %.3f) physics(particle %.3f skin %.3f "
        "rectangle %.3f) normals %.3f render-surface %.3f total %.3f\n",
        median(timing[0]),median(timing[1]),median(timing[2]),median(timing[3]),median(timing[4]),
        median(timing[5]),median(timing[6]),median(timing[7]));
    require(audit.counts_valid, "course particle or skin count changed");
    require(audit.maximum_finite_failures==0U &&
        audit.positions_finite,"course physics became non-finite");
    require(audit.maximum_particles_outside == 0U, "course boundary reported escaped particles");
    require(audit.maximum_particle_speed<=options.maximum_particle_speed+1.0e-4F &&
            audit.maximum_skin_speed<=options.maximum_skin_speed+1.0e-4F,
        "course particle or skin speed cap was exceeded");
    require(audit.particle_board>=-clearance_tolerance &&
            audit.skin_board>=-clearance_tolerance,"course body penetrated the board or rails");
    require(audit.final_center.z < audit.initial_z - 0.5F, "droplet made no downhill progress");
    require(std::isfinite(audit.frame300_height) && audit.frame300_height>=0.75F &&
            audit.frame300_height<3.0F,"closed skin collapsed or expanded unreasonably by frame 300");
    require(audit.particle_post < 0.01F || audit.skin_post < 0.01F,
        "script never reached a rigid course post");
    require(audit.maximum_edge_stretch<=1.5F,"course skin edge strain exceeded the measured regime");
    require(audit.minimum_triangle_area > 0.0F, "course skin contains a degenerate triangle");
    require(audit.minimum_relative_triangle_area>=0.1F,"course skin triangle collapsed below 10% area");
    require(audit.minimum_face_alignment>0.0F,"course skin face reversed relative to best-fit rotation");
}
} // namespace
int main(int argc, char** argv) {
    try {
        unsigned frames=600U, iterations=0U; bool host_only=false;
        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "--host-only") == 0) host_only = true;
            else if (std::strcmp(argv[i], "--frames") == 0 && i+1 < argc)
                frames = static_cast<unsigned>(std::strtoul(argv[++i], nullptr, 10));
            else if (std::strcmp(argv[i], "--iterations") == 0 && i+1 < argc)
                iterations = static_cast<unsigned>(std::strtoul(argv[++i], nullptr, 10));
            else throw std::invalid_argument(
                "usage: parallel-mater-course-tests [--host-only] [--frames N] [--iterations N]");
        }
        test_analytic_course_boundaries(); test_best_fit_rotation(); test_course_preset(); test_motion_controls(); test_projection(); test_ground_cloth_pit_projection();
        if (host_only) { std::puts("PASS parallel-mater-course-tests host"); return 0; }
        int devices = 0;
        if (cudaGetDeviceCount(&devices)!=cudaSuccess || devices==0)
            { std::puts("SKIP parallel-mater-course-tests GPU: no CUDA device"); return 77; }
        test_runtime_motion_controls();
        require(frames >= 300U, "GPU course test requires at least 300 frames");
        run_gpu(frames, iterations);
        std::puts("PASS parallel-mater-course-tests"); return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr,"FAIL parallel-mater-course-tests: %s\n",error.what()); return 1;
    }
}
