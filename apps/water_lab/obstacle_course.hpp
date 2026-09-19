// SPDX-License-Identifier: MIT
#pragma once

#include <vector_functions.h>
#include <cmath>
#include <cstdint>

namespace waterlab {

enum class GalleryArena : unsigned {
    none,
    course,
    bowl,
    slope,
    ground,
    enclosed_box,
    water_wheel,
    cloth_basin,
    ground_box,
    low_ceiling_box,
    rope_post,
    rope_bridge,
};

struct CourseContact {
    float distance;
    float3 normal;
};

inline constexpr float3 bowl_center{0.0F, 0.15F, -1.8F};
inline constexpr float bowl_inner_radius = 1.65F;
inline constexpr float bowl_wall_thickness = 0.08F;
inline constexpr std::uint32_t bowl_peg_count = 4U;
inline constexpr float bowl_peg_radius = 0.12F;
inline constexpr float bowl_peg_height = 0.55F;
__host__ __device__ inline float3 bowl_peg(std::uint32_t index)
{
    // Four vertical capped cylinders are seated on the inner hemisphere and
    // moved inward so the heavy sphere and fluid must actually negotiate them.
    constexpr float radial = 0.62F;
    const float base_y = bowl_center.y -
        sqrtf(bowl_inner_radius * bowl_inner_radius - radial * radial);
    const float y = base_y + 0.5F * bowl_peg_height;
    switch (index & 3U) {
    case 0U: return make_float3( radial, y, bowl_center.z);
    case 1U: return make_float3(-radial, y, bowl_center.z);
    case 2U: return make_float3(0.0F, y, bowl_center.z + radial);
    default: return make_float3(0.0F, y, bowl_center.z - radial);
    }
}

__host__ __device__ inline CourseContact bowl_peg_contact(
    float3 p, std::uint32_t peg_index)
{
    const float3 center = bowl_peg(peg_index);
    const float dx = p.x - center.x;
    const float dz = p.z - center.z;
    const float radial_length = sqrtf(dx * dx + dz * dz);
    const float radial_distance = radial_length - bowl_peg_radius;
    const float cap_distance = fabsf(p.y - center.y) - 0.5F * bowl_peg_height;
    const float outside_radial = fmaxf(radial_distance, 0.0F);
    const float outside_cap = fmaxf(cap_distance, 0.0F);
    const float distance = sqrtf(outside_radial * outside_radial +
        outside_cap * outside_cap) +
        fminf(fmaxf(radial_distance, cap_distance), 0.0F);
    const float3 radial = radial_length > 1.0e-8F
        ? make_float3(dx / radial_length, 0.0F, dz / radial_length)
        : make_float3(1.0F, 0.0F, 0.0F);
    float3 normal{};
    if (outside_radial > 0.0F && outside_cap > 0.0F) {
        const float inverse = 1.0F / sqrtf(
            outside_radial * outside_radial + outside_cap * outside_cap);
        const float cap_sign = p.y >= center.y ? 1.0F : -1.0F;
        normal = make_float3(radial.x * outside_radial * inverse,
            cap_sign * outside_cap * inverse,
            radial.z * outside_radial * inverse);
    } else if (radial_distance > cap_distance) {
        normal = radial;
    } else {
        normal = make_float3(0.0F, p.y >= center.y ? 1.0F : -1.0F, 0.0F);
    }
    return {distance, normal};
}
inline constexpr float slope_start_y = -0.85F;
inline constexpr float slope_gradient = -0.17632698F; // tan(10 degrees)
inline constexpr float3 gallery_box_center{0.0F, 0.575F, -1.2F};
inline constexpr float3 gallery_box_half_extents{2.8F, 1.625F, 3.2F};
inline constexpr float hanging_ceiling_y = -0.15F;
inline constexpr float3 low_gallery_box_center{
    gallery_box_center.x,
    0.5F * ((gallery_box_center.y - gallery_box_half_extents.y) + hanging_ceiling_y),
    gallery_box_center.z};
inline constexpr float3 low_gallery_box_half_extents{
    gallery_box_half_extents.x,
    0.5F * (hanging_ceiling_y -
        (gallery_box_center.y - gallery_box_half_extents.y)),
    gallery_box_half_extents.z};
inline constexpr float3 rope_post_center{0.0F, -0.30F, -1.2F};
inline constexpr float rope_post_radius = 0.16F;
inline constexpr float rope_post_height = 1.50F;
inline constexpr float3 rope_anchor{
    rope_post_center.x + rope_post_radius + 0.035F,
    rope_post_center.y + 0.5F * rope_post_height - 0.10F,
    rope_post_center.z};

__host__ __device__ inline CourseContact rope_post_contact(float3 p)
{
    const float dx = p.x - rope_post_center.x;
    const float dz = p.z - rope_post_center.z;
    const float radial_length = sqrtf(dx * dx + dz * dz);
    const float radial_distance = radial_length - rope_post_radius;
    const float cap_distance = fabsf(p.y - rope_post_center.y) -
        0.5F * rope_post_height;
    const float outside_radial = fmaxf(radial_distance, 0.0F);
    const float outside_cap = fmaxf(cap_distance, 0.0F);
    const float distance = sqrtf(outside_radial * outside_radial +
        outside_cap * outside_cap) +
        fminf(fmaxf(radial_distance, cap_distance), 0.0F);
    const float3 radial = radial_length > 1.0e-8F
        ? make_float3(dx / radial_length, 0.0F, dz / radial_length)
        : make_float3(1.0F, 0.0F, 0.0F);
    float3 normal{};
    if (outside_radial > 0.0F && outside_cap > 0.0F) {
        const float inverse = 1.0F / sqrtf(
            outside_radial * outside_radial + outside_cap * outside_cap);
        const float sign = p.y >= rope_post_center.y ? 1.0F : -1.0F;
        normal = make_float3(radial.x * outside_radial * inverse,
            sign * outside_cap * inverse, radial.z * outside_radial * inverse);
    } else if (radial_distance > cap_distance) {
        normal = radial;
    } else {
        normal = make_float3(0.0F,
            p.y >= rope_post_center.y ? 1.0F : -1.0F, 0.0F);
    }
    return {distance, normal};
}
// The context-7 ground cloth spans this opening. Its pinned border rests on
// the room floor while the free interior can sag into the finite pit.
inline constexpr float2 ground_pit_center{0.0F, -0.55F};
inline constexpr float2 ground_pit_half_extents{1.30F, 1.30F};
inline constexpr float ground_pit_bottom_y = -2.30F;

__host__ __device__ inline bool inside_ground_pit(float3 p, float inset = 0.0F)
{
    return fabsf(p.x - ground_pit_center.x) < ground_pit_half_extents.x - inset &&
        fabsf(p.z - ground_pit_center.y) < ground_pit_half_extents.y - inset;
}
// Context 5 uses a doubled-spacing 3x3 cloth grid and surrounds it with a
// separate rigid curb. The cloth supplies a flexible floor; these four walls
// provide containment without becoming visible geometry.
inline constexpr float3 cloth_basin_center{0.0F, 0.05F, -1.2F};
inline constexpr float2 cloth_basin_inner_half_extents{
    0.5F * 27.0F * 0.13F, 0.5F * 21.0F * 0.13F};
inline constexpr float cloth_basin_wall_bottom = -1.05F;
inline constexpr float cloth_basin_wall_top = 2.20F;
inline constexpr float cloth_basin_wall_thickness = 0.08F;
inline constexpr std::uint32_t cloth_snake_wall_count = 2U;
inline constexpr float cloth_snake_wall_half_thickness = 0.045F;
inline constexpr float cloth_snake_gap_width =
    2.0F * cloth_basin_inner_half_extents.x / 3.0F;
inline constexpr float cloth_snake_wall_top = 0.30F;
inline constexpr float cloth_snake_collision_top = 2.20F;
inline constexpr float3 cloth_goal_center{
    cloth_basin_center.x + 1.38F, 0.55F, cloth_basin_center.z - 1.00F};
inline constexpr float3 cloth_goal_half_extents{0.32F, 0.48F, 0.32F};
__host__ __device__ inline float cloth_snake_wall_z(std::uint32_t wall)
{
    const float panel_depth = 2.0F * cloth_basin_inner_half_extents.y / 3.0F;
    return cloth_basin_center.z + 0.5F * panel_depth -
        panel_depth * static_cast<float>(wall);
}
inline constexpr float water_wheel_ground_y = -1.05F;
// The inlet meets the wheel at its leftmost point, exactly 2.5 m above the
// lower collector. The two ramps are separate finite planes: the upper ramp
// feeds the wheel and the lower ramp carries discharged water away.
inline constexpr float3 water_wheel_center{0.50F, 1.45F, 0.0F};
inline constexpr float water_wheel_radius = 1.35F;
inline constexpr float water_wheel_hub_radius = 0.24F;
inline constexpr float water_wheel_shell_radius = 1.85F;
inline constexpr float water_wheel_fin_outer_radius = 1.80F;
inline constexpr float water_wheel_fin_thickness = 0.060F;
inline constexpr float water_wheel_fin_barrier_distance = 0.080F;
inline constexpr float water_wheel_half_depth = 0.52F;
inline constexpr float water_wheel_ground_half_depth = 0.58F;
inline constexpr float water_wheel_cross_offset = 1.35F;
inline constexpr float water_wheel_axle_half_length = 1.70F;
inline constexpr float water_wheel_outer_disk_offset = water_wheel_cross_offset;
inline constexpr float water_wheel_outer_disk_half_thickness = 0.30F;
inline constexpr float water_wheel_outer_disk_radius = 0.675F;
inline constexpr float water_wheel_stage_z =
    water_wheel_center.z + water_wheel_cross_offset;
inline constexpr float water_wheel_top_platform_y =
    water_wheel_center.y + water_wheel_outer_disk_radius - 0.10F;
inline constexpr float water_wheel_top_platform_gap_half_width = 0.34F;
inline constexpr float water_wheel_top_platform_outer_x = 3.10F;
inline constexpr float water_wheel_top_platform_half_depth = 0.30F;
inline constexpr float water_wheel_top_bumper_thickness = 0.055F;
inline constexpr float water_wheel_top_bumper_height = 0.16F;
inline constexpr std::uint32_t water_wheel_fin_count = 8U;
inline constexpr float water_wheel_inlet_start_x = -5.20F;
inline constexpr float water_wheel_entry_x =
    water_wheel_center.x - water_wheel_shell_radius;
inline constexpr float water_wheel_exit_x = water_wheel_center.x;
inline constexpr float water_wheel_collector_start_x = water_wheel_exit_x - 0.85F;
inline constexpr float water_wheel_collector_end_x = 4.60F;
inline constexpr float water_wheel_ramp_gradient = -0.17632698F; // tan(-10 degrees)

__host__ __device__ inline bool water_wheel_back_face_contact(
    float local_tangent, float relative_back_normal_speed, float particle_radius)
{
    return local_tangent <= 0.0F &&
        local_tangent > -(water_wheel_fin_thickness + particle_radius +
            water_wheel_fin_barrier_distance) &&
        relative_back_normal_speed < 0.0F;
}

__host__ __device__ inline bool water_wheel_back_face_overlap(
    float local_tangent, float particle_radius)
{
    return local_tangent <= 0.0F &&
        local_tangent > -(water_wheel_fin_thickness + particle_radius +
            water_wheel_fin_barrier_distance);
}

// Inelastic one-way fin response. The speculative shell may cancel only the
// velocity already approaching the trailing face; it can never create extra
// separating speed. A true overlap receives full cancellation, while a nearby
// particle receives a smooth fraction. Geometric overlap is resolved in
// position separately and therefore cannot inject correction/dt energy.
__host__ __device__ inline float water_wheel_fin_response_delta(
    float local_tangent, float relative_back_normal_speed, float particle_radius)
{
    if (!water_wheel_back_face_contact(
            local_tangent, relative_back_normal_speed, particle_radius)) return 0.0F;
    const float separation = -(local_tangent + water_wheel_fin_thickness);
    const float hard_depth = fmaxf(0.0F, particle_radius - separation);
    if (hard_depth > 0.0F) return -relative_back_normal_speed;
    const float shell_depth = fmaxf(0.0F,
        particle_radius + water_wheel_fin_barrier_distance - separation);
    const float activation = fminf(1.0F,
        shell_depth / water_wheel_fin_barrier_distance);
    return activation * -relative_back_normal_speed;
}

struct WaterWheelState {
    float angle{};
    float angular_velocity{};
    float inertia{180.0F};
    float rim_angle{};
    float rim_angular_velocity{};
    float rim_inertia{260.0F};
    float rim_drag{0.55F};
    float cross_reaction_torque{};
};

__host__ __device__ inline float water_wheel_inlet_height(float x)
{
    return water_wheel_center.y +
        water_wheel_ramp_gradient * (x - water_wheel_entry_x);
}

__host__ __device__ inline float water_wheel_collector_height(float x)
{
    return water_wheel_ground_y +
        water_wheel_ramp_gradient * (x - water_wheel_exit_x);
}

__host__ __device__ inline bool water_wheel_support_height(float x, float& height)
{
    if (x >= water_wheel_inlet_start_x && x <= water_wheel_entry_x) {
        height = water_wheel_inlet_height(x);
        return true;
    }
    if (x >= water_wheel_collector_start_x &&
        x <= water_wheel_collector_end_x) {
        height = water_wheel_collector_height(x);
        return true;
    }
    return false;
}

__host__ __device__ inline bool water_wheel_top_platform_contact(
    float3 p, float radius)
{
    const float local_x = fabsf(p.x - water_wheel_center.x);
    return local_x >= water_wheel_top_platform_gap_half_width &&
        local_x <= water_wheel_top_platform_outer_x &&
        fabsf(p.z - water_wheel_stage_z) <=
            water_wheel_top_platform_half_depth &&
        p.y >= water_wheel_top_platform_y - 2.0F * radius &&
        p.y < water_wheel_top_platform_y + radius;
}

// Context 9: ten 0.3 m tile rows with 0.2 m rope gaps produce a 4.8 m span.
inline constexpr float rope_bridge_tile_size = 0.30F;
inline constexpr float rope_bridge_gap = 0.20F;
inline constexpr std::uint32_t rope_bridge_columns = 4U;
inline constexpr std::uint32_t rope_bridge_rows = 10U;
inline constexpr float rope_bridge_pitch = rope_bridge_tile_size + rope_bridge_gap;
inline constexpr float rope_bridge_width =
    rope_bridge_columns * rope_bridge_tile_size +
    (rope_bridge_columns - 1U) * rope_bridge_gap;
inline constexpr float rope_bridge_length =
    rope_bridge_rows * rope_bridge_tile_size +
    (rope_bridge_rows - 1U) * rope_bridge_gap;
inline constexpr float rope_bridge_deck_y = -0.30F;
inline constexpr float rope_bridge_land_y = rope_bridge_deck_y - 0.02F;
inline constexpr float rope_bridge_land_inner_z = 0.5F * rope_bridge_length + 0.10F;
inline constexpr float rope_bridge_land_outer_z = rope_bridge_land_inner_z + 1.50F;
inline constexpr float rope_bridge_land_half_width = 1.70F;

// Water arrives at nine o'clock and follows gravity through the lower-left
// quadrant to a vertical six-o'clock outlet. This arc is the containing side
// of that path; putting it on the lower-right leaves the incoming water on the
// unbounded side of the wheel.
__host__ __device__ inline bool water_wheel_shell_present(float3 relative)
{
    // The housing follows the loaded inlet-to-outlet path on the lower-left.
    // Its upper endpoint reaches the horizontal high ground at axle height;
    // the lower-right remains open so discharged water cannot be trapped.
    // Extend the circular guide above the axle so its authored beginning is
    // visible above the high inlet ground rather than terminating at 9 o'clock.
    const bool lower_left = relative.x <= 0.0F &&
        relative.y <= 0.65F * water_wheel_shell_radius;
    const bool horizontal_inlet =
        relative.x < -0.82F * water_wheel_shell_radius &&
        fabsf(relative.y) < 0.24F * water_wheel_shell_radius;
    const bool vertical_outlet =
        relative.y < -0.82F * water_wheel_shell_radius &&
        fabsf(relative.x) < 0.24F * water_wheel_shell_radius;
    return lower_left && !horizontal_inlet && !vertical_outlet;
}

__host__ __device__ inline bool water_wheel_shell_opening(float3 relative)
{
    return !water_wheel_shell_present(relative);
}

// One immutable layout shared by collision, rendering, and the driving test.
inline constexpr float course_floor_y = -1.05F;
inline constexpr float course_half_width = 3.4F;
inline constexpr float course_near_z = 1.8F;
inline constexpr float course_far_z = -10.8F;
inline constexpr float course_floor_thickness = 0.18F;
inline constexpr float course_rail_thickness = 0.14F;
inline constexpr float course_rail_height = 0.38F;
inline constexpr float course_goal_z = -9.4F;
inline constexpr float course_goal_radius = 0.85F;
inline constexpr float course_peg_radius = 0.30F;
inline constexpr float course_peg_height = 1.65F;
inline constexpr unsigned course_peg_count = 8U;

__host__ __device__ inline float3 course_peg(unsigned index)
{
    // Gaps are wider than the settled droplet; alternating center pegs ask for
    // steering without forcing the water through an impossible aperture.
    switch (index) {
    case 0: return make_float3(-1.25F, course_floor_y, -2.0F);
    case 1: return make_float3( 1.25F, course_floor_y, -2.0F);
    case 2: return make_float3( 0.0F, course_floor_y, -4.0F);
    case 3: return make_float3(-2.5F, course_floor_y, -4.5F);
    case 4: return make_float3( 2.5F, course_floor_y, -4.5F);
    case 5: return make_float3(-1.3F, course_floor_y, -6.6F);
    case 6: return make_float3( 1.3F, course_floor_y, -6.6F);
    default:return make_float3(-1.7F, course_floor_y, -8.5F);
    }
}

// Context 1 uses one immutable analytic contact set: floor, full-height
// containment rails, and eight rigid posts. Other gallery contexts select a
// different arena and therefore do not inherit these obstacles.
inline constexpr unsigned course_boundary_contact_count = 5U;
inline constexpr unsigned course_contact_count =
    course_boundary_contact_count + course_peg_count;

// Signed distance from a point to one rigid cylindrical course post. The
// collision solid continues into the floor and has a finite top; this avoids
// selecting a downward bottom-cap normal that would fight the floor contact.
// Positive is outside. Keeping this definition beside course_peg() makes the
// physics and renderer consume the same authored obstacle dimensions.
__host__ __device__ inline CourseContact rigid_course_post_contact(
    float3 p, unsigned peg_index)
{
    const float3 base = course_peg(peg_index);
    const float dx = p.x - base.x;
    const float dz = p.z - base.z;
    const float radial_length = sqrtf(dx * dx + dz * dz);
    const float radial_distance = radial_length - course_peg_radius;
    const float cap_distance =
        p.y - (course_floor_y + course_peg_height);
    const float outside_radial = fmaxf(radial_distance, 0.0F);
    const float outside_cap = fmaxf(cap_distance, 0.0F);
    const float distance = sqrtf(
        outside_radial * outside_radial + outside_cap * outside_cap) +
        fminf(fmaxf(radial_distance, cap_distance), 0.0F);
    const float3 radial = radial_length > 1.0e-8F
        ? make_float3(dx / radial_length, 0.0F, dz / radial_length)
        : make_float3(1.0F, 0.0F, 0.0F);
    float3 normal{};
    if (outside_radial > 0.0F && outside_cap > 0.0F) {
        const float inverse = 1.0F / sqrtf(
            outside_radial * outside_radial + outside_cap * outside_cap);
        normal = make_float3(
            radial.x * outside_radial * inverse,
            outside_cap * inverse,
            radial.z * outside_radial * inverse);
    } else if (radial_distance > cap_distance) {
        normal = radial;
    } else {
        normal = make_float3(0.0F, 1.0F, 0.0F);
    }
    return {distance, normal};
}

__host__ __device__ inline CourseContact course_contact(float3 p, unsigned index)
{
    switch (index) {
    case 0: return {p.y - course_floor_y, make_float3(0, 1, 0)};
    case 1: return {p.x + course_half_width, make_float3(1, 0, 0)};
    case 2: return {course_half_width - p.x, make_float3(-1, 0, 0)};
    case 3: return {course_near_z - p.z, make_float3(0, 0, -1)};
    case 4: return {p.z - course_far_z, make_float3(0, 0, 1)};
    default:
        return index < course_contact_count
            ? rigid_course_post_contact(p, index - course_boundary_contact_count)
            : CourseContact{INFINITY, make_float3(0, 0, 0)};
    }
}

// Unilateral, zero-restitution static contact. Clamp only inward normal
// velocity; bounded Coulomb friction removes tangential kinetic energy and
// gives the water surface traction. This is used identically for fluid and
// skin particles, not as an additional catch-up/rejection solver.
__host__ __device__ inline void project_course_contact(
    float3& p, float3& v, float radius)
{
    for (unsigned i = 0; i < course_contact_count; ++i) {
        const CourseContact c = course_contact(p, i);
        if (c.distance >= radius) continue;
        const float depth = radius - c.distance;
        p.x += c.normal.x * depth;
        p.y += c.normal.y * depth;
        p.z += c.normal.z * depth;
        const float vn = v.x * c.normal.x + v.y * c.normal.y + v.z * c.normal.z;
        if (vn >= 0.0F) continue;
        v.x -= vn * c.normal.x;
        v.y -= vn * c.normal.y;
        v.z -= vn * c.normal.z;
        const float speed = sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
        const float fraction = fmaxf(0.0F, 1.0F - 0.35F * (-vn) / fmaxf(speed, 1.0e-8F));
        v.x *= fraction;
        v.y *= fraction;
        v.z *= fraction;
    }
}

__host__ __device__ inline void project_gallery_contact(
    float3& p, float3& v, float radius, GalleryArena arena)
{
    if (arena == GalleryArena::course) {
        project_course_contact(p, v, radius);
        return;
    }
    float depth = 0.0F;
    float3 normal{};
    if (arena == GalleryArena::bowl && p.y <= bowl_center.y + radius) {
        const float3 from_center = make_float3(
            p.x - bowl_center.x, p.y - bowl_center.y, p.z - bowl_center.z);
        const float distance = sqrtf(from_center.x * from_center.x +
            from_center.y * from_center.y + from_center.z * from_center.z);
        const float limit = bowl_inner_radius - radius;
        if (distance > limit && distance > 1.0e-8F) {
            depth = distance - limit;
            normal = make_float3(-from_center.x / distance,
                -from_center.y / distance, -from_center.z / distance);
        }
    } else if (arena == GalleryArena::slope) {
        const float plane_y = slope_start_y + slope_gradient * p.x;
        const float signed_height = p.y - plane_y;
        normal = make_float3(-slope_gradient, 1.0F, 0.0F);
        const float scale = 1.0F / sqrtf(normal.x * normal.x + 1.0F);
        normal.x *= scale;
        normal.y *= scale;
        if (signed_height * scale < radius) {
            depth = radius - signed_height * scale;
        }
    } else if (arena == GalleryArena::water_wheel) {
        // The front outer rim is a true torus collider shared by the visible
        // stage and the rigid sphere. Water remains on the central z slice and
        // therefore never sees this deliberately extruded interaction rim.
        const float dx = p.x - water_wheel_center.x;
        const float dy = p.y - water_wheel_center.y;
        const float radial = sqrtf(dx * dx + dy * dy);
        const float radial_offset = radial-water_wheel_outer_disk_radius;
        const float axial_offset = p.z-water_wheel_stage_z;
        constexpr float rim_half_width=0.065F;
        const float radial_excess=fabsf(radial_offset)-rim_half_width;
        const float axial_excess=fabsf(axial_offset)-
            water_wheel_outer_disk_half_thickness;
        const float outside_radial=fmaxf(radial_excess,0.0F);
        const float outside_axial=fmaxf(axial_excess,0.0F);
        const float outside_length=sqrtf(outside_radial*outside_radial+
            outside_axial*outside_axial);
        const float rim_distance=outside_length+
            fminf(fmaxf(radial_excess,axial_excess),0.0F);
        if (rim_distance < radius && radial > 1.0e-8F) {
            float radial_normal{};
            float axial_normal{};
            if (outside_length>1.0e-8F) {
                radial_normal=(radial_offset>=0.0F ? 1.0F : -1.0F)*
                    outside_radial/outside_length;
                axial_normal=(axial_offset>=0.0F ? 1.0F : -1.0F)*
                    outside_axial/outside_length;
            } else if (radial_excess>axial_excess) {
                radial_normal=radial_offset>=0.0F ? 1.0F : -1.0F;
            } else {
                axial_normal=axial_offset>=0.0F ? 1.0F : -1.0F;
            }
            const float correction = radius-rim_distance;
            const float3 rim_normal = make_float3(
                radial_normal*dx/radial,radial_normal*dy/radial,axial_normal);
            p.x += correction * rim_normal.x;
            p.y += correction * rim_normal.y;
            p.z += correction * rim_normal.z;
            const float vn = v.x * rim_normal.x + v.y * rim_normal.y +
                v.z * rim_normal.z;
            if (vn < 0.0F) {
                v.x -= vn * rim_normal.x;
                v.y -= vn * rim_normal.y;
                v.z -= vn * rim_normal.z;
            }
        }
        // Only the large game sphere uses the front stage. Keep it between
        // the two visible bumper rails while leaving the central water slice
        // untouched. The open left end is the level exit.
        if (radius > 0.10F &&
            p.x >= water_wheel_center.x-water_wheel_top_platform_outer_x-radius &&
            p.x <= water_wheel_center.x+water_wheel_top_platform_outer_x+radius &&
            p.y >= water_wheel_top_platform_y-radius &&
            p.y <= water_wheel_top_platform_y+4.0F*radius) {
            const float z_limit = water_wheel_top_platform_half_depth-radius;
            if (p.z < water_wheel_stage_z-z_limit) {
                p.z = water_wheel_stage_z-z_limit;
                if (v.z < 0.0F) v.z = 0.0F;
            } else if (p.z > water_wheel_stage_z+z_limit) {
                p.z = water_wheel_stage_z+z_limit;
                if (v.z > 0.0F) v.z = 0.0F;
            }
            const float right_limit = water_wheel_center.x+
                water_wheel_top_platform_outer_x-radius;
            if (p.x > right_limit) {
                p.x = right_limit;
                if (v.x > 0.0F) v.x = 0.0F;
            }
        }
        float plane_y{};
        if (water_wheel_top_platform_contact(p, radius)) {
            plane_y = water_wheel_top_platform_y;
        } else if (!water_wheel_support_height(p.x, plane_y)) {
            return;
        }
        const float signed_height = p.y - plane_y;
        normal = make_float3(-water_wheel_ramp_gradient, 1.0F, 0.0F);
        const float scale = 1.0F / sqrtf(normal.x * normal.x + 1.0F);
        normal.x *= scale;
        normal.y *= scale;
        if (signed_height * scale < radius)
            depth = radius - signed_height * scale;
    } else if (arena == GalleryArena::rope_bridge) {
        const bool on_land = fabsf(p.z) >= rope_bridge_land_inner_z - radius;
        if (on_land && p.y < rope_bridge_land_y + radius) {
            normal = make_float3(0.0F, 1.0F, 0.0F);
            depth = rope_bridge_land_y + radius - p.y;
        }
        const float side_limit = rope_bridge_land_half_width - radius;
        if (p.x < -side_limit) {
            p.x = -side_limit;
            if (v.x < 0.0F) v.x = 0.0F;
        } else if (p.x > side_limit) {
            p.x = side_limit;
            if (v.x > 0.0F) v.x = 0.0F;
        }
    } else if ((arena == GalleryArena::ground || arena == GalleryArena::rope_post) &&
               p.y < course_floor_y + radius) {
        normal = make_float3(0.0F, 1.0F, 0.0F);
        depth = course_floor_y + radius - p.y;
    } else if (arena == GalleryArena::enclosed_box ||
               arena == GalleryArena::ground_box ||
               arena == GalleryArena::cloth_basin ||
               arena == GalleryArena::low_ceiling_box) {
        const float3 box_center = arena == GalleryArena::low_ceiling_box
            ? low_gallery_box_center : gallery_box_center;
        const float3 box_half_extents = arena == GalleryArena::low_ceiling_box
            ? low_gallery_box_half_extents : gallery_box_half_extents;
        const float lower_y = arena == GalleryArena::ground_box &&
                inside_ground_pit(p, radius)
            ? ground_pit_bottom_y + radius
            : box_center.y - box_half_extents.y + radius;
        const float3 minimum = make_float3(
            box_center.x - box_half_extents.x + radius,
            lower_y,
            box_center.z - box_half_extents.z + radius);
        const float3 maximum = make_float3(
            box_center.x + box_half_extents.x - radius,
            box_center.y + box_half_extents.y - radius,
            box_center.z + box_half_extents.z - radius);
        // Project against all axes. This treats the room as one closed volume,
        // including its fixed floor and ceiling, rather than six unrelated
        // one-sided faces that can be skipped at corners.
        if (p.x < minimum.x) {
            depth = minimum.x - p.x;
            normal = make_float3(1.0F, 0.0F, 0.0F);
        } else if (p.x > maximum.x) {
            depth = p.x - maximum.x;
            normal = make_float3(-1.0F, 0.0F, 0.0F);
        }
        if (depth > 0.0F) {
            p.x += normal.x * depth;
            const float vn = v.x * normal.x;
            if (vn < 0.0F) v.x -= vn * normal.x;
            v.y *= 0.985F;
            v.z *= 0.985F;
        }
        depth = 0.0F;
        if (p.y < minimum.y) {
            depth = minimum.y - p.y;
            normal = make_float3(0.0F, 1.0F, 0.0F);
        } else if (p.y > maximum.y) {
            depth = p.y - maximum.y;
            normal = make_float3(0.0F, -1.0F, 0.0F);
        }
        if (depth > 0.0F) {
            p.y += normal.y * depth;
            const float vn = v.y * normal.y;
            if (vn < 0.0F) v.y -= vn * normal.y;
            // This projection runs once per solver substep. The former 0.985
            // multiplier compounded to less than 0.1% retained speed over a
            // two-second approach, making the room floor behave like the
            // invisible bar reported in context 3.
            v.x *= 0.9995F;
            v.z *= 0.9995F;
        }
        depth = 0.0F;
        if (p.z < minimum.z) {
            depth = minimum.z - p.z;
            normal = make_float3(0.0F, 0.0F, 1.0F);
        } else if (p.z > maximum.z) {
            depth = p.z - maximum.z;
            normal = make_float3(0.0F, 0.0F, -1.0F);
        }
        if (depth > 0.0F) {
            p.z += normal.z * depth;
            const float vn = v.z * normal.z;
            if (vn < 0.0F) v.z -= vn * normal.z;
            v.x *= 0.985F;
            v.y *= 0.985F;
        }
        if (arena == GalleryArena::cloth_basin) {
            const float minimum_x = cloth_basin_center.x -
                cloth_basin_inner_half_extents.x + radius;
            const float maximum_x = cloth_basin_center.x +
                cloth_basin_inner_half_extents.x - radius;
            const float minimum_z = cloth_basin_center.z -
                cloth_basin_inner_half_extents.y + radius;
            const float maximum_z = cloth_basin_center.z +
                cloth_basin_inner_half_extents.y - radius;
            const bool in_exit = p.x > maximum_x - 0.34F &&
                p.z < minimum_z + 0.34F;
            if (p.x < minimum_x) {
                p.x = minimum_x;
                if (v.x < 0.0F) v.x = 0.0F;
            } else if (p.x > maximum_x && !in_exit) {
                p.x = maximum_x;
                if (v.x > 0.0F) v.x = 0.0F;
            }
            if (p.z < minimum_z && !in_exit) {
                p.z = minimum_z;
                if (v.z < 0.0F) v.z = 0.0F;
            } else if (p.z > maximum_z) {
                p.z = maximum_z;
                if (v.z > 0.0F) v.z = 0.0F;
            }
            // The visible rail is deliberately short, but its collision volume
            // reaches the room ceiling. Both sphere and water therefore obey
            // the same two-panel snake even when a fast substep climbs above it.
            if (p.y + radius > cloth_basin_center.y - 0.08F &&
                p.y - radius < cloth_snake_collision_top) {
                for (std::uint32_t wall = 0U;
                     wall < cloth_snake_wall_count; ++wall) {
                    const bool gap_right = (wall & 1U) == 0U;
                    const float wall_min_x = minimum_x +
                        (gap_right ? 0.0F : cloth_snake_gap_width);
                    const float wall_max_x = maximum_x -
                        (gap_right ? cloth_snake_gap_width : 0.0F);
                    const float center_z = cloth_snake_wall_z(wall);
                    if (p.x < wall_min_x - radius || p.x > wall_max_x + radius ||
                        fabsf(p.z - center_z) >=
                            cloth_snake_wall_half_thickness + radius) continue;
                    const float below = center_z - cloth_snake_wall_half_thickness - radius;
                    const float above = center_z + cloth_snake_wall_half_thickness + radius;
                    if (fabsf(p.z - below) < fabsf(above - p.z)) {
                        p.z = below;
                        if (v.z > 0.0F) v.z = 0.0F;
                    } else {
                        p.z = above;
                        if (v.z < 0.0F) v.z = 0.0F;
                    }
                }
            }
            v.x *= 0.985F;
            v.z *= 0.985F;
        }
        return;
    }
    if (arena == GalleryArena::bowl) {
        // The visible hemisphere is open, but the minigame owns an invisible
        // vertical continuation. Without it a sufficiently energetic particle
        // can clear the rim and is then outside the hemispherical SDF forever.
        // The cylinder uses the exact bowl radius, so it adds no visible ledge.
        const float dx = p.x - bowl_center.x;
        const float dz = p.z - bowl_center.z;
        const float radial_distance = sqrtf(dx * dx + dz * dz);
        const float radial_limit = bowl_inner_radius - radius;
        if (radial_distance > radial_limit && radial_distance > 1.0e-8F) {
            const float inverse = 1.0F / radial_distance;
            const float3 inward = make_float3(-dx * inverse, 0.0F, -dz * inverse);
            const float correction = radial_distance - radial_limit;
            p.x += inward.x * correction;
            p.z += inward.z * correction;
            const float vn = v.x * inward.x + v.z * inward.z;
            if (vn < 0.0F) {
                v.x -= vn * inward.x;
                v.z -= vn * inward.z;
            }
        }
        // The bowl and its four capped-cylinder pegs form one analytic contact
        // set. Apply all overlaps so corner contacts cannot be skipped.
        for (std::uint32_t peg = 0U; peg < bowl_peg_count; ++peg) {
            const CourseContact contact = bowl_peg_contact(p, peg);
            if (!(contact.distance < radius)) continue;
            const float correction = radius - contact.distance;
            const float3 peg_normal = contact.normal;
            p.x += peg_normal.x * correction;
            p.y += peg_normal.y * correction;
            p.z += peg_normal.z * correction;
            const float peg_vn = v.x*peg_normal.x + v.y*peg_normal.y + v.z*peg_normal.z;
            if (peg_vn < 0.0F) {
                v.x -= peg_vn * peg_normal.x;
                v.y -= peg_vn * peg_normal.y;
                v.z -= peg_vn * peg_normal.z;
            }
        }
    }
    if (depth > 0.0F) {
        p.x += normal.x * depth;
        p.y += normal.y * depth;
        p.z += normal.z * depth;
        const float vn = v.x * normal.x + v.y * normal.y + v.z * normal.z;
        if (vn < 0.0F) {
            v.x -= vn * normal.x;
            v.y -= vn * normal.y;
            v.z -= vn * normal.z;
        // Ground contact runs once per substep. A 0.93 multiplier at four to
        // sixteen substeps removed most horizontal momentum in one frame and
        // made the soft sphere buckle in place instead of rolling. Keep mild
        // contact drag here; material damping remains the main dissipation.
        // The bowl is a smooth rigid vessel. Tangential contact drag made
        // particles adhere to the curved wall and hold an artificial raised
        // annulus; bulk particle damping already dissipates the fluid.
            const float tangential_retention =
                arena == GalleryArena::bowl ? 1.0F : 0.997F;
            v.x *= tangential_retention;
            v.y *= tangential_retention;
            v.z *= tangential_retention;
        }
    }
    if (arena == GalleryArena::rope_post) {
        const CourseContact contact = rope_post_contact(p);
        if (contact.distance < radius) {
            const float correction = radius - contact.distance;
            p.x += contact.normal.x * correction;
            p.y += contact.normal.y * correction;
            p.z += contact.normal.z * correction;
            const float vn = v.x * contact.normal.x + v.y * contact.normal.y +
                v.z * contact.normal.z;
            if (vn < 0.0F) {
                v.x -= vn * contact.normal.x;
                v.y -= vn * contact.normal.y;
                v.z -= vn * contact.normal.z;
            }
        }
    }
}

} // namespace waterlab
