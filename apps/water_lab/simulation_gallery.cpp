// SPDX-License-Identifier: MIT
#include "simulation_gallery.hpp"

#include "cloth.hpp"
#include "obstacle_course.hpp"

#include <vector_functions.h>

#include <cstddef>
#include <algorithm>
#include <cmath>
#include <string>
#include <utility>

namespace waterlab::gallery {

const meshprep::sim::ExampleContextInfo& context_info(
    meshprep::sim::ExampleContext context) noexcept
{
    return meshprep::sim::example_contexts[
        static_cast<std::size_t>(context) - 1U];
}

bool context_has(
    meshprep::sim::ExampleContext context,
    meshprep::sim::Component component) noexcept
{
    return meshprep::sim::has_component(context_info(context).components, component);
}

HybridOptions make_context_physics(
    meshprep::sim::ExampleContext context,
    const ContextPhysicsOverrides& overrides) noexcept
{
    using meshprep::sim::ExampleContext;
    const bool course = context == ExampleContext::water_course;
    HybridOptions options = course ? course_options() : HybridOptions{};
    if (course) {
        // Context 1's authored neutral preset.  These values are deliberately
        // inside their live-control ranges so the default can be tuned in
        // either direction instead of starting at a limit.
        options.particle_count = 3'000U;
        options.physical_skin_frequency = 10U;
        options.render_skin_frequency = 10U;
        options.particle_repulsion = 20.0F;
    }
    if (!course) {
        options.physics_iterations = 4U;
        options.gravity = make_float3(0.0F, -9.81F, 0.0F);
        options.physical_skin_frequency = 3U;
        options.render_skin_frequency = 12U;
    }
    if (context == ExampleContext::particle_bowl) {
        options.gravity = make_float3(0.0F, -19.62F, 0.0F);
        options.particle_count = 15'000U;
        options.particle_capacity = 15'000U;
        options.particle_initial_center = make_float3(0.0F, 1.05F, -1.8F);
        // A lower-repulsion HCP fill retains a compact pile instead of
        // spreading across the bowl under the authored two-g load.
        options.particle_repulsion = 20.0F;
    } else if (context == ExampleContext::particles_cloth) {
        options.particle_count = 2'500U;
        options.particle_initial_center = make_float3(
            catch_cloth_center.x, 0.85F, catch_cloth_center.z);
    } else if (context == ExampleContext::soft_body_fluid) {
        options.particle_count = 2'000U;
        const float spawn_x = water_wheel_entry_x - 2.15F;
        options.particle_initial_center = make_float3(
            spawn_x, water_wheel_inlet_height(spawn_x) + 0.32F, 0.0F);
    }
    options.fixed_dt = overrides.fixed_dt;
    if (overrides.physics_iterations.has_value()) {
        options.physics_iterations = *overrides.physics_iterations;
    }
    if (overrides.gravity.has_value()) options.gravity = *overrides.gravity;
    if (overrides.particle_count.has_value()) {
        options.particle_count = *overrides.particle_count;
        options.particle_capacity = std::max(
            options.particle_capacity, options.particle_count);
    }
    if (overrides.physical_skin_frequency.has_value()) {
        options.physical_skin_frequency = *overrides.physical_skin_frequency;
        options.render_skin_frequency = std::max(
            options.render_skin_frequency, options.physical_skin_frequency);
    }
    options.obstacle_course = course;
    if (course) options.arena = GalleryArena::course;
    else if (context == ExampleContext::particle_bowl)
        options.arena = GalleryArena::bowl;
    else if (context == ExampleContext::soft_body_fluid)
        options.arena = GalleryArena::water_wheel;
    else if (context == ExampleContext::particles_cloth)
        options.arena = GalleryArena::cloth_basin;
    else if (context == ExampleContext::soft_body_rigid)
        options.arena = GalleryArena::low_ceiling_box;
    else if (context == ExampleContext::cloth_rigid)
        options.arena = GalleryArena::enclosed_box;
    else if (context == ExampleContext::soft_body_cloth)
        options.arena = GalleryArena::ground_box;
    else if (context == ExampleContext::rope_rigid)
        options.arena = GalleryArena::rope_post;
    options.particle_skin_coupling =
        context != ExampleContext::particle_bowl &&
        context != ExampleContext::particles_cloth &&
        context != ExampleContext::soft_body_fluid;
    return options;
}

std::unique_ptr<SoftBodyCourse> make_context_deformable(
    meshprep::sim::ExampleContext context,
    const HybridOptions& physics,
    std::string_view soft_body_asset_path,
    std::uint32_t rope_node_count)
{
    SoftBodyOptions options;
    options.fixed_dt = physics.fixed_dt;
    options.solver_substeps = physics.physics_iterations;
    if (context == meshprep::sim::ExampleContext::soft_body_rigid ||
        context == meshprep::sim::ExampleContext::soft_body_fluid ||
        context == meshprep::sim::ExampleContext::soft_body_cloth ||
        context == meshprep::sim::ExampleContext::particles_cloth ||
        context == meshprep::sim::ExampleContext::rope_rigid)
        options.spring_solver_iterations = 16U;
    options.maximum_speed = physics.maximum_skin_speed;
    // The cloth carries its own weight from two pins before any impact.
    options.strength_multiplier =
        context_has(context, meshprep::sim::Component::cloth) ||
        context == meshprep::sim::ExampleContext::soft_body_rigid ||
        context == meshprep::sim::ExampleContext::soft_body_fluid ||
        context == meshprep::sim::ExampleContext::soft_body_cloth
        ? 1.5F : 0.5F;
    if (context == meshprep::sim::ExampleContext::soft_body_rigid) {
        // A bonded column must support gravity before any impact. These values
        // sit well below the live maxima, leaving meaningful room for both the
        // SOFT SPRING and SOFT BOND controls to strengthen it further.
        options.spring_stiffness = 80'000.0F;
        options.strength_multiplier = 8.0F;
        options.velocity_damping = 0.45F;
    } else if (context == meshprep::sim::ExampleContext::soft_body_cloth) {
        // The free sphere is a load-bearing volume, not a tearable cloth. Its
        // dense rest graph and non-bonded barrier preserve volume while the
        // low ground drag above lets that volume translate and roll.
        options.spring_stiffness = 20'000.0F;
        options.strength_multiplier = 16.0F;
        options.ground_friction = 10.0F;
        options.unbonded_voxel_collisions = true;
        options.velocity_damping = 0.45F;
    } else if (context == meshprep::sim::ExampleContext::cloth_rigid) {
        // Impact strain is sampled before projection in this scene; a higher
        // threshold keeps the tear local around the sphere instead of erasing
        // the entire hanging sheet.
        options.spring_stiffness = 4'000.0F;
        options.strength_multiplier = 4.0F;
        options.fracture_persistence_substeps = 32U;
    } else if (context == meshprep::sim::ExampleContext::particles_cloth) {
        // A 2,500-particle load is shared by sixteen support junctions. Use the same
        // converged fixed-graph solve as the anchored post instead of letting
        // residual stretch accumulate into an early fracture cascade.
        options.spring_stiffness = 4'500.0F;
        options.strength_multiplier = 4.0F;
        options.velocity_damping = 0.45F;
    } else if (context == meshprep::sim::ExampleContext::rope_rigid) {
        options.spring_stiffness = 18'000.0F;
        options.strength_multiplier = 64.0F;
        options.spring_damping_ratio = 0.9F;
        options.velocity_damping = 0.35F;
        options.render_internal_members = true;
    }
    options.course_board_collisions =
        context != meshprep::sim::ExampleContext::soft_body_fluid;
    options.arena = physics.arena;
    options.render_internal_members =
        context == meshprep::sim::ExampleContext::soft_body_rigid ||
        context == meshprep::sim::ExampleContext::rope_rigid;
    options.fracture_before_projection =
        context == meshprep::sim::ExampleContext::cloth_rigid;
    options.preserve_fractured_triangle_shape =
        context == meshprep::sim::ExampleContext::cloth_rigid;

    using meshprep::sim::ExampleContext;
    if (context == ExampleContext::particle_bowl ||
        context == ExampleContext::water_course) return {};

    options.instance_count = 1U;
    options.use_course_layout = false;
    options.require_1000_voxels = false;
    if (context == ExampleContext::rope_rigid) {
        rope_node_count = std::clamp(rope_node_count, 8U, 512U);
        SoftBodyAsset rope = make_soft_rope(
            rope_node_count, rope_length / static_cast<float>(rope_node_count - 1U));
        rope = translate_soft_body_asset(std::move(rope), rope_anchor);
        return std::make_unique<SoftBodyCourse>(std::move(rope), options);
    }
    if (context == ExampleContext::cloth_rigid ||
        context == ExampleContext::particles_cloth) {
        ClothGridOptions cloth;
        cloth.columns = context == ExampleContext::cloth_rigid
            ? 28U : catch_cloth_columns;
        cloth.rows = context == ExampleContext::cloth_rigid
            ? 30U : catch_cloth_rows;
        cloth.spacing = context == ExampleContext::cloth_rigid
            ? 0.065F : catch_cloth_spacing;
        cloth.top_center = context == ExampleContext::particles_cloth
            ? make_float3(0.0F, 0.0F, 0.0F)
            : make_float3(0.0F,
                course_floor_y + static_cast<float>(cloth.rows - 1U) * cloth.spacing,
                -0.8F);
        SoftBodyAsset asset = make_cloth_grid(cloth);
        if (context == ExampleContext::particles_cloth) {
            asset = rotate_cloth_to_horizontal(std::move(asset));
            const float half_span_z = 0.5F *
                static_cast<float>(cloth.rows - 1U) * cloth.spacing;
            asset = translate_soft_body_asset(
                std::move(asset),
                make_float3(catch_cloth_center.x, catch_cloth_center.y,
                    catch_cloth_center.z - half_span_z));
            // One horizontal sheet divided into a real 3x3 panel grid. Every
            // node on the four-by-four support-line lattice is fixed, so each
            // of the nine cells is an independently flexible cloth panel in
            // a square support rather than one sagging sheet with 16 pins.
            for (std::uint32_t& flags : asset.voxel_flags)
                flags &= ~soft_body_voxel_pinned;
            const std::uint32_t support_columns[4]{
                0U, (cloth.columns - 1U) / 3U,
                2U * (cloth.columns - 1U) / 3U, cloth.columns - 1U};
            const std::uint32_t support_rows[4]{
                0U, (cloth.rows - 1U) / 3U,
                2U * (cloth.rows - 1U) / 3U, cloth.rows - 1U};
            for (std::uint32_t row = 0U; row < cloth.rows; ++row) {
                for (std::uint32_t column = 0U; column < cloth.columns; ++column) {
                    const bool support_row = std::find(std::begin(support_rows),
                        std::end(support_rows), row) != std::end(support_rows);
                    const bool support_column = std::find(std::begin(support_columns),
                        std::end(support_columns), column) != std::end(support_columns);
                    if (support_row || support_column)
                        asset.voxel_flags[row * cloth.columns + column] |=
                            soft_body_voxel_pinned;
                }
            }
        } else {
            // Context 3 is a tensioned target: both horizontal borders are
            // fixed so the rigid sphere passes through the unsupported center
            // instead of carrying the whole curtain out of the room.
            const std::uint32_t bottom_row = (cloth.rows - 1U) * cloth.columns;
            for (std::uint32_t column = 0U; column < cloth.columns; ++column) {
                asset.voxel_flags[column] |= soft_body_voxel_pinned;
                const std::uint32_t bottom = bottom_row + column;
                asset.voxel_flags[bottom] |= soft_body_voxel_pinned;
                // Bury the fixed node spheres below the floor. Their previous
                // centers sat on the floor, so collision radius protruded as
                // an invisible horizontal bar in front of the cloth.
                asset.rest_voxels[bottom].y -= cloth.spacing;
                asset.render_positions[bottom].y -= cloth.spacing;
            }
            for (SoftBodyEdge& edge : asset.edges) {
                const float3 a = asset.rest_voxels[edge.vertices.x];
                const float3 b = asset.rest_voxels[edge.vertices.y];
                edge.rest_length = std::sqrt(
                    (a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y) +
                    (a.z-b.z)*(a.z-b.z));
            }
        }
        return std::make_unique<SoftBodyCourse>(std::move(asset), options);
    }

    if (context == ExampleContext::soft_body_cloth) {
        SoftBodyAsset sphere = make_soft_sphere();
        const float spacing = sphere.nominal_spacing;
        sphere = translate_soft_body_asset(std::move(sphere),
            make_float3(0.0F, course_floor_y + 0.46F, -0.35F));
        ClothGridOptions cloth;
        cloth.columns = 24U;
        cloth.rows = 24U;
        cloth.spacing = spacing;
        cloth.top_center = make_float3(0.0F, 1.05F, -1.8F);
        SoftBodyAsset curtain = make_cloth_grid(cloth);
        for (std::uint32_t column = 0U; column < cloth.columns; ++column)
            curtain.voxel_flags[column] |= soft_body_voxel_pinned;
        curtain.file_flags |= soft_body_asset_free_body;

        // Add a second, horizontal cloth surface over the rigid ground. Its
        // pinned perimeter lets the sphere roll across and visibly load it;
        // the rigid plane beneath remains the final containment fallback.
        ClothGridOptions ground_options;
        ground_options.columns = 40U;
        ground_options.rows = 40U;
        ground_options.spacing = spacing;
        ground_options.top_center = {};
        SoftBodyAsset ground_cloth = rotate_cloth_to_horizontal(
            make_cloth_grid(ground_options));
        for (std::uint32_t& flags : ground_cloth.voxel_flags)
            flags &= ~soft_body_voxel_pinned;
        for (std::uint32_t row = 0U; row < ground_options.rows; ++row) {
            for (std::uint32_t column = 0U; column < ground_options.columns; ++column) {
                if (row == 0U || row + 1U == ground_options.rows ||
                    column == 0U || column + 1U == ground_options.columns) {
                    ground_cloth.voxel_flags[
                        row * ground_options.columns + column] |= soft_body_voxel_pinned;
                }
            }
        }
        const float ground_half_span = 0.5F *
            static_cast<float>(ground_options.rows - 1U) * spacing;
        ground_cloth = translate_soft_body_asset(std::move(ground_cloth),
            make_float3(0.0F, course_floor_y + 0.025F,
                -0.55F - ground_half_span));
        ground_cloth.file_flags |= soft_body_asset_free_body;

        options.cross_source_nodes = 1'000U;
        options.cross_target_triangle_first =
            static_cast<std::uint32_t>(sphere.render_triangles.size());
        SoftBodyAsset cloth_surfaces = merge_soft_body_assets(curtain, ground_cloth);
        auto result = std::make_unique<SoftBodyCourse>(
            merge_soft_body_assets(sphere, cloth_surfaces), options);
        initialize_context_motion(context, *result);
        return result;
    }

    if (context == ExampleContext::soft_body_fluid) {
        SoftBodyAsset front = translate_soft_body_asset(make_soft_cross(), make_float3(
            water_wheel_center.x, water_wheel_center.y,
            water_wheel_center.z + water_wheel_cross_offset));
        SoftBodyAsset back = translate_soft_body_asset(make_soft_cross(), make_float3(
            water_wheel_center.x, water_wheel_center.y,
            water_wheel_center.z - water_wheel_cross_offset));
        SoftBodyAsset cross = merge_soft_body_assets(front, back);
        // The axle cross is a compliant load path, not a consumable fracture
        // target. Keep visible flex while preventing ordinary wheel pressure
        // from being interpreted as thousands of permanent breaks.
        options.spring_stiffness = 8'000.0F;
        options.strength_multiplier = 16.0F;
        options.unbonded_voxel_collisions = true;
        return std::make_unique<SoftBodyCourse>(std::move(cross), options);
    }

    SoftBodyAsset cylinder = load_soft_body_asset(std::string(soft_body_asset_path));

    if (context == ExampleContext::soft_body_rigid) {
        // Hang the column from the room ceiling.  Only the authored top cap is
        // prescribed; the lower end is free so the sphere can strike it.
        const float maximum_y = std::max_element(cylinder.rest_voxels.begin(),
            cylinder.rest_voxels.end(), [](float3 a, float3 b) {
                return a.y < b.y;
            })->y;
        for (std::size_t node = 0U; node < cylinder.rest_voxels.size(); ++node) {
            cylinder.voxel_flags[node] &= ~soft_body_voxel_pinned;
            if (cylinder.rest_voxels[node].y >=
                maximum_y - 0.65F * cylinder.nominal_spacing) {
                cylinder.voxel_flags[node] |= soft_body_voxel_pinned;
            }
        }
        const float ceiling = hanging_ceiling_y;
        options.instance_origins[0] = make_float3(
            0.0F, ceiling - 0.02F - maximum_y, -1.8F);
    } else {
        // The authored bottom ring remains pinned in the obstacle course.
        options.instance_origins[0] = make_float3(
            0.0F, course_floor_y + 0.825F, -1.8F);
    }
    return std::make_unique<SoftBodyCourse>(std::move(cylinder), options);
}

FluidDisplay default_context_display(meshprep::sim::ExampleContext context) noexcept
{
    return context == meshprep::sim::ExampleContext::water_course ||
        context == meshprep::sim::ExampleContext::particle_bowl ||
        context == meshprep::sim::ExampleContext::particles_cloth
        ? FluidDisplay::Surface : FluidDisplay::Particles;
}

RigidSphereState initial_rigid_sphere(
    meshprep::sim::ExampleContext context, std::uint32_t rope_node_count) noexcept
{
    RigidSphereState sphere;
    sphere.radius = 0.40F;
    sphere.mass = 20.0F;
    if (context == meshprep::sim::ExampleContext::particle_bowl) {
        sphere.radius = 0.34F;
        // One particle has unit simulation mass. This exceeds the roughly
        // 2,500 particles displaced by the sphere, so it settles on the bowl
        // instead of floating at the free surface.
        sphere.mass = 3'500.0F;
        sphere.center = make_float3(0.0F, 0.80F, -1.8F);
        sphere.velocity = make_float3(0.35F, -0.15F, 0.0F);
    } else if (context == meshprep::sim::ExampleContext::cloth_rigid) {
        // Start far enough from the tensioned cloth to make the approach
        // legible. Its lower pinned row is buried below the floor rather than
        // protruding into the sphere's path as an invisible curb.
        sphere.center = make_float3(0.0F, 0.18F, 1.0F);
        sphere.velocity = make_float3(0.0F, 0.0F, 0.0F);
        sphere.mass = 250.0F;
    } else if (context == meshprep::sim::ExampleContext::particles_cloth) {
        sphere.radius = 0.17F;
        sphere.mass = 12.0F;
        sphere.center = make_float3(
            catch_cloth_center.x, 1.35F, catch_cloth_center.z);
        sphere.velocity = make_float3(0.0F, -1.20F, 0.0F);
    } else if (context == meshprep::sim::ExampleContext::soft_body_rigid) {
        sphere.radius = 0.34F;
        sphere.mass = 80.0F;
        sphere.center = make_float3(-2.0F, course_floor_y + sphere.radius, -1.8F);
        sphere.velocity = make_float3(1.8F, 0.0F, 0.0F);
    } else if (context == meshprep::sim::ExampleContext::rope_rigid) {
        rope_node_count = std::clamp(rope_node_count, 8U, 512U);
        const float node_radius = 0.36F * rope_length /
            static_cast<float>(rope_node_count - 1U);
        sphere.radius = 0.32F;
        sphere.mass = 18.0F;
        sphere.center = make_float3(
            rope_anchor.x + rope_length + sphere.radius + node_radius,
            rope_anchor.y, rope_anchor.z);
        sphere.velocity = {};
    } else {
        // A grazing track loads the breakable wall while not demanding that a
        // floor-height ball tunnel through the post's immovable foundation.
        sphere.center = make_float3(-2.15F, course_floor_y + sphere.radius, -1.25F);
        sphere.velocity = make_float3(2.0F, 0.0F, 0.0F);
    }
    return sphere;
}

void initialize_context_motion(
    meshprep::sim::ExampleContext context, SoftBodyCourse& body)
{
    // Context 7 is driven by the same live gravity tilt as every other
    // gallery scene. It starts from rest so translation and rotation come
    // from gravity plus ground contact, not a prescribed launch velocity.
    (void)context;
    (void)body;
}

} // namespace waterlab::gallery
