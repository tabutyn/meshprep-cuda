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

const parallel_mater::examples::SimulationRecipeInfo& recipe_info(
    parallel_mater::examples::SimulationRecipe context) noexcept
{
    for (const auto& item : parallel_mater::examples::simulation_recipes) {
        if (item.recipe == context) return item;
    }
    return parallel_mater::examples::simulation_recipes.front();
}

bool recipe_has(
    parallel_mater::examples::SimulationRecipe context,
    parallel_mater::examples::Component component) noexcept
{
    return parallel_mater::examples::has_component(recipe_info(context).components, component);
}

HybridOptions make_recipe_physics(
    parallel_mater::examples::SimulationRecipe context,
    const RecipePhysicsOverrides& overrides) noexcept
{
    using parallel_mater::examples::SimulationRecipe;
    const bool course = context == SimulationRecipe::water_cloth;
    HybridOptions options = course ? course_options() : HybridOptions{};
    if (course) {
        // Water-Cloth's authored neutral preset. These values are deliberately
        // inside their live-control ranges so the default can be tuned in
        // either direction instead of starting at a limit.
        options.particle_count = 3'000U;
        options.physical_skin_frequency = 10U;
        options.render_skin_frequency = 10U;
        options.particle_repulsion = 20.0F;
        options = with_course_motion_multiplier(options, 8.0F);
        // Preserve the authored material control while selecting the requested
        // 8x motion preset; bracket changes continue scaling it from here.
        options.particle_repulsion = 20.0F;
    }
    if (!course) {
        options.physics_iterations = 4U;
        options.gravity = make_float3(0.0F, -9.81F, 0.0F);
        options.physical_skin_frequency = 3U;
        options.render_skin_frequency = 12U;
    }
    if (context == SimulationRecipe::water) {
        options.gravity = make_float3(0.0F, -19.62F, 0.0F);
        options.particle_count = 20'000U;
        options.particle_capacity = 20'000U;
        options.particle_initial_center = make_float3(0.0F, 1.05F, -1.8F);
        // A lower-repulsion HCP fill retains a compact pile instead of
        // spreading across the bowl under the authored two-g load.
        options.particle_repulsion = 50.0F;
    } else if (context == SimulationRecipe::water_rope) {
        options.particle_count = 40'000U;
        options.particle_capacity = 40'000U;
        options.particle_initial_center = make_float3(0.0F,-0.78F,-1.15F);
        options.particle_initial_scale = make_float3(2.05F,0.90F,0.50F);
        // Repulsion controls compressibility, not fill level. 120 is a useful
        // authored stiffness at four substeps; 500 merely hides a sparse
        // initial fill and makes the explicit particle solve much harsher.
        options.particle_repulsion = 120.0F;
    } else if (context == SimulationRecipe::water_soft_body) {
        options.particle_count = 20'000U;
        options.particle_capacity = 20'000U;
        const float spawn_x = water_wheel_entry_x - 2.15F;
        options.particle_initial_center = make_float3(
            spawn_x, water_wheel_inlet_height(spawn_x) + 0.32F, 0.0F);
    } else if (context == SimulationRecipe::fluid_smoke) {
        // A shallow pool sits on the heated floor while the independent smoke
        // system emits steam above it. Keeping the two particle populations
        // separate makes the phase boundary explicit in the public API.
        options.particle_count = 8'000U;
        options.particle_capacity = 20'000U;
        options.particle_initial_center = make_float3(0.0F,-0.72F,-1.2F);
        options.particle_initial_scale = make_float3(1.40F,0.20F,1.05F);
        options.particle_repulsion = 60.0F;
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
    else if (context == SimulationRecipe::water)
        options.arena = GalleryArena::bowl;
    else if (context == SimulationRecipe::water_soft_body)
        options.arena = GalleryArena::water_wheel;
    else if (context == SimulationRecipe::water_rope)
        options.arena = GalleryArena::fishing_tank;
    else if (context == SimulationRecipe::soft_body)
        options.arena = GalleryArena::low_ceiling_box;
    else if (context == SimulationRecipe::cloth)
        options.arena = GalleryArena::enclosed_box;
    else if (context == SimulationRecipe::cloth_soft_body)
        options.arena = GalleryArena::ground_box;
    else if (context == SimulationRecipe::rope)
        options.arena = GalleryArena::rope_post;
    else if (context == SimulationRecipe::soft_body_rope)
        options.arena = GalleryArena::rope_bridge;
    else if (context == SimulationRecipe::cloth_rope)
        options.arena = GalleryArena::rope_bridge;
    else if (context == SimulationRecipe::smoke ||
            context == SimulationRecipe::cloth_smoke)
        options.arena = GalleryArena::ground;
    else if (context == SimulationRecipe::soft_body_smoke)
        options.arena = GalleryArena::grass;
    else if (context == SimulationRecipe::fluid_smoke)
        options.arena = GalleryArena::hot_pan;
    else if (context == SimulationRecipe::rope_smoke)
        options.arena = GalleryArena::rope_bridge;
    options.particle_skin_coupling =
        context != SimulationRecipe::water &&
        context != SimulationRecipe::water_rope &&
        context != SimulationRecipe::water_soft_body;
    return options;
}

std::unique_ptr<SoftBodyCourse> make_recipe_deformable(
    parallel_mater::examples::SimulationRecipe context,
    const HybridOptions& physics,
    std::string_view soft_body_asset_path,
    std::uint32_t rope_node_count,
    std::uint32_t cloth_detail,
    std::uint32_t bridge_columns,
    std::uint32_t bridge_rows,
    std::uint32_t cylinder_columns,
    std::uint32_t cylinder_rows)
{
    SoftBodyOptions options;
    options.fixed_dt = physics.fixed_dt;
    options.solver_substeps = physics.physics_iterations;
    if (context == parallel_mater::examples::SimulationRecipe::soft_body ||
        context == parallel_mater::examples::SimulationRecipe::water_soft_body ||
        context == parallel_mater::examples::SimulationRecipe::cloth_soft_body ||
        context == parallel_mater::examples::SimulationRecipe::water_rope ||
        context == parallel_mater::examples::SimulationRecipe::rope ||
        context == parallel_mater::examples::SimulationRecipe::cloth_rope ||
        context == parallel_mater::examples::SimulationRecipe::soft_body_rope)
        options.spring_solver_iterations = 16U;
    options.maximum_speed = physics.maximum_skin_speed;
    // The cloth carries its own weight from two pins before any impact.
    options.strength_multiplier =
        recipe_has(context, parallel_mater::examples::Component::cloth) ||
        context == parallel_mater::examples::SimulationRecipe::soft_body ||
        context == parallel_mater::examples::SimulationRecipe::water_soft_body ||
        context == parallel_mater::examples::SimulationRecipe::cloth_soft_body
        ? 1.5F : 0.5F;
    if (context == parallel_mater::examples::SimulationRecipe::soft_body) {
        // A bonded column must support gravity before any impact. These values
        // sit well below the live maxima, leaving meaningful room for both the
        // SOFT SPRING and SOFT BOND controls to strengthen it further.
        options.spring_stiffness = 80'000.0F;
        options.strength_multiplier = 8.0F;
        options.velocity_damping = 0.45F;
    } else if (context == parallel_mater::examples::SimulationRecipe::cloth_soft_body) {
        // The free sphere is a load-bearing volume, not a tearable cloth. Its
        // dense rest graph and non-bonded barrier preserve volume while the
        // low ground drag above lets that volume translate and roll.
        options.spring_stiffness = 20'000.0F;
        options.strength_multiplier = 16.0F;
        options.ground_friction = 10.0F;
        options.unbonded_voxel_collisions = true;
        options.velocity_damping = 0.45F;
        options.cross_source_mass_multiplier = 8.0F;
    } else if (context == parallel_mater::examples::SimulationRecipe::cloth) {
        // Impact strain is sampled before projection in this scene. Require a
        // persistent but locally reachable strain so rolling contact can tear
        // the sheet without treating one gravity-loaded solve as damage.
        options.spring_stiffness = 4'000.0F;
        options.ground_friction = 5.0F;
        // Rest/gravity load must not count as damage. Fracture is driven by
        // the measured rigid-contact impulse in the contact pass below.
        options.strength_multiplier = 8.0F;
        options.fracture_persistence_substeps = 16U;
    } else if (context == parallel_mater::examples::SimulationRecipe::water_rope) {
        options.spring_stiffness = 22'000.0F;
        options.strength_multiplier = 64.0F;
        options.spring_damping_ratio = 0.92F;
        options.velocity_damping = 0.65F;
        options.render_internal_members = true;
    } else if (context == parallel_mater::examples::SimulationRecipe::water_soft_body) {
        // Shared by the top stage and the extruded outer wheel. It controls
        // rigid-ball rolling only; water and wheel gravity remain authored.
        options.ground_friction = 10.0F;
    } else if (context == parallel_mater::examples::SimulationRecipe::rope) {
        options.spring_stiffness = 18'000.0F;
        options.strength_multiplier = 64.0F;
        options.spring_damping_ratio = 0.9F;
        options.velocity_damping = 0.35F;
        options.render_internal_members = true;
    } else if (context == parallel_mater::examples::SimulationRecipe::cloth_rope ||
               context == parallel_mater::examples::SimulationRecipe::soft_body_rope ||
               context == parallel_mater::examples::SimulationRecipe::rope_smoke) {
        options.spring_stiffness = 32'000.0F;
        options.strength_multiplier = 64.0F;
        options.spring_damping_ratio = 0.92F;
        options.velocity_damping = 0.55F;
        options.render_internal_members = true;
    }
    options.course_board_collisions =
        context != parallel_mater::examples::SimulationRecipe::water_soft_body;
    options.arena = physics.arena;
    options.render_internal_members =
        context == parallel_mater::examples::SimulationRecipe::water_rope ||
        context == parallel_mater::examples::SimulationRecipe::rope ||
        context == parallel_mater::examples::SimulationRecipe::cloth_rope ||
        context == parallel_mater::examples::SimulationRecipe::soft_body_rope ||
        context == parallel_mater::examples::SimulationRecipe::rope_smoke;
    options.fracture_before_projection =
        context == parallel_mater::examples::SimulationRecipe::cloth ||
        context == parallel_mater::examples::SimulationRecipe::cloth_soft_body;
    options.preserve_fractured_triangle_shape =
        context == parallel_mater::examples::SimulationRecipe::cloth ||
        context == parallel_mater::examples::SimulationRecipe::cloth_soft_body;

    using parallel_mater::examples::SimulationRecipe;
    if (context == SimulationRecipe::water ||
        context == SimulationRecipe::water_cloth ||
        context == SimulationRecipe::smoke ||
        context == SimulationRecipe::fluid_smoke) return {};

    options.instance_count = 1U;
    options.use_course_layout = false;
    options.require_1000_voxels = false;
    if (context == SimulationRecipe::rope) {
        rope_node_count = std::clamp(rope_node_count, 8U, 512U);
        const std::uint32_t authored_nodes=std::max(16U,rope_node_count);
        const std::uint32_t trunk_nodes=std::max(8U,authored_nodes/2U);
        const std::uint32_t branch_nodes=std::max(
            5U,(authored_nodes-trunk_nodes)/2U+1U);
        options.cage_first_node=trunk_nodes+branch_nodes-1U;
        options.cage_node_count=8U;
        SoftBodyAsset rope = make_y_rope_cage(
            authored_nodes,rope_length);
        rope = translate_soft_body_asset(std::move(rope), rope_anchor);
        return std::make_unique<SoftBodyCourse>(std::move(rope), options);
    }
    if (context == SimulationRecipe::water_rope) {
        rope_node_count=std::clamp(rope_node_count,16U,512U);
        const float spacing=3.85F/static_cast<float>(rope_node_count-1U);
        SoftBodyAsset fishing_rope=make_soft_rope(rope_node_count,spacing);
        SoftBodyRigidTransform hanging;
        // The generic rope is authored along +X.  Fishing needs its one pinned
        // endpoint at the tank head and its free hook directly below it.
        hanging.x_axis=make_float3(0.0F,-1.0F,0.0F);
        hanging.y_axis=make_float3(1.0F,0.0F,0.0F);
        hanging.z_axis=make_float3(0.0F,0.0F,1.0F);
        hanging.translation=fishing_rope_anchor;
        fishing_rope=transform_soft_body_asset(
            std::move(fishing_rope),hanging);
        return std::make_unique<SoftBodyCourse>(std::move(fishing_rope),options);
    }
    if (context == SimulationRecipe::cloth_rope ||
        context == SimulationRecipe::soft_body_rope ||
        context == SimulationRecipe::rope_smoke) {
        bridge_columns=std::clamp(bridge_columns,2U,16U);
        bridge_rows=std::clamp(bridge_rows,2U,64U);
        options.rope_bridge_columns=bridge_columns;
        options.rope_bridge_rows=bridge_rows;
        options.rope_bridge_nodes_per_tile=
            context==SimulationRecipe::soft_body_rope ? 256U : 36U;
        SoftBodyAsset bridge = context==SimulationRecipe::soft_body_rope
            ? make_dense_tile_rope_bridge(bridge_columns,bridge_rows)
            : make_rope_bridge(bridge_columns,bridge_rows,true);
        return std::make_unique<SoftBodyCourse>(std::move(bridge), options);
    }
    if (context == SimulationRecipe::cloth_smoke) {
        // Four independent cloth blades share a pinned hub. The blades are
        // pitched 30 degrees about their radial axes so the smoke stream has
        // a non-zero tangential load instead of striking a symmetric sheet.
        ClothGridOptions blade_options;
        blade_options.columns=6U;
        blade_options.rows=13U;
        blade_options.spacing=0.075F;
        SoftBodyAsset blade=make_cloth_grid(blade_options);
        for (std::uint32_t& flags:blade.voxel_flags)
            flags&=~soft_body_voxel_pinned;
        for (std::uint32_t column=0U;column<blade_options.columns;++column)
            blade.voxel_flags[column]|=
                soft_body_voxel_pinned|soft_body_voxel_rim_anchor;
        constexpr float pi=3.14159265358979323846F;
        constexpr float pitch=pi/6.0F;
        SoftBodyAsset windmill;
        for (std::uint32_t arm=0U;arm<4U;++arm) {
            const float angle=0.5F*pi*static_cast<float>(arm);
            SoftBodyRigidTransform transform;
            transform.x_axis=make_float3(cosf(angle)*cosf(pitch),
                sinf(angle)*cosf(pitch),-sinf(pitch));
            transform.y_axis=make_float3(-sinf(angle),cosf(angle),0.0F);
            transform.z_axis=make_float3(cosf(angle)*sinf(pitch),
                sinf(angle)*sinf(pitch),cosf(pitch));
            transform.translation=make_float3(0.0F,0.15F,-1.20F);
            SoftBodyAsset arm_asset=transform_soft_body_asset(blade,transform);
            windmill=arm==0U ? std::move(arm_asset)
                : merge_soft_body_assets(windmill,arm_asset);
        }
        options.spring_stiffness=5'000.0F;
        options.strength_multiplier=64.0F;
        options.velocity_damping=0.6F;
        return std::make_unique<SoftBodyCourse>(std::move(windmill),options);
    }
    if (context == SimulationRecipe::cloth) {
        cloth_detail = cloth_detail == 0U
            ? default_cloth_detail(context)
            : std::clamp(cloth_detail, 1U, 8U);
        // Detail refines one physical sheet; it must not multiply its mass by
        // detail squared. Keeping total areal mass constant also prevents a
        // high-detail hanging cloth tearing under its own newly-added weight.
        options.voxel_mass /= static_cast<float>(cloth_detail*cloth_detail);
        options.spring_solver_iterations = std::min(256U,
            std::max(options.spring_solver_iterations,16U*cloth_detail));
        ClothGridOptions cloth;
        const bool hanging = true;
        cloth.columns = 27U * cloth_detail + 1U;
        cloth.rows = (hanging ? 29U : 21U) * cloth_detail + 1U;
        cloth.spacing = (hanging ? 0.065F : 0.13F) /
            static_cast<float>(cloth_detail);
        cloth.top_center = make_float3(0.0F,
            course_floor_y + static_cast<float>(cloth.rows - 1U) * cloth.spacing,
            -1.40F);
        SoftBodyAsset asset = make_cloth_grid(cloth);
        // The tensioned target uses both horizontal borders. With the authored
        // rest shape supported at both ends, gravity cannot pre-damage it.
        const std::uint32_t bottom_row = (cloth.rows - 1U) * cloth.columns;
        for (std::uint32_t column = 0U; column < cloth.columns; ++column) {
            asset.voxel_flags[column] |= soft_body_voxel_pinned;
            const std::uint32_t bottom = bottom_row + column;
            asset.voxel_flags[bottom] |= soft_body_voxel_pinned;
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
        return std::make_unique<SoftBodyCourse>(std::move(asset), options);
    }

    if (context == SimulationRecipe::cloth_soft_body) {
        SoftBodyAsset sphere = make_soft_sphere();
        const float spacing = sphere.nominal_spacing;
        sphere = translate_soft_body_asset(std::move(sphere),
            make_float3(0.0F, course_floor_y + 0.55F, -0.35F));
        ClothGridOptions cloth;
        cloth.columns = 24U;
        cloth.rows = 24U;
        cloth.spacing = spacing;
        cloth.top_center = make_float3(0.0F,
            course_floor_y + static_cast<float>(cloth.rows - 1U) * spacing,
            cloth_soft_body_goal_z);
        SoftBodyAsset curtain = make_cloth_grid(cloth);
        for (std::uint32_t column = 0U; column < cloth.columns; ++column) {
            curtain.voxel_flags[column] |= soft_body_voxel_pinned;
            curtain.voxel_flags[(cloth.rows - 1U) * cloth.columns + column] |=
                soft_body_voxel_pinned;
        }
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
        options.fracture_node_first=options.cross_source_nodes;
        // The leading sphere remains unbreakable by range; this multiplier
        // now controls only the cloth's tear threshold.
        options.strength_multiplier=1.5F;
        options.cross_target_triangle_first =
            static_cast<std::uint32_t>(sphere.render_triangles.size());
        options.surface_triangle_split = options.cross_target_triangle_first;
        options.secondary_surface_triangle_split = options.surface_triangle_split +
            static_cast<std::uint32_t>(curtain.render_triangles.size());
        SoftBodyAsset cloth_surfaces = merge_soft_body_assets(curtain, ground_cloth);
        auto result = std::make_unique<SoftBodyCourse>(
            merge_soft_body_assets(sphere, cloth_surfaces), options);
        initialize_recipe_motion(context, *result);
        return result;
    }

    if (context == SimulationRecipe::water_soft_body) {
        // Compact cross arms terminate at the smaller external inertial wheel.
        SoftBodyAsset front = translate_soft_body_asset(
            make_soft_cross(17U, 5U, 0.085F), make_float3(
            water_wheel_center.x, water_wheel_center.y,
            water_wheel_center.z + water_wheel_cross_offset));
        SoftBodyAsset back = translate_soft_body_asset(
            make_soft_cross(17U, 5U, 0.085F), make_float3(
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

    if (context == SimulationRecipe::soft_body_smoke) {
        // Reuse the imported volumetric lattice as a field of short green
        // bristles. Each instance retains its pinned root and all internal
        // bonds, so smoke and the rolling sphere bend actual soft bodies.
        for (float3& point:cylinder.rest_voxels) {
            point.x*=0.13F; point.y*=0.42F; point.z*=0.13F;
        }
        for (float3& point:cylinder.render_positions) {
            point.x*=0.13F; point.y*=0.42F; point.z*=0.13F;
        }
        cylinder.voxel_radius*=0.13F;
        cylinder.nominal_spacing*=0.13F;
        for (SoftBodyEdge& edge:cylinder.edges) {
            const float3 a=cylinder.rest_voxels[edge.vertices.x];
            const float3 b=cylinder.rest_voxels[edge.vertices.y];
            edge.rest_length=std::sqrt((a.x-b.x)*(a.x-b.x)+
                (a.y-b.y)*(a.y-b.y)+(a.z-b.z)*(a.z-b.z));
        }
        constexpr std::uint32_t columns=10U;
        constexpr std::uint32_t rows=6U;
        options.instance_count=columns*rows;
        options.spring_stiffness=9'000.0F;
        options.strength_multiplier=64.0F;
        options.unbonded_voxel_collisions=true;
        for (std::uint32_t row=0U;row<rows;++row)
            for (std::uint32_t column=0U;column<columns;++column)
                options.instance_origins[row*columns+column]=make_float3(
                    -1.35F+0.30F*column,course_floor_y+0.03F,-2.0F+0.30F*row);
        return std::make_unique<SoftBodyCourse>(std::move(cylinder),options);
    }

    if (context == SimulationRecipe::soft_body) {
        cylinder_columns=std::clamp(cylinder_columns,1U,16U);
        cylinder_rows=std::clamp(cylinder_rows,1U,16U);
        while (cylinder_columns*cylinder_rows>SoftBodyOptions::maximum_instances)
            cylinder_rows=std::max(1U,cylinder_rows-1U);
        const float scale_x=0.20F*5.0F/static_cast<float>(cylinder_columns);
        const float scale_z=0.20F*4.0F/static_cast<float>(cylinder_rows);
        // Fit the requested N x M curtain into the same volume. Increasing
        // quantity makes each cylinder thinner instead of expanding the room.
        for (float3& point : cylinder.rest_voxels) {
            point.x *= scale_x;
            point.y *= 0.50F;
            point.z *= scale_z;
        }
        for (float3& point : cylinder.render_positions) {
            point.x *= scale_x;
            point.y *= 0.50F;
            point.z *= scale_z;
        }
        const float radial_scale=std::min(scale_x,scale_z);
        cylinder.voxel_radius *= radial_scale;
        cylinder.nominal_spacing *= radial_scale;
        for (SoftBodyEdge& edge : cylinder.edges) {
            const float3 a = cylinder.rest_voxels[edge.vertices.x];
            const float3 b = cylinder.rest_voxels[edge.vertices.y];
            edge.rest_length = std::sqrt(
                (a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y) +
                (a.z-b.z)*(a.z-b.z));
        }
        options.instance_count = cylinder_columns*cylinder_rows;
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
        for (std::uint32_t row = 0U; row < cylinder_rows; ++row) {
            for (std::uint32_t column = 0U; column < cylinder_columns; ++column) {
                const std::uint32_t instance = row*cylinder_columns+column;
                const float alpha_x=cylinder_columns==1U ? 0.5F :
                    static_cast<float>(column)/static_cast<float>(cylinder_columns-1U);
                const float alpha_z=cylinder_rows==1U ? 0.5F :
                    static_cast<float>(row)/static_cast<float>(cylinder_rows-1U);
                options.instance_origins[instance] = make_float3(
                    -0.88F+1.76F*alpha_x,
                    ceiling - 0.02F - maximum_y,
                    -2.46F+1.32F*alpha_z);
            }
        }
    } else {
        // The authored bottom ring remains pinned in the obstacle course.
        options.instance_origins[0] = make_float3(
            0.0F, course_floor_y + 0.825F, -1.8F);
    }
    return std::make_unique<SoftBodyCourse>(std::move(cylinder), options);
}

FluidDisplay default_recipe_display(parallel_mater::examples::SimulationRecipe context) noexcept
{
    return context == parallel_mater::examples::SimulationRecipe::water_cloth ||
        context == parallel_mater::examples::SimulationRecipe::water ||
        context == parallel_mater::examples::SimulationRecipe::water_rope ||
        context == parallel_mater::examples::SimulationRecipe::water_soft_body ||
        context == parallel_mater::examples::SimulationRecipe::fluid_smoke
        ? FluidDisplay::Surface : FluidDisplay::Particles;
}

RigidSphereState initial_rigid_sphere(
    parallel_mater::examples::SimulationRecipe context, std::uint32_t rope_node_count) noexcept
{
    RigidSphereState sphere;
    sphere.radius = 0.40F;
    sphere.mass = 20.0F;
    if (context == parallel_mater::examples::SimulationRecipe::water) {
        sphere.radius = 0.34F;
        // One particle has unit simulation mass. This exceeds the roughly
        // 2,500 particles displaced by the sphere, so it settles on the bowl
        // instead of floating at the free surface.
        sphere.mass = 3'500.0F;
        sphere.center = make_float3(0.0F, 0.80F, -1.8F);
        sphere.velocity = make_float3(0.35F, -0.15F, 0.0F);
    } else if (context == parallel_mater::examples::SimulationRecipe::cloth) {
        // Start far enough from the tensioned cloth to make the approach
        // legible. Its lower pinned row is buried below the floor rather than
        // protruding into the sphere's path as an invisible curb.
        sphere.center = make_float3(0.0F, 0.18F, 1.45F);
        sphere.velocity = make_float3(0.0F, 0.0F, 0.0F);
        sphere.mass = 250.0F;
    } else if (context == parallel_mater::examples::SimulationRecipe::water_rope) {
        // A sphere remains the conservative fluid/contact proxy while the
        // renderer presents it as a treasure chest.
        sphere.radius = fishing_chest_radius;
        // Heavier than the displaced particle volume, so the chest settles to
        // the tank floor until the rope supplies an upward reaction.
        sphere.mass = 3'500.0F;
        sphere.center = fishing_chest_start;
        sphere.velocity = {};
    } else if (context == parallel_mater::examples::SimulationRecipe::soft_body) {
        sphere.radius = 0.34F;
        sphere.mass = 80.0F;
        sphere.center = make_float3(-2.0F, course_floor_y + sphere.radius, -1.8F);
        sphere.velocity = make_float3(1.8F, 0.0F, 0.0F);
    } else if (context == parallel_mater::examples::SimulationRecipe::water_soft_body) {
        // Start on the right stage and cross over the wheel toward the open
        // left exit. Bumper rails confine lateral motion to the stage.
        sphere.radius = 0.26F;
        sphere.mass = 24.0F;
        sphere.center = make_float3(
            0.5F * (water_wheel_center.x + water_wheel_top_platform_outer_x +
                water_wheel_center.x + water_wheel_top_platform_gap_half_width),
            water_wheel_top_platform_y + sphere.radius,
            water_wheel_stage_z);
        sphere.velocity = {};
    } else if (context == parallel_mater::examples::SimulationRecipe::rope) {
        rope_node_count = std::clamp(rope_node_count, 16U, 512U);
        const std::uint32_t trunk_nodes=std::max(8U,rope_node_count/2U);
        const std::uint32_t branch_nodes=std::max(
            5U,(rope_node_count-trunk_nodes)/2U+1U);
        const float node_radius=0.36F*std::min(
            0.5F*rope_length/static_cast<float>(trunk_nodes-1U),
            0.5F*rope_length/static_cast<float>(branch_nodes-1U));
        const float inverse_direction=1.0F/std::sqrt(1.0F+0.55F*0.55F);
        const float3 endpoint=make_float3(
            rope_anchor.x+
            0.5F*rope_length*(1.0F+inverse_direction),rope_anchor.y,
            rope_anchor.z+0.5F*rope_length*0.55F*inverse_direction);
        const float3 direction=make_float3(
            inverse_direction,0.0F,0.55F*inverse_direction);
        sphere.radius = 0.32F;
        sphere.mass = 18.0F;
        sphere.center=make_float3(
            endpoint.x+direction.x*(sphere.radius+node_radius),
            endpoint.y,
            endpoint.z+direction.z*(sphere.radius+node_radius));
        sphere.velocity = {};
    } else if (context == parallel_mater::examples::SimulationRecipe::cloth_rope ||
               context == parallel_mater::examples::SimulationRecipe::soft_body_rope ||
               context == parallel_mater::examples::SimulationRecipe::rope_smoke) {
        sphere.radius = 0.34F;
        sphere.mass = context==parallel_mater::examples::SimulationRecipe::cloth_rope
            ? 180.0F : 45.0F;
        sphere.center = make_float3(0.0F,
            rope_bridge_land_y + sphere.radius,
            rope_bridge_land_inner_z + 0.65F);
        sphere.velocity = {};
    } else if (context == parallel_mater::examples::SimulationRecipe::smoke) {
        sphere.radius=0.36F;
        sphere.mass=28.0F;
        sphere.center=make_float3(-1.8F,course_floor_y+sphere.radius,-1.2F);
        sphere.velocity=make_float3(1.35F,0.0F,0.0F);
    } else if (context == parallel_mater::examples::SimulationRecipe::soft_body_smoke) {
        sphere.radius=0.30F;
        sphere.mass=45.0F;
        sphere.center=make_float3(-1.75F,course_floor_y+sphere.radius,-1.25F);
        sphere.velocity=make_float3(0.9F,0.0F,0.0F);
    } else if (context == parallel_mater::examples::SimulationRecipe::cloth_smoke) {
        sphere.center=make_float3(3.0F,course_floor_y+sphere.radius,-1.2F);
    } else if (context == parallel_mater::examples::SimulationRecipe::fluid_smoke) {
        sphere.center=make_float3(3.0F,course_floor_y+sphere.radius,-1.2F);
    } else {
        // A grazing track loads the breakable wall while not demanding that a
        // floor-height ball tunnel through the post's immovable foundation.
        sphere.center = make_float3(-2.15F, course_floor_y + sphere.radius, -1.25F);
        sphere.velocity = make_float3(2.0F, 0.0F, 0.0F);
    }
    return sphere;
}

void initialize_recipe_motion(
    parallel_mater::examples::SimulationRecipe context, SoftBodyCourse& body)
{
    // Cloth-Softbody is driven by the same live gravity tilt as every other
    // gallery scene. It starts from rest so translation and rotation come
    // from gravity plus ground contact, not a prescribed launch velocity.
    (void)context;
    (void)body;
}

} // namespace waterlab::gallery
