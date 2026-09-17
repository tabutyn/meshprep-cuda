#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Create and convert deterministic GLB soft-body assets with Blender.

Run through Blender, for example:

  blender --background --python tools/blender/softbody_asset_pipeline.py -- \
    reproduce --glb assets/softbody/checker_cylinder.glb \
    --asset assets/softbody/checker_cylinder.msb

The converter deliberately uses only Blender's bundled Python modules.  Output
coordinates are right-handed, +Y up runtime coordinates: (x, z, -y) from
Blender space.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import tempfile
from typing import Iterable, NamedTuple, Sequence

try:
    import bpy
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
except ImportError as exc:  # pragma: no cover - produces a useful CLI error
    raise SystemExit("Run this script with Blender's Python: blender --background --python ...") from exc


MAGIC = b"MSBODY1\0"
VERSION = 1
ENDIAN_TAG = 0x01020304
HEADER = struct.Struct("<8s12IfI")
VOXEL = struct.Struct("<3fB3x")
EDGE = struct.Struct("<IIf")
CSR_OFFSET = struct.Struct("<I")
NEIGHBOR = struct.Struct("<II")
RENDER_VERTEX = struct.Struct("<3f2f4I4f")
TRIANGLE = struct.Struct("<3I")

FLAG_SURFACE = 1 << 0
FLAG_PINNED = 1 << 1
ASSET_FLAG_Y_UP = 1 << 0
ASSET_FLAG_BINDING_DELTA_SKINNING = 1 << 1


class SourceMesh(NamedTuple):
    positions: list[Vector]
    triangles: list[tuple[int, int, int]]
    uvs: list[tuple[float, float]]


class SoftBodyData(NamedTuple):
    voxels: list[Vector]
    voxel_flags: list[int]
    edges: list[tuple[int, int, float]]
    render_positions: list[Vector]
    render_uvs: list[tuple[float, float]]
    render_triangles: list[tuple[int, int, int]]
    bindings: list[tuple[tuple[int, int, int, int], tuple[float, float, float, float]]]
    nominal_spacing: float
    relax_iterations: int


def blender_arguments() -> list[str]:
    if "--" not in sys.argv:
        return []
    return sys.argv[sys.argv.index("--") + 1 :]


def clean_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for datablocks in (bpy.data.meshes, bpy.data.materials, bpy.data.cameras, bpy.data.lights):
        for datablock in list(datablocks):
            if datablock.users == 0:
                datablocks.remove(datablock)


def look_at(obj: bpy.types.Object, target: Vector) -> None:
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


def create_checker_image(path: Path, size: int = 128, squares: int = 8) -> bpy.types.Image:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = bpy.data.images.new("softbody_checker", width=size, height=size, alpha=True)
    pixels: list[float] = []
    light = (0.88, 0.91, 0.95, 1.0)
    dark = (0.055, 0.075, 0.11, 1.0)
    cell = max(1, size // squares)
    for y in range(size):
        for x in range(size):
            pixels.extend(light if ((x // cell) + (y // cell)) % 2 == 0 else dark)
    image.pixels.foreach_set(pixels)
    image.filepath_raw = str(path.resolve())
    image.file_format = "PNG"
    image.save()
    return image


def create_uv_cylinder(radial_segments: int = 48, height_segments: int = 16) -> bpy.types.Object:
    radius = 0.30
    height = 1.65
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, ...]] = []
    face_uvs: list[list[tuple[float, float]]] = []

    # Shared geometric rings; loop UVs preserve the seam without duplicating geometry.
    for row in range(height_segments + 1):
        z = -0.5 * height + height * row / height_segments
        for column in range(radial_segments):
            angle = 2.0 * math.pi * column / radial_segments
            vertices.append((radius * math.cos(angle), radius * math.sin(angle), z))

    for row in range(height_segments):
        v0 = row / height_segments
        v1 = (row + 1) / height_segments
        for column in range(radial_segments):
            next_column = (column + 1) % radial_segments
            a = row * radial_segments + column
            b = row * radial_segments + next_column
            c = (row + 1) * radial_segments + next_column
            d = (row + 1) * radial_segments + column
            faces.append((a, b, c, d))
            u0 = column / radial_segments
            u1 = (column + 1) / radial_segments
            face_uvs.append([(u0 * 6.0, v0 * 10.0), (u1 * 6.0, v0 * 10.0),
                             (u1 * 6.0, v1 * 10.0), (u0 * 6.0, v1 * 10.0)])

    bottom_center = len(vertices)
    vertices.append((0.0, 0.0, -0.5 * height))
    top_center = len(vertices)
    vertices.append((0.0, 0.0, 0.5 * height))
    for column in range(radial_segments):
        next_column = (column + 1) % radial_segments
        bottom_a = column
        bottom_b = next_column
        top_a = height_segments * radial_segments + column
        top_b = height_segments * radial_segments + next_column
        faces.append((bottom_center, bottom_b, bottom_a))
        faces.append((top_center, top_a, top_b))

        def cap_uv(index: int) -> tuple[float, float]:
            angle = 2.0 * math.pi * index / radial_segments
            return (2.0 + 2.0 * math.cos(angle), 2.0 + 2.0 * math.sin(angle))

        face_uvs.append([(2.0, 2.0), cap_uv(next_column), cap_uv(column)])
        face_uvs.append([(2.0, 2.0), cap_uv(column), cap_uv(next_column)])

    mesh = bpy.data.meshes.new("CheckerCylinderMesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update(calc_edges=True)
    uv_layer = mesh.uv_layers.new(name="UVMap")
    for polygon, polygon_uvs in zip(mesh.polygons, face_uvs):
        for loop_index, uv in zip(polygon.loop_indices, polygon_uvs):
            uv_layer.data[loop_index].uv = uv
    for polygon in mesh.polygons:
        polygon.use_smooth = polygon.loop_total == 4

    obj = bpy.data.objects.new("CheckerSoftBodyCylinder", mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    return obj


def create_example_glb(glb_path: Path, texture_path: Path, preview_path: Path | None) -> None:
    clean_scene()
    obj = create_uv_cylinder()
    image = create_checker_image(texture_path)
    material = bpy.data.materials.new("CheckerSoftBodyMaterial")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    texture = nodes.new("ShaderNodeTexImage")
    texture.image = image
    texture.interpolation = "Closest"
    links.new(texture.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.48
    bsdf.inputs["Metallic"].default_value = 0.0
    obj.data.materials.append(material)

    glb_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.export_scene.gltf(
        filepath=str(glb_path.resolve()),
        export_format="GLB",
        use_selection=True,
        export_texcoords=True,
        export_normals=True,
        export_materials="EXPORT",
        export_yup=True,
    )

    if preview_path is not None:
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.mesh.primitive_plane_add(size=5.0, location=(0.0, 0.0, -0.835))
        plane = bpy.context.object
        floor_material = bpy.data.materials.new("PreviewFloor")
        floor_material.diffuse_color = (0.025, 0.03, 0.045, 1.0)
        plane.data.materials.append(floor_material)

        bpy.ops.object.camera_add(location=(2.25, -2.5, 1.55))
        camera = bpy.context.object
        camera.data.lens = 55
        look_at(camera, Vector((0.0, 0.0, 0.0)))
        bpy.context.scene.camera = camera
        bpy.ops.object.light_add(type="AREA", location=(1.8, -1.5, 2.7))
        key = bpy.context.object
        key.data.energy = 900.0
        key.data.shape = "DISK"
        key.data.size = 3.0
        look_at(key, Vector((0.0, 0.0, 0.0)))
        bpy.ops.object.light_add(type="AREA", location=(-1.4, 0.8, 0.5))
        fill = bpy.context.object
        fill.data.energy = 500.0
        fill.data.size = 2.0
        look_at(fill, Vector((0.0, 0.0, 0.0)))

        scene = bpy.context.scene
        scene.render.engine = "BLENDER_EEVEE_NEXT"
        scene.render.resolution_x = 512
        scene.render.resolution_y = 512
        scene.render.resolution_percentage = 100
        scene.render.image_settings.file_format = "PNG"
        scene.render.filepath = str(preview_path.resolve())
        scene.render.film_transparent = False
        scene.world.color = (0.008, 0.012, 0.022)
        bpy.ops.render.render(write_still=True)


def triangle_area(a: Vector, b: Vector, c: Vector) -> float:
    return 0.5 * (b - a).cross(c - a).length


def signed_volume(positions: Sequence[Vector], triangles: Sequence[tuple[int, int, int]]) -> float:
    return sum(positions[a].dot(positions[b].cross(positions[c])) for a, b, c in triangles) / 6.0


def import_source_mesh(glb_path: Path) -> SourceMesh:
    clean_scene()
    bpy.ops.import_scene.gltf(filepath=str(glb_path.resolve()))
    objects = sorted((obj for obj in bpy.context.scene.objects if obj.type == "MESH"), key=lambda obj: obj.name)
    if not objects:
        raise ValueError(f"{glb_path} contains no mesh objects")

    positions: list[Vector] = []
    uvs: list[tuple[float, float]] = []
    triangles: list[tuple[int, int, int]] = []
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for obj in objects:
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
        try:
            mesh.calc_loop_triangles()
            uv_layer = mesh.uv_layers.active
            split_vertices: dict[tuple[int, int, int], int] = {}
            for loop_triangle in mesh.loop_triangles:
                result_triangle: list[int] = []
                for loop_index in loop_triangle.loops:
                    vertex_index = mesh.loops[loop_index].vertex_index
                    if uv_layer is None:
                        uv = (0.0, 0.0)
                    else:
                        loop_uv = uv_layer.data[loop_index].uv
                        uv = (float(loop_uv.x), float(loop_uv.y))
                    key = (vertex_index, round(uv[0] * 1_000_000), round(uv[1] * 1_000_000))
                    output_index = split_vertices.get(key)
                    if output_index is None:
                        output_index = len(positions)
                        split_vertices[key] = output_index
                        positions.append(obj.matrix_world @ mesh.vertices[vertex_index].co)
                        uvs.append(uv)
                    result_triangle.append(output_index)
                triangles.append(tuple(result_triangle))
        finally:
            evaluated.to_mesh_clear()

    volume = signed_volume(positions, triangles)
    if abs(volume) < 1.0e-9:
        raise ValueError("source mesh has zero signed volume; input must be a closed solid")
    if volume < 0.0:
        triangles = [(a, c, b) for a, b, c in triangles]
    return SourceMesh(positions, triangles, uvs)


def make_bvh(source: SourceMesh) -> BVHTree:
    return BVHTree.FromPolygons(source.positions, source.triangles, all_triangles=True)


def bounds(points: Sequence[Vector]) -> tuple[Vector, Vector]:
    return (
        Vector(tuple(min(point[axis] for point in points) for axis in range(3))),
        Vector(tuple(max(point[axis] for point in points) for axis in range(3))),
    )


def is_inside(point: Vector, bvh: BVHTree, diagonal: float) -> bool:
    # Parity ray with a non-axis direction avoids alignment with authored grids.
    direction = Vector((1.0, 0.371390676, 0.527046277)).normalized()
    origin = point.copy()
    intersections = 0
    epsilon = max(diagonal * 1.0e-7, 1.0e-8)
    remaining = diagonal * 4.0
    for _ in range(128):
        hit, _normal, _index, distance = bvh.ray_cast(origin, direction, remaining)
        if hit is None:
            break
        intersections += 1
        advance = distance + epsilon
        origin += direction * advance
        remaining -= advance
        if remaining <= 0.0:
            break
    return (intersections & 1) == 1


def area_sample_candidates(source: SourceMesh, candidate_count: int) -> list[Vector]:
    areas = [triangle_area(source.positions[a], source.positions[b], source.positions[c])
             for a, b, c in source.triangles]
    cumulative: list[float] = []
    running = 0.0
    for area in areas:
        running += area
        cumulative.append(running)
    if running <= 0.0:
        raise ValueError("source has no positive-area triangles")

    candidates = [point.copy() for point in source.positions]
    for sample in range(candidate_count):
        area_coordinate = (sample + 0.5) * running / candidate_count
        triangle_index = min(bisect.bisect_left(cumulative, area_coordinate), len(source.triangles) - 1)
        a_index, b_index, c_index = source.triangles[triangle_index]
        a, b, c = source.positions[a_index], source.positions[b_index], source.positions[c_index]
        r1 = ((sample + 1) * 0.7548776662466927) % 1.0
        r2 = ((sample + 1) * 0.5698402909980532) % 1.0
        root = math.sqrt(r1)
        wa = 1.0 - root
        wb = root * (1.0 - r2)
        wc = root * r2
        candidates.append(a * wa + b * wb + c * wc)

    unique: dict[tuple[int, int, int], Vector] = {}
    for point in candidates:
        key = tuple(round(component * 10_000_000) for component in point)
        unique.setdefault(key, point)
    return [unique[key] for key in sorted(unique)]


def farthest_sample(candidates: Sequence[Vector], count: int, seeds: Sequence[Vector] = ()) -> list[Vector]:
    if count > len(candidates):
        raise ValueError(f"requested {count} samples from only {len(candidates)} candidates")
    chosen: list[Vector] = []
    selected = [False] * len(candidates)
    if seeds:
        min_distance2 = [min((point - seed).length_squared for seed in seeds) for point in candidates]
    else:
        centroid = sum(candidates, Vector()) / len(candidates)
        min_distance2 = [(point - centroid).length_squared for point in candidates]

    for _ in range(count):
        best = max((index for index in range(len(candidates)) if not selected[index]),
                   key=lambda index: (min_distance2[index], -index))
        selected[best] = True
        point = candidates[best].copy()
        chosen.append(point)
        for index, candidate in enumerate(candidates):
            if not selected[index]:
                min_distance2[index] = min(min_distance2[index], (candidate - point).length_squared)
    return chosen


def grid_candidates(source: SourceMesh, bvh: BVHTree, nominal_spacing: float) -> list[Vector]:
    minimum, maximum = bounds(source.positions)
    diagonal = (maximum - minimum).length
    spacing = nominal_spacing * 0.52
    counts = [max(2, int(math.ceil((maximum[axis] - minimum[axis]) / spacing))) for axis in range(3)]
    actual = [(maximum[axis] - minimum[axis]) / counts[axis] for axis in range(3)]
    candidates: list[Vector] = []
    for z in range(counts[2]):
        for y in range(counts[1]):
            for x in range(counts[0]):
                point = Vector((minimum[0] + (x + 0.5) * actual[0],
                                minimum[1] + (y + 0.5) * actual[1],
                                minimum[2] + (z + 0.5) * actual[2]))
                if is_inside(point, bvh, diagonal):
                    nearest, _normal, _face, _distance = bvh.find_nearest(point)
                    if nearest is not None and (point - nearest).length >= nominal_spacing * 0.28:
                        candidates.append(point)
    return candidates


def relax_interior(points: list[Vector], surface_count: int, bvh: BVHTree,
                   nominal_spacing: float, iterations: int) -> None:
    target = nominal_spacing * 0.78
    cell_size = target
    margin = nominal_spacing * 0.18
    for iteration in range(iterations):
        cells: dict[tuple[int, int, int], list[int]] = {}
        for index, point in enumerate(points):
            key = tuple(math.floor(component / cell_size) for component in point)
            cells.setdefault(key, []).append(index)
        corrections = [Vector() for _ in points]
        phase = 0.35 + 0.65 * (1.0 - iteration / max(1, iterations))
        for index in range(surface_count, len(points)):
            point = points[index]
            base = tuple(math.floor(component / cell_size) for component in point)
            correction = Vector()
            for dz in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        for other_index in cells.get((base[0] + dx, base[1] + dy, base[2] + dz), ()):
                            if other_index == index:
                                continue
                            delta = point - points[other_index]
                            distance = delta.length
                            if 1.0e-9 < distance < target:
                                correction += delta * ((target - distance) / (distance * target))
                            elif distance <= 1.0e-9:
                                axis = (index + other_index) % 3
                                correction[axis] += 1.0
            corrections[index] = correction * (nominal_spacing * 0.065 * phase)

        for index in range(surface_count, len(points)):
            correction = corrections[index]
            if correction.length > nominal_spacing * 0.16:
                correction.normalize()
                correction *= nominal_spacing * 0.16
            candidate = points[index] + correction
            nearest, normal, _face, _distance = bvh.find_nearest(candidate)
            if nearest is None:
                continue
            signed_distance = (candidate - nearest).dot(normal)
            if signed_distance > -margin:
                candidate = nearest - normal * margin
            points[index] = candidate


def build_edges(points: Sequence[Vector], nominal_spacing: float, neighbors: int = 14) -> list[tuple[int, int, float]]:
    edges: set[tuple[int, int]] = set()
    radius2 = (nominal_spacing * 2.25) ** 2
    for index, point in enumerate(points):
        distances = sorted(((point - other).length_squared, other_index)
                           for other_index, other in enumerate(points) if other_index != index)
        selected = [(distance2, other_index) for distance2, other_index in distances if distance2 <= radius2][:neighbors]
        if len(selected) < 6:
            selected = distances[:6]
        for _distance2, other_index in selected:
            edges.add((min(index, other_index), max(index, other_index)))

    # Ensure one connected graph.  Components are linked by their nearest pair.
    parent = list(range(len(points)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a: int, b: int) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for a, b in edges:
        union(a, b)
    while len({find(index) for index in range(len(points))}) > 1:
        components: dict[int, list[int]] = {}
        for index in range(len(points)):
            components.setdefault(find(index), []).append(index)
        roots = sorted(components)
        root_a = roots[0]
        best: tuple[float, int, int] | None = None
        for a in components[root_a]:
            for root_b in roots[1:]:
                for b in components[root_b]:
                    candidate = ((points[a] - points[b]).length_squared, min(a, b), max(a, b))
                    if best is None or candidate < best:
                        best = candidate
        assert best is not None
        _distance2, a, b = best
        edges.add((a, b))
        union(a, b)

    return [(a, b, (points[a] - points[b]).length) for a, b in sorted(edges)]


def build_bindings(render_positions: Sequence[Vector], voxels: Sequence[Vector],
                   surface_count: int) -> list[tuple[tuple[int, int, int, int], tuple[float, float, float, float]]]:
    bindings = []
    for render_position in render_positions:
        nearest = sorted(((render_position - voxels[index]).length_squared, index)
                         for index in range(surface_count))[:4]
        if nearest[0][0] <= 1.0e-16:
            indices = (nearest[0][1], nearest[1][1], nearest[2][1], nearest[3][1])
            weights = (1.0, 0.0, 0.0, 0.0)
        else:
            raw = [1.0 / max(distance2, 1.0e-12) for distance2, _index in nearest]
            total = sum(raw)
            indices = tuple(index for _distance2, index in nearest)
            weights = tuple(value / total for value in raw)
        bindings.append((indices, weights))
    return bindings


def make_soft_body(source: SourceMesh, voxel_count: int, relax_iterations: int) -> SoftBodyData:
    if voxel_count != 1000:
        raise ValueError("this course asset contract requires exactly 1000 voxels")
    bvh = make_bvh(source)
    volume = abs(signed_volume(source.positions, source.triangles))
    surface_area = sum(triangle_area(source.positions[a], source.positions[b], source.positions[c])
                       for a, b, c in source.triangles)
    nominal_spacing = (volume / voxel_count) ** (1.0 / 3.0) * 0.92
    estimated_surface = round(surface_area / max(nominal_spacing * nominal_spacing * 1.35, 1.0e-12))
    surface_count = max(192, min(voxel_count // 2, estimated_surface))

    surface_candidates = area_sample_candidates(source, max(surface_count * 5, 2500))
    surface_points = farthest_sample(surface_candidates, surface_count)
    interior_candidates = grid_candidates(source, bvh, nominal_spacing)
    interior_count = voxel_count - surface_count
    if len(interior_candidates) < interior_count:
        raise ValueError(f"voxel grid produced {len(interior_candidates)} interior candidates; need {interior_count}")
    interior_points = farthest_sample(interior_candidates, interior_count, surface_points)
    voxels = surface_points + interior_points
    relax_interior(voxels, surface_count, bvh, nominal_spacing, relax_iterations)

    minimum, maximum = bounds(source.positions)
    # Hold the entire physical foundation band, including interior nodes.
    # Connections above the band remain ordinary breakable spring bonds.
    pin_threshold = minimum.z + nominal_spacing * 1.25
    pinned = {index for index in range(voxel_count)
              if voxels[index].z <= pin_threshold}
    minimum_pins = max(12, surface_count // 32)
    if len(pinned) < minimum_pins:
        pinned = set(sorted(range(voxel_count),
                            key=lambda index: (voxels[index].z, index))[:minimum_pins])
    flags = [(FLAG_SURFACE if index < surface_count else 0) |
             (FLAG_PINNED if index in pinned else 0)
             for index in range(voxel_count)]
    edges = build_edges(voxels, nominal_spacing)
    bindings = build_bindings(source.positions, voxels, surface_count)
    return SoftBodyData(voxels, flags, edges, source.positions, source.uvs,
                        source.triangles, bindings, nominal_spacing, relax_iterations)


def runtime_vector(point: Vector) -> tuple[float, float, float]:
    return (float(point.x), float(point.z), float(-point.y))


def write_asset(path: Path, data: SoftBodyData, source_path: Path | None, write_manifest: bool = True) -> None:
    surface_count = sum(bool(flags & FLAG_SURFACE) for flags in data.voxel_flags)
    pinned_count = sum(bool(flags & FLAG_PINNED) for flags in data.voxel_flags)
    adjacency: list[list[tuple[int, int]]] = [[] for _ in data.voxels]
    for edge_id, (a, b, _rest_length) in enumerate(data.edges):
        adjacency[a].append((b, edge_id))
        adjacency[b].append((a, edge_id))
    for row in adjacency:
        row.sort()
    directed_neighbor_count = sum(map(len, adjacency))
    header = HEADER.pack(
        MAGIC,
        VERSION,
        ENDIAN_TAG,
        HEADER.size,
        len(data.voxels),
        len(data.edges),
        directed_neighbor_count,
        len(data.render_positions),
        len(data.render_triangles),
        surface_count,
        pinned_count,
        data.relax_iterations,
        ASSET_FLAG_Y_UP | ASSET_FLAG_BINDING_DELTA_SKINNING,
        data.nominal_spacing,
        0,
    )
    chunks = [header]
    for point, flags in zip(data.voxels, data.voxel_flags):
        chunks.append(VOXEL.pack(*runtime_vector(point), flags))
    for a, b, rest_length in data.edges:
        chunks.append(EDGE.pack(a, b, rest_length))
    running_offset = 0
    chunks.append(CSR_OFFSET.pack(0))
    for row in adjacency:
        running_offset += len(row)
        chunks.append(CSR_OFFSET.pack(running_offset))
    for row in adjacency:
        for neighbor, edge_id in row:
            chunks.append(NEIGHBOR.pack(neighbor, edge_id))
    for point, uv, binding in zip(data.render_positions, data.render_uvs, data.bindings):
        indices, weights = binding
        chunks.append(RENDER_VERTEX.pack(*runtime_vector(point), *uv, *indices, *weights))
    for triangle in data.render_triangles:
        # Runtime transform (x,z,-y) reverses handedness, so reverse winding.
        chunks.append(TRIANGLE.pack(triangle[0], triangle[2], triangle[1]))
    payload = b"".join(chunks)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)

    if write_manifest:
        voxel_runtime = [runtime_vector(point) for point in data.voxels]
        manifest = {
            "schema": "meshprep.softbody-asset.v1",
            "binary": path.name,
            "binarySha256": hashlib.sha256(payload).hexdigest(),
            "source": source_path.name if source_path is not None else None,
            "sourceSha256": hashlib.sha256(source_path.read_bytes()).hexdigest() if source_path is not None else None,
            "coordinates": "right-handed +Y up; Blender (x,y,z) maps to runtime (x,z,-y)",
            "voxelCount": len(data.voxels),
            "surfaceVoxelCount": surface_count,
            "pinnedVoxelCount": pinned_count,
            "edgeCount": len(data.edges),
            "directedNeighborCount": directed_neighbor_count,
            "renderVertexCount": len(data.render_positions),
            "renderTriangleCount": len(data.render_triangles),
            "relaxIterations": data.relax_iterations,
            "nominalSpacing": data.nominal_spacing,
            "bounds": {
                "minimum": [min(point[axis] for point in voxel_runtime) for axis in range(3)],
                "maximum": [max(point[axis] for point in voxel_runtime) for axis in range(3)],
            },
        }
        manifest_path = path.with_suffix(path.suffix + ".json")
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def convert(glb_path: Path, asset_path: Path, voxel_count: int, relax_iterations: int,
            write_manifest: bool = True) -> SoftBodyData:
    source = import_source_mesh(glb_path)
    data = make_soft_body(source, voxel_count, relax_iterations)
    write_asset(asset_path, data, glb_path, write_manifest)
    return data


def read_asset(path: Path) -> dict:
    payload = path.read_bytes()
    if len(payload) < HEADER.size:
        raise ValueError("asset is shorter than its header")
    unpacked = HEADER.unpack_from(payload)
    (magic, version, endian, header_bytes, voxel_count, edge_count, directed_neighbor_count, render_count,
     triangle_count, surface_count, pinned_count, relax_iterations, flags,
     nominal_spacing, reserved) = unpacked
    if magic != MAGIC or version != VERSION or endian != ENDIAN_TAG or header_bytes != HEADER.size:
        raise ValueError("invalid soft-body asset header")
    if reserved != 0:
        raise ValueError("reserved header values must be zero")
    expected_bytes = (HEADER.size + voxel_count * VOXEL.size + edge_count * EDGE.size +
                      (voxel_count + 1) * CSR_OFFSET.size + directed_neighbor_count * NEIGHBOR.size +
                      render_count * RENDER_VERTEX.size + triangle_count * TRIANGLE.size)
    if len(payload) != expected_bytes:
        raise ValueError(f"asset size is {len(payload)} bytes; expected {expected_bytes}")

    offset = HEADER.size
    voxels = []
    for _ in range(voxel_count):
        x, y, z, voxel_flags = VOXEL.unpack_from(payload, offset)
        offset += VOXEL.size
        voxels.append(((x, y, z), voxel_flags))
    edges = []
    for _ in range(edge_count):
        edges.append(EDGE.unpack_from(payload, offset))
        offset += EDGE.size
    csr_offsets = []
    for _ in range(voxel_count + 1):
        csr_offsets.append(CSR_OFFSET.unpack_from(payload, offset)[0])
        offset += CSR_OFFSET.size
    neighbors = []
    for _ in range(directed_neighbor_count):
        neighbors.append(NEIGHBOR.unpack_from(payload, offset))
        offset += NEIGHBOR.size
    render_vertices = []
    for _ in range(render_count):
        render_vertices.append(RENDER_VERTEX.unpack_from(payload, offset))
        offset += RENDER_VERTEX.size
    triangles = []
    for _ in range(triangle_count):
        triangles.append(TRIANGLE.unpack_from(payload, offset))
        offset += TRIANGLE.size
    return {
        "payload": payload,
        "flags": flags,
        "nominal_spacing": nominal_spacing,
        "relax_iterations": relax_iterations,
        "surface_count": surface_count,
        "pinned_count": pinned_count,
        "voxels": voxels,
        "edges": edges,
        "csr_offsets": csr_offsets,
        "neighbors": neighbors,
        "render_vertices": render_vertices,
        "triangles": triangles,
    }


def validate(path: Path, require_1000: bool = True) -> dict:
    data = read_asset(path)
    voxels = data["voxels"]
    edges = data["edges"]
    csr_offsets = data["csr_offsets"]
    neighbors = data["neighbors"]
    render_vertices = data["render_vertices"]
    triangles = data["triangles"]
    if require_1000 and len(voxels) != 1000:
        raise ValueError(f"expected exactly 1000 voxels, got {len(voxels)}")
    if not (data["flags"] & ASSET_FLAG_Y_UP and data["flags"] & ASSET_FLAG_BINDING_DELTA_SKINNING):
        raise ValueError("asset is missing required coordinate or binding flags")

    surface_ids = {index for index, (_point, flags) in enumerate(voxels) if flags & FLAG_SURFACE}
    pinned_ids = {index for index, (_point, flags) in enumerate(voxels) if flags & FLAG_PINNED}
    if len(surface_ids) != data["surface_count"] or len(pinned_ids) != data["pinned_count"]:
        raise ValueError("header surface/pinned counts disagree with voxel flags")
    if not pinned_ids:
        raise ValueError("mounted posts need a non-empty pinned foundation")
    for point, flags in voxels:
        if flags & ~(FLAG_SURFACE | FLAG_PINNED):
            raise ValueError("voxel uses unknown flags")
        if not all(math.isfinite(value) for value in point):
            raise ValueError("voxel contains a non-finite coordinate")

    seen_edges: set[tuple[int, int]] = set()
    adjacency = [[] for _ in voxels]
    maximum_rest_error = 0.0
    for a, b, rest_length in edges:
        if not (a < b < len(voxels)) or (a, b) in seen_edges:
            raise ValueError("edge IDs must be canonical, valid, and unique")
        if not math.isfinite(rest_length) or rest_length <= 0.0:
            raise ValueError("edge rest length must be finite and positive")
        seen_edges.add((a, b))
        adjacency[a].append(b)
        adjacency[b].append(a)
        pa, pb = voxels[a][0], voxels[b][0]
        measured = math.sqrt(sum((pa[axis] - pb[axis]) ** 2 for axis in range(3)))
        maximum_rest_error = max(maximum_rest_error, abs(measured - rest_length))
    if maximum_rest_error > 2.0e-6:
        raise ValueError(f"edge rest length error {maximum_rest_error} is too large")
    reached = {0}
    stack = [0]
    while stack:
        current = stack.pop()
        for neighbor in adjacency[current]:
            if neighbor not in reached:
                reached.add(neighbor)
                stack.append(neighbor)
    if len(reached) != len(voxels):
        raise ValueError("voxel neighbor graph is disconnected")
    if csr_offsets[0] != 0 or csr_offsets[-1] != len(neighbors):
        raise ValueError("CSR offset endpoints do not cover the neighbor records")
    if any(csr_offsets[index] > csr_offsets[index + 1] for index in range(len(voxels))):
        raise ValueError("CSR offsets must be monotonic")
    for voxel_id in range(len(voxels)):
        row = neighbors[csr_offsets[voxel_id] : csr_offsets[voxel_id + 1]]
        if list(row) != sorted(row):
            raise ValueError("CSR rows must be sorted by neighbor ID and edge ID")
        if len({neighbor for neighbor, _edge_id in row}) != len(row):
            raise ValueError("CSR row contains duplicate neighbors")
        expected_row = sorted(adjacency[voxel_id])
        if len(row) != len(expected_row):
            raise ValueError("CSR row degree disagrees with the undirected edge table")
        for (neighbor, edge_id), expected_neighbor in zip(row, expected_row):
            if neighbor != expected_neighbor or edge_id >= len(edges):
                raise ValueError("CSR neighbor or edge ID is invalid")
            edge_a, edge_b, _rest_length = edges[edge_id]
            if (min(voxel_id, neighbor), max(voxel_id, neighbor)) != (edge_a, edge_b):
                raise ValueError("CSR edge ID does not name its undirected edge")

    for render in render_vertices:
        position_uv = render[:5]
        indices = render[5:9]
        weights = render[9:13]
        if not all(math.isfinite(value) for value in position_uv + weights):
            raise ValueError("render vertex contains non-finite data")
        if any(index not in surface_ids for index in indices):
            raise ValueError("render vertex binding references a non-surface voxel")
        if any(weight < 0.0 for weight in weights) or abs(sum(weights) - 1.0) > 2.0e-5:
            raise ValueError("render vertex weights must be nonnegative and sum to one")
    minimum_triangle_area = math.inf
    for triangle in triangles:
        if len(set(triangle)) != 3 or any(index >= len(render_vertices) for index in triangle):
            raise ValueError("render triangle has invalid indices")
        points = [Vector(render_vertices[index][:3]) for index in triangle]
        area = triangle_area(*points)
        if area <= 1.0e-9:
            raise ValueError("render triangle is degenerate")
        minimum_triangle_area = min(minimum_triangle_area, area)

    minimum_voxel_distance = math.inf
    for a in range(len(voxels)):
        for b in range(a + 1, len(voxels)):
            pa, pb = voxels[a][0], voxels[b][0]
            distance = math.sqrt(sum((pa[axis] - pb[axis]) ** 2 for axis in range(3)))
            minimum_voxel_distance = min(minimum_voxel_distance, distance)
    if minimum_voxel_distance < data["nominal_spacing"] * 0.15:
        raise ValueError("voxel relaxation left an unreasonably close pair")

    result = {
        "asset": str(path),
        "sha256": hashlib.sha256(data["payload"]).hexdigest(),
        "bytes": len(data["payload"]),
        "voxels": len(voxels),
        "surfaceVoxels": len(surface_ids),
        "pinnedVoxels": len(pinned_ids),
        "edges": len(edges),
        "directedNeighbors": len(neighbors),
        "minimumDegree": min(map(len, adjacency)),
        "maximumDegree": max(map(len, adjacency)),
        "renderVertices": len(render_vertices),
        "renderTriangles": len(triangles),
        "nominalSpacing": data["nominal_spacing"],
        "minimumVoxelDistance": minimum_voxel_distance,
        "minimumTriangleArea": minimum_triangle_area,
        "maximumRestLengthError": maximum_rest_error,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_arguments(arguments: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    example = subparsers.add_parser("create-example", help="create a textured cylinder GLB")
    example.add_argument("--glb", type=Path, required=True)
    example.add_argument("--texture", type=Path, required=True)
    example.add_argument("--preview", type=Path)

    convert_parser = subparsers.add_parser("convert", help="convert a closed GLB to .msb")
    convert_parser.add_argument("--input", type=Path, required=True)
    convert_parser.add_argument("--output", type=Path, required=True)
    convert_parser.add_argument("--voxels", type=int, default=1000)
    convert_parser.add_argument("--relax-iterations", type=int, default=24)

    validate_parser = subparsers.add_parser("validate", help="validate a .msb asset")
    validate_parser.add_argument("--input", type=Path, required=True)

    reproduce = subparsers.add_parser("reproduce", help="create, convert, validate, and check determinism")
    reproduce.add_argument("--glb", type=Path, required=True)
    reproduce.add_argument("--asset", type=Path, required=True)
    reproduce.add_argument("--texture", type=Path)
    reproduce.add_argument("--preview", type=Path)
    reproduce.add_argument("--relax-iterations", type=int, default=24)
    return parser.parse_args(arguments)


def main() -> None:
    arguments = parse_arguments(blender_arguments())
    if arguments.command == "create-example":
        create_example_glb(arguments.glb, arguments.texture, arguments.preview)
    elif arguments.command == "convert":
        convert(arguments.input, arguments.output, arguments.voxels, arguments.relax_iterations)
        validate(arguments.output)
    elif arguments.command == "validate":
        validate(arguments.input)
    elif arguments.command == "reproduce":
        texture = arguments.texture or arguments.glb.with_name(arguments.glb.stem + "_checker.png")
        create_example_glb(arguments.glb, texture, arguments.preview)
        convert(arguments.glb, arguments.asset, 1000, arguments.relax_iterations)
        first = validate(arguments.asset)
        with tempfile.TemporaryDirectory(prefix="meshprep-softbody-") as temporary_directory:
            repeated = Path(temporary_directory) / arguments.asset.name
            convert(arguments.glb, repeated, 1000, arguments.relax_iterations, write_manifest=False)
            second_hash = hashlib.sha256(repeated.read_bytes()).hexdigest()
        if first["sha256"] != second_hash:
            raise RuntimeError("determinism check failed: repeated conversion changed the asset bytes")
        print(f"DETERMINISM PASS {second_hash}")


if __name__ == "__main__":
    main()
