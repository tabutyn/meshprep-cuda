# Blender soft-body asset pipeline

`softbody_asset_pipeline.py` turns a closed GLB render mesh into a deterministic
voxel soft body. The committed example is an upright checkerboard cylinder for
the obstacle-course posts. The script, procedural cylinder, checker texture,
converted runtime data, and previews in this repository are project-owned and
distributed under the repository's MIT license.

Regenerate and validate everything with Blender 4.5 or newer:

```bash
./scripts/generate_softbody_assets.sh
```

The command builds the textured GLB, converts it twice, requires byte-identical
results, validates all indices and graph invariants, and writes:

- `assets/softbody/checker_cylinder.glb`: UV-mapped source with an embedded
  checker texture.
- `assets/softbody/checker_cylinder.msb`: runtime soft-body data.
- `assets/softbody/checker_cylinder.msb.json`: source/output hashes and counts.
- `assets/softbody/checker_cylinder_checker.png`: source texture.
- `assets/softbody/checker_cylinder_preview.png`: Blender Form-gate preview.

Convert another watertight GLB without rebuilding the example:

```bash
/usr/local/bin/blender --background \
  --python tools/blender/softbody_asset_pipeline.py -- \
  convert --input model.glb --output model.msb \
  --voxels 1000 --relax-iterations 24
```

The v1 course contract intentionally requires exactly 1,000 voxels per asset.
The converter samples the authored surface, fills the interior from a fixed
grid, then performs deterministic short-range repulsion while keeping surface
voxels fixed. Bottom surface voxels are marked pinned. It constructs a connected
static neighbor graph; each undirected edge has a stable ID and rest length so
runtime can break it without changing adjacency storage.

## `.msb` v1 binary format

All values are little-endian. Output is right-handed and +Y up; Blender
`(x, y, z)` maps to runtime `(x, z, -y)`.

The 64-byte header is packed as `<8s12IfI>`:

| Field | Type |
|---|---|
| magic (`MSBODY1\0`) | `char[8]` |
| version (`1`) | `uint32` |
| endian tag (`0x01020304`) | `uint32` |
| header bytes (`64`) | `uint32` |
| voxel count | `uint32` |
| undirected edge count | `uint32` |
| directed neighbor count | `uint32` |
| render vertex count | `uint32` |
| render triangle count | `uint32` |
| surface voxel count | `uint32` |
| pinned voxel count | `uint32` |
| relaxation iterations | `uint32` |
| asset flags | `uint32` |
| nominal voxel spacing | `float32` |
| reserved (`0`) | `uint32` |

Payload sections follow in this exact order:

1. `voxel[voxelCount]`, `<3fB3x>`: rest position and flags (`surface=1`,
   `pinned=2`).
2. `edge[edgeCount]`, `<IIf>`: canonical endpoints `a < b` and rest length.
   Array index is the shared edge ID.
3. `csrOffset[voxelCount + 1]`, `<I>`.
4. `neighbor[directedNeighborCount]`, `<II>`: neighbor voxel and shared edge
   ID. Each row is sorted by neighbor ID.
5. `renderVertex[renderVertexCount]`, `<3f2f4I4f>`: rest position, UV, four
   surface-voxel IDs, and four normalized weights.
6. `triangle[triangleCount]`, `<3I>`: CCW runtime render indices.

The file retains four normalized candidate bindings so contact can distribute
reaction impulses over nearby surface voxels. Runtime rendering selects the
strongest candidate as one material anchor and two bonded candidates as a local
corotational frame. This preserves the authored GLB at rest without blending a
render vertex across separated fracture components:

```text
render_position = current_anchor
                + current_frame * inverse(rest_frame)
                  * (render_rest - rest_anchor)
```

The validator requires a connected graph, canonical unique edges, reciprocal
CSR entries, surface-only normalized bindings, finite data, nondegenerate
triangles, valid rest lengths, and no abnormally close voxel pair.
