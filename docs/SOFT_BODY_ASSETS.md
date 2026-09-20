# Soft-body assets

`parallel_mater::physics::SoftBody` consumes the versioned `.msb` fixed-topology
format. The file stores simulation nodes, canonical bonds, reciprocal sorted
neighbor CSR, a triangle surface, UV coordinates, and four-node surface
bindings. Scene placement, colliders, rendering, controls, and gameplay are not
stored in the asset.

The public API accepts either a filesystem path or a `SoftBodyAssetView` over
memory owned by the caller. Initialization validates and copies the complete
payload, so the input bytes may be released when initialization returns.

The checked-in `checker_cylinder.msb` fixture contains 1,000 nodes, 7,704
bonds, 931 surface vertices, and 1,632 surface triangles. It is original
project work and exists to exercise the loader, package examples, and tests.

Regenerate the fixture and its metadata with Blender 4.5 or newer:

```bash
./scripts/generate_softbody_assets.sh
```

Convert another closed GLB:

```bash
blender --background \
  --python tools/blender/softbody_asset_pipeline.py -- \
  convert --input model.glb --output model.msb \
  --voxels 1000 --relax-iterations 24
```

The converter is an offline authoring tool, not part of the installed runtime
API. The binary layout and validation rules are documented in
[`tools/blender/README.md`](../tools/blender/README.md).
