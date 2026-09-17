<!-- SPDX-License-Identifier: MIT -->

# Soft-body fracture stability

The saved 245-frame collision is now a headless regression instead of a visual
anecdote. The audit reconstructs every render triangle, then starts a fresh
simulation from the authored cylinder and reapplies the recorded gravity,
controller, material, and water-contact inputs. It never boots from the
exploded final snapshot.

## Five tested hypotheses

1. **One transient stretch caused an irreversible fracture avalanche.** Edge
   damage now has to remain above the break strain for eight consecutive
   substeps, and fracture is tested after graph projection.
2. **A broken bond made the surviving graph overload and collapse.** Jacobi
   corrections use immutable rest degree, each proposal is bounded to 20% of
   voxel spacing, and a broken bond retains a unilateral compression barrier.
3. **The same contact projection became momentum twice.** Position correction
   is converted to velocity once, blended with predicted velocity, and the
   post-spring result is clamped again after damping.
4. **Deep triangle contact sometimes chose an inward or excessively large
   reaction.** Contact normals are oriented from the current bound voxel
   support, the pair uses one inverse-mass-weighted multiplier, and transfer is
   capped to 20% of voxel radius per substep.
5. **Render bindings continued to bridge separated voxel components.** A
   render vertex now follows one deterministic dominant voxel. Two bonded
   candidates define a corotational local frame, so the authored surface offset
   rotates with the material instead of being left behind in world axes. If
   that frame breaks, the vertex falls back to its dominant voxel; every
   authored triangle remains active. Area, edge length, and orientation are
   measurements, never reasons to hide material.

## Recorded result

RTX 3050 Ti Laptop GPU, CUDA 13.1, Release build:

| Metric | Saved failing run | Fresh current solver |
| --- | ---: | ---: |
| Broken bonds | 4,230 | 483 |
| Maximum breaks in one frame | 205 | 39 |
| Supported triangle area range | 0.0646–709.22× | 0.6039–1.3759× |
| Supported triangle p99 area | 1.0000× | 1.0009× |
| Maximum supported edge | 58.12× | 1.2780× |
| Minimum supported surface | 91.94% | 94.75% |
| Maximum voxel speed | 8.57 | 2.74 |
| Maximum contact penetration | 0.07863 m | 0.07863 m |
| Mean fresh replay wall time | — | 10.32 ms/frame |
| Finite-state gate | pass | pass |
| Active triangle-size gate | **fail** | **pass** |
| Material-relative orientation gate | **fail** (3,756 active observations) | **pass** (0 active observations) |

The contact penetration did not improve, so these changes are not a
collision-accuracy improvement. Across every triangle—including detached
fracture seams retained by the audit—the maximum area is `3.8350×`, p99 area is
`1.0073×`, and maximum edge is `4.8477×`. Those unrestricted maxima are no
longer allowed to make the supported gate pass by disappearing: the runtime
activity mask is now derived only from broken support bonds. The 16 remaining
orientation observations all occur after such a bond has broken; supported
triangles record zero inversions. Explicit component-aware fracture topology
would still be better than fixed render bindings for fully detached pieces.

For a direct non-circular comparison, the same current solver with four-way
translation skinning and support-only activity produced all-triangle maximum
area `273.64×`, p99 area `1.0241×`, and maximum edge `35.71×`; its supported
range was `0.0291–21.02×`, supported edge maximum `33.97×`, and it recorded 509
supported inversion observations. Corotational dominant support reduces those
to the final values above without using any quality-dependent culling.

The five changes were evaluated cumulatively against the recorded trajectory;
the table does not assign an independent speedup or causal percentage to any
single change. The focused isolated impact remains about `0.63 ms` and keeps
at least 97.9% of triangles supported during its deliberately tearing case.
That supplemental fixture still reaches a localized `0.337×` minimum area for
0.2% of triangles; its maximum area (`1.988×`) and edge (`1.913×`) stay bounded.
The exact captured regression retains the stricter `[0.5, 2.0]` all-supported
gate and is the release-blocking collision case.

## Filled surfaces and live-lattice regression (2026-09-15)

Each cylinder already contains 1,000 physical nodes: 500 on its surface and
500 in its interior. Normal display now renders the filled checker surface in
every gallery context. `V` shows water particles together with all physical
nodes and only live bonds; detached surface triangles are not drawn as red
tethers. This changes display, not the fracture threshold or solver forces.

A severed-bond fixture separates a tetrahedral graph into two free pieces and
checks 30 frames against free drift. The intact-bond control pulls inward.
GPU render tests check filled-mesh occlusion and disappearance of disconnected
faces in both course and lab rendering. All five focused soft-body, hybrid,
simulation API, runtime, and gallery test suites passed. A clean installed-package
consumer also built and ran. Native filled and lattice views were inspected.

Fracture still removes broken surface support rather than creating new crack-cap
triangles. Interior nodes are physical fill, not a generated solid fracture
surface; exposed cuts can therefore remain open.

## Reproduce

```bash
cmake --build build -j --target \
  meshprep-soft-body-capture-audit meshprep-soft-body-tests
./build/meshprep-soft-body-capture-audit \
  --csv /tmp/meshprep-hybrid-captures/latest-soft-body-audit.csv
./build/meshprep-soft-body-tests
```

The detailed CSV is temporary and can be removed with:

```bash
rm -f /tmp/meshprep-hybrid-captures/latest-soft-body-audit.csv
```
