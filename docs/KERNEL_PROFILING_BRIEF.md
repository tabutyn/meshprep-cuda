# Bounded-force kernel brief

The frame entry point is
[`HybridDroplet::step`](../apps/water_lab/hybrid_kernels.cu#L933). All arrays are
device-resident and all launches use one CUDA stream. Force evaluation reads
immutable state; later integration kernels write the next state.

| Kernel or primitive | Inputs | Outputs | Purpose and profiling question | Code |
|---|---|---|---|---|
| `particle_forces_kernel` | Particle positions/velocities, fluid hierarchy, physical/rest skin positions and normals, skin hierarchy, incident-triangle CSR, force constants | One force and three barycentrically weighted skin-reaction rows per particle; owner and neighbor/finite counters | Traverses both hierarchies, evaluates gas-like repulsion and damping, uses the closest vertex to seed a closest local triangle query, and emits the opposite boundary force across that triangle. This is the main physics kernel; inspect branch efficiency, local-memory stack traffic, memory throughput, and occupancy. | [`hybrid_kernels.cu`](../apps/water_lab/hybrid_kernels.cu#L235) |
| `integrate_particles_kernel` | Particle force, position, velocity, inverse mass, `1/(60N)` substep, caps | Updated particle positions and velocities | Bounded semi-implicit Euler. It is a simple bandwidth/control baseline. | [`hybrid_kernels.cu`](../apps/water_lab/hybrid_kernels.cu#L427) |
| `skin_spring_forces_kernel` | Physical skin state and precomputed one-ring CSR/rest lengths | Initial force and purple debug component per skin vertex | Evaluates fixed-topology spring/damping forces. Check whether the short irregular rows coalesce well. | [`hybrid_kernels.cu`](../apps/water_lab/hybrid_kernels.cu#L455) |
| CUB radix sort + reduce-by-key | Three `(triangle vertex, weighted reaction)` rows per particle | Deterministically reduced force per touched skin vertex | Replaces floating-point contact atomics with a fixed reduction order. Inspect temporary-storage traffic and compare its cost with force generation. | [`hybrid_kernels.cu`](../apps/water_lab/hybrid_kernels.cu#L1006) |
| `apply_skin_reactions_kernel` | Reduced vertex IDs/forces and run count | Accumulated skin forces and yellow debug component | Sparse scatter of one already-reduced row per touched vertex. | [`hybrid_kernels.cu`](../apps/water_lab/hybrid_kernels.cu#L486) |
| `skin_rectangle_forces_kernel` | Skin state, dynamic rectangle state, box stiffness/damping and force limits | Skin ejection and blue debug forces; equal/opposite rectangle force and torque rows | Evaluates the exact OBB plus a 2 mm smooth contact shell. Defaults use stiffness `3000`, damping `64`, and per-vertex cap `200`. Inspect divergence during sparse contact. | [`hybrid_kernels.cu`](../apps/water_lab/hybrid_kernels.cu#L504) |
| `integrate_skin_kernel` | Combined spring, particle, and rectangle forces | Updated physical skin positions/velocities | Applies mass, damping, force cap, speed cap, and the fixed timestep. | [`hybrid_kernels.cu`](../apps/water_lab/hybrid_kernels.cu#L587) |
| CUB force/torque reductions | One row per physical skin vertex | Net rectangle force and yaw torque | Deterministically gathers skin contact onto the rigid body. | [`hybrid_kernels.cu`](../apps/water_lab/hybrid_kernels.cu#L1038) |
| `integrate_rectangle_kernel` | Net skin reaction, user force/torque, finite mass/inertia | Rectangle pose and velocity | One-thread rigid-body update. Kernel latency matters more than throughput. | [`hybrid_kernels.cu`](../apps/water_lab/hybrid_kernels.cu#L621) |
| `emit_bounds_kernel` | Current particle or skin-vertex positions and radius | Primitive AABBs | Feeds the two point-compatible hierarchy rebuilds. It should be bandwidth-bound. | [`hybrid_kernels.cu`](../apps/water_lab/hybrid_kernels.cu#L223) |
| `embed_render_surface_kernel` | Coarse skin displacement and fixed barycentric embedding | 20,252 detailed render positions | Linear coarse-to-detailed gather. The following normal and 40,500-triangle hierarchy builds dominate the combined render-prep stage. | [`hybrid_kernels.cu`](../apps/water_lab/hybrid_kernels.cu#L656) |
| `render_kernel` | Detailed skin/hierarchy/normals, optional particle hierarchy, rectangle, camera | RGBA framebuffer | Hierarchy ray traversal, smooth refraction, checkerboard room, and wireframe OBB edge composition. Inspect divergent rays and traversal stack traffic. | [`water_kernels.cu`](../apps/water_lab/water_kernels.cu#L570) |

The hierarchy and normal operations also launch kernels from
[`src/meshprep.cu`](../src/meshprep.cu). Use Nsight Systems first to find their
actual contribution before collecting expensive Compute metrics.
