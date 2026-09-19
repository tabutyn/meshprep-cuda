# Public soft-body performance

This is a narrow package-level baseline, not a claim about the eight gallery
scenes. It measures the public `meshprep::physics::SoftBody` frame protocol and
the custom analytic ground kernel in `examples/soft_body.cu`.

## Configuration

- NVIDIA GeForce RTX 3050 Ti Laptop GPU, compute capability 8.6, 4 GiB;
- driver 590.44.01 and CUDA Toolkit 13.1 (`nvcc` 13.1.80);
- Release build with `-O3` CUDA compilation;
- one generated checker-cylinder asset: 1,000 physical nodes and 1,632 render
  triangles;
- fixed `1/60 s` frame, four substeps, eight graph iterations;
- 10 warmup frames followed by 120 measured frames;
- CUDA-event stage timing reported by the solver; profiler overhead excluded.

## Result

| GPU frame p5 | Median | p95 | Retained allocation | Broken bonds | Non-finite failures |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1.049 ms | 1.327 ms | 1.445 ms | 4.15 MiB | 0 | 0 |

The measured total includes lattice prediction/constraints, surface
deformation and normals, and render/member hierarchy refits. It excludes
application rendering and CPU wall time. This fixture is intentionally small;
instance count, asset topology, solver iterations, and application contact
kernels all change the cost.

Reproduce from the repository root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DMESHPREP_BUILD_EXAMPLES=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --target meshprep-soft-body-example -j
./build/meshprep-soft-body-example assets/softbody/checker_cylinder.msb
```

The executable prints its own p5/median/p95 values and retained allocation so
new measurements do not silently reuse this machine's result.
