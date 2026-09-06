# Benchmarks

`meshprep-benchmark` accepts a triangulated OBJ or generates a deterministic grid. Mesh upload and OBJ parsing occur before timing. Warmups retain all output and workspace allocations.

```bash
./build/meshprep-benchmark --obj benchmarks/data/mesh/sibenik.obj \
  --operation hierarchy --warmups 5 --iterations 30
./build/meshprep-benchmark --triangles 9963191 \
  --operation hierarchy --warmups 1 --iterations 3
```

Each `RESULT` row contains CUDA-event time, wall time, throughput, topology statistics, peak retained workspace, and output capacity. Use `summarize.py` to summarize one or more captured CSV streams.

The benchmark intentionally does not download scenes during configure/build and does not bundle external assets.
