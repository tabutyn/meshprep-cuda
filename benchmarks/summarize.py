#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Summarize parallel-mater-benchmark RESULT rows without third-party packages."""

import csv
import statistics
import sys
from collections import defaultdict


def percentile(values: list[float], percent: int) -> float:
    return statistics.quantiles(values, n=100, method="inclusive")[percent - 1]


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} RESULTS.csv", file=sys.stderr)
        return 2
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    with open(sys.argv[1], newline="", encoding="utf-8") as source:
        rows = [line.removeprefix("RESULT,") for line in source if line.startswith("RESULT,")]
    header = (
        "operation,source,vertices,triangles,iteration,gpu_ms,wall_ms,"
        "mtriangles_per_second,nodes,leaves,branches,max_depth,workspace_bytes,output_bytes"
    )
    for row in csv.DictReader([header, *rows]):
        groups[(row["operation"], row["source"])].append(row)
    print("operation,source,samples,median_gpu_ms,p5_gpu_ms,p95_gpu_ms,median_mtriangles_per_second")
    for (operation, source), group in sorted(groups.items()):
        times = [float(row["gpu_ms"]) for row in group]
        throughput = [float(row["mtriangles_per_second"]) for row in group]
        print(
            f"{operation},{source},{len(group)},{statistics.median(times):.6f},"
            f"{percentile(times, 5):.6f},{percentile(times, 95):.6f},"
            f"{statistics.median(throughput):.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
