#!/usr/bin/env python
"""Benchmark the production spatial and Torch mode-merge primitives."""

import argparse
import hashlib
import json
import resource
import time

import numpy as np

from em3dfold.pipeline.spatial_hash import membership_spatial, select_centers_spatial
from em3dfold.pipeline.torch_merge import membership_torch


def make_paired_points(n_points):
    n_centers = (n_points + 1) // 2
    side = int(np.ceil(n_centers ** (1.0 / 3.0)))
    index = np.arange(n_centers, dtype=np.int64)
    centers = np.column_stack(
        (index % side, (index // side) % side, index // (side * side))
    ).astype(np.float32)
    centers *= 3.0
    points = np.repeat(centers, 2, axis=0)[:n_points]
    points[1::2, 0] += 0.1
    densities = np.empty(n_points, dtype=np.float32)
    densities[0::2] = np.linspace(1.0, 0.6, len(densities[0::2]), dtype=np.float32)
    densities[1::2] = np.linspace(0.4, 0.0, len(densities[1::2]), dtype=np.float32)
    return points, densities


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument("--points", type=int, required=True)
parser.add_argument("--backend", choices=["spatial", "torch"], required=True)
parser.add_argument("--device", default="auto")
parser.add_argument("--block-size", type=int, default=256)
args = parser.parse_args()

points, densities = make_paired_points(args.points)
if args.backend == "torch":
    import torch

    warm = torch.zeros((8, 3), device=args.device)
    _ = torch.sum(warm * warm).cpu()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
start = time.perf_counter()
center_ids = select_centers_spatial(points, densities, 1.0)
selection_seconds = time.perf_counter() - start
centers = points[center_ids]

start = time.perf_counter()
if args.backend == "torch":
    membership = membership_torch(points, centers, args.device, args.block_size)
else:
    membership = membership_spatial(
        points,
        centers,
        1.0,
        known_within_radius=np.ones(len(points), dtype=bool),
    )
membership_seconds = time.perf_counter() - start

result = {
    "backend": args.backend,
    "points": len(points),
    "centers": len(center_ids),
    "selection_seconds": selection_seconds,
    "membership_seconds": membership_seconds,
    "total_seconds": selection_seconds + membership_seconds,
    "center_digest": digest(center_ids),
    "membership_digest": digest(membership),
    "peak_cpu_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
}
if args.backend == "torch":
    result["peak_gpu_allocated_mib"] = torch.cuda.max_memory_allocated() / 1024**2
    result["peak_gpu_reserved_mib"] = torch.cuda.max_memory_reserved() / 1024**2
print(json.dumps(result, sort_keys=True))
