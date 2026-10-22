#!/usr/bin/env python
"""Run one mode-merge backend on a captured EM3DFold input."""

import argparse
import hashlib
import json

import numpy as np

from em3dfold.pipeline import meanshift


def digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument("capture")
parser.add_argument("--backend", choices=["spatial", "torch", "ckdtree"], required=True)
parser.add_argument("--device", default="auto")
parser.add_argument("--block-size", type=int, default=256)
args = parser.parse_args()

data = np.load(args.capture)
points = np.asarray(data["points"], dtype=np.float32)
densities = np.asarray(data["densities"], dtype=np.float32)
mrc = meanshift.MRCMap(
    data=np.zeros((1, 1, 1), dtype=np.float32),
    origin=np.zeros(3, dtype=np.float32),
    voxel_size=np.asarray(data["voxel_size"], dtype=np.float32),
)
params = meanshift.Params(
    rmerge=float(data["rmerge"]),
    filter_fraction=float(data["filter_fraction"]),
    device=args.device,
    merge_backend=args.backend,
    merge_block_size=args.block_size,
)
centers, center_densities, membership = meanshift._merge_modes(
    points, densities, params, mrc
)
print(
    json.dumps(
        {
            "backend": args.backend,
            "points": len(points),
            "centers": len(centers),
            "centers_sha256": digest(centers),
            "center_densities_sha256": digest(center_densities),
            "membership_sha256": digest(membership),
        },
        sort_keys=True,
    )
)
