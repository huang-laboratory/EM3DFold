#!/usr/bin/env python3
"""Benchmark dense or sparse global tracing from saved Stage2 diagnostics."""

import argparse
import hashlib
import json
import os
import resource
import time

import numpy as np

from em3dfold.infer.flood_fill import global_path_cover_with_edge_dual_protein
from em3dfold.polymer_utils.polymer import get_polymer_from_file_path


def _rss_mib():
    with open("/proc/self/status", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return float("nan")


def _load_edge_dict(edge_data):
    edge_dict = {}
    for source, destination, probabilities in zip(
        edge_data["source_original_index"],
        edge_data["destination_original_index"],
        edge_data["probabilities"],
    ):
        edge_dict.setdefault(int(source), {})[int(destination)] = np.asarray(
            probabilities, dtype=np.float32
        )
    return edge_dict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recycle-dir", required=True)
    parser.add_argument("--backend", choices=["dense", "sparse"], required=True)
    parser.add_argument("--minimum-edge-score", type=float, default=0.60)
    parser.add_argument(
        "--edge-score-mode",
        choices=[
            "reciprocal_log_odds",
            "reciprocal_log_probability",
            "reciprocal_probability",
        ],
        default="reciprocal_probability",
    )
    parser.add_argument("--geometry-weight", type=float, default=0.0)
    parser.add_argument("--bond-distance-threshold", type=float, default=1.50)
    args = parser.parse_args()

    recycle_dir = os.path.abspath(args.recycle_dir)
    polymer = get_polymer_from_file_path(
        os.path.join(recycle_dir, "protein_before_existence_filter.cif")
    )
    edge_data = np.load(os.path.join(recycle_dir, "protein_edge_scores.npz"))
    survived = np.asarray(edge_data["survived_original_indices"], dtype=np.int32)
    protein_mask = np.asarray(
        edge_data["protein_mask_after_existence"], dtype=bool
    )
    atom_positions = np.asarray(polymer.atomc_positions, dtype=np.float32)[survived]
    atom_positions = atom_positions[protein_mask]
    original_indices = survived[protein_mask]
    edge_dict = _load_edge_dict(edge_data)

    rss_before = _rss_mib()
    start = time.perf_counter()
    chains = global_path_cover_with_edge_dual_protein(
        atom_positions,
        edge_dict,
        original_indices,
        bond_distance_threshold=args.bond_distance_threshold,
        edge_score_mode=args.edge_score_mode,
        geometry_weight=args.geometry_weight,
        minimum_edge_score=args.minimum_edge_score,
        assignment_backend=args.backend,
    )
    elapsed = time.perf_counter() - start
    serialized = json.dumps(chains, separators=(",", ":")).encode("utf-8")
    result = {
        "backend": args.backend,
        "nodes": len(atom_positions),
        "input_edge_predictions": int(
            sum(len(destination_map) for destination_map in edge_dict.values())
        ),
        "chains": len(chains),
        "selected_edges": int(sum(max(0, len(chain) - 1) for chain in chains)),
        "chain_sha256": hashlib.sha256(serialized).hexdigest(),
        "elapsed_seconds": elapsed,
        "rss_before_mib": rss_before,
        "rss_after_mib": _rss_mib(),
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
