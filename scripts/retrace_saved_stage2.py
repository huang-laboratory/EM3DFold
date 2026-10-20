#!/usr/bin/env python3
"""Rerun protein tracing/HMM postprocessing from saved Stage2 diagnostics."""

import argparse
import os

import numpy as np

from em3dfold.infer.denovo import (
    _build_protein_chain_outputs,
    _compute_residue_type_scores,
    _pred_rmsd_to_confidence,
    remove_overlapping_ca,
    _split_chain_outputs_at_backbone_discontinuities,
    _split_chain_outputs_at_sequence_discontinuities,
)
from em3dfold.infer.flood_fill import (
    flood_fill_with_edge_dual,
    global_path_cover_with_edge_dual_protein,
)
from em3dfold.io.pdbio import chains_atom_pos_to_pdb
from em3dfold.io.seqio import read_fasta
from em3dfold.polymer_utils.polymer import get_polymer_from_file_path


def _load_edge_dict(npz):
    edge_dict = {}
    for source, destination, probabilities in zip(
        npz["source_original_index"],
        npz["destination_original_index"],
        npz["probabilities"],
    ):
        edge_dict.setdefault(int(source), {})[int(destination)] = np.asarray(
            probabilities, dtype=np.float32
        )
    return edge_dict


def _write_chains(output_path, atom_positions, atom_mask, confidence, chains, chain_types):
    chains_atom_pos = [atom_positions[chain] for chain in chains]
    chains_atom_mask = [atom_mask[chain] for chain in chains]
    chains_res_idx = [np.arange(len(chain), dtype=np.int32) for chain in chains]
    chains_bfactor = [
        np.repeat(confidence[chain, None], atom_mask.shape[1], axis=1)
        for chain in chains
    ]
    chains_atom_pos_to_pdb(
        filename=output_path,
        chains_atom_pos=chains_atom_pos,
        chains_atom_mask=chains_atom_mask,
        chains_res_type=chain_types,
        chains_res_idx=chains_res_idx,
        chains_bfactor=chains_bfactor,
        suffix="cif",
    )


def _split_at_low_confidence_edges(
    chains,
    chain_types,
    chain_sequence_indices,
    survived_original_indices,
    edge_dict,
    minimum_probability,
):
    """Split final HMM chains without changing coordinates or residue types."""
    split_chains = []
    split_types = []
    split_sequence_indices = []
    num_splits = 0
    default_probability = np.asarray([0.8, 0.1, 0.1], dtype=np.float32)
    for chain, residue_types, sequence_indices in zip(
        chains, chain_types, chain_sequence_indices
    ):
        chain = np.asarray(chain, dtype=np.int32)
        residue_types = np.asarray(residue_types)
        sequence_indices = np.asarray(sequence_indices, dtype=np.int32)
        segment_start = 0
        for offset in range(1, len(chain)):
            source = int(survived_original_indices[int(chain[offset - 1])])
            destination = int(survived_original_indices[int(chain[offset])])
            forward = np.asarray(
                edge_dict.get(source, {}).get(destination, default_probability),
                dtype=np.float32,
            )
            reverse = np.asarray(
                edge_dict.get(destination, {}).get(source, default_probability),
                dtype=np.float32,
            )
            probability = float(
                np.sqrt(max(float(forward[1]), 0.0) * max(float(reverse[2]), 0.0))
            )
            if probability < minimum_probability:
                split_chains.append(chain[segment_start:offset])
                split_types.append(residue_types[segment_start:offset])
                split_sequence_indices.append(sequence_indices[segment_start:offset])
                segment_start = offset
                num_splits += 1
        if len(chain):
            split_chains.append(chain[segment_start:])
            split_types.append(residue_types[segment_start:])
            split_sequence_indices.append(sequence_indices[segment_start:])
    return split_chains, split_types, split_sequence_indices, num_splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recycle-dir", required=True)
    parser.add_argument("--protein-seq", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--backend",
        choices=[
            "greedy",
            "geometry_greedy",
            "reciprocal_greedy",
            "global_path_cover",
        ],
        default="global_path_cover",
    )
    parser.add_argument("--minimum-edge-score", type=float, default=0.60)
    parser.add_argument("--bond-distance-threshold", type=float, default=1.50)
    parser.add_argument(
        "--assignment-backend",
        choices=["dense", "sparse"],
        default="sparse",
    )
    parser.add_argument(
        "--assignment-objective",
        choices=["penalized", "max_cardinality"],
        default="penalized",
    )
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
    parser.add_argument(
        "--post-hmm-minimum-edge-probability",
        type=float,
        default=None,
        help="Split final chains below this reciprocal NEXT/PREV probability",
    )
    parser.add_argument(
        "--existence-radius",
        type=float,
        default=None,
        help="Recompute confidence-ordered CA NMS at this radius instead of reusing the saved mask",
    )
    parser.add_argument("--split-sequence-discontinuities", action="store_true")
    parser.add_argument(
        "--sequence-split-mode",
        choices=["all", "nonmonotonic"],
        default="all",
    )
    args = parser.parse_args()

    recycle_dir = os.path.abspath(args.recycle_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    raw_polymer = get_polymer_from_file_path(
        os.path.join(recycle_dir, "protein_before_existence_filter.cif")
    )
    score_data = np.load(
        os.path.join(recycle_dir, "residue_type_scores_before_existence_filter.npz")
    )
    edge_data = np.load(os.path.join(recycle_dir, "protein_edge_scores.npz"))
    raw_atom_positions = np.asarray(raw_polymer.atomc_positions, dtype=np.float32)
    raw_atom_mask = np.asarray(raw_polymer.atomc_mask, dtype=np.float32)
    raw_pred_aatype = np.asarray(score_data["pred_aatype"], dtype=np.float32)
    raw_prot_mask = np.asarray(score_data["prot_mask"], dtype=bool)
    raw_pred_rmsd = np.asarray(score_data["pred_rmsd"], dtype=np.float32)
    raw_confidence = _pred_rmsd_to_confidence(raw_pred_rmsd, raw_prot_mask)
    if args.existence_radius is None:
        survived = np.asarray(edge_data["survived_original_indices"], dtype=np.int32)
    else:
        existence_mask = np.ones(len(raw_atom_positions), dtype=bool)
        existence_mask[raw_prot_mask] = remove_overlapping_ca(
            raw_atom_positions[raw_prot_mask, 1],
            raw_confidence[raw_prot_mask],
            radius_threshold=args.existence_radius,
        )
        survived = np.flatnonzero(existence_mask).astype(np.int32)

    atom_positions = raw_atom_positions[survived]
    atom_mask = raw_atom_mask[survived]
    pred_aatype = raw_pred_aatype[survived]
    prot_mask = raw_prot_mask[survived]
    pred_rmsd = raw_pred_rmsd[survived]
    confidence = _pred_rmsd_to_confidence(pred_rmsd, prot_mask)
    residue_type_entropy, residue_type_confidence = _compute_residue_type_scores(
        pred_aatype, prot_mask
    )
    edge_dict = _load_edge_dict(edge_data)
    original_indices = survived[prot_mask]
    protein_atom_positions = atom_positions[prot_mask]
    protein_confidence = confidence[prot_mask]

    if args.backend == "global_path_cover":
        traced_local = global_path_cover_with_edge_dual_protein(
            protein_atom_positions,
            edge_dict,
            original_indices,
            bond_distance_threshold=args.bond_distance_threshold,
            edge_score_mode=args.edge_score_mode,
            geometry_weight=args.geometry_weight,
            minimum_edge_score=args.minimum_edge_score,
            assignment_backend=args.assignment_backend,
            assignment_objective=args.assignment_objective,
        )
    else:
        greedy_edge_dict = {} if args.backend == "geometry_greedy" else edge_dict
        traced_local = flood_fill_with_edge_dual(
            protein_atom_positions,
            protein_confidence,
            bond_distance_threshold=args.bond_distance_threshold,
            edge_existence_dict=greedy_edge_dict,
            idx_exist_to_original=original_indices,
            use_reciprocal_edges=args.backend == "reciprocal_greedy",
        )
    protein_global_indices = np.flatnonzero(prot_mask)
    traced_chains = [protein_global_indices[chain] for chain in traced_local]
    reordered_results = {
        "pred_aatype": pred_aatype,
        "residue_type_entropy": residue_type_entropy,
        "residue_type_confidence": residue_type_confidence,
    }
    protein_sequences = [sequence.upper() for sequence in read_fasta(args.protein_seq)]
    (
        before_chains,
        before_types,
        before_sequence_indices,
        after_chains,
        after_types,
        after_sequence_indices,
        _parent_hmm_match_score_by_node,
    ) = _build_protein_chain_outputs(
        reordered_results,
        traced_chains,
        protein_sequences,
        os.path.join(args.output_dir, "hmm"),
        atom_positions,
    )
    if args.post_hmm_minimum_edge_probability is not None:
        (
            after_chains,
            after_types,
            after_sequence_indices,
            confidence_splits,
        ) = _split_at_low_confidence_edges(
            after_chains,
            after_types,
            after_sequence_indices,
            survived,
            edge_dict,
            args.post_hmm_minimum_edge_probability,
        )
    else:
        confidence_splits = 0
    if args.split_sequence_discontinuities:
        (
            after_chains,
            after_types,
            after_sequence_indices,
            sequence_splits,
        ) = _split_chain_outputs_at_sequence_discontinuities(
            after_chains,
            after_types,
            after_sequence_indices,
            split_forward_gaps=args.sequence_split_mode == "all",
        )
    else:
        sequence_splits = 0
    after_chains, after_types, backbone_splits = (
        _split_chain_outputs_at_backbone_discontinuities(
            after_chains,
            after_types,
            atom_positions,
        )
    )
    output_path = os.path.join(args.output_dir, "protein_after_prune.cif")
    _write_chains(
        output_path,
        atom_positions,
        atom_mask,
        confidence,
        after_chains,
        after_types,
    )
    print(
        {
            "backend": args.backend,
            "minimum_edge_score": args.minimum_edge_score,
            "bond_distance_threshold": args.bond_distance_threshold,
            "assignment_backend": args.assignment_backend,
            "assignment_objective": args.assignment_objective,
            "edge_score_mode": args.edge_score_mode,
            "geometry_weight": args.geometry_weight,
            "post_hmm_minimum_edge_probability": args.post_hmm_minimum_edge_probability,
            "confidence_splits": confidence_splits,
            "raw_traced_chains": len(traced_chains),
            "output_chains": len(after_chains),
            "output_residues": int(sum(len(chain) for chain in after_chains)),
            "sequence_splits": sequence_splits,
            "backbone_splits": backbone_splits,
            "output": output_path,
        }
    )


if __name__ == "__main__":
    main()
