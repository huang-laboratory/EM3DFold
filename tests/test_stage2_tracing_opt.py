import argparse
import inspect
from types import SimpleNamespace

import numpy as np

import em3dfold.infer.inferlm_common as inferlm_common
from em3dfold.infer.denovo import (
    _parent_hmm_match_score_by_node,
    _prune_low_confidence_short_protein_chains,
    _should_try_adaptive_greedy,
    _should_use_adaptive_greedy,
    _surviving_original_indices,
    _trace_na_chains_local,
    _split_chain_outputs_at_backbone_discontinuities,
    _split_chain_outputs_at_sequence_discontinuities,
)
from em3dfold.infer.inferlm_common import prepare_common_args
from em3dfold.infer.flood_fill import (
    EDGE_NEXT,
    EDGE_PREV,
    edge_correction,
    global_path_cover_with_edge_dual_protein,
)


def test_existence_filter_keeps_pre_filter_edge_indices():
    mask = np.array([True, False, True, True, False, True])
    survivors = _surviving_original_indices(mask)
    assert survivors.tolist() == [0, 2, 3, 5]
    assert survivors[np.array([True, False, True, True])].tolist() == [0, 3, 5]


def test_na_tracing_remains_local_with_only_corrected_original_indices(monkeypatch):
    captured = {}

    def fake_local_trace(atom_positions, b_factors, **kwargs):
        captured["atom_positions"] = atom_positions.copy()
        captured["b_factors"] = b_factors.copy()
        captured.update(kwargs)
        return [[1, 0]]

    monkeypatch.setattr(
        "em3dfold.infer.denovo.flood_fill_with_edge_dual", fake_local_trace
    )
    atom_positions = np.arange(4 * 9 * 3, dtype=np.float32).reshape(4, 9, 3)
    confidence = np.arange(4, dtype=np.float32)
    na_mask = np.array([False, True, False, True])
    surviving_original_indices = np.array([2, 5, 7, 11], dtype=np.int32)

    chains = _trace_na_chains_local(
        atom_positions,
        confidence,
        na_mask,
        edge_existence_dict={"sentinel": True},
        survived_original_indices=surviving_original_indices,
    )

    assert [chain.tolist() for chain in chains] == [[3, 1]]
    assert captured["idx_exist_to_original"].tolist() == [5, 11]
    assert captured["bond_distance_threshold"] == 3.6
    assert captured["last_idx"] == 6
    assert captured["next_idx"] == 8
    assert captured["use_reciprocal_edges"] is False
    assert captured["edge_existence_dict"] == {"sentinel": True}
    np.testing.assert_array_equal(captured["atom_positions"], atom_positions[na_mask])
    np.testing.assert_array_equal(captured["b_factors"], confidence[na_mask])


def test_backbone_discontinuity_split_preserves_residues_and_types():
    atom_positions = np.zeros((4, 3, 3), dtype=np.float32)
    atom_positions[0, 2] = [0.0, 0.0, 0.0]
    atom_positions[1, 0] = [1.3, 0.0, 0.0]
    atom_positions[1, 2] = [2.0, 0.0, 0.0]
    atom_positions[2, 0] = [8.0, 0.0, 0.0]
    atom_positions[2, 2] = [9.0, 0.0, 0.0]
    atom_positions[3, 0] = [10.3, 0.0, 0.0]

    chains, residue_types, num_splits = (
        _split_chain_outputs_at_backbone_discontinuities(
            [np.arange(4)],
            [np.arange(10, 14)],
            atom_positions,
        )
    )

    assert num_splits == 1
    assert [chain.tolist() for chain in chains] == [[0, 1], [2, 3]]
    assert [types.tolist() for types in residue_types] == [[10, 11], [12, 13]]


def test_parent_hmm_scores_are_propagated_to_nodes():
    scores = _parent_hmm_match_score_by_node(
        7,
        [np.array([1, 3, 5]), np.array([0, 2])],
        [0.6, 0.9],
    )
    np.testing.assert_allclose(scores, [0.9, 0.6, 0.9, 0.6, 1.0, 0.6, 1.0])


def test_confidence_aware_short_chain_pruning_uses_two_of_three_votes():
    chains = [
        np.arange(0, 5),
        np.arange(5, 10),
        np.arange(10, 16),
    ]
    chain_types = [np.full(len(chain), index) for index, chain in enumerate(chains)]
    confidence = np.full((16, 23), 0.95, dtype=np.float32)
    residue_type_confidence = np.full((16,), 0.99, dtype=np.float32)
    parent_hmm_score = np.full((16,), 0.95, dtype=np.float32)

    # Chain 0 is short and has two low signals, so it is removed.
    confidence[chains[0]] = 0.70
    residue_type_confidence[chains[0]] = 0.80
    # Chain 1 has only one low signal and must remain.
    parent_hmm_score[chains[1]] = 0.50
    # Chain 2 has all three low signals but is longer than five residues.
    confidence[chains[2]] = 0.50
    residue_type_confidence[chains[2]] = 0.50
    parent_hmm_score[chains[2]] = 0.50

    kept_chains, kept_types, removed = (
        _prune_low_confidence_short_protein_chains(
            chains,
            chain_types,
            {
                "confidence": confidence,
                "residue_type_confidence": residue_type_confidence,
            },
            parent_hmm_score,
        )
    )

    assert [chain.tolist() for chain in kept_chains] == [
        chains[1].tolist(),
        chains[2].tolist(),
    ]
    assert [types.tolist() for types in kept_types] == [
        chain_types[1].tolist(),
        chain_types[2].tolist(),
    ]
    assert len(removed) == 1
    assert removed[0]["chain_index"] == 0
    assert removed[0]["low_confidence_votes"] == 2


def test_stage2_cli_enables_short_chain_pruning_by_default():
    parser = prepare_common_args(argparse.ArgumentParser())
    args = parser.parse_args(["--map", "input.mrc", "--polymer", "input.cif"])

    assert args.prune_low_confidence_short_protein_chains is True
    assert args.use_final_pruned_structure is False


def test_stage2_cli_can_disable_pruning_or_select_pruned_output():
    parser = prepare_common_args(argparse.ArgumentParser())
    disabled_args = parser.parse_args(
        [
            "--map",
            "input.mrc",
            "--polymer",
            "input.cif",
            "--no-prune-low-confidence-short-protein-chains",
        ]
    )
    pruned_output_args = parser.parse_args(
        [
            "--map",
            "input.mrc",
            "--polymer",
            "input.cif",
            "--use-final-pruned-structure",
        ]
    )

    assert disabled_args.prune_low_confidence_short_protein_chains is False
    assert pruned_output_args.use_final_pruned_structure is True


def _run_mocked_stage2_recycles(monkeypatch, tmp_path, use_final_pruned_structure):
    prune_flags = []
    merge_calls = {}

    def fake_run_inference_loop(args, *_unused):
        recycle_idx = len(prune_flags)
        prune_flags.append(args.prune_low_confidence_short_protein_chains)
        output_dir = str(tmp_path / f"recycle_{recycle_idx}")
        output_info = {
            "protein_after_prune_path": f"{output_dir}/protein_after_prune.cif",
            "protein_after_num_res": 100,
            "protein_before_final_prune_path": (
                f"{output_dir}/protein_before_final_prune.cif"
            ),
            "protein_after_final_prune_path": (
                f"{output_dir}/protein_after_final_prune.cif"
            ),
            "after_final_prune_entropy_score_path": None,
            "protein_after_final_prune_num_res": 95,
            "protein_after_quality": 0.9 if recycle_idx == 2 else 0.8,
            "protein_after_mean_confidence": 0.9,
            "protein_after_mean_residue_type_confidence": 0.9,
            "na_after_num_res": 50,
            "na_after_prune_path": f"{output_dir}/na_after_prune.cif",
            "before_prune_path": f"{output_dir}/before_prune.cif",
            "output_path": f"{output_dir}/output.cif",
            "output_entropy_score_path": None,
            "next_round_state": {},
        }
        return output_info

    def fake_write_merged(output_path, protein_path=None, na_path=None):
        merge_calls[output_path] = (protein_path, na_path)
        return output_path

    monkeypatch.setattr(inferlm_common, "run_inference_loop", fake_run_inference_loop)
    monkeypatch.setattr(
        inferlm_common, "_write_merged_best_so_far_output", fake_write_merged
    )
    monkeypatch.setattr(
        inferlm_common, "_write_final_chain_hmm_profiles", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(inferlm_common, "_resolve_chain_hmm_profile_dir", lambda: None)
    monkeypatch.setattr(inferlm_common, "clear_cuda_cache", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(inferlm_common, "seed_everything", lambda *_args, **_kwargs: None)

    args = SimpleNamespace(
        model_dir=str(tmp_path / "weights"),
        output_dir=str(tmp_path),
        recycle=4,
        device="cpu",
        prune_low_confidence_short_protein_chains=True,
        use_final_pruned_structure=use_final_pruned_structure,
        pass_prev_aa_probs=False,
        pass_prev_rmsd=False,
        pass_prev_node=False,
    )
    inferlm_common.run_main(args, object, {}, lambda: None)
    return prune_flags, merge_calls


def test_stage2_runs_confidence_pruning_in_every_recycle(monkeypatch, tmp_path):
    prune_flags, merge_calls = _run_mocked_stage2_recycles(
        monkeypatch, tmp_path, use_final_pruned_structure=False
    )

    assert prune_flags == [True, True, True, True]
    assert merge_calls[str(tmp_path / "before_final_prune.cif")][0].endswith(
        "recycle_2/protein_after_prune.cif"
    )
    assert merge_calls[str(tmp_path / "after_final_prune.cif")][0].endswith(
        "recycle_2/protein_after_final_prune.cif"
    )
    assert merge_calls[str(tmp_path / "output_final_prune.cif")][0].endswith(
        "recycle_2/protein_after_final_prune.cif"
    )
    assert merge_calls[str(tmp_path / "output.cif")][0].endswith(
        "recycle_2/protein_after_prune.cif"
    )
    assert merge_calls[str(tmp_path / "output.cif")][1].endswith(
        "recycle_3/na_after_prune.cif"
    )
    assert merge_calls[str(tmp_path / "output_final_prune.cif")][1].endswith(
        "recycle_3/na_after_prune.cif"
    )


def test_stage2_option_selects_final_pruned_structure_for_output(monkeypatch, tmp_path):
    _, merge_calls = _run_mocked_stage2_recycles(
        monkeypatch, tmp_path, use_final_pruned_structure=True
    )

    assert merge_calls[str(tmp_path / "output.cif")][0].endswith(
        "recycle_2/protein_after_final_prune.cif"
    )


def test_sequence_split_can_be_limited_to_nonmonotonic_registration():
    chains = [np.arange(5)]
    residue_types = [np.arange(5)]
    sequence_indices = [np.array([10, 11, 14, 15, 8])]

    all_chains, _, _, all_splits = _split_chain_outputs_at_sequence_discontinuities(
        chains,
        residue_types,
        sequence_indices,
    )
    monotonic_chains, _, _, monotonic_splits = (
        _split_chain_outputs_at_sequence_discontinuities(
            chains,
            residue_types,
            sequence_indices,
            split_forward_gaps=False,
        )
    )

    assert all_splits == 2
    assert [chain.tolist() for chain in all_chains] == [[0, 1], [2, 3], [4]]
    assert monotonic_splits == 1
    assert [chain.tolist() for chain in monotonic_chains] == [[0, 1, 2, 3], [4]]


def test_reciprocal_edge_score_breaks_a_one_sided_tie():
    edge_dict = {
        0: {
            1: np.array([0.05, 0.05, 0.90]),
            2: np.array([0.05, 0.05, 0.90]),
        },
        1: {0: np.array([0.05, 0.90, 0.05])},
        2: {0: np.array([0.90, 0.05, 0.05])},
    }
    order = edge_correction(
        possible_indices=np.array([1, 2]),
        idx=0,
        dmat=np.array([1.35, 1.35]),
        edge_existence_dict=edge_dict,
        idx_exist_to_original=np.arange(3),
        edge_class_idx=EDGE_PREV,
        reciprocal_edge_class_idx=EDGE_NEXT,
        use_reciprocal=True,
    )
    assert order.tolist() == [0, 1]


def test_edge_correction_tolerates_missing_source_entry():
    order = edge_correction(
        possible_indices=np.array([1, 2]),
        idx=0,
        dmat=np.array([1.35, 1.60]),
        edge_existence_dict={},
        idx_exist_to_original=np.arange(3),
        edge_class_idx=EDGE_PREV,
        reciprocal_edge_class_idx=EDGE_NEXT,
        use_reciprocal=True,
    )
    assert order.tolist() == [0, 1]


def test_adaptive_greedy_requires_low_retention_and_material_gain():
    assert _should_try_adaptive_greedy(79, 100)
    assert not _should_try_adaptive_greedy(80, 100)
    assert not _should_try_adaptive_greedy(0, 0)
    assert _should_use_adaptive_greedy(89, 100)
    assert not _should_use_adaptive_greedy(90, 100)


def test_global_path_cover_uses_reciprocal_next_prev_edges():
    atom_positions = np.zeros((3, 3, 3), dtype=np.float32)
    for index in range(3):
        atom_positions[index, 0] = [index * 3.8, 0.0, 0.0]
        atom_positions[index, 1] = [index * 3.8 + 1.45, 0.0, 0.0]
        atom_positions[index, 2] = [index * 3.8 + 2.47, 0.0, 0.0]
    edge_dict = {index: {} for index in range(3)}
    for index in range(2):
        edge_dict[index][index + 1] = np.array([0.01, 0.98, 0.01])
        edge_dict[index + 1][index] = np.array([0.01, 0.01, 0.98])

    chains = global_path_cover_with_edge_dual_protein(
        atom_positions,
        edge_dict,
        np.arange(3),
    )
    assert chains == [[0, 1, 2]]


def test_sparse_global_path_cover_matches_dense_for_unique_scores():
    atom_positions = np.zeros((8, 3, 3), dtype=np.float32)
    for index in range(8):
        atom_positions[index, 0] = [index * 3.8, 0.0, 0.0]
        atom_positions[index, 1] = [index * 3.8 + 1.45, 0.0, 0.0]
        atom_positions[index, 2] = [index * 3.8 + 2.47, 0.0, 0.0]
    edge_dict = {index: {} for index in range(8)}
    for index in range(7):
        confidence = 0.97 - index * 0.01
        edge_dict[index][index + 1] = np.array(
            [0.01, confidence, 0.02 + index * 0.01]
        )
        edge_dict[index + 1][index] = np.array(
            [0.01, 0.02 + index * 0.01, confidence]
        )

    dense_chains = global_path_cover_with_edge_dual_protein(
        atom_positions,
        edge_dict,
        np.arange(8),
        assignment_backend="dense",
    )
    sparse_chains = global_path_cover_with_edge_dual_protein(
        atom_positions,
        edge_dict,
        np.arange(8),
        assignment_backend="sparse",
    )

    assert dense_chains == sparse_chains == [list(range(8))]


def test_global_path_cover_defaults_to_sparse():
    atom_positions = np.zeros((3, 3, 3), dtype=np.float32)
    for index in range(3):
        atom_positions[index, 0] = [index * 3.8, 0.0, 0.0]
        atom_positions[index, 1] = [index * 3.8 + 1.45, 0.0, 0.0]
        atom_positions[index, 2] = [index * 3.8 + 2.47, 0.0, 0.0]
    edge_dict = {index: {} for index in range(3)}
    for index in range(2):
        edge_dict[index][index + 1] = np.array([0.01, 0.98, 0.01])
        edge_dict[index + 1][index] = np.array([0.01, 0.01, 0.98])

    default_chains = global_path_cover_with_edge_dual_protein(
        atom_positions,
        edge_dict,
        np.arange(3),
    )
    sparse_chains = global_path_cover_with_edge_dual_protein(
        atom_positions,
        edge_dict,
        np.arange(3),
        assignment_backend="sparse",
    )

    assert default_chains == sparse_chains == [[0, 1, 2]]


def test_global_path_cover_defaults_match_validated_generalization_settings():
    defaults = inspect.signature(
        global_path_cover_with_edge_dual_protein
    ).parameters
    assert defaults["bond_distance_threshold"].default == 1.5
    assert defaults["edge_score_mode"].default == "reciprocal_probability"
    assert defaults["geometry_weight"].default == 0.0
    assert defaults["minimum_edge_score"].default == 0.60
    assert defaults["assignment_backend"].default == "sparse"


def test_max_cardinality_keeps_a_geometrically_feasible_low_score_edge():
    atom_positions = np.zeros((2, 3, 3), dtype=np.float32)
    atom_positions[0, 0] = [-1.45, 0.0, 0.0]
    atom_positions[0, 1] = [0.0, 0.0, 0.0]
    atom_positions[0, 2] = [1.02, 0.0, 0.0]
    atom_positions[1, 0] = [2.35, 0.0, 0.0]
    atom_positions[1, 1] = [3.80, 0.0, 0.0]
    atom_positions[1, 2] = [4.82, 0.0, 0.0]
    edge_dict = {
        0: {1: np.array([0.80, 0.10, 0.10])},
        1: {0: np.array([0.80, 0.10, 0.10])},
    }

    penalized = global_path_cover_with_edge_dual_protein(
        atom_positions,
        edge_dict,
        np.arange(2),
        assignment_objective="penalized",
    )
    max_cardinality = global_path_cover_with_edge_dual_protein(
        atom_positions,
        edge_dict,
        np.arange(2),
        assignment_objective="max_cardinality",
    )

    assert penalized == [[0], [1]]
    assert max_cardinality == [[0, 1]]
