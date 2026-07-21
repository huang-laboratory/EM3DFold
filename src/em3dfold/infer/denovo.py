import os
import json
import copy
import shutil

import numpy as np
from scipy.spatial import cKDTree

from em3dfold.infer.fixing import fix_match
from em3dfold.infer.flood_fill import (
    beam_trace_with_edge_dual_protein,
    flood_fill_with_edge_dual,
)
from em3dfold.infer.gnn_inference_utils import sample_na_aa_logits
from em3dfold.infer.hmm.hmm_sequence_align import (
    MatchToSequence,
    best_match_to_sequences,
    fix_chains_pipeline,
    prune_and_connect_chains,
)
from em3dfold.infer.hmm.aa_probs_to_hmm import dump_aa_logits_to_hmm_file
from em3dfold.io.pdbio import chains_atom_pos_to_pdb
from em3dfold.io.seqio import read_fasta
from em3dfold.polymer_utils.residue_constants import num_prot, select_torsion_angles
from em3dfold.utils.misc_utils import abspath, pjoin
from em3dfold.utils.to_all_atom import affines_and_torsion_angles_to_atomc_pos


FALLBACK_CONFIDENCE_SENTINEL = 0.0


def _disable_na_pp_rescue() -> bool:
    flag = os.environ.get("EM3DFOLD_NA_DISABLE_PP_RESCUE", "")
    return flag.lower() in {"1", "true", "yes", "on"}


def _require_vanilla_support_for_na_pp() -> bool:
    flag = os.environ.get("EM3DFOLD_NA_PP_REQUIRE_VANILLA_SUPPORT", "")
    if flag == "":
        return True
    return flag.lower() in {"1", "true", "yes", "on"}


def _na_pp_vanilla_support_threshold() -> float:
    value = os.environ.get("EM3DFOLD_NA_PP_VANILLA_SUPPORT_THRESHOLD", "0.65")
    try:
        return float(value)
    except ValueError:
        return 0.65


def _enable_na_pp_moderate_rescue() -> bool:
    flag = os.environ.get("EM3DFOLD_NA_PP_ENABLE_MODERATE_RESCUE", "")
    return flag.lower() in {"1", "true", "yes", "on"}


def _na_pp_moderate_exact_threshold() -> float:
    value = os.environ.get("EM3DFOLD_NA_PP_MODERATE_EXACT_THRESHOLD", "0.15")
    try:
        return float(value)
    except ValueError:
        return 0.15


def _na_pp_moderate_score_threshold() -> float:
    value = os.environ.get("EM3DFOLD_NA_PP_MODERATE_SCORE_THRESHOLD", "0.50")
    try:
        return float(value)
    except ValueError:
        return 0.50


def _na_pp_moderate_min_gain() -> float:
    value = os.environ.get("EM3DFOLD_NA_PP_MODERATE_MIN_GAIN", "0.35")
    try:
        return float(value)
    except ValueError:
        return 0.35


def remove_overlapping_ca(
    ca_positions: np.ndarray,
    bfactors,
    existence_mask=None,
    radius_threshold: float = 1.5,
) -> np.ndarray:
    kdtree = cKDTree(ca_positions)
    bfactors_copy = np.copy(bfactors)
    sorted_indices = np.argsort(bfactors_copy)[::-1]
    if existence_mask is None:
        existence_mask = np.ones(len(ca_positions), dtype=bool)

    for i in sorted_indices:
        if existence_mask[i]:
            too_close = np.array(
                kdtree.query_ball_point(ca_positions[i], r=radius_threshold)
            )
            too_close = too_close[too_close != i]
            existence_mask[too_close] = False
    return existence_mask


def normalize_local_confidence_score(
    local_confidence_score: np.ndarray,
    best_value: float = 0.20,
    worst_value: float = 1.00,
) -> np.ndarray:
    normalized_score = (worst_value - local_confidence_score) / (
        worst_value - best_value
    )
    normalized_score = np.clip(normalized_score, 0, 1)
    return normalized_score


def _inverse_minmax_confidence(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    if max_value - min_value < 1e-8:
        return np.ones_like(values, dtype=np.float32) * 0.5
    return 1.0 - (values - min_value) / (max_value - min_value)


def _pred_rmsd_to_confidence(pred_rmsd: np.ndarray, prot_mask: np.ndarray) -> np.ndarray:
    pred_rmsd = np.asarray(pred_rmsd, dtype=np.float32)
    prot_mask = np.asarray(prot_mask, dtype=bool)
    confidence = np.zeros((len(pred_rmsd),), dtype=np.float32)

    if np.any(prot_mask):
        prot_conf = normalize_local_confidence_score(
            pred_rmsd[prot_mask],
            best_value=0.20,
            worst_value=1.20,
        )
        if float(np.max(prot_conf)) <= 1e-6 or float(np.std(prot_conf)) <= 1e-6:
            prot_conf = _inverse_minmax_confidence(pred_rmsd[prot_mask])
        confidence[prot_mask] = prot_conf

    na_mask = ~prot_mask
    if np.any(na_mask):
        na_conf = normalize_local_confidence_score(
            pred_rmsd[na_mask],
            best_value=0.40,
            worst_value=1.40,
        )
        if float(np.max(na_conf)) <= 1e-6 or float(np.std(na_conf)) <= 1e-6:
            na_conf = _inverse_minmax_confidence(pred_rmsd[na_mask])
        confidence[na_mask] = na_conf

    return np.clip(confidence, 0.0, 1.0).astype(np.float32)


def _read_sequences(sequence_path, label):
    if sequence_path is None:
        return []
    print(f"# Read {label} sequence(s)")
    seqs = [seq.upper() for seq in read_fasta(sequence_path)]
    print(f"# Found {len(seqs)} {label} sequences")
    lengths = [len(seq) for seq in seqs]
    if len(lengths) > 0:
        print(f"# {label} sequence lengths = {lengths}")
    return seqs


def _sanitize_na_sequence(seq):
    seq = seq.upper().replace("T", "U")
    seq = "".join([s if s in ["A", "C", "G", "U"] else "N" for s in seq])
    return seq


def _reconstruct_chains(reordered_final_results, chains, chains_res_type, bfactor_key="plddt"):
    chains_atom_pos = []
    chains_atom_mask = []
    chains_res_idx = []
    chains_bfactor = []
    for chain, aatype in zip(chains, chains_res_type):
        pred_torsions = select_torsion_angles(
            reordered_final_results["pred_torsions"][chain],
            aatype=aatype,
            normalize=True,
        )
        pred_atom_pos, pred_atom_mask = affines_and_torsion_angles_to_atomc_pos(
            reordered_final_results["pred_affines"][chain],
            pred_torsions,
            aatype=aatype,
        )
        chains_atom_pos.append(pred_atom_pos.cpu().numpy())
        chains_atom_mask.append(pred_atom_mask.cpu().numpy())
        chains_res_idx.append(np.arange(len(chain), dtype=np.int32))
        chains_bfactor.append(reordered_final_results[bfactor_key][chain])
    return chains_atom_pos, chains_atom_mask, chains_res_idx, chains_bfactor


def _get_chain_stats(chains):
    if len(chains) == 0:
        return {
            "num_chains": 0,
            "total_residues": 0,
            "max_len": 0,
            "min_len": 0,
            "mean_len": 0.0,
        }
    lengths = np.array([len(chain) for chain in chains], dtype=np.int32)
    return {
        "num_chains": int(len(chains)),
        "total_residues": int(lengths.sum()),
        "max_len": int(lengths.max()),
        "min_len": int(lengths.min()),
        "mean_len": float(lengths.mean()),
    }


def _print_chain_stats(label, chains):
    stats = _get_chain_stats(chains)
    print(
        "# {}: chains={}, total_residues={}, longest={}, shortest={}, mean={:.2f}".format(
            label,
            stats["num_chains"],
            stats["total_residues"],
            stats["max_len"],
            stats["min_len"],
            stats["mean_len"],
        )
    )


def _write_chain_file(
    filename,
    reordered_final_results,
    chains,
    chains_res_type,
    bfactor_key="plddt",
):
    chains_atom_pos, chains_atom_mask, chains_res_idx, chains_bfactor = _reconstruct_chains(
        reordered_final_results,
        chains,
        chains_res_type,
        bfactor_key=bfactor_key,
    )
    chains_atom_pos_to_pdb(
        filename=filename,
        chains_atom_pos=chains_atom_pos,
        chains_atom_mask=chains_atom_mask,
        chains_res_type=chains_res_type,
        chains_res_idx=chains_res_idx,
        chains_bfactor=chains_bfactor,
        suffix="cif",
    )
    print(f"# Output chains to {filename}")


def _resolve_chain_hmm_profile_dir():
    value = os.environ.get("EM3DFOLD_CHAIN_HMM_PROFILE_DIR", "")
    if str(value).strip() == "":
        return None
    return abspath(value)


def _infer_na_hmm_alphabet(chain_res_type):
    chain_res_type = np.asarray(chain_res_type, dtype=np.int32)
    if chain_res_type.size == 0:
        return "RNA"
    dna_votes = int(np.sum((chain_res_type >= num_prot) & (chain_res_type < num_prot + 4)))
    rna_votes = int(np.sum((chain_res_type >= num_prot + 4) & (chain_res_type < num_prot + 8)))
    return "DNA" if dna_votes > rna_votes else "RNA"


def _dump_chain_hmm_profiles(
    profile_root_dir,
    reordered_final_results,
    protein_chains,
    protein_chain_types,
    na_chains,
    na_chain_types,
    *,
    profile_set_name,
):
    if profile_root_dir is None:
        return
    profile_dir = os.path.join(profile_root_dir, profile_set_name)
    if os.path.isdir(profile_dir):
        shutil.rmtree(profile_dir)
    os.makedirs(profile_dir, exist_ok=True)

    pred_aatype = np.asarray(reordered_final_results["pred_aatype"], dtype=np.float32)
    confidence = reordered_final_results.get("confidence")
    residue_confidence = None
    if confidence is not None:
        confidence = np.asarray(confidence, dtype=np.float32)
        residue_confidence = confidence[:, 0] if confidence.ndim >= 2 else confidence

    summary = []

    for chain_idx, chain in enumerate(protein_chains):
        chain = np.asarray(chain, dtype=np.int32)
        chain_name = f"protein_chain_{chain_idx:04d}"
        output_file = os.path.join(profile_dir, f"{chain_name}.hmm")
        chain_conf = None if residue_confidence is None else np.asarray(residue_confidence[chain], dtype=np.float32)
        dump_aa_logits_to_hmm_file(
            pred_aatype[chain],
            output_file,
            confidence=chain_conf,
            name=chain_name,
            alphabet_type="amino",
        )
        summary.append({
            "name": chain_name,
            "type": "protein",
            "alphabet": "amino",
            "length": int(len(chain)),
            "path": output_file,
        })

    for chain_idx, (chain, chain_res_type) in enumerate(zip(na_chains, na_chain_types)):
        chain = np.asarray(chain, dtype=np.int32)
        alphabet = _infer_na_hmm_alphabet(chain_res_type)
        chain_name = f"na_chain_{chain_idx:04d}"
        chain_logits = np.asarray(pred_aatype[chain], dtype=np.float32)
        if chain_logits.shape[-1] == (num_prot + 4):
            alphabet = "RNA"
            expanded_logits = np.full((chain_logits.shape[0], num_prot + 8), -100.0, dtype=np.float32)
            expanded_logits[:, num_prot + 4 : num_prot + 8] = chain_logits[:, num_prot : num_prot + 4]
            chain_logits = expanded_logits
        output_file = os.path.join(profile_dir, f"{chain_name}.{alphabet.lower()}.hmm")
        chain_conf = None if residue_confidence is None else np.asarray(residue_confidence[chain], dtype=np.float32)
        dump_aa_logits_to_hmm_file(
            chain_logits,
            output_file,
            confidence=chain_conf,
            name=chain_name,
            alphabet_type=alphabet,
        )
        summary.append({
            "name": chain_name,
            "type": "na",
            "alphabet": alphabet,
            "length": int(len(chain)),
            "path": output_file,
        })

    summary_path = os.path.join(profile_dir, "profiles.json")
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"# Output chain HMM profiles to {profile_dir}")


def _softmax_numpy(logits):
    logits = np.asarray(logits, dtype=np.float32)
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    probs = np.exp(logits)
    probs = probs / np.clip(np.sum(probs, axis=-1, keepdims=True), a_min=1e-8, a_max=None)
    return probs


def _logsumexp_pair_numpy(x, y):
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    m = np.maximum(x, y)
    return m + np.log(np.exp(x - m) + np.exp(y - m))


def _merge_na_logits_to_acgu(na_logits):
    na_logits = np.asarray(na_logits, dtype=np.float32)
    if na_logits.shape[-1] >= 8:
        return np.stack(
            [
                _logsumexp_pair_numpy(na_logits[..., 0], na_logits[..., 4]),
                _logsumexp_pair_numpy(na_logits[..., 1], na_logits[..., 5]),
                _logsumexp_pair_numpy(na_logits[..., 2], na_logits[..., 6]),
                _logsumexp_pair_numpy(na_logits[..., 3], na_logits[..., 7]),
            ],
            axis=-1,
        )
    if na_logits.shape[-1] >= 4:
        return na_logits[..., :4]
    return na_logits


def _compute_residue_type_scores(pred_aatype, prot_mask):
    pred_aatype = np.asarray(pred_aatype, dtype=np.float32)
    prot_mask = np.asarray(prot_mask, dtype=bool)

    entropy = np.zeros((len(pred_aatype),), dtype=np.float32)
    confidence = np.zeros((len(pred_aatype),), dtype=np.float32)

    if np.any(prot_mask):
        prot_probs = _softmax_numpy(pred_aatype[prot_mask, :num_prot])
        prot_entropy = -np.sum(prot_probs * np.log(np.clip(prot_probs, 1e-8, None)), axis=-1)
        prot_max_entropy = np.log(float(num_prot))
        entropy[prot_mask] = prot_entropy
        confidence[prot_mask] = 1.0 - prot_entropy / prot_max_entropy

    na_mask = ~prot_mask
    if np.any(na_mask):
        na_logits = _merge_na_logits_to_acgu(pred_aatype[na_mask, num_prot:])
        na_vocab = na_logits.shape[-1]
        if na_vocab <= 1:
            na_entropy = np.zeros((len(na_logits),), dtype=np.float32)
            na_conf = np.ones((len(na_logits),), dtype=np.float32)
        else:
            na_probs = _softmax_numpy(na_logits)
            na_entropy = -np.sum(na_probs * np.log(np.clip(na_probs, 1e-8, None)), axis=-1)
            na_max_entropy = np.log(float(na_vocab))
            na_conf = 1.0 - na_entropy / na_max_entropy
        entropy[na_mask] = na_entropy
        confidence[na_mask] = na_conf

    confidence = np.clip(confidence, 0.0, 1.0).astype(np.float32)
    return entropy.astype(np.float32), confidence


def _expand_residue_scalar_to_atom_bfactor(scores, num_atoms=23):
    scores = np.asarray(scores, dtype=np.float32).reshape(-1, 1)
    return np.repeat(scores, num_atoms, axis=-1)


def _write_residue_type_score_npz(filename, final_results):
    np.savez_compressed(
        filename,
        residue_type_entropy=np.asarray(final_results["residue_type_entropy"], dtype=np.float32),
        residue_type_confidence=np.asarray(final_results["residue_type_confidence"], dtype=np.float32),
        pred_rmsd=np.asarray(final_results["pred_rmsd"], dtype=np.float32),
        prot_mask=np.asarray(final_results["prot_mask"], dtype=bool),
        pred_aatype=np.asarray(final_results["pred_aatype"], dtype=np.float32),
    )
    print(f"# Output residue type scores to {filename}")


def _build_chain_order_recycle_state(reordered_final_results, chains):
    if len(chains) == 0:
        return None

    flat_indices = np.concatenate(chains, axis=0)
    pred_aatype = np.asarray(
        reordered_final_results["pred_aatype"][flat_indices],
        dtype=np.float32,
    )
    prot_mask = np.asarray(reordered_final_results["prot_mask"][flat_indices], dtype=bool)

    prev_aa_probs = np.zeros_like(pred_aatype, dtype=np.float32)
    if np.any(prot_mask):
        prev_aa_probs[prot_mask, :num_prot] = _softmax_numpy(
            pred_aatype[prot_mask, :num_prot]
        )
    if np.any(~prot_mask):
        prev_aa_probs[~prot_mask, num_prot:] = _softmax_numpy(
            pred_aatype[~prot_mask, num_prot:]
        )

    prev_rmsd = np.asarray(
        reordered_final_results["pred_rmsd"][flat_indices],
        dtype=np.float32,
    ).reshape(-1, 1)

    recycle_state = {
        "prev_aa_probs": prev_aa_probs,
        "prev_rmsd": prev_rmsd,
    }
    if "recycle_node_state" in reordered_final_results:
        recycle_state["prev_node"] = np.asarray(
            reordered_final_results["recycle_node_state"][flat_indices],
            dtype=np.float32,
        )

    return recycle_state


def _write_residue_subset_file(
    filename,
    atom_pos,
    atom_mask,
    aatype,
    residue_index=None,
    bfactor=None,
):
    if len(atom_pos) == 0:
        return

    if residue_index is None:
        residue_index = np.arange(len(atom_pos), dtype=np.int32)
    if bfactor is None:
        bfactor = np.zeros_like(atom_mask, dtype=np.float32)

    chains_atom_pos_to_pdb(
        filename=filename,
        chains_atom_pos=[atom_pos],
        chains_atom_mask=[atom_mask],
        chains_res_type=[aatype],
        chains_res_idx=[residue_index],
        chains_bfactor=[bfactor],
        suffix="cif",
    )
    print(f"# Output raw residue subset to {filename}")


def _build_protein_chain_outputs(
    reordered_final_results,
    prot_chains,
    prot_seqs,
    hmm_temp_dir,
    dummy_atom_pos,
):
    if len(prot_chains) == 0:
        return [], [], [], []

    if len(prot_seqs) == 0:
        chains_res_type = [
            np.argmax(reordered_final_results["pred_aatype"][chain][..., :num_prot], axis=-1)
            for chain in prot_chains
        ]
        return prot_chains, chains_res_type, prot_chains, chains_res_type

    # Keep CA positions in the same global index space as `prot_chains`.
    ca_pos = dummy_atom_pos[..., 1, :]
    chains_prot_mask = [np.ones(len(chain), dtype=bool) for chain in prot_chains]
    chains_aa_logits = [reordered_final_results["pred_aatype"][chain] for chain in prot_chains]

    fix_chains_output = fix_chains_pipeline(
        prot_sequences=prot_seqs,
        rna_sequences=[],
        dna_sequences=[],
        chains=prot_chains,
        chain_aa_logits=chains_aa_logits,
        ca_pos=ca_pos,
        chain_prot_mask=chains_prot_mask,
        base_dir=hmm_temp_dir,
        postprocess=False,
    )

    print("# Fix protein chain match")
    fix_chains_output = fix_chains_output._replace(
        best_match_output=fix_match(
            fix_chains_output.best_match_output,
            prot_seqs,
            match_score_cutoff=0.60,
            len_cutoff=10,
        )
    )

    before_chains = fix_chains_output.chains
    before_chain_types = fix_chains_output.best_match_output.new_sequences

    after_fix_chains_output = prune_and_connect_chains(
        chains=fix_chains_output.chains,
        best_match_output=fix_chains_output.best_match_output,
        ca_pos=ca_pos,
        aggressive_pruning=True,
        chain_prune_length=4,
    )

    return (
        before_chains,
        before_chain_types,
        after_fix_chains_output.chains,
        after_fix_chains_output.best_match_output.new_sequences,
    )


def _map_na_sequence_to_res_type(sequence, is_dna):
    if is_dna:
        mapping = {"A": 20, "C": 21, "G": 22, "U": 23, "N": 29}
    else:
        mapping = {"A": 24, "C": 25, "G": 26, "U": 27, "N": 30}
    return np.array([mapping.get(ch, mapping["N"]) for ch in sequence], dtype=np.int32)


def _maybe_override_homopolymer_dna_chain_type(
    aatype,
    chain_logits,
    matched_sequence,
    *,
    target_vote_threshold=0.60,
    at_vote_threshold=0.85,
    target_prob_threshold=0.55,
    target_margin_threshold=0.10,
):
    matched_sequence = (matched_sequence or "").upper()
    if len(matched_sequence) == 0:
        return aatype, None
    unique_chars = set(matched_sequence)
    if unique_chars == {"A"}:
        target_char = "A"
        target_restype = 20
        target_idx = 0
    elif unique_chars == {"U"}:
        target_char = "T"
        target_restype = 23
        target_idx = 3
    else:
        return aatype, None

    chain_logits = np.asarray(chain_logits)
    if chain_logits.ndim != 2 or chain_logits.shape[-1] < num_prot + 4:
        return aatype, None

    dna_logits = chain_logits[..., num_prot : num_prot + 4]
    if dna_logits.shape[0] == 0:
        return aatype, None

    dna_logits = dna_logits - np.max(dna_logits, axis=-1, keepdims=True)
    dna_probs = np.exp(dna_logits)
    dna_probs = dna_probs / np.clip(dna_probs.sum(axis=-1, keepdims=True), 1e-8, None)

    mean_probs = dna_probs.mean(axis=0)
    argmax_idx = np.argmax(dna_probs, axis=-1)
    target_vote = float(np.mean(argmax_idx == target_idx))
    at_vote = float(np.mean(np.isin(argmax_idx, [0, 3])))
    target_prob = float(mean_probs[target_idx])
    other_best = float(np.max(np.delete(mean_probs, target_idx)))
    target_margin = target_prob - other_best

    should_override = (
        target_vote >= target_vote_threshold
        and at_vote >= at_vote_threshold
        and target_prob >= target_prob_threshold
        and target_margin >= target_margin_threshold
    )
    if not should_override:
        return aatype, {
            "target_char": target_char,
            "target_vote": target_vote,
            "at_vote": at_vote,
            "target_prob": target_prob,
            "target_margin": target_margin,
            "applied": False,
        }

    overridden = np.full_like(aatype, fill_value=target_restype)
    return overridden, {
        "target_char": target_char,
        "target_vote": target_vote,
        "at_vote": at_vote,
        "target_prob": target_prob,
        "target_margin": target_margin,
        "applied": True,
    }


def _dna_homopolymer_target_char(sequence):
    sequence = (sequence or "").upper()
    if len(sequence) == 0:
        return None
    unique_chars = set(sequence)
    if unique_chars == {"A"}:
        return "A"
    if unique_chars == {"U"}:
        return "T"
    return None


def _summarize_dna_chain_logits(chain_logits):
    chain_logits = np.asarray(chain_logits)
    if chain_logits.ndim != 2 or chain_logits.shape[-1] < num_prot + 4:
        return None
    dna_logits = chain_logits[..., num_prot : num_prot + 4]
    if dna_logits.shape[0] == 0:
        return None
    dna_logits = dna_logits - np.max(dna_logits, axis=-1, keepdims=True)
    dna_probs = np.exp(dna_logits)
    dna_probs = dna_probs / np.clip(dna_probs.sum(axis=-1, keepdims=True), 1e-8, None)
    mean_probs = dna_probs.mean(axis=0)
    argmax_idx = np.argmax(dna_probs, axis=-1)
    return {
        "a_prob": float(mean_probs[0]),
        "t_prob": float(mean_probs[3]),
        "at_vote": float(np.mean(np.isin(argmax_idx, [0, 3]))),
    }


def _apply_global_homopolymer_dna_assignment(
    chains_res_type,
    chains_aa_logits,
    sequence_idxs,
    seqs,
    seqs_is_dna,
    *,
    at_vote_threshold=0.85,
    assigned_prob_threshold=0.50,
    assigned_margin_threshold=0.05,
):
    del sequence_idxs
    if not seqs or not all(seqs_is_dna):
        return chains_res_type

    homopolymer_targets = [_dna_homopolymer_target_char(seq) for seq in seqs]
    if any(target is None for target in homopolymer_targets):
        return chains_res_type

    total_a_len = sum(len(seq) for seq, target in zip(seqs, homopolymer_targets) if target == "A")
    total_t_len = sum(len(seq) for seq, target in zip(seqs, homopolymer_targets) if target == "T")
    total_len = total_a_len + total_t_len
    if total_len <= 0:
        return chains_res_type

    candidate_summaries = []
    for chain_idx, chain_logits in enumerate(chains_aa_logits):
        summary = _summarize_dna_chain_logits(chain_logits)
        if summary is None:
            continue
        if summary["at_vote"] < at_vote_threshold:
            continue
        summary = dict(summary)
        summary["chain_idx"] = chain_idx
        summary["delta_a_minus_t"] = summary["a_prob"] - summary["t_prob"]
        candidate_summaries.append(summary)

    if not candidate_summaries:
        return chains_res_type

    candidate_summaries.sort(key=lambda item: item["delta_a_minus_t"])
    t_fraction = total_t_len / float(total_len)
    if total_a_len > 0 and total_t_len > 0:
        num_t = int(round(len(candidate_summaries) * t_fraction))
        num_t = max(1, min(len(candidate_summaries) - 1, num_t))
    elif total_t_len > 0:
        num_t = len(candidate_summaries)
    else:
        num_t = 0

    assignments = []
    for summary in candidate_summaries[:num_t]:
        assignments.append((summary, "T", 23, summary["t_prob"], summary["t_prob"] - summary["a_prob"]))
    for summary in candidate_summaries[num_t:]:
        assignments.append((summary, "A", 20, summary["a_prob"], summary["a_prob"] - summary["t_prob"]))

    for summary, target_char, target_restype, assigned_prob, assigned_margin in assignments:
        applied = assigned_prob >= assigned_prob_threshold and assigned_margin >= assigned_margin_threshold
        print(
            "# Global homopolymer DNA assignment chain={} target={} applied={} a_prob={:.3f} t_prob={:.3f} at_vote={:.3f} margin={:.3f}".format(
                summary["chain_idx"],
                target_char,
                applied,
                summary["a_prob"],
                summary["t_prob"],
                summary["at_vote"],
                assigned_margin,
            )
        )
        if applied:
            chains_res_type[summary["chain_idx"]][:] = target_restype

    return chains_res_type


def _filter_short_chains(chains, chains_res_type, min_len):
    if min_len <= 1:
        return chains, chains_res_type
    kept = [
        (chain, res_type)
        for chain, res_type in zip(chains, chains_res_type)
        if len(chain) >= min_len
    ]
    if len(kept) == 0:
        return [], []
    new_chains, new_chain_types = zip(*kept)
    return list(new_chains), list(new_chain_types)


def _filter_short_chains_with_fallback_masks(chains, chains_res_type, fallback_masks, min_len):
    if min_len <= 1:
        return chains, chains_res_type, fallback_masks
    kept = [
        (chain, res_type, fallback_mask)
        for chain, res_type, fallback_mask in zip(chains, chains_res_type, fallback_masks)
        if len(chain) >= min_len
    ]
    if len(kept) == 0:
        return [], [], []
    new_chains, new_chain_types, new_fallback_masks = zip(*kept)
    return list(new_chains), list(new_chain_types), list(new_fallback_masks)


def _predict_na_chain_types(reordered_final_results, na_chains, *, rna_only=False):
    chain_types = []
    for chain in na_chains:
        na_logits = np.asarray(reordered_final_results["pred_aatype"][chain][..., num_prot:], dtype=np.float32)
        if rna_only and na_logits.shape[-1] >= 8:
            shared_logits = np.stack(
                [
                    na_logits[..., 0] + na_logits[..., 4],
                    na_logits[..., 1] + na_logits[..., 5],
                    na_logits[..., 2] + na_logits[..., 6],
                    na_logits[..., 3] + na_logits[..., 7],
                ],
                axis=-1,
            )
            chain_types.append(np.argmax(shared_logits, axis=-1) + num_prot + 4)
        elif rna_only and na_logits.shape[-1] >= 4:
            chain_types.append(np.argmax(na_logits[..., -4:], axis=-1) + num_prot + 4)
        else:
            chain_types.append(np.argmax(na_logits, axis=-1) + num_prot)
    return chain_types


def _pack_shared_na_logits(na_logits):
    na_logits = np.asarray(na_logits, dtype=np.float32)
    full_logits = np.full(
        na_logits.shape[:-1] + (num_prot + 4,),
        fill_value=-100.0,
        dtype=np.float32,
    )
    full_logits[..., num_prot:] = na_logits
    return full_logits


def _sample_vanilla_na_chain_logits(reordered_final_results, na_chains, na_aa_logits_data):
    if na_aa_logits_data is None:
        return None

    vanilla_chain_logits = []
    for chain in na_chains:
        c4_positions = reordered_final_results["_dummy_atom_pos"][chain][..., 1, :]
        chain_logits = sample_na_aa_logits(
            na_aa_logits_data["map"],
            c4_positions,
            na_aa_logits_data["origin"],
            na_aa_logits_data["voxel_size"],
        )
        vanilla_chain_logits.append(_pack_shared_na_logits(chain_logits))
    return vanilla_chain_logits


def _summarize_na_mode_selection(debug_records, chains):
    total_residues = 0
    exact_residues = 0
    pp_residues = 0
    for chain_idx, chain in enumerate(chains):
        chain_len = len(chain)
        total_residues += chain_len
        chosen_mode = "exact"
        if debug_records is not None and chain_idx < len(debug_records):
            chosen_mode = str(debug_records[chain_idx].get("chosen_mode", "exact"))
        if chosen_mode == "pp":
            pp_residues += chain_len
        else:
            exact_residues += chain_len
    denom = max(total_residues, 1)
    return {
        "total_residues": int(total_residues),
        "exact_residues": int(exact_residues),
        "pp_residues": int(pp_residues),
        "exact_ratio": float(exact_residues / denom),
        "pp_ratio": float(pp_residues / denom),
    }



def _select_best_na_match_output(
    tta_output,
    vanilla_output,
    chains,
    *,
    tta_debug_records=None,
    vanilla_debug_records=None,
    pp_mode_penalty=0.05,
    vanilla_enable_tta_score_threshold=0.50,
    vanilla_min_score_gain=0.15,
    vanilla_pp_min_score=0.85,
    vanilla_exact_min_score=0.70,
):
    tta_match = tta_output.best_match_output
    tta_score_sum = float(np.sum(tta_match.match_scores))

    if vanilla_output is None:
        summary = {
            "tta_score_sum": tta_score_sum,
            "vanilla_score_sum": None,
            "tta_residue_ratio": 1.0,
            "vanilla_residue_ratio": 0.0,
            "exact_ratio": None,
            "pp_ratio": None,
            "pp_mode_penalty": float(pp_mode_penalty),
            "vanilla_enable_tta_score_threshold": float(vanilla_enable_tta_score_threshold),
            "vanilla_min_score_gain": float(vanilla_min_score_gain),
            "vanilla_pp_min_score": float(vanilla_pp_min_score),
            "vanilla_exact_min_score": float(vanilla_exact_min_score),
            "vanilla_available": False,
        }
        selected_debug = []
        mode_summary = _summarize_na_mode_selection(tta_debug_records or [], chains)
        summary["exact_ratio"] = float(mode_summary["exact_ratio"])
        summary["pp_ratio"] = float(mode_summary["pp_ratio"])
        for chain_idx, chain in enumerate(chains):
            chain_len = len(chain)
            tta_record = (
                dict(tta_debug_records[chain_idx])
                if tta_debug_records is not None and chain_idx < len(tta_debug_records)
                else {
                    "chain_idx": int(chain_idx),
                    "source": "tta",
                    "exact_score": float(tta_match.match_scores[chain_idx]),
                    "pp_score": None,
                    "chosen_mode": "exact",
                    "chosen_raw_score": float(tta_match.match_scores[chain_idx]),
                    "chosen_effective_score": float(tta_match.match_scores[chain_idx]),
                }
            )
            selected_debug.append({
                "chain_idx": int(chain_idx),
                "chain_len": int(chain_len),
                "tta_exact_score": tta_record.get("exact_score"),
                "tta_pp_score": tta_record.get("pp_score"),
                "tta_selected_mode": tta_record.get("chosen_mode", "exact"),
                "tta_selected_raw_score": tta_record.get("chosen_raw_score", float(tta_match.match_scores[chain_idx])),
                "tta_selected_effective_score": tta_record.get("chosen_effective_score", float(tta_match.match_scores[chain_idx])),
                "vanilla_exact_score": None,
                "vanilla_pp_score": None,
                "vanilla_selected_mode": None,
                "vanilla_selected_raw_score": None,
                "vanilla_selected_effective_score": None,
                "chosen_source": "tta",
                "chosen_mode": tta_record.get("chosen_mode", "exact"),
                "chosen_raw_score": tta_record.get("chosen_raw_score", float(tta_match.match_scores[chain_idx])),
                "chosen_effective_score": tta_record.get("chosen_effective_score", float(tta_match.match_scores[chain_idx])),
                "score_margin_vs_other_source": None,
            })
        return tta_output, np.ones((len(chains),), dtype=bool), selected_debug, summary

    if len(tta_output.chains) != len(vanilla_output.chains):
        print("# WARN NA TTA/vanilla chain counts differ, keep TTA branch selection")
        return _select_best_na_match_output(
            tta_output,
            None,
            chains,
            tta_debug_records=tta_debug_records,
            vanilla_debug_records=vanilla_debug_records,
            pp_mode_penalty=pp_mode_penalty,
            vanilla_enable_tta_score_threshold=vanilla_enable_tta_score_threshold,
            vanilla_min_score_gain=vanilla_min_score_gain,
            vanilla_pp_min_score=vanilla_pp_min_score,
            vanilla_exact_min_score=vanilla_exact_min_score,
        )

    vanilla_match = vanilla_output.best_match_output
    vanilla_score_sum = float(np.sum(vanilla_match.match_scores))
    print("# NA TTA branch score = {:.4f}".format(tta_score_sum))
    print("# NA vanilla branch score = {:.4f}".format(vanilla_score_sum))

    use_tta = []
    selected = {
        "new_sequences": [],
        "residue_idxs": [],
        "sequence_idxs": [],
        "key_start_matches": [],
        "key_end_matches": [],
        "match_scores": [],
        "hmm_output_match_sequences": [],
        "exists_in_sequence_mask": [],
        "is_nucleotide": [],
    }
    selected_debug = []
    tta_residue_count = 0
    vanilla_residue_count = 0
    exact_residue_count = 0
    pp_residue_count = 0
    total_residue_count = 0

    for chain_idx in range(len(chains)):
        chain_len = len(chains[chain_idx])
        total_residue_count += chain_len
        tta_record = (
            dict(tta_debug_records[chain_idx])
            if tta_debug_records is not None and chain_idx < len(tta_debug_records)
            else {
                "chain_idx": int(chain_idx),
                "source": "tta",
                "exact_score": float(tta_match.match_scores[chain_idx]),
                "pp_score": None,
                "chosen_mode": "exact",
                "chosen_raw_score": float(tta_match.match_scores[chain_idx]),
                "chosen_effective_score": float(tta_match.match_scores[chain_idx]),
            }
        )
        vanilla_record = (
            dict(vanilla_debug_records[chain_idx])
            if vanilla_debug_records is not None and chain_idx < len(vanilla_debug_records)
            else {
                "chain_idx": int(chain_idx),
                "source": "vanilla",
                "exact_score": float(vanilla_match.match_scores[chain_idx]),
                "pp_score": None,
                "chosen_mode": "exact",
                "chosen_raw_score": float(vanilla_match.match_scores[chain_idx]),
                "chosen_effective_score": float(vanilla_match.match_scores[chain_idx]),
            }
        )

        tta_raw_score = float(tta_record.get("chosen_raw_score", float(tta_match.match_scores[chain_idx])))
        vanilla_raw_score = float(vanilla_record.get("chosen_raw_score", float(vanilla_match.match_scores[chain_idx])))
        tta_mode = str(tta_record.get("chosen_mode", "exact"))
        vanilla_mode = str(vanilla_record.get("chosen_mode", "exact"))
        tta_effective_score = float(tta_raw_score - (pp_mode_penalty if tta_mode == "pp" else 0.0))
        vanilla_effective_score = float(vanilla_raw_score - (pp_mode_penalty if vanilla_mode == "pp" else 0.0))

        vanilla_has_rescue_score = (
            vanilla_raw_score >= vanilla_exact_min_score
            if vanilla_mode == "exact"
            else vanilla_raw_score >= vanilla_pp_min_score
        )
        allow_vanilla_override = (
            tta_raw_score < vanilla_enable_tta_score_threshold
            and vanilla_has_rescue_score
            and (vanilla_raw_score - tta_raw_score) >= vanilla_min_score_gain
        )

        if not allow_vanilla_override:
            choose_tta = True
        elif tta_effective_score > vanilla_effective_score:
            choose_tta = True
        elif vanilla_effective_score > tta_effective_score:
            choose_tta = False
        elif tta_mode != vanilla_mode:
            choose_tta = (tta_mode == "exact")
        elif tta_raw_score != vanilla_raw_score:
            choose_tta = (tta_raw_score >= vanilla_raw_score)
        else:
            choose_tta = True

        use_tta.append(choose_tta)
        source_match = tta_match if choose_tta else vanilla_match
        chosen_record = tta_record if choose_tta else vanilla_record
        chosen_source = "tta" if choose_tta else "vanilla"
        chosen_mode = str(chosen_record.get("chosen_mode", "exact"))
        chosen_raw_score = tta_raw_score if choose_tta else vanilla_raw_score
        chosen_effective_score = tta_effective_score if choose_tta else vanilla_effective_score

        if choose_tta:
            tta_residue_count += chain_len
        else:
            vanilla_residue_count += chain_len
        if chosen_mode == "pp":
            pp_residue_count += chain_len
        else:
            exact_residue_count += chain_len

        selected["new_sequences"].append(source_match.new_sequences[chain_idx])
        selected["residue_idxs"].append(source_match.residue_idxs[chain_idx])
        selected["sequence_idxs"].append(source_match.sequence_idxs[chain_idx])
        selected["key_start_matches"].append(source_match.key_start_matches[chain_idx])
        selected["key_end_matches"].append(source_match.key_end_matches[chain_idx])
        selected["match_scores"].append(source_match.match_scores[chain_idx])
        selected["hmm_output_match_sequences"].append(source_match.hmm_output_match_sequences[chain_idx])
        selected["exists_in_sequence_mask"].append(source_match.exists_in_sequence_mask[chain_idx])
        selected["is_nucleotide"].append(source_match.is_nucleotide[chain_idx])
        selected_debug.append({
            "chain_idx": int(chain_idx),
            "chain_len": int(chain_len),
            "tta_exact_score": tta_record.get("exact_score"),
            "tta_pp_score": tta_record.get("pp_score"),
            "tta_selected_mode": tta_mode,
            "tta_selected_raw_score": tta_raw_score,
            "tta_selected_effective_score": tta_effective_score,
            "vanilla_exact_score": vanilla_record.get("exact_score"),
            "vanilla_pp_score": vanilla_record.get("pp_score"),
            "vanilla_selected_mode": vanilla_mode,
            "vanilla_selected_raw_score": vanilla_raw_score,
            "vanilla_selected_effective_score": vanilla_effective_score,
            "chosen_source": chosen_source,
            "chosen_mode": chosen_mode,
            "chosen_raw_score": chosen_raw_score,
            "chosen_effective_score": chosen_effective_score,
            "score_margin_vs_other_source": float(abs(tta_effective_score - vanilla_effective_score)),
            "allow_vanilla_override": bool(allow_vanilla_override),
            "vanilla_has_rescue_score": bool(vanilla_has_rescue_score),
        })

    denom = max(total_residue_count, 1)
    print(
        "# Using TTA ratio = {:.4f} using vanilla ratio = {:.4f}".format(
            tta_residue_count / denom,
            vanilla_residue_count / denom,
        )
    )
    print(
        "# Using exact ratio = {:.4f} using PP ratio = {:.4f}".format(
            exact_residue_count / denom,
            pp_residue_count / denom,
        )
    )

    merged_output = MatchToSequence(
        new_sequences=selected["new_sequences"],
        residue_idxs=selected["residue_idxs"],
        sequence_idxs=np.array(selected["sequence_idxs"]),
        key_start_matches=np.array(selected["key_start_matches"]),
        key_end_matches=np.array(selected["key_end_matches"]),
        match_scores=np.array(selected["match_scores"]),
        hmm_output_match_sequences=selected["hmm_output_match_sequences"],
        exists_in_sequence_mask=selected["exists_in_sequence_mask"],
        is_nucleotide=selected["is_nucleotide"],
    )
    summary = {
        "tta_score_sum": float(tta_score_sum),
        "vanilla_score_sum": float(vanilla_score_sum),
        "tta_residue_ratio": float(tta_residue_count / denom),
        "vanilla_residue_ratio": float(vanilla_residue_count / denom),
        "exact_ratio": float(exact_residue_count / denom),
        "pp_ratio": float(pp_residue_count / denom),
        "pp_mode_penalty": float(pp_mode_penalty),
        "vanilla_enable_tta_score_threshold": float(vanilla_enable_tta_score_threshold),
        "vanilla_min_score_gain": float(vanilla_min_score_gain),
        "vanilla_pp_min_score": float(vanilla_pp_min_score),
        "vanilla_exact_min_score": float(vanilla_exact_min_score),
        "vanilla_available": True,
    }
    return (
        tta_output._replace(best_match_output=merged_output),
        np.asarray(use_tta, dtype=bool),
        selected_debug,
        summary,
    )



def _select_best_na_pp_rescue_output(
    exact_output,
    pp_rna_output,
    pp_dna_output,
    seqs_is_dna,
    *,
    source_label="tta",
    exact_match_score_threshold=0.50,
    pp_match_score_threshold=0.70,
    min_score_gain=0.20,
    corroboration_debug_records=None,
    corroboration_score_threshold=0.65,
):
    disable_pp_rescue = _disable_na_pp_rescue()
    if disable_pp_rescue:
        print(f"# NA PP rescue disabled by env for source={source_label}")
    enable_moderate_rescue = _enable_na_pp_moderate_rescue()
    moderate_exact_threshold = _na_pp_moderate_exact_threshold()
    moderate_score_threshold = _na_pp_moderate_score_threshold()
    moderate_min_gain = _na_pp_moderate_min_gain()
    if enable_moderate_rescue:
        print(
            "# NA moderate PP rescue enabled for source={} exact<thr {:.3f} pp>=thr {:.3f} gain>=thr {:.3f}".format(
                source_label,
                moderate_exact_threshold,
                moderate_score_threshold,
                moderate_min_gain,
            )
        )

    selected = {
        "new_sequences": [],
        "residue_idxs": [],
        "sequence_idxs": [],
        "key_start_matches": [],
        "key_end_matches": [],
        "match_scores": [],
        "hmm_output_match_sequences": [],
        "exists_in_sequence_mask": [],
        "is_nucleotide": [],
    }
    use_pp = []
    debug_records = []

    exact_match = exact_output.best_match_output
    if pp_rna_output is not None and len(exact_output.chains) != len(pp_rna_output.chains):
        print("# WARN NA exact/PP-RNA chain counts differ, skip RNA PP rescue")
        pp_rna_output = None
    if pp_dna_output is not None and len(exact_output.chains) != len(pp_dna_output.chains):
        print("# WARN NA exact/PP-DNA chain counts differ, skip DNA PP rescue")
        pp_dna_output = None

    for chain_idx in range(len(exact_output.chains)):
        exact_score = float(exact_match.match_scores[chain_idx])
        exact_seq_idx = int(exact_match.sequence_idxs[chain_idx])
        pp_match = None
        label = None
        if 0 <= exact_seq_idx < len(seqs_is_dna):
            if seqs_is_dna[exact_seq_idx]:
                pp_match = None if pp_dna_output is None else pp_dna_output.best_match_output
                label = "DNA"
            else:
                pp_match = None if pp_rna_output is None else pp_rna_output.best_match_output
                label = "RNA"

        take_pp = False
        pp_score = None
        pp_seq_idx = None
        corroboration_support = None
        if pp_match is not None:
            pp_score = float(pp_match.match_scores[chain_idx])
            pp_seq_idx = int(pp_match.sequence_idxs[chain_idx])
            take_pp = (
                (not disable_pp_rescue)
                and exact_score < exact_match_score_threshold
                and pp_score >= pp_match_score_threshold
                and (pp_score - exact_score) >= min_score_gain
            )
            corr_pp_score = None
            if corroboration_debug_records is not None and chain_idx < len(corroboration_debug_records):
                corr = corroboration_debug_records[chain_idx]
                corr_exact = corr.get("exact_score")
                corr_pp = corr.get("pp_score")
                corr_best = max(
                    float(corr_exact) if corr_exact is not None else -1.0,
                    float(corr_pp) if corr_pp is not None else -1.0,
                )
                corr_pp_score = float(corr_pp) if corr_pp is not None else None
                if take_pp:
                    corroboration_support = corr_best
                    if corr_best < corroboration_score_threshold:
                        take_pp = False
            if (
                (not take_pp)
                and enable_moderate_rescue
                and corroboration_debug_records is not None
                and exact_score < moderate_exact_threshold
                and pp_score >= moderate_score_threshold
                and (pp_score - exact_score) >= moderate_min_gain
                and corr_pp_score is not None
                and corr_pp_score >= moderate_score_threshold
            ):
                take_pp = True
                corroboration_support = corr_pp_score

        chosen = pp_match if take_pp and pp_match is not None else exact_match
        chosen_raw_score = float(chosen.match_scores[chain_idx])
        chosen_mode = "pp" if take_pp else "exact"
        use_pp.append(take_pp)
        selected["new_sequences"].append(chosen.new_sequences[chain_idx])
        selected["residue_idxs"].append(chosen.residue_idxs[chain_idx])
        selected["sequence_idxs"].append(chosen.sequence_idxs[chain_idx])
        selected["key_start_matches"].append(chosen.key_start_matches[chain_idx])
        selected["key_end_matches"].append(chosen.key_end_matches[chain_idx])
        selected["match_scores"].append(chosen.match_scores[chain_idx])
        selected["hmm_output_match_sequences"].append(chosen.hmm_output_match_sequences[chain_idx])
        selected["exists_in_sequence_mask"].append(chosen.exists_in_sequence_mask[chain_idx])
        selected["is_nucleotide"].append(chosen.is_nucleotide[chain_idx])
        debug_records.append({
            "chain_idx": int(chain_idx),
            "source": source_label,
            "sequence_label": label,
            "exact_score": float(exact_score),
            "pp_score": None if pp_score is None else float(pp_score),
            "exact_sequence_idx": int(exact_seq_idx),
            "pp_sequence_idx": None if pp_seq_idx is None else int(pp_seq_idx),
            "chosen_mode": chosen_mode,
            "chosen_raw_score": float(chosen_raw_score),
            "chosen_effective_score": float(chosen_raw_score),
            "pp_used": bool(take_pp),
            "pp_disabled": bool(disable_pp_rescue),
            "corroboration_support": None if corroboration_support is None else float(corroboration_support),
            "corroboration_score_threshold": float(corroboration_score_threshold),
            "moderate_rescue_enabled": bool(enable_moderate_rescue),
            "moderate_exact_threshold": float(moderate_exact_threshold),
            "moderate_score_threshold": float(moderate_score_threshold),
            "moderate_min_gain": float(moderate_min_gain),
        })
        if take_pp and pp_score is not None:
            print(
                "# NA PP rescue chain={} source={} type={} exact_match_score={:.4f} pp_match_score={:.4f}".format(
                    chain_idx,
                    source_label,
                    label,
                    exact_score,
                    pp_score,
                )
            )

    merged_output = MatchToSequence(
        new_sequences=selected["new_sequences"],
        residue_idxs=selected["residue_idxs"],
        sequence_idxs=np.array(selected["sequence_idxs"]),
        key_start_matches=np.array(selected["key_start_matches"]),
        key_end_matches=np.array(selected["key_end_matches"]),
        match_scores=np.array(selected["match_scores"]),
        hmm_output_match_sequences=selected["hmm_output_match_sequences"],
        exists_in_sequence_mask=selected["exists_in_sequence_mask"],
        is_nucleotide=selected["is_nucleotide"],
    )
    return exact_output._replace(best_match_output=merged_output), np.asarray(use_pp, dtype=bool), debug_records



def _build_na_chain_outputs(
    reordered_final_results,
    na_chains,
    rna_seqs,
    dna_seqs,
    hmm_temp_dir,
    na_aa_logits_data=None,
    min_chain_len=1,
    fallback_to_predicted_na_types=True,
):
    if len(na_chains) == 0:
        return [], [], [], [], [], []

    if len(rna_seqs) + len(dna_seqs) == 0:
        before_chains = list(na_chains)
        before_chain_types = _predict_na_chain_types(reordered_final_results, before_chains, rna_only=True)
        before_fallback_masks = [
            np.ones((len(chain),), dtype=bool) for chain in before_chains
        ]
        print(
            "# NA sequences not provided; keep predicted NA types and apply length-only filtering (min_len = {})".format(
                min_chain_len
            )
        )
        after_chains, after_chain_types, after_fallback_masks = _filter_short_chains_with_fallback_masks(
            before_chains,
            before_chain_types,
            before_fallback_masks,
            min_chain_len,
        )
        return (
            before_chains,
            before_chain_types,
            after_chains,
            after_chain_types,
            before_fallback_masks,
            after_fallback_masks,
        )

    rna_seqs_sanitized = [_sanitize_na_sequence(seq) for seq in rna_seqs]
    dna_seqs_sanitized = [_sanitize_na_sequence(seq) for seq in dna_seqs]
    seqs = rna_seqs_sanitized + dna_seqs_sanitized
    seqs_is_dna = [False] * len(rna_seqs_sanitized) + [True] * len(dna_seqs_sanitized)
    first_character = seqs[0][0]
    seqs_is_all_same = np.all(
        np.asarray([all(ch == first_character for ch in seq) for seq in seqs], dtype=bool)
    )
    print(f"# NA seqs is all same = {seqs_is_all_same}")

    # Keep C1'/CA proxy positions in the same global index space as `na_chains`.
    ca_pos = reordered_final_results["_dummy_atom_pos"][..., 1, :]
    chains_prot_mask = [np.zeros(len(chain), dtype=bool) for chain in na_chains]
    chains_aa_logits = [reordered_final_results["pred_aatype"][chain] for chain in na_chains]

    tta_exact_output = fix_chains_pipeline(
        prot_sequences=[],
        rna_sequences=rna_seqs_sanitized,
        dna_sequences=dna_seqs_sanitized,
        chains=na_chains,
        chain_aa_logits=chains_aa_logits,
        ca_pos=ca_pos,
        chain_prot_mask=chains_prot_mask,
        chain_confidences=None,
        base_dir=hmm_temp_dir,
        postprocess=False,
        do_pp=False,
    )
    tta_pp_chain_logits = [reordered_final_results["pred_aatype"][chain] for chain in tta_exact_output.chains]
    tta_pp_chain_masks = [np.zeros(len(chain), dtype=bool) for chain in tta_exact_output.chains]
    tta_pp_rna_output = None
    tta_pp_dna_output = None
    if len(rna_seqs_sanitized) > 0:
        tta_pp_rna_output = fix_chains_pipeline(
            prot_sequences=[],
            rna_sequences=rna_seqs_sanitized,
            dna_sequences=[],
            chains=tta_exact_output.chains,
            chain_aa_logits=tta_pp_chain_logits,
            ca_pos=ca_pos,
            chain_prot_mask=tta_pp_chain_masks,
            chain_confidences=None,
            base_dir=hmm_temp_dir,
            postprocess=False,
            do_pp=True,
        )
    if len(dna_seqs_sanitized) > 0:
        tta_pp_dna_output = fix_chains_pipeline(
            prot_sequences=[],
            rna_sequences=[],
            dna_sequences=dna_seqs_sanitized,
            chains=tta_exact_output.chains,
            chain_aa_logits=tta_pp_chain_logits,
            ca_pos=ca_pos,
            chain_prot_mask=tta_pp_chain_masks,
            chain_confidences=None,
            base_dir=hmm_temp_dir,
            postprocess=False,
            do_pp=True,
        )
    tta_best_output, tta_use_pp, tta_debug_records = _select_best_na_pp_rescue_output(
        tta_exact_output,
        tta_pp_rna_output,
        tta_pp_dna_output,
        seqs_is_dna,
        source_label="tta",
    )
    tta_mode_summary = _summarize_na_mode_selection(tta_debug_records, tta_best_output.chains)
    print("# NA TTA exact score = {:.4f}".format(float(np.sum(tta_exact_output.best_match_output.match_scores))))
    print("# NA TTA selected score = {:.4f}".format(float(np.sum(tta_best_output.best_match_output.match_scores))))
    print(
        "# NA TTA exact ratio = {:.4f} PP ratio = {:.4f}".format(
            tta_mode_summary["exact_ratio"],
            tta_mode_summary["pp_ratio"],
        )
    )

    vanilla_exact_output = None
    vanilla_best_output = None
    vanilla_debug_records = None
    vanilla_mode_summary = None
    vanilla_chain_logits = _sample_vanilla_na_chain_logits(
        reordered_final_results,
        tta_exact_output.chains,
        na_aa_logits_data,
    )
    if vanilla_chain_logits is not None:
        vanilla_exact_output = fix_chains_pipeline(
            prot_sequences=[],
            rna_sequences=rna_seqs_sanitized,
            dna_sequences=dna_seqs_sanitized,
            chains=tta_best_output.chains,
            chain_aa_logits=vanilla_chain_logits,
            ca_pos=ca_pos,
            chain_prot_mask=[np.zeros(len(chain), dtype=bool) for chain in tta_best_output.chains],
            chain_confidences=None,
            base_dir=hmm_temp_dir,
            postprocess=False,
            do_pp=False,
        )
        vanilla_pp_chain_logits = _sample_vanilla_na_chain_logits(
            reordered_final_results,
            vanilla_exact_output.chains,
            na_aa_logits_data,
        )
        vanilla_pp_chain_masks = [np.zeros(len(chain), dtype=bool) for chain in vanilla_exact_output.chains]
        vanilla_pp_rna_output = None
        vanilla_pp_dna_output = None
        if vanilla_pp_chain_logits is not None and len(rna_seqs_sanitized) > 0:
            vanilla_pp_rna_output = fix_chains_pipeline(
                prot_sequences=[],
                rna_sequences=rna_seqs_sanitized,
                dna_sequences=[],
                chains=vanilla_exact_output.chains,
                chain_aa_logits=vanilla_pp_chain_logits,
                ca_pos=ca_pos,
                chain_prot_mask=vanilla_pp_chain_masks,
                chain_confidences=None,
                base_dir=hmm_temp_dir,
                postprocess=False,
                do_pp=True,
            )
        if vanilla_pp_chain_logits is not None and len(dna_seqs_sanitized) > 0:
            vanilla_pp_dna_output = fix_chains_pipeline(
                prot_sequences=[],
                rna_sequences=[],
                dna_sequences=dna_seqs_sanitized,
                chains=vanilla_exact_output.chains,
                chain_aa_logits=vanilla_pp_chain_logits,
                ca_pos=ca_pos,
                chain_prot_mask=vanilla_pp_chain_masks,
                chain_confidences=None,
                base_dir=hmm_temp_dir,
                postprocess=False,
                do_pp=True,
            )
        vanilla_best_output, vanilla_use_pp, vanilla_debug_records = _select_best_na_pp_rescue_output(
            vanilla_exact_output,
            vanilla_pp_rna_output,
            vanilla_pp_dna_output,
            seqs_is_dna,
            source_label="vanilla",
        )
        if _require_vanilla_support_for_na_pp():
            support_threshold = _na_pp_vanilla_support_threshold()
            print(f"# Require vanilla support for TTA PP rescue = True threshold={support_threshold:.3f}")
            tta_best_output, tta_use_pp, tta_debug_records = _select_best_na_pp_rescue_output(
                tta_exact_output,
                tta_pp_rna_output,
                tta_pp_dna_output,
                seqs_is_dna,
                source_label="tta",
                corroboration_debug_records=vanilla_debug_records,
                corroboration_score_threshold=support_threshold,
            )
            tta_mode_summary = _summarize_na_mode_selection(tta_debug_records, tta_best_output.chains)
            print("# NA TTA selected score (with vanilla corroboration) = {:.4f}".format(float(np.sum(tta_best_output.best_match_output.match_scores))))
            print(
                "# NA TTA exact ratio = {:.4f} PP ratio = {:.4f} (with vanilla corroboration)".format(
                    tta_mode_summary["exact_ratio"],
                    tta_mode_summary["pp_ratio"],
                )
            )
        vanilla_mode_summary = _summarize_na_mode_selection(vanilla_debug_records, vanilla_best_output.chains)
        print("# NA vanilla exact score = {:.4f}".format(float(np.sum(vanilla_exact_output.best_match_output.match_scores))))
        print("# NA vanilla selected score = {:.4f}".format(float(np.sum(vanilla_best_output.best_match_output.match_scores))))
        print(
            "# NA vanilla exact ratio = {:.4f} PP ratio = {:.4f}".format(
                vanilla_mode_summary["exact_ratio"],
                vanilla_mode_summary["pp_ratio"],
            )
        )

    fix_chains_output, use_tta, na_alignment_debug_records, na_alignment_summary = _select_best_na_match_output(
        tta_best_output,
        vanilla_best_output,
        tta_best_output.chains,
        tta_debug_records=tta_debug_records,
        vanilla_debug_records=vanilla_debug_records,
    )
    del use_tta

    predicted_chain_types = _predict_na_chain_types(
        reordered_final_results,
        fix_chains_output.chains,
    )

    print("# Fix NA chain match")
    fixed_match = fix_match(
        fix_chains_output.best_match_output,
        [seq.lower() for seq in seqs],
        match_score_cutoff=0.40,
        len_cutoff=6,
    )

    chains_res_type = []
    chain_fallback_masks = []
    num_fallback_chains = 0
    for chain_idx, aatype in enumerate(fixed_match.new_sequences):
        aatype = aatype.copy()
        chain = fix_chains_output.chains[chain_idx]
        seq_idx = fixed_match.sequence_idxs[chain_idx]
        has_valid_sequence = 0 <= seq_idx < len(seqs)
        if not has_valid_sequence:
            if fallback_to_predicted_na_types:
                chains_res_type.append(predicted_chain_types[chain_idx].astype(np.int32))
                chain_fallback_masks.append(np.ones((len(chain),), dtype=bool))
                num_fallback_chains += 1
            else:
                chains_res_type.append(aatype.astype(np.int32))
                chain_fallback_masks.append(np.zeros((len(chain),), dtype=bool))
            continue
        if seqs_is_dna[seq_idx]:
            aatype[aatype == 24] = 20
            aatype[aatype == 25] = 21
            aatype[aatype == 26] = 22
            aatype[aatype == 27] = 23
            aatype[aatype >= 28] = 21
            aatype, homopolymer_override = _maybe_override_homopolymer_dna_chain_type(
                aatype=aatype,
                chain_logits=chains_aa_logits[chain_idx],
                matched_sequence=seqs[seq_idx],
            )
            if homopolymer_override is not None:
                print(
                    "# Homopolymer DNA override chain={} seq_idx={} target={} applied={} target_vote={:.3f} at_vote={:.3f} target_prob={:.3f} target_margin={:.3f}".format(
                        chain_idx,
                        seq_idx,
                        homopolymer_override["target_char"],
                        homopolymer_override["applied"],
                        homopolymer_override["target_vote"],
                        homopolymer_override["at_vote"],
                        homopolymer_override["target_prob"],
                        homopolymer_override["target_margin"],
                    )
                )
        else:
            aatype[aatype >= 28] = 21 + 4
        chains_res_type.append(aatype.astype(np.int32))
        chain_fallback_masks.append(np.zeros((len(chain),), dtype=bool))

    chains_res_type = _apply_global_homopolymer_dna_assignment(
        chains_res_type=chains_res_type,
        chains_aa_logits=chains_aa_logits,
        sequence_idxs=fixed_match.sequence_idxs,
        seqs=seqs,
        seqs_is_dna=seqs_is_dna,
    )

    if seqs_is_all_same:
        for chain_idx, chain_types in enumerate(chains_res_type):
            seq_idx = fixed_match.sequence_idxs[chain_idx]
            if not (0 <= seq_idx < len(seqs)):
                continue
            fill_value = _map_na_sequence_to_res_type(seqs[seq_idx], seqs_is_dna[seq_idx])[0]
            chain_types[:] = fill_value

    if fallback_to_predicted_na_types and num_fallback_chains > 0:
        print(
            "# Fallback to predicted NA types for {} unmatched chain(s)".format(
                num_fallback_chains
            )
        )

    before_chains = fix_chains_output.chains
    before_chain_types = chains_res_type
    before_fallback_masks = chain_fallback_masks
    print(
        "# Use HMM-aligned NA chain types, then apply length-only filtering (min_len = {})".format(
            min_chain_len
        )
    )
    after_chains, after_chain_types, after_fallback_masks = _filter_short_chains_with_fallback_masks(
        before_chains,
        before_chain_types,
        before_fallback_masks,
        min_chain_len,
    )

    keep_chain_indices = {
        int(chain_idx)
        for chain_idx, chain in enumerate(before_chains)
        if len(chain) >= min_chain_len
    }
    for chain_idx, record in enumerate(na_alignment_debug_records):
        record["final_sequence_idx_after_fix_match"] = int(fixed_match.sequence_idxs[chain_idx])
        record["final_match_score_after_fix_match"] = float(fixed_match.match_scores[chain_idx])
        record["has_valid_sequence_after_fix_match"] = bool(0 <= fixed_match.sequence_idxs[chain_idx] < len(seqs))
        record["fallback_used"] = bool(before_fallback_masks[chain_idx].any())
        record["kept_after_length_filter"] = bool(chain_idx in keep_chain_indices)
        record["final_chain_len"] = int(len(before_chains[chain_idx]))

    na_alignment_payload = {
        "summary": {
            **na_alignment_summary,
            "tta_exact_score_sum": float(np.sum(tta_exact_output.best_match_output.match_scores)),
            "tta_selected_score_sum": float(np.sum(tta_best_output.best_match_output.match_scores)),
            "tta_exact_ratio": float(tta_mode_summary["exact_ratio"]),
            "tta_pp_ratio": float(tta_mode_summary["pp_ratio"]),
            "vanilla_exact_score_sum": None if vanilla_exact_output is None else float(np.sum(vanilla_exact_output.best_match_output.match_scores)),
            "vanilla_selected_score_sum": None if vanilla_best_output is None else float(np.sum(vanilla_best_output.best_match_output.match_scores)),
            "vanilla_exact_ratio": None if vanilla_mode_summary is None else float(vanilla_mode_summary["exact_ratio"]),
            "vanilla_pp_ratio": None if vanilla_mode_summary is None else float(vanilla_mode_summary["pp_ratio"]),
            "fallback_chain_count": int(num_fallback_chains),
            "before_chain_count": int(len(before_chains)),
            "after_chain_count": int(len(after_chains)),
            "min_chain_len": int(min_chain_len),
            "fallback_to_predicted_na_types": bool(fallback_to_predicted_na_types),
            "seqs_is_all_same": bool(seqs_is_all_same),
        },
        "chains": na_alignment_debug_records,
    }
    na_alignment_debug_path = os.path.join(hmm_temp_dir, "na_alignment_selection.json")
    _write_json(na_alignment_debug_path, na_alignment_payload)

    return (
        before_chains,
        before_chain_types,
        after_chains,
        after_chain_types,
        before_fallback_masks,
        after_fallback_masks,
    )


def _json_default(obj):
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, set):
        return sorted(list(obj))
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _histogram_summary(values, bins=10):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return {"bin_edges": [], "counts": []}
    bins = int(max(1, min(bins, max(1, values.size))))
    counts, bin_edges = np.histogram(values, bins=bins)
    return {
        "bin_edges": [float(x) for x in bin_edges.tolist()],
        "counts": [int(x) for x in counts.tolist()],
    }


def _serializable_candidate(candidate):
    serializable = dict(candidate)
    if "overlap_metadata" in serializable:
        overlap_metadata = dict(serializable["overlap_metadata"])
        overlap_metadata["conflict_candidate_ids"] = [
            int(x) if isinstance(x, (np.integer, int)) else x
            for x in overlap_metadata.get("conflict_candidate_ids", [])
        ]
        overlap_metadata["dedup_removed_candidate_ids"] = [
            int(x) if isinstance(x, (np.integer, int)) else x
            for x in overlap_metadata.get("dedup_removed_candidate_ids", [])
        ]
        serializable["overlap_metadata"] = overlap_metadata
    return serializable


def _decorate_protein_beam_candidate(
    candidate,
    round_idx,
    protein_local_indices,
    protein_global_indices,
    protein_original_indices,
):
    local_path = np.asarray(candidate["node_indices"], dtype=np.int32)
    return {
        "candidate_uid": f"round{int(round_idx)}_cand{int(candidate['candidate_id'])}",
        "source_round_idx": int(round_idx),
        "source_candidate_id": int(candidate["candidate_id"]),
        "node_indices_local": [int(x) for x in local_path.tolist()],
        "node_indices_filtered": [
            int(x) for x in protein_global_indices[local_path].astype(np.int32).tolist()
        ],
        "node_indices_original": [
            int(x) for x in protein_original_indices[local_path].astype(np.int32).tolist()
        ],
        "path_length": int(candidate["path_length"]),
        "tail_idx_local": int(local_path[-1]),
        "tail_idx_filtered": int(protein_global_indices[local_path[-1]]),
        "tail_idx_original": int(protein_original_indices[local_path[-1]]),
        "score_total": float(candidate["score_total"]),
        "score_breakdown": {
            "edge": float(candidate["score_breakdown"]["edge"]),
            "cn": float(candidate["score_breakdown"]["cn"]),
            "ca": float(candidate["score_breakdown"]["ca"]),
            "confidence": float(candidate["score_breakdown"]["confidence"]),
        },
        "edge_sources": [str(x) for x in candidate.get("edge_sources", [])],
        "edge_scores": candidate.get("edge_scores", []),
        "num_ca_rescue_edges": int(candidate.get("num_ca_rescue_edges", 0)),
        "num_both_edges": int(candidate.get("num_both_edges", 0)),
        "overlap_metadata": {
            "duplicate_of": None,
            "dedup_removed_candidate_ids": [],
            "conflict_candidate_ids": [],
        },
    }


def _beam_candidate_overlap(a_nodes, b_nodes):
    a_set = set(int(x) for x in a_nodes)
    b_set = set(int(x) for x in b_nodes)
    if len(a_set) == 0 or len(b_set) == 0:
        return 0.0
    return len(a_set & b_set) / float(min(len(a_set), len(b_set)))


def _deduplicate_protein_beam_candidates(candidates, overlap_threshold=0.95):
    ranked = sorted(
        candidates,
        key=lambda cand: (cand["score_total"], cand["path_length"]),
        reverse=True,
    )
    kept = []
    duplicate_map = {}

    for candidate in ranked:
        candidate_nodes = candidate["node_indices_filtered"]
        candidate_set = set(candidate_nodes)
        duplicate_of = None

        for kept_candidate in kept:
            kept_nodes = kept_candidate["node_indices_filtered"]
            kept_set = set(kept_nodes)
            if candidate_nodes == kept_nodes:
                duplicate_of = kept_candidate["candidate_uid"]
                break

            overlap = _beam_candidate_overlap(candidate_nodes, kept_nodes)
            if overlap >= overlap_threshold:
                if candidate_nodes[0] == kept_nodes[0] and candidate_nodes[-1] == kept_nodes[-1]:
                    duplicate_of = kept_candidate["candidate_uid"]
                    break

            if candidate_set.issubset(kept_set) and len(candidate_nodes) < len(kept_nodes):
                if candidate["score_total"] <= kept_candidate["score_total"] + 1e-6:
                    duplicate_of = kept_candidate["candidate_uid"]
                    break

        if duplicate_of is not None:
            candidate["overlap_metadata"]["duplicate_of"] = duplicate_of
            duplicate_map.setdefault(duplicate_of, []).append(candidate["candidate_uid"])
            continue

        kept.append(candidate)

    for candidate in kept:
        candidate["overlap_metadata"]["dedup_removed_candidate_ids"] = duplicate_map.get(
            candidate["candidate_uid"], []
        )

    return kept, duplicate_map


def _attach_protein_beam_conflicts(candidates):
    node_to_candidate_ids = {}
    for candidate in candidates:
        candidate_nodes = set(int(x) for x in candidate["node_indices_filtered"])
        for node_idx in candidate_nodes:
            node_to_candidate_ids.setdefault(node_idx, []).append(candidate["candidate_uid"])

    conflict_map = {candidate["candidate_uid"]: set() for candidate in candidates}
    for candidate_ids in node_to_candidate_ids.values():
        if len(candidate_ids) < 2:
            continue
        for candidate_id in candidate_ids:
            conflict_map[candidate_id].update(
                other_id for other_id in candidate_ids if other_id != candidate_id
            )

    for candidate in candidates:
        candidate["overlap_metadata"]["conflict_candidate_ids"] = sorted(
            conflict_map[candidate["candidate_uid"]]
        )

    return candidates, conflict_map


def _summarize_protein_beam_candidates(candidates):
    if len(candidates) == 0:
        return {
            "raw_candidate_count": 0,
            "candidate_count": 0,
            "path_length_histogram": {"bin_edges": [], "counts": []},
            "score_histogram": {"bin_edges": [], "counts": []},
            "num_candidates_with_ca_rescue": 0,
            "fraction_candidates_with_ca_rescue": 0.0,
            "fraction_edges_from_ca_rescue": 0.0,
        }

    lengths = np.asarray([cand["path_length"] for cand in candidates], dtype=np.float32)
    scores = np.asarray([cand["score_total"] for cand in candidates], dtype=np.float32)
    rescue_candidates = [cand for cand in candidates if cand["num_ca_rescue_edges"] > 0]
    rescue_edges = sum(int(cand["num_ca_rescue_edges"]) for cand in candidates)
    total_edges = sum(max(int(cand["path_length"]) - 1, 0) for cand in candidates)

    return {
        "raw_candidate_count": int(len(candidates)),
        "candidate_count": int(len(candidates)),
        "path_length_histogram": _histogram_summary(lengths),
        "score_histogram": _histogram_summary(scores),
        "num_candidates_with_ca_rescue": int(len(rescue_candidates)),
        "fraction_candidates_with_ca_rescue": float(len(rescue_candidates) / len(candidates)),
        "fraction_edges_from_ca_rescue": float(rescue_edges / max(total_edges, 1)),
    }


def _annotate_protein_beam_candidates_with_hmm(
    candidates,
    reordered_final_results,
    prot_seqs,
    hmm_temp_dir,
):
    if len(candidates) == 0 or len(prot_seqs) == 0:
        return candidates, None

    chain_aa_logits = []
    chain_confidences = []
    chain_prot_mask = []
    for candidate in candidates:
        node_indices = np.asarray(candidate["node_indices_filtered"], dtype=np.int32)
        chain_aa_logits.append(
            np.asarray(reordered_final_results["pred_aatype"][node_indices], dtype=np.float32)
        )
        if "confidence" in reordered_final_results:
            chain_confidences.append(
                np.asarray(reordered_final_results["confidence"][node_indices][:, 0], dtype=np.float32)
            )
        else:
            chain_confidences.append(None)
        chain_prot_mask.append(np.ones((len(node_indices),), dtype=bool))

    hmm_output = best_match_to_sequences(
        prot_sequences=prot_seqs,
        rna_sequences=[],
        dna_sequences=[],
        chain_prot_mask=chain_prot_mask,
        chain_aa_logits=chain_aa_logits,
        chain_confidences=chain_confidences,
        base_dir=hmm_temp_dir,
        do_pp=False,
    )

    for candidate_idx, candidate in enumerate(candidates):
        seq_idx = int(hmm_output.sequence_idxs[candidate_idx])
        residue_idxs = np.asarray(hmm_output.residue_idxs[candidate_idx], dtype=np.int32)
        exists_mask = np.asarray(
            hmm_output.exists_in_sequence_mask[candidate_idx], dtype=np.int32
        )
        coverage = float(np.mean(exists_mask.astype(np.float32))) if len(exists_mask) > 0 else 0.0
        candidate["hmm_annotation"] = {
            "match_score": float(hmm_output.match_scores[candidate_idx]),
            "matched_sequence_id": int(seq_idx) if 0 <= seq_idx < len(prot_seqs) else -1,
            "matched_residue_span": [
                int(residue_idxs.min()) if len(residue_idxs) > 0 else -1,
                int(residue_idxs.max()) if len(residue_idxs) > 0 else -1,
            ],
            "aligned_coverage": coverage,
            "exists_in_sequence_mask_summary": {
                "count": int(np.sum(exists_mask > 0)),
                "length": int(len(exists_mask)),
            },
            "hmm_output_match_sequence": hmm_output.hmm_output_match_sequences[candidate_idx],
        }

    return candidates, hmm_output


def _rank_protein_beam_candidates(candidates):
    if len(candidates) == 0:
        return candidates

    def _sort_key(candidate):
        hmm_annotation = candidate.get("hmm_annotation")
        if hmm_annotation is None:
            return (candidate["score_total"], candidate["path_length"])
        return (
            float(hmm_annotation.get("match_score", 0.0)),
            float(hmm_annotation.get("aligned_coverage", 0.0)),
            float(candidate["score_total"]),
            int(candidate["path_length"]),
        )

    return sorted(candidates, key=_sort_key, reverse=True)


def _select_anchor_candidates_for_iterative_refine(candidates, max_anchors=32):
    if len(candidates) == 0:
        return []

    ranked = sorted(
        candidates,
        key=lambda candidate: (
            candidate.get("hmm_annotation", {}).get("match_score", candidate["score_total"]),
            candidate.get("hmm_annotation", {}).get("aligned_coverage", 0.0),
            candidate["score_total"],
            candidate["path_length"],
        ),
        reverse=True,
    )

    selected = []
    used_nodes = set()
    for candidate in ranked:
        candidate_nodes = set(int(x) for x in candidate["node_indices_local"])
        if len(candidate_nodes) == 0:
            continue
        if candidate_nodes & used_nodes:
            continue
        hmm_annotation = candidate.get("hmm_annotation")
        if hmm_annotation:
            if hmm_annotation.get("match_score", 0.0) < 0.55:
                continue
            if hmm_annotation.get("aligned_coverage", 0.0) < 0.4:
                continue
        elif candidate["score_total"] < 0.0:
            continue
        selected.append(candidate)
        used_nodes.update(candidate_nodes)
        if len(selected) >= max_anchors:
            break

    return selected


def _write_json(filename, payload):
    with open(filename, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=_json_default)
    print(f"# Output JSON to {filename}")


def final_results_align_to_sequence_beam_protein(
    final_results,
    protein_seq_dir=None,
    dna_seq_dir=None,
    rna_seq_dir=None,
    output_dir="output",
    flag_prune_and_connect_chains=True,
    protein_radius_threshold=1.5,
    na_radius_threshold=1.5,
    min_na_chain_len=1,
    fallback_to_predicted_na_types=True,
    na_aa_logits_data=None,
    beam_width_protein=16,
    trace_max_candidates=5000,
    cn_radius_protein=2.1,
    ca_rescue_radius_protein=4.8,
    beam_protein_trace_dump_candidates=True,
    beam_protein_trace_hmm_rerank=False,
    beam_protein_trace_iterative_refine=False,
    beam_protein_trace_max_refine_rounds=3,
):
    del dna_seq_dir
    del rna_seq_dir
    del flag_prune_and_connect_chains
    del na_radius_threshold
    del min_na_chain_len
    del fallback_to_predicted_na_types
    del na_aa_logits_data

    parent_out_dir = abspath(output_dir)
    out_dir = pjoin(parent_out_dir, "postprocess_beam_protein")
    hmm_temp_dir = pjoin(out_dir, "hmm")
    os.makedirs(hmm_temp_dir, exist_ok=True)
    print(f"# Beam protein postprocess directory = {out_dir}")
    print(f"# Beam protein HMM temp dir = {hmm_temp_dir}")

    prot_seqs = (
        _read_sequences(protein_seq_dir, "protein")
        if protein_seq_dir is not None
        else []
    )

    working_results = dict(final_results)
    prot_mask_full = np.asarray(final_results["prot_mask"], dtype=bool)
    if not np.any(prot_mask_full):
        summary_path = pjoin(out_dir, "beam_summary.json")
        candidates_path = pjoin(out_dir, "protein_candidates.json")
        empty_summary = {
            "raw_candidate_count": 0,
            "processed_candidate_count": 0,
            "truncated_candidate_count": 0,
            "beam_settings": {
                "beam_width_protein": int(beam_width_protein),
                "trace_max_candidates": int(trace_max_candidates),
                "cn_radius_protein": float(cn_radius_protein),
                "ca_rescue_radius_protein": float(ca_rescue_radius_protein),
                "beam_protein_trace_hmm_rerank": bool(beam_protein_trace_hmm_rerank),
                "beam_protein_trace_iterative_refine": bool(
                    beam_protein_trace_iterative_refine
                ),
                "beam_protein_trace_max_refine_rounds": int(
                    beam_protein_trace_max_refine_rounds
                ),
            },
            "round_summaries": [],
            "message": "No protein residues found in final_results.",
        }
        if beam_protein_trace_dump_candidates:
            _write_json(
                candidates_path,
                {
                    "raw_candidates": [],
                    "processed_candidates": [],
                    "dedup_removed_mapping": {},
                },
            )
        _write_json(summary_path, empty_summary)
        return {
            "output_dir": out_dir,
            "protein_candidates_json": candidates_path
            if beam_protein_trace_dump_candidates
            else None,
            "beam_summary_json": summary_path,
            "protein_candidates_topk_cif": None,
            "raw_candidate_count": 0,
            "processed_candidate_count": 0,
        }

    na_mask_full = ~prot_mask_full
    residue_type_entropy, residue_type_confidence = _compute_residue_type_scores(
        final_results["pred_aatype"],
        prot_mask_full,
    )
    working_results["residue_type_entropy"] = residue_type_entropy
    working_results["residue_type_confidence"] = residue_type_confidence
    working_results["entropy_score_bfactor"] = _expand_residue_scalar_to_atom_bfactor(
        residue_type_confidence
    )

    confidence = _pred_rmsd_to_confidence(final_results["pred_rmsd"], prot_mask_full)
    working_results["confidence"] = np.repeat(confidence[..., None], 23, axis=-1)
    working_results["plddt"] = working_results["confidence"]

    dummy_aatype = np.zeros((len(final_results["pred_aatype"]),), dtype=np.int32)
    if np.any(prot_mask_full):
        dummy_aatype[prot_mask_full] = np.argmax(
            final_results["pred_aatype"][prot_mask_full][..., :num_prot],
            axis=-1,
        )
    if np.any(na_mask_full):
        dummy_aatype[na_mask_full] = (
            np.argmax(final_results["pred_aatype"][na_mask_full][..., num_prot:], axis=-1)
            + num_prot
            + 4
        )

    pred_torsions = select_torsion_angles(
        final_results["pred_torsions"],
        aatype=dummy_aatype,
        normalize=True,
    )
    pred_atom_pos, pred_atom_mask = affines_and_torsion_angles_to_atomc_pos(
        final_results["pred_affines"],
        pred_torsions,
        aatype=dummy_aatype,
    )
    pred_atom_pos = pred_atom_pos.cpu().numpy()
    pred_atom_mask = pred_atom_mask.cpu().numpy()

    existence_mask = np.ones_like(confidence, dtype=bool)
    if np.any(prot_mask_full):
        existence_mask[prot_mask_full] = remove_overlapping_ca(
            final_results["pred_affines"][prot_mask_full][..., :, -1],
            bfactors=confidence[prot_mask_full],
            existence_mask=None,
            radius_threshold=protein_radius_threshold,
        )

    pred_atom_pos = pred_atom_pos[existence_mask]
    pred_atom_mask = pred_atom_mask[existence_mask]
    prot_mask = prot_mask_full[existence_mask]
    confidence = confidence[existence_mask]
    survived_original_indices = np.nonzero(existence_mask)[0].astype(np.int32)

    reordered_final_results = {}
    for key, value in working_results.items():
        if key in {"pred_edge_existence_dict", "pred_pairing_dict"}:
            continue
        reordered_final_results[key] = value[existence_mask]
    reordered_final_results["_dummy_atom_pos"] = pred_atom_pos

    if not np.any(prot_mask):
        summary_path = pjoin(out_dir, "beam_summary.json")
        candidates_path = pjoin(out_dir, "protein_candidates.json")
        empty_summary = {
            "raw_candidate_count": 0,
            "processed_candidate_count": 0,
            "truncated_candidate_count": 0,
            "beam_settings": {
                "beam_width_protein": int(beam_width_protein),
                "trace_max_candidates": int(trace_max_candidates),
                "cn_radius_protein": float(cn_radius_protein),
                "ca_rescue_radius_protein": float(ca_rescue_radius_protein),
                "beam_protein_trace_hmm_rerank": bool(beam_protein_trace_hmm_rerank),
                "beam_protein_trace_iterative_refine": bool(
                    beam_protein_trace_iterative_refine
                ),
                "beam_protein_trace_max_refine_rounds": int(
                    beam_protein_trace_max_refine_rounds
                ),
            },
            "round_summaries": [],
            "message": "No protein residues survived the existence filter.",
        }
        if beam_protein_trace_dump_candidates:
            _write_json(
                candidates_path,
                {
                    "raw_candidates": [],
                    "processed_candidates": [],
                    "dedup_removed_mapping": {},
                },
            )
        _write_json(summary_path, empty_summary)
        return {
            "output_dir": out_dir,
            "protein_candidates_json": candidates_path
            if beam_protein_trace_dump_candidates
            else None,
            "beam_summary_json": summary_path,
            "protein_candidates_topk_cif": None,
            "raw_candidate_count": 0,
            "processed_candidate_count": 0,
        }

    protein_global_indices = np.arange(len(pred_atom_pos), dtype=np.int32)[prot_mask]
    protein_original_indices = survived_original_indices[prot_mask]
    protein_atom_pos = pred_atom_pos[prot_mask]
    protein_confidence = confidence[prot_mask]

    all_raw_candidates = []
    round_summaries = []
    frozen_mask = np.zeros((len(protein_atom_pos),), dtype=bool)
    num_rounds = (
        max(int(beam_protein_trace_max_refine_rounds), 1)
        if beam_protein_trace_iterative_refine
        else 1
    )
    last_beam_settings = None

    for round_idx in range(num_rounds):
        remaining_budget = max(int(trace_max_candidates) - len(all_raw_candidates), 0)
        if remaining_budget <= 0:
            round_summaries.append(
                {
                    "round_idx": int(round_idx),
                    "num_candidates": 0,
                    "stopped_reason": "candidate_cap_reached",
                    "num_frozen_nodes": int(np.sum(frozen_mask)),
                }
            )
            break

        beam_output = beam_trace_with_edge_dual_protein(
            protein_atom_pos,
            b_factors=protein_confidence,
            bond_distance_threshold=cn_radius_protein,
            last_idx=2,
            next_idx=0,
            edge_existence_dict=final_results["pred_edge_existence_dict"],
            idx_exist_to_original=protein_original_indices,
            beam_width=beam_width_protein,
            max_candidates=remaining_budget,
            max_path_length=len(protein_atom_pos),
            ca_rescue_radius=ca_rescue_radius_protein,
            frozen_mask=frozen_mask,
        )
        last_beam_settings = dict(beam_output["settings"])
        round_candidates = [
            _decorate_protein_beam_candidate(
                candidate,
                round_idx=round_idx,
                protein_local_indices=np.arange(len(protein_atom_pos), dtype=np.int32),
                protein_global_indices=protein_global_indices,
                protein_original_indices=protein_original_indices,
            )
            for candidate in beam_output["candidates"]
        ]

        if len(round_candidates) == 0:
            round_summaries.append(
                {
                    "round_idx": int(round_idx),
                    "num_candidates": 0,
                    "stopped_reason": "no_candidates",
                    "num_frozen_nodes": int(np.sum(frozen_mask)),
                }
            )
            break

        if beam_protein_trace_iterative_refine and len(prot_seqs) > 0:
            round_candidates, _ = _annotate_protein_beam_candidates_with_hmm(
                round_candidates,
                reordered_final_results,
                prot_seqs,
                hmm_temp_dir,
            )

        all_raw_candidates.extend(round_candidates)
        round_summary = {
            "round_idx": int(round_idx),
            "num_candidates": int(len(round_candidates)),
            "num_frozen_nodes": int(np.sum(frozen_mask)),
            "num_candidates_with_ca_rescue": int(
                sum(cand["num_ca_rescue_edges"] > 0 for cand in round_candidates)
            ),
        }

        if not beam_protein_trace_iterative_refine:
            round_summary["stopped_reason"] = "single_round_mode"
            round_summaries.append(round_summary)
            break

        anchors = _select_anchor_candidates_for_iterative_refine(round_candidates)
        round_summary["num_selected_anchors"] = int(len(anchors))
        round_summary["selected_anchor_candidate_uids"] = [
            cand["candidate_uid"] for cand in anchors
        ]
        if len(anchors) == 0:
            round_summary["stopped_reason"] = "no_anchor_candidates"
            round_summaries.append(round_summary)
            break

        new_frozen_mask = frozen_mask.copy()
        for anchor_candidate in anchors:
            anchor_local_indices = np.asarray(
                anchor_candidate["node_indices_local"], dtype=np.int32
            )
            new_frozen_mask[anchor_local_indices] = True

        if np.array_equal(new_frozen_mask, frozen_mask):
            round_summary["stopped_reason"] = "no_new_frozen_nodes"
            round_summaries.append(round_summary)
            break

        frozen_mask = new_frozen_mask
        round_summary["stopped_reason"] = "continue_refine"
        round_summaries.append(round_summary)

    for candidate_idx, candidate in enumerate(all_raw_candidates):
        candidate["candidate_id"] = int(candidate_idx)

    raw_candidates_snapshot = copy.deepcopy(all_raw_candidates)
    processed_candidates = copy.deepcopy(all_raw_candidates)

    if beam_protein_trace_hmm_rerank and len(processed_candidates) > 0 and len(prot_seqs) > 0:
        processed_candidates, _ = _annotate_protein_beam_candidates_with_hmm(
            processed_candidates,
            reordered_final_results,
            prot_seqs,
            hmm_temp_dir,
        )

    processed_candidates = _rank_protein_beam_candidates(processed_candidates)
    processed_candidates, duplicate_map = _deduplicate_protein_beam_candidates(
        processed_candidates
    )
    processed_candidates, conflict_map = _attach_protein_beam_conflicts(
        processed_candidates
    )
    processed_candidates = _rank_protein_beam_candidates(processed_candidates)

    topk_candidates_path = None
    if len(raw_candidates_snapshot) > 0:
        topk_candidates = sorted(
            raw_candidates_snapshot,
            key=lambda cand: (cand["score_total"], cand["path_length"]),
            reverse=True,
        )[: min(64, len(raw_candidates_snapshot))]
        topk_chains = [
            np.asarray(cand["node_indices_filtered"], dtype=np.int32)
            for cand in topk_candidates
            if cand["path_length"] > 0
        ]
        topk_chain_types = [
            np.argmax(
                np.asarray(reordered_final_results["pred_aatype"][chain], dtype=np.float32)[
                    ..., :num_prot
                ],
                axis=-1,
            ).astype(np.int32)
            for chain in topk_chains
        ]
        if len(topk_chains) > 0:
            topk_candidates_path = pjoin(out_dir, "protein_candidates_topk.cif")
            _write_chain_file(
                topk_candidates_path,
                reordered_final_results,
                topk_chains,
                topk_chain_types,
            )

    candidates_json_path = pjoin(out_dir, "protein_candidates.json")
    if beam_protein_trace_dump_candidates:
        _write_json(
            candidates_json_path,
            {
                "raw_candidates": [
                    _serializable_candidate(candidate)
                    for candidate in raw_candidates_snapshot
                ],
                "processed_candidates": [
                    _serializable_candidate(candidate)
                    for candidate in processed_candidates
                ],
                "dedup_removed_mapping": duplicate_map,
                "conflict_map": {
                    candidate_id: sorted(list(conflict_ids))
                    for candidate_id, conflict_ids in conflict_map.items()
                },
            },
        )

    raw_summary = _summarize_protein_beam_candidates(raw_candidates_snapshot)
    processed_summary = _summarize_protein_beam_candidates(processed_candidates)
    summary_payload = {
        "raw_candidate_count": int(len(raw_candidates_snapshot)),
        "processed_candidate_count": int(len(processed_candidates)),
        "truncated_candidate_count": int(min(len(raw_candidates_snapshot), int(trace_max_candidates))),
        "beam_settings": {
            "beam_width_protein": int(beam_width_protein),
            "trace_max_candidates": int(trace_max_candidates),
            "cn_radius_protein": float(cn_radius_protein),
            "ca_rescue_radius_protein": float(ca_rescue_radius_protein),
            "beam_protein_trace_dump_candidates": bool(
                beam_protein_trace_dump_candidates
            ),
            "beam_protein_trace_hmm_rerank": bool(beam_protein_trace_hmm_rerank),
            "beam_protein_trace_iterative_refine": bool(
                beam_protein_trace_iterative_refine
            ),
            "beam_protein_trace_max_refine_rounds": int(
                beam_protein_trace_max_refine_rounds
            ),
            "backend_settings": last_beam_settings,
        },
        "round_summaries": round_summaries,
        "raw_path_length_histogram": raw_summary["path_length_histogram"],
        "raw_score_histogram": raw_summary["score_histogram"],
        "processed_path_length_histogram": processed_summary["path_length_histogram"],
        "processed_score_histogram": processed_summary["score_histogram"],
        "raw_fraction_candidates_with_ca_rescue": raw_summary[
            "fraction_candidates_with_ca_rescue"
        ],
        "raw_fraction_edges_from_ca_rescue": raw_summary[
            "fraction_edges_from_ca_rescue"
        ],
        "processed_fraction_candidates_with_ca_rescue": processed_summary[
            "fraction_candidates_with_ca_rescue"
        ],
        "processed_fraction_edges_from_ca_rescue": processed_summary[
            "fraction_edges_from_ca_rescue"
        ],
        "num_protein_nodes_after_existence_filter": int(len(protein_atom_pos)),
        "num_duplicate_groups": int(len(duplicate_map)),
        "num_conflicting_processed_candidates": int(
            sum(
                len(candidate["overlap_metadata"]["conflict_candidate_ids"]) > 0
                for candidate in processed_candidates
            )
        ),
    }
    summary_path = pjoin(out_dir, "beam_summary.json")
    _write_json(summary_path, summary_payload)

    return {
        "output_dir": out_dir,
        "protein_candidates_json": candidates_json_path
        if beam_protein_trace_dump_candidates
        else None,
        "beam_summary_json": summary_path,
        "protein_candidates_topk_cif": topk_candidates_path,
        "raw_candidate_count": int(len(raw_candidates_snapshot)),
        "processed_candidate_count": int(len(processed_candidates)),
    }


def final_results_align_to_sequence(
    final_results,
    protein_seq_dir=None,
    dna_seq_dir=None,
    rna_seq_dir=None,
    output_dir="output",
    flag_prune_and_connect_chains=True,
    protein_radius_threshold=1.5,
    na_radius_threshold=1.5,
    min_na_chain_len=1,
    fallback_to_predicted_na_types=True,
    na_aa_logits_data=None,
    force_protein_mode=False,
    force_na_mode=False,
):
    out_dir = abspath(output_dir)
    hmm_temp_dir = pjoin(out_dir, "hmm")
    os.makedirs(hmm_temp_dir, exist_ok=True)
    print(f"# Setting output directory to {out_dir}")
    print(f"# Setting HMM temp dir to {hmm_temp_dir}")

    prot_seqs = _read_sequences(protein_seq_dir, "protein") if protein_seq_dir is not None else []
    dna_seqs = _read_sequences(dna_seq_dir, "DNA") if dna_seq_dir is not None else []
    rna_seqs = _read_sequences(rna_seq_dir, "RNA") if rna_seq_dir is not None else []

    has_protein_input = len(prot_seqs) > 0
    has_na_input = (len(dna_seqs) + len(rna_seqs)) > 0
    if force_protein_mode or force_na_mode:
        has_protein_input = bool(force_protein_mode)
        has_na_input = bool(force_na_mode)
        print(
            "# No-seq force mode: protein={} na={}".format(
                has_protein_input,
                has_na_input,
            )
        )

    prot_mask = np.asarray(final_results["prot_mask"], dtype=bool)
    if has_protein_input and (not has_na_input):
        print("# Protein-only mode: disable nucleic-acid postprocess outputs")
        prot_mask = np.ones_like(prot_mask, dtype=bool)
        na_mask = np.zeros_like(prot_mask, dtype=bool)
    elif has_na_input and (not has_protein_input):
        print("# Nucleic-acid-only mode: disable protein postprocess outputs")
        prot_mask = np.zeros_like(prot_mask, dtype=bool)
        na_mask = np.ones_like(prot_mask, dtype=bool)
    elif has_na_input or has_protein_input:
        na_mask = ~prot_mask
    else:
        na_mask = ~prot_mask

    residue_type_entropy, residue_type_confidence = _compute_residue_type_scores(
        final_results["pred_aatype"],
        prot_mask,
    )
    final_results["residue_type_entropy"] = residue_type_entropy
    final_results["residue_type_confidence"] = residue_type_confidence
    final_results["entropy_score_bfactor"] = _expand_residue_scalar_to_atom_bfactor(
        residue_type_confidence
    )

    confidence = _pred_rmsd_to_confidence(final_results["pred_rmsd"], prot_mask)

    existence_mask = np.ones_like(confidence).astype(bool)
    if np.any(prot_mask):
        existence_mask[prot_mask] = remove_overlapping_ca(
            final_results["pred_affines"][prot_mask][..., :, -1],
            bfactors=confidence[prot_mask],
            existence_mask=None,
            radius_threshold=protein_radius_threshold,
        )
    if np.any(na_mask):
        existence_mask[na_mask] = remove_overlapping_ca(
            final_results["pred_affines"][na_mask][..., :, -1],
            bfactors=confidence[na_mask],
            existence_mask=None,
            radius_threshold=na_radius_threshold,
        )

    final_results["confidence"] = np.repeat(confidence[..., None], 23, axis=-1)
    final_results["plddt"] = final_results["confidence"]

    dummy_aatype = np.zeros((len(final_results["pred_aatype"]),), dtype=np.int32)
    if np.any(prot_mask):
        dummy_aatype[prot_mask] = np.argmax(
            final_results["pred_aatype"][prot_mask][..., :num_prot],
            axis=-1,
        )
    if np.any(na_mask):
        dummy_aatype[na_mask] = (
            np.argmax(final_results["pred_aatype"][na_mask][..., num_prot:], axis=-1)
            + num_prot
            + 4
        )

    pred_torsions = select_torsion_angles(
        final_results["pred_torsions"],
        aatype=dummy_aatype,
        normalize=True,
    )
    pred_atom_pos, pred_atom_mask = affines_and_torsion_angles_to_atomc_pos(
        final_results["pred_affines"],
        pred_torsions,
        aatype=dummy_aatype,
    )
    pred_atom_pos = pred_atom_pos.cpu().numpy()
    pred_atom_mask = pred_atom_mask.cpu().numpy()

    raw_before_existence_filter_path = None
    raw_protein_before_existence_filter_path = None
    raw_na_before_existence_filter_path = None
    residue_type_scores_before_existence_filter_path = os.path.join(
        out_dir, "residue_type_scores_before_existence_filter.npz"
    )
    _write_residue_type_score_npz(
        residue_type_scores_before_existence_filter_path,
        final_results,
    )
    if np.any(prot_mask):
        raw_protein_before_existence_filter_path = os.path.join(
            out_dir, "protein_before_existence_filter.cif"
        )
        _write_residue_subset_file(
            raw_protein_before_existence_filter_path,
            pred_atom_pos[prot_mask],
            pred_atom_mask[prot_mask],
            dummy_aatype[prot_mask],
            residue_index=np.arange(int(np.sum(prot_mask)), dtype=np.int32),
            bfactor=final_results["plddt"][prot_mask],
        )
        print("# Raw protein residues before existence filter = {}".format(int(np.sum(prot_mask))))
    if np.any(na_mask):
        raw_na_before_existence_filter_path = os.path.join(
            out_dir, "na_before_existence_filter.cif"
        )
        _write_residue_subset_file(
            raw_na_before_existence_filter_path,
            pred_atom_pos[na_mask],
            pred_atom_mask[na_mask],
            dummy_aatype[na_mask],
            residue_index=np.arange(int(np.sum(na_mask)), dtype=np.int32),
            bfactor=final_results["plddt"][na_mask],
        )
        print("# Raw NA residues before existence filter = {}".format(int(np.sum(na_mask))))

    raw_before_existence_filter_path = os.path.join(out_dir, "before_existence_filter.cif")
    _write_residue_subset_file(
        raw_before_existence_filter_path,
        pred_atom_pos,
        pred_atom_mask,
        dummy_aatype,
        residue_index=np.arange(len(pred_atom_pos), dtype=np.int32),
        bfactor=final_results["plddt"],
    )
    print("# Raw all residues before existence filter = {}".format(len(pred_atom_pos)))

    pred_atom_pos = pred_atom_pos[existence_mask]
    pred_atom_mask = pred_atom_mask[existence_mask]

    prot_mask = prot_mask[existence_mask]
    na_mask = na_mask[existence_mask]
    confidence = confidence[existence_mask]

    reordered_final_results = {}
    for key, value in final_results.items():

        # Must not reorder sparse dict outputs here.
        # Afterwards, we have idx_exist_to_original to indicate the correspondence between the original and the existed.
        if key in {"pred_edge_existence_dict", "pred_pairing_dict"}:
            continue
        reordered_final_results[key] = value[existence_mask]
        
    reordered_final_results["_dummy_atom_pos"] = pred_atom_pos

    residue_type_scores_after_existence_filter_path = os.path.join(
        out_dir, "residue_type_scores_after_existence_filter.npz"
    )
    _write_residue_type_score_npz(
        residue_type_scores_after_existence_filter_path,
        reordered_final_results,
    )

    prot_chains = []
    if has_protein_input and np.any(prot_mask):
        idx_exist_to_original = np.arange(len(pred_atom_pos), dtype=np.int32)[prot_mask]
        traced = flood_fill_with_edge_dual(
            pred_atom_pos[prot_mask],
            confidence[prot_mask],
            bond_distance_threshold=2.1,
            last_idx=2,
            next_idx=0,
            edge_existence_dict=final_results["pred_edge_existence_dict"],
            idx_exist_to_original=idx_exist_to_original,
        )
        idxs = np.arange(len(pred_atom_pos))[prot_mask]
        prot_chains = [idxs[c] for c in traced]
        print(f"# Trace protein into {len(prot_chains)} dummy chains")

    na_chains = []
    if has_na_input and np.any(na_mask):
        idx_exist_to_original = np.arange(len(pred_atom_pos), dtype=np.int32)[na_mask]
        traced = flood_fill_with_edge_dual(
            pred_atom_pos[na_mask],
            confidence[na_mask],
            bond_distance_threshold=3.6,
            last_idx=6,
            next_idx=8,
            edge_existence_dict=final_results["pred_edge_existence_dict"],
            idx_exist_to_original=idx_exist_to_original,
        )
        idxs = np.arange(len(pred_atom_pos))[na_mask]
        na_chains = [idxs[c] for c in traced]
        print(f"# Trace NA into {len(na_chains)} dummy chains")

    (
        prot_before_chains,
        prot_before_chain_types,
        prot_after_chains,
        prot_after_chain_types,
    ) = _build_protein_chain_outputs(
        reordered_final_results,
        prot_chains,
        prot_seqs,
        hmm_temp_dir,
        pred_atom_pos,
    )
    (
        na_before_chains,
        na_before_chain_types,
        na_after_chains,
        na_after_chain_types,
        na_before_fallback_masks,
        na_after_fallback_masks,
    ) = _build_na_chain_outputs(
        reordered_final_results,
        na_chains,
        rna_seqs,
        dna_seqs,
        hmm_temp_dir,
        na_aa_logits_data=na_aa_logits_data,
        min_chain_len=min_na_chain_len,
        fallback_to_predicted_na_types=fallback_to_predicted_na_types,
    )

    before_prune_path = None
    after_prune_path = None
    protein_before_prune_path = None
    protein_after_prune_path = None
    na_before_prune_path = None
    na_after_prune_path = None
    if len(prot_before_chains) > 0:
        _print_chain_stats("Protein before prune", prot_before_chains)
        protein_before_prune_path = os.path.join(out_dir, "protein_before_prune.cif")
        _write_chain_file(
            protein_before_prune_path,
            reordered_final_results,
            prot_before_chains,
            prot_before_chain_types,
        )

    if len(prot_after_chains) > 0:
        _print_chain_stats("Protein after prune", prot_after_chains)
        protein_after_prune_path = os.path.join(out_dir, "protein_after_prune.cif")
        _write_chain_file(
            protein_after_prune_path,
            reordered_final_results,
            prot_after_chains,
            prot_after_chain_types,
        )

    if len(na_before_chains) > 0:
        _print_chain_stats("NA before prune", na_before_chains)
        na_before_prune_path = os.path.join(out_dir, "na_before_prune.cif")
        _write_chain_file(
            na_before_prune_path,
            reordered_final_results,
            na_before_chains,
            na_before_chain_types,
        )

    if len(na_after_chains) > 0:
        _print_chain_stats("NA after prune", na_after_chains)
        na_after_prune_path = os.path.join(out_dir, "na_after_prune.cif")
        _write_chain_file(
            na_after_prune_path,
            reordered_final_results,
            na_after_chains,
            na_after_chain_types,
        )

    profile_root_dir = _resolve_chain_hmm_profile_dir()
    if profile_root_dir is not None:
        profile_root_dir = os.path.join(profile_root_dir, os.path.basename(out_dir))
        _dump_chain_hmm_profiles(
            profile_root_dir,
            reordered_final_results,
            prot_after_chains,
            prot_after_chain_types,
            na_after_chains,
            na_after_chain_types,
            profile_set_name="after_prune",
        )

    all_before_chains = []
    all_before_chain_types = []
    if len(prot_before_chains) > 0:
        all_before_chains.extend(prot_before_chains)
        all_before_chain_types.extend(prot_before_chain_types)
    if len(na_before_chains) > 0:
        all_before_chains.extend(na_before_chains)
        all_before_chain_types.extend(na_before_chain_types)

    if len(all_before_chains) > 0:
        before_prune_path = os.path.join(out_dir, "before_prune.cif")
        _print_chain_stats("All before prune", all_before_chains)
        _write_chain_file(
            before_prune_path,
            reordered_final_results,
            all_before_chains,
            all_before_chain_types,
        )
    before_prune_state = _build_chain_order_recycle_state(
        reordered_final_results,
        all_before_chains,
    )

    all_after_chains = []
    all_after_chain_types = []
    if len(prot_after_chains) > 0:
        all_after_chains.extend(prot_after_chains)
        all_after_chain_types.extend(prot_after_chain_types)
    if len(na_after_chains) > 0:
        all_after_chains.extend(na_after_chains)
        all_after_chain_types.extend(na_after_chain_types)

    if len(all_after_chains) > 0:
        after_prune_path = os.path.join(out_dir, "after_prune.cif")
        _print_chain_stats("All after prune", all_after_chains)
        _write_chain_file(
            after_prune_path,
            reordered_final_results,
            all_after_chains,
            all_after_chain_types,
        )

    all_chains = []
    all_chain_types = []
    if len(prot_after_chains) > 0:
        all_chains.extend(prot_after_chains)
        all_chain_types.extend(prot_after_chain_types)
    if len(na_after_chains) > 0:
        all_chains.extend(na_after_chains)
        all_chain_types.extend(na_after_chain_types)

    fpdbout = os.path.join(out_dir, "output.cif")
    _write_chain_file(
        fpdbout,
        reordered_final_results,
        all_chains,
        all_chain_types,
    )
    entropy_final_results = dict(reordered_final_results)
    entropy_score_bfactor = np.array(
        reordered_final_results["entropy_score_bfactor"],
        copy=True,
    )
    fallback_entropy_mask = np.zeros((len(entropy_score_bfactor),), dtype=bool)
    for chain, fallback_mask in zip(na_after_chains, na_after_fallback_masks):
        if len(chain) == 0 or len(fallback_mask) == 0:
            continue
        fallback_entropy_mask[np.asarray(chain, dtype=np.int32)[np.asarray(fallback_mask, dtype=bool)]] = True
    if np.any(fallback_entropy_mask):
        entropy_score_bfactor[fallback_entropy_mask] = FALLBACK_CONFIDENCE_SENTINEL
    entropy_final_results["entropy_score_bfactor"] = entropy_score_bfactor
    entropy_output_path = os.path.join(out_dir, "output_entropy_score.cif")
    _write_chain_file(
        entropy_output_path,
        entropy_final_results,
        all_chains,
        all_chain_types,
        bfactor_key="entropy_score_bfactor",
    )
    output_state = _build_chain_order_recycle_state(
        reordered_final_results,
        all_chains,
    )
    next_round_state = before_prune_state if before_prune_path is not None else output_state
    protein_after_num_res = int(sum(len(chain) for chain in prot_after_chains))
    na_after_num_res = int(sum(len(chain) for chain in na_after_chains))
    return {
        "output_path": fpdbout,
        "output_entropy_score_path": entropy_output_path,
        "before_prune_path": before_prune_path,
        "after_prune_path": after_prune_path,
        "raw_before_existence_filter_path": raw_before_existence_filter_path,
        "raw_protein_before_existence_filter_path": raw_protein_before_existence_filter_path,
        "raw_na_before_existence_filter_path": raw_na_before_existence_filter_path,
        "residue_type_scores_before_existence_filter_path": residue_type_scores_before_existence_filter_path,
        "residue_type_scores_after_existence_filter_path": residue_type_scores_after_existence_filter_path,
        "protein_before_prune_path": protein_before_prune_path,
        "protein_after_prune_path": protein_after_prune_path,
        "na_before_prune_path": na_before_prune_path,
        "na_after_prune_path": na_after_prune_path,
        "protein_after_num_res": protein_after_num_res,
        "na_after_num_res": na_after_num_res,
        "before_prune_state": before_prune_state,
        "output_state": output_state,
        "next_round_state": next_round_state,
    }
