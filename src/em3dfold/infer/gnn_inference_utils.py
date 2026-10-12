import torch
import torch.nn.functional as F
import argparse

import numpy as np
from contextlib import nullcontext
from scipy.spatial import cKDTree

from em3dfold.utils.affine_utils import (
    get_affine_translation,
    get_affine_rot,
    get_affine,
    init_random_affine_from_translation,
)

from em3dfold.polymer_utils.polymer import Polymer, get_polymer_empty_except
from em3dfold.polymer_utils.residue_constants import (
    num_net_torsions,
    canonical_num_residues,
)

def load_translation_from_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    trans = []
    prot_mask = []

    if filename.endswith(".pdb"):
        for line in lines:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            assert len(line) >= 54, "Wrong PDB format"
            atom_name = line[12:16].strip()
            if len(atom_name) == 0:
                continue

            trans.append(np.asarray([float(x) for x in [line[i: i + 8] for i in [30, 38, 46]]], dtype=np.float32))

            if atom_name == "CA":
                prot_mask.append(1)
            elif atom_name == "C4'":
                prot_mask.append(0)
            else:
                raise Exception(f"Unknown atom type = {atom_name}, only support CA or C4'")

    elif filename.endswith(".cif"):
        for line in lines:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            fields = line.strip().split()
            assert len(fields) >= 13, "Wrong CIF format"
            atom_name = fields[3]
            if len(atom_name) == 0:
                continue

            if atom_name not in ["CA", "C4'"]:
                continue

            trans.append(np.array([fields[10], fields[11], fields[12]], dtype=np.float32))

            if atom_name == "CA":
                prot_mask.append(1)
            elif atom_name == "C4'":
                prot_mask.append(0)
            else:
                raise Exception(f"Unknown atom type = {atom_name}, only support CA or C4'")

    else:
        raise NotImplementedError
    trans = np.asarray(trans, dtype=np.float32)
    prot_mask = np.asarray(prot_mask, dtype=bool)
    return trans, prot_mask


def argmin_random(
    count_tensor: torch.Tensor,
    neighbours: torch.LongTensor,
    batch_size: int = 1,
    repeat_per_residue: int = 3,
):
    # We first look at the individual counts for each residue
    counts = count_tensor.clamp(max=repeat_per_residue)
    # If the proportion of clamped counts is too high, we use the full count tensor
    if torch.sum(counts == repeat_per_residue).item() / len(counts) > 0.7:
        neighbour_counts = count_tensor
    else:
        neighbour_counts = counts[neighbours].sum(dim=-1)
    rand_idxs = torch.randperm(len(neighbour_counts))
    corr_idxs = torch.arange(len(neighbour_counts))[rand_idxs]
    random_argmin = neighbour_counts[rand_idxs].argsort()[:batch_size]
    original_argmin = corr_idxs[random_argmin]
    return original_argmin


def get_neighbour_idxs(polymer, k: int, idxs=None):
    # Get an initial set of pointers to neighbours for more efficient inference
    backbone_frames = polymer.rigidgroups_gt_frames[:, 0]  # (num_res, 3, 4)
    translation = get_affine_translation(backbone_frames)
    kd = cKDTree(translation)
    if idxs is None:
        _, init_neighbours = kd.query(translation, k=k)
    else:
        _, init_neighbours = kd.query(translation[idxs], k=k)
    return torch.from_numpy(init_neighbours)


def sample_na_aa_logits(logits_map, coords, origin, voxel_size):
    coords = np.asarray(coords, dtype=np.float32)
    if coords.size == 0:
        return np.zeros(coords.shape[:-1] + (logits_map.shape[0],), dtype=np.float32)

    logits_map = np.asarray(logits_map, dtype=np.float32)
    origin = np.asarray(origin, dtype=np.float32)
    voxel_size = np.asarray(voxel_size, dtype=np.float32)

    flat_coords = coords.reshape(-1, 3)
    grid_xyz = (flat_coords - origin[None, :]) / voxel_size[None, :]
    grid_idx = np.rint(grid_xyz).astype(np.int32)

    sampled = np.zeros((len(flat_coords), logits_map.shape[0]), dtype=np.float32)
    nxyz = np.asarray(logits_map.shape[1:][::-1], dtype=np.int32)
    valid = np.all((grid_idx >= 0) & (grid_idx < nxyz[None, :]), axis=-1)
    valid_idx = grid_idx[valid]
    sampled[valid] = logits_map[
        :,
        valid_idx[:, 2],
        valid_idx[:, 1],
        valid_idx[:, 0],
    ].T
    return sampled.reshape(coords.shape[:-1] + (logits_map.shape[0],))


def init_empty_collate_results(
    num_predicted_residues, unified_seq_len=None, device="cpu"
):
    result = {}
    result["counts"] = torch.zeros(num_predicted_residues, device=device)
    result["pred_positions"] = torch.zeros(num_predicted_residues, 3, device=device)
    result["pred_affines"] = torch.zeros(num_predicted_residues, 3, 4, device=device)
    result["pred_torsions"] = torch.zeros(
        num_predicted_residues, num_net_torsions, 2, device=device
    )

    # rmsd
    result["pred_rmsd"] = torch.zeros(
        num_predicted_residues, 1, device=device
    )

    # aatype
    result["pred_aatype"] = torch.zeros(
        num_predicted_residues, 24, device=device
    )

    # DNA/RNA logits
    result["pred_na_type_logits"] = torch.zeros(
        num_predicted_residues, 2, device=device
    )

    # recycle node state
    result["recycle_node_state"] = None

    # prot mask
    result["prot_mask"] = -1 * torch.ones(
        num_predicted_residues, device=device
    )

    # node existence
    result["pred_node_existence"] = torch.zeros(
        num_predicted_residues, 1, device=device
    )

    # edge existence dict
    result["pred_edge_existence_dict"] = dict()
    for i in range(num_predicted_residues):
        result["pred_edge_existence_dict"][i] = dict()

    # NA pairing stats keyed by undirected residue pair (min_idx, max_idx)
    result["pred_pairing_stats"] = dict()

    return result


def get_inference_data(
    polymer, grid_data, idxs, crop_length=200, num_devices: int = 1, na_aa_logits_data=None,
):
    cryo_grids = torch.from_numpy(grid_data.grid[None])  # Add channel dim
    backbone_frames = polymer.rigidgroups_gt_frames[:, 0]  # (num_res, 3, 4)
    translation = get_affine_translation(backbone_frames)
    picked_indices = np.arange(len(translation), dtype=int)

    batch = None
    batch_num = 1
    output_list = []
    batch_num_per_device = len(idxs) // num_devices
    for j in range(num_devices):
        if len(translation) >= crop_length:
            kd = cKDTree(translation)
            _, picked_indices = kd.query(
                translation[idxs[j * batch_num_per_device: (j + 1) * batch_num_per_device]], k=crop_length
            )
            batch_num = batch_num_per_device
            batch = torch.concat(
                [torch.ones(crop_length, dtype=torch.long) * i for i in range(batch_num)],
                dim=0,
            )


        # centering
        affines = backbone_frames[picked_indices]
        origin = grid_data.origin.astype(np.float32)

        output_dict = {
            "affines": torch.from_numpy(affines),
            "cryo_grids": cryo_grids,

            "cryo_global_origins": torch.from_numpy(origin),
            "cryo_voxel_sizes": torch.from_numpy(
                grid_data.voxel_size.astype(np.float32)
            ),
            "indices": torch.from_numpy(picked_indices),
            "prot_mask": torch.from_numpy(polymer.prot_mask[picked_indices]),
            "num_nodes": len(picked_indices),
            "batch_num": batch_num,
            "batch": batch,

        }

        if na_aa_logits_data is not None:
            selected_trans = get_affine_translation(affines)
            output_dict["na_aa_logits"] = torch.from_numpy(
                sample_na_aa_logits(
                    na_aa_logits_data["map"],
                    selected_trans,
                    na_aa_logits_data["origin"],
                    na_aa_logits_data["voxel_size"],
                )
            )

        if getattr(polymer, "prot_seq_embedding", None) is not None:
            output_dict["prot_seq_embed"] = torch.from_numpy(polymer.prot_seq_embedding)
        if getattr(polymer, "na_seq_embedding", None) is not None:
            output_dict["na_seq_embed"] = torch.from_numpy(polymer.na_seq_embedding)
        if getattr(polymer, "prev_aa_probs", None) is not None:
            output_dict["prev_aa_probs"] = torch.from_numpy(
                np.asarray(polymer.prev_aa_probs[picked_indices], dtype=np.float32)
            )
        if getattr(polymer, "prev_rmsd", None) is not None:
            output_dict["prev_rmsd"] = torch.from_numpy(
                np.asarray(polymer.prev_rmsd[picked_indices], dtype=np.float32)
            )
        if getattr(polymer, "prev_node", None) is not None:
            output_dict["prev_node"] = torch.from_numpy(
                np.asarray(polymer.prev_node[picked_indices], dtype=np.float32)
            )

        # residue index and chain index
        if polymer.residue_index is not None and \
            polymer.chain_index is not None:

            # add extra key and values
            output_dict["chain_index"] = torch.from_numpy( polymer.chain_index[picked_indices] )
            output_dict["residue_index"] = torch.from_numpy( polymer.residue_index[picked_indices] )

        output_list.append(output_dict)
    return output_list


def update_polymer_gt_frames(
    polymer, update_indices: np.ndarray, update_affines: np.ndarray
):
    # Index the original array directly: chained advanced indexing writes to a copy.
    polymer.rigidgroups_gt_frames[update_indices, 0] = update_affines
    return polymer


def collate_nn_results(
    collated_results, results, indices, polymer, num_pred_residues=50, offset=0,
):
    update_slice = np.s_[offset : num_pred_residues + offset]
    collated_results["counts"][indices[update_slice]] += 1

    # update positions
    collated_results["pred_positions"][indices[update_slice]] += results[
        "pred_positions"
    ][-1][update_slice]

    # update torsions
    #collated_results["pred_torsions"][indices[update_slice]] += F.normalize(
    #    results["pred_torsions"][update_slice], p=2, dim=-1
    #)
    collated_results["pred_torsions"][indices[update_slice]] += F.normalize(
        results["pred_torsions"][-1][update_slice], p=2, dim=-1
    )

    curr_pos_avg = (
        collated_results["pred_positions"][indices[update_slice]]
        / collated_results["counts"][indices[update_slice]][..., None]
    )

    # update affines
    collated_results["pred_affines"][indices[update_slice]] = get_affine(
        get_affine_rot(results["pred_affines"][-1][update_slice]).cpu(), curr_pos_avg
    )

    # update rmsd
    if "pred_rmsd" in results:
        collated_results["pred_rmsd"][indices[update_slice]] += results["pred_rmsd"][update_slice]

    # update aatype
    if "pred_aatype" in results:
        collated_results["pred_aatype"][indices[update_slice]] += results["pred_aatype"][update_slice]

    # update DNA/RNA logits
    if "pred_na_type_logits" in results and results["pred_na_type_logits"] is not None:
        collated_results["pred_na_type_logits"][indices[update_slice]] += results["pred_na_type_logits"][update_slice]

    # update recycle node state
    if "recycle_node_state" in results and results["recycle_node_state"] is not None:
        if collated_results["recycle_node_state"] is None:
            d_node = results["recycle_node_state"].shape[-1]
            collated_results["recycle_node_state"] = torch.zeros(
                len(collated_results["counts"]),
                d_node,
                device=collated_results["counts"].device,
            )
        collated_results["recycle_node_state"][indices[update_slice]] += results[
            "recycle_node_state"
        ][update_slice]

    # update prot mask
    if "prot_mask" in results:
        collated_results["prot_mask"][indices[update_slice]] = results["prot_mask"][update_slice]

    # update node existence += -> =
    if "pred_node_existence" in results and results["pred_node_existence"] is not None:
        collated_results["pred_node_existence"][indices[update_slice]] = results["pred_node_existence"][update_slice]


    # update edge must using =
    if "pred_edge_existence" in results:
        pred_edge_existence = torch.softmax(
            results["pred_edge_existence"],
            dim=-1,
        ).cpu().numpy()  # (n, k, 3), [NO_EDGE, NEXT, PREV]

        edge_index = results["edge_index"].long().cpu().numpy()  # (n, k)

        n, k = edge_index.shape
        indices_np = indices.cpu().numpy()

        src_local = np.repeat(np.arange(n), k)         # (n*k,)
        dst_local = edge_index.reshape(-1)             # (n*k,)
        scores = pred_edge_existence.reshape(-1, 3)    # (n*k, 3)

        src_global = indices_np[src_local]
        dst_global = indices_np[dst_local]

        # mask for acceleration:
        # only keep edges whose NEXT or PREV prob is not too low
        edge_prob = scores[:, 1:].max(axis=-1)         # max(NEXT, PREV), shape (n*k,)
        mask = edge_prob >= 0.1

        for s, d, p in zip(src_global[mask], dst_global[mask], scores[mask, 0:]):
            collated_results["pred_edge_existence_dict"][s][d] = p

        # print for debug
        #for key, value in collated_results["pred_edge_existence_dict"].items():
        #    print(key, value)
        #exit()

    # update NA pairing stats using undirected pair averaging
    if "pred_pairing" in results and results["pred_pairing"] is not None:
        pred_pairing = torch.sigmoid(results["pred_pairing"]).reshape(-1).cpu().numpy()
        edge_index = results["edge_index"].long().cpu().numpy()
        prot_mask_local = results["prot_mask"].bool().cpu().numpy()

        n, k = edge_index.shape
        indices_np = indices.cpu().numpy()

        src_local = np.repeat(np.arange(n), k)
        dst_local = edge_index.reshape(-1)

        src_global = indices_np[src_local]
        dst_global = indices_np[dst_local]

        na_na_mask = (~prot_mask_local[src_local]) & (~prot_mask_local[dst_local])
        prob_mask = pred_pairing > 0.10
        mask = na_na_mask & prob_mask

        for s, d, p in zip(src_global[mask], dst_global[mask], pred_pairing[mask]):
            s = int(s)
            d = int(d)
            if s == d:
                continue
            key = (s, d) if s < d else (d, s)
            if key not in collated_results["pred_pairing_stats"]:
                collated_results["pred_pairing_stats"][key] = [0.0, 0]
            collated_results["pred_pairing_stats"][key][0] += float(p)
            collated_results["pred_pairing_stats"][key][1] += 1

    # update polymer += -> =
    polymer = update_polymer_gt_frames(
        polymer,
        indices[update_slice].numpy(),
        collated_results["pred_affines"][indices[update_slice]].numpy(),
    )
    return collated_results, polymer


@torch.no_grad()
def run_inference_on_data(
    module,
    meta_batch_list,
    run_iters: int = 2,
    seq_attention_batch_size: int = 200,
    fp16: bool = False,
    using_cache: bool = False,
):
    with_seq = "seq_embed" in meta_batch_list[0]
    meta_input_list = []
    for data in meta_batch_list:
        affines = data["affines"]
        kwargs = {
            "affines": affines,
            "run_iters": run_iters,
        }

        if with_seq:
            kwargs["seq_attention_batch_size"] = seq_attention_batch_size

        # for residue index and chain index
        if "residue_index" in data and \
            "chain_index" in data:

            # add new key and values
            kwargs["chain_index"] = data["chain_index"]
            kwargs["residue_index"] = data["chain_index"] * int(1e5) + data["residue_index"]

        # mol type
        kwargs["prot_mask"] = data["prot_mask"].long()

        # others
        if data["batch_num"] == 1:
            if with_seq:
                kwargs["seq_embed"] = data["seq_embed"][None]
                kwargs["seq_embed_mask"] = torch.ones(1, data["seq_embed"].shape[0])

            kwargs["batch"] = None
            kwargs["cryo_grids"] = [data["cryo_grids"]]
            kwargs["cryo_global_origins"] = [data["cryo_global_origins"]]
            kwargs["cryo_voxel_sizes"] = [data["cryo_voxel_sizes"]]
        else:
            if with_seq:
                kwargs["seq_embed"] = (
                    data["seq_embed"][None].expand(data["batch_num"], -1, -1,)
                )
                kwargs["seq_embed_mask"] = torch.ones(
                    data["batch_num"], data["seq_embed"].shape[0],
                )

            kwargs["batch"] = data["batch"]
            kwargs["cryo_grids"] = [
                data["cryo_grids"] for _ in range(data["batch_num"])
            ]
            kwargs["cryo_global_origins"] = [
                data["cryo_global_origins"] for _ in range(data["batch_num"])
            ]
            kwargs["cryo_voxel_sizes"] = [
                data["cryo_voxel_sizes"] for _ in range(data["batch_num"])
            ]

        meta_input_list.append(kwargs)
    result = module(meta_input_list)
    return result


def init_polymer_from_translation(filename: str):
    translation, prot_mask = load_translation_from_file(filename) # (N, 3)

    rigidgroups_gt_frames = np.zeros((len(translation), 1, 3, 4), dtype=np.float32)
    rigidgroups_gt_frames[:, 0] = init_random_affine_from_translation(
        torch.from_numpy(translation),
    ).numpy()
    rigidgroups_gt_exists = np.ones((len(translation), 1), dtype=np.float32)
    residue_mask = np.ones(len(rigidgroups_gt_exists), dtype=bool)

    return get_polymer_empty_except(
        rigidgroups_gt_frames=rigidgroups_gt_frames[residue_mask],
        rigidgroups_gt_exists=rigidgroups_gt_exists[residue_mask],
        prot_mask=prot_mask[residue_mask],
    )


def get_final_nn_results(collated_results):
    final_results = {}

    final_results["pred_positions"] = (
        collated_results["pred_positions"] / collated_results["counts"][..., None]
    )
    final_results["pred_torsions"] = (
        collated_results["pred_torsions"] / collated_results["counts"][..., None, None]
    )
    final_results["pred_affines"] = get_affine(
        get_affine_rot(collated_results["pred_affines"]),
        final_results["pred_positions"],
    )

    # rmsd
    if "pred_rmsd" in collated_results:
        final_results["pred_rmsd"] = (
            collated_results["pred_rmsd"] / collated_results["counts"][..., None]
        )[..., 0]

    # aatype
    if "pred_aatype" in collated_results:
        final_results["pred_aatype"] = (
            collated_results["pred_aatype"] / collated_results["counts"][..., None]
        )

    # DNA/RNA logits
    if "pred_na_type_logits" in collated_results:
        final_results["pred_na_type_logits"] = (
            collated_results["pred_na_type_logits"] / collated_results["counts"][..., None]
        )

    if collated_results.get("recycle_node_state", None) is not None:
        final_results["recycle_node_state"] = (
            collated_results["recycle_node_state"]
            / collated_results["counts"][..., None]
        )

    # prot mask
    if "prot_mask" in collated_results:
        final_results["prot_mask"] = collated_results["prot_mask"]

    # node existence
    if "pred_node_existence" in collated_results:
        final_results["pred_node_existence"] = collated_results["pred_node_existence"][..., 0]

    if "pred_edge_existence_dict" in collated_results:
        final_results["pred_edge_existence_dict"] = collated_results["pred_edge_existence_dict"]

    if "pred_pairing_stats" in collated_results:
        pred_pairing_dict = dict()
        for (i, j), (prob_sum, prob_count) in collated_results["pred_pairing_stats"].items():
            if prob_count <= 0:
                continue
            mean_prob = float(prob_sum) / float(prob_count)
            if i not in pred_pairing_dict:
                pred_pairing_dict[i] = dict()
            if j not in pred_pairing_dict:
                pred_pairing_dict[j] = dict()
            pred_pairing_dict[i][j] = mean_prob
            pred_pairing_dict[j][i] = mean_prob
        final_results["pred_pairing_dict"] = pred_pairing_dict

        # print for debug
        #for key, value in final_results["pred_pairing_dict"].items():
        #    print(key, value)
        #exit()

    return dict([(k, v.numpy()) if torch.is_tensor(v) else (k, v) for (k, v) in final_results.items()])


