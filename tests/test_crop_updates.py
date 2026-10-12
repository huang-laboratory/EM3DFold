from types import SimpleNamespace

import numpy as np
import torch

from em3dfold.infer.gnn_inference_utils import (
    collate_nn_results,
    get_inference_data,
    init_empty_collate_results,
    update_polymer_gt_frames,
)


def make_polymer():
    frames = np.zeros((5, 2, 3, 4), dtype=np.float32)
    frames[:, :, :3, :3] = np.eye(3)
    frames[:, 0, 0, 3] = np.arange(5)
    return SimpleNamespace(
        rigidgroups_gt_frames=frames,
        prot_mask=np.ones(5, dtype=bool),
        residue_index=None,
        chain_index=None,
    )


def test_frame_update_writes_original_selected_backbone_frames():
    polymer = make_polymer()
    original = polymer.rigidgroups_gt_frames
    before = original.copy()
    indices = np.array([3, 1])
    updated = before[indices, 0].copy()
    updated[:, :, 3] += 100
    returned = update_polymer_gt_frames(polymer, indices, updated)
    assert returned is polymer and polymer.rigidgroups_gt_frames is original
    np.testing.assert_array_equal(original[indices, 0], updated)
    np.testing.assert_array_equal(original[[0, 2, 4]], before[[0, 2, 4]])
    np.testing.assert_array_equal(original[:, 1], before[:, 1])


def test_collation_changes_next_crop_geometry_and_neighbors():
    polymer = make_polymer()
    grid = SimpleNamespace(
        grid=np.zeros((2, 2, 2), dtype=np.float32),
        origin=np.zeros(3, dtype=np.float32),
        voxel_size=np.ones(3, dtype=np.float32),
    )
    first = get_inference_data(polymer, grid, torch.tensor([0]), crop_length=3)[0]
    assert first['indices'].tolist() == [0, 1, 2]
    predicted = first['affines'].clone()
    predicted[1, 0, 3] = 100
    predicted[2, 0, 3] = 900
    collated = init_empty_collate_results(5)
    results = {
        'pred_positions': [predicted[:, :, 3]],
        'pred_affines': [predicted],
        'pred_torsions': [torch.ones(3, collated['pred_torsions'].shape[1], 2)],
    }
    collated, polymer = collate_nn_results(
        collated, results, first['indices'], polymer, num_pred_residues=2,
    )
    assert collated['counts'].tolist() == [1, 1, 0, 0, 0]
    assert polymer.rigidgroups_gt_frames[1, 0, 0, 3] == 100
    assert polymer.rigidgroups_gt_frames[2, 0, 0, 3] == 2
    # The next crop queries the newly written coordinates, including new context.
    second = get_inference_data(polymer, grid, torch.tensor([0]), crop_length=3)[0]
    assert second['indices'].tolist() == [0, 2, 3]
    moved_center = get_inference_data(polymer, grid, torch.tensor([1]), crop_length=3)[0]
    assert moved_center['affines'][0, 0, 3] == 100
    np.testing.assert_array_equal(
        polymer.rigidgroups_gt_frames[first['indices'][:2], 0],
        collated['pred_affines'][first['indices'][:2]].numpy(),
    )
    predicted_again = predicted.clone()
    predicted_again[1, 0, 3] = 200
    results['pred_positions'] = [predicted_again[:, :, 3]]
    results['pred_affines'] = [predicted_again]
    collated, polymer = collate_nn_results(
        collated, results, first['indices'], polymer, num_pred_residues=2,
    )
    assert collated['counts'].tolist() == [2, 2, 0, 0, 0]
    assert polymer.rigidgroups_gt_frames[1, 0, 0, 3] == 150
    assert first['affines'][1, 0, 3] == 1
