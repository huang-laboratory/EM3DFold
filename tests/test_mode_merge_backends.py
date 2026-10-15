import os
import unittest
from unittest.mock import patch

import numpy as np

from em3dfold.pipeline import meanshift
from em3dfold.pipeline.spatial_hash import (
    membership_spatial,
    select_centers_spatial,
)
from em3dfold.pipeline.torch_merge import membership_torch


def numpy_reference(points, densities, radius):
    order = np.argsort(-densities)
    alive = np.ones(len(points), dtype=bool)
    centers = []
    radius2 = radius * radius
    for idx in order:
        if not alive[idx]:
            continue
        centers.append(int(idx))
        delta = points - points[idx][None, :]
        alive[np.sum(delta * delta, axis=1) <= radius2] = False
    center_ids = np.asarray(centers, dtype=np.int32)
    delta = points[:, None, :] - points[center_ids][None, :, :]
    membership = np.argmin(np.sum(delta * delta, axis=2), axis=1).astype(np.int32)
    return center_ids, membership


class SpatialHashTests(unittest.TestCase):
    def test_random_duplicates_negative_coordinates_and_radius_boundary(self):
        rng = np.random.default_rng(17)
        points = rng.normal(size=(300, 3)).astype(np.float32)
        points[:4] = np.asarray(
            [[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 0, 0]],
            dtype=np.float32,
        )
        densities = rng.random(len(points), dtype=np.float32)
        densities[:4] = [1.0, 0.8, 0.7, 0.6]
        expected_centers, expected_membership = numpy_reference(
            points, densities, 1.0
        )
        centers = select_centers_spatial(points, densities, 1.0)
        membership = membership_spatial(
            points,
            points[centers],
            1.0,
            known_within_radius=np.ones(len(points), dtype=bool),
        )
        np.testing.assert_array_equal(centers, expected_centers)
        np.testing.assert_array_equal(membership, expected_membership)

    def test_filtered_points_and_equal_distance_tie(self):
        centers = np.asarray([[0, 0, 0], [2, 0, 0]], dtype=np.float32)
        points = np.asarray([[0, 0, 0], [2, 0, 0], [1, 0, 0], [8, 0, 0]], dtype=np.float32)
        membership = membership_spatial(
            points,
            centers,
            radius=1.0,
            known_within_radius=np.asarray([True, True, False, False]),
        )
        np.testing.assert_array_equal(membership, [0, 1, 0, 1])


class MergeIntegrationTests(unittest.TestCase):
    def test_spatial_merge_matches_numpy_reference(self):
        points = np.asarray(
            [[0, 0, 0], [0.2, 0, 0], [3, 0, 0], [3.2, 0, 0]],
            dtype=np.float32,
        )
        densities = np.asarray([1.0, 0.1, 0.8, 0.2], dtype=np.float32)
        mrc = meanshift.MRCMap(
            data=np.zeros((1, 1, 1), dtype=np.float32),
            origin=np.zeros(3, dtype=np.float32),
            voxel_size=np.ones(3, dtype=np.float32),
        )
        params = meanshift.Params(rmerge=1.0, merge_backend="spatial")
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("EM3DFOLD_MODE_MERGE_BACKEND", None)
            centers, _, membership = meanshift._merge_modes(
                points, densities, params, mrc
            )
        expected_ids, expected_membership = numpy_reference(
            points, meanshift._normalize_densities(densities), 1.0
        )
        np.testing.assert_array_equal(centers, points[expected_ids])
        np.testing.assert_array_equal(membership, expected_membership)


class TorchMembershipTests(unittest.TestCase):
    def test_gpu_membership_matches_numpy_and_repairs_tie(self):
        if meanshift.torch is None or not meanshift.torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")
        centers = np.asarray([[0, 0, 0], [2, 0, 0], [5, 0, 0]], dtype=np.float32)
        points = np.asarray(
            [[0, 0, 0], [2, 0, 0], [1, 0, 0], [4.9, 0, 0]],
            dtype=np.float32,
        )
        result = membership_torch(points, centers, device="cuda", block_size=2)
        delta = points[:, None, :] - centers[None, :, :]
        expected = np.argmin(np.sum(delta * delta, axis=2), axis=1)
        np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
