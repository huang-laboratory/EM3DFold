"""Exact CPU spatial index for fixed-radius mode merging.

The cell edge is the merge radius. Every point within that radius is in the
query cell or one of its 26 neighbours.
"""

from itertools import product

import numpy as np


_OFFSETS_27 = np.asarray(list(product((-1, 0, 1), repeat=3)), dtype=np.int64)


def _cell_index(points: np.ndarray, cell_size: float) -> np.ndarray:
    if not np.isfinite(cell_size) or cell_size <= 0.0:
        raise ValueError("cell_size must be finite and greater than zero")
    return np.floor(points / cell_size).astype(np.int64)


def _cell_map(cell_index: np.ndarray) -> dict[tuple[int, int, int], np.ndarray]:
    buckets: dict[tuple[int, int, int], list[int]] = {}
    for idx, cell in enumerate(cell_index):
        key = (int(cell[0]), int(cell[1]), int(cell[2]))
        buckets.setdefault(key, []).append(idx)
    return {key: np.asarray(ids, dtype=np.int32) for key, ids in buckets.items()}


def _neighbour_ids(
    cell: np.ndarray,
    buckets: dict[tuple[int, int, int], np.ndarray],
) -> np.ndarray:
    parts = []
    for offset in _OFFSETS_27:
        key_array = cell + offset
        ids = buckets.get((int(key_array[0]), int(key_array[1]), int(key_array[2])))
        if ids is not None:
            parts.append(ids)
    if not parts:
        return np.empty(0, dtype=np.int32)
    if len(parts) == 1:
        return parts[0]
    return np.concatenate(parts)


def select_centers_spatial(
    points_world: np.ndarray,
    densities: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Greedy radius suppression with NumPy-compatible order and boundary."""
    cells = _cell_index(points_world, radius)
    buckets = _cell_map(cells)
    order = np.argsort(-densities)
    alive = np.ones(len(points_world), dtype=bool)
    centers_idx: list[int] = []
    radius2 = radius * radius

    for idx_value in order:
        idx = int(idx_value)
        if not alive[idx]:
            continue
        centers_idx.append(idx)
        neighbour_ids = _neighbour_ids(cells[idx], buckets)
        delta = points_world[neighbour_ids] - points_world[idx][None, :]
        distance2 = np.sum(delta * delta, axis=1)
        alive[neighbour_ids[distance2 <= radius2]] = False

    return np.asarray(centers_idx, dtype=np.int32)


def _best_candidate(
    point: np.ndarray,
    centers: np.ndarray,
    ids: np.ndarray,
) -> tuple[np.generic, int]:
    delta = centers[ids] - point[None, :]
    distance2 = np.sum(delta * delta, axis=1)
    best_distance2 = np.min(distance2)
    # Bucket order differs from center-array order. Preserve np.argmin's first
    # (lowest center index) result for exactly equal distances.
    best_id = int(np.min(ids[distance2 == best_distance2]))
    return best_distance2, best_id


def _nearest_by_shells(
    point: np.ndarray,
    query_cell: np.ndarray,
    centers: np.ndarray,
    buckets: dict[tuple[int, int, int], np.ndarray],
    cell_size: float,
) -> int:
    """Exact nearest-cell search for points excluded by density filtering."""
    best_distance2 = np.float32(np.inf)
    best_id = np.iinfo(np.int32).max
    shell = 0
    while True:
        parts = []
        for dx in range(-shell, shell + 1):
            for dy in range(-shell, shell + 1):
                for dz in range(-shell, shell + 1):
                    if shell and max(abs(dx), abs(dy), abs(dz)) != shell:
                        continue
                    key = (
                        int(query_cell[0]) + dx,
                        int(query_cell[1]) + dy,
                        int(query_cell[2]) + dz,
                    )
                    ids = buckets.get(key)
                    if ids is not None:
                        parts.append(ids)
        if parts:
            ids = parts[0] if len(parts) == 1 else np.concatenate(parts)
            shell_distance2, shell_id = _best_candidate(point, centers, ids)
            if shell_distance2 < best_distance2 or (
                shell_distance2 == best_distance2 and shell_id < best_id
            ):
                best_distance2 = shell_distance2
                best_id = shell_id

        # All unvisited cells are outside this closed cube. Continue on
        # equality so a tied center with a lower index cannot be missed.
        low = (query_cell - shell) * cell_size
        high = (query_cell + shell + 1) * cell_size
        distance_to_unvisited = np.min(np.minimum(point - low, high - point))
        if best_id != np.iinfo(np.int32).max and best_distance2 < distance_to_unvisited**2:
            return best_id
        shell += 1


def membership_spatial(
    points_world: np.ndarray,
    centers_world: np.ndarray,
    radius: float,
    known_within_radius: np.ndarray | None = None,
) -> np.ndarray:
    """Find exact nearest centers without constructing an N-by-C matrix.

    ``known_within_radius`` marks candidates that participated in greedy
    suppression. Every such point has a selected center within ``radius``;
    other points use an exact expanding-shell search.
    """
    if len(centers_world) == 0:
        raise ValueError("centers_world must not be empty")
    point_cells = _cell_index(points_world, radius)
    center_cells = _cell_index(centers_world, radius)
    center_buckets = _cell_map(center_cells)
    if known_within_radius is None:
        known_within_radius = np.zeros(len(points_world), dtype=bool)
    else:
        known_within_radius = np.asarray(known_within_radius, dtype=bool)
        if known_within_radius.shape != (len(points_world),):
            raise ValueError("known_within_radius has the wrong shape")

    membership = np.empty(len(points_world), dtype=np.int32)
    known_ids = np.flatnonzero(known_within_radius).astype(np.int32, copy=False)
    if len(known_ids):
        # Group queries by cell and reuse each adjacent-center list.
        query_buckets: dict[tuple[int, int, int], list[int]] = {}
        for idx in known_ids:
            cell = point_cells[idx]
            key = (int(cell[0]), int(cell[1]), int(cell[2]))
            query_buckets.setdefault(key, []).append(int(idx))
        for key, query_list in query_buckets.items():
            query_ids = np.asarray(query_list, dtype=np.int32)
            center_ids = _neighbour_ids(np.asarray(key, dtype=np.int64), center_buckets)
            if len(center_ids) == 0:
                raise RuntimeError("greedy-suppressed point has no nearby center cell")
            delta = points_world[query_ids, None, :] - centers_world[center_ids][None, :, :]
            distance2 = np.sum(delta * delta, axis=2)
            sort_order = np.argsort(center_ids)
            sorted_ids = center_ids[sort_order]
            local = np.argmin(distance2[:, sort_order], axis=1)
            membership[query_ids] = sorted_ids[local]

    for idx in np.flatnonzero(~known_within_radius):
        membership[idx] = _nearest_by_shells(
            points_world[idx],
            point_cells[idx],
            centers_world,
            center_buckets,
            radius,
        )
    return membership
