"""Blocked Torch GPU membership assignment for mode merging."""

from __future__ import annotations

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - optional runtime dependency
    torch = None


def _resolve_device(device: str):
    if torch is None:
        raise RuntimeError("Torch mode merging was requested but torch is unavailable")
    requested = str(device)
    if requested.isdigit():
        requested = f"cuda:{requested}"
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    resolved = torch.device(requested)
    if resolved.type != "cuda":
        raise RuntimeError(f"Torch mode merging requires a CUDA device, got {resolved}")
    if not torch.cuda.is_available():
        raise RuntimeError("Torch mode merging requires CUDA, but CUDA is unavailable")
    return resolved


def _repair_ambiguous_memberships(
    points: np.ndarray,
    centers: np.ndarray,
    membership: np.ndarray,
    ambiguous: np.ndarray,
) -> None:
    """Recompute near-ties with NumPy to preserve its exact argmin rule."""
    for row in np.flatnonzero(ambiguous):
        delta = centers - points[row][None, :]
        distance2 = np.sum(delta * delta, axis=1)
        membership[row] = int(np.argmin(distance2))


def membership_torch(
    points_world: np.ndarray,
    centers_world: np.ndarray,
    device: str = "auto",
    block_size: int = 256,
) -> np.ndarray:
    """Assign points to centers with bounded GPU memory.

    Distances are evaluated in the input NumPy dtype. Queries whose two best
    GPU distances are within a conservative floating-point error band are
    recomputed with NumPy. This preserves NumPy's lowest-index tie behavior
    without giving up blocked GPU execution for ordinary points.
    """
    if block_size <= 0:
        raise ValueError("block_size must be greater than zero")
    points = np.ascontiguousarray(points_world)
    centers = np.ascontiguousarray(centers_world)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_world must have shape (N, 3)")
    if centers.ndim != 2 or centers.shape[1] != 3 or len(centers) == 0:
        raise ValueError("centers_world must have non-empty shape (C, 3)")
    if points.dtype not in (np.float32, np.float64):
        points = points.astype(np.float32)
    if centers.dtype != points.dtype:
        centers = centers.astype(points.dtype)

    resolved_device = _resolve_device(device)
    points_gpu = torch.as_tensor(points, device=resolved_device)
    centers_gpu = torch.as_tensor(centers, device=resolved_device)
    membership = np.empty(len(points), dtype=np.int32)
    epsilon = torch.finfo(points_gpu.dtype).eps

    for start in range(0, len(points), block_size):
        stop = min(start + block_size, len(points))
        delta = points_gpu[start:stop, None, :] - centers_gpu[None, :, :]
        distance2 = torch.sum(delta * delta, dim=2)
        if len(centers) == 1:
            membership[start:stop] = 0
            continue

        best2, ids2 = torch.topk(distance2, k=2, dim=1, largest=False, sorted=True)
        selected = ids2[:, 0].to(dtype=torch.int32).cpu().numpy()
        scale = torch.maximum(best2[:, 0].abs(), best2[:, 1].abs()).clamp_min(1.0)
        # Three squared components are accumulated. A factor of 32 is
        # intentionally conservative and sends all exact/near ties to NumPy.
        ambiguous = (best2[:, 1] - best2[:, 0] <= 32.0 * epsilon * scale).cpu().numpy()
        if np.any(ambiguous):
            _repair_ambiguous_memberships(
                points[start:stop], centers, selected, ambiguous
            )
        membership[start:stop] = selected

    torch.cuda.synchronize(resolved_device)
    return membership
