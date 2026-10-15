from __future__ import annotations

import argparse
import math
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

try:
    from scipy import ndimage as ndi
    from scipy.spatial import cKDTree
except ImportError:  # pragma: no cover - optional runtime dependency
    ndi = None
    cKDTree = None

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - optional runtime dependency
    torch = None
    F = None

from em3dfold.pipeline.spatial_hash import membership_spatial, select_centers_spatial
from em3dfold.pipeline.torch_merge import membership_torch
from em3dfold.utils.cryo_utils import parse_map


@dataclass
class MRCMap:
    data: np.ndarray  # z, y, x
    origin: np.ndarray  # x, y, z
    voxel_size: np.ndarray  # x, y, z


@dataclass
class Params:
    threshold: float = 5.0
    resolution: float = 6.0
    rshift: float = 10.0
    rmerge: float = 1.0
    filter_fraction: float = 0.00
    backend: str = "auto"
    device: str = "auto"
    max_shift_iterations: int = 1000
    merge_backend: str = "auto"
    merge_block_size: int = 256


def log_message(message: str) -> None:
    print(f"# {message}", flush=True)


@contextmanager
def stage_timer(name: str):
    start = time.time()
    yield
    end = time.time()
    log_message(f"{name} finished in {end - start:.4f}s")


def _read_pdb_points(filename: str) -> np.ndarray:
    coords = []
    with open(filename, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(("ATOM", "HETATM")):
                coords.append(
                    [
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]),
                    ]
                )
    return np.asarray(coords, dtype=np.float32)


def _write_pdb_points(
    filename: str,
    coords: np.ndarray,
    bfactors: np.ndarray,
    atom_name: str = "CA",
    res_name: str = "GLY",
    chain_id: str = "A",
    element: str | None = None,
) -> None:
    atom_name = atom_name.strip()
    res_name = res_name.strip()
    chain_id = (chain_id or "A")[0]
    if element is None:
        element = atom_name.replace("'", "").strip()[0]

    atom_name_pdb = f"{atom_name:>4s}"[:4]
    res_name_pdb = f"{res_name:>3s}"[:3]
    element_pdb = f"{element:>2s}"[:2]

    with open(filename, "w", encoding="utf-8") as handle:
        for idx, (coord, b) in enumerate(zip(coords, bfactors), start=1):
            atom_serial = min(idx, 99999)
            residue_serial = min(idx, 9999)
            handle.write(
                "ATOM  {:>5d} {:4s} {:3s} {}{:>4d}    {:8.3f}{:8.3f}{:8.3f}{:>6.2f}{:>6.2f}          {:>2s}\n".format(
                    atom_serial,
                    atom_name_pdb,
                    res_name_pdb,
                    chain_id,
                    residue_serial,
                    float(coord[0]),
                    float(coord[1]),
                    float(coord[2]),
                    float(b),
                    float(b),
                    element_pdb,
                )
            )
            handle.write("TER\n")


def _load_mrc_map(map_path: str) -> MRCMap:
    data, origin, _, voxel_size = parse_map(map_path, False, None)
    return MRCMap(
        data=np.asarray(data, dtype=np.float32),
        origin=np.asarray(origin, dtype=np.float32),
        voxel_size=np.asarray(voxel_size, dtype=np.float32),
    )


def _voxel_step(mrc: MRCMap) -> float:
    voxel_size = np.asarray(mrc.voxel_size, dtype=np.float32)
    if np.max(np.abs(voxel_size - voxel_size[0])) > 1e-4:
        log_message(
            "WARNING anisotropic voxel size detected; using mean voxel size for shift and merge radii"
        )
    return float(np.mean(voxel_size))


def _resolution_to_kernel_scale(resolution: float) -> float:
    if resolution <= 4.0:
        return 2.0
    if resolution <= 5.0:
        return 2.5
    if resolution <= 6.0:
        return 3.0
    if resolution <= 7.0:
        return 3.5
    if resolution <= 8.0:
        return 4.0
    return 4.5


def _kernel_params(mrc: MRCMap, params: Params) -> tuple[float, int, float]:
    gstep = _voxel_step(mrc)
    dreso = _resolution_to_kernel_scale(params.resolution)
    fs = ((dreso / gstep) * 0.5) ** 2
    kernel_coef = -1.5 / fs
    radius = max(1, int(math.ceil((dreso / gstep) * 2.0)))
    return kernel_coef, radius, gstep


def _thresholded_map(mrc: MRCMap, threshold: float) -> np.ndarray:
    data = np.asarray(mrc.data, dtype=np.float32)
    return np.where(data > threshold, data, 0.0).astype(np.float32, copy=False)


def map_to_grid_points(mrc: MRCMap, threshold: float) -> np.ndarray:
    valid_zyx = np.argwhere(mrc.data > threshold)
    if valid_zyx.size == 0:
        raise ValueError("No valid grid points were found above the current threshold.")
    # NumPy returns z-y-x indices; getp works in x-y-z voxel coordinates.
    return valid_zyx[:, ::-1].astype(np.float32, copy=False)


def _normalize_densities(densities: np.ndarray) -> np.ndarray:
    if len(densities) == 0:
        return np.zeros((0,), dtype=np.float32)
    densities = np.asarray(densities, dtype=np.float32)
    min_dens = float(np.min(densities))
    max_dens = float(np.max(densities))
    if max_dens - min_dens < 1e-8:
        return np.ones(len(densities), dtype=np.float32)
    return ((densities - min_dens) / (max_dens - min_dens + 1e-6)).astype(np.float32, copy=False)


def _kernel_1d_np(kernel_coef: float, radius: int) -> np.ndarray:
    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    return np.exp(kernel_coef * np.square(offsets)).astype(np.float32, copy=False)


def _gaussian_kernel_1d_torch(kernel_coef: float, radius: int, device: torch.device) -> torch.Tensor:
    offsets = torch.arange(-radius, radius + 1, device=device, dtype=torch.float64)
    return torch.exp(kernel_coef * offsets.square())


def _resolve_torch_device(device: str) -> str:
    requested = str(device)
    if requested.isdigit():
        requested = f"cuda:{requested}"
    if requested == "auto":
        return "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
    if requested.startswith("cuda") and (torch is None or not torch.cuda.is_available()):
        log_message(f"WARNING requested torch device '{requested}' is unavailable; fall back to cpu")
        return "cpu"
    return requested


def _conv3d_separable(volume: torch.Tensor, kernel_1d: torch.Tensor, radius: int) -> torch.Tensor:
    kernel_x = kernel_1d.view(1, 1, 1, 1, -1)
    kernel_y = kernel_1d.view(1, 1, 1, -1, 1)
    kernel_z = kernel_1d.view(1, 1, -1, 1, 1)
    out = F.conv3d(volume, kernel_x, padding=(0, 0, radius))
    out = F.conv3d(out, kernel_y, padding=(0, radius, 0))
    out = F.conv3d(out, kernel_z, padding=(radius, 0, 0))
    return out


def _sample_torch(field: torch.Tensor, positions_zyx: torch.Tensor, shape_zyx: tuple[int, int, int]) -> torch.Tensor:
    nz, ny, nx = shape_zyx
    z = 2.0 * positions_zyx[:, 0] / max(nz - 1, 1) - 1.0
    y = 2.0 * positions_zyx[:, 1] / max(ny - 1, 1) - 1.0
    x = 2.0 * positions_zyx[:, 2] / max(nx - 1, 1) - 1.0
    grid = torch.stack((x, y, z), dim=-1).view(1, -1, 1, 1, 3)
    sampled = F.grid_sample(field, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return sampled.view(-1)


def _filter1d_separable(volume: np.ndarray, kernel_1d: np.ndarray) -> np.ndarray:
    if ndi is None:
        raise RuntimeError("scipy is required for the scipy mean-shift backend.")
    out = ndi.convolve1d(volume, kernel_1d, axis=2, mode="constant", cval=0.0)
    out = ndi.convolve1d(out, kernel_1d, axis=1, mode="constant", cval=0.0)
    out = ndi.convolve1d(out, kernel_1d, axis=0, mode="constant", cval=0.0)
    return out.astype(np.float32, copy=False)


def _mean_shift_torch(
    map_data: np.ndarray,
    mrc: MRCMap,
    grid_points_xyz: np.ndarray,
    params: Params,
    device_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    if torch is None or F is None:
        raise RuntimeError("PyTorch is not available for the torch mean-shift backend.")

    device = torch.device(device_name)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    # Use float64 to make the torch implementation closer to scipy / torch CPU.
    volume = torch.from_numpy(np.ascontiguousarray(map_data).astype(np.float64, copy=False)).to(device=device).unsqueeze(0).unsqueeze(0)
    nz, ny, nx = volume.shape[-3:]

    kernel_coef, radius, gstep = _kernel_params(mrc, params)
    kernel_1d = _gaussian_kernel_1d_torch(kernel_coef, radius, device)

    xs = torch.arange(nx, device=device, dtype=torch.float64).view(1, 1, 1, 1, nx)
    ys = torch.arange(ny, device=device, dtype=torch.float64).view(1, 1, 1, ny, 1)
    zs = torch.arange(nz, device=device, dtype=torch.float64).view(1, 1, nz, 1, 1)

    smooth0 = _conv3d_separable(volume, kernel_1d, radius)
    smoothx = _conv3d_separable(volume * xs, kernel_1d, radius)
    smoothy = _conv3d_separable(volume * ys, kernel_1d, radius)
    smoothz = _conv3d_separable(volume * zs, kernel_1d, radius)

    positions = torch.from_numpy(grid_points_xyz.astype(np.float64, copy=False)).to(device=device)
    start = positions.clone()
    shift2_limit = (params.rshift / gstep) ** 2

    for _ in range(params.max_shift_iterations):
        sample_zyx = torch.stack((positions[:, 2], positions[:, 1], positions[:, 0]), dim=1)
        denom = torch.clamp(_sample_torch(smooth0, sample_zyx, (nz, ny, nx)), min=1e-8)
        after = torch.stack(
            (
                _sample_torch(smoothx, sample_zyx, (nz, ny, nx)) / denom,
                _sample_torch(smoothy, sample_zyx, (nz, ny, nx)) / denom,
                _sample_torch(smoothz, sample_zyx, (nz, ny, nx)) / denom,
            ),
            dim=1,
        )
        step2 = torch.sum((after - positions) ** 2, dim=1)
        total2 = torch.sum((after - start) ** 2, dim=1)
        positions = after
        if torch.all((step2 < 1e-3) | (total2 > shift2_limit)):
            break

    sample_zyx = torch.stack((positions[:, 2], positions[:, 1], positions[:, 0]), dim=1)
    densities = _sample_torch(smooth0, sample_zyx, (nz, ny, nx))
    return (
        positions.detach().cpu().numpy().astype(np.float32, copy=False),
        densities.detach().cpu().numpy().astype(np.float32, copy=False),
    )


def _mean_shift_scipy(
    map_data: np.ndarray,
    mrc: MRCMap,
    grid_points_xyz: np.ndarray,
    params: Params,
) -> tuple[np.ndarray, np.ndarray]:
    if ndi is None:
        raise RuntimeError("scipy is required for the scipy mean-shift backend.")
    kernel_coef, radius, gstep = _kernel_params(mrc, params)
    kernel_1d = _kernel_1d_np(kernel_coef, radius)

    zz, yy, xx = np.indices(map_data.shape, dtype=np.float32)
    smooth0 = _filter1d_separable(map_data, kernel_1d)
    smoothx = _filter1d_separable(map_data * xx, kernel_1d)
    smoothy = _filter1d_separable(map_data * yy, kernel_1d)
    smoothz = _filter1d_separable(map_data * zz, kernel_1d)

    positions = grid_points_xyz.astype(np.float32, copy=True)
    start = positions.copy()
    shift2_limit = (params.rshift / gstep) ** 2

    for _ in range(params.max_shift_iterations):
        sample_zyx = np.stack((positions[:, 2], positions[:, 1], positions[:, 0]), axis=0)
        denom = np.maximum(ndi.map_coordinates(smooth0, sample_zyx, order=1, mode="nearest"), 1e-8)
        after = np.column_stack(
            [
                ndi.map_coordinates(smoothx, sample_zyx, order=1, mode="nearest") / denom,
                ndi.map_coordinates(smoothy, sample_zyx, order=1, mode="nearest") / denom,
                ndi.map_coordinates(smoothz, sample_zyx, order=1, mode="nearest") / denom,
            ]
        ).astype(np.float32, copy=False)
        step2 = np.sum((after - positions) ** 2, axis=1)
        total2 = np.sum((after - start) ** 2, axis=1)
        positions = after
        if np.all((step2 < 1e-3) | (total2 > shift2_limit)):
            break

    sample_zyx = np.stack((positions[:, 2], positions[:, 1], positions[:, 0]), axis=0)
    densities = ndi.map_coordinates(smooth0, sample_zyx, order=1, mode="nearest").astype(np.float32, copy=False)
    return positions, densities


def _resolve_mode_merge_backend(params: Params) -> tuple[str, bool]:
    configured = os.environ.get(
        "EM3DFOLD_MODE_MERGE_BACKEND", params.merge_backend
    ).strip().lower()
    aliases = {"cpu": "spatial", "cuda": "torch", "gpu": "torch"}
    configured = aliases.get(configured, configured)
    choices = {"auto", "spatial", "torch", "ckdtree"}
    if configured not in choices:
        raise ValueError(
            "mode-merge backend must be one of auto, spatial, torch, or ckdtree"
        )
    automatic = configured == "auto"
    if automatic:
        resolved_device = _resolve_torch_device(params.device)
        configured = "torch" if str(resolved_device).startswith("cuda") else "spatial"
    if configured == "ckdtree" and cKDTree is None:
        raise RuntimeError("cKDTree mode merging was requested but SciPy is unavailable")
    return configured, automatic


def _merge_block_size(params: Params) -> int:
    value = os.environ.get(
        "EM3DFOLD_MODE_MERGE_BLOCK_SIZE", str(params.merge_block_size)
    )
    try:
        block_size = int(value)
    except ValueError as exc:
        raise ValueError("mode-merge block size must be an integer") from exc
    if block_size <= 0:
        raise ValueError("mode-merge block size must be greater than zero")
    return block_size


def _merge_modes(
    points_xyz: np.ndarray,
    densities: np.ndarray,
    params: Params,
    mrc: MRCMap,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(points_xyz) == 0:
        raise ValueError("Mean-shift produced no candidate points.")

    normalized_densities = _normalize_densities(densities)
    valid = normalized_densities >= params.filter_fraction
    candidates = points_xyz[valid]
    candidate_dens = normalized_densities[valid]
    if len(candidates) == 0:
        raise ValueError("No candidate points survived the current filter threshold.")

    if len(candidates) == 1:
        membership = np.zeros(len(points_xyz), dtype=np.int32)
        return (
            candidates.astype(np.float32, copy=False),
            candidate_dens.astype(np.float32, copy=False),
            membership,
        )

    merge_radius = float(params.rmerge)
    candidates_world = candidates * mrc.voxel_size[None, :]
    merge_backend, automatic = _resolve_mode_merge_backend(params)
    if merge_backend == "ckdtree":
        order = np.argsort(-candidate_dens)
        alive = np.ones(len(candidates), dtype=bool)
        centers_idx: list[int] = []
        tree = cKDTree(candidates_world)
        for idx in order:
            if not alive[idx]:
                continue
            centers_idx.append(int(idx))
            neighbors = np.asarray(tree.query_ball_point(candidates_world[idx], r=merge_radius), dtype=np.int32)
            alive[neighbors] = False
    else:
        centers_idx = select_centers_spatial(
            candidates_world, candidate_dens, merge_radius
        ).tolist()

    center_ids = np.asarray(centers_idx, dtype=np.int32)
    centers = candidates[center_ids]
    center_densities = candidate_dens[center_ids]

    points_world = points_xyz * mrc.voxel_size[None, :]
    centers_world = centers * mrc.voxel_size[None, :]
    if merge_backend == "ckdtree":
        member_tree = cKDTree(centers * mrc.voxel_size[None, :])
        membership = member_tree.query(points_world)[1].astype(np.int32, copy=False)
    elif merge_backend == "torch":
        try:
            membership = membership_torch(
                points_world,
                centers_world,
                device=_resolve_torch_device(params.device),
                block_size=_merge_block_size(params),
            )
        except RuntimeError as exc:
            if not automatic:
                raise
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
            log_message(
                f"WARNING Torch GPU mode merging failed ({exc}); falling back to CPU spatial hash"
            )
            merge_backend = "spatial"
            membership = membership_spatial(
                points_world,
                centers_world,
                merge_radius,
                known_within_radius=valid,
            )
    else:
        membership = membership_spatial(
            points_world,
            centers_world,
            merge_radius,
            known_within_radius=valid,
        )
    log_message(f"mode-merge backend = {merge_backend}")
    return (
        centers.astype(np.float32, copy=False),
        center_densities.astype(np.float32, copy=False),
        membership,
    )


def extract_points(
    mrc: MRCMap,
    params: Params,
    init_coords_xyz: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    thresholded = _thresholded_map(mrc, params.threshold)
    work_mrc = MRCMap(data=thresholded, origin=mrc.origin, voxel_size=mrc.voxel_size)

    if init_coords_xyz is None or len(init_coords_xyz) == 0:
        with stage_timer("grid point selection"):
            grid_points_xyz = map_to_grid_points(work_mrc, 0.0)
        log_message(f"valid grid points: {len(grid_points_xyz)}")
    else:
        with stage_timer("initial point projection"):
            grid_points_xyz = (
                (np.asarray(init_coords_xyz, dtype=np.float32) - mrc.origin[None, :])
                / np.maximum(mrc.voxel_size[None, :], 1e-6)
            )
            max_xyz = np.asarray(work_mrc.data.shape[::-1], dtype=np.float32) - 1.0
            grid_points_xyz = np.clip(grid_points_xyz, 0.0, max_xyz[None, :])
        log_message(f"initial points: {len(grid_points_xyz)}")

    backend = params.backend
    if backend == "auto":
        backend = "torch" if (torch is not None and torch.cuda.is_available()) else "scipy"

    if backend == "torch":
        device = _resolve_torch_device(params.device)
        log_message(f"mean-shift backend: torch ({device})")
        if str(device).startswith("cuda"):
            log_message(
                "torch GPU mean-shift may produce slightly different results from scipy and torch CPU implementations"
            )
            log_message("torch GPU mean-shift will use float64 and disable TF32 to better match CPU behavior")
        with stage_timer("mean-shift"):
            shifted, shifted_dens = _mean_shift_torch(thresholded, work_mrc, grid_points_xyz, params, device)
    elif backend == "scipy":
        log_message("mean-shift backend: scipy")
        with stage_timer("mean-shift"):
            shifted, shifted_dens = _mean_shift_scipy(thresholded, work_mrc, grid_points_xyz, params)
    else:
        raise ValueError(f"Unsupported backend: {params.backend}")

    with stage_timer("mode merging"):
        centers, center_densities, membership = _merge_modes(shifted, shifted_dens, params, work_mrc)

    points = centers * work_mrc.voxel_size[None, :] + work_mrc.origin[None, :]
    return points.astype(np.float32, copy=False), center_densities, membership


def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--in", "--map", dest="map", required=True, help="Input cryo-EM map")
    parser.add_argument("--out", "--output", dest="output", required=True, help="Output PDB path")
    parser.add_argument("--thresh", type=float, default=5.0, help="Map threshold")
    parser.add_argument("--res", type=float, default=6.0, help="Resolution")
    parser.add_argument("--rmax", type=float, default=10.0, help="Maximum shift radius in Angstrom")
    parser.add_argument("--dmerge", type=float, default=1.0, help="Merge distance in Angstrom")
    parser.add_argument("--filter", type=float, default=0.00, help="Low-bound normalized density filter")
    parser.add_argument("--pdb", type=str, default=None, help="Optional initial points in PDB format")
    parser.add_argument("--backend", choices=["auto", "torch", "scipy"], default="auto")
    parser.add_argument("--device", default="auto", help="Torch device, e.g. cuda:0 or cpu")
    parser.add_argument("--max-shift-iterations", type=int, default=5000)
    parser.add_argument(
        "--merge-backend",
        choices=["auto", "spatial", "torch", "ckdtree"],
        default="auto",
        help="Mode-merge backend; auto uses Torch on CUDA and spatial hash on CPU",
    )
    parser.add_argument(
        "--merge-block-size",
        type=int,
        default=256,
        help="Query block size for Torch GPU membership assignment",
    )
    parser.add_argument("--atom-name", default="CA")
    parser.add_argument("--res-name", default="GLY")
    parser.add_argument("--chain-id", default="A")
    parser.add_argument("--element", default=None)
    return parser


def main(args: argparse.Namespace) -> None:
    log_message("reading map")
    mrc = _load_mrc_map(args.map)

    init_coords = None
    if args.pdb:
        log_message(f"reading initial coordinates from {args.pdb}")
        init_coords = _read_pdb_points(args.pdb)

    params = Params(
        threshold=float(args.thresh),
        resolution=float(args.res),
        rshift=float(args.rmax),
        rmerge=float(args.dmerge),
        filter_fraction=float(args.filter),
        backend=str(args.backend),
        device=str(args.device),
        max_shift_iterations=int(args.max_shift_iterations),
        merge_backend=str(args.merge_backend),
        merge_block_size=int(args.merge_block_size),
    )

    log_message(f"threshold = {params.threshold:.4f}")
    log_message(f"resolution = {params.resolution:.4f}")
    log_message(f"rmax = {params.rshift:.4f}")
    log_message(f"dmerge = {params.rmerge:.4f}")
    log_message(f"filter = {params.filter_fraction:.4f}")
    log_message(f"merge backend = {params.merge_backend}")

    points, densities, _ = extract_points(mrc, params, init_coords_xyz=init_coords)
    log_message(f"retained points = {len(points)}")
    _write_pdb_points(
        args.output,
        points,
        densities,
        atom_name=args.atom_name,
        res_name=args.res_name,
        chain_id=args.chain_id,
        element=args.element,
    )
    log_message(f"wrote points to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Python mean-shift version of getp",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    main(add_args(parser).parse_args())
