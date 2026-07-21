import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from em3dfold.models.v3x2.bde import BackboneDistanceEmbedding
from em3dfold.scunet.scunet import ConvTransBlock
from em3dfold.utils.affine_utils import (
    sample_centered_cube_rot_matrix,
    sample_centered_rectangle_along_vector,
)
from em3dfold.utils.torch_utils import get_batches_to_idx


class EM3DSpatialMean(nn.Module):
    def forward(self, x):
        return x.mean(dim=[-3, -2, -1])


class EM3DLocalConvStem(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation_class=nn.ReLU,
        checkpoint: bool = True,
    ):
        super().__init__()
        self.use_checkpoint = checkpoint
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.norm = nn.InstanceNorm3d(out_channels, affine=True)
        self.act = activation_class()

    def _forward_impl(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

    def forward(self, x: torch.Tensor):
        if self.use_checkpoint and x.requires_grad:
            return torch_checkpoint(
                self._forward_impl,
                x,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        return self._forward_impl(x)


def _stride2_output_size(size: int) -> int:
    return (size + 1) // 2


def _round_up_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 1:
        return value
    return ((value + multiple - 1) // multiple) * multiple


def _select_trans_dim(
    channels: int,
    head_dim: int,
    trans_ratio: float,
    max_trans_dim: int,
) -> int:
    trans_dim = int(round(channels * trans_ratio))
    trans_dim = min(trans_dim, max_trans_dim, channels - 1)
    trans_dim = max(trans_dim, head_dim)
    trans_dim = (trans_dim // head_dim) * head_dim
    if trans_dim <= 0:
        trans_dim = head_dim
    if trans_dim >= channels:
        trans_dim = ((channels - 1) // head_dim) * head_dim
    if trans_dim <= 0 or trans_dim >= channels:
        raise ValueError(
            f"Cannot split channels={channels} into conv/trans branches with head_dim={head_dim}"
        )
    return trans_dim


class EM3DSwinConvMixer(nn.Module):
    def __init__(
        self,
        channels: int,
        input_resolution: int,
        head_dim: int = 16,
        window_size: int = 3,
        drop_path: float = 0.0,
        trans_ratio: float = 0.25,
        max_trans_dim: int = 128,
        block_type: str = "W",
        checkpoint: bool = True,
    ):
        super().__init__()
        self.window_size = window_size
        self.use_checkpoint = checkpoint

        padded_resolution = _round_up_to_multiple(
            max(input_resolution, window_size),
            window_size,
        )
        trans_dim = _select_trans_dim(
            channels=channels,
            head_dim=head_dim,
            trans_ratio=trans_ratio,
            max_trans_dim=max_trans_dim,
        )
        conv_dim = channels - trans_dim
        self.block = ConvTransBlock(
            conv_dim=conv_dim,
            trans_dim=trans_dim,
            head_dim=head_dim,
            window_size=window_size,
            drop_path=drop_path,
            type=block_type,
            input_resolution=padded_resolution,
        )

    def _forward_impl(self, x: torch.Tensor):
        _, _, z, y, xdim = x.shape
        pad_z = (_round_up_to_multiple(z, self.window_size) - z)
        pad_y = (_round_up_to_multiple(y, self.window_size) - y)
        pad_x = (_round_up_to_multiple(xdim, self.window_size) - xdim)
        if pad_z > 0 or pad_y > 0 or pad_x > 0:
            x = F.pad(x, (0, pad_x, 0, pad_y, 0, pad_z))
        x = self.block(x)
        return x[..., :z, :y, :xdim]

    def forward(self, x: torch.Tensor):
        if self.use_checkpoint and x.requires_grad:
            return torch_checkpoint(
                self._forward_impl,
                x,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        return self._forward_impl(x)


class EM3DSwinConvPyramidBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        input_resolution: int,
        activation_class=nn.ReLU,
        checkpoint: bool = True,
        cube_scunet_head_dim: int = 16,
        cube_scunet_window_size: int = 3,
        cube_scunet_drop_path: float = 0.0,
        cube_scunet_trans_ratio: float = 0.25,
        cube_scunet_max_trans_dim: int = 128,
    ):
        super().__init__()
        self.use_checkpoint = checkpoint
        self.act = activation_class()
        self.conv0 = nn.Conv3d(in_channels, out_channels * 3, kernel_size=1, bias=False)
        self.norm0 = nn.InstanceNorm3d(out_channels * 3, affine=True)
        self.conv1 = nn.Conv3d(
            out_channels * 3,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.norm1 = nn.InstanceNorm3d(out_channels, affine=True)

        output_resolution = _stride2_output_size(input_resolution)
        self.mixer_w = EM3DSwinConvMixer(
            channels=out_channels,
            input_resolution=output_resolution,
            head_dim=cube_scunet_head_dim,
            window_size=cube_scunet_window_size,
            drop_path=cube_scunet_drop_path,
            trans_ratio=cube_scunet_trans_ratio,
            max_trans_dim=cube_scunet_max_trans_dim,
            block_type="W",
            checkpoint=checkpoint,
        )
        self.mixer_sw = EM3DSwinConvMixer(
            channels=out_channels,
            input_resolution=output_resolution,
            head_dim=cube_scunet_head_dim,
            window_size=cube_scunet_window_size,
            drop_path=cube_scunet_drop_path,
            trans_ratio=cube_scunet_trans_ratio,
            max_trans_dim=cube_scunet_max_trans_dim,
            block_type="SW",
            checkpoint=checkpoint,
        )

    def _forward_impl(self, x: torch.Tensor):
        x = self.act(self.norm0(self.conv0(x)))
        x = self.act(self.norm1(self.conv1(x)))
        x = self.mixer_w(x)
        x = self.mixer_sw(x)
        return x

    def forward(self, x: torch.Tensor):
        if self.use_checkpoint and x.requires_grad:
            return torch_checkpoint(
                self._forward_impl,
                x,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        return self._forward_impl(x)


class EM3DMultiScaleSwinConvEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        cube_size: int,
        activation_class=nn.ReLU,
        checkpoint: bool = True,
        cube_scunet_head_dim: int = 16,
        cube_scunet_window_size: int = 3,
        cube_scunet_drop_path: float = 0.0,
        cube_scunet_trans_ratio: float = 0.25,
        cube_scunet_max_trans_dim: int = 128,
    ):
        super().__init__()
        self.use_checkpoint = checkpoint
        self.avg_pool = EM3DSpatialMean()
        self.conv0 = EM3DLocalConvStem(
            in_channels,
            hidden_channels * 4,
            activation_class,
            checkpoint=checkpoint,
        )
        self.conv1 = EM3DSwinConvPyramidBlock(
            hidden_channels * 4,
            hidden_channels,
            input_resolution=cube_size,
            activation_class=activation_class,
            checkpoint=checkpoint,
            cube_scunet_head_dim=cube_scunet_head_dim,
            cube_scunet_window_size=cube_scunet_window_size,
            cube_scunet_drop_path=cube_scunet_drop_path,
            cube_scunet_trans_ratio=cube_scunet_trans_ratio,
            cube_scunet_max_trans_dim=cube_scunet_max_trans_dim,
        )
        self.vision1 = nn.Conv3d(hidden_channels, out_channels, kernel_size=7, bias=False)

        self.conv2 = EM3DSwinConvPyramidBlock(
            hidden_channels,
            hidden_channels * 4,
            input_resolution=_stride2_output_size(cube_size),
            activation_class=activation_class,
            checkpoint=checkpoint,
            cube_scunet_head_dim=cube_scunet_head_dim,
            cube_scunet_window_size=cube_scunet_window_size,
            cube_scunet_drop_path=cube_scunet_drop_path,
            cube_scunet_trans_ratio=cube_scunet_trans_ratio,
            cube_scunet_max_trans_dim=cube_scunet_max_trans_dim,
        )
        self.vision2 = nn.Conv3d(hidden_channels * 4, out_channels, kernel_size=5, bias=False)

        self.conv3 = EM3DSwinConvPyramidBlock(
            hidden_channels * 4,
            hidden_channels * 8,
            input_resolution=_stride2_output_size(_stride2_output_size(cube_size)),
            activation_class=activation_class,
            checkpoint=checkpoint,
            cube_scunet_head_dim=cube_scunet_head_dim,
            cube_scunet_window_size=cube_scunet_window_size,
            cube_scunet_drop_path=cube_scunet_drop_path,
            cube_scunet_trans_ratio=cube_scunet_trans_ratio,
            cube_scunet_max_trans_dim=cube_scunet_max_trans_dim,
        )
        self.vision3 = nn.Conv3d(hidden_channels * 8, out_channels, kernel_size=3, bias=False)
        self.norm = nn.LayerNorm(out_channels)

    def _vision_forward(self, vision_module: nn.Module, x: torch.Tensor):
        y = vision_module(x)
        return self.avg_pool(y)

    def forward(self, x: torch.Tensor):
        x = self.conv0(x)
        x = self.conv1(x)
        if self.use_checkpoint and x.requires_grad:
            x1 = torch_checkpoint(
                lambda t: self._vision_forward(self.vision1, t),
                x,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            x1 = self._vision_forward(self.vision1, x)

        x = self.conv2(x)
        if self.use_checkpoint and x.requires_grad:
            x2 = torch_checkpoint(
                lambda t: self._vision_forward(self.vision2, t),
                x,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            x2 = self._vision_forward(self.vision2, x)

        x = self.conv3(x)
        if self.use_checkpoint and x.requires_grad:
            x3 = torch_checkpoint(
                lambda t: self._vision_forward(self.vision3, t),
                x,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            x3 = self._vision_forward(self.vision3, x)

        return self.norm(x1 + x2 + x3)


class CryoInit(nn.Module):
    """
    v3x CryoInit: keep the v3 multi-scale pyramid, but replace each cube stage
    with a lightweight SCUNet-style Swin-Conv mixer.
    """

    def __init__(
        self,
        d_node: int = 256,
        d_edge: int = 128,
        d_cryo_emb: int = 256,
        c_grid: int = 1,
        cube_size: int = 17,
        rectangle_length: int = 12,
        k: int = 32,
        activation_class: nn.Module = nn.ReLU,
        checkpoint: bool = True,
        cube_scunet_head_dim: int = 16,
        cube_scunet_window_size: int = 3,
        cube_scunet_drop_path: float = 0.0,
        cube_scunet_trans_ratio: float = 0.25,
        cube_scunet_max_trans_dim: int = 128,
        **kwargs,
    ):
        super().__init__()

        assert d_cryo_emb % 4 == 0

        self.k = k
        self.d_node = d_node
        self.d_edge = d_edge
        self.c_length = cube_size
        self.r_length = rectangle_length
        self.d_cryo_emb = d_cryo_emb
        self.c_grid = c_grid
        self.use_checkpoint = checkpoint

        self.conv_cube = EM3DMultiScaleSwinConvEncoder(
            in_channels=self.c_grid,
            hidden_channels=self.d_cryo_emb // 4,
            out_channels=self.d_node,
            cube_size=self.c_length,
            activation_class=activation_class,
            checkpoint=checkpoint,
            cube_scunet_head_dim=cube_scunet_head_dim,
            cube_scunet_window_size=cube_scunet_window_size,
            cube_scunet_drop_path=cube_scunet_drop_path,
            cube_scunet_trans_ratio=cube_scunet_trans_ratio,
            cube_scunet_max_trans_dim=cube_scunet_max_trans_dim,
        )

        self.conv_rectangle_pre = nn.Conv3d(
            in_channels=self.c_grid,
            out_channels=self.d_cryo_emb // 4,
            kernel_size=3,
            bias=False,
        )
        self.conv_rectangle_post = nn.Sequential(
            nn.LayerNorm(self.d_cryo_emb // 4 * (self.r_length - 2)),
            activation_class(),
            nn.Linear(self.d_cryo_emb // 4 * (self.r_length - 2), self.d_edge, bias=False),
        )

        self.backbone_distance_emb = BackboneDistanceEmbedding()

    def _run_with_checkpoint(self, module: nn.Module, x: torch.Tensor):
        if self.use_checkpoint and x.requires_grad:
            return torch_checkpoint(
                module,
                x,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        return module(x)

    def forward(
        self,
        affines,
        cryo_grids=None,
        cryo_global_origins=None,
        cryo_voxel_sizes=None,
        edge_index=None,
        full_edge_index=None,
        batch=None,
        **kwargs,
    ):
        assert cryo_grids is not None
        n = len(affines)
        k = min(n - 1, self.k)

        batch_to_idx = (
            get_batches_to_idx(batch)
            if batch is not None
            else [torch.arange(0, len(affines), dtype=int, device=affines.device)]
        )

        bde_out = self.backbone_distance_emb(
            affines,
            edge_index=edge_index,
            full_edge_index=full_edge_index,
            batch=batch,
            k=k,
        )

        with torch.no_grad():
            batch_cryo_grids = [
                cg.expand(len(b), -1, -1, -1, -1)
                for (cg, b) in zip(cryo_grids, batch_to_idx)
            ]
            cryo_points = [
                (bde_out.positions[b].reshape(-1, 3) - go) / vz
                for (b, go, vz) in zip(
                    batch_to_idx, cryo_global_origins, cryo_voxel_sizes
                )
            ]
            cryo_points_rot_matrices = [
                affines[b][..., :3, :3].reshape(-1, 3, 3) for b in batch_to_idx
            ]
            cryo_points_cube = sample_centered_cube_rot_matrix(
                batch_cryo_grids,
                cryo_points_rot_matrices,
                cryo_points,
                cube_side=self.c_length,
            )

        node_repr = self.conv_cube(cryo_points_cube.requires_grad_())

        with torch.no_grad():
            batch_cryo_grids = [
                cg.expand(len(b) * k, -1, -1, -1, -1)
                for (cg, b) in zip(cryo_grids, batch_to_idx)
            ]
            cryo_vectors = bde_out.neighbour_positions.detach()
            cryo_vectors = [cryo_vectors[b].reshape(-1, 3) for b in batch_to_idx]
            cryo_vectors_center_positions = [
                (
                    bde_out.positions[b]
                    .unsqueeze(1)
                    .expand(len(b), k, 3)
                    .reshape(-1, 3)
                    - go
                )
                / vz
                for (b, go, vz) in zip(
                    batch_to_idx, cryo_global_origins, cryo_voxel_sizes
                )
            ]
            cryo_vectors_rec = sample_centered_rectangle_along_vector(
                batch_cryo_grids,
                cryo_vectors,
                cryo_vectors_center_positions,
                rectangle_length=self.r_length,
            )

        edge_repr = self._run_with_checkpoint(
            self.conv_rectangle_pre,
            cryo_vectors_rec.requires_grad_(),
        )
        edge_repr = einops.rearrange(
            edge_repr,
            "(b kz) c z y x -> b kz (c z y x)",
            kz=k,
            c=self.d_cryo_emb // 4,
            z=(self.r_length - 2),
            y=1,
            x=1,
        )
        edge_repr = self._run_with_checkpoint(self.conv_rectangle_post, edge_repr)
        return node_repr, edge_repr, bde_out
