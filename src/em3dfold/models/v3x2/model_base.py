import sys

import torch
import torch.nn as nn
from typing import List
import contextlib

from em3dfold.models.v3x2.cryo_init_base import CryoInit
from em3dfold.models.v3x2.track import TrackBlock, rbf
from em3dfold.models.v3x2.sequence_attention_base import SequenceAttention
from em3dfold.models.mol import MolTypeEmbedder
from em3dfold.polymer_utils import polymer
from em3dfold.polymer_utils import residue_constants as rc

class Output:
    def __init__(self, **kwargs):
        self.data = dict()
        for k, v in kwargs.items():
            self.data[k] = v

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __contains__(self, key):
        return key in self.data

    def to(self, device_or_dtype):
        for k, v in self.data.items():
            if torch.is_tensor(v):
                self.data[k] = v.to(device_or_dtype)
            elif isinstance(v, list):
                self.data[k] = [x.to(device_or_dtype) for x in v]
        return self

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    def update(self, **kwargs):
        self.data.update(kwargs)

    def __repr__(self):
        return f"Output({self.data})"


class FusedAAPredictor(nn.Module):
    def __init__(
        self,
        d_node: int = 256,
        n_classes: int = 4,
    ):
        super().__init__()
        self.linear1 = nn.Linear(d_node, d_node)
        self.linear2 = nn.Linear(d_node, d_node)

        self.aa_head = nn.Sequential(
            nn.Linear(d_node, d_node),
            nn.ReLU(),
            nn.Linear(d_node, d_node),
            nn.ReLU(),
            nn.Linear(d_node, n_classes),
        )

    def forward(self, node_density, node):
        x = self.linear1(node_density) + self.linear2(node)
        return self.aa_head(x)


class Model(nn.Module):
    def __init__(
        self,
        d_node=256,
        d_edge=256,
        d_head=48,
        d_seq=1280,
        d_seq_na=1280,
        n_qk_point=4,
        n_v_point=8,
        n_head=8,
        n_block=8,
        k=32,
        p_drop=0.10,
        c_grid=1,
        pred_node_exist=False,
        pred_edge_exist=True,
        pred_pairing=False,
        use_checkpoint=False,
        use_seq_attn=True,
        seq_attn_every=2,
        geometry_update_every=4,
        recycle_use_full_structure=True,
    ):
        super().__init__()
        if n_block % 4 != 0:
            raise ValueError(f"n_block must be a multiple of 4, got {n_block}")
        if use_seq_attn and seq_attn_every <= 0:
            raise ValueError(f"seq_attn_every must be positive, got {seq_attn_every}")
        if geometry_update_every <= 0:
            raise ValueError(f"geometry_update_every must be positive, got {geometry_update_every}")
        if n_block % geometry_update_every != 0:
            raise ValueError(
                f"n_block ({n_block}) must be divisible by geometry_update_every ({geometry_update_every})"
            )

        self.use_checkpoint = use_checkpoint
        self.d_node = d_node
        self.d_edge = d_edge
        self.k = k
        self.d_seq = d_seq
        self.d_seq_na = d_seq if d_seq_na is None else d_seq_na
        self.use_seq_attn = use_seq_attn
        self.seq_attn_every = seq_attn_every
        self.geometry_update_every = geometry_update_every
        self.recycle_use_full_structure = recycle_use_full_structure
        self.has_intermediate_refresh = any(
            (((i + 1) % self.geometry_update_every == 0) or (i == n_block - 1))
            and (i != n_block - 1)
            for i in range(n_block)
        )

        # Feature init
        self.init = CryoInit(
            d_node=d_node,
            d_edge=d_edge,
            d_cryo_emb=d_node,
            c_grid=c_grid,
            k=k,
        )

        # Embed mol type
        self.embed_mol_type = MolTypeEmbedder(
            d_node=d_node,
            d_edge=d_edge,
            num_classes=2,
        )

        # Track blocks
        self.n_block = n_block
        self.blocks = nn.ModuleList()
        for i in range(self.n_block):
            is_geometry_block = ((i + 1) % self.geometry_update_every == 0) or (i == self.n_block - 1)
            self.blocks.append(TrackBlock(
                d_node=d_node,
                d_edge=d_edge,
                d_head=d_head,
                n_head=n_head,
                n_qk_point=n_qk_point,
                n_v_point=n_v_point,
                p_drop=p_drop,
                k=k,
                enable_geometry_update=is_geometry_block,
            ))

        # Sequence attention
        if self.use_seq_attn:
            self.seq_attn_blocks = nn.ModuleList([
                SequenceAttention(
                    d=d_node,
                    d_seq=d_seq,
                    d_seq_na=self.d_seq_na,
                    d_head=d_head,
                    n_head=n_head,
                    checkpoint=use_checkpoint,
                )
                for _ in range(self.n_block // self.seq_attn_every)
            ])

        if self.has_intermediate_refresh:
            self.refresh_node_embed = nn.Sequential(
                nn.LayerNorm(d_node),
                nn.Linear(d_node, d_node, bias=False),
            )
            self.refresh_node_head_embed = nn.Sequential(
                nn.LayerNorm(d_node),
                nn.Linear(d_node, d_node, bias=False),
            )
            self.refresh_edge_embed = nn.Sequential(
                nn.LayerNorm(d_edge),
                nn.Linear(d_edge, d_edge, bias=False),
            )
        self.edge_bias_rbf_bins = 32
        proxy_prot_atom_names = rc.restype_name_to_atomc_names["ALA"]
        proxy_na_atom_names = rc.restype_name_to_atomc_names["DA"]
        self.register_buffer(
            "proxy_atom_indices",
            torch.tensor(
                [
                    [
                        proxy_prot_atom_names.index("CA"),
                        proxy_prot_atom_names.index("CB"),
                    ],
                    [
                        proxy_na_atom_names.index("C4'"),
                        proxy_na_atom_names.index("P"),
                    ],
                ],
                dtype=torch.long,
            ),
            persistent=False,
        )
        self.edge_bias_embed = nn.Sequential(
            nn.Linear(4 * self.edge_bias_rbf_bins, d_edge, bias=False),
        )

        # Fused AA predictor
        self.aa_predictor = FusedAAPredictor(
            d_node=d_node,
            n_classes=20 + 4,
        )

        # pLDDT
        self.rmsd_predictor = nn.Sequential(
            nn.Linear(d_node, d_node),
            nn.ReLU(),
            nn.Linear(d_node, d_node),
            nn.ReLU(),
            nn.Linear(d_node, 1),
        )

        # Node existence pred
        self.pred_node_exist = pred_node_exist
        if self.pred_node_exist:
            self.node_existence_predictor = nn.Sequential(
                nn.Linear(d_node, d_node),
                nn.ReLU(),
                nn.Linear(d_node, d_node),
                nn.ReLU(),
                nn.Linear(d_node, 1),
            )

        # Edge existence pred
        self.pred_edge_exist = pred_edge_exist
        if self.pred_edge_exist:
            self.edge_existence_predictor = nn.Sequential(
                nn.Linear(d_edge, d_edge),
                nn.ReLU(),
                nn.Linear(d_edge, d_edge),
                nn.ReLU(),
                nn.Linear(d_edge, 3),
            )

        # NA pairing pred
        self.pred_pairing = pred_pairing
        if self.pred_pairing:
            self.pairing_predictor = nn.Sequential(
                nn.Linear(d_edge, d_edge),
                nn.ReLU(),
                nn.Linear(d_edge, d_edge),
                nn.ReLU(),
                nn.Linear(d_edge, 1),
            )

        # Previous cycle node embedder
        self.embed_cycle_node = nn.Sequential(
            nn.LayerNorm(d_node),
            nn.Linear(d_node, d_node, bias=False),
        )

        # Previous cycle edge embedder
        self.embed_cycle_edge = nn.Sequential(
            nn.LayerNorm(d_edge),
            nn.Linear(d_edge, d_edge, bias=False),
        )

    def _compute_edge_bias(self, affines, torsion_angles, prot_mask, edge_index):
        with torch.no_grad():
            device = affines.device
            anchor_pos = affines[..., :3, 3]

            if not self.recycle_use_full_structure:
                torsion_angles = None

            if torsion_angles is None:
                anchor_dist = torch.norm(
                    anchor_pos[:, None, :] - anchor_pos[edge_index],
                    dim=-1,
                )
                atom_pair_dist = torch.zeros(
                    anchor_dist.shape[0],
                    anchor_dist.shape[1],
                    4,
                    device=device,
                    dtype=affines.dtype,
                )
                atom_pair_dist[..., 0] = anchor_dist
                atom_pair_rbf = rbf(atom_pair_dist, d_count=self.edge_bias_rbf_bins)
                atom_pair_rbf = atom_pair_rbf.reshape(
                    atom_pair_rbf.shape[0],
                    atom_pair_rbf.shape[1],
                    -1,
                )
            else:
                proxy_aatype = torch.where(
                    prot_mask.bool(),
                    torch.zeros_like(prot_mask, dtype=torch.long),
                    torch.full_like(prot_mask, 20, dtype=torch.long),
                )
                proxy_torsion_angles = rc.select_torsion_angles(
                    torsion_angles,
                    proxy_aatype,
                    normalize=True,
                )
                proxy_frames = polymer.torsion_angles_to_frames(
                    proxy_aatype.detach().cpu().numpy(),
                    affines,
                    proxy_torsion_angles,
                )
                proxy_atomc_positions = polymer.frames_and_literature_positions_to_atomc_pos(
                    proxy_aatype.detach().cpu().numpy(),
                    proxy_frames,
                )

                atom_index_selector = self.proxy_atom_indices[prot_mask.long()].to(device)
                gather_index = atom_index_selector[..., None].expand(-1, -1, 3)
                anchor_atoms = torch.gather(proxy_atomc_positions, dim=1, index=gather_index)
                neighbor_atoms = anchor_atoms[edge_index]

                atom_pair_dist = torch.norm(
                    anchor_atoms[:, None, :, None, :] - neighbor_atoms[:, :, None, :, :],
                    dim=-1,
                )
                atom_pair_rbf = rbf(
                    atom_pair_dist.reshape(atom_pair_dist.shape[0], atom_pair_dist.shape[1], -1),
                    d_count=self.edge_bias_rbf_bins,
                )
                atom_pair_rbf = atom_pair_rbf.reshape(
                    atom_pair_rbf.shape[0],
                    atom_pair_rbf.shape[1],
                    -1,
                )

        return self.edge_bias_embed(atom_pair_rbf)

    def _run_cryo_init(
        self,
        affines,
        cryo_grids,
        cryo_global_origins,
        cryo_voxel_sizes,
        edge_index,
        full_edge_index,
        batch,
    ):
        node_density, edge_density, bde_out = self.init(
            affines=affines,
            cryo_grids=cryo_grids,
            cryo_global_origins=cryo_global_origins,
            cryo_voxel_sizes=cryo_voxel_sizes,
            edge_index=edge_index,
            full_edge_index=full_edge_index,
            batch=batch,
        )
        return node_density, edge_density, bde_out

    def forward(
        self,
        affines: torch.Tensor, # (n, 3, 4) rot and trans
        prot_mask: torch.Tensor, # (n, )
        cryo_grids: List[torch.Tensor] = None,
        cryo_global_origins: List[torch.Tensor] = None,
        cryo_voxel_sizes: List[torch.Tensor] = None,
        prot_seq_embed: torch.Tensor = None,
        prot_seq_embed_mask: torch.Tensor = None,
        na_seq_embed: torch.Tensor = None,
        na_seq_embed_mask: torch.Tensor = None,
        batch=None,
        run_iters=1,
        **kwargs,
    ):

        if batch is None:
            batch = torch.zeros( len(affines) ).to(affines.device).long()

        if cryo_grids is None or cryo_global_origins is None or cryo_voxel_sizes is None:
            raise ValueError("Cryo grid inputs must be provided.")

        init_affines = affines
        init_node = torch.zeros( (len(affines), self.d_node) ).to(affines.device)
        max_k = min(self.k, len(affines) - 1)
        init_edge = torch.zeros( (len(affines), max_k, self.d_edge) ).to(affines.device)

        # Edge index should be fixed during network run (for edge feat consistency)
        fixed_edge_index = None
        fixed_full_edge_index = None

        for run_iter in range(run_iters):
            with torch.no_grad() if run_iter < run_iters - 1 else contextlib.nullcontext():
                # Initial features from map
                # Edge index will be initialized the first time running cryo init
                node_density, edge_density, bde_out = self._run_cryo_init(
                    affines=init_affines,
                    cryo_grids=cryo_grids,
                    cryo_global_origins=cryo_global_origins,
                    cryo_voxel_sizes=cryo_voxel_sizes,
                    edge_index=fixed_edge_index,
                    full_edge_index=fixed_full_edge_index,
                    batch=batch,
                )
                if fixed_edge_index is None:
                    fixed_edge_index = bde_out.edge_index
                    fixed_full_edge_index = bde_out.full_edge_index

                node = node_density
                edge = edge_density
                node_density_state = node_density

                # Add mol type embedding
                node_type, edge_type = self.embed_mol_type(prot_mask, edge_index=bde_out.edge_index)
                node = node + node_type
                edge = edge + edge_type

                # Add previous cycle node and edge embedding
                node_prev = self.embed_cycle_node(init_node)
                edge_prev = self.embed_cycle_edge(init_edge)
                node = node + node_prev
                edge = edge + edge_prev

                torsion_angles = None
                # Initial geometry bias is injected once per refresh stage.
                edge = edge + self._compute_edge_bias(
                    init_affines,
                    torsion_angles,
                    prot_mask,
                    bde_out.edge_index,
                )

                # Staged Track blocks
                affines_list = []
                torsion_list = []
                affines = init_affines
                seq_attn_idx = 0
                for i in range(self.n_block):
                    do_seq_attn = self.use_seq_attn and self.seq_attn_every > 0 and ((i + 1) % self.seq_attn_every == 0)
                    do_geometry_update = self.blocks[i].enable_geometry_update

                    if do_seq_attn:
                        node = self.seq_attn_blocks[seq_attn_idx](
                            node,
                            prot_mask=prot_mask.bool(),
                            batch=batch,
                            prot_seq_emb=prot_seq_embed,
                            prot_seq_mask=prot_seq_embed_mask,
                            na_seq_emb=na_seq_embed,
                            na_seq_mask=na_seq_embed_mask,
                        )
                        seq_attn_idx += 1

                    node, edge, affines, torsion_angles = self.blocks[i](
                        node,
                        edge,
                        affines,
                        pos_emb=bde_out.pos3d_emb,
                        edge_index=bde_out.edge_index,
                        use_checkpoint=self.use_checkpoint,
                    )

                    # Refresh density features after geometry updates, except
                    # after the final geometry prediction where no later layers
                    # remain to consume new map features.
                    if do_geometry_update and i != self.n_block - 1:
                        node_density, edge_density, _ = self._run_cryo_init(
                            affines=affines,
                            cryo_grids=cryo_grids,
                            cryo_global_origins=cryo_global_origins,
                            cryo_voxel_sizes=cryo_voxel_sizes,
                            edge_index=fixed_edge_index,
                            full_edge_index=fixed_full_edge_index,
                            batch=batch,
                        )
                        if not self.has_intermediate_refresh:
                            raise RuntimeError("Intermediate density refresh triggered without refresh modules.")
                        node_density_state = node_density_state + self.refresh_node_head_embed(node_density)
                        node = node + self.refresh_node_embed(node_density)
                        edge = edge + self.refresh_edge_embed(edge_density)
                        edge = edge + self._compute_edge_bias(
                            affines,
                            torsion_angles,
                            prot_mask,
                            bde_out.edge_index,
                        )

                    if do_geometry_update:
                        affines_list.append(affines)
                        torsion_list.append(torsion_angles)

                # AA prediction
                pred_aatype = self.aa_predictor(node_density_state, node)

                # pLDDT prediction
                rmsd = self.rmsd_predictor(node)

                # Node existence prediction
                if self.pred_node_exist:
                    pred_node_existence = self.node_existence_predictor(node)

                # Edge existence prediction
                if self.pred_edge_exist:
                    pred_edge_existence = self.edge_existence_predictor(
                        edge,
                    )

                # For NAs, pairing prediction
                if self.pred_pairing:
                    pred_pairing = self.pairing_predictor(
                        edge,
                    )

                # For next iteration
                init_affines = affines
                init_node = node
                init_edge = edge


        return Output(
            prot_mask=prot_mask,
            pred_aatype=pred_aatype,
            pred_torsions=torsion_list,
            pred_affines=affines_list,
            pred_positions=[affines[..., :3, -1] for affines in affines_list],
            pred_rmsd=rmsd,
            pred_node_existence=pred_node_existence if self.pred_node_exist else None,
            pred_edge_existence=pred_edge_existence if self.pred_edge_exist else None,
            edge_index=bde_out.edge_index,
            full_edge_index=bde_out.full_edge_index,
            pred_pairing=pred_pairing if self.pred_pairing else None,
        )

import time
if __name__ == '__main__':
    model = Model(n_block=16, use_checkpoint=True).cuda()
    print(sum(p.numel() for p in model.parameters()))

    n = 200
    for i in range(10):
        t0 = time.time()
        batch = torch.zeros(n).long().cuda()

        out = model(
            affines=torch.randn(n, 3, 4).cuda(),
            prot_mask=torch.randint(0, 2, (n, )).cuda(),
            cryo_grids=[torch.zeros(1, 1, 120, 120, 120).cuda()],
            cryo_global_origins=[torch.zeros(3).cuda()],
            cryo_voxel_sizes=[torch.ones(3).cuda()],
            prot_seq_embed=torch.randn(1, 2000, 1280).cuda(),
            prot_seq_embed_mask=torch.ones(1, 2000).bool().cuda(),
            na_seq_embed=torch.randn(1, 1200, 1280).cuda(),
            na_seq_embed_mask=torch.ones(1, 1200).bool().cuda(),
            batch=batch,
            run_iters=2,
        )
        t1 = time.time()


        out["pred_affines"][-1].mean().backward()

        for k, v in out.items():
            if hasattr(v, "shape"):
                print(k)
                print(v.shape)

        print("{:.4f}".format(t1 - t0))

        time.sleep(1)

        #exit()
