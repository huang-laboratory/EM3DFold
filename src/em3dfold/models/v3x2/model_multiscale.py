import contextlib

import torch
import torch.nn as nn

from em3dfold.models.v3x2.model_base import Model as V2Model
from em3dfold.models.v3x2.model_base import Output
from em3dfold.models.v3x2.cryo_init_multiscale import CryoInit
from em3dfold.models.v3x2.sequence_attention import SequenceAttention


class FusedEdgeAAPredictor(nn.Module):
    def __init__(self, d_edge: int = 256, n_classes: int = 24):
        super().__init__()
        self.n_classes = n_classes
        self.linear1 = nn.Linear(d_edge, d_edge)
        self.linear2 = nn.Linear(d_edge, d_edge)
        self.head = nn.Sequential(
            nn.Linear(d_edge, d_edge),
            nn.ReLU(),
            nn.Linear(d_edge, d_edge),
            nn.ReLU(),
            nn.Linear(d_edge, n_classes * n_classes),
        )

    def forward(self, edge_density, edge):
        x = self.linear1(edge_density) + self.linear2(edge)
        x = self.head(x)
        return x.view(*x.shape[:-1], self.n_classes, self.n_classes)


class Model(V2Model):
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
        n_block=16,
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
        d_cryo_emb=None,
        cube_size=23,
        rectangle_length=15,
    ):
        super().__init__(
            d_node=d_node,
            d_edge=d_edge,
            d_head=d_head,
            d_seq=d_seq,
            d_seq_na=d_seq_na,
            n_qk_point=n_qk_point,
            n_v_point=n_v_point,
            n_head=n_head,
            n_block=n_block,
            k=k,
            p_drop=p_drop,
            c_grid=c_grid,
            pred_node_exist=pred_node_exist,
            pred_edge_exist=pred_edge_exist,
            pred_pairing=pred_pairing,
            use_checkpoint=use_checkpoint,
            use_seq_attn=use_seq_attn,
            seq_attn_every=seq_attn_every,
            geometry_update_every=geometry_update_every,
            recycle_use_full_structure=recycle_use_full_structure,
        )

        self.num_res_types = 24

        self.init = CryoInit(
            d_node=d_node,
            d_edge=d_edge,
            d_cryo_emb=d_node if d_cryo_emb is None else d_cryo_emb,
            c_grid=c_grid,
            cube_size=cube_size,
            rectangle_length=rectangle_length,
            k=k,
            checkpoint=use_checkpoint,
        )

        if self.use_seq_attn:
            self.seq_attn_blocks = nn.ModuleList(
                [
                    SequenceAttention(
                        d=d_node,
                        d_seq=d_seq,
                        d_seq_na=self.d_seq_na,
                        d_head=d_head,
                        n_head=n_head,
                        checkpoint=use_checkpoint,
                    )
                    for _ in range(self.n_block // self.seq_attn_every)
                ]
            )

        self.edge_aa_predictor = FusedEdgeAAPredictor(
            d_edge=d_edge,
            n_classes=self.num_res_types,
        )

        self.embed_cycle_aa = nn.Sequential(
            nn.LayerNorm(self.num_res_types),
            nn.Linear(self.num_res_types, d_node, bias=False),
        )
        self.embed_cycle_edge_aa = nn.Sequential(
            nn.LayerNorm(self.num_res_types * self.num_res_types),
            nn.Linear(self.num_res_types * self.num_res_types, d_edge, bias=False),
        )
        self.embed_cycle_conf = nn.Sequential(
            nn.LayerNorm(2),
            nn.Linear(2, d_node, bias=False),
        )
        self.embed_cycle_edge_conf = nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(4, d_edge, bias=False),
        )

        if self.has_intermediate_refresh:
            self.refresh_edge_head_embed = nn.Sequential(
                nn.LayerNorm(d_edge),
                nn.Linear(d_edge, d_edge, bias=False),
            )

    def _build_confidence_features(
        self,
        prev_rmsd: torch.Tensor,
        edge_index: torch.Tensor,
    ):
        prev_rmsd = torch.clamp(prev_rmsd, min=0.0)
        prev_conf = torch.exp(-prev_rmsd)

        node_conf_feat = torch.cat([prev_rmsd, prev_conf], dim=-1)

        neighbor_rmsd = prev_rmsd[edge_index]
        neighbor_conf = prev_conf[edge_index]
        src_rmsd = prev_rmsd[:, None, :].expand_as(neighbor_rmsd)
        src_conf = prev_conf[:, None, :].expand_as(neighbor_conf)
        edge_conf_feat = torch.cat(
            [
                0.5 * (src_rmsd + neighbor_rmsd),
                torch.abs(src_rmsd - neighbor_rmsd),
                0.5 * (src_conf + neighbor_conf),
                torch.abs(src_conf - neighbor_conf),
            ],
            dim=-1,
        )
        return node_conf_feat, edge_conf_feat

    def forward(
        self,
        affines: torch.Tensor,
        prot_mask: torch.Tensor,
        cryo_grids=None,
        cryo_global_origins=None,
        cryo_voxel_sizes=None,
        prot_seq_embed: torch.Tensor = None,
        prot_seq_embed_mask: torch.Tensor = None,
        na_seq_embed: torch.Tensor = None,
        na_seq_embed_mask: torch.Tensor = None,
        prev_aa_probs: torch.Tensor = None,
        prev_rmsd: torch.Tensor = None,
        prev_node: torch.Tensor = None,
        batch=None,
        run_iters=1,
        **kwargs,
    ):
        if batch is None:
            batch = torch.zeros(len(affines), device=affines.device, dtype=torch.long)

        if cryo_grids is None or cryo_global_origins is None or cryo_voxel_sizes is None:
            raise ValueError("Cryo grid inputs must be provided.")

        init_affines = affines
        init_node = torch.zeros((len(affines), self.d_node), device=affines.device)
        max_k = min(self.k, len(affines) - 1)
        init_edge = torch.zeros((len(affines), max_k, self.d_edge), device=affines.device)
        init_aa_logits = torch.zeros(
            (len(affines), self.num_res_types),
            device=affines.device,
        )
        init_edge_aa_logits = torch.zeros(
            (len(affines), max_k, self.num_res_types * self.num_res_types),
            device=affines.device,
        )
        init_rmsd = torch.zeros(
            (len(affines), 1),
            device=affines.device,
        )
        if prev_node is not None:
            if prev_node.shape != init_node.shape:
                raise ValueError(
                    f"prev_node must have shape {tuple(init_node.shape)}, "
                    f"got {tuple(prev_node.shape)}"
                )
            init_node = prev_node.to(
                device=affines.device,
                dtype=init_node.dtype,
            )
        if prev_aa_probs is not None:
            if prev_aa_probs.shape != init_aa_logits.shape:
                raise ValueError(
                    f"prev_aa_probs must have shape {tuple(init_aa_logits.shape)}, "
                    f"got {tuple(prev_aa_probs.shape)}"
                )
            init_aa_logits = prev_aa_probs.to(
                device=affines.device,
                dtype=init_aa_logits.dtype,
            )
        if prev_rmsd is not None:
            if prev_rmsd.shape == (len(affines),):
                prev_rmsd = prev_rmsd[..., None]
            if prev_rmsd.shape != init_rmsd.shape:
                raise ValueError(
                    f"prev_rmsd must have shape {tuple(init_rmsd.shape)}, "
                    f"got {tuple(prev_rmsd.shape)}"
                )
            init_rmsd = prev_rmsd.to(
                device=affines.device,
                dtype=init_rmsd.dtype,
            )

        fixed_edge_index = None
        fixed_full_edge_index = None

        pred_aatype = None
        pred_edge_aa_logits = None
        pred_node_existence = None
        pred_edge_existence = None
        pred_pairing = None

        for run_iter in range(run_iters):
            with torch.no_grad() if run_iter < run_iters - 1 else contextlib.nullcontext():
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
                edge_density_state = edge_density

                node_type, edge_type = self.embed_mol_type(prot_mask, edge_index=bde_out.edge_index)
                node = node + node_type
                edge = edge + edge_type

                node = node + self.embed_cycle_node(init_node)
                edge = edge + self.embed_cycle_edge(init_edge)
                node = node + self.embed_cycle_aa(init_aa_logits)
                edge = edge + self.embed_cycle_edge_aa(init_edge_aa_logits)
                node_conf_feat, edge_conf_feat = self._build_confidence_features(
                    init_rmsd,
                    bde_out.edge_index,
                )
                node = node + self.embed_cycle_conf(node_conf_feat)
                edge = edge + self.embed_cycle_edge_conf(edge_conf_feat)

                torsion_angles = None
                edge = edge + self._compute_edge_bias(
                    init_affines,
                    torsion_angles,
                    prot_mask,
                    bde_out.edge_index,
                )

                affines_list = []
                torsion_list = []
                affines = init_affines
                seq_attn_idx = 0
                for i in range(self.n_block):
                    do_seq_attn = (
                        self.use_seq_attn
                        and self.seq_attn_every > 0
                        and ((i + 1) % self.seq_attn_every == 0)
                    )
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
                            raise RuntimeError(
                                "Intermediate density refresh triggered without refresh modules."
                            )
                        node_density_state = node_density_state + self.refresh_node_head_embed(node_density)
                        edge_density_state = edge_density_state + self.refresh_edge_head_embed(edge_density)
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

                pred_aatype = self.aa_predictor(node_density_state, node)
                pred_edge_aa_logits = self.edge_aa_predictor(edge_density_state, edge)
                rmsd = self.rmsd_predictor(node)

                if self.pred_node_exist:
                    pred_node_existence = self.node_existence_predictor(node)

                if self.pred_edge_exist:
                    pred_edge_existence = self.edge_existence_predictor(edge)

                if self.pred_pairing:
                    pred_pairing = self.pairing_predictor(edge)

                init_affines = affines
                init_node = node
                init_edge = edge

                with torch.no_grad():
                    # Original implementation kept here for reference:
                    # init_aa_logits = torch.softmax(pred_aatype, dim=-1)
                    prot_probs = torch.softmax(pred_aatype[..., :20], dim=-1)
                    na_probs = torch.softmax(pred_aatype[..., 20:], dim=-1)
                    init_aa_logits = torch.zeros_like(pred_aatype)
                    prot_mask_bool = prot_mask.bool().unsqueeze(-1)
                    init_aa_logits[..., :20] = torch.where(
                        prot_mask_bool,
                        prot_probs,
                        torch.zeros_like(prot_probs),
                    )
                    init_aa_logits[..., 20:] = torch.where(
                        ~prot_mask_bool,
                        na_probs,
                        torch.zeros_like(na_probs),
                    )

                    # Original implementation kept here for reference:
                    # init_edge_aa_logits = torch.softmax(
                    #     pred_edge_aa_logits.reshape(
                    #         pred_edge_aa_logits.shape[0],
                    #         pred_edge_aa_logits.shape[1],
                    #         -1,
                    #     ),
                    #     dim=-1,
                    # )
                    init_edge_aa_logits = torch.softmax(
                        pred_edge_aa_logits.reshape(
                            pred_edge_aa_logits.shape[0],
                            pred_edge_aa_logits.shape[1],
                            -1,
                        ),
                        dim=-1,
                    )

                with torch.no_grad():
                    init_rmsd = rmsd

        return Output(
            prot_mask=prot_mask,
            pred_aatype=pred_aatype,
            pred_aa_logits=pred_aatype,
            pred_edge_aa_logits=pred_edge_aa_logits,
            pred_torsions=torsion_list,
            pred_affines=affines_list,
            pred_positions=[x[..., :3, -1] for x in affines_list],
            pred_rmsd=rmsd,
            recycle_node_state=node,
            pred_node_existence=pred_node_existence if self.pred_node_exist else None,
            pred_edge_existence=pred_edge_existence if self.pred_edge_exist else None,
            edge_index=bde_out.edge_index,
            full_edge_index=bde_out.full_edge_index,
            pred_pairing=pred_pairing if self.pred_pairing else None,
        )
