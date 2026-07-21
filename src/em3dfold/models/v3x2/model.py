import torch
import torch.nn as nn

from em3dfold.models.v3x2.track import rbf
from em3dfold.models.v3x2.model_scunet import Model as V3XModel
from em3dfold.polymer_utils import polymer
from em3dfold.polymer_utils import residue_constants as rc


class Model(V3XModel):
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
        n_block=12,
        k=32,
        p_drop=0.10,
        c_grid=1,
        pred_node_exist=False,
        pred_edge_exist=True,
        pred_pairing=False,
        pred_ss=True,
        use_checkpoint=False,
        use_seq_attn=True,
        seq_attn_every=2,
        geometry_update_every=4,
        recycle_use_full_structure=True,
        d_cryo_emb=None,
        cube_size=23,
        rectangle_length=15,
        cube_scunet_head_dim=16,
        cube_scunet_window_size=3,
        cube_scunet_drop_path=0.0,
        cube_scunet_trans_ratio=0.25,
        cube_scunet_max_trans_dim=128,
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
            d_cryo_emb=d_node if d_cryo_emb is None else d_cryo_emb,
            cube_size=cube_size,
            rectangle_length=rectangle_length,
            cube_scunet_head_dim=cube_scunet_head_dim,
            cube_scunet_window_size=cube_scunet_window_size,
            cube_scunet_drop_path=cube_scunet_drop_path,
            cube_scunet_trans_ratio=cube_scunet_trans_ratio,
            cube_scunet_max_trans_dim=cube_scunet_max_trans_dim,
        )

        self.pred_ss = pred_ss
        if self.pred_ss:
            self.ss_predictor = nn.Sequential(
                nn.Linear(d_node, d_node),
                nn.ReLU(),
                nn.Linear(d_node, d_node),
                nn.ReLU(),
                nn.Linear(d_node, 3),
            )

        proxy_prot_atom_names = rc.restype_name_to_atomc_names["ALA"]
        proxy_na_atom_names = rc.restype_name_to_atomc_names["DA"]
        self.proxy_atom_indices = torch.tensor(
            [
                [
                    proxy_prot_atom_names.index("N"),
                    proxy_prot_atom_names.index("CA"),
                    proxy_prot_atom_names.index("C"),
                    proxy_prot_atom_names.index("O"),
                    proxy_prot_atom_names.index("CB"),
                ],
                [
                    proxy_na_atom_names.index("P"),
                    proxy_na_atom_names.index("C4'"),
                    proxy_na_atom_names.index("O3'"),
                    proxy_na_atom_names.index("C1'"),
                    proxy_na_atom_names.index("N9"),
                ],
            ],
            dtype=torch.long,
            device=self.proxy_atom_indices.device,
        )
        self.num_proxy_atoms = int(self.proxy_atom_indices.shape[-1])
        self.edge_bias_embed = nn.Sequential(
            nn.Linear(self.num_proxy_atoms * self.num_proxy_atoms * self.edge_bias_rbf_bins, d_edge, bias=False),
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
                atom_pair_dist = anchor_dist[..., None, None].expand(
                    -1,
                    -1,
                    self.num_proxy_atoms,
                    self.num_proxy_atoms,
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

    def forward(self, *args, **kwargs):
        output = super().forward(*args, **kwargs)
        if self.pred_ss:
            output["pred_ss_logits"] = self.ss_predictor(output["recycle_node_state"])
        else:
            output["pred_ss_logits"] = None
        return output
