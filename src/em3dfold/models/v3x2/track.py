import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

import time
import einops

from em3dfold.models.v3x2.attention import NodeUpdate, EdgeUpdate, OutProductMean, Transition
from em3dfold.models.v3x2.ipa import InvariantPointAttention, IPATransition
from em3dfold.models.v3x2.backbone_update import BackboneUpdate

def rbf(d, d_count=64, d_min=0.5, d_max=30.5):
    device = d.device
    d_mu = torch.linspace(d_min, d_max, d_count, device=device)
    d_mu = d_mu.view([1, 1, -1])
    d_sigma = (d_max - d_min) / d_count
    d_expand = torch.unsqueeze(d, -1)
    RBF = torch.exp(-((d_expand - d_mu) / d_sigma)**2)
    return RBF

def _angle(a, b, c, eps=1e-8):
    v1 = a - b
    v2 = c - b
    v1 = F.normalize(v1, dim=-1, eps=eps)
    v2 = F.normalize(v2, dim=-1, eps=eps)
    cosang = (v1 * v2).sum(dim=-1).clamp(-1.0 + eps, 1.0 - eps)
    return torch.acos(cosang)

def _dihedral(a, b, c, d, eps=1e-8):
    b0 = a - b
    b1 = c - b
    b2 = d - c

    b1 = F.normalize(b1, dim=-1, eps=eps)
    v = b0 - (b0 * b1).sum(dim=-1, keepdim=True) * b1
    w = b2 - (b2 * b1).sum(dim=-1, keepdim=True) * b1
    v = F.normalize(v, dim=-1, eps=eps)
    w = F.normalize(w, dim=-1, eps=eps)

    x = (v * w).sum(dim=-1)
    y = (torch.cross(b1, v, dim=-1) * w).sum(dim=-1)
    return torch.atan2(y, x)

class TorsionNet(nn.Module):
    def __init__(
        self,
        d_node=256,
        d_head=128,
        p_drop=0.15,
        n_tors=10,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_node)
        self.linear_0 = nn.Linear(d_node, d_head)

        # ResNet layers
        self.linear_1 = nn.Linear(d_head, d_head)
        self.linear_2 = nn.Linear(d_head, d_head)
        self.linear_3 = nn.Linear(d_head, d_head)
        self.linear_4 = nn.Linear(d_head, d_head)

        # Final outputs
        self.n_tors = n_tors
        self.linear_out = nn.Linear(d_head, self.n_tors * 2)

    def forward(self, node):
        node = self.norm(node)
        node = self.linear_0(node)

        node = node + self.linear_2(F.relu_(self.linear_1(F.relu_(node))))
        node = node + self.linear_4(F.relu_(self.linear_3(F.relu_(node))))

        tors = self.linear_out(F.relu_(node))
        tors = einops.rearrange(tors, "... (d x) -> ... d x", d=self.n_tors, x=2)

        return tors


class TrackBlock(nn.Module):
    def __init__(
        self,
        d_node=256,
        d_edge=256,
        d_bias=None,
        d_head=48,
        n_head=8,
        n_qk_point=4,
        n_v_point=8,
        p_drop=0.10,
        k=32,
        enable_geometry_update=True,
    ):

        super().__init__()
        self.enable_geometry_update = enable_geometry_update

        # node update
        self.node_update = NodeUpdate(
            in_features=d_node,
            in_features_edge=d_edge,
            num_neighbors=k,
            num_heads=n_head,
            head_dim=d_head,
            p_drop=p_drop,
        )

        # node transition
        self.node_transition = Transition(
            in_features=d_node,
            norm=nn.LayerNorm,
            n=2,
            p_drop=p_drop,
        )

        # out product mean
        self.out_product = OutProductMean(
            in_features=d_node,
            in_features_edge=d_edge,
            c=d_head,
            p_drop=p_drop,
        )

        # edge update
        self.edge_update = EdgeUpdate(
            in_features=d_edge,
            num_neighbors=k,
            num_heads=n_head,
            head_dim=d_head,
            p_drop=p_drop,
        )

        # edge transition
        self.edge_transition = Transition(
            in_features=d_edge,
            norm=nn.LayerNorm,
            n=2,
            p_drop=p_drop,
        )

        # ipa
        self.ipa = InvariantPointAttention(
            in_features=d_node,
            in_features_edge=d_edge,
            attention_heads=n_head,
            c=d_head,
            query_points=n_qk_point,
            point_values=n_v_point,
        )

        self.ipa_transition = IPATransition(
            in_features=d_node,
            n=2,
        )

        if self.enable_geometry_update:
            self.bb_update = BackboneUpdate(d_node)

            # torsion update
            self.torsion_update = TorsionNet(
                d_node=d_node,
                d_head=d_head,
                n_tors=10 * 28,
            )

    def forward(
        self,
        node,
        edge,
        affines,
        pos_emb=None,
        edge_index=None,
        use_checkpoint=False,
    ):
        if use_checkpoint and torch.is_grad_enabled() and (
            node.requires_grad or edge.requires_grad or affines.requires_grad
        ):
            new_forward = lambda node, edge, affines: self.forward_normal(
                node,
                edge,
                affines,
                pos_emb=pos_emb,
                edge_index=edge_index,
            )
            return checkpoint.checkpoint(
                new_forward,
                node,
                edge,
                affines,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            return self.forward_normal(
                node,
                edge,
                affines,
                pos_emb,
                edge_index,
            )

    def forward_normal(
        self,
        node,
        edge,
        affines,
        pos_emb=None,
        edge_index=None,
    ):
        """
            node: (..., n, d)
            edge: (..., n, k, d)
            afines: (..., n, 3, 4)
        """

        if edge_index is None:
            raise ValueError("edge_index is required for TrackBlock")

        # node update
        node = self.node_update(
            node,
            edge,
            pos_emb,
            edge_index,
        )
        node = self.node_transition(node)

        # out product mean
        edge = self.out_product(
            node,
            edge,
            edge_index,
        )

        # edge update
        edge = self.edge_update(
            edge,
            edge_index,
        )
        edge = self.edge_transition(edge)

        # no grad for rotation
        affines = torch.cat(
            [
                affines[..., :3, :3].detach(), # no grad for rotation
                affines[..., :3, -1][..., None],
            ],
            dim=-1,
        )

        # ipa
        node = self.ipa(
            node,
            edge,
            affines,
            pos_emb,
            edge_index,
        )

        # transition
        node = self.ipa_transition(node)

        torsion_angles_sin_cos = None
        if self.enable_geometry_update:
            # bb update
            affines = self.bb_update(node, affines)

            # torsion update
            torsion_angles_sin_cos = self.torsion_update(node)

        return node, edge, affines, torsion_angles_sin_cos


import time
if __name__ == '__main__':
    model = TrackBlock().cuda()
    print(sum(p.numel() for p in model.parameters()))

    for i in range(10):
        t0 = time.time()
        node, edge, affines, torsion_angles_sin_cos = model(
            node=torch.randn(100, 256).cuda(),
            edge=torch.randn(100, 32, 256).cuda(),
            affines=torch.randn(100, 3, 4).cuda(),
            pos_emb=torch.randn(100, 32).cuda(),
            edge_index=torch.randint(0, 100, (100, 32)).cuda(),
            use_checkpoint=False,
        )
        t1 = time.time()

        print("{:.4f}".format(t1 - t0))

        time.sleep(2)
