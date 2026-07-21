import math

import torch
from einops.layers.torch import Rearrange
from torch import nn

from em3dfold.utils.affine_utils import affine_mul_vecs, invert_affine
from em3dfold.models.modules import qk_modulation

class InvariantPointAttention(nn.Module):
    def __init__(
        self,
        in_features: int,
        in_features_edge: int,
        attention_heads: int = 12,
        c: int = 48,
        query_points: int = 4,
        point_values: int = 8,
    ):
        super().__init__()
        self.in_features = in_features
        self.in_features_edge = in_features_edge
        self.num_heads = attention_heads
        self.head_dim = c
        self.num_query_points = query_points
        self.num_value_points = point_values
        self.attention_scale = math.sqrt(self.head_dim)

        # Learnable weighting for point-distance attention term.
        self.gamma = nn.Parameter(math.sqrt(2) * torch.ones((1, 1, attention_heads)))

        # Scalar attention projections.
        self.query_scalar_proj = nn.Sequential(
            nn.Linear(self.in_features, self.num_heads * self.head_dim, bias=False),
            Rearrange("N (h d) -> N h d", h=self.num_heads, d=self.head_dim),
        )
        self.key_scalar_proj = nn.Sequential(
            nn.Linear(self.in_features, self.num_heads * self.head_dim, bias=False),
            Rearrange("N (h d) -> N h d", h=self.num_heads, d=self.head_dim),
        )
        self.value_scalar_proj = nn.Sequential(
            nn.Linear(self.in_features, self.num_heads * self.head_dim, bias=False),
            Rearrange("N (h d) -> N h d", h=self.num_heads, d=self.head_dim),
        )

        # Point attention projections.
        self.query_point_proj = nn.Sequential(
            nn.Linear(self.in_features, self.num_heads * self.num_query_points * 3, bias=False),
            Rearrange(
                "N (k h q d) -> N k h q d",
                k=1,
                h=self.num_heads,
                q=self.num_query_points,
                d=3,
            ),
        )
        self.key_point_proj = nn.Sequential(
            nn.Linear(self.in_features, self.num_heads * self.num_query_points * 3, bias=False),
            Rearrange("N (h q d) -> N h q d", h=self.num_heads, q=self.num_query_points, d=3),
        )
        self.value_point_proj = nn.Sequential(
            nn.Linear(self.in_features, self.num_heads * self.num_value_points * 3, bias=False),
            Rearrange("N (h v d) -> N h v d", h=self.num_heads, v=self.num_value_points, d=3),
        )

        self.attn_bias_proj = nn.Linear(in_features_edge, attention_heads, bias=False)
        self.output_proj = nn.Linear(
            self.num_heads * self.head_dim
            + self.num_heads * (self.in_features_edge)
            + self.num_heads * self.num_value_points * 4,
            self.in_features,
            bias=False,
        )
        self.residual = nn.Identity()
        self.norm = nn.LayerNorm(in_features)

    def forward(self, node, edge, affines, pos_emb, edge_index):
        # Scalar Q/K/V attention terms.
        query_scalar = self.query_scalar_proj(node)  # N h d
        key_scalar = self.key_scalar_proj(node)[edge_index]  # N k h d
        query_scalar, key_scalar = qk_modulation(query_scalar, key_scalar, pos_emb, edge_index)
        value_scalar = self.value_scalar_proj(node)[edge_index]  # N k h d

        # Point Q/K/V attention terms in local frames.
        query_point = self.query_point_proj(node)  # N 1 h q 3
        key_point = self.key_point_proj(node)[edge_index]  # N k h q 3
        value_point = self.value_point_proj(node)  # N h v 3

        bias = self.attn_bias_proj(edge)
        point_scale = math.sqrt(2 / (9 * self.num_query_points))
        total_scale = math.sqrt(1 / 3)

        # Scalar attention score.
        scalar_score = (
            torch.einsum("nhd,nkhd->nkh", query_scalar, key_scalar) / self.attention_scale
        ) + bias

        # Point-distance attention score in global coordinates.
        query_point_global = affine_mul_vecs(affines, query_point)
        key_point_global = affine_mul_vecs(affines, key_point)
        point_score = -torch.sum(
            torch.square(query_point_global - key_point_global),
            dim=-1,
        ).sum(dim=-1)  # N k h

        # Combined attention.
        attention_scores = total_scale * (scalar_score + point_scale * self.gamma * point_score)
        attention_weights = torch.softmax(attention_scores, dim=1)

        # Aggregate scalar value, pair value, and point value features.
        output_scalar = torch.einsum("nkh,nkhd->nhd", attention_weights, value_scalar)
        output_pair = torch.einsum("nkh,nki->nhi", attention_weights, edge)

        value_point_global = affine_mul_vecs(affines, value_point)[edge_index]
        output_point = torch.einsum("nkh,nkhvd->nhvd", attention_weights, value_point_global)
        output_point_local = affine_mul_vecs(invert_affine(affines), output_point)
        output_point_norm = torch.norm(output_point_local, dim=-1, p=2)

        # Merge all IPA outputs, then residual + layernorm.
        out = self.output_proj(
            torch.cat(
                (
                    output_scalar.flatten(1),
                    output_pair.flatten(1),
                    output_point_local.flatten(1),
                    output_point_norm.flatten(1),
                ),
                dim=1,
            )
        )
        out = self.norm(math.sqrt(2) * self.residual(node) + out)
        return out


class IPATransition(nn.Module):
    def __init__(self, in_features: int, n: int = 2):
        super().__init__()
        self.transition = nn.Sequential(
            nn.Linear(in_features, in_features * n),
            nn.ReLU(),
            nn.Linear(in_features * n, in_features * n),
            nn.ReLU(),
            nn.Linear(in_features * n, in_features, bias=False),
        )
        self.residual = nn.Identity()
        self.norm = nn.LayerNorm(in_features)

    def forward(self, x):
        # Feed-forward transition with residual connection.
        y = self.transition(x)
        y = self.norm(y + math.sqrt(2) * self.residual(x))
        return y

