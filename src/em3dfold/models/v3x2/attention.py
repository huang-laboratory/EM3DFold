# Change the original node/edge attention in the TTA module to more-efficient kNN-based node/edge attention
# Change the original node_rope into qk_modulation
# More cleaner version

import math
import einops
import torch
from torch import nn
from einops.layers.torch import Rearrange

from em3dfold.models.modules import qk_modulation

class NodeUpdate(nn.Module):
    def __init__(
        self,
        in_features: int,
        in_features_edge: int,
        num_neighbors: int,
        num_heads: int,
        head_dim: int,
        p_drop: float = 0.0,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.in_features = in_features
        self.num_neighbors = num_neighbors
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.attention_scale = math.sqrt(self.head_dim)
        self.residual = nn.Identity()
        self.attn_dropout = nn.Dropout(p_drop)
        self.output_dropout = nn.Dropout(p_drop)

        # Linear projections for attention.
        self.query_proj = nn.Sequential(
            nn.Linear(self.in_features, self.num_heads * self.head_dim, bias=False),
            Rearrange("n (h d) -> n h d", h=self.num_heads, d=self.head_dim),
        )
        self.key_proj = nn.Sequential(
            nn.Linear(self.in_features, self.num_heads * self.head_dim, bias=False),
            Rearrange("n k (h d) -> n k h d", h=self.num_heads, d=self.head_dim),
        )
        self.value_proj = nn.Sequential(
            nn.Linear(self.in_features, self.num_heads * self.head_dim, bias=False),
            Rearrange("n k (h d) -> n k h d", h=self.num_heads, d=self.head_dim),
        )

        # Additive bias and gating.
        self.attn_bias = nn.Sequential(
            nn.LayerNorm(in_features_edge),
            nn.Linear(in_features_edge, num_heads, bias=False),
        )
        self.gate = nn.Sequential(
            nn.Linear(self.in_features, self.num_heads * self.head_dim, bias=True),
            Rearrange("n (h d) -> n h d", h=self.num_heads, d=self.head_dim),
            nn.Sigmoid(),
        )

        # Output projection.
        self.output_proj = nn.Sequential(
            Rearrange("n h d -> n (h d)", h=self.num_heads, d=self.head_dim),
            nn.Linear(self.num_heads * self.head_dim, self.in_features),
        )

    def forward(self, node, edge, pos_emb, edge_index):
        # Gather neighbor features for node-wise attention.
        neighbor_feat = node[edge_index]  # N k in_features
        query = self.query_proj(node)  # N h d
        key = self.key_proj(neighbor_feat)  # N k h d
        query, key = qk_modulation(query, key, pos_emb, edge_index)
        value = self.value_proj(neighbor_feat)  # N k h d

        bias = self.attn_bias(edge)
        gate = self.gate(node)

        # Scaled dot-product attention over neighbors.
        scores = torch.einsum("n h d, n k h d -> n k h", query, key) / self.attention_scale + bias
        weights = torch.softmax(scores, dim=1)
        weights = self.attn_dropout(weights)
        out = gate * torch.einsum("n k h, n k h d -> n h d", weights, value)

        # Residual + norm.
        out = self.output_dropout(self.output_proj(out))
        out = self.norm(out + math.sqrt(2) * self.residual(node))
        return out


class Transition(nn.Module):
    def __init__(self, in_features: int, norm: nn.Module, n: int = 4, p_drop: float = 0.0):
        super().__init__()
        self.norm = norm(in_features)
        self.w1 = nn.Linear(in_features, in_features * n, bias=False)
        self.w2 = nn.Linear(in_features, in_features * n, bias=False)
        self.w3 = nn.Linear(in_features * n, in_features, bias=False)
        self.residual = nn.Identity()
        self.dropout = nn.Dropout(p_drop)

    def forward(self, x):
        # Gated MLP block.
        y = nn.functional.silu(self.w1(x)) * self.w2(x)
        y = self.dropout(y)
        y = self.w3(y)
        y = self.norm(y + math.sqrt(2) * self.residual(x))
        return y


class OutProductMean(nn.Module):
    def __init__(self, in_features: int, in_features_edge: int, c: int = 32, p_drop: float = 0.0):
        super().__init__()
        self.proj = nn.Linear(in_features, c * 2)
        self.channel = c
        self.out_proj = nn.Linear(c**2, in_features_edge)
        self.norm = nn.LayerNorm(in_features_edge)
        self.dropout = nn.Dropout(p_drop)

    def forward(self, node, edge, edge_index):
        # Outer-product over channels for pair features.
        left, right = self.proj(node).chunk(2, dim=-1)
        right_neighbors = right[edge_index]
        out = torch.einsum("n i, n k j -> n k i j", left, right_neighbors)
        out = einops.rearrange(out, "n k i j -> n k (i j)", i=self.channel, j=self.channel)
        out = self.out_proj(out)
        out = self.dropout(out)

        out = self.norm(out + math.sqrt(2) * edge)
        return out


class EdgeUpdate(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_neighbors: int = 32,
        num_heads: int = 4,
        head_dim: int = 32,
        p_drop: float = 0.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_neighbors = num_neighbors
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.norm = nn.LayerNorm(in_features)
        self.attention_scale = math.sqrt(self.head_dim)
        self.residual = nn.Identity()
        self.attn_dropout = nn.Dropout(p_drop)
        self.output_dropout = nn.Dropout(p_drop)

        # Linear projections for attention.
        self.query_proj = nn.Sequential(
            nn.Linear(self.in_features, self.num_heads * self.head_dim, bias=False),
            Rearrange("n k (h d) -> n k h d", h=self.num_heads, d=self.head_dim),
        )
        self.key_proj = nn.Sequential(
            nn.Linear(self.in_features, self.num_heads * self.head_dim, bias=False),
            Rearrange("n k (h d) -> n k h d", h=self.num_heads, d=self.head_dim),
        )
        self.value_proj = nn.Sequential(
            nn.Linear(self.in_features, self.num_heads * self.head_dim, bias=False),
            Rearrange("n k (h d) -> n k h d", h=self.num_heads, d=self.head_dim),
        )

        # Additive bias and gating.
        self.attn_bias = nn.Linear(in_features, num_heads, bias=False)
        self.gate = nn.Sequential(
            nn.Linear(self.in_features, self.num_heads * self.head_dim, bias=True),
            Rearrange("n k (h d) -> n k h d", h=self.num_heads, d=self.head_dim),
            nn.Sigmoid(),
        )

        # Output projection.
        self.output_proj = nn.Sequential(
            Rearrange("n k h d -> n k (h d)", h=self.num_heads, d=self.head_dim),
            nn.Linear(self.num_heads * self.head_dim, self.in_features),
        )

    def forward(self, edge, edge_index):
        # Build self-edges for each node in the k-neighborhood.
        edge_self = (
            torch.arange(len(edge), device=edge.device)
            .unsqueeze(1)
            .expand(-1, edge.size(1))
        )

        query = self.query_proj(edge)  # n k h d
        key = self.key_proj(edge)  # n k h d
        value = self.value_proj(edge)  # n k h d

        # Concatenate self and neighbor keys/values.
        key = torch.cat((key[edge_self], key[edge_index]), dim=2)  # n k 2k h d
        value = torch.cat((value[edge_self], value[edge_index]), dim=2)  # n k 2k h d
        bias = self.attn_bias(edge)
        bias = torch.cat((bias[edge_self], bias[edge_index]), dim=2)  # n k 2k h

        gate = self.gate(edge)
        scores = torch.einsum("n k h c, n k j h c -> n k j h", query, key) / self.attention_scale + bias
        weights = torch.softmax(scores, dim=2)
        weights = self.attn_dropout(weights)
        out = gate * torch.einsum("n k j h, n k j h c -> n k h c", weights, value)

        # Residual + norm.
        out = self.output_dropout(self.output_proj(out))
        out = self.norm(out + math.sqrt(2) * self.residual(edge))
        return out

