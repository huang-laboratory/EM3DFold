import math
from functools import partial

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from em3dfold.utils.torch_utils import get_batch_slices, padded_sequence_softmax


def get_batched_sequence_attention_scores(
    sequence_query,  # n h d
    sequence_key,  # b s h d
    batch,  # n
    attention_scale,
    batch_size=200,
    device="cpu",
):
    output = torch.zeros(
        sequence_query.shape[0], sequence_key.shape[1], sequence_key.shape[2], device=device
    )  # n s h

    seq_batches = get_batch_slices(output.shape[1], batch_size)
    sequence_query = sequence_query[:, None]

    for seq_batch in seq_batches:
        output[:, seq_batch] = (
            sequence_query * sequence_key[:, seq_batch][batch]
        ).sum(dim=-1) / attention_scale

    return output


def get_batched_sequence_attention_features(
    sequence_attention_weights,  # n s h
    sequence_value,  # b s h d
    batch,  # n
    batch_size=200,
    device="cpu",
):
    output = torch.zeros(
        sequence_attention_weights.shape[0], sequence_value.shape[2], sequence_value.shape[3], device=device
    )  # n h d
    seq_batches = get_batch_slices(sequence_attention_weights.shape[1], batch_size)

    for seq_batch in seq_batches:
        output += (
            sequence_attention_weights[:, seq_batch][..., None]
            * sequence_value[:, seq_batch][batch]
        ).sum(dim=1)
    return output


class SequenceAttention(nn.Module):
    def __init__(
        self,
        d: int = 256,
        d_seq: int = 1280,
        d_seq_na: int | None = None,
        d_head: int = 48,
        n_head: int = 8,
        activation_class: nn.Module = nn.ReLU,
        checkpoint: bool = True,
    ):
        super().__init__()
        self.d = d
        self.d_seq_prot = d_seq
        self.d_seq_na = d_seq if d_seq_na is None else d_seq_na
        self.d_seq_joint = self.d_seq_prot + self.d_seq_na
        self.d_head = d_head
        self.n_head = n_head

        self.attention_scale = math.sqrt(self.d_head)
        self.norm = nn.LayerNorm(self.d)

        self.q = nn.Sequential(
            nn.Linear(self.d, self.n_head * self.d_head, bias=False),
            Rearrange("... (h d) -> ... h d", h=self.n_head, d=self.d_head),
        )

        self.k = nn.Sequential(
            nn.Linear(self.d_seq_joint, self.n_head * self.d_head, bias=False),
            Rearrange("... (h d) -> ... h d", h=self.n_head, d=self.d_head),
        )
        self.v = nn.Sequential(
            nn.Linear(self.d_seq_joint, self.n_head * self.d_head, bias=False),
            Rearrange("... (h d) -> ... h d", h=self.n_head, d=self.d_head),
        )

        self.gate = nn.Sequential(
            nn.Linear(self.d, self.n_head * self.d_head, bias=True),
            Rearrange("... (h d) -> ... h d", h=self.n_head, d=self.d_head),
            nn.Sigmoid(),
        )

        self.back = nn.Sequential(
            Rearrange("... h d -> ... (h d)", h=self.n_head, d=self.d_head),
            nn.Linear(self.n_head * self.d_head, self.d),
        )
        self.forward = self.forward_checkpoint if checkpoint else self.forward_normal

    def _prepare_sequence_inputs(
        self,
        seq_emb,
        seq_mask,
        d_seq,
        batch_size,
        device,
        dtype,
    ):
        if seq_emb is None:
            seq_emb = torch.zeros(batch_size, 3, d_seq, device=device, dtype=dtype)
            seq_mask = torch.ones(batch_size, 3, device=device, dtype=torch.bool)
            return seq_emb, seq_mask

        if seq_mask is None:
            seq_mask = torch.ones(seq_emb.shape[:2], device=seq_emb.device, dtype=torch.bool)

        return seq_emb, seq_mask

    def _merge_sequence_inputs(
        self,
        prot_seq_emb,
        prot_seq_mask,
        na_seq_emb,
        na_seq_mask,
    ):
        prot_pad = torch.zeros(
            prot_seq_emb.shape[0],
            prot_seq_emb.shape[1],
            self.d_seq_na,
            device=prot_seq_emb.device,
            dtype=prot_seq_emb.dtype,
        )
        na_pad = torch.zeros(
            na_seq_emb.shape[0],
            na_seq_emb.shape[1],
            self.d_seq_prot,
            device=na_seq_emb.device,
            dtype=na_seq_emb.dtype,
        )

        prot_joint = torch.cat([prot_seq_emb, prot_pad], dim=-1)
        na_joint = torch.cat([na_pad, na_seq_emb], dim=-1)
        joint_seq_emb = torch.cat([prot_joint, na_joint], dim=1)

        prot_only_mask = torch.cat(
            [
                prot_seq_mask,
                torch.zeros_like(na_seq_mask),
            ],
            dim=1,
        )
        na_only_mask = torch.cat(
            [
                torch.zeros_like(prot_seq_mask),
                na_seq_mask,
            ],
            dim=1,
        )
        return joint_seq_emb, prot_only_mask, na_only_mask

    def _attend_subset(
        self,
        node_subset,
        batch_subset,
        sequence_emb,
        sequence_mask,
        attention_batch_size,
        device,
    ):
        if node_subset.shape[0] == 0:
            return torch.zeros_like(node_subset)

        y = self.norm(node_subset)
        sequence_query = self.q(y)
        sequence_key = self.k(sequence_emb)
        sequence_value = self.v(sequence_emb)

        sequence_attention_scores = get_batched_sequence_attention_scores(
            sequence_query,
            sequence_key,
            batch_subset,
            self.attention_scale,
            batch_size=attention_batch_size,
            device=device,
        )

        batched_mask = sequence_mask[batch_subset].unsqueeze(-1)
        sequence_attention_weights = padded_sequence_softmax(
            sequence_attention_scores, batched_mask, dim=1
        )

        new_features_attention = get_batched_sequence_attention_features(
            sequence_attention_weights,
            sequence_value,
            batch_subset,
            batch_size=attention_batch_size,
            device=device,
        )

        gate = self.gate(y)
        new_features = self.back(gate * new_features_attention)
        return new_features

    def forward_normal(
        self,
        node,
        prot_mask=None,
        batch=None,
        attention_batch_size=200,
        prot_seq_emb=None,
        prot_seq_mask=None,
        na_seq_emb=None,
        na_seq_mask=None,
        **kwargs,
    ):
        return self._intern_forward(
            node=node,
            prot_mask=prot_mask,
            batch=batch,
            attention_batch_size=attention_batch_size,
            prot_seq_emb=prot_seq_emb,
            prot_seq_mask=prot_seq_mask,
            na_seq_emb=na_seq_emb,
            na_seq_mask=na_seq_mask,
        )

    def forward_checkpoint(
        self,
        node: torch.Tensor,
        prot_mask: torch.Tensor | None = None,
        batch=None,
        attention_batch_size: int = 200,
        prot_seq_emb=None,
        prot_seq_mask=None,
        na_seq_emb=None,
        na_seq_mask=None,
        **kwargs,
    ):
        new_forward = partial(
            self._intern_forward,
            prot_mask=prot_mask,
            batch=batch,
            attention_batch_size=attention_batch_size,
            prot_seq_emb=prot_seq_emb,
            prot_seq_mask=prot_seq_mask,
            na_seq_emb=na_seq_emb,
            na_seq_mask=na_seq_mask,
        )
        return torch.utils.checkpoint.checkpoint(
            new_forward,
            node,
            preserve_rng_state=False,
        )

    def _intern_forward(
        self,
        node: torch.Tensor,
        prot_mask: torch.Tensor | None,
        batch,
        attention_batch_size: int,
        prot_seq_emb,
        prot_seq_mask,
        na_seq_emb,
        na_seq_mask,
    ):
        device = node.device
        if prot_mask is None:
            raise ValueError("prot_mask must be provided for mixed protein/NA sequence attention.")

        if batch is None:
            batch = torch.zeros(node.shape[0], dtype=torch.long, device=device)

        prot_mask = prot_mask.bool()
        na_mask = ~prot_mask
        batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 1

        prot_seq_emb, prot_seq_mask = self._prepare_sequence_inputs(
            prot_seq_emb,
            prot_seq_mask,
            self.d_seq_prot,
            batch_size,
            device,
            node.dtype,
        )
        na_seq_emb, na_seq_mask = self._prepare_sequence_inputs(
            na_seq_emb,
            na_seq_mask,
            self.d_seq_na,
            batch_size,
            device,
            node.dtype,
        )

        new_features_all = torch.zeros_like(node)

        joint_seq_emb, prot_only_mask, na_only_mask = self._merge_sequence_inputs(
            prot_seq_emb,
            prot_seq_mask,
            na_seq_emb,
            na_seq_mask,
        )

        prot_features = self._attend_subset(
            node_subset=node[prot_mask],
            batch_subset=batch[prot_mask],
            sequence_emb=joint_seq_emb,
            sequence_mask=prot_only_mask,
            attention_batch_size=attention_batch_size,
            device=device,
        )
        if prot_features.shape[0] > 0:
            new_features_all[prot_mask] = prot_features

        na_features = self._attend_subset(
            node_subset=node[na_mask],
            batch_subset=batch[na_mask],
            sequence_emb=joint_seq_emb,
            sequence_mask=na_only_mask,
            attention_batch_size=attention_batch_size,
            device=device,
        )
        if na_features.shape[0] > 0:
            new_features_all[na_mask] = na_features

        node = math.sqrt(2) * node + new_features_all
        return node


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = SequenceAttention(checkpoint=False, d_seq=1280, d_seq_na=640).to(device)

    n = 16
    node = torch.randn(n, 256, device=device, requires_grad=True)
    prot_mask = torch.tensor([True] * 8 + [False] * 8, device=device)
    batch = torch.zeros(n, dtype=torch.long, device=device)

    prot_seq_embed = torch.randn(1, 32, 1280, device=device)
    prot_seq_mask = torch.ones(1, 32, dtype=torch.bool, device=device)
    na_seq_embed = torch.randn(1, 24, 640, device=device)
    na_seq_mask = torch.ones(1, 24, dtype=torch.bool, device=device)

    out = model(
        node=node,
        prot_mask=prot_mask,
        batch=batch,
        prot_seq_emb=prot_seq_embed,
        prot_seq_mask=prot_seq_mask,
        na_seq_emb=na_seq_embed,
        na_seq_mask=na_seq_mask,
    )
    out.sum().backward()
    print(out.shape)

